"""Chat memory survives restarts, rebuilt managers keep the user's level, and
AI calls no longer run on the event loop."""
import asyncio
import json
import uuid

import pytest


def _signup(client):
    r = client.post("/api/auth/signup", json={
        "email": f"p_{uuid.uuid4().hex[:8]}@example.com", "password": "Passw0rd!long",
        "name": "P", "consent": True})
    assert r.status_code == 200, r.text
    body = r.json()
    return {"Authorization": f"Bearer {body['token']}"}, body["user"]["user_id"]


class FakeOrchestrator:
    def __init__(self):
        self.seen = []

    def invoke(self, payload, config=None):
        msgs = list(payload["messages"])
        self.seen.append(msgs)

        class M:
            content = f"reply {len(self.seen)}"
            type = "ai"
        return {"messages": msgs + [M()]}


def _simulate_restart(app_mod, uid):
    """What a deploy does to a user: every process-local cache is gone."""
    app_mod._user_chat_history.pop(uid, None)
    app_mod._user_orchestrators.pop(uid, None)


@pytest.fixture
def quiet_tone(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "observe_tone", lambda *_a, **_k: {})


def _chat(client, h, text):
    r = client.post("/api/chat", headers=h, json={"message": text})
    assert r.status_code == 200, r.text
    return r.json()


def test_manager_remembers_the_conversation_after_a_restart(client, app_mod, monkeypatch, quiet_tone):
    h, uid = _signup(client)
    client.post("/api/start", headers=h, json={"career": "cybersecurity", "experience_level": "experienced", "reset": True})
    orc = FakeOrchestrator()
    app_mod._user_orchestrators[uid] = orc
    _chat(client, h, "I escalated the brute force alert")

    _simulate_restart(app_mod, uid)
    orc2 = FakeOrchestrator()
    monkeypatch.setattr(app_mod, "create_orchestrator", lambda *a, **k: orc2)
    _chat(client, h, "Did you see my escalation?")

    sent = orc2.seen[-1]
    assert ("human", "I escalated the brute force alert") in sent
    assert ("ai", "reply 1") in sent
    assert sent[-1] == ("human", "Did you see my escalation?")


def test_a_reset_is_not_undone_by_a_restart(client, app_mod, monkeypatch, quiet_tone):
    h, uid = _signup(client)
    client.post("/api/start", headers=h, json={"career": "cybersecurity", "experience_level": "experienced", "reset": True})
    app_mod._user_orchestrators[uid] = FakeOrchestrator()
    _chat(client, h, "old simulation message")
    client.post("/api/start", headers=h, json={"career": "cybersecurity", "experience_level": "experienced", "reset": True})

    _simulate_restart(app_mod, uid)
    orc = FakeOrchestrator()
    monkeypatch.setattr(app_mod, "create_orchestrator", lambda *a, **k: orc)
    _chat(client, h, "new simulation message")
    assert all("old simulation" not in str(m) for m in orc.seen[-1])


@pytest.mark.parametrize("level", ["beginner", "experienced"])
def test_rebuilt_manager_keeps_the_users_level(client, app_mod, monkeypatch, level):
    h, uid = _signup(client)
    client.post("/api/start", headers=h, json={"career": "cybersecurity", "experience_level": level, "reset": True})
    _simulate_restart(app_mod, uid)
    built = []
    monkeypatch.setattr(app_mod, "create_orchestrator", lambda career, lvl="studied": built.append((career, lvl)) or FakeOrchestrator())
    client.get("/api/inbox", headers=h)
    assert built == [("cybersecurity", level)]


def _on_event_loop() -> bool:
    """True when called from the thread running the server's event loop.
    (TestClient runs the app off the main thread, so thread names prove nothing.)"""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_slack_tone_check_runs_off_the_event_loop(client, auth, app_mod, monkeypatch):
    client.post("/api/start", headers=auth, json={"career": "cybersecurity", "experience_level": "experienced", "reset": True})
    where = []
    monkeypatch.setattr(app_mod, "observe_tone", lambda *_a, **_k: where.append(_on_event_loop()) or {})
    r = client.post("/api/slack/post", headers=auth, json={"channel": "#soc-alerts", "text": "Escalating ALERT-2 now"})
    assert r.status_code == 200, r.text
    assert where == [False]


def test_jd_model_calls_run_in_worker_threads(client, auth, app_mod, monkeypatch):
    where = []

    class Fake:
        def invoke(self, *_a, **_k):
            where.append(_on_event_loop())
            raise RuntimeError("404 model not found")
    monkeypatch.setattr(app_mod, "llm", Fake())
    r = client.post("/api/scenario/from-jd", headers=auth,
                    json={"career": "auto", "experience_level": "experienced", "text": "Process supplier invoices daily."})
    assert r.status_code == 503
    assert where and not any(where)


def test_second_jd_while_first_is_running_is_refused(app_mod, client):
    auth, uid = _signup(client)
    # Hold the per-user JD lock as if a paste were mid-extraction.
    lk = app_mod._async_lock(app_mod._user_locks, f"{uid}:jd")

    async def hold():
        await lk.acquire()
    asyncio.run(hold())
    try:
        r = client.post("/api/scenario/from-jd", headers=auth,
                        json={"career": "cybersecurity", "experience_level": "experienced", "text": "SOC analyst role"})
        assert r.status_code == 429
        assert "still being processed" in r.json()["error"]["message"]
    finally:
        lk.release()


class SeqLLM:
    def __init__(self, *steps):
        self.steps, self.calls = list(steps), 0

    def invoke(self, *_a, **_k):
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step

        class R:
            content = step
        return R()


def _job(career="cybersecurity"):
    from scenario_engine import JobPosting
    return JobPosting(ext_id="t:1", career=career, title="SOC Analyst", company="Acme",
                      description="Triage SIEM alerts and escalate incidents.", location="",
                      seniority="junior", source="test", tools_mentioned=[], raw={})


def test_extractor_repairs_bad_json_once(app_mod, monkeypatch):
    from scenario_engine import ScenarioExtractor
    fake = SeqLLM("Sure! here you go: {broken", '{"tasks": [], "starter_emails": []}')
    ex = ScenarioExtractor(fake)
    tpl, reason = ex.extract_with_reason(_job())
    assert fake.calls == 2           # one repair call, no more
    assert tpl is None and reason == "invalid"  # repaired JSON parsed, then failed validation honestly


def test_extractor_repair_outage_is_reported_as_llm(app_mod):
    from scenario_engine import ScenarioExtractor
    ex = ScenarioExtractor(SeqLLM("not json", RuntimeError("503 upstream")))
    assert ex.extract_with_reason(_job()) == (None, "llm")


def test_extractor_reason_does_not_live_on_shared_state(app_mod):
    from scenario_engine import ScenarioExtractor
    ex = ScenarioExtractor(SeqLLM())
    assert ex.extract_with_reason(_job("hr")) == (None, "no_palette")
    assert ex.last_failure is None   # only the legacy extract() sets it


def test_requirements_are_exactly_pinned():
    import pathlib
    lines = [l.strip() for l in (pathlib.Path(app_mod_root()) / "requirements.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert lines and all("==" in l for l in lines), lines


def app_mod_root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_eval_script_scores_and_gates(capsys):
    import importlib.util, os
    path = os.path.join(app_mod_root(), "scripts", "eval_career_classifier.py")
    spec = importlib.util.spec_from_file_location("evalcc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = json.load(open(os.path.join(app_mod_root(), "scripts", "eval_career_cases.json")))

    class Perfect:
        def __init__(self):
            self.i = 0

        def invoke(self, *_a, **_k):
            c = cases[self.i]
            self.i += 1

            class R:
                content = c["expected"] + "."
            return R()

    class AlwaysPm:
        def invoke(self, *_a, **_k):
            class R:
                content = "pm"
            return R()

    assert mod.main([], llm=Perfect()) == 0
    assert "accuracy 16/16" in capsys.readouterr().out
    assert mod.main([], llm=AlwaysPm()) == 1
    # the eval must use production's prompt, not a copy
    prompt, valid = mod.load_prompt_and_careers()
    from app import _CAREER_CLASSIFIER_PROMPT, _VALID_CAREERS
    assert prompt == _CAREER_CLASSIFIER_PROMPT and valid == _VALID_CAREERS
