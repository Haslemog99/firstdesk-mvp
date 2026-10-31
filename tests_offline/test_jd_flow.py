"""Paste-a-JD flow: honest errors instead of a silent 'pm' and a misleading 422."""
import pytest

JD = ("About Us: Sutaka UK Ltd is a wholesale and export company. The Wholesale Admin "
      "Assistant processes orders, raises invoices, chases suppliers and keeps stock records.")


class SeqLLM:
    """Each invoke() takes the next behaviour: a string reply, or an Exception to raise."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = 0

    def invoke(self, messages, config=None, **_kw):
        self.calls += 1
        step = self.steps.pop(0) if self.steps else "unused"
        if isinstance(step, Exception):
            raise step

        class R:
            content = step
        return R()


NOT_FOUND = RuntimeError("Error code: 404 - Model not found, inaccessible, and/or not deployed")


@pytest.fixture
def use_llm(app_mod, monkeypatch):
    def _install(fake):
        monkeypatch.setattr(app_mod, "llm", fake)
        engine = app_mod.get_scenario_engine()
        monkeypatch.setattr(engine.extractor, "llm", fake)
        return fake
    return _install


def _post(client, auth, career="auto", text=JD):
    return client.post("/api/scenario/from-jd", headers=auth,
                       json={"career": career, "experience_level": "experienced", "text": text})


def test_model_outage_is_reported_as_our_fault(client, auth, use_llm):
    use_llm(SeqLLM(NOT_FOUND))
    r = _post(client, auth)
    assert r.status_code == 503, r.text
    msg = r.json()["error"]["message"]
    assert "not with your job description" in msg
    assert "more detailed" not in msg


def test_outage_does_not_burn_the_rate_limit(client, auth, use_llm):
    use_llm(SeqLLM(NOT_FOUND, NOT_FOUND))
    assert _post(client, auth).status_code == 503
    assert _post(client, auth).status_code == 503  # not 429


def test_unsupported_track_says_so_and_logs_demand(client, auth, use_llm, app_mod):
    fake = use_llm(SeqLLM("procurement"))
    r = _post(client, auth)
    assert r.status_code == 422, r.text
    body = r.json()["error"]
    assert body["code"] == "jd_role_not_supported"
    assert body["career"] == "procurement"
    assert "isn't a problem with what you pasted" in body["message"]
    assert "Procurement" in body["message"]
    assert fake.calls == 1  # extractor never called for an unsupported track
    row = app_mod.analytics_col.find_one({"event": "jd_unsupported_role"}, sort=[("ts", -1)])
    assert row and row["data"]["career"] == "procurement"


def test_explicit_unsupported_career_is_honest_too(client, auth, use_llm):
    use_llm(SeqLLM())
    r = _post(client, auth, career="hr")
    assert r.status_code == 422
    assert "isn't a problem with what you pasted" in r.json()["error"]["message"]


def test_unreadable_classifier_reply_asks_user_to_pick(client, auth, use_llm):
    use_llm(SeqLLM("I think this is probably an admin role"))
    r = _post(client, auth)
    assert r.status_code == 422
    assert "Pick the closest career" in r.json()["error"]["message"]


def test_extractor_outage_on_supported_track_is_503(client, auth, use_llm):
    fake = use_llm(SeqLLM("cybersecurity", NOT_FOUND))
    r = _post(client, auth)
    assert r.status_code == 503, r.text
    assert fake.calls == 2


def test_bad_extractor_output_keeps_the_old_message(client, auth, use_llm):
    use_llm(SeqLLM("cybersecurity", "not json at all"))
    r = _post(client, auth)
    assert r.status_code == 422
    assert "more detailed" in r.json()["error"]["message"]


def test_classifier_never_silently_defaults_to_pm(app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "llm", SeqLLM(NOT_FOUND))
    with pytest.raises(app_mod.CareerClassifierUnavailable):
        app_mod._classify_career_from_jd(JD, "Admin")
