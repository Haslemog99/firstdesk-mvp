"""Ground truth must stay on the server: no answer-key fields in client JSON."""
import pytest

CAREERS = ["accountant", "cybersecurity", "marketing", "developer",
           "procurement", "hr", "pm", "grc"]

# Fields that state the right answer outright. The UI never reads them.
FORBIDDEN = {
    "is_real_threat", "correct_action", "correct_actions", "threat_type",
    "false_positive_reason", "level_trap", "correct_amount", "correct_verdict",
    "correct_reject_reason", "is_flawed", "flaw_type", "is_correct",
    "expected_artifacts", "expected_substrings", "expected_min_rows",
    "expected_command_patterns", "points_correct", "points_wrong",
}

LIST_ENDPOINTS = ["/api/inbox", "/api/tasks", "/api/siem", "/api/edr", "/api/jira",
                  "/api/tools", "/api/invoices", "/api/campaigns", "/api/grc",
                  "/api/meetings", "/api/session"]


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, f"{path}.{k}"
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


@pytest.mark.parametrize("career", CAREERS)
def test_no_answer_keys_in_any_list_endpoint(client, auth, career):
    r = client.post("/api/start", json={"career": career, "experience_level": "experienced", "reset": True},
                    headers=auth)
    assert r.status_code == 200, r.text
    leaks = []
    for _ in range(5):  # walk every day so later-day content is covered too
        for ep in LIST_ENDPOINTS:
            resp = client.get(ep, headers=auth)
            if resp.status_code != 200:
                continue
            for key, where in _walk_keys(resp.json()):
                if key in FORBIDDEN:
                    leaks.append(f"{ep}{where}")
        nd = client.post("/api/simulation/next-day", headers=auth)
        if nd.status_code != 200:
            break
    assert not leaks, f"{len(leaks)} answer-key leaks, e.g. {leaks[:8]}"


def test_scoring_still_uses_ground_truth(client, auth, app_mod):
    client.post("/api/start", json={"career": "cybersecurity", "experience_level": "experienced", "reset": True},
                headers=auth)
    alerts = client.get("/api/siem", headers=auth).json()["alerts"]
    assert alerts
    # Read the hidden truth straight from the DB, the way the server does.
    a = alerts[0]
    doc = app_mod.siem_col.find_one({"id": a["id"]})
    right = (doc.get("correct_action") or ("escalate" if doc.get("is_real_threat") else "resolve"))
    if right not in ("escalate", "resolve"):
        pytest.skip(f"first alert expects {right}")
    r = client.post(f"/api/siem/{a['id']}/triage", headers=auth, json={
        "action": right, "rationale": "Checked the context and telemetry before deciding."})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["correct"] is True and d["base_points"] > 0
    assert app_mod.evals_col.find_one({"type": "alert_triage", "alert_id": a["id"], "correct": True})


def _first_alert_where(app_mod, client, auth, pred):
    client.post("/api/start", json={"career": "cybersecurity", "experience_level": "experienced", "reset": True},
                headers=auth)
    for a in client.get("/api/siem", headers=auth).json()["alerts"]:
        doc = app_mod.siem_col.find_one({"id": a["id"]})
        if pred(doc):
            return doc
    pytest.skip("no matching alert")


def test_take_ownership_neither_reveals_nor_penalises(client, auth, app_mod):
    doc = _first_alert_where(app_mod, client, auth,
                             lambda d: d.get("correct_action") in ("escalate", "resolve"))
    r = client.post(f"/api/siem/{doc['id']}/triage", headers=auth, json={"action": "investigate"})
    d = r.json()
    assert r.status_code == 200
    assert d["points"] == 0 and d["correct"] is None
    assert not d.get("expected")
    assert doc["correct_action"] not in d["message"]
    assert not app_mod.evals_col.find_one({"type": "alert_triage", "alert_id": doc["id"], "action": "investigate"})


def test_investigate_still_scores_when_it_is_the_answer(client, auth, app_mod):
    doc = _first_alert_where(app_mod, client, auth, lambda d: d.get("correct_action") == "investigate")
    d = client.post(f"/api/siem/{doc['id']}/triage", headers=auth, json={"action": "investigate"}).json()
    assert d["correct"] is True and d["base_points"] > 0


def test_manager_tools_do_not_see_open_answers(client, auth, app_mod):
    client.post("/api/start", json={"career": "cybersecurity", "experience_level": "experienced", "reset": True},
                headers=auth)
    open_alert = client.get("/api/siem", headers=auth).json()["alerts"][0]
    doc = app_mod.siem_col.find_one({"id": open_alert["id"]})
    line = app_mod._format_siem_tool_line(doc)
    assert "correct_action" not in line and "judged=pending" in line
    closed = {**doc, "triage_status": "reviewed", "status": doc.get("correct_action")}
    assert "judged=correct" in app_mod._format_siem_tool_line(closed)
    redacted = app_mod._redact_answer_keys({"data": {"rows": [{"correct_action": "x", "keep": 1}]}})
    assert redacted == {"data": {"rows": [{"keep": 1}]}}
