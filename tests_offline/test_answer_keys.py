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
    assert "Correct" in r.text
