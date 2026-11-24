#!/usr/bin/env python3
"""Contracts for the security-review fixes. No Mongo required.

Run: python3 test_review_fixes.py
"""
import ast
import base64
import hashlib
import hmac
import os
import socket
import time
from datetime import datetime, timezone

os.environ.setdefault("AUTH_SECRET", "test-secret-for-review-fixes")
os.environ.setdefault("MONGO_URI", "mongodb://127.0.0.1:27017")
os.environ.setdefault("FIREWORKS_API_KEY", "test")

import httpx
from fastapi import HTTPException

import app
from scenario_engine import JobPosting, ScenarioExtractor
from tool_dispatch import normalise_tool_spec

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
        print("FAIL", msg)
    else:
        print("ok", msg)


class _Req:
    def __init__(self, peer, xff=None):
        self.headers = {} if xff is None else {"x-forwarded-for": xff}
        self.client = type("C", (), {"host": peer})()


def test_client_ip():
    old = os.environ.get("TRUSTED_PROXY_HOPS")
    try:
        os.environ["TRUSTED_PROXY_HOPS"] = "0"
        ip = app._client_ip(_Req("10.1.1.1", "1.2.3.4, 9.9.9.9"))
        check(ip == "10.1.1.1", "socket IP is used when no proxy is trusted")
        os.environ["TRUSTED_PROXY_HOPS"] = "1"
        ip = app._client_ip(_Req("10.1.1.1", "1.2.3.4, 9.9.9.9"))
        check(ip == "9.9.9.9", "last X-Forwarded-For hop is the trusted proxy's client")
        check(ip != "1.2.3.4", "first X-Forwarded-For hop is not trusted")
        ip = app._client_ip(_Req("10.1.1.1", None))
        check(ip == "10.1.1.1", "missing X-Forwarded-For falls back to the socket")
    finally:
        if old is None:
            os.environ.pop("TRUSTED_PROXY_HOPS", None)
        else:
            os.environ["TRUSTED_PROXY_HOPS"] = old


def test_login_cap():
    email = "lock-" + hashlib.sha256(str(time.time()).encode()).hexdigest()[:12] + "@example.com"
    try:
        allowed = [app._reserve_login_attempt(email) for _ in range(app.LOGIN_MAX_FAILURES)]
        check(all(allowed), "the first batch of guesses is counted and allowed")
        check(not app._reserve_login_attempt(email), "the next guess for that email is refused")
        app._clear_login_failures(email)
        check(app._reserve_login_attempt(email), "a successful login can clear the counter")
    finally:
        app._clear_login_failures(email)


def test_ip_hourly():
    old = os.environ.get("TRUSTED_PROXY_HOPS")
    os.environ["TRUSTED_PROXY_HOPS"] = "0"
    try:
        req = _Req("203.0.113.10")
        # Isolate this peer from any earlier calls.
        app._ip_hourly.pop("signup:203.0.113.10", None)
        oks = [app._ip_hourly_ok("signup", req) for _ in range(app.SIGNUP_PER_IP_HOURLY)]
        check(all(oks) and not app._ip_hourly_ok("signup", req), "signup is capped per socket IP")
        app._ip_hourly.pop("guest:203.0.113.10", None)
        goks = [app._ip_hourly_ok("guest", req) for _ in range(app.GUEST_PER_IP_HOURLY)]
        check(all(goks) and not app._ip_hourly_ok("guest", req), "guest creation is capped per socket IP")
    finally:
        app._ip_hourly.pop("signup:203.0.113.10", None)
        app._ip_hourly.pop("guest:203.0.113.10", None)
        if old is None:
            os.environ.pop("TRUSTED_PROXY_HOPS", None)
        else:
            os.environ["TRUSTED_PROXY_HOPS"] = old


def _legacy_token(user_id, skew=0):
    expiry = int(time.time()) + app.TOKEN_TTL + skew
    payload = f"{user_id}.{expiry}"
    sig = hmac.new(app.AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}.{sig}".encode()).decode().rstrip("=")


def test_tokens():
    tok = app._make_token("u_new", 0)
    parsed = app._parse_token(tok)
    check(parsed and parsed[0] == "u_new" and parsed[2] == 0, "new tokens carry a version")
    user = {"user_id": "u_new", "token_version": 0}
    check(not app._token_revoked(user, tok), "a current token is accepted")
    user["token_version"] = 1
    check(app._token_revoked(user, tok), "a bumped token_version rejects the old token")

    legacy = _legacy_token("u_old")
    check(app._parse_token(legacy)[2] == 0, "tokens from before the version field still parse")
    old_user = {"user_id": "u_old"}
    check(not app._token_revoked(old_user, legacy), "a legacy token works until something revokes it")
    stale = _legacy_token("u_old", skew=-120)
    changed = {"user_id": "u_old", "password_changed_at": datetime.now(timezone.utc).isoformat()}
    check(app._token_revoked(changed, stale), "password_changed_at rejects tokens issued before it")
    fresh = app._make_token("u_reset", 3)
    iat = app._parse_token(fresh)[1] - app.TOKEN_TTL
    same = {"user_id": "u_reset", "token_version": 3,
            "password_changed_at": datetime.fromtimestamp(iat, timezone.utc).isoformat()}
    check(not app._token_revoked(same, fresh), "the token minted by the reset itself stays valid")
    app._remember_token_epoch("u_old", 4)
    try:
        check(app._token_revoked(old_user, legacy), "logout's epoch rejects the token even if the cache is stale")
    finally:
        app._token_epoch.pop("u_old", None)


def test_public_ip_and_allowlist():
    check(app._jd_host_allowed("boards.greenhouse.io"), "allowlisted host")
    check(app._jd_host_allowed("jobs.boards.greenhouse.io"), "subdomain of an allowlisted host")
    check(not app._jd_host_allowed("boards.greenhouse.io.evil.com"), "suffix trick is not allowlisted")
    check(not app._jd_host_allowed("evil.example"), "other hosts are refused")
    check(not app._jd_host_allowed("127.0.0.1"), "raw loopback is not allowlisted")
    for bad in ("127.0.0.1", "10.0.0.5", "192.168.1.9", "169.254.169.254", "::1", "fe80::1",
                "64:ff9b::a9fe:a9fe", "::ffff:127.0.0.1", "0.0.0.0"):
        check(not app._ip_is_public(bad), f"{bad} is not a public address")
    for good in ("8.8.8.8", "1.1.1.1", "2001:4860:4860::8888"):
        check(app._ip_is_public(good), f"{good} is public")

    def boom(url):
        try:
            app._resolve_public_or_raise(httpx.URL(url))
            return None
        except HTTPException as ex:
            return ex.status_code

    check(boom("file:///etc/passwd") == 400, "non-http schemes are refused")
    check(boom("http://127.0.0.1/latest") == 400, "loopback URL is refused")
    check(boom("https://evil.example/job") == 400, "host outside the allowlist is refused")
    check(boom("https://boards.greenhouse.io:8443/job") == 400, "non-web ports are refused")

    real = socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):
        ip = "10.0.0.8" if host == "boards.greenhouse.io" else "1.1.1.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    socket.getaddrinfo = fake_getaddrinfo
    try:
        check(boom("https://boards.greenhouse.io/acme") == 400,
              "an allowlisted name that resolves to a private IP is refused")
        socket.getaddrinfo = lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port)),
        ]
        check(boom("https://jobs.lever.co/acme") == 400,
              "one private DNS answer rejects the host")
        socket.getaddrinfo = lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port))
        ]
        ip = app._resolve_public_or_raise(httpx.URL("https://jobs.lever.co/acme"))
        check(ip == "1.1.1.1", "a public DNS answer is the address we would connect to")
    finally:
        socket.getaddrinfo = real
    src = open("app.py").read()
    check("follow_redirects=False" in src and "follow_redirects=True" not in src.split("def _fetch_jd_from_url")[1].split("\nclass ")[0],
          "the JD fetch does not follow redirects on its own")


def test_answer_keys():
    alert = {"id": "a", "title": "scan", "is_real_threat": False, "correct_action": "resolve",
             "playbook_steps": ["Mark resolved"], "threat_type": "false_positive",
             "false_positive_indicators": "scheduled scan", "severity": "LOW"}
    pub = app._siem_public(alert)
    for key in ("is_real_threat", "correct_action", "playbook_steps", "threat_type", "false_positive_indicators"):
        check(key not in pub, f"SIEM response drops {key}")
    check(pub["title"] == "scan" and pub["severity"] == "LOW", "SIEM response keeps the alert itself")
    leaked = app._invoice_public({
        "id": "INV-1", "amount": 12400, "correct_amount": 11800, "notes": "VAT applied in error",
        "vat_applied": 600, "client": "Crawford",
    })
    check("correct_amount" not in leaked and "notes" not in leaked and "vat_applied" not in leaked,
          "a queried invoice does not include the correction")
    check(leaked["client"] == "Crawford" and leaked["amount"] == 12400, "invoice amounts stay visible")
    kept = app._invoice_public({
        "id": "INV-2", "amount": 9800, "correct_amount": 9800, "notes": "32 days overdue — chase AP",
    })
    check(kept.get("notes", "").startswith("32 days") and "correct_amount" not in kept,
          "an operational note stays when it is not the answer")


class _Col:
    def __init__(self, docs):
        self.docs = docs

    def _match(self, doc, q):
        for k, v in q.items():
            if isinstance(v, dict):
                if "$ne" in v and doc.get(k) == v["$ne"]:
                    return False
                if "$nin" in v and doc.get(k) in v["$nin"]:
                    return False
            elif doc.get(k) != v:
                return False
        return True

    def find_one(self, q, proj=None):
        for d in self.docs:
            if self._match(d, q):
                return dict(d)
        return None

    def update_one(self, q, upd):
        class R:
            modified_count = 0
        r = R()
        for d in self.docs:
            if not self._match(d, q):
                continue
            changed = False
            for k, v in (upd.get("$set") or {}).items():
                if d.get(k) != v:
                    changed = True
                d[k] = v
            for k, v in (upd.get("$inc") or {}).items():
                d[k] = d.get(k, 0) + v
                changed = True
            for k, v in (upd.get("$addToSet") or {}).items():
                cur = d.setdefault(k, [])
                if v not in cur:
                    cur.append(v)
                    changed = True
            if changed:
                r.modified_count = 1
            break
        return r


def test_completion_paid_once():
    tasks = _Col([
        {"id": "t1", "simulation_id": "sim_ua", "title": "Reconcile", "points": 20,
         "status": "pending", "review_status": "open"},
        {"id": "t1", "simulation_id": "sim_ub", "title": "Other sim", "points": 20,
         "status": "pending", "review_status": "open"},
    ])
    sessions = _Col([{"session_id": "sess_ua", "score_adj": 0, "tasks_completed": []}])
    old = (app.tasks_col, app.sessions_col, app._apply_level_weight, app.emit)
    app.tasks_col = tasks
    app.sessions_col = sessions
    app._apply_level_weight = lambda pts, kind: pts
    app.emit = lambda *a, **k: None
    try:
        app.set_scope("ua")
        outcome, _t, credit = app._mark_task_done("t1")
        check(outcome == "done" and credit == 4, "first completion pays 20% once")
        outcome, _t, credit = app._mark_task_done("t1")
        check(outcome == "already" and credit == 0, "completing again pays nothing")
        score = sessions.docs[0]["score_adj"]
        check(score == 4, "the session score moved only on the first transition")
        # Rework reopens the task. The credit flag must hold.
        tasks.docs[0]["status"] = "pending"
        tasks.docs[0]["review_status"] = "needs_rework"
        outcome, _t, credit = app._mark_task_done("t1")
        check(outcome == "done" and credit == 0, "resubmitting after rework does not pay again")
        check(sessions.docs[0]["score_adj"] == 4, "score stays put on resubmit")
        check(tasks.docs[1]["status"] == "pending", "the same seeded id in another sim is untouched")
    finally:
        app.tasks_col, app.sessions_col, app._apply_level_weight, app.emit = old


def test_model_and_classifier():
    src = open("app.py").read()
    check('accounts/fireworks/models/deepseek-v4p1-flash-1017' in src, "default model is the current serverless id")
    check(src.split("FIREWORKS_MODEL = os.environ.get", 1)[1].split(")", 1)[0].count("deepseek-v4-pro-0813") == 0,
          "the retired model is not the default")
    check("deepseek-v4p1-flash-1017" in open(".env.example").read(), ".env.example matches the code default")

    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("404 model not found: deepseek-v4-pro-0813")

    class Say:
        def __init__(self, text):
            self.text = text
        def invoke(self, *a, **k):
            return type("R", (), {"content": self.text})()

    old = app.llm
    try:
        app.llm = Boom()
        raised = False
        try:
            app._classify_career_from_jd("A SOC analyst role " * 30, "SOC analyst")
        except app.CareerClassifierUnavailable:
            raised = True
        check(raised, "a model error is raised instead of a silent 'pm'")
        app.llm = Say("not-a-career")
        check(app._classify_career_from_jd("some role " * 20, "Role") is None,
              "an unreadable classifier reply is not turned into pm")
        app.llm = Say("cybersecurity, because it mentions a SIEM")
        check(app._classify_career_from_jd("SIEM " * 20, "Analyst") == "cybersecurity",
              "a real career word is still accepted")
    finally:
        app.llm = old

    class BoomLLM:
        def invoke(self, *a, **k):
            raise RuntimeError("model_not_found")

    ext = ScenarioExtractor(BoomLLM())
    job = JobPosting(ext_id="t", career="accountant", title="Junior accountant",
                     company="Acme", description="Reconcile the ledger " * 40)
    check(ext.extract(job) is None and ext.last_failure == "llm",
          "extractor records an LLM failure instead of a generic miss")


def test_tool_ids_and_queries_and_deletion():
    spec = {"tool_type": "alerts", "name": "Queue", "icon": 'x" onmouseover=alert(1)', "tint": "red;}</",
            "data": {"alerts": [{"id": "A1');alert(1);('", "severity": "HIGH", "title": "t",
                                 "details": "d", "correct_action": "escalate", "points": 5}]}}
    out = normalise_tool_spec(spec)
    check(out is not None and out["icon"] != spec["icon"] and out["tint"].startswith("#"),
          "generated icon and tint cannot break an attribute")
    check(all(ch not in out["data"]["alerts"][0]["id"] for ch in ("'", ")", ";")),
          "generated alert ids are stripped of quote and call characters")
    src = open("app.py").read()
    tree = ast.parse(src)
    scope = {"simulation_id", "session_id", "user_id", "token_hash", "scenario_id",
             "fingerprint", "ext_id", "email"}
    global_cols = {"users_col", "progress_col", "password_resets_col", "jobs_col",
                   "scenarios_col", "careers_col", "interviews_col", "analytics_col"}
    ops = {"find_one", "find", "update_one", "update_many", "delete_one", "delete_many",
           "find_one_and_update"}
    bad = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ops
                and isinstance(n.func.value, ast.Name) and n.func.value.id.endswith("_col")
                and n.func.value.id not in global_cols and n.args and isinstance(n.args[0], ast.Dict)):
            keys = {k.value for k in n.args[0].keys if isinstance(k, ast.Constant)}
            if "id" in keys and not (keys & scope):
                bad.append(f"{n.func.value.id}.{n.func.attr}:{n.lineno}")
    check(not bad, "every id query on sim data also carries simulation_id")
    for name in ("hr_onboarding_col", "training_col", "it_setup_col", "files_col", "grc_col"):
        check(name in app._USER_DATA_COLS_BY_SIM_ID and name in app._USER_DATA_COLS_BY_USER_ID,
              f"account deletion removes {name}")
    check("analytics_col" in app._USER_DATA_COLS_BY_USER_ID, "account deletion removes analytics")
    html = open("static/index.html").read()
    check("function jsq(" in html and "&#39;" in html, "esc() covers quotes and jsq() exists")
    check("${t.tool_id}" not in html and "'${t.tool_id}'" not in html,
          "tool_id is not interpolated raw into a handler")


if __name__ == "__main__":
    test_client_ip()
    test_login_cap()
    test_ip_hourly()
    test_tokens()
    test_public_ip_and_allowlist()
    test_answer_keys()
    test_completion_paid_once()
    test_model_and_classifier()
    test_tool_ids_and_queries_and_deletion()
    if fails:
        print(f"\n{len(fails)} failed")
        raise SystemExit(1)
    print("\nall review-fix checks passed")
