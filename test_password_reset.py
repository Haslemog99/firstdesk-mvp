#!/usr/bin/env python3
"""Password reset contracts — token hashing, link building, who may reset, copy.

Run: python3 test_password_reset.py   (exit 0 pass / 1 fail)

AST-extracts pure helpers from app.py so this never boots Mongo/FastAPI.
Written so the reset flow cannot quietly regress into the classic failures:
storing raw tokens, resetting the wrong account, or mailing a dead link.
"""
import ast
import hashlib
import sys
from html import escape as _escape
from typing import Optional as _Optional
from urllib.parse import quote as _quote

SRC = open("app.py").read()


def _extract(names: set):
    tree = ast.parse(SRC)
    segs = []
    for node in tree.body:
        got = set()
        if isinstance(node, ast.Assign):
            got = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign):
            t = node.target
            if isinstance(t, ast.Name):
                got = {t.id}
        elif isinstance(node, ast.FunctionDef):
            got = {node.name}
        if got & names:
            segs.append(ast.get_source_segment(SRC, node))
    defined = {n for n in names if any(
        (isinstance(node, ast.FunctionDef) and node.name == n)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == n for t in node.targets))
        or (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == n)
        for node in tree.body)}
    ns = {"hashlib": hashlib, "_html_escape": _escape, "quote": _quote,
          "Optional": _Optional}
    exec("\n\n".join(s for s in segs if s), ns)
    return ns, names - defined


NEED = {
    "_normalize_email",
    "_hash_reset_token",
    "_reset_link",
    "_can_request_reset",
    "_reset_email_html",
    "_RESET_TTL_SEC",
}

NS, MISSING = _extract(NEED)

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


if MISSING:
    print("FAIL — implement the password-reset helpers first:")
    for m in sorted(MISSING):
        print("  missing", m)
    sys.exit(1)

norm = NS["_normalize_email"]
hash_tok = NS["_hash_reset_token"]
link_for = NS["_reset_link"]
may_reset = NS["_can_request_reset"]
email_html = NS["_reset_email_html"]
ttl = int(NS["_RESET_TTL_SEC"])


# ── 1. Tokens are stored hashed, never raw ───────────────────────
raw = "0Fq3xZ_tokenvalue-abc123"
check(hash_tok(raw) == hash_tok(raw), "hashing the same token twice must agree")
check(hash_tok(raw) != raw, "the raw token must never be what we store")
check(hash_tok(raw) != hash_tok(raw + "x"), "different tokens must hash differently")
check(len(hash_tok(raw)) == 64 and all(c in "0123456789abcdef" for c in hash_tok(raw)),
      "token hash must be sha256 hex so it fits a unique index")


# ── 2. The emailed link has to actually resolve ──────────────────
check(link_for("https://app.example.com", "tok123") == "https://app.example.com/?reset=tok123",
      "reset link must point at the SPA with the token as ?reset=")
check(link_for("https://app.example.com/", "tok123") == "https://app.example.com/?reset=tok123",
      "a trailing slash on the base URL must not produce a double slash")
check("tok+/=abc" not in link_for("https://x.io", "tok+/=abc"),
      "token must be URL-encoded into the link, not pasted raw")


# ── 3. Only real accounts can be reset ───────────────────────────
real = {"user_id": "u_1", "email": "sam@org.com", "password_hash": "c2FsdA==$aGFzaA=="}
check(may_reset(real) is True, "a normal account can request a reset")
check(may_reset(None) is False, "an unknown email cannot request a reset")
check(may_reset({**real, "user_id": "g_1", "email": "g_1@guest.local", "is_guest": True}) is False,
      "guest accounts have no password, so no reset")
check(may_reset({**real, "user_id": "g_2", "email": "g_2@guest.local"}) is False,
      "a guest.local address is a guest even without the flag")
check(may_reset({"user_id": "u_2", "email": "x@org.com", "password_hash": ""}) is False,
      "an account with an empty password hash has nothing to reset")
check(may_reset({"user_id": "u_3", "email": "y@org.com"}) is False,
      "a doc with password_hash projected away must fail closed")


# ── 4. Lookups must match how signup stored the address ──────────
check(norm("  Name@Org.com ") == "name@org.com",
      "email lookup must lowercase and strip, or a locked-out user silently gets nothing")
check(norm(None) == "", "a missing email must normalise to empty, not explode")


# ── 5. The email itself has to do its job ────────────────────────
body = email_html("Sam", "https://app.example.com/?reset=tok123", 60)
check("https://app.example.com/?reset=tok123" in body, "the email must contain the reset link")
check("60" in body, "the email must state how long the link lasts")
check("did not request" in body.lower(), "the email must tell a victim what to do")
check("—" not in body and "–" not in body, "site copy carries no em or en dashes")
xss = email_html('<script>alert(1)</script>', "https://x.io/?reset=t", 60)
check("<script>" not in xss, "the display name must be escaped into the email HTML")


# ── 6. A reset window measured in hours is a liability ───────────
check(0 < ttl <= 3600, "reset tokens must expire within an hour")


if fails:
    print(f"FAIL {len(fails)}")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("OK  password reset contracts")
sys.exit(0)
