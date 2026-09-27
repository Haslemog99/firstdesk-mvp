"""
FirstDesk — Generic Tool Dispatcher
====================================

A unified validator + side-effect engine for the 8 generic tool surfaces
that scenarios can spawn into the simulator's desktop:

    table | sql | kanban | dashboard | alerts | editor | terminal | crm

Each tool document lives in `tools_col` with the envelope:

    {
        "tool_id":         "tl_xxxxxxxx",
        "simulation_id":   "sim_<user>",
        "scenario_id":     "scn_xxxxxxxxxx" | None,
        "name":            "Sage Spreadsheet",
        "tool_type":       "table",
        "icon":            "table"            # frontend SVG key
        "tint":            "#0D9488",
        "category":        "finance|security|engineering|marketing|generic",
        "data":            { ...type-specific... },
        "actions_log":     [ {ts, action, params, ok, points_delta} ],
        "created_at":      iso8601,
    }

`dispatch_action(tool, action, params)` returns a `DispatchResult`:

    {
        "ok":               bool,
        "message":          str,                  # human-readable result
        "points_delta":     int,                  # award / penalty
        "manager_feedback": Optional[str],        # if non-empty, materialised
                                                  # as a manager email flag
        "side_effects":     list[dict]            # cross-tool deltas applied
                                                  # to `data` BEFORE persist
        "slack_summary":    Optional[str],        # broadcast summary
        "events":           list[(name, payload)] # to relay through emit()
    }

Validators are pure: they never touch Mongo. The caller (app.py) persists
the mutated `tool` document and emits the events. This separation keeps
the dispatcher unit-testable without a database.

Designed to be extended: every validator returns the same envelope, so a
new `tool_type` only requires (a) a builder shape and (b) a validator.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from utctime import utcnow
from typing import Any, Callable, Optional
import re


SUPPORTED_TYPES: tuple[str, ...] = (
    "table", "sql", "kanban", "dashboard",
    "alerts", "editor", "terminal", "crm",
)

# Hint to the frontend on which icon to render when a scenario doesn't
# specify one. Lower-case keys match the SVG sprite in index.html.
DEFAULT_ICON: dict[str, str] = {
    "table": "table",
    "sql": "database",
    "kanban": "kanban",
    "dashboard": "chart",
    "alerts": "shield",
    "editor": "code",
    "terminal": "terminal",
    "crm": "users",
}

# Default tints — Apple-quality palette, all desaturated, all readable
# against the simulator's slate-grey window chrome.
DEFAULT_TINT: dict[str, str] = {
    "table":     "#0D9488",
    "sql":       "#6366F1",
    "kanban":    "#0EA5E9",
    "dashboard": "#F59E0B",
    "alerts":    "#EF4444",
    "editor":    "#10B981",
    "terminal":  "#475569",
    "crm":       "#EC4899",
}


@dataclass
class DispatchResult:
    ok: bool
    message: str = ""
    points_delta: int = 0
    manager_feedback: Optional[str] = None
    side_effects: list[dict] = field(default_factory=list)
    slack_summary: Optional[str] = None
    events: list[tuple[str, dict]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict turns tuples in `events` into lists; re-tuple is unnecessary
        # since we serialise to JSON for the API anyway.
        return d


# ──────────────────────────────────────────────────────────────────
# Per-type validators. Each receives (tool_doc, params) and returns
# a DispatchResult. They mutate tool_doc["data"] in-place when the
# action succeeds; the caller then persists.
# ──────────────────────────────────────────────────────────────────

def _err(msg: str) -> DispatchResult:
    return DispatchResult(ok=False, message=msg, points_delta=0)


def validate_table_action(tool: dict, action: str, params: dict) -> DispatchResult:
    """
    table.update_cell  → params: {row: int, column: str, value: str|number}

    The data shape is:
        data.columns          : [str]
        data.rows             : [[cells]]      (parallel to columns)
        data.expected_corrections : [
            {row, column, expected_value, points, label}
        ]
    """
    data = tool.get("data") or {}
    if action != "update_cell":
        return _err(f"unsupported table action: {action}")

    rows = data.get("rows") or []
    cols = data.get("columns") or []
    try:
        ridx = int(params.get("row"))
    except (TypeError, ValueError):
        return _err("row must be an integer index")
    col_name = (params.get("column") or "").strip()
    new_val = params.get("value")

    if not (0 <= ridx < len(rows)):
        return _err(f"row {ridx} out of range (0..{len(rows)-1})")
    if col_name not in cols:
        return _err(f"unknown column: {col_name!r}")
    cidx = cols.index(col_name)

    old = rows[ridx][cidx] if cidx < len(rows[ridx]) else None
    while len(rows[ridx]) <= cidx:
        rows[ridx].append(None)
    rows[ridx][cidx] = new_val

    # Score against expected_corrections
    expected = data.get("expected_corrections") or []
    match = next(
        (c for c in expected
         if int(c.get("row", -1)) == ridx
         and c.get("column") == col_name
         and not c.get("done")),
        None,
    )
    if match and _values_equivalent(new_val, match.get("expected_value")):
        match["done"] = True
        pts = int(match.get("points") or 8)
        label = match.get("label") or f"{col_name} on row {ridx}"
        return DispatchResult(
            ok=True,
            message=f"Correct edit — {label} (+{pts}pts).",
            points_delta=pts,
            slack_summary=f"{tool.get('name','Spreadsheet')}: {label} corrected.",
            events=[("tool_table_corrected", {"tool_id": tool["tool_id"], "label": label})],
            side_effects=[{"path": "rows", "row": ridx, "col": col_name, "old": old, "new": new_val}],
        )

    # Edited a cell that isn't in expected_corrections — neutral.
    if not match:
        return DispatchResult(
            ok=True,
            message="Cell updated.",
            points_delta=0,
            side_effects=[{"path": "rows", "row": ridx, "col": col_name, "old": old, "new": new_val}],
        )

    # Wrong value for an expected correction → penalty + manager flag.
    return DispatchResult(
        ok=True,
        message="Value recorded but does not match expectation.",
        points_delta=-3,
        manager_feedback=(
            f"You changed {col_name} on row {ridx+1} of {tool.get('name','the table')} "
            f"to {new_val!r}, but the figure should be {match.get('expected_value')!r}. "
            "Re-check the source artefact before posting."
        ),
        side_effects=[{"path": "rows", "row": ridx, "col": col_name, "old": old, "new": new_val}],
    )


def validate_sql_action(tool: dict, action: str, params: dict) -> DispatchResult:
    """
    sql.run_query  → params: {query: str}

    data shape:
        data.engine        : "postgres"|"bigquery"|"mysql"|"sqlite"
        data.schema        : [{name, columns:[{name,type}]}]
        data.sample_data   : { "<table>": [ {col:val} ... ] }
        data.challenges    : [
            {id, prompt, expected_substrings:[...], expected_min_rows:int,
             points, hint?}
        ]
    """
    data = tool.get("data") or {}
    if action != "run_query":
        return _err(f"unsupported sql action: {action}")

    q = (params.get("query") or "").strip()
    if not q:
        return _err("query cannot be empty")
    if len(q) > 4000:
        return _err("query too long (max 4000 chars)")
    # Read-only sandbox: any DDL/DML rejected.
    forbidden = re.compile(r"\b(drop|truncate|delete|update|insert|alter|grant)\b", re.I)
    if forbidden.search(q):
        return DispatchResult(
            ok=True, message="Read-only sandbox — write statements are blocked.",
            points_delta=-2,
            manager_feedback="Production-safety reminder: SQL sandbox rejects writes. Use SELECT only.",
        )

    challenges = [c for c in (data.get("challenges") or []) if not c.get("done")]
    qlow = q.lower()
    matched = None
    for c in challenges:
        subs = [s.lower() for s in (c.get("expected_substrings") or []) if s]
        if subs and all(s in qlow for s in subs):
            matched = c
            break

    # Approximate row count — naive but enough for a learning sandbox: count
    # rows in the FROM table that mention any value from a WHERE literal.
    sample = data.get("sample_data") or {}
    approx_rows = _approximate_query_rows(q, sample)

    if matched is not None:
        if approx_rows < int(matched.get("expected_min_rows") or 0):
            return DispatchResult(
                ok=True, message=f"Query syntactically on-target but only matched {approx_rows} rows; expected at least {matched.get('expected_min_rows')}.",
                points_delta=0,
            )
        matched["done"] = True
        pts = int(matched.get("points") or 12)
        return DispatchResult(
            ok=True, message=f"Query passed: {matched.get('prompt','challenge')} (+{pts}pts).",
            points_delta=pts,
            slack_summary=f"{tool.get('name','SQL')}: '{matched.get('prompt','query')}' resolved.",
            events=[("tool_sql_solved", {"tool_id": tool["tool_id"], "challenge_id": matched.get("id"), "rows": approx_rows})],
        )

    return DispatchResult(
        ok=True,
        message=f"Query ran. {approx_rows} rows estimated. (No challenge matched.)",
        points_delta=0,
    )


def validate_kanban_action(tool: dict, action: str, params: dict) -> DispatchResult:
    """
    kanban.move_card  → params: {card_id: str, to_column: str}

    data shape:
        data.columns       : ["Backlog","In Progress","Review","Done"]
        data.cards         : [{id,title,column,priority,desc?, expected_column?}]
    """
    data = tool.get("data") or {}
    if action != "move_card":
        return _err(f"unsupported kanban action: {action}")

    card_id = (params.get("card_id") or "").strip()
    to_col = (params.get("to_column") or "").strip()
    if not card_id or not to_col:
        return _err("card_id and to_column are required")
    if to_col not in (data.get("columns") or []):
        return _err(f"unknown column: {to_col!r}")
    card = next((c for c in (data.get("cards") or []) if c.get("id") == card_id), None)
    if not card:
        return _err(f"card not found: {card_id}")

    old_col = card.get("column")
    card["column"] = to_col

    expected = card.get("expected_column")
    pts = 0
    msg = f"Moved {card_id} → {to_col}."
    feedback = None
    if expected:
        if expected == to_col:
            pts = int(card.get("points") or 8)
            msg += f" Correct destination (+{pts}pts)."
        elif old_col == expected:
            # Regression — moved a correctly-placed card away.
            pts = -2
            feedback = (
                f"Card {card_id} ({card.get('title','?')}) was already in the correct column "
                f"({expected}). Moving it back to {to_col} regresses progress."
            )

    return DispatchResult(
        ok=True, message=msg, points_delta=pts, manager_feedback=feedback,
        slack_summary=f"{tool.get('name','Board')}: {card.get('title',card_id)} moved {old_col} → {to_col}.",
        events=[("tool_kanban_moved", {"tool_id": tool["tool_id"], "card_id": card_id,
                                       "from": old_col, "to": to_col,
                                       "title": card.get("title",""),
                                       "is_correct": (expected == to_col) if expected else None})],
        side_effects=[{"path": "cards", "card_id": card_id, "from": old_col, "to": to_col}],
    )


def validate_dashboard_action(tool: dict, action: str, params: dict) -> DispatchResult:
    """
    dashboard.answer_question  → params: {question_id: str, answer: str|number}

    data shape:
        data.metrics    : [{label, value, trend?}]
        data.charts     : [{title, type, series:[...] }]
        data.questions  : [{id, prompt, expected, points, tolerance?, done?}]
    """
    data = tool.get("data") or {}
    if action != "answer_question":
        return _err(f"unsupported dashboard action: {action}")

    qid = (params.get("question_id") or "").strip()
    ans = params.get("answer")
    questions = data.get("questions") or []
    q = next((x for x in questions if x.get("id") == qid), None)
    if not q:
        return _err(f"unknown question id: {qid!r}")
    if q.get("done"):
        return _err("question already answered")

    expected = q.get("expected")
    tol = float(q.get("tolerance") or 0.0)
    if _answer_matches(ans, expected, tol):
        q["done"] = True
        pts = int(q.get("points") or 10)
        return DispatchResult(
            ok=True, message=f"Correct (+{pts}pts).", points_delta=pts,
            slack_summary=f"{tool.get('name','Dashboard')}: {q.get('prompt','question')} answered.",
            events=[("tool_dashboard_answered", {"tool_id": tool["tool_id"], "question_id": qid})],
        )
    return DispatchResult(
        ok=True, message=f"Answer recorded but not correct. Expected {expected!r}.",
        points_delta=-2,
        manager_feedback=(
            f"On the {tool.get('name','dashboard')}, you answered '{ans}' to "
            f"{q.get('prompt','a question')}. The chart shows the figure is {expected}."
        ),
    )


def validate_alerts_action(tool: dict, action: str, params: dict) -> DispatchResult:
    """
    alerts.triage  → params: {alert_id: str, action: 'escalate'|'resolve'|'investigate'}

    data shape:
        data.alerts      : [{id, severity, title, details, status,
                             correct_action: 'escalate'|'resolve', points}]
    """
    data = tool.get("data") or {}
    if action != "triage":
        return _err(f"unsupported alerts action: {action}")

    aid = (params.get("alert_id") or "").strip()
    user_action = (params.get("action") or "").strip().lower()
    if user_action not in ("escalate", "resolve", "investigate"):
        return _err("action must be escalate | resolve | investigate")
    alert = next((a for a in (data.get("alerts") or []) if a.get("id") == aid), None)
    if not alert:
        return _err(f"unknown alert: {aid}")

    alert["status"] = user_action
    correct = alert.get("correct_action") or "resolve"
    base = int(alert.get("points") or 10)
    if user_action == correct or (correct == "escalate" and user_action == "investigate"):
        return DispatchResult(
            ok=True, message=f"Alert {aid} {user_action}d (+{base}pts).",
            points_delta=base,
            slack_summary=f"{tool.get('name','SIEM')}: [{alert.get('severity','?').upper()}] {alert.get('title','')} {user_action.upper()}D.",
            events=[("tool_alert_triaged", {"tool_id": tool["tool_id"], "alert_id": aid,
                                            "action": user_action, "alert": alert,
                                            "correct": True})],
        )
    return DispatchResult(
        ok=True,
        message=f"You {user_action}d a {alert.get('severity','?').upper()} alert that should be {correct}d.",
        points_delta=-int(base * 0.5),
        manager_feedback=(
            f"Alert {aid} ({alert.get('title','')}) was a {alert.get('severity','?').upper()} "
            f"item that needed to be {correct}d, not {user_action}d. Review the details: {alert.get('details','')[:160]}"
        ),
        events=[("tool_alert_triaged", {"tool_id": tool["tool_id"], "alert_id": aid,
                                        "action": user_action, "alert": alert,
                                        "correct": False})],
    )


def validate_editor_action(tool: dict, action: str, params: dict) -> DispatchResult:
    """
    editor.submit_code  → params: {file: str, content: str}

    data shape:
        data.language          : "python"|"javascript"|"typescript"|"sql"|...
        data.files             : [{name, content, readonly?}]
        data.expected_substrings: [str]      # all must appear in content
        data.forbidden_substrings: [str]     # none must appear
        data.points            : int (default 15)
        data.solved            : bool
    """
    data = tool.get("data") or {}
    if action != "submit_code":
        return _err(f"unsupported editor action: {action}")
    if data.get("solved"):
        return _err("already submitted and solved")

    fname = (params.get("file") or "").strip()
    content = params.get("content") or ""
    if not isinstance(content, str):
        return _err("content must be a string")
    if len(content) > 20_000:
        return _err("submission too large (max 20k chars)")
    files = data.get("files") or []
    file_doc = next((f for f in files if f.get("name") == fname), None)
    if not file_doc:
        return _err(f"unknown file: {fname!r}")
    if file_doc.get("readonly"):
        return _err(f"{fname} is read-only")

    file_doc["content"] = content

    expected_subs = data.get("expected_substrings") or []
    forbidden_subs = data.get("forbidden_substrings") or []
    missing = [s for s in expected_subs if s not in content]
    illegal = [s for s in forbidden_subs if s in content]

    if not missing and not illegal:
        data["solved"] = True
        pts = int(data.get("points") or 15)
        return DispatchResult(
            ok=True, message=f"Submission accepted (+{pts}pts).", points_delta=pts,
            slack_summary=f"{tool.get('name','Editor')}: {fname} submitted and accepted.",
            events=[("tool_editor_solved", {"tool_id": tool["tool_id"], "file": fname})],
        )

    parts = []
    if missing:
        parts.append("Missing required: " + ", ".join(missing[:3]))
    if illegal:
        parts.append("Should not contain: " + ", ".join(illegal[:3]))
    return DispatchResult(
        ok=True, message="Submission saved but does not pass review. " + " | ".join(parts),
        points_delta=-2,
        manager_feedback=f"Code review on {fname}: " + " ".join(parts),
    )


def validate_terminal_action(tool: dict, action: str, params: dict) -> DispatchResult:
    """
    terminal.run_command  → params: {command: str}

    data shape:
        data.shell           : "bash"|"zsh"|"powershell"
        data.scrollback      : [{cmd, output, ts}]
        data.challenge       : "Restart the failing pod and confirm rollout"
        data.expected_command_patterns : [str]   # regex (any match)
        data.forbidden_patterns        : [str]
        data.points          : int
        data.solved          : bool
    """
    data = tool.get("data") or {}
    if action != "run_command":
        return _err(f"unsupported terminal action: {action}")
    if data.get("solved"):
        return _err("challenge already solved")

    cmd = (params.get("command") or "").strip()
    if not cmd:
        return _err("command cannot be empty")
    if len(cmd) > 1000:
        return _err("command too long")

    # Append to scrollback regardless.
    scrollback = data.setdefault("scrollback", [])
    forbidden = data.get("forbidden_patterns") or []
    expected = data.get("expected_command_patterns") or []
    out_text = ""
    pts = 0
    feedback = None

    bad = next((p for p in forbidden if re.search(p, cmd)), None)
    if bad:
        out_text = f"$ {cmd}\nrefused: pattern matches forbidden command policy ({bad})"
        pts = -3
        feedback = f"Terminal: command '{cmd}' violates policy ({bad}). Choose a safer approach."
    else:
        good = next((p for p in expected if re.search(p, cmd)), None)
        if good is not None:
            data["solved"] = True
            pts = int(data.get("points") or 12)
            out_text = f"$ {cmd}\nok — challenge resolved (+{pts}pts)"
        else:
            out_text = f"$ {cmd}\n(no error, no progress against the challenge)"

    scrollback.append({"cmd": cmd, "output": out_text, "ts": utcnow().isoformat()})
    # Cap scrollback to last 80 entries.
    if len(scrollback) > 80:
        del scrollback[: len(scrollback) - 80]

    return DispatchResult(
        ok=True, message=out_text, points_delta=pts,
        manager_feedback=feedback,
        slack_summary=(f"{tool.get('name','Terminal')}: challenge '{data.get('challenge','task')}' resolved."
                       if data.get("solved") else None),
        events=[("tool_terminal_solved", {"tool_id": tool["tool_id"]})] if data.get("solved") else [],
    )


def validate_crm_action(tool: dict, action: str, params: dict) -> DispatchResult:
    """
    crm.advance_stage  → params: {contact_id: str, stage: str, note?: str}

    data shape:
        data.stages       : ["Lead","Qualified","Proposal","Closed Won","Closed Lost"]
        data.contacts     : [{id, name, company, owner, stage,
                              last_contact_days, expected_stage?, value?}]
        data.stalled_threshold_days : int  (default 14)
    """
    data = tool.get("data") or {}
    if action != "advance_stage":
        return _err(f"unsupported crm action: {action}")

    cid = (params.get("contact_id") or "").strip()
    new_stage = (params.get("stage") or "").strip()
    if new_stage not in (data.get("stages") or []):
        return _err(f"unknown stage: {new_stage!r}")
    contact = next((c for c in (data.get("contacts") or []) if c.get("id") == cid), None)
    if not contact:
        return _err(f"contact not found: {cid}")

    old = contact.get("stage")
    contact["stage"] = new_stage
    contact["last_contact_days"] = 0

    expected = contact.get("expected_stage")
    pts = 0
    feedback = None
    if expected:
        if expected == new_stage:
            pts = int(contact.get("points") or 10)
        elif (data.get("stages") or []).index(new_stage) > (data.get("stages") or []).index(expected):
            pts = -3
            feedback = (
                f"You advanced {contact.get('name','?')} to {new_stage}, but their "
                f"history only supports {expected}. Don't inflate pipeline stages."
            )

    msg = f"{contact.get('name','?')}: {old} → {new_stage}" + (f" (+{pts}pts)" if pts > 0 else "")
    return DispatchResult(
        ok=True, message=msg, points_delta=pts, manager_feedback=feedback,
        slack_summary=f"{tool.get('name','CRM')}: {contact.get('name','?')} moved {old} → {new_stage}.",
        events=[("tool_crm_advanced", {"tool_id": tool["tool_id"], "contact_id": cid,
                                       "from": old, "to": new_stage})],
    )


# ──────────────────────────────────────────────────────────────────
# Dispatcher entry point
# ──────────────────────────────────────────────────────────────────

VALIDATORS: dict[str, Callable[[dict, str, dict], DispatchResult]] = {
    "table":     validate_table_action,
    "sql":       validate_sql_action,
    "kanban":    validate_kanban_action,
    "dashboard": validate_dashboard_action,
    "alerts":    validate_alerts_action,
    "editor":    validate_editor_action,
    "terminal":  validate_terminal_action,
    "crm":       validate_crm_action,
}


def dispatch_action(tool: dict, action: str, params: dict | None = None) -> DispatchResult:
    """Validate and apply a write action against `tool["data"]` (in-place)."""
    params = params or {}
    if not isinstance(action, str) or not action:
        return _err("action is required")
    tt = (tool.get("tool_type") or "").lower()
    fn = VALIDATORS.get(tt)
    if not fn:
        return _err(f"no validator for tool_type {tt!r}")
    try:
        return fn(tool, action.strip().lower(), params)
    except Exception as ex:
        return _err(f"dispatch error: {type(ex).__name__}: {ex}")


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _values_equivalent(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a == b
    # Numeric equivalence to ±0.005
    try:
        return abs(float(a) - float(b)) < 0.01
    except (TypeError, ValueError):
        return str(a).strip().lower() == str(b).strip().lower()


def _answer_matches(ans: Any, expected: Any, tol: float) -> bool:
    if isinstance(expected, (int, float)):
        try:
            return abs(float(ans) - float(expected)) <= max(tol, 0.0001)
        except (TypeError, ValueError):
            return False
    return str(ans).strip().lower() == str(expected).strip().lower()


def _approximate_query_rows(query: str, sample: dict) -> int:
    """Tiny SELECT executor for the SQL sandbox.

    Not a real engine — just enough to give the user feedback on whether
    their WHERE clause narrowed results. Picks the first table mentioned
    in FROM and counts rows whose values intersect with literal values
    quoted in the query. Returns total row count if no literals.
    """
    qlow = query.lower()
    m = re.search(r"\bfrom\s+([a-z_][a-z0-9_]*)", qlow)
    if not m:
        return 0
    table = m.group(1)
    rows = sample.get(table) or []
    literals = re.findall(r"'([^']{1,80})'", query)
    if not literals:
        # If no WHERE literal, return all (LIMIT honoured loosely).
        m_lim = re.search(r"\blimit\s+(\d+)", qlow)
        if m_lim:
            return min(int(m_lim.group(1)), len(rows))
        return len(rows)
    lits = [s.lower() for s in literals]
    n = 0
    for row in rows:
        hay = " ".join(str(v).lower() for v in row.values())
        if any(l in hay for l in lits):
            n += 1
    return n


# ──────────────────────────────────────────────────────────────────
# Builders — convert short LLM-friendly specs into well-formed tool docs
# ──────────────────────────────────────────────────────────────────

def normalise_tool_spec(spec: dict) -> Optional[dict]:
    """Sanitise a tool spec coming from the scenario extractor.

    Returns None if the spec is unusable. Output is safe to insert into
    `tools_col` (caller adds tool_id, simulation_id, scenario_id, created_at).
    """
    if not isinstance(spec, dict):
        return None
    tt = (spec.get("tool_type") or spec.get("type") or "").lower().strip()
    if tt not in SUPPORTED_TYPES:
        return None
    name = (spec.get("name") or tt.title())[:80]
    icon = (spec.get("icon") or DEFAULT_ICON.get(tt) or "tool")[:24]
    tint = (spec.get("tint") or DEFAULT_TINT.get(tt) or "#475569")[:9]
    category = (spec.get("category") or "generic")[:24]
    data = spec.get("data") or {}
    if not isinstance(data, dict):
        return None
    cleaned = _normalise_data_for_type(tt, data)
    if cleaned is None:
        return None
    return {
        "name": name,
        "tool_type": tt,
        "icon": icon,
        "tint": tint,
        "category": category,
        "data": cleaned,
        "actions_log": [],
    }


def _normalise_data_for_type(tt: str, d: dict) -> Optional[dict]:
    if tt == "table":
        cols = [str(c)[:40] for c in (d.get("columns") or [])][:12]
        if not cols:
            return None
        rows = [
            [_short(v) for v in (r if isinstance(r, list) else [])][: len(cols)]
            for r in (d.get("rows") or [])[:200]
        ]
        ec = []
        for c in (d.get("expected_corrections") or [])[:24]:
            try:
                ec.append({
                    "row": int(c.get("row")),
                    "column": str(c.get("column"))[:40],
                    "expected_value": c.get("expected_value"),
                    "points": int(c.get("points") or 8),
                    "label": str(c.get("label") or "")[:80],
                    "done": False,
                })
            except (TypeError, ValueError):
                continue
        return {"columns": cols, "rows": rows, "expected_corrections": ec}

    if tt == "sql":
        engine = (d.get("engine") or "postgres")[:24]
        schema = []
        for t in (d.get("schema") or [])[:8]:
            tname = str(t.get("name") or "")[:40]
            if not tname:
                continue
            cols = [{"name": str(c.get("name"))[:40], "type": str(c.get("type") or "text")[:24]}
                    for c in (t.get("columns") or [])[:16]]
            schema.append({"name": tname, "columns": cols})
        sample = {}
        for tname, rows in (d.get("sample_data") or {}).items():
            if not isinstance(rows, list):
                continue
            sample[str(tname)[:40]] = [
                {str(k)[:40]: _short(v) for k, v in row.items()}
                for row in rows[:80] if isinstance(row, dict)
            ]
        challenges = []
        for c in (d.get("challenges") or [])[:6]:
            challenges.append({
                "id": str(c.get("id") or f"ch_{len(challenges)+1}")[:24],
                "prompt": str(c.get("prompt") or "")[:200],
                "expected_substrings": [str(s)[:40] for s in (c.get("expected_substrings") or [])][:6],
                "expected_min_rows": int(c.get("expected_min_rows") or 1),
                "points": int(c.get("points") or 12),
                "hint": str(c.get("hint") or "")[:200],
                "done": False,
            })
        return {"engine": engine, "schema": schema, "sample_data": sample, "challenges": challenges}

    if tt == "kanban":
        cols = [str(c)[:40] for c in (d.get("columns") or ["Backlog", "In Progress", "Review", "Done"])][:8]
        cards = []
        for c in (d.get("cards") or [])[:60]:
            col = c.get("column")
            if col not in cols:
                col = cols[0]
            cards.append({
                "id": str(c.get("id") or f"k_{len(cards)+1}")[:24],
                "title": str(c.get("title") or "")[:120],
                "column": col,
                "priority": str(c.get("priority") or "medium")[:12],
                "desc": str(c.get("desc") or "")[:400],
                "expected_column": (c.get("expected_column") if c.get("expected_column") in cols else None),
                "points": int(c.get("points") or 8),
            })
        return {"columns": cols, "cards": cards}

    if tt == "dashboard":
        metrics = [{"label": str(m.get("label") or "")[:40],
                    "value": _short(m.get("value")),
                    "trend": str(m.get("trend") or "")[:24]}
                   for m in (d.get("metrics") or [])[:12]]
        charts = []
        for ch in (d.get("charts") or [])[:6]:
            charts.append({
                "title": str(ch.get("title") or "")[:60],
                "type": (ch.get("type") or "bar")[:16],
                "series": [
                    {"label": str(s.get("label") or "")[:40],
                     "value": _to_number(s.get("value"))}
                    for s in (ch.get("series") or [])[:24]
                ],
            })
        questions = []
        for q in (d.get("questions") or [])[:8]:
            questions.append({
                "id": str(q.get("id") or f"q_{len(questions)+1}")[:24],
                "prompt": str(q.get("prompt") or "")[:240],
                "expected": q.get("expected"),
                "tolerance": float(q.get("tolerance") or 0.0),
                "points": int(q.get("points") or 10),
                "done": False,
            })
        return {"metrics": metrics, "charts": charts, "questions": questions}

    if tt == "alerts":
        # Map LLM-invented verbs back onto the validator's allowed set
        # {escalate, resolve, investigate}.
        _CA_ALIASES = {
            "escalate": "escalate", "page": "escalate", "alert": "escalate",
            "raise": "escalate", "report": "escalate",
            "resolve": "resolve", "close": "resolve", "dismiss": "resolve",
            "ignore": "resolve", "fix": "resolve",
            "investigate": "investigate", "acknowledge": "investigate",
            "ack": "investigate", "triage": "investigate", "review": "investigate",
        }
        alerts = []
        for a in (d.get("alerts") or [])[:24]:
            raw = str(a.get("correct_action") or "resolve").strip().lower()
            ca = _CA_ALIASES.get(raw, "investigate")
            alerts.append({
                "id": str(a.get("id") or f"a_{len(alerts)+1}")[:24],
                "severity": str(a.get("severity") or "MEDIUM")[:8].upper(),
                "title": str(a.get("title") or "")[:160],
                "details": str(a.get("details") or "")[:600],
                "status": "open",
                "correct_action": ca,
                "points": int(a.get("points") or 10),
            })
        return {"alerts": alerts}

    if tt == "editor":
        files = []
        for f in (d.get("files") or [])[:6]:
            files.append({
                "name": str(f.get("name") or "main")[:60],
                "content": str(f.get("content") or "")[:8000],
                "readonly": bool(f.get("readonly")),
            })
        if not files:
            return None
        return {
            "language": (d.get("language") or "python")[:16],
            "files": files,
            "instructions": str(d.get("instructions") or "")[:600],
            "expected_substrings": [str(s)[:80] for s in (d.get("expected_substrings") or [])][:8],
            "forbidden_substrings": [str(s)[:80] for s in (d.get("forbidden_substrings") or [])][:8],
            "points": int(d.get("points") or 15),
            "solved": False,
        }

    if tt == "terminal":
        return {
            "shell": (d.get("shell") or "bash")[:16],
            "scrollback": [],
            "challenge": str(d.get("challenge") or "")[:240],
            "expected_command_patterns": [str(p)[:80] for p in (d.get("expected_command_patterns") or [])][:6],
            "forbidden_patterns": [str(p)[:80] for p in (d.get("forbidden_patterns") or [])][:6],
            "points": int(d.get("points") or 12),
            "solved": False,
        }

    if tt == "crm":
        stages = [str(s)[:24] for s in (d.get("stages") or ["Lead", "Qualified", "Proposal", "Closed Won", "Closed Lost"])][:8]
        contacts = []
        for c in (d.get("contacts") or [])[:40]:
            stage = c.get("stage") if c.get("stage") in stages else stages[0]
            expected = c.get("expected_stage") if c.get("expected_stage") in stages else None
            contacts.append({
                "id": str(c.get("id") or f"cn_{len(contacts)+1}")[:24],
                "name": str(c.get("name") or "")[:80],
                "company": str(c.get("company") or "")[:80],
                "owner": str(c.get("owner") or "you")[:40],
                "stage": stage,
                "value": _to_number(c.get("value")),
                "last_contact_days": int(c.get("last_contact_days") or 0),
                "expected_stage": expected,
                "points": int(c.get("points") or 8),
                "notes": str(c.get("notes") or "")[:240],
            })
        return {
            "stages": stages,
            "contacts": contacts,
            "stalled_threshold_days": int(d.get("stalled_threshold_days") or 14),
        }

    return None


def _short(v: Any) -> Any:
    if isinstance(v, str):
        return v[:200]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)[:200]


def _to_number(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
