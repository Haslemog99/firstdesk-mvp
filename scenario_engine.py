"""
FirstDesk — Adaptive Scenario Engine

Turns real (or seeded) job descriptions into multi-tool workplace scenarios.
The engine is the moat: instead of four hard-coded simulations, every user
gets a fresh day's work that mirrors what an actual junior in that role does.

Pipeline
────────
   JobSource              ScenarioExtractor          Materializer
   ─────────              ─────────────────          ────────────
   AdzunaSource    ──►    LLM(JD) → tasks    ──►    Insert into:
   SeedFileSource         + npcs + tools             emails_col
                          + tool_workflow            tasks_col
                          → scenarios_col            slack_col
                                                     siem/jira/sage/...

Each generated task carries `tools_required` (3–5 tools) and `tool_workflow`
(the cross-tool steps the user must touch). The existing FirstDesk event
bus (discrepancy → ledger correction email, alert → incident ticket,
campaign update → manager review) means those tools end up *talking to
each other* the moment the user starts working.

Design decisions
────────────────
• Dependency injection: the engine takes Mongo collections + an LLM as
  arguments. No imports from `app`, no circular references.
• Source plug-in surface: anything that yields `JobPosting` objects can be
  registered. Adzuna ships behind two env vars; SeedFile is always-on so
  the demo never blocks on network.
• Strict JSON schema for extraction. The LLM returns one envelope; we
  reject anything that doesn't parse and fall back gracefully.
• Idempotent ingestion: jobs are upserted by `ext_id`, scenarios by hash.
• Per-user uniqueness ledger so a returning user never sees the same
  scenario twice until the bank is exhausted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import httpx

from tool_dispatch import normalise_tool_spec

log = logging.getLogger("firstdesk.scenario")

# ──────────────────────────────────────────────────────────────────
# Domain types
# ──────────────────────────────────────────────────────────────────

VALID_CAREERS = ("accountant", "cybersecurity", "marketing", "developer")

# Each career maps to the canonical tool palette the simulator already supports.
# Generated tasks must reference 3–5 of these. Anything outside is dropped.
CAREER_TOOL_PALETTE: dict[str, tuple[str, ...]] = {
    "accountant":    ("outlook", "sage",       "tasks", "slack", "sheets"),
    "cybersecurity": ("outlook", "siem",       "tasks", "slack", "jira"),
    "marketing":     ("outlook", "campaigns",  "tasks", "slack", "analytics"),
    "developer":     ("outlook", "jira",       "tasks", "slack", "ci"),
}

DEFAULT_SLACK_CHANNELS: dict[str, tuple[str, ...]] = {
    "accountant":    ("#finance", "#general"),
    "cybersecurity": ("#soc-alerts", "#incidents", "#general"),
    "marketing":     ("#marketing", "#partnerships", "#general"),
    "developer":     ("#engineering", "#incidents", "#general"),
}


@dataclass
class JobPosting:
    """A normalised job description, regardless of source."""
    ext_id: str
    career: str
    title: str
    company: str
    description: str
    location: str = ""
    seniority: str = "junior"
    source: str = "unknown"
    tools_mentioned: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable hash used to dedupe scenarios derived from the same JD."""
        h = hashlib.sha256()
        h.update(self.career.encode())
        h.update(b"\x1f")
        h.update(self.title.lower().strip().encode())
        h.update(b"\x1f")
        h.update(self.description.strip()[:4000].encode())
        return h.hexdigest()[:16]


@dataclass
class ScenarioTemplate:
    """A materialisable workplace day. Persisted in `scenarios_col`."""
    scenario_id: str
    career: str
    title: str
    company: str
    fingerprint: str
    source_job_ext_id: str
    emails: list[dict]
    tasks: list[dict]                      # each task: tools_required + tool_workflow
    tool_data: dict[str, list[dict]]       # keyed by tool name → seed rows
    system_events: list[dict]
    tools: list[dict] = field(default_factory=list)  # generic tool docs (table/sql/...)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ──────────────────────────────────────────────────────────────────
# Job sources (plug-in surface)
# ──────────────────────────────────────────────────────────────────

class JobSource(ABC):
    """Abstract source. Implementations yield `JobPosting`s on demand."""

    name: str = "abstract"

    @abstractmethod
    def fetch(self, career: str, limit: int = 10) -> list[JobPosting]:
        """Return up to `limit` postings for `career`. Must not raise on empty."""


class SeedFileSource(JobSource):
    """Ships with the repo. Reads `scenarios_seed.json`. Always available."""

    name = "seed_file"

    def __init__(self, path: str | Path = "scenarios_seed.json") -> None:
        self.path = Path(path)

    def fetch(self, career: str, limit: int = 10) -> list[JobPosting]:
        if career not in VALID_CAREERS:
            return []
        if not self.path.exists():
            log.warning("seed_source.missing path=%s", self.path)
            return []
        try:
            data = json.loads(self.path.read_text())
        except Exception as ex:
            log.exception("seed_source.parse_failed path=%s err=%s", self.path, ex)
            return []
        out: list[JobPosting] = []
        for j in data.get("jobs", []):
            if j.get("career") != career:
                continue
            out.append(JobPosting(
                ext_id=str(j["ext_id"]),
                career=j["career"],
                title=j.get("title", "Untitled"),
                company=j.get("company", "Anonymous Co"),
                description=j.get("description", ""),
                location=j.get("location", ""),
                seniority=j.get("seniority", "junior"),
                source="seed_file",
                tools_mentioned=list(j.get("tools_mentioned", [])),
                raw=j,
            ))
            if len(out) >= limit:
                break
        return out


class AdzunaSource(JobSource):
    """Live Adzuna API. Disabled unless ADZUNA_APP_ID + ADZUNA_APP_KEY are set."""

    name = "adzuna"
    BASE = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"

    # Map our internal careers to Adzuna search terms.
    QUERY: dict[str, str] = {
        "accountant":    "junior accountant",
        "cybersecurity": "junior soc analyst",
        "marketing":     "junior marketing coordinator",
        "developer":     "junior software engineer",
    }

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_key: Optional[str] = None,
        country: str = "gb",
        timeout: float = 6.0,
    ) -> None:
        self.app_id = app_id or os.environ.get("ADZUNA_APP_ID", "")
        self.app_key = app_key or os.environ.get("ADZUNA_APP_KEY", "")
        self.country = country
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.app_id and self.app_key)

    def fetch(self, career: str, limit: int = 10) -> list[JobPosting]:
        if not self.enabled or career not in self.QUERY:
            return []
        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": self.QUERY[career],
            "results_per_page": min(max(limit, 1), 50),
            "content-type": "application/json",
        }
        try:
            url = self.BASE.format(country=self.country)
            r = httpx.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            payload = r.json()
        except Exception as ex:
            log.warning("adzuna.fetch_failed career=%s kind=%s msg=%s",
                        career, type(ex).__name__, ex)
            return []
        out: list[JobPosting] = []
        for item in payload.get("results", []):
            ext_id = str(item.get("id") or item.get("redirect_url") or uuid.uuid4().hex)
            out.append(JobPosting(
                ext_id=f"adzuna:{ext_id}",
                career=career,
                title=item.get("title", "Untitled"),
                company=(item.get("company") or {}).get("display_name", "Anonymous Co"),
                description=item.get("description", "") or "",
                location=(item.get("location") or {}).get("display_name", ""),
                seniority="junior",
                source="adzuna",
                tools_mentioned=[],
                raw=item,
            ))
            if len(out) >= limit:
                break
        return out


# ──────────────────────────────────────────────────────────────────
# Scenario extraction (LLM-driven, JSON-strict)
# ──────────────────────────────────────────────────────────────────

EXTRACTOR_SYSTEM_PROMPT = """You convert a real junior-level job description into a one-day workplace simulation for a brand-new starter.

Output ONLY a single JSON object matching this schema (no prose, no markdown fences):

{
  "title": "Short scenario title, 6-10 words",
  "company_summary": "1-sentence positioning of the company",
  "starter_emails": [
    {
      "from_role": "manager|client|teammate|it|hr|director|system",
      "subject": "string",
      "body": "string, 2-5 short paragraphs, realistic UK workplace tone",
      "priority": "high|medium|low",
      "category": "task|client|teammate|onboarding|info",
      "requires_action": true
    }
  ],
  "tasks": [
    {
      "title": "Imperative, specific. References real artifacts.",
      "description": "1-3 sentences. Must hint at which tools to touch.",
      "priority": "high|medium|low",
      "deadline": "HH:MM or 'tomorrow' or weekday name",
      "points": 10-40,
      "tools_required": ["outlook","sage|siem|campaigns|jira","tasks","slack", "..."],
      "tool_workflow": [
        {"step": 1, "tool": "outlook",  "action": "Concrete action sentence"},
        {"step": 2, "tool": "sage",     "action": "..."},
        {"step": 3, "tool": "slack",    "action": "..."}
      ]
    }
  ],
  "slack_seed": [
    {"channel": "#finance", "from": "Senior Accountant", "text": "morning all — month-end close starts today, ping me if you need help"}
  ],
  "system_events": [
    {"tool": "sage|siem|campaigns|jira", "trigger_at": 6, "end_at": 9,
     "start_msg": "Brief outage banner",
     "end_msg": "Resolution banner",
     "points_for_report": 5}
  ],
  "tools": [
    /* 2-5 generic interactive tool surfaces. Pick the types that match the
       JD: a ledger spreadsheet → table; a SQL workbench → sql; a Jira board
       → kanban; an analytics screen → dashboard; a SOC console → alerts;
       a code repo → editor; a DevOps shell → terminal; a sales pipeline
       → crm. Every tool MUST contain at least one *graded* element
       (expected_corrections / challenges / questions / correct_action /
        expected_substrings / expected_command_patterns / expected_stage). */
    {
      "name": "Sage Spreadsheet",
      "tool_type": "table",
      "icon": "table", "tint": "#0D9488", "category": "finance",
      "data": {
        "columns": ["Date","Description","Debit","Credit","Ref"],
        "rows": [["2026-03-21","Conference ACCA","1580","","EXP-0345"]],
        "expected_corrections": [
          {"row": 0, "column": "Debit", "expected_value": 1850, "points": 10,
           "label": "Conference ACCA understated by £270"}
        ]
      }
    },
    {
      "name": "Analytics Workbench",
      "tool_type": "sql",
      "icon": "database", "tint": "#6366F1", "category": "data",
      "data": {
        "engine": "postgres",
        "schema": [{"name":"orders","columns":[{"name":"id","type":"int"},{"name":"status","type":"text"},{"name":"amount","type":"numeric"}]}],
        "sample_data": {"orders":[{"id":1,"status":"refunded","amount":120}]},
        "challenges": [{"id":"ch1","prompt":"List refunded orders","expected_substrings":["select","from orders","status = 'refunded'"],"expected_min_rows":1,"points":12}]
      }
    },
    {
      "name": "Sprint Board",
      "tool_type": "kanban",
      "icon": "kanban", "tint": "#0EA5E9", "category": "engineering",
      "data": {
        "columns": ["Backlog","In Progress","Review","Done"],
        "cards": [{"id":"BUG-301","title":"Fix Safari login crash","column":"In Progress","priority":"high","desc":"...","expected_column":"Review","points":10}]
      }
    },
    {
      "name": "SOC Triage Console",
      "tool_type": "alerts",
      "icon": "shield", "tint": "#EF4444", "category": "security",
      "data": {
        "alerts":[{"id":"A1","severity":"HIGH","title":"Outbound 14GB","details":"...","correct_action":"escalate","points":12}]
      }
    },
    {
      "name": "Postmortem Editor",
      "tool_type": "editor",
      "icon": "code", "tint": "#10B981", "category": "engineering",
      "data": {
        "language":"markdown",
        "files":[{"name":"postmortem.md","content":"# Incident\\n\\n## Timeline\\n\\n## Root cause\\n"}],
        "instructions":"Fill in timeline, root cause, action items.",
        "expected_substrings":["Root cause","Action items","Timeline"],
        "points":15
      }
    }
  ]
}

HARD RULES:
- Generate EXACTLY 8 starter_emails and 8-10 tasks.
- Every task MUST list 3 to 5 tools_required from the allowed palette: {palette}.
- Every task MUST include a tool_workflow with one step per required tool, in execution order.
- Slack is required in at least 60% of tasks because the team broadcasts updates.
- The "tools" array MUST contain 2-5 entries appropriate for this career.
- Each tool MUST embed at least one graded element so the user earns points by interacting with it.
- Use UK spelling. Be realistic about deadlines and points.
- No emojis. No markdown. JSON only."""


def _palette_for(career: str) -> str:
    return ", ".join(CAREER_TOOL_PALETTE.get(career, ()))


class ScenarioExtractor:
    """Wraps an LLM, produces a `ScenarioTemplate` from a `JobPosting`."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def extract(self, job: JobPosting) -> Optional[ScenarioTemplate]:
        palette = _palette_for(job.career)
        if not palette:
            log.warning("extractor.no_palette career=%s", job.career)
            return None
        sys_prompt = EXTRACTOR_SYSTEM_PROMPT.replace("{palette}", palette)
        human = (
            f"CAREER: {job.career}\n"
            f"COMPANY: {job.company}\n"
            f"TITLE: {job.title}\n"
            f"LOCATION: {job.location}\n\n"
            f"JOB DESCRIPTION:\n{job.description.strip()[:4000]}\n\n"
            "Now emit the JSON envelope."
        )
        try:
            resp = self.llm.invoke([("system", sys_prompt), ("human", human)])
            raw = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as ex:
            log.warning("extractor.llm_failed ext_id=%s kind=%s", job.ext_id, type(ex).__name__)
            return None
        envelope = self._parse_envelope(raw)
        if envelope is None:
            log.warning("extractor.parse_failed ext_id=%s preview=%r", job.ext_id, raw[:200] if raw else "")
            return None
        return self._validate_and_build(job, envelope)

    @staticmethod
    def _parse_envelope(text: str) -> Optional[dict]:
        if not text:
            return None
        s = text.strip()
        # Strip ``` fences if the model added them despite instruction.
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.MULTILINE)
        # Find the outermost {...}
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            # One retry: brace-balance heuristic for trailing commas etc.
            cleaned = re.sub(r",(\s*[}\]])", r"\1", s[start:end + 1])
            try:
                return json.loads(cleaned)
            except Exception:
                return None

    def _validate_and_build(self, job: JobPosting, env: dict) -> Optional[ScenarioTemplate]:
        palette = set(CAREER_TOOL_PALETTE.get(job.career, ()))
        tasks_raw = env.get("tasks") or []
        emails_raw = env.get("starter_emails") or []
        if len(tasks_raw) < 6:
            log.warning("extractor.too_few_tasks ext_id=%s n=%d", job.ext_id, len(tasks_raw))
            return None
        if len(emails_raw) < 5:
            log.warning("extractor.too_few_emails ext_id=%s n=%d", job.ext_id, len(emails_raw))
            return None

        clean_tasks: list[dict] = []
        for i, t in enumerate(tasks_raw):
            tools = [x.lower().strip() for x in (t.get("tools_required") or []) if isinstance(x, str)]
            tools = [x for x in tools if x in palette]
            if not (3 <= len(tools) <= 5):
                continue
            wf_raw = t.get("tool_workflow") or []
            workflow: list[dict] = []
            for j, step in enumerate(wf_raw, 1):
                tool = (step.get("tool") or "").lower().strip()
                if tool not in palette:
                    continue
                workflow.append({
                    "step": int(step.get("step") or j),
                    "tool": tool,
                    "action": str(step.get("action") or "")[:300],
                    "done": False,
                })
            if len(workflow) < 3:
                continue
            points = int(t.get("points") or 15)
            points = max(5, min(50, points))
            priority = (t.get("priority") or "medium").lower()
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            clean_tasks.append({
                "id": f"t_{uuid.uuid4().hex[:6]}",
                "title": str(t.get("title") or f"Task {i+1}")[:160],
                "description": str(t.get("description") or "")[:600],
                "priority": priority,
                "deadline": str(t.get("deadline") or "17:00")[:32],
                "status": "pending",
                "points": points,
                "tools_required": tools,
                "tool_workflow": workflow,
                "generated": True,
            })

        if len(clean_tasks) < 6:
            log.warning("extractor.task_validation_dropped ext_id=%s remaining=%d",
                        job.ext_id, len(clean_tasks))
            return None

        clean_emails: list[dict] = []
        for i, e in enumerate(emails_raw):
            from_role = (e.get("from_role") or "").lower().strip()
            if from_role not in ("manager", "client", "teammate", "it", "hr", "director", "system"):
                from_role = "manager"
            priority = (e.get("priority") or "medium").lower()
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            clean_emails.append({
                "from_role_key": from_role,                  # filled with real persona by Materializer
                "subject": str(e.get("subject") or "Welcome")[:140],
                "body": str(e.get("body") or "")[:2400],
                "priority": priority,
                "category": str(e.get("category") or "task")[:32],
                "requires_action": bool(e.get("requires_action", True)),
                "ord": i,
            })

        slack_seed: list[dict] = []
        for s in (env.get("slack_seed") or []):
            ch = str(s.get("channel") or "#general").strip()
            if not ch.startswith("#"):
                ch = "#" + ch
            slack_seed.append({
                "channel": ch[:40],
                "from": str(s.get("from") or "Teammate")[:60],
                "text": str(s.get("text") or "")[:600],
            })

        sys_events: list[dict] = []
        for ev in (env.get("system_events") or []):
            tool = (ev.get("tool") or "").lower().strip()
            if tool not in palette:
                continue
            try:
                trig = max(2, min(20, int(ev.get("trigger_at") or 6)))
                end = int(ev.get("end_at") or trig + 3)
                if end <= trig:
                    end = trig + 3
            except (TypeError, ValueError):
                trig, end = 6, 9
            sys_events.append({
                "type": "outage",
                "tool": tool,
                "trigger_at": trig,
                "end_at": end,
                "start_msg": str(ev.get("start_msg") or f"{tool} brief outage")[:300],
                "end_msg": str(ev.get("end_msg") or f"{tool} restored")[:300],
                "points_for_report": int(ev.get("points_for_report") or 5),
                "triggered": False,
                "resolved": False,
            })

        # Generic interactive tool surfaces (table/sql/kanban/...).
        clean_tools: list[dict] = []
        for raw in (env.get("tools") or []):
            normalised = normalise_tool_spec(raw)
            if normalised:
                clean_tools.append(normalised)
            if len(clean_tools) >= 6:
                break

        scenario_id = f"scn_{uuid.uuid4().hex[:10]}"
        return ScenarioTemplate(
            scenario_id=scenario_id,
            career=job.career,
            title=str(env.get("title") or job.title)[:160],
            company=job.company,
            fingerprint=job.fingerprint(),
            source_job_ext_id=job.ext_id,
            emails=clean_emails,
            tasks=clean_tasks,
            tool_data={"slack_seed": slack_seed},
            system_events=sys_events,
            tools=clean_tools,
        )


# ──────────────────────────────────────────────────────────────────
# Materialiser — writes a scenario into the live simulation collections
# ──────────────────────────────────────────────────────────────────

@dataclass
class MaterializerCollections:
    """Bundle of Mongo collection handles the materialiser writes to.

    Keeping this as a simple struct lets `app.py` inject the collections it
    already owns, without the engine importing app or knowing about Mongo
    setup details.
    """
    emails_col: Any
    tasks_col: Any
    sheets_col: Any
    invoices_col: Any
    siem_col: Any
    jira_col: Any
    campaigns_col: Any
    slack_col: Any
    system_events_col: Any
    tools_col: Any = None  # generic tool surfaces (table/sql/kanban/...)


class Materializer:
    """Seeds a simulation from a `ScenarioTemplate`, mapping abstract NPCs
    onto the existing CAREER_CONFIGS personas so all downstream tools and
    event handlers continue to work unchanged."""

    def __init__(self, cols: MaterializerCollections, career_configs: dict) -> None:
        self.cols = cols
        self.career_configs = career_configs

    def materialise(self, template: ScenarioTemplate, sim_id: str) -> dict:
        cfg = self.career_configs[template.career]
        # Overlay the scenario's company onto the persona pack — names stay,
        # but emails and the "company" string reflect the new scenario.
        company = template.company

        emails_to_insert = []
        now = datetime.utcnow().strftime("%H:%M")
        for e in template.emails:
            persona = self._persona_for(cfg, e["from_role_key"])
            eid = f"e_{uuid.uuid4().hex[:6]}"
            emails_to_insert.append({
                "id": eid,
                "simulation_id": sim_id,
                "from_name": persona["name"],
                "from_role": persona["role"],
                "from_email": persona["email"],
                "to": "you",
                "subject": e["subject"],
                "body": e["body"],
                "time": _email_time_for(e["ord"]),
                "read": False,
                "replied": False,
                "priority": e["priority"],
                "category": e["category"],
                "requires_action": e["requires_action"],
                "scenario_id": template.scenario_id,
            })
        if emails_to_insert:
            self.cols.emails_col.insert_many(emails_to_insert)

        tasks_to_insert = []
        for t in template.tasks:
            tasks_to_insert.append({
                "id": t["id"],
                "simulation_id": sim_id,
                "title": t["title"],
                "description": t["description"],
                "priority": t["priority"],
                "deadline": t["deadline"],
                "status": "pending",
                "points": t["points"],
                "tools_required": t["tools_required"],
                "tool_workflow": t["tool_workflow"],
                "generated": True,
                "scenario_id": template.scenario_id,
            })
        if tasks_to_insert:
            self.cols.tasks_col.insert_many(tasks_to_insert)

        # Slack seed — every scenario gets the canonical channels for the
        # career, plus any extra rows the LLM proposed.
        channels = list(DEFAULT_SLACK_CHANNELS.get(template.career, ("#general",)))
        seed_rows = []
        for ch in channels:
            seed_rows.append(_slack_msg(sim_id, ch, "FirstDesk Bot",
                                        f"Welcome to {ch} — daily standup at 09:30."))
        for s in template.tool_data.get("slack_seed", []):
            seed_rows.append(_slack_msg(sim_id, s["channel"], s["from"], s["text"]))
        if seed_rows:
            self.cols.slack_col.insert_many(seed_rows)

        # System events
        if template.system_events:
            ev_rows = []
            for ev in template.system_events:
                ev_rows.append({**ev, "id": f"ev_{uuid.uuid4().hex[:5]}", "simulation_id": sim_id})
            self.cols.system_events_col.insert_many(ev_rows)

        # Generic tool surfaces — only when the engine collection was wired.
        tool_docs: list[dict] = []
        if self.cols.tools_col is not None and template.tools:
            for spec in template.tools:
                tool_docs.append({
                    "tool_id": f"tl_{uuid.uuid4().hex[:8]}",
                    "simulation_id": sim_id,
                    "scenario_id": template.scenario_id,
                    "name": spec["name"],
                    "tool_type": spec["tool_type"],
                    "icon": spec.get("icon"),
                    "tint": spec.get("tint"),
                    "category": spec.get("category", "generic"),
                    "data": spec["data"],
                    "actions_log": [],
                    "created_at": datetime.utcnow().isoformat(),
                })
            if tool_docs:
                self.cols.tools_col.insert_many(tool_docs)

        log.info(
            "materializer.done sim=%s scenario=%s career=%s emails=%d tasks=%d slack=%d events=%d tools=%d",
            sim_id, template.scenario_id, template.career,
            len(emails_to_insert), len(tasks_to_insert), len(seed_rows),
            len(template.system_events), len(tool_docs),
        )

        return {
            "scenario_id": template.scenario_id,
            "career": template.career,
            "company": company,
            "emails": len(emails_to_insert),
            "tasks": len(tasks_to_insert),
            "tools_palette": list(CAREER_TOOL_PALETTE.get(template.career, ())),
            "generic_tools": [
                {"tool_id": t["tool_id"], "name": t["name"],
                 "tool_type": t["tool_type"], "icon": t.get("icon"),
                 "tint": t.get("tint")}
                for t in tool_docs
            ],
        }

    @staticmethod
    def _persona_for(cfg: dict, role_key: str) -> dict:
        # role_key is one of: manager | client | teammate | it | hr | director | system
        if role_key == "system":
            return {"name": "FirstDesk Bot", "role": "System", "email": "system@firstdesk.app"}
        return cfg.get(role_key) or cfg["manager"]


def _email_time_for(ordinal: int) -> str:
    """Spread starter emails across the morning so the inbox feels real."""
    base_minutes = 8 * 60 + 15  # 08:15
    minutes = base_minutes + ordinal * 18
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _slack_msg(sim_id: str, channel: str, author: str, text: str) -> dict:
    return {
        "id": f"sl_{uuid.uuid4().hex[:8]}",
        "simulation_id": sim_id,
        "channel": channel,
        "author": author,
        "text": text,
        "ts": datetime.utcnow().isoformat(),
    }


# ──────────────────────────────────────────────────────────────────
# Engine — the public façade `app.py` calls
# ──────────────────────────────────────────────────────────────────

class ScenarioEngine:
    """Top-level façade. Instantiated once at app boot."""

    def __init__(
        self,
        *,
        sources: Sequence[JobSource],
        extractor: ScenarioExtractor,
        materializer: Materializer,
        jobs_col: Any,
        scenarios_col: Any,
        user_scenarios_col: Any,
    ) -> None:
        self.sources = list(sources)
        self.extractor = extractor
        self.materializer = materializer
        self.jobs_col = jobs_col
        self.scenarios_col = scenarios_col
        self.user_scenarios_col = user_scenarios_col

    # ── INGESTION ────────────────────────────────────────────────
    def ingest(self, career: str, source_name: Optional[str] = None,
               limit: int = 5) -> dict:
        """Pull jobs, extract scenarios, persist to Mongo. Idempotent."""
        if career not in VALID_CAREERS:
            return {"ok": False, "error": f"unknown career: {career}"}
        sources = [s for s in self.sources if (source_name is None or s.name == source_name)]
        if not sources:
            return {"ok": False, "error": f"no source matched: {source_name}"}

        fetched = 0
        new_jobs = 0
        new_scenarios = 0
        skipped = 0
        for src in sources:
            try:
                postings = src.fetch(career, limit=limit)
            except Exception as ex:
                log.exception("ingest.fetch_failed source=%s career=%s kind=%s",
                              src.name, career, type(ex).__name__)
                continue
            fetched += len(postings)
            log.info("ingest.fetched source=%s career=%s n=%d", src.name, career, len(postings))
            for job in postings:
                # cache the JD
                self.jobs_col.update_one(
                    {"ext_id": job.ext_id},
                    {"$set": {**asdict(job), "cached_at": datetime.utcnow().isoformat()}},
                    upsert=True,
                )
                new_jobs += 1
                # skip if we already have a scenario for this fingerprint
                fp = job.fingerprint()
                if self.scenarios_col.find_one({"fingerprint": fp}, {"_id": 1}):
                    skipped += 1
                    continue
                template = self.extractor.extract(job)
                if template is None:
                    skipped += 1
                    continue
                self.scenarios_col.insert_one({
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
                })
                new_scenarios += 1
        return {
            "ok": True, "career": career,
            "fetched": fetched, "cached_jobs": new_jobs,
            "new_scenarios": new_scenarios, "skipped": skipped,
        }

    # ── PICK ────────────────────────────────────────────────────
    def pick_for_user(self, user_id: str, career: str) -> Optional[ScenarioTemplate]:
        """Return a scenario the user has not seen yet, or None."""
        seen = set()
        ledger = self.user_scenarios_col.find_one({"user_id": user_id}) or {}
        for entry in ledger.get("seen", []):
            if entry.get("career") == career:
                seen.add(entry.get("scenario_id"))
        cursor = self.scenarios_col.find({"career": career}, {"_id": 0})
        candidates = [s for s in cursor if s["scenario_id"] not in seen]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        return ScenarioTemplate(
            scenario_id=chosen["scenario_id"],
            career=chosen["career"],
            title=chosen["title"],
            company=chosen["company"],
            fingerprint=chosen["fingerprint"],
            source_job_ext_id=chosen["source_job_ext_id"],
            emails=chosen.get("emails", []),
            tasks=chosen.get("tasks", []),
            tool_data=chosen.get("tool_data", {}),
            system_events=chosen.get("system_events", []),
            tools=chosen.get("tools", []),
            created_at=chosen.get("created_at", datetime.utcnow().isoformat()),
        )

    def mark_seen(self, user_id: str, scenario_id: str, career: str) -> None:
        self.user_scenarios_col.update_one(
            {"user_id": user_id},
            {"$push": {"seen": {
                "scenario_id": scenario_id,
                "career": career,
                "ts": datetime.utcnow().isoformat(),
            }}},
            upsert=True,
        )

    # ── ORCHESTRATE: the call site `/api/start` uses this ───────
    def start_for_user(self, user_id: str, career: str, sim_id: str) -> Optional[dict]:
        """Pick + materialise. Returns a summary dict or None when the bank is empty."""
        template = self.pick_for_user(user_id, career)
        if template is None:
            return None
        summary = self.materializer.materialise(template, sim_id)
        self.mark_seen(user_id, template.scenario_id, career)
        return summary


# ──────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────

def build_default_engine(
    *,
    llm: Any,
    cols: MaterializerCollections,
    jobs_col: Any,
    scenarios_col: Any,
    user_scenarios_col: Any,
    career_configs: dict,
    seed_path: str | Path = "scenarios_seed.json",
) -> ScenarioEngine:
    """Wire the default sources, extractor, materializer."""
    sources: list[JobSource] = [SeedFileSource(path=seed_path)]
    adzuna = AdzunaSource()
    if adzuna.enabled:
        sources.append(adzuna)
        log.info("scenario_engine.adzuna_enabled")
    else:
        log.info("scenario_engine.adzuna_disabled reason=missing_credentials")
    return ScenarioEngine(
        sources=sources,
        extractor=ScenarioExtractor(llm=llm),
        materializer=Materializer(cols=cols, career_configs=career_configs),
        jobs_col=jobs_col,
        scenarios_col=scenarios_col,
        user_scenarios_col=user_scenarios_col,
    )
