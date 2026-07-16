# workplace_physics.py
# Cross-tool causality engine. Drop-in addition to FirstDesk's existing
# tool-action pipeline. Bolted onto /api/tool/{tool_id}/action.
#
# Each rule key is "<tool>.<action>" (e.g. "siem.escalate_alert_correct").
# Each rule has optional top-level "conditions" (gate the whole rule) and a
# list of "triggers" (each may have its own per-trigger "condition").
#
# Templating: any string field in a trigger may include {placeholder} tokens
# that get filled from the action's context dict at runtime.

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Optional

# ═══════════════════════════════════════════════════════════
# RULES
# ═══════════════════════════════════════════════════════════

WORKPLACE_PHYSICS: dict[str, dict] = {

    # ─── SAGE / ACCOUNTING ─────────────────────────────────
    "sage.update_invoice": {
        "triggers": [
            {"type": "tool_update", "tool": "bank_reconciliation", "action": "flag_entry",
             "message": "Invoice {invoice_id} updated — reconciliation entry flagged for review"},
            {"type": "agent_email", "from": "manager", "condition": "amount > 1000",
             "subject": "Invoice update — please confirm",
             "body": "I noticed you updated invoice {invoice_id} for £{amount}. Can you confirm the reason for the change before end of day? This is above our auto-approval threshold."},
            {"type": "slack_message", "channel": "#finance",
             "message": "📊 Invoice {invoice_id} updated — amount changed to £{amount}"},
        ],
    },
    "sage.fix_discrepancy_correct": {
        "triggers": [
            {"type": "performance_note", "note": "Correctly identified and resolved a financial discrepancy", "points": 15},
            {"type": "agent_email", "from": "manager", "subject": "Good catch on the reconciliation",
             "body": "That discrepancy you found on the {account} account — well spotted. That would have caused real problems in the month-end report. Keep that attention to detail."},
        ],
    },
    "sage.fix_discrepancy_wrong": {
        "triggers": [
            {"type": "performance_note", "note": "Attempted to fix discrepancy but entry is still incorrect", "points": -10},
            {"type": "agent_email", "from": "manager", "subject": "Check your numbers — still off",
             "body": "I've reviewed the entry you made on {account}. The figures still don't reconcile — the issue is on line {line}. The bank shows £{bank_amount} but our ledger shows £{ledger_amount}. Take another look."},
        ],
    },
    "sage.wrong_entry": {
        "triggers": [
            {"type": "performance_note", "note": "Incorrect ledger entry — wrong account or amount", "points": -10},
            {"type": "agent_email", "from": "manager", "subject": "Ledger entry needs correction",
             "body": "Quick one — the entry you posted to {account} doesn't look right. The amount doesn't match the supporting document. Can you review and correct before the client call at 3pm?"},
        ],
    },

    # ─── SIEM / CYBERSECURITY ──────────────────────────────
    "siem.escalate_alert_correct": {
        "triggers": [
            {"type": "create_jira_ticket", "title": "INC: {alert_title}", "priority": "{alert_severity}",
             "description": "Escalated from SIEM alert {alert_id}. Initial analysis: {analyst_notes}. Assigned to: {user_name}.",
             "status": "open"},
            {"type": "slack_message", "channel": "#security-incidents",
             "message": "🚨 ALERT ESCALATED: {alert_title} (Severity: {alert_severity}) — Jira ticket created by {user_name}"},
            {"type": "agent_email", "from": "manager", "condition": "alert_severity == CRITICAL",
             "subject": "P1 Incident — situation report needed",
             "body": "I see you escalated {alert_title} as P1. Good call — I've notified the CISO. Can you send me a 3-line situation report in the next 15 minutes? Format: (1) What happened (2) Scope (3) Immediate action taken."},
            {"type": "performance_note", "note": "Correctly escalated a real security threat", "points": 20},
        ],
    },
    "siem.escalate_alert_wrong": {
        "triggers": [
            {"type": "agent_email", "from": "manager", "subject": "False positive escalation — quick note",
             "body": "I reviewed the escalation on {alert_id}. That one was actually a false positive — {false_positive_reason}. No harm done but worth knowing for next time. Check the confidence score and cross-reference with known FPs before escalating."},
            {"type": "performance_note", "note": "Escalated a known false positive — unnecessary noise", "points": -10},
        ],
    },
    "siem.resolve_alert_correct": {
        "triggers": [
            {"type": "performance_note", "note": "Correctly resolved false positive — good discrimination", "points": 10},
            {"type": "slack_message", "channel": "#security-incidents",
             "message": "✅ Alert {alert_id} resolved as false positive by {user_name}"},
        ],
    },
    "siem.resolve_alert_wrong": {
        "triggers": [
            {"type": "performance_note", "note": "CRITICAL: Resolved a real threat as false positive — danger", "points": -25},
            {"type": "agent_email", "from": "manager", "subject": "URGENT: Incorrectly resolved alert",
             "body": "I need to speak to you immediately. Alert {alert_id} — you marked it as a false positive but our threat intel confirms it was a real incident. We need to re-open this and contain now. What was your reasoning for resolving it?"},
            {"type": "slack_message", "channel": "#security-incidents",
             "message": "⚠️ Alert {alert_id} was incorrectly resolved — re-opening for investigation"},
        ],
    },
    "siem.correlate_alerts": {
        "triggers": [
            {"type": "performance_note", "note": "Identified correlation between multiple alerts — advanced analytical skill", "points": 15},
            {"type": "slack_message", "channel": "#security-incidents",
             "message": "🔗 Correlation identified: {alert_1} + {alert_2} share source IP {ip_address} — possible coordinated attack"},
            {"type": "agent_email", "from": "manager", "subject": "Good analysis on the correlation",
             "body": "That link you spotted between {alert_1} and {alert_2} — that's exactly the kind of pattern recognition that separates good analysts from great ones. Makes me think this is coordinated. Keep watching {ip_address}."},
        ],
    },

    # ─── JIRA / SOFTWARE DEV ──────────────────────────────
    "jira.close_ticket_correct": {
        "triggers": [
            {"type": "slack_message", "channel": "#engineering",
             "message": "✅ {ticket_id} closed by {user_name} — {ticket_title}"},
            {"type": "tool_update", "tool": "jira", "action": "update_sprint_velocity",
             "message": "Sprint velocity recalculated after {ticket_id} closure"},
            {"type": "agent_email", "from": "manager", "condition": "ticket_priority == HIGH",
             "subject": "Nice work on {ticket_id}",
             "body": "That was a tricky one — the client has been waiting on that fix since Tuesday. I'll let them know it's resolved."},
            {"type": "performance_note", "note": "Closed ticket correctly and on time", "points": 10},
        ],
    },
    "jira.move_to_in_progress": {
        "triggers": [
            {"type": "agent_email", "from": "teammate", "condition": "has_dependency == True",
             "subject": "RE: {ticket_id} — heads up",
             "body": "Hey, I saw you picked up {ticket_id}. Just a heads up — there's a dependency on {dependency_ticket} that needs to be merged first or you'll hit a conflict. Worth checking before you go too deep into it."},
        ],
    },
    "jira.wrong_priority_assigned": {
        "triggers": [
            {"type": "agent_email", "from": "manager", "subject": "Ticket priority — quick note",
             "body": "I noticed you marked {ticket_id} as {wrong_priority}. Given the client impact on this one, it should be {correct_priority}. Can you update it? Affects how we plan the sprint."},
            {"type": "performance_note", "note": "Assigned incorrect priority to ticket", "points": -5},
        ],
    },

    # ─── EMAIL / OUTLOOK ──────────────────────────────────
    "email.client_ignored": {
        "conditions": {"interactions_without_client_reply": 8},
        "triggers": [
            {"type": "agent_email", "from": "client", "to": "manager", "cc": "user",
             "subject": "RE: {original_subject} — following up",
             "body": "Hi {manager_name}, I wanted to follow up on my earlier message. I've been waiting to hear back from your team since {original_time}. Could you let me know the status? This is becoming quite urgent on our end. Best, {client_name}"},
            {"type": "agent_email", "from": "manager", "subject": "Client chasing — please respond",
             "body": "I've just been CC'd by {client_name}. They've been waiting all morning. Please respond to their query within the next 30 minutes and keep me copied. Client satisfaction is a priority."},
            {"type": "performance_note", "note": "Client ignored for extended period — communication failure", "points": -15},
        ],
    },
    "email.unprofessional_tone": {
        "triggers": [
            {"type": "agent_email", "from": "manager", "subject": "Quick note on your email",
             "body": "Your reply to {recipient_name} — the content was right but the tone came across as a bit informal for an external communication. Worth keeping in mind. Generally, external emails should be more formal than internal ones."},
            {"type": "performance_note", "note": "Unprofessional email tone detected in external communication", "points": -8},
        ],
    },
    "email.correct_cc_manager": {
        "triggers": [
            {"type": "performance_note", "note": "Correctly CC'd manager on client escalation — good judgment", "points": 5},
        ],
    },
    "email.professional_client_response": {
        "triggers": [
            {"type": "performance_note", "note": "Professional and timely client communication", "points": 8},
            {"type": "agent_email", "from": "client", "subject": "RE: {subject}",
             "body": "Thanks for getting back to me so quickly — really appreciate it. That all makes sense. I'll proceed on that basis. Let me know if anything changes."},
        ],
    },

    # ─── TASKS ────────────────────────────────────────────
    "task.completed_on_time": {
        "triggers": [
            {"type": "performance_note", "note": "Task completed on time", "points": 5},
            {"type": "slack_message", "channel": "#general", "message": "✅ {task_title} marked complete"},
        ],
    },
    "task.completed_early": {
        "triggers": [
            {"type": "performance_note", "note": "Task completed ahead of deadline — efficient", "points": 8},
        ],
    },
    "task.deadline_missed": {
        "triggers": [
            {"type": "agent_email", "from": "manager", "subject": "{task_title} — deadline passed",
             "body": "The deadline for {task_title} was {deadline}. I don't see it marked as complete. What's the current status? If you're blocked on something, let me know immediately."},
            {"type": "performance_note", "note": "Task deadline missed without communication", "points": -10},
        ],
    },

    # ─── IT SYSTEM ────────────────────────────────────────
    "it.outage_reported": {
        "triggers": [
            {"type": "agent_email", "from": "it", "subject": "RE: {tool} access issue",
             "body": "Hi, thanks for flagging. We're aware of the {tool} issue and a fix is underway — estimated 20 minutes. In the meantime you can use the read-only backup at {backup_url}. Apologies for the disruption."},
            {"type": "performance_note", "note": "Proactively reported system outage to IT — correct process", "points": 5},
        ],
    },
    "it.outage_ignored": {
        "conditions": {"interactions_without_reporting": 5},
        "triggers": [
            {"type": "agent_email", "from": "manager", "subject": "System outage — did you log this?",
             "body": "I heard {tool} went down earlier during your shift. Did you log it with IT? System outages need to be reported immediately so the team can plan around them. Always raise a ticket straight away."},
            {"type": "performance_note", "note": "Failed to report system outage — process not followed", "points": -5},
        ],
    },
}


# ═══════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════

class WorkplacePhysicsEngine:
    """Processes a single tool action and fires cascading workplace effects.

    Stateless — instantiated per request. Pulls collection refs and the active
    career_config from the caller so this module stays decoupled from app.py
    module globals.
    """

    def __init__(
        self,
        *,
        emails_col,
        tasks_col,
        tools_col,
        slack_col,
        evals_col,
        jira_col,
        sessions_col,
        career_config: dict,
        sim_id: str,
        session_id: str,
        user_id: str,
        user_name: str = "you",
    ):
        self.emails_col = emails_col
        self.tasks_col = tasks_col
        self.tools_col = tools_col
        self.slack_col = slack_col
        self.evals_col = evals_col
        self.jira_col = jira_col
        self.sessions_col = sessions_col
        self.career_config = career_config or {}
        self.sim_id = sim_id
        self.session_id = session_id
        self.user_id = user_id
        self.user_name = user_name

    def process(self, action_key: str, context: dict, skip_types: Optional[set] = None) -> list[dict]:
        """Entry point. Returns the list of effects actually fired.
        skip_types: trigger types to suppress (e.g. {'performance_note'} when the
        caller's own scoring pipeline already awards points for this action)."""
        rules = WORKPLACE_PHYSICS.get(action_key)
        if not rules:
            return []
        # Top-level gate
        for cond_key, cond_val in (rules.get("conditions") or {}).items():
            if not self._check_top_condition(cond_key, cond_val, context):
                return []
        # Inject user_name default for templates
        ctx = {"user_name": self.user_name, **(context or {})}
        effects = []
        for trigger in rules.get("triggers", []):
            if skip_types and trigger.get("type") in skip_types:
                continue
            cond = trigger.get("condition")
            if cond and not self._eval_condition(cond, ctx):
                continue
            try:
                effect = self._fire(trigger, ctx)
                if effect:
                    effects.append(effect)
            except Exception as ex:
                # One bad trigger must not break the whole cascade.
                effects.append({"type": "error", "trigger": trigger.get("type"), "error": str(ex)[:120]})
        return effects

    # ── fire ──────────────────────────────────────────────
    def _fire(self, trigger: dict, ctx: dict) -> Optional[dict]:
        t = trigger["type"]
        filled = self._fill(trigger, ctx)
        if t == "agent_email":       return self._send_email(filled)
        if t == "slack_message":     return self._post_slack(filled)
        if t == "create_jira_ticket": return self._create_ticket(filled)
        if t == "tool_update":       return self._update_tool(filled)
        if t == "performance_note":  return self._record_eval(filled)
        return None

    def _send_email(self, t: dict) -> dict:
        from_agent = t.get("from", "manager")
        agent = self.career_config.get(from_agent, {}) if isinstance(self.career_config, dict) else {}
        eid = f"e_phys_{uuid.uuid4().hex[:8]}"
        email = {
            "id": eid,
            "simulation_id": self.sim_id,
            "from_name":  agent.get("name", from_agent.title()),
            "from_role":  agent.get("role", from_agent.title()),
            "from_email": agent.get("email", f"{from_agent}@company.com"),
            "to": "you",
            "cc": t.get("cc", ""),
            "subject": t.get("subject", "Update"),
            "body": t.get("body", ""),
            "time": datetime.utcnow().strftime("%H:%M"),
            "read": False,
            "replied": False,
            "priority": t.get("priority", "medium"),
            "category": "physics_triggered",
            "requires_action": True,
            "triggered_by": "workplace_physics",
            "created_at": datetime.utcnow().isoformat(),
        }
        self.emails_col.insert_one(email)
        return {"type": "agent_email", "id": eid, "from": from_agent, "subject": email["subject"]}

    def _post_slack(self, t: dict) -> dict:
        sid = f"s_phys_{uuid.uuid4().hex[:8]}"
        msg = {
            "id": sid,
            "simulation_id": self.sim_id,
            "channel": t.get("channel", "#general"),
            "text": t.get("message", ""),
            "sender": "System",
            "sender_type": "bot",
            "timestamp": datetime.utcnow().isoformat(),
            "triggered_by": "workplace_physics",
        }
        self.slack_col.insert_one(msg)
        return {"type": "slack_message", "channel": msg["channel"], "message": msg["text"][:120]}

    def _create_ticket(self, t: dict) -> dict:
        n = self.jira_col.count_documents({"simulation_id": self.sim_id}) + 1
        tid = f"INC-{n:04d}"
        doc = {
            "id": tid,
            "simulation_id": self.sim_id,
            "title": t.get("title", "Incident"),
            "priority": (t.get("priority") or "MEDIUM").upper(),
            "description": t.get("description", ""),
            "status": "open",
            "type": "incident",
            "created_at": datetime.utcnow().isoformat(),
            "triggered_by": "workplace_physics",
        }
        self.jira_col.insert_one(doc)
        return {"type": "create_jira_ticket", "ticket_id": tid, "title": doc["title"]}

    def _update_tool(self, t: dict) -> dict:
        tool_name = t.get("tool", "")
        action = t.get("action", "")
        # Try display_name OR tool_type match (looser)
        tool = self.tools_col.find_one({
            "simulation_id": self.sim_id,
            "$or": [
                {"display_name": {"$regex": tool_name, "$options": "i"}},
                {"name":         {"$regex": tool_name, "$options": "i"}},
                {"tool_type":    {"$regex": tool_name, "$options": "i"}},
            ],
        })
        if tool:
            self.tools_col.update_one(
                {"_id": tool["_id"]},
                {"$push": {"physics_flags": {"action": action, "message": t.get("message", ""), "ts": datetime.utcnow().isoformat()}}}
            )
        return {"type": "tool_update", "tool": tool_name, "action": action, "applied": bool(tool)}

    def _record_eval(self, t: dict) -> dict:
        pts = int(t.get("points", 0) or 0)
        self.evals_col.insert_one({
            "simulation_id": self.sim_id,
            "user_id": self.user_id,
            "type": "physics_note",
            "note": t.get("note", ""),
            "points": pts,
            "source": "workplace_physics",
            "ts": datetime.utcnow().isoformat(),
        })
        if pts:
            # Roll into the same per-session score_adj the rest of the app uses,
            # so EOD totals already aggregate physics impact.
            self.sessions_col.update_one(
                {"session_id": self.session_id},
                {"$inc": {"score_adj": pts, "physics_score": pts}},
            )
        return {"type": "performance_note", "note": t.get("note"), "points": pts}

    # ── templates / conditions ────────────────────────────
    _TOKEN_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    def _fill(self, trigger: dict, ctx: dict) -> dict:
        """Replace {placeholder} tokens recursively. Safe against curly-brace
        collisions in body text (we use a single named-token regex, not
        Python str.format which would choke on stray { in copy)."""
        def repl(s: str) -> str:
            return self._TOKEN_RE.sub(lambda m: str(ctx.get(m.group(1), "")), s)
        out: dict[str, Any] = {}
        for k, v in trigger.items():
            if isinstance(v, str):
                out[k] = repl(v)
            else:
                out[k] = v
        return out

    def _check_top_condition(self, key: str, expected: Any, ctx: dict) -> bool:
        # numeric "at least" semantics for counters
        actual = ctx.get(key)
        if isinstance(expected, int) and isinstance(actual, (int, float)):
            return actual >= expected
        return actual == expected

    def _eval_condition(self, condition: str, ctx: dict) -> bool:
        """Mini DSL: 'left op right' where left is a context key and right
        is a literal. Supports > < >= <= == !=."""
        try:
            parts = condition.split(maxsplit=2)
            if len(parts) != 3:
                return True
            left, op, right = parts
            left_val = ctx.get(left)
            if left_val is None:
                return False
            right_lit = right.strip("'\"")
            # numeric ops
            if op in (">", "<", ">=", "<="):
                lv, rv = float(left_val), float(right_lit)
                return {">": lv > rv, "<": lv < rv, ">=": lv >= rv, "<=": lv <= rv}[op]
            if op == "==": return str(left_val) == right_lit
            if op == "!=": return str(left_val) != right_lit
        except Exception:
            return False
        return True
