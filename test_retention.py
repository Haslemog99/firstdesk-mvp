#!/usr/bin/env python3
"""Retention contracts — first minute, resume, idle clock, meetings, SIEM evidence.

Run: python3 test_retention.py   (exit 0 pass / 1 fail)

AST-extracts pure helpers from app.py so this never boots Mongo/FastAPI.
Written to lock the five bounce modes live users actually hit.
"""
import ast
import sys
from datetime import datetime, timedelta

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
    missing = names - {n for n in names if any(
        (isinstance(node, ast.FunctionDef) and node.name == n)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == n for t in node.targets))
        or (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == n)
        for node in ast.parse(SRC).body)}
    from typing import Optional as _Optional
    ns = {"datetime": datetime, "timedelta": timedelta, "Optional": _Optional}
    exec("\n\n".join(s for s in segs if s), ns)
    return ns, missing


NEED = {
    "_should_resume_session",
    "_parse_iso_dt",
    "_tick_idle_pause",
    "_IDLE_PAUSE_AFTER_SEC",
    "_sim_offset_sec",
    "_SIM_DAY_START_HOUR",
    "_SIM_DAY_SEC",
    "_REAL_DAY_SEC",
    "_REAL_DAY_SEC_BY_LEVEL",
    "_real_day_sec",
    "_MEETING_WINDOWS",
    "_MEETING_ONBOARD_REAL_SEC",
    "_meeting_runtime",
    "_format_siem_tool_line",
}

NS, MISSING = _extract(NEED)

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


check(not MISSING, f"app.py is missing helpers: {sorted(MISSING)}")

if MISSING:
    print("FAIL — implement the retention helpers first:")
    for m in sorted(MISSING):
        print("  missing", m)
    sys.exit(1)

resume = NS["_should_resume_session"]
tick = NS["_tick_idle_pause"]
idle_after = int(NS["_IDLE_PAUSE_AFTER_SEC"])
runtime = NS["_meeting_runtime"]
siem_line = NS["_format_siem_tool_line"]
onboard_sec = int(NS["_MEETING_ONBOARD_REAL_SEC"])


# ── 1. Start over vs resume ──────────────────────────────────────
check(resume({"career": "cybersecurity", "sim_id": "sim_x", "real_start_time": "t"},
             "cybersecurity", False) is True,
      "same career with a live session resumes")
check(resume({"career": "cybersecurity", "sim_id": "sim_x"},
             "cybersecurity", True) is False,
      "reset=True always starts over")
check(resume({"career": "cybersecurity", "sim_id": "sim_x"},
             "accountant", False) is False,
      "switching career is a new start")
check(resume({}, "cybersecurity", False) is False,
      "empty session cannot resume")
check(resume({"career": "cybersecurity"}, "cybersecurity", False) is False,
      "career with no sim_id is not a resumable desk")


# ── 2. Idle wall-clock must not eat the day ──────────────────────
check(idle_after >= 120, "idle pause starts after at least 2 real minutes")
start = datetime(2026, 8, 31, 10, 0, 0)
sess = {
    "real_start_time": start.isoformat(),
    "last_activity_at": start.isoformat(),
    "last_clock_tick": start.isoformat(),
    "clock_paused_sec": 0,
}
# 4 minutes on the page, last click 30s ago — still active, no pause
sess["last_activity_at"] = (start + timedelta(minutes=3, seconds=30)).isoformat()
out = tick(sess, start + timedelta(minutes=4))
check(out["clock_paused_sec"] == 0, "active polling does not pause the clock")

# then 47 minutes idle with polls still arriving (tab left open)
sess2 = dict(sess)
sess2["last_activity_at"] = start.isoformat()
sess2["last_clock_tick"] = (start + timedelta(minutes=4)).isoformat()
sess2["clock_paused_sec"] = 0
out2 = tick(sess2, start + timedelta(minutes=51))
check(out2["clock_paused_sec"] >= 40 * 60,
      "a 47-minute idle gap while the tab is open must pause ~the idle tail, not run the sim clock")

# coming back must KEEP the pause (not dump 47 minutes onto elapsed)
sess3 = dict(sess2)
sess3["clock_paused_sec"] = out2["clock_paused_sec"]
sess3["last_clock_tick"] = out2["last_clock_tick"]
sess3["last_activity_at"] = (start + timedelta(minutes=51)).isoformat()
out3 = tick(sess3, start + timedelta(minutes=52))
check(out3["clock_paused_sec"] == sess3["clock_paused_sec"],
      "returning from idle must not unwind the accumulated pause")


# ── 3. Meetings must not punish the first-hour reader ────────────
standup = {
    "id": "cyber-d1-standup",
    "title": "Morning standup",
    "sim_time": "11:00",
    "ask": "Your triage summary.",
    "points": 10,
}
studied = {"experience_level": "studied", "current_day": 1, "meetings": {},
           "emails_read": [], "alerts_triaged": []}
# 11:45 sim on studied = 2h45 sim after 09:00. Window is 30 sim-min after open,
# so without grace this is already "missed".
clock_late = {"sim_elapsed_sec": (2 * 3600) + (45 * 60), "elapsed_real_sec": 12 * 60}
rt = runtime(studied, clock_late, standup)
check(rt["state"] != "missed",
      "day-1 standup must not miss while the user is still onboarding (no mail opened, <20 real min)")

studied2 = dict(studied)
studied2["emails_read"] = ["e1"]
clock_earned = {"sim_elapsed_sec": (3 * 3600), "elapsed_real_sec": onboard_sec + 600}
rt2 = runtime(studied2, clock_earned, standup)
check(rt2["state"] == "missed",
      "after onboarding grace, a no-show still misses")

already = {"experience_level": "studied", "current_day": 1,
           "meetings": {"cyber-d1-standup": {"state": "attended", "submitted_at": "t"}},
           "emails_read": ["e1"]}
rt3 = runtime(already, clock_earned, standup)
check(rt3["state"] == "attended", "attended stays attended")


# ── 4. SIEM work must be visible to the manager ──────────────────
line = siem_line({
    "id": "ALERT-2026-0218-002", "severity": "HIGH", "source": "EDR",
    "title": "SSH brute force", "status": "escalated",
    "triage_status": "reviewed", "verdict_rationale": "C2 + authorized_keys",
    "correct_action": "escalate",
})
for token in ("ALERT-2026-0218-002", "triage=reviewed", "escalat", "C2 + authorized_keys"):
    check(token.lower() in line.lower() or token in line,
          f"siem tool line must expose {token!r} so Slack theatre is not required")


if fails:
    print(f"FAIL {len(fails)}")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("OK  retention contracts")
sys.exit(0)
