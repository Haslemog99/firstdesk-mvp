#!/usr/bin/env python3
"""Contract test for _generate_noise_alerts — written BEFORE the generator.

Run: python3 test_noise_alerts.py   (exits 0 on pass, 1 on fail)

Extracts _generate_noise_alerts + _enrich_soc_alert (and their constants)
from app.py via AST so the test never imports app.py's side effects
(Mongo client, FastAPI app, STRICT_ENV checks).
"""
import ast, re, sys, json, hashlib
from datetime import datetime, timedelta

SRC = open("app.py").read()

def _extract(names: set):
    """Pull top-level defs/assigns from app.py source into a namespace."""
    tree = ast.parse(SRC)
    segs = []
    for node in tree.body:
        got = set()
        if isinstance(node, ast.Assign):
            got = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.FunctionDef, ast.AnnAssign)):
            got = {getattr(node, "name", None) or getattr(getattr(node, "target", None), "id", None)}
        if got & names:
            segs.append(ast.get_source_segment(SRC, node))
    ns = {"re": re, "hashlib": hashlib, "datetime": datetime, "timedelta": timedelta, "json": json}
    import random as _random
    ns["random"] = _random
    exec("\n\n".join(segs), ns)
    return ns

NS = _extract({
    "_SOC_TA_CODES", "_SOC_RULE_NAMES", "_SOC_IP_REPUTATION", "_RFC1918_RE",
    "_enrich_soc_alert",
    "_NOISE_POOLS", "_noise_templates", "_generate_noise_alerts",
})

fails = []
def check(cond, msg):
    if not cond:
        fails.append(msg)

# ── the authored schema noise must match exactly ──────────────────────
# 32 authored fields minus log_source (only ALERT-014 sets it explicitly)
# minus simulation_id (stamped by the insertion path, not the generator).
EXPECTED_KEYS = {
    "id", "timestamp", "severity", "confidence", "source", "source_detail",
    "title", "description", "details",
    "src_ip", "src_host", "src_user", "dst_ip", "dst_geo", "dst_reputation",
    "affected_asset", "mitre_tactic", "mitre_technique", "mitre_name",
    "status", "triage_status", "sla_hours",
    "is_real_threat", "threat_type", "correct_action", "points",
    "related_alerts", "playbook", "playbook_steps",
    "raw_log", "analyst_notes", "false_positive_indicators",
}

RFC1918 = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)")
IPRE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
HOSTPAT = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)+$")
IDPAT = re.compile(r"^NOISE-2026-D1-\d{3}$")

gen = NS.get("_generate_noise_alerts")
if gen is None:
    print("RED: _generate_noise_alerts does not exist yet (expected before Step 4).")
    sys.exit(1)
enrich = NS["_enrich_soc_alert"]

batch = gen(day=1, count=10, seed=12345)

# 1 ── count and shape ─────────────────────────────────────────────────
check(isinstance(batch, list) and len(batch) == 10, f"expected 10 alerts, got {len(batch) if isinstance(batch, list) else type(batch)}")

seen_ids = set()
for a in batch:
    aid = a.get("id", "?")

    # 2 ── exact schema: no missing, no extra keys ─────────────────────
    keys = set(a.keys())
    check(keys == EXPECTED_KEYS,
          f"{aid}: schema mismatch — missing {sorted(EXPECTED_KEYS - keys)}, extra {sorted(keys - EXPECTED_KEYS)}")

    # 3 ── invariant values every noise alert must carry ───────────────
    check(a["is_real_threat"] is False, f"{aid}: is_real_threat must be False")
    check(a["correct_action"] == "resolve", f"{aid}: correct_action must be 'resolve'")
    check(a["threat_type"] == "noise", f"{aid}: threat_type must be 'noise'")
    check(2 <= int(a["points"]) <= 5, f"{aid}: points {a['points']} outside 2-5")
    check(a["severity"] in ("LOW", "MEDIUM"), f"{aid}: severity {a['severity']} not LOW/MEDIUM")
    check(a["status"] == "open" and a["triage_status"] == "unreviewed", f"{aid}: wrong initial status")
    check(a["related_alerts"] == [], f"{aid}: related_alerts must be []")
    check(bool(str(a["false_positive_indicators"]).strip()), f"{aid}: false_positive_indicators must be populated")
    check(len(a["playbook_steps"]) >= 2, f"{aid}: playbook needs >=2 steps")

    # 4 ── deterministic, unique, well-formed IDs ──────────────────────
    check(IDPAT.match(aid), f"bad id format: {aid}")
    check(aid not in seen_ids, f"duplicate id: {aid}")
    seen_ids.add(aid)

    # 5 ── timestamps on the right date, within the shift ──────────────
    check(a["timestamp"].startswith("2026-02-18T"), f"{aid}: day-1 timestamp {a['timestamp']} not on 2026-02-18")
    hour = int(a["timestamp"][11:13])
    check(0 <= hour <= 17, f"{aid}: timestamp hour {hour} outside shift")

    # 6 ── data conventions (pre-enrichment) ───────────────────────────
    src_ip, dst_ip, sh = str(a["src_ip"]), str(a["dst_ip"]), str(a["src_host"])
    if IPRE.match(src_ip):
        if sh.startswith("("):
            check(not RFC1918.match(src_ip), f"{aid}: external src_ip {src_ip} is RFC1918")
        else:
            check(RFC1918.match(src_ip), f"{aid}: internal src_ip {src_ip} not RFC1918")
    if sh and not sh.startswith("(") and sh != "-":
        check(HOSTPAT.match(sh), f"{aid}: unstructured hostname {sh!r}")
    if dst_ip not in ("-", "") and "/" not in dst_ip:
        check(IPRE.match(dst_ip), f"{aid}: dst_ip {dst_ip!r} not an IP")
        if not RFC1918.match(dst_ip):
            check(str(a["dst_reputation"]).strip() not in ("", "-"),
                  f"{aid}: public dst {dst_ip} needs a dst_reputation string")
    # no live URLs / fangable IOCs in analyst-facing fields (raw_log exempt)
    for k in ("title", "description", "details", "false_positive_indicators"):
        check(not re.search(r"https?://", str(a[k])), f"{aid}: live URL in {k}")
    # correct Windows Event IDs where lockouts are referenced
    blob = " ".join(str(a[k]) for k in ("title", "description", "details", "raw_log")).lower()
    if "lockout" in blob or "locked out" in blob:
        check("4740" in blob, f"{aid}: lockout referenced without Event ID 4740")

    # 7 ── enrichment: passes, idempotent, derives SOC fields ──────────
    e = enrich(dict(a))
    check(e["alert_id"].startswith("SOC-2026-"), f"{aid}: enrichment alert_id failed")
    check(e["severity_label"] in ("Low", "Medium"), f"{aid}: severity_label {e.get('severity_label')}")
    check(e["verdict"] is None, f"{aid}: verdict must start null")
    check(e["event_time"] == a["timestamp"] and e["detection_time"] >= a["timestamp"],
          f"{aid}: event/detection time derivation broken")
    check(enrich(dict(e)) == e, f"{aid}: enrichment not idempotent")
    check(a["severity"] == e["severity"], f"{aid}: enrichment mutated legacy severity")

# 8 ── determinism: same seed identical, different seed different ──────
again = gen(day=1, count=10, seed=12345)
check(json.dumps(batch, sort_keys=True, default=str) == json.dumps(again, sort_keys=True, default=str),
      "same seed must produce byte-identical output")
other = gen(day=1, count=10, seed=99999)
check(json.dumps(batch, sort_keys=True, default=str) != json.dumps(other, sort_keys=True, default=str),
      "different seed must produce different output")

# 9 ── day parameter drives the date ───────────────────────────────────
d2 = gen(day=2, count=5, seed=12345)
check(all(x["timestamp"].startswith("2026-02-19T") for x in d2), "day=2 timestamps must be on 2026-02-19")
check(all(re.match(r"^NOISE-2026-D2-\d{3}$", x["id"]) for x in d2), "day=2 ids must carry D2")

# 10 ── template variety in a batch of 10 ──────────────────────────────
check(len({a["title"].split(" — ")[0] for a in batch}) >= 3, "a batch of 10 should draw from >=3 distinct templates")

if fails:
    print(f"FAIL — {len(fails)} assertion(s):")
    for f in fails[:25]:
        print("  ✗", f)
    sys.exit(1)
print("PASS — 10 noise alerts, schema-exact, convention-clean, deterministic.")
