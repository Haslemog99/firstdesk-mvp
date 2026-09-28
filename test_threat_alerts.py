#!/usr/bin/env python3
"""Contract test for threat_engine.generate_day_threats.

Run: python3 test_threat_alerts.py   (exits 0 on pass, 1 on fail)

Generated threats must be indistinguishable in shape from the authored ones in
_CYBER_DAY_PACKS: same schema, same conventions (RFC1918 internals, public
externals, structured hostnames), correct ground truth, and telemetry the EDR
console can group. Imports threat_engine directly (pure module, no deps) and
AST-extracts _enrich_soc_alert from app.py to prove enrichment still applies.
"""
import ast
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timedelta

from threat_engine import (FP_TEMPLATES, THREAT_TEMPLATES, WINDOWS_SERVER_PREFIXES,
                           generate_day_threats, summarise, threat_seed)

SRC = open("app.py").read()


def _ns(names):
    tree = ast.parse(SRC)
    segs = []
    for node in tree.body:
        got = set()
        if isinstance(node, ast.Assign):
            got = {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.FunctionDef):
            got = {node.name}
        if got & names:
            segs.append(ast.get_source_segment(SRC, node))
    ns = {"re": re, "hashlib": hashlib, "random": random,
          "datetime": datetime, "timedelta": timedelta}
    exec("\n\n".join(segs), ns)
    return ns


ns = _ns({"_SOC_TA_CODES", "_SOC_RULE_NAMES", "_SOC_IP_REPUTATION",
          "_RFC1918_RE", "_enrich_soc_alert", "_NOISE_POOLS"})
enrich = ns["_enrich_soc_alert"]
POOLS = ns["_NOISE_POOLS"]

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


# Same schema the noise layer is pinned to (test_noise_alerts.py), plus the
# two fields only the signal layer carries.
EXPECTED_KEYS = {
    "id", "timestamp", "severity", "confidence", "source", "source_detail",
    "title", "description", "details",
    "src_ip", "src_host", "src_user", "dst_ip", "dst_geo", "dst_reputation",
    "affected_asset", "mitre_tactic", "mitre_technique", "mitre_name",
    "status", "triage_status", "sla_hours",
    "is_real_threat", "threat_type", "correct_action", "points",
    "related_alerts", "playbook", "playbook_steps",
    "raw_log", "analyst_notes", "false_positive_indicators",
    "difficulty", "generated",
}
OPTIONAL_KEYS = {"edr_telemetry"}

RFC1918 = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)")
IPRE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
HOSTPAT = re.compile(r"^[A-Z][A-Z0-9]*(-[A-Z0-9]+)+$")
TSRE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

DAYS = [1, 2, 3, 5, 9, 40]
runs = {d: generate_day_threats(d, threat_seed("sim-abc", d), POOLS) for d in DAYS}

# ── 1. determinism ────────────────────────────────────────────────────
for d in DAYS:
    again = generate_day_threats(d, threat_seed("sim-abc", d), POOLS)
    check(json.dumps(again, sort_keys=True) == json.dumps(runs[d], sort_keys=True),
          f"day {d}: same (day, seed) did not reproduce byte-identical output")

other = generate_day_threats(3, threat_seed("sim-zzz", 3), POOLS)
check(json.dumps(other, sort_keys=True) != json.dumps(runs[3], sort_keys=True),
      "two different simulations generated identical day-3 threats")

# ── 2. every day produces work, including past the authored day 5 ─────
for d in DAYS:
    alerts = runs[d]
    check(len(alerts) >= 3, f"day {d}: only {len(alerts)} alerts generated")
    s = summarise(alerts)
    check(s["real"] >= 2, f"day {d}: {s['real']} real threats, expected at least 2")
    check(s["false_positives"] >= 1, f"day {d}: no hard false positive to catch out a trigger-happy analyst")

# ── 3. no repeats within a day, unique ids across days ────────────────
for d in DAYS:
    types = [a["threat_type"] for a in runs[d]]
    check(len(types) == len(set(types)), f"day {d}: duplicate threat_type in one day: {types}")
    ids = [a["id"] for a in runs[d]]
    check(len(ids) == len(set(ids)), f"day {d}: duplicate alert ids")
    check(all(f"D{d}-" in i for i in ids), f"day {d}: id does not carry its day")

# ── 4. schema and conventions ─────────────────────────────────────────
for d, alerts in runs.items():
    for a in alerts:
        extra = set(a) - EXPECTED_KEYS - OPTIONAL_KEYS
        missing = EXPECTED_KEYS - set(a)
        check(not extra, f"day {d} {a['id']}: unexpected keys {sorted(extra)}")
        check(not missing, f"day {d} {a['id']}: missing keys {sorted(missing)}")
        check(TSRE.match(a["timestamp"]), f"{a['id']}: bad timestamp {a['timestamp']!r}")
        check(a["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW"), f"{a['id']}: bad severity")
        check(0 < a["confidence"] <= 100, f"{a['id']}: confidence out of range")
        check(IPRE.match(a["src_ip"]) and IPRE.match(a["dst_ip"]), f"{a['id']}: malformed IP")
        check(RFC1918.match(a["src_ip"]) or RFC1918.match(a["dst_ip"]),
              f"{a['id']}: neither end of the connection is internal")
        check(a["mitre_technique"].startswith("T"), f"{a['id']}: bad MITRE technique {a['mitre_technique']!r}")
        check(len(a["playbook_steps"]) >= 3, f"{a['id']}: playbook has fewer than 3 steps")
        check(a["raw_log"] and len(a["raw_log"]) > 40, f"{a['id']}: raw log too thin to analyse")
        check(a["status"] == "open" and a["triage_status"] == "unreviewed",
              f"{a['id']}: should start untriaged")
        host = a["src_host"]
        check(HOSTPAT.match(host) or host.startswith("(external")
              or host in {h for h, _ in POOLS["workstations"] + POOLS["servers"]},
              f"{a['id']}: hostname {host!r} breaks the naming convention")

# ── 5. ground truth is coherent — this is what grading depends on ─────
for d, alerts in runs.items():
    for a in alerts:
        if a["is_real_threat"]:
            check(a["correct_action"] == "escalate", f"{a['id']}: real threat not marked escalate")
            check(a["false_positive_indicators"] is None,
                  f"{a['id']}: real threat carries false-positive indicators")
        else:
            check(a["correct_action"] == "resolve", f"{a['id']}: false positive not marked resolve")
            check(a["false_positive_indicators"],
                  f"{a['id']}: false positive gives the analyst nothing to reason from")
        check(a["points"] > 0, f"{a['id']}: non-positive points")

# ── 6. entities come from the sim's own inventory ─────────────────────
known_hosts = {h for h, _ in POOLS["workstations"] + POOLS["servers"]}
known_ips = {ip for _, ip in POOLS["workstations"] + POOLS["servers"]}
for d, alerts in runs.items():
    for a in alerts:
        internal = a["src_ip"] if RFC1918.match(a["src_ip"]) else a["dst_ip"]
        # Injected infrastructure hosts (pentest VM, SCCM) are deliberately
        # outside the noise pools; at least one end must still be a real asset.
        check(internal in known_ips or a["src_ip"] in known_ips or a["dst_ip"] in known_ips,
              f"{a['id']}: touches no host from the simulation inventory")
        if a["src_host"] in known_hosts:
            check((a["src_ip"], a["src_host"]) in {(i, h) for h, i in
                                                   POOLS["workstations"] + POOLS["servers"]},
                  f"{a['id']}: host {a['src_host']} paired with the wrong IP {a['src_ip']}")

# ── 7. telemetry is stitched and internally consistent ────────────────
for d, alerts in runs.items():
    for a in alerts:
        t = a.get("edr_telemetry")
        check(t, f"{a['id']}: no EDR telemetry — the EDR console would not show it")
        if not t:
            continue
        tree = t["process_tree"]
        check(len(tree) >= 2, f"{a['id']}: process tree is not a chain")
        pids = {p["pid"] for p in tree}
        check(len(pids) == len(tree), f"{a['id']}: duplicate pids in process tree")
        for p in tree[1:]:
            check(p["ppid"] in pids, f"{a['id']}: pid {p['pid']} has orphan ppid {p['ppid']}")
        check(all(len(p["hash"]) == 64 for p in tree), f"{a['id']}: process hash is not sha256-shaped")
        times = [p["start_ts"] for p in tree]
        check(times == sorted(times), f"{a['id']}: process tree is not chronological")
        check(t["collected_at"] > a["timestamp"], f"{a['id']}: telemetry collected before the event")
        server_hosts = {h for h, _ in POOLS["servers"]}
        if a["src_host"] in server_hosts:
            win_host = a["src_host"].startswith(WINDOWS_SERVER_PREFIXES)
            check(win_host == t["os"].startswith("Windows"),
                  f"{a['id']}: {a['src_host']} reported as {t['os']}")
        if a["is_real_threat"]:
            check(any(p["malicious"] for p in tree), f"{a['id']}: real threat has no malicious process")
        else:
            check(not any(p["malicious"] for p in tree),
                  f"{a['id']}: false positive flags a process as malicious")

# ── 8. difficulty ramps: early days stay readable ─────────────────────
check(max(a["difficulty"] for a in runs[1]) <= 2, "day 1 includes a difficulty-3 incident")
check(max(a["difficulty"] for a in runs[2]) <= 2, "day 2 includes a difficulty-3 incident")
declared = {fn.__name__: diff for fn, _srv, diff in THREAT_TEMPLATES + FP_TEMPLATES}
check(len(declared) == len(THREAT_TEMPLATES) + len(FP_TEMPLATES), "duplicate template registered")

# ── 9. enrichment still applies cleanly ───────────────────────────────
for d, alerts in runs.items():
    soc_ids = [enrich(dict(a))["alert_id"] for a in alerts]
    check(len(soc_ids) == len(set(soc_ids)),
          f"day {d}: two alerts enrich to the same SOC id: {soc_ids}")

for a in [x for v in runs.values() for x in v]:
    e = enrich(dict(a))
    check(e["alert_id"].startswith("SOC-2026-"), f"{a['id']}: enrichment produced no SOC id")
    check(e["rule_name"] and e["rule_name"] != "Unclassified Detection Rule",
          f"{a['id']}: threat_type {a['threat_type']!r} has no rule name in _SOC_RULE_NAMES")
    check(e["hostname"] and e["hostname"] != "-", f"{a['id']}: enrichment found no hostname")
    check(e["mitre_tactic_full"] != "-", f"{a['id']}: MITRE tactic {a['mitre_tactic']!r} not in _SOC_TA_CODES")
    check(e["detection_time"] >= e["event_time"], f"{a['id']}: detection precedes the event")

if fails:
    for f in fails:
        print("FAIL —", f)
    print(f"\n{len(fails)} failure(s)")
    sys.exit(1)

total = sum(len(v) for v in runs.values())
print(f"OK  {total} generated alerts across days {DAYS} — schema, ground truth, "
      f"telemetry, determinism and difficulty ramp all hold")
