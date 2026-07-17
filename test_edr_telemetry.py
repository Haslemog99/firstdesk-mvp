#!/usr/bin/env python3
"""Contract test for edr_telemetry on signal alerts.

Run: python3 test_edr_telemetry.py   (exit 0 pass / 1 fail)

AST-extracts the day-1 alert list + _enrich_soc_alert + _generate_noise_alerts
from app.py — no Mongo, no FastAPI, no env. Asserts telemetry is stitched to
its parent alert and that noise carries none.
"""
import ast, re, sys, hashlib, random
from datetime import datetime, timedelta

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
    ns = {"re": re, "hashlib": hashlib, "random": random, "datetime": datetime, "timedelta": timedelta}
    exec("\n\n".join(segs), ns)
    return ns

ns = _ns({"_SOC_TA_CODES", "_SOC_RULE_NAMES", "_SOC_IP_REPUTATION", "_RFC1918_RE",
          "_enrich_soc_alert", "_NOISE_POOLS", "_noise_templates", "_generate_noise_alerts"})
enrich = ns["_enrich_soc_alert"]

m = re.search(r'siem_col\.insert_many\(\[_enrich_soc_alert\(a\) for a in \[(.*?)\]\]\)', SRC, re.S)
day1 = [enrich(a) for a in ast.literal_eval('[' + m.group(1).replace('"simulation_id":_sim,', '') + ']')]
by_id = {a["id"]: a for a in day1}

fails = []
def check(c, m):
    if not c: fails.append(m)

SHA256 = re.compile(r"^[0-9a-f]{64}$")

def validate_telemetry(a):
    """Full stitching + structural validation for one alert's telemetry."""
    aid = a["id"]
    t = a.get("edr_telemetry")
    check(isinstance(t, dict), f"{aid}: edr_telemetry missing/not a dict")
    if not isinstance(t, dict):
        return

    # 1 ── non-empty process tree ──────────────────────────────────────
    pt = t.get("process_tree") or []
    check(len(pt) >= 1, f"{aid}: process_tree empty")

    # 2 ── sensor / os / status / collected_at ─────────────────────────
    # sensor: equals the alerting source when that source IS an endpoint
    # sensor (EDR/AV); otherwise (SIEM/DAM/IAM-sourced alerts) telemetry is
    # pulled from the host's EDR agent.
    allowed_sensors = {a.get("source_detail")}
    if a.get("source") not in ("EDR", "AV"):
        allowed_sensors |= {"CrowdStrike Falcon", "Windows Defender"}
    check(t.get("sensor") in allowed_sensors, f"{aid}: sensor {t.get('sensor')!r} not in {allowed_sensors}")
    check(bool(t.get("os")), f"{aid}: os missing")
    check(t.get("host_status") in ("active", "contained"), f"{aid}: bad host_status {t.get('host_status')}")
    check(t.get("collected_at", "") >= a.get("detection_time", ""), f"{aid}: collected_at before detection_time")

    # 3 ── every node well-formed; hashes are sha256 ───────────────────
    pids = {}
    for p in pt:
        for k in ("pid", "ppid", "name", "cmdline", "user", "start_ts", "signed", "hash", "malicious"):
            check(k in p, f"{aid}: process node {p.get('pid','?')} missing '{k}'")
        check(isinstance(p.get("pid"), int), f"{aid}: pid not int")
        check(SHA256.match(str(p.get("hash", ""))), f"{aid}: pid {p.get('pid')} hash not sha256")
        check(a["timestamp"][:10] in str(p.get("start_ts", "")) or p.get("start_ts", "") < a["timestamp"],
              f"{aid}: pid {p.get('pid')} start_ts {p.get('start_ts')} off the alert date")
        pids[p["pid"]] = p

    # 4 ── ppid→pid forms a valid tree: >=1 root, no cycles, every ─────
    #      non-root ppid resolves to a real pid in the same tree.
    roots = [p for p in pt if p["ppid"] not in pids]
    check(len(roots) >= 1, f"{aid}: no root process (every ppid points inside the tree — cycle?)")
    # reachability from roots covers all nodes ⇒ acyclic & connected-to-root
    seen, stack = set(), [r["pid"] for r in roots]
    while stack:
        cur = stack.pop()
        if cur in seen:  # a pid reached twice ⇒ cycle
            check(False, f"{aid}: cycle detected at pid {cur}")
            break
        seen.add(cur)
        stack += [p["pid"] for p in pt if p["ppid"] == cur]
    check(seen == set(pids), f"{aid}: {len(set(pids) - seen)} node(s) not reachable from a root (orphaned ppid)")

    # 5 ── malicious chain user == affected_user (root daemons exempt) ──
    mal_users = {p["user"] for p in pt if p.get("malicious") and p["user"] not in ("root", "SYSTEM")}
    check(mal_users <= {a.get("affected_user")},
          f"{aid}: malicious-chain users {mal_users} != affected_user {a.get('affected_user')!r}")
    check(any(p.get("malicious") for p in pt), f"{aid}: no process flagged malicious")

    # 6 ── network events stitch to source_ip / dest_ip ────────────────
    for e in (t.get("network_events") or []):
        for k in ("ts", "src_ip", "dst_ip", "dst_port", "direction", "bytes", "protocol"):
            check(k in e, f"{aid}: network event missing '{k}'")
        check(e["direction"] in ("inbound", "outbound"), f"{aid}: bad direction {e.get('direction')}")
        endpoints = {e.get("src_ip"), e.get("dst_ip")}
        # each event must touch the alert's own src or dst IP
        check(bool(endpoints & {a.get("source_ip"), a.get("dest_ip")}),
              f"{aid}: network event {e.get('src_ip')}→{e.get('dst_ip')} touches neither source_ip nor dest_ip")
        if e["direction"] == "inbound":
            check(e.get("src_ip") == a.get("source_ip"),
                  f"{aid}: inbound src {e.get('src_ip')} != source_ip {a.get('source_ip')}")

    # 7 ── file event process names exist in the tree ──────────────────
    names = {p["name"] for p in pt}
    for e in (t.get("file_events") or []):
        for k in ("ts", "path", "action", "process"):
            check(k in e, f"{aid}: file event missing '{k}'")
        check(e.get("process") in names,
              f"{aid}: file event process {e.get('process')!r} not in process_tree")
        check(not re.search(r"\[\.\]|hxxp", str(e.get("path", ""))),
              f"{aid}: file path is defanged (raw telemetry should be verbatim)")

# ── the one bootstrapped alert ─────────────────────────────────────────
check("SOC-2026-0002" in [a["alert_id"] for a in day1] or by_id.get("ALERT-2026-0218-002"),
      "SOC-2026-0002 (ALERT-2026-0218-002) not found")
target = by_id["ALERT-2026-0218-002"]
validate_telemetry(target)

# ── noise carries NO telemetry ─────────────────────────────────────────
noise = ns["_generate_noise_alerts"](day=1, count=20, seed=555)
for n in noise:
    check("edr_telemetry" not in n, f"{n['id']}: noise must NOT have edr_telemetry")
# and after enrichment it still doesn't sprout one
for n in noise[:5]:
    check("edr_telemetry" not in enrich(dict(n)), f"{n['id']}: enrichment added telemetry to noise")

# ── any OTHER signal alert that later grows telemetry also validates ───
for a in day1:
    if a["id"] != "ALERT-2026-0218-002" and a.get("edr_telemetry"):
        validate_telemetry(a)

# ── day-pack (Days 2–5) telemetried alerts validate identically ────────
mp = re.search(r'_CYBER_DAY_PACKS: dict\[int, list\[dict\]\] = (\{.*?\n\})\n', SRC, re.S)
packs = ast.literal_eval(mp.group(1))
pack_tel = 0
for day, lst in packs.items():
    for raw in lst:
        a = enrich(dict(raw))
        if a.get("edr_telemetry"):
            pack_tel += 1
            validate_telemetry(a)
check(pack_tel >= 3, f"expected >=3 telemetried day-pack alerts, found {pack_tel}")

if fails:
    print(f"FAIL — {len(fails)} assertion(s):")
    for f in fails[:30]:
        print("  ✗", f)
    sys.exit(1)
print("PASS — edr_telemetry stitched to parent, valid process tree, noise clean.")
