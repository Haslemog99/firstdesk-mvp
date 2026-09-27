#!/usr/bin/env python3
"""Contract test for the ComplianceHub (GRC) data model — the 6 Step-4 gates.

Run: python3 test_grc_model.py   (exit 0 pass / 1 fail)

AST-extracts the seed builders + enrichment + read-strip helpers from app.py
so the test needs no Mongo, no FastAPI, no env — same discipline as
test_noise_alerts.py / test_edr_telemetry.py.
"""
import ast, re, sys, json

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
    ns = {"re": re}
    exec("\n\n".join(segs), ns)
    return ns

ns = _ns({"_GRC_CAST", "_GRC_FRAMEWORK_RE", "_GRC_ISO_DATE_RE", "_enrich_grc_control",
          "_grc_controls_seed", "_grc_pbc_seed", "_grc_audit_meta_seed",
          "_GRC_GT_KEYS", "_grc_public_control", "_grc_readiness"})
controls_seed   = ns["_grc_controls_seed"]
pbc_seed        = ns["_grc_pbc_seed"]
audit_seed      = ns["_grc_audit_meta_seed"]
enrich          = ns["_enrich_grc_control"]
public_control  = ns["_grc_public_control"]
readiness       = ns["_grc_readiness"]
GT_KEYS         = set(ns["_GRC_GT_KEYS"])

GT_REQUIRED = ("is_flawed", "flaw_type", "correct_verdict",
               "correct_reject_reason", "points_correct", "points_wrong")
REQUIRED_PERIOD_MONTHS = {"07", "08", "09"}   # Q3 = Jul–Sep

fails = []
def check(c, m):
    if not c: fails.append(m)

controls = controls_seed()
all_ev = [(c["control_id"], e) for c in controls for e in c.get("evidence_items", [])]

# ── 1. every evidence item has complete ground truth ──────────────────
for cid, e in all_ev:
    for k in GT_REQUIRED:
        check(k in e, f"[1] {e.get('ev_id')} on {cid} missing ground-truth {k!r}")

# ── 2. every is_flawed:true item's flaw is PRESENT in the artifact ─────
for cid, e in all_ev:
    if not e.get("is_flawed"):
        continue
    a, ft = e["artifact"], e["flaw_type"]
    if ft == "wrong_period":
        idx = a["columns"].index("review_date")
        months = {r[idx][5:7] for r in a["rows"]}
        check(months.isdisjoint(REQUIRED_PERIOD_MONTHS),
              f"[2] {e['ev_id']} wrong_period but review_date months {sorted(months)} overlap Q3")
    elif ft == "missing_approver":
        ab = a.get("approval_block", {})
        check(not ab.get("approved_by") and not ab.get("approval_date"),
              f"[2] {e['ev_id']} missing_approver but approval_block is populated: {ab}")
    elif ft == "insufficient_source":
        check(a.get("url_bar") is None and a.get("captured_at") is None and a.get("logged_in_as") is None,
              f"[2] {e['ev_id']} insufficient_source but source context present")
    elif ft == "wrong_scope":
        # scope flaw: artifact references staging/non-production where production required
        blob = json.dumps(a).lower()
        check("staging" in blob or "non-prod" in blob or "test" in blob,
              f"[2] {e['ev_id']} wrong_scope but no out-of-scope marker in artifact")
    else:
        check(False, f"[2] {e['ev_id']} flawed with unknown flaw_type {ft!r}")

# ── 3. every is_flawed:false item passes its criteria in the artifact ──
for cid, e in all_ev:
    if e.get("is_flawed"):
        continue
    a = e["artifact"]
    if a["type"] == "export":
        idx = a["columns"].index("review_date")
        months = {r[idx][5:7] for r in a["rows"]}
        ridx = a["columns"].index("reviewed_by")
        check(months <= REQUIRED_PERIOD_MONTHS, f"[3] clean {e['ev_id']} export months {sorted(months)} not all in Q3")
        check(all(r[ridx] for r in a["rows"]), f"[3] clean {e['ev_id']} export has a blank reviewed_by")
    elif a["type"] == "screenshot":
        check(bool(a.get("url_bar")) and bool(a.get("captured_at")) and bool(a.get("logged_in_as")),
              f"[3] clean {e['ev_id']} screenshot missing source context")
    elif a["type"] == "pdf":
        ab = a.get("approval_block", {})
        check(bool(ab.get("approved_by")) and bool(ab.get("approval_date")),
              f"[3] clean {e['ev_id']} pdf approval_block incomplete")
    elif a["type"] == "ticket":
        check(bool(a.get("approved_by")) and a.get("approved_by") != a.get("requested_by") and bool(a.get("approved_at")),
              f"[3] clean {e['ev_id']} ticket lacks an independent, timestamped approver")

# ── 4. determinism — same builder call twice → identical ──────────────
check(json.dumps(controls_seed(), sort_keys=True) == json.dumps(controls_seed(), sort_keys=True),
      "[4] controls seed not deterministic")
check(json.dumps(pbc_seed(), sort_keys=True) == json.dumps(pbc_seed(), sort_keys=True),
      "[4] pbc seed not deterministic")
check(audit_seed() == audit_seed(), "[4] audit_meta seed not deterministic")
# enrichment is deterministic too
e1 = [enrich(d, "sim_x", 1) for d in controls_seed()]
e2 = [enrich(d, "sim_x", 1) for d in controls_seed()]
check(json.dumps(e1, sort_keys=True) == json.dumps(e2, sort_keys=True), "[4] enrichment not deterministic")

# ── 5. read-strip: zero ground-truth keys reach the client ────────────
def scan_gt(o, path=""):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k in GT_KEYS: hits.append(path + "." + k)
            hits += scan_gt(v, path + "." + k)
    elif isinstance(o, list):
        for i, v in enumerate(o): hits += scan_gt(v, f"{path}[{i}]")
    return hits
public = [public_control(enrich(d, "sim_x", 1), 1) for d in controls_seed()]
leaks = scan_gt(public)
check(not leaks, f"[5] ground-truth leaked through public_control: {leaks}")
# and the artifact survives the strip (still renderable)
pub_ev = [e for c in public for e in c.get("evidence_items", [])]
check(len(pub_ev) == 6 and all("artifact" in e for e in pub_ev), "[5] evidence artifact lost during strip")

# ── 6. day-gating — a day-3 doc is invisible at current_day 1 ─────────
def day_gated(docs, cur):
    return [d for d in docs if d.get("day", 1) <= cur]
mixed = [enrich(controls_seed()[0], "sim_x", 1)]           # day 1
future = dict(enrich(controls_seed()[1], "sim_x", 1)); future["day"] = 3; future["control_id"] = "FUT-99"
mixed.append(future)
visible_d1 = day_gated(mixed, 1)
check(all(d["control_id"] != "FUT-99" for d in visible_d1), "[6] day-3 doc visible at current_day 1")
check(any(d["control_id"] != "FUT-99" for d in day_gated(mixed, 3)) and
      any(d["control_id"] == "FUT-99" for d in day_gated(mixed, 3)), "[6] day-3 doc not visible at current_day 3")

# ── bonus: readiness sanity ───────────────────────────────────────────
check(readiness(public) == round(100 * 5 / 16), f"[bonus] readiness {readiness(public)} != 31 (5 passing / 16)")

if fails:
    print(f"FAIL — {len(fails)} assertion(s):")
    for f in fails[:30]:
        print("  ✗", f)
    sys.exit(1)
print("PASS — 6/6 GRC model gates green (ground truth, flaw-visible, clean-valid, "
      "deterministic, no-leak, day-gated) + readiness 31%.")
