#!/usr/bin/env python3
"""Contract test for the ComplianceHub Step-5 write-back endpoints.

    POST /api/grc/{control_id}/evidence/{ev_id}/verdict
    POST /api/grc/{pbc_id}/request
    POST /api/grc/{pbc_id}/remind
    POST /api/grc/{pbc_id}/escalate

Run: python3 test_grc_writeback.py    (exit 0 pass / 1 fail / 2 prereq missing)

Unlike test_grc_model.py (which AST-extracts the seed builders and needs
nothing), these are endpoints that mutate Mongo — so this one needs a live
server on :8000 and MONGO_URI in the environment:

    set -a && . ./.env && set +a && python3 test_grc_writeback.py

It provisions its own throwaway user and asserts against that sim only, so it
is safe to run repeatedly against a shared dev database.

The gates, in priority order:
  1. SILENCE      accepting flawed evidence must produce NO eval row, NO score
                  change, and a message identical to the clean-accept message.
                  This is the core teaching mechanic — if it ever starts
                  leaking, the trap is dead.
  2. IDEMPOTENCE  re-dispositioning already-dispositioned evidence must not
                  score twice.
  3. SCORING      exact point values on all four reject/accept branches.
  4. STATUS       control status recomputation, incl. that a pending exception
                  does NOT make a control pass.
  5. GATES        validation + chase ordering (can't remind before requesting,
                  escalating before chasing costs points).
  6. DIEGETIC     no response tells the user they were right or wrong.
"""
import json
import os
import sys
import uuid

FAILS = []
CHECKS = [0]


def check(name, cond, detail=""):
    CHECKS[0] += 1
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}" + (f" — {detail}" if detail else ""))
        FAILS.append(name)


def main():
    try:
        import requests
        from pymongo import MongoClient
    except ImportError as ex:
        print(f"PREREQ — missing dependency: {ex}")
        return 2

    base = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("PREREQ — MONGO_URI not set. Run: set -a && . ./.env && set +a")
        return 2
    try:
        requests.get(base, timeout=5)
    except Exception:
        print(f"PREREQ — no server at {base}. Start it: nohup bash run.sh > server.log 2>&1 &")
        return 2

    db = MongoClient(uri, serverSelectionTimeoutMS=5000)["firstdesk"]

    # ---- provision an isolated grc sim -----------------------------------
    email = f"grcwb_{uuid.uuid4().hex[:10]}@test.local"
    tok = requests.post(f"{base}/api/auth/signup", timeout=30, json={
        "name": "GRC WB", "email": email, "password": "testpass123", "consent": True}).json()["token"]
    H = {"Authorization": f"Bearer {tok}"}
    r = requests.post(f"{base}/api/start", headers=H, timeout=200,
                      json={"career": "grc", "experience_level": "studied", "reset": True})
    if r.status_code != 200:
        print(f"PREREQ — /api/start failed: {r.status_code} {r.text[:200]}")
        return 2

    uid = db.users.find_one({"email": email})["user_id"]
    sim, sess = f"sim_{uid}", f"sess_{uid}"

    def score():
        return (db.sessions.find_one({"session_id": sess}) or {}).get("score_adj", 0)

    def evals(kind):
        # NB: the collection is "evaluations", not "evals" — using the wrong
        # name makes every eval-row assertion pass vacuously (0 == 0).
        return db.evaluations.count_documents({"simulation_id": sim, "type": kind})

    def ctrl(cid):
        return db.grc_controls.find_one({"simulation_id": sim, "doc_type": "control", "control_id": cid})

    def ev(cid, evid):
        c = ctrl(cid)
        return next((e for e in (c.get("evidence_items") or []) if e.get("ev_id") == evid), None)

    def pbc(pid):
        return db.grc_controls.find_one({"simulation_id": sim, "doc_type": "pbc", "pbc_id": pid})

    def verdict(cid, evid, **body):
        body.setdefault("rationale", "Checked the period, the approver and the source system of record.")
        return requests.post(f"{base}/api/grc/{cid}/evidence/{evid}/verdict",
                             headers=H, json=body, timeout=30)

    # The seed gives 3 controls, each with one flawed and one clean item @15pts.
    FLAWED = {"AC-02": ("EV-0041", "wrong_period"),
              "CFG-01": ("EV-0042", "insufficient_source"),
              "GV-01": ("EV-0043", "missing_approver")}
    CLEAN = {"AC-02": "EV-0044", "CFG-01": "EV-0045", "GV-01": "EV-0046"}

    for cid, (evid, reason) in FLAWED.items():
        got = ev(cid, evid)
        if not got or not got.get("is_flawed") or got.get("correct_reject_reason") != reason:
            print(f"PREREQ — seed drift: {cid}/{evid} is not the expected flawed item ({got})")
            return 2

    # ---- 1. validation gates --------------------------------------------
    print("\nvalidation")
    check("unknown control -> 404", verdict("NOPE-99", "EV-0041", verdict="accept").status_code == 404)
    check("unknown evidence -> 404", verdict("AC-02", "EV-9999", verdict="accept").status_code == 404)

    r = verdict("GV-01", "EV-0046", verdict="accept", rationale="too short")
    check("short rationale -> 400 rationale_required",
          r.status_code == 400 and (r.json().get("error") or {}).get("code", "") == "rationale_required"
          or "rationale_required" in r.text, f"{r.status_code} {r.text[:120]}")

    r = verdict("GV-01", "EV-0046", verdict="reject")
    check("reject without reason -> 400 reject_reason_required",
          r.status_code == 400 and "reject_reason_required" in r.text, f"{r.status_code} {r.text[:120]}")

    check("failed validation did not mutate evidence",
          ev("GV-01", "EV-0046").get("status") == "awaiting_review")

    # ---- 2. accept-clean -------------------------------------------------
    print("\naccept-clean (AC-02 / EV-0044)")
    s0, e0 = score(), evals("grc_verdict")
    r = verdict("AC-02", "EV-0044", verdict="accept")
    clean_body = r.json()
    check("200", r.status_code == 200, r.text[:150])
    check("scores +15", score() - s0 == 15, f"delta={score()-s0}")
    check("writes one eval row", evals("grc_verdict") - e0 == 1)
    check("evidence -> accepted", ev("AC-02", "EV-0044").get("status") == "accepted")
    check("control -> passing", ctrl("AC-02").get("status") == "passing",
          f"got {ctrl('AC-02').get('status')}")
    check("no pending exception armed", ev("AC-02", "EV-0044").get("pending_exception_day") is None)

    # ---- 3. idempotence --------------------------------------------------
    print("\nidempotence")
    s1, e1 = score(), evals("grc_verdict")
    r = verdict("AC-02", "EV-0044", verdict="reject", reject_reason="wrong_period")
    check("repeat disposition -> already", r.status_code == 200 and r.json().get("already") == "accepted",
          r.text[:150])
    check("no double score", score() == s1, f"delta={score()-s1}")
    check("no extra eval row", evals("grc_verdict") == e1)

    # ---- 4. THE SILENCE MECHANIC ----------------------------------------
    print("\naccept-flawed — the silence mechanic (CFG-01 / EV-0042)")
    s2, e2 = score(), evals("grc_verdict")
    r = verdict("CFG-01", "EV-0042", verdict="accept")
    flawed_body = r.json()
    check("200", r.status_code == 200, r.text[:150])
    check("SILENT: no score change", score() == s2, f"delta={score()-s2}")
    check("SILENT: no eval row", evals("grc_verdict") == e2, f"delta={evals('grc_verdict')-e2}")
    check("arms delayed exception", ev("CFG-01", "EV-0042").get("pending_exception_day") is not None)
    check("pending exception does NOT pass the control",
          ctrl("CFG-01").get("status") != "passing", f"got {ctrl('CFG-01').get('status')}")
    check("message identical to clean accept (no tell)",
          flawed_body.get("message", "").replace("CFG-01", "X") ==
          clean_body.get("message", "").replace("AC-02", "X"),
          f"clean={clean_body.get('message')!r} flawed={flawed_body.get('message')!r}")
    check("response keys identical to clean accept",
          set(flawed_body) == set(clean_body),
          f"{sorted(set(flawed_body) ^ set(clean_body))}")

    # ---- 5. reject branches ---------------------------------------------
    print("\nreject-correct, right reason (AC-02 / EV-0041, wrong_period)")
    s3 = score()
    r = verdict("AC-02", "EV-0041", verdict="reject", reject_reason="wrong_period")
    check("200", r.status_code == 200, r.text[:150])
    check("full points +15", score() - s3 == 15, f"delta={score()-s3}")
    check("schedules a correction", ev("AC-02", "EV-0041").get("correction_owed_day") is not None)

    print("\nreject-correct, WRONG reason (GV-01 / EV-0043, expects missing_approver)")
    s4 = score()
    r = verdict("GV-01", "EV-0043", verdict="reject", reject_reason="wrong_period")
    check("200", r.status_code == 200, r.text[:150])
    check("half points +7", score() - s4 == 7, f"delta={score()-s4}")
    check("no correction scheduled", ev("GV-01", "EV-0043").get("correction_owed_day") is None)

    print("\nreject-clean (CFG-01 / EV-0045 — the evidence was fine)")
    s5 = score()
    r = verdict("CFG-01", "EV-0045", verdict="reject", reject_reason="wrong_scope")
    check("200", r.status_code == 200, r.text[:150])
    check("penalty -5", score() - s5 == -5, f"delta={score()-s5}")

    # ---- 6. diegetic responses ------------------------------------------
    print("\ndiegetic (no response reveals correctness)")
    # Word-boundary matching: "A correction request has been logged" is in-world
    # copy and must NOT trip this, but a bare "correct" would.
    import re as _re
    blob = json.dumps([clean_body, flawed_body, r.json()]).lower()
    leak = [w for w in ("correct", "incorrect", "wrong", "well done", "mistake", "points", "score")
            if _re.search(r"\b" + _re.escape(w) + r"\b", blob)]
    check("no correctness leak in verdict responses", not leak, f"leaked: {leak}")

    # ---- 7. PBC chase ordering ------------------------------------------
    print("\nPBC chase ordering")
    check("unknown pbc -> 404",
          requests.post(f"{base}/api/grc/PBC-XXX/request", headers=H, timeout=30).status_code == 404)

    # PBC-019 starts at chase_state 0
    check("fixture PBC-019 starts unrequested", (pbc("PBC-019") or {}).get("chase_state", 0) == 0,
          f"got {(pbc('PBC-019') or {}).get('chase_state')}")

    r = requests.post(f"{base}/api/grc/PBC-019/remind", headers=H, timeout=30)
    check("remind before request -> 400 not_requested",
          r.status_code == 400 and "not_requested" in r.text, f"{r.status_code} {r.text[:120]}")

    s6 = score()
    r = requests.post(f"{base}/api/grc/PBC-019/escalate", headers=H, timeout=30)
    check("escalate before chasing -> escalated_too_early",
          r.status_code == 200 and r.json().get("outcome") == "escalated_too_early", r.text[:150])
    check("early escalation penalty -5", score() - s6 == -5, f"delta={score()-s6}")
    check("early escalation leaves state untouched", (pbc("PBC-019") or {}).get("chase_state", 0) == 0)

    r = requests.post(f"{base}/api/grc/PBC-019/request", headers=H, timeout=30)
    check("request -> chase_state 1", r.status_code == 200 and r.json().get("chase_state") == 1, r.text[:150])
    check("request logged", any(x.get("action") == "requested" for x in (pbc("PBC-019").get("chase_log") or [])))

    r = requests.post(f"{base}/api/grc/PBC-019/request", headers=H, timeout=30)
    check("double request -> 400 already_requested",
          r.status_code == 400 and "already_requested" in r.text, f"{r.status_code} {r.text[:120]}")

    r = requests.post(f"{base}/api/grc/PBC-019/remind", headers=H, timeout=30)
    check("remind -> chase_state 2", r.status_code == 200 and r.json().get("chase_state") == 2, r.text[:150])

    r = requests.post(f"{base}/api/grc/PBC-019/escalate", headers=H, timeout=30)
    check("escalate after chasing -> escalated_properly",
          r.status_code == 200 and r.json().get("outcome") == "escalated_properly", r.text[:150])
    check("escalation -> chase_state 3", (pbc("PBC-019") or {}).get("chase_state") == 3)
    check("escalation sets evidence_owed_day", pbc("PBC-019").get("evidence_owed_day") is not None)

    # ---- 8. day-gating still holds --------------------------------------
    print("\nday gating")
    body = requests.get(f"{base}/api/grc", headers=H, timeout=30).json()
    days = [c.get("day") for c in (body.get("controls") or []) if c.get("day") is not None]
    check("no future-day controls leak on day 1", all(d <= 1 for d in days), f"days={sorted(set(days))}")

    # ---- cleanup ---------------------------------------------------------
    try:
        requests.delete(f"{base}/api/auth/account", headers=H, timeout=30)
    except Exception:
        pass

    print()
    if FAILS:
        print(f"FAIL — {len(FAILS)}/{CHECKS[0]} checks failed: {', '.join(FAILS)}")
        return 1
    print(f"PASS — {CHECKS[0]}/{CHECKS[0]} ComplianceHub write-back checks green "
          "(silence mechanic, idempotence, scoring, status recompute, chase ordering, diegetic).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
