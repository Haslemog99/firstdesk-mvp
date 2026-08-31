#!/usr/bin/env python3
"""End-to-end day-progression test: all 8 careers, days 1 -> 5.

Run: set -a && . ./.env && set +a && python3 test_e2e_days.py
     (exit 0 pass / 1 fail / 2 prereq missing)

Walks each career through the real user journey — /api/start, then four
/api/simulation/next-day rollovers — and asserts at every step:

  1. CONTENT      day 1 is not empty, and every later day reveals something new
  2. GATING       nothing from a future day is visible early
  3. MONOTONIC    revealed content only grows; nothing vanishes on rollover
  4. ROLLOVER     the endpoint reports the day it actually moved to, and the
                  sim timer is reset so the new day starts from 09:00
  5. TERMINAL     day 5 is the last one; a 6th rollover is refused

Optional: pass career names to limit the run, e.g.
     python3 test_e2e_days.py grc pm
"""
import os
import sys
import time
import uuid

CAREERS = ["accountant", "cybersecurity", "marketing", "developer",
           "procurement", "hr", "pm", "grc"]

# Which list endpoints matter per career (inbox/tasks are universal).
EXTRA = {
    "accountant": [("invoices", "/api/invoices", "invoices")],
    "cybersecurity": [("alerts", "/api/siem", "alerts")],
    "developer": [("jira", "/api/jira", "tickets")],
    "marketing": [("campaigns", "/api/campaigns", "campaigns")],
    "grc": [("controls", "/api/grc", "controls")],
}

FAILS = []


def fail(career, msg):
    FAILS.append(f"[{career}] {msg}")
    print(f"      FAIL {msg}")


def main():
    try:
        import requests
    except ImportError as ex:
        print(f"PREREQ — {ex}")
        return 2
    base = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
    try:
        requests.get(base, timeout=5)
    except Exception:
        print(f"PREREQ — no server at {base}")
        return 2

    careers = [c for c in sys.argv[1:] if c in CAREERS] or CAREERS
    print(f"end-to-end day progression — {len(careers)} career(s), days 1->5\n")

    def req(method, url, H, **kw):
        """One request, with backoff on the global per-IP rate limit.

        The app enforces GLOBAL_API_LIMIT (default 120/min/IP). A thorough
        sweep like this will trip it, so back off rather than reporting a
        429 as a product failure.
        """
        for attempt in range(6):
            r = getattr(requests, method)(base + url, headers=H, **kw)
            if r.status_code == 429 or "rate_limited" in (r.text[:200] if r.text else ""):
                time.sleep(6 * (attempt + 1))
                continue
            return r
        return r

    def probe(H, url, key):
        """Fetch ONCE and derive both the count and the set of day values —
        fetching twice is what tripped the rate limiter."""
        try:
            r = req("get", url, H, timeout=45)
            if r.status_code != 200:
                return None, set()
            items = r.json().get(key)
            if not isinstance(items, list):
                return None, set()
            days = {i.get("day") for i in items if isinstance(i, dict) and i.get("day")}
            return len(items), days
        except Exception:
            return None, set()

    grand_t0 = time.time()
    for career in careers:
        t0 = time.time()
        print(f"── {career}")
        email = f"e2e_{career[:4]}_{uuid.uuid4().hex[:8]}@test.local"
        try:
            tok = req("post", "/api/auth/signup", None, timeout=30, json={
                "name": "E2E", "email": email, "password": "testpass123",
                "consent": True}).json()["token"]
        except Exception as ex:
            fail(career, f"signup failed: {ex}")
            continue
        H = {"Authorization": f"Bearer {tok}"}

        r = req("post", "/api/start", H, timeout=300,
                json={"career": career, "experience_level": "studied", "reset": True})
        if r.status_code != 200:
            fail(career, f"/api/start {r.status_code}: {r.text[:120]}")
            continue

        probes = [("emails", "/api/inbox", "emails"), ("tasks", "/api/tasks", "tasks")]
        probes += EXTRA.get(career, [])

        prev = None
        for day in range(1, 6):
            snap, seen = {}, {}
            for name, url, key in probes:
                snap[name], seen[name] = probe(H, url, key)
            total = sum(v for v in snap.values() if isinstance(v, int))
            shown = "  ".join(f"{n}={snap[n] if snap[n] is not None else 'ERR'}" for n in snap)
            delta = "" if prev is None else f"  (+{total - prev})"
            print(f"   day {day}: {shown}   total={total}{delta}")

            # 1. content exists
            if total == 0:
                fail(career, f"day {day} has no content at all")
            for n, v in snap.items():
                if v is None:
                    fail(career, f"day {day}: {n} endpoint errored")

            # 2. gating — nothing from the future is visible
            for name in seen:
                future = {d for d in seen[name] if isinstance(d, int) and d > day}
                if future:
                    fail(career, f"day {day}: {name} leaks future days {sorted(future)}")

            # 3. monotonic growth
            if prev is not None:
                if total < prev:
                    fail(career, f"day {day}: content shrank {prev} -> {total}")
                elif total == prev:
                    fail(career, f"day {day}: rollover revealed nothing new (still {total})")
            prev = total

            if day == 5:
                break

            # 4. rollover
            rr = req("post", "/api/simulation/next-day", H, json={}, timeout=120)
            if rr.status_code != 200:
                fail(career, f"rollover {day}->{day+1} returned {rr.status_code}: {rr.text[:120]}")
                break
            got = rr.json().get("current_day")
            if got != day + 1:
                fail(career, f"rollover reported current_day={got}, expected {day+1}")
            clk = req("get", "/api/sim_clock", H, timeout=30).json()
            if clk.get("elapsed_real_sec", 999) > 60:
                fail(career, f"day {day+1} timer not reset (elapsed={clk.get('elapsed_real_sec')}s)")

        # 5. terminal
        rr = req("post", "/api/simulation/next-day", H, json={}, timeout=120)
        if rr.status_code == 200:
            fail(career, "a 6th day was allowed — day 5 should be terminal")
        elif "max_day_reached" not in rr.text:
            fail(career, f"day-6 refusal had unexpected shape: {rr.text[:120]}")

        try:
            req("delete", "/api/auth/account", H, timeout=30)
        except Exception:
            pass
        print(f"   ({time.time()-t0:.0f}s)")
        time.sleep(4)   # stay under the per-IP limit between careers

    print(f"\ntotal {time.time()-grand_t0:.0f}s")
    if FAILS:
        print(f"\nFAIL — {len(FAILS)} problem(s):")
        for f in FAILS:
            print("  " + f)
        return 1
    print(f"\nPASS — {len(careers)} careers walked days 1->5: content present every day, "
          "new content each rollover, no future-day leakage, day 5 terminal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
