#!/usr/bin/env python3
"""Marketing page is the front door. The simulator stays on the same origin.

Run: python3 test_landing.py   (exit 0 pass / 1 fail)
"""
import sys

from fastapi.testclient import TestClient

import app as appmod

client = TestClient(appmod.app)
_failed = 0


def check(name, ok, detail=""):
    global _failed
    if ok:
        print(f"  ok  {name}")
        return
    _failed += 1
    print(f" FAIL {name} {detail}")


def test_root_is_marketing_for_anonymous_visitors():
    r = client.get("/")
    body = r.text
    check("root status", r.status_code == 200, r.status_code)
    check("root html", "text/html" in r.headers.get("content-type", ""))
    check("root headline", "first" in body and "before your first job" in body)
    check("root careers", "Junior Accountant" in body and "Junior GRC Analyst" in body)
    check("root sign-in link", 'href="/?app=1"' in body)
    check("root is not the spa", "authScreen" not in body)
    check("origin substituted", "{{ORIGIN}}" not in body)
    check("seo tags", 'rel="canonical"' in body and 'property="og:title"' in body)
    head = client.head("/")
    check("root head is the landing page", head.status_code == 200 and int(head.headers.get("content-length", "0")) < 100000, head.headers.get("content-length"))


def test_landing_path_serves_the_same_page():
    r = client.get("/landing")
    check("landing status", r.status_code == 200, r.status_code)
    check("landing copy", "Scenario engine" in r.text and "authScreen" not in r.text)


def test_app_query_still_serves_the_simulator():
    r = client.get("/?app=1")
    check("app status", r.status_code == 200, r.status_code)
    check("app shell", "authScreen" in r.text)
    check("signup mode hook", "mode==='signup'" in r.text)


def test_signup_entry_and_password_reset_stay_on_the_simulator():
    signup = client.get("/?app=1&mode=signup")
    check("signup entry", "authScreen" in signup.text)
    reset = client.get("/?reset=abc")
    check("reset status", reset.status_code == 200, reset.status_code)
    check("reset is the spa", "authScreen" in reset.text)
    check("reset is not marketing", "Your <em>first</em> job" not in reset.text)


def test_existing_static_app_files_still_load():
    index = client.get("/index.html")
    check("index.html", index.status_code == 200 and "authScreen" in index.text)
    interview = client.get("/interview.html")
    check("interview.html", interview.status_code == 200, interview.status_code)


if __name__ == "__main__":
    test_root_is_marketing_for_anonymous_visitors()
    test_landing_path_serves_the_same_page()
    test_app_query_still_serves_the_simulator()
    test_signup_entry_and_password_reset_stay_on_the_simulator()
    test_existing_static_app_files_still_load()
    if _failed:
        print(f"\n{_failed} failed")
        sys.exit(1)
    print("\nall passed")
