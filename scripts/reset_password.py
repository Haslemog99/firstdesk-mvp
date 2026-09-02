#!/usr/bin/env python3
"""Admin password reset. Last resort when someone is locked out of FirstDesk.

Usage:
    python3 scripts/reset_password.py --email someone@example.com
    python3 scripts/reset_password.py --email someone@example.com --password 'their-choice'
    python3 scripts/reset_password.py --email someone@example.com --dry-run

Prints the new password once. Nothing is emailed, so hand it over on a channel
you trust and tell them to change it after signing in.

Reads MONGO_URI from the environment, falling back to the repo's .env. Lifts
_hash_pw out of app.py with the AST so the stored hash can never drift from
what the running app expects, and so this never boots FastAPI or LangChain.
"""
import argparse
import ast
import os
import re
import secrets
import string
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP_PY = REPO / "app.py"
DB_NAME = "firstdesk"


def _load_dotenv() -> None:
    """Mirror run.sh: source .env when the variable is not already exported."""
    env_file = REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


def _load_hash_pw():
    """AST-extract _hash_pw from app.py so hashing stays byte-identical."""
    src = APP_PY.read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_hash_pw":
            ns = {"secrets": secrets, "hashlib": __import__("hashlib"),
                  "base64": __import__("base64")}
            exec(ast.get_source_segment(src, node), ns)
            return ns["_hash_pw"]
    sys.exit("Could not find _hash_pw in app.py. Has it been renamed?")


def _generate_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def main() -> int:
    ap = argparse.ArgumentParser(description="Reset a FirstDesk user's password.")
    ap.add_argument("--email", required=True, help="account email (case-insensitive)")
    ap.add_argument("--password", help="password to set; generated if omitted")
    ap.add_argument("--mongo-uri", help="overrides MONGO_URI")
    ap.add_argument("--dry-run", action="store_true",
                    help="look the account up and report, but change nothing")
    args = ap.parse_args()

    _load_dotenv()
    uri = args.mongo_uri or os.environ.get("MONGO_URI")
    if not uri:
        return int(bool(sys.stderr.write(
            "MONGO_URI is not set. Pass --mongo-uri or fill it in .env.\n")))

    new_pw = args.password or _generate_password()
    if len(new_pw) < 8:
        return int(bool(sys.stderr.write(
            "Password must be at least 8 characters to match the signup rule.\n")))

    from pymongo import MongoClient  # imported late so --help works without pymongo

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    users = client[DB_NAME]["users"]

    email = args.email.strip()
    user = users.find_one({"email": {"$regex": f"^{re.escape(email)}$", "$options": "i"}})
    if not user:
        return int(bool(sys.stderr.write(f"No account found for {email!r}.\n")))
    if user.get("is_guest"):
        return int(bool(sys.stderr.write(
            f"{email!r} is a guest account and has no password to reset.\n")))

    print(f"account   {user['email']}")
    print(f"user_id   {user['user_id']}")
    print(f"name      {user.get('name') or '(none)'}")

    if args.dry_run:
        print("\nDry run. Nothing was changed.")
        return 0

    users.update_one({"user_id": user["user_id"]},
                     {"$set": {"password_hash": _load_hash_pw()(new_pw)}})
    # Any reset link already in their inbox should stop working now.
    client[DB_NAME]["password_resets"].delete_many({"user_id": user["user_id"]})

    print(f"\npassword  {new_pw}")
    print("\nPassword updated. Share it over a trusted channel and ask them to")
    print("change it after signing in. Any pending reset links are now dead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
