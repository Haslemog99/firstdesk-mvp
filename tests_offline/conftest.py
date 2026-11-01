"""Offline test rig: runs app.py with an in-memory Mongo and a scripted LLM.

No network, no real keys. Import order matters: pymongo.MongoClient is swapped
for mongomock before app.py creates its client at import time.
"""
import os
import sys

import mongomock
import pymongo
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("MONGO_URI", "mongodb://offline-test")
os.environ.setdefault("FIREWORKS_API_KEY", "offline-test-key")
os.environ.setdefault("AUTH_SECRET", "offline-test-secret-offline-test-secret")
os.environ["FIRSTDESK_SKIP_LLM_PROBE"] = "1"
os.environ["GLOBAL_API_LIMIT"] = "100000"
os.environ["SIGNUP_PER_IP_HOURLY"] = "100000"
os.environ["GUEST_PER_IP_HOURLY"] = "100000"
os.environ.pop("LANGSMITH_API_KEY", None)
os.environ.pop("LANGCHAIN_API_KEY", None)

_shared = mongomock.MongoClient()


def _fake_client(*_a, **_kw):
    return _shared


pymongo.MongoClient = _fake_client  # type: ignore[assignment]

import app as app_module  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


class ScriptedLLM:
    """Stands in for ChatFireworks. `mode` decides what invoke() does."""

    def __init__(self):
        self.mode = "ok"
        self.reply = "cybersecurity"
        self.calls = []

    def invoke(self, messages, config=None, **_kw):
        self.calls.append(messages)
        if self.mode == "not_found":
            raise RuntimeError("FireworksModelNotFoundError: Error code: 404 - model not found")

        class _R:
            pass

        r = _R()
        r.content = self.reply
        return r


@pytest.fixture
def app_mod():
    return app_module


@pytest.fixture
def client(app_mod):
    with TestClient(app_mod.app) as c:
        yield c


@pytest.fixture
def auth(client):
    import uuid
    email = f"t_{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/signup", json={
        "email": email, "password": "Passw0rd!long", "name": "T", "consent": True,
    })
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}
