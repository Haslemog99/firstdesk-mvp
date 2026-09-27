# utctime.py
# One source of "now" for the whole app.

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Current UTC time as a naive datetime.

    Identical to the value `datetime.utcnow()` returned, but without the
    deprecation. Naive on purpose: timestamps are stored as bare ISO strings
    and compared against each other, so adding an offset here would break
    every value already in Mongo.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
