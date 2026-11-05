"""Session lifecycle errors and acknowledgement states shared with clients."""

from enum import StrEnum


class SessionError(RuntimeError):
    """An operation conflicts with the current session lifecycle."""


class CapacityError(SessionError):
    """The configured session or commit capacity is exhausted."""


class UnknownSessionError(SessionError):
    """The addressed session does not exist.

    Separate from SessionError because recovery branches on it: a reply lost
    for a session the worker never had means the entry is safe to drop, while
    every other SessionError describes a session that still exists and must be
    kept. The control channel maps unknown worker errors onto SessionError, so
    the two are only distinguishable by class.
    """


class ReceiptStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    ABORTED = "aborted"


class TransactionStatus(StrEnum):
    IDLE = "idle"
    OPEN = "open"
    PUBLISHED = "published"
    ABORTED = "aborted"
