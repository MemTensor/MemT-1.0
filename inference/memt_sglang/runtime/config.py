"""Deployment capacity, control policy, and server protocol limits.

Wire-format values shared with the client are in memt_sglang.protocol.
"""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields

# Tokenizer -> worker control round trip.
CONTROL_TIMEOUT_SECONDS = 180
# Wait for a cancelled or finished request to leave the worker (and its KV).
RETIRE_TIMEOUT_SECONDS = 30
RETIRE_POLL_SECONDS = 0.01
# String arguments of /memt/control (lease, transaction, request ID, file).
MAX_CONTROL_ARGUMENT_LENGTH = 512
# Commit requests per transaction before it must be published or aborted.
MAX_TRANSACTION_RECEIPTS = 1024
# Snapshot file names: [A-Za-z0-9_.-]{1,MAX_SNAPSHOT_NAME_LENGTH}.safetensors.
MAX_SNAPSHOT_NAME_LENGTH = 120
# Largest single SSE event accepted from the native stream.
MAX_SSE_EVENT_BYTES = 16 * 1024 * 1024


def control_timeout_seconds():
    """Deployment timeout; independent of model structure and memory capacity."""
    value = float(
        os.environ.get("MEMT_CONTROL_TIMEOUT_SECONDS", CONTROL_TIMEOUT_SECONDS)
    )
    if not math.isfinite(value) or value <= 0:
        raise ValueError("MEMT_CONTROL_TIMEOUT_SECONDS must be finite and positive")
    return value


@dataclass(frozen=True)
class RuntimeConfig:
    max_sessions_per_rank: int = 1
    max_inflight_commits_per_rank: int = 1
    min_kv_tokens: int = 1

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"memt_runtime.{field.name} must be a positive integer"
                )

    @classmethod
    def from_hf_config(cls, hf_config) -> "RuntimeConfig":
        # SGLang v0.5.16 (fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1):
        # configs/model_config.py:274-291 keeps the outer HF config;
        # utils/hf_transformers/config.py:256-257 applies a shallow update.
        # Never read runtime fields from memt or from a nested text_config.
        value = (
            hf_config.get("memt_runtime")
            if isinstance(hf_config, Mapping)
            else getattr(hf_config, "memt_runtime", None)
        )
        if not isinstance(value, Mapping):
            raise ValueError("HF config requires a top-level memt_runtime object")
        expected = {field.name for field in fields(cls)}
        missing, unknown = expected - value.keys(), value.keys() - expected
        if missing or unknown:
            raise ValueError(
                "Invalid memt_runtime fields: "
                f"missing={sorted(missing)}, unknown={sorted(unknown, key=str)}"
            )
        return cls(**value)
