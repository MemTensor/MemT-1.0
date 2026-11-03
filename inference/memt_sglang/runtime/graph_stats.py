"""Per-rank admission counters for the opt-in CUDA graph experiments.

Counters exist so that a recommendation can state what actually replayed and
why the rest fell back. They are a pure side channel: nothing here decides
whether a graph runs, and no counter can change a boolean returned to native.

Buckets are keyed by the value native uses to pick a captured shape, which is
  - prefill: len(batch.input_ids), the token rows in this forward, and
  - decode:  the batch size native feeds its graph key (for DP attention this
    is the cross-rank maximum, not the local batch size).

Each bucket records both the key (what was asked for) and the gear that was
actually selected, so a reviewer can tell "fell back" apart from "padded to a
larger gear".

Host-only by construction: no tensor reads, no device synchronization, no
collectives, and no per-batch logging. Counters are printed once per rank when
the worker process exits.
"""

import atexit
import json

MAX_BUCKETS = 256
_OVERFLOW = "overflow"

_fallback_reasons = frozenset(
    {
        "memt_query_veto",
        "over_max_gear",
        "below_padding_factor",
        "native_veto_other",
    }
)

_phases = {name: {} for name in ("prefill", "decode")}
_overflow = {name: 0 for name in ("prefill", "decode")}
_enabled = False


def record(*, phase, key, gear, admitted, reasons=()):
    """Add one admission decision to this rank's counters.

    key is the shape value native used to choose a captured graph, gear is the
    captured shape actually selected (None when the batch fell back), and
    reasons explains a fallback. Acceptance is never derived from here.
    """
    if phase not in _phases:
        raise ValueError(f"Unknown graph phase: {phase!r}")
    if type(key) is not int or key < 0:
        raise ValueError("Graph statistics require a non-negative integer key")
    value = _phases[phase].get(key)
    if value is None:
        if len(_phases[phase]) >= MAX_BUCKETS:
            _overflow[phase] += 1
            return
        value = _phases[phase][key] = {
            "gear": None, "admitted": 0, "fallback": 0, "reasons": {}
        }
    if admitted:
        value["admitted"] += 1
        if gear is not None:
            value["gear"] = gear
        return
    value["fallback"] += 1
    for reason in reasons or ("native_veto_other",):
        value["reasons"][reason] = value["reasons"].get(reason, 0) + 1


def snapshot():
    """Copy of the counters; safe to print while the worker keeps running."""
    return {
        "phases": {
            phase: {
                str(key): {
                    "gear": value["gear"],
                    "admitted": value["admitted"],
                    "fallback": value["fallback"],
                    "reasons": dict(value["reasons"]),
                }
                for key, value in buckets.items()
            }
            for phase, buckets in _phases.items()
        },
        "overflow_buckets": dict(_overflow),
    }


def summarize(*, model, rank, emit):
    """Emit one line per phase; separate lines because the keys differ in unit.

    rank may be a callable, resolved here because parallel state is not yet
    initialized when the worker bootstraps.
    """
    if callable(rank):
        rank = rank()
    result = snapshot()
    for phase, buckets in result["phases"].items():
        if not buckets:
            continue
        emit(
            {
                "component": "memt_graph_stats",
                "model": model,
                "phase": phase,
                "rank": rank,
                "buckets": buckets,
                "overflow_buckets": result["overflow_buckets"][phase],
            }
        )


def install_exit_summary(*, model, rank, emit):
    """Print the counters once when this worker process exits.

    The normal shutdown path returns from run_scheduler_process, so atexit
    runs. A SIGKILLed worker skips it; counters from a wedged rank are not
    worth preserving, so no stronger mechanism is used.
    """
    global _enabled
    if _enabled:
        return
    _enabled = True

    def _emit_summary():
        try:
            summarize(model=model, rank=rank, emit=emit)
        except Exception:
            # Exit-time reporting must never change the process exit status.
            pass

    atexit.register(_emit_summary)


def dumps(event):
    return json.dumps(event, sort_keys=True)
