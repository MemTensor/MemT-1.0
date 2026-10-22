"""Opt-in graph and slot readers; metadata caching is enabled by default."""

import os
from dataclasses import replace


def enabled(name, *, environment=None):
    environment = os.environ if environment is None else environment
    return environment.get("MEMT_EXPERIMENT_" + name,
                           "1" if name == "METADATA_CACHE" else "0") == "1"


def sync_graph_config(args):
    """Use resolved native backends for worker hooks; other backends run eager."""
    graph = args.cuda_graph_config
    phases = {
        phase: replace(
            getattr(graph, phase),
            backend=("breakable" if getattr(graph, phase).backend == "breakable"
                     else "disabled"),
        )
        for phase in ("prefill", "decode")
    }
    args.cuda_graph_config = replace(graph, **phases)
    os.environ.update({
        "MEMT_EXPERIMENT_" + phase.upper() + "_GRAPH":
            "1" if config.backend == "breakable" else "0"
        for phase, config in phases.items()
    })
