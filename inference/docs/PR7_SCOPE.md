# Review and branch scope

PR7 depends on PR4 (`61f56f5`); PR4 remains open. Mainline merging must account for
that dependency. The original PR7 head `7a0566b` remains preserved remotely.

| Branch | Purpose |
| --- | --- |
| delivery/pr7-production-20261102 | Atomic memory writer plus memory decode graph, fast default, optional deterministic arithmetic, dependency patch and CI |
| followup/continuous-commit-20261102 | Global/tail continuous-KV scheduling and recovery tests; core fixes through 156373d integrated, 304 CPU tests plus 12 subtests pass |
| followup/flash-prefill-experiments-20261102 | Preserved prefill capture experiment and tests; contains source-hash/AST and shape-specific policies needing redesign |

Attention TP/head-local reads remain an independently tested experimental patch,
not a delivered core capability. Its rank-local projection and read paths need
proper integration before deployment. The core adapter retains DP attention;
removing that invariant alone would produce an unsupported model layout.

Tests for an extracted feature travel with its branch. Removed production
reference modes and diagnostic injection points must not be reintroduced merely
to reuse a benchmark. Standalone acceptance/profiling tools stay outside the
installed memt_sglang package and record any instrumentation separately.

Current execution constraints and their implementation reasons are documented in
[INTEGRATION_CONSTRAINTS.md](INTEGRATION_CONSTRAINTS.md).
