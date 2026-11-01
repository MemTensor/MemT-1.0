# Validation and evidence scope

This production-cleanup branch is under validation. Final-commit GPU capture,
numerical and performance checks remain pending. The historical results below
belong to independently frozen versions, not automatically to this branch.

## Historical fast-mode evidence (2026-11-02)

Flash, DP4/TP4/EP4, B300, BF16 model and FP8 KV. Both fast and strict enabled
native and memory decode graphs, with actual owner retention on all four ranks.
One fixed nonzero S snapshot was shared by graph/eager arms. At C1/C32/C96,
16,512 complete-vocabulary positions were bit-identical after matching the
initial batch composition and row order. Uncontrolled C32 disagreement is
retained: it begins in eager prefill. Fast mode does not promise invariance to
batch composition or identical answers across independent deployments.

Exclusive timings, diverse prompts, 512 input/256 output, 10 rounds per point:

| Point | Strict TPOT ms | Fast TPOT ms | Reduction |
| --- | ---: | ---: | ---: |
| C1 empty memory | 28.241 | 16.422 | 41.85% |
| C1 nonempty memory | 28.196 | 16.550 | 41.30% |
| C32 empty memory | 37.411 | 26.180 | 30.02% |

These compare arithmetic policies with graphs on in both arms. They do not
isolate memory graph gains, owner retention cost, or SWA capacity tuning.
Full LocoMo: 1540 answers per policy, strict F1 25.9900%, fast 25.8632%, paired
conversation bootstrap delta CI95 [-0.4365, +0.1573] percentage points. This
single generation per arm does not prove quality equivalence. Team business
acceptance remains pending. Dataset category IDs are not quality scores.

No isolated experiment establishes owner retention as the cause of historical
answer changes. The dependency fix is justified by capture pointer lifetime.

## Final source validation requirements

Run `ruff check memt_sglang tests tools`, `ruff format --check memt_sglang tests tools`
and `pytest tests -q`. Report skips separately. Verify the patch in a fresh
SGLang checkout. GPU checks cover both models, nonzero S, graph/eager, actual
per-rank ownership and graph segments, slot reuse/retirement and larger buckets.
Performance checks run without diagnostic probes on an exclusive node. Source,
dependency patch, configuration, initial state and tools must be hashed together.

SWA budgets are deployment choices: measure actual SWA/full pool capacities and
request retraction for the offered concurrency. A tested ratio is not a universal
minimum. Attention TP requires a separate read/injection adaptation; removing
DP guards alone is not support. Global/tail keep native KV and compression state
within a long commit; inter-Agent history is carried by published MemT S, not
by permanent native KV. This accepted semantic difference must remain explicit.

## Example for the measured Flash layout

This is a capacity example for the measured DP4 workload, not a hardware or
correctness allowlist. Check free capacity on your own model and hardware.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH=1 \
sh start_memt_flash.sh MODEL_PATH \
  --tp-size 4 --ep 4 --dp-size 4 --max-running-requests 32 \
  --max-sessions-per-rank 8 --max-inflight-commits-per-rank 8 \
  --max-total-tokens 131072 --mem-fraction-static 0.8 \
  --swa-full-tokens-ratio 0.25 \
  --cuda-graph-config '{"prefill":{"backend":"disabled"},"decode":{"backend":"breakable","bs":[1,2,3,4,7,8,16,24,32],"max_bs":32}}'
```

The script no longer forces custom all-reduce off. Use
`--deterministic-communication` when that specific policy is requested. It does
not by itself guarantee batch-invariant model outputs. `MEMT_NUMERICS=strict`
validates the full optional DSV4 arithmetic profile and the native deterministic
flag; it is separate from the communication-only option.

## Control timeout

Agent validation previously used a 180-second tokenizer/worker control timeout;
this delivery makes that the default and exposes `--control-timeout-seconds`
(or `MEMT_CONTROL_TIMEOUT_SECONDS`). The startup identity records the effective
value. Timeout changes do not alter model math, numeric criteria, or successful
transaction semantics. An uncertain write still requires receipt reconciliation;
increasing or configuring the timeout does not authorize replaying it. Repeated
timeouts require diagnosis rather than repeatedly increasing this value.
