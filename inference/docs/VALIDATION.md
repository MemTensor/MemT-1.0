# Validation and evidence scope

This branch has CPU, dependency-installation and GPU functional evidence. Current exclusive baselines and generation timelines are available below;
paired candidate measurements and later workload timelines remain in progress.
Evidence below identifies its source revision; results from different revisions
and comparisons must not be combined into a single speedup.

## Production-cleanup validation (2026-11-02)

At `156373d`, core CPU checks pass: 286 tests and 12 subtests, with no skips.
Ruff lint and format pass. GitHub CI also checks the bundled patch against seven
files fetched from the pinned public SGLang reference, including executable AST
checks of the actual owner-retention expression. The continuous-commit branch is
separate and passes 304 tests plus 12 subtests after integrating these core fixes.

The numerical matrix uses frozen adapter `c55344c`, fast arithmetic, nonzero
published S and a shared prompt. Graph and eager arms share the same initial
batch composition and row order. Each session contributes 128 contiguous forced
history positions and its complete FP32 vocabulary:

| Model | Concurrency | Unique positions | Bit-identical positions |
| --- | ---: | ---: | ---: |
| Nano | 1 | 128 | 128 |
| Nano | 32 | 4,096 | 4,096 |
| Flash | 1 | 128 | 128 |
| Flash | 32 | 4,096 | 4,096 |
| Flash | 96 | 12,288 | 12,288 |

All 20,736 positions have finite logits, top-1 agreement 100%, relative L2 error
zero and Pearson correlation one. Duplicate Nano TP-rank evidence is checked but
not counted twice. These instrumented runs establish graph/eager equality under
controlled histories; they are neither timing nor business-quality measurements.
The numerical driver ignored its inherited DIVERSE setting: this matrix does not
qualify concurrent distinct prompts. Separate timeline/timing collectors actually
rotate prompts. The later distinct-prompt numerical matrix is recorded below.
Later changes through `156373d` concern configuration, timeout, coalescing and
capability validation; the memory tensor implementations are unchanged. They do
not extend this numerical result to alternative selection policies.

Six functional smoke attempts at `a935789` passed: Nano/Flash optional strict
profiles, Nano TP1 control and coalescing, Nano TP4 coalescing, and Flash alternative
native MHC settings. They are small functional checks run on separate GPU groups,
not performance evidence or qualification of every topology.

Independent installation at `156373d` extracted the complete public SGLang
`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` archive into a fresh directory and
applied only the bundled seven-file patch. All seven patched hashes match the
tested dependency. Real-weight, nonzero-memory smoke passed for Nano fast,
Nano strict and Flash fast. No private overlay source was copied into this tree.
The Flash attempt's historical directory name ends in `native-topk`, but the
runner actually selected stable C4: that attempt is evidence for the default
policy only. A first launcher attempt omitted an experiment-directory variable
and exited before model startup; it remains preserved separately.

Raw evidence root on the experiment node:
`memt-dynamic-prefill-20261027/codex-production-20261102`, including
`final-{nano,flash}-c{1,32,96}-numerics.json`, `whole-overlay-audit.json`,
`public-reproduction.json` and per-attempt logs/results in the sibling
`claude-queue-20261029/out` directory. The external tools and immutable source
copies are kept outside the production Python package.

## Follow-up validation and measurements

Core `6054aa8` passes 305 CPU tests and 12 subtests. It fixes case-insensitive
native profile booleans, removes the optional GPU qualification list and
64/16384 mHC model-shape restriction, derives query budgets from native pages,
and preserves native prewarm for positive chunk budgets. The released fixed
partition remains 64; a different partition requires separate model validation.
The integration constraints and their reasons are listed in
[INTEGRATION_CONSTRAINTS.md](INTEGRATION_CONSTRAINTS.md).

Actual distinct-prompt, nonzero-memory comparisons use 32 sessions and 128
controlled history positions per session. Nano and Flash core graph/eager
comparisons each have 4,096 bit-identical complete-vocabulary positions. An
independent Nano graph replica is also bit-identical. Flash query-only candidate
`fa0f806` matches at C32 (4,096 positions) and C1 (128 positions). Ordered-overlap
candidate `d8befb8` matches both models at C32. Per-session prompts, forcing
formula, tool hashes and initial state are compared by the audit. These are
numerical results; the sampling probe synchronizes and does not establish
asynchronous overlap safety. Candidate changes remain separate from this PR.

Raw audits: `diverse-{nano,flash}-core-c32-audit.json`,
`diverse-nano-replica-c32-audit.json`,
`diverse-flash-queryonly-c{1,32}-audit.json` and
`diverse-{nano,flash}-overlap-c32-audit.json` under the evidence root above.

Source `e0bf1d4` passes real-weight functional checks for Nano TP2 and Flash DP4:
512/1,024/1,536/4,096-token queries with nonempty memory, no cached prompt tokens,
and a complete 2,048-token atomic write followed by publication and closure.
The per-rank query budget is 512; Flash uses `TRUE` profile switches with optional
strict arithmetic. These checks establish functionality, not graph/eager
numerical equality for every chunk budget. Attempts are
`chunk-e0bf1d4-{nano,flash}-c1-r2`. Their first attempts failed in the external
collector preflight before model startup and remain preserved.

The native mHC kernel separately executes width/partition pairs 8192/32,
16384/64, 16384/32 and 28672/7 with finite outputs. This only checks kernel
execution; it is not model-level numerical acceptance of those partitions.

Exclusive current baselines at `156373d`: 512 input, 256 output, distinct prompts,
nonempty S, ten complete measured rounds per point and separate warmup:

| Model | Concurrency | TPOT ms | Completed-output token/s |
| --- | ---: | ---: | ---: |
| Nano | 1 | 5.214 | 182.04 |
| Nano | 32 | 9.414 | 3043.48 |
| Flash | 1 | 16.628 | 58.44 |
| Flash | 32 | 26.134 | 1128.01 |

These are baselines, not a new optimization gain. Raw source is
`final-baselines-summary-v1.json` plus per-round outputs. The new paired cohort
reuses immutable S and starts fresh native KV for every query; comparisons must
use the same collector and initialization policy on every arm. Full-tensor
state audits run outside query timing. Control/setup cost remains separately
reportable. Current final-source and candidate brackets are pending; do not
combine their figures with this older collector to calculate a speedup.

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
