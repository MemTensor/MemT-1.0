# MemT SGLang

[English](README.md) | [简体中文](README_zh.md)

`memt-sglang` provides native session memory for MemT-1.0 models on SGLang 0.5.16. The Python import namespace is `memt_sglang`.

| Model | Backbone | Example deployment configuration | Startup script |
| --- | --- | --- | --- |
| MemT-1.0-Nano | Qwen3.6-35B-A3B | 2 GPUs, TP2 / EP1 / DP1 | [start_memt_nano.sh](start_memt_nano.sh) |
| MemT-1.0-Flash | DeepSeek-V4-Flash | 8 GPUs, TP8 / EP8 / DP8, attention TP1 | [start_memt_flash.sh](start_memt_flash.sh) |

The scripts provide one runnable starting configuration. They have not been deeply optimized; adjust parallelism, token budgets and concurrency for your hardware and workload. The Nano backbone's HF configuration identifier is `qwen3_5_moe`.

- [Installation](#installation)
- [Start a server](#start-a-server)
- [Python client](#python-client)
- [HTTP reference](#http-reference)
- [Checkpoint and snapshot formats](#checkpoint-and-snapshot-formats)
- [Licensing](#licensing)

## Installation

Server deployment requires Linux, NVIDIA CUDA and Python 3.11 or newer. In your selected environment, install from this directory:

```bash
uv pip install '.[server]'
```

For a client talking to an existing server:

```bash
uv pip install '.[client]'
```

The `server` extra pins PyTorch 2.11.0, SGLang 0.5.16, Transformers 5.12.1, flashinfer-python[cu13] 0.6.14 and sglang-kernel 0.4.5. The `client` extra requires Transformers without installing PyTorch. The package includes a vendored subset of flash-linear-attention 0.5.2 (MIT).

Model weights and tokenizer assets are distributed through Hugging Face:

- MemT-1.0-Nano: <https://huggingface.co/MemTensor/MemT-1.0-Nano>
- MemT-1.0-Flash: <https://huggingface.co/MemTensor/MemT-1.0-Flash>

## Start a server

Use a complete local MemT checkpoint directory containing matching BF16 safetensors, configuration and tokenizer assets:

```bash
sh start_memt_nano.sh /models/MemT-1.0-Nano
sh start_memt_flash.sh /models/MemT-1.0-Flash
```

Run one command for the model you intend to serve. Both scripts call `python -m memt_sglang.launch`, default to `127.0.0.1:30000` and store memory snapshots in `./memt-state`. Extra launcher arguments are appended, for example:

```bash
sh start_memt_flash.sh /models/MemT-1.0-Flash \
  --port 30001 --state-directory /data/memt-state
```

### Example script settings

| Setting | Nano | Flash |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1` | `0,1,2,3,4,5,6,7` |
| TP / EP / DP | 2 / 1 / 1 | 8 / 8 / 8 |
| Sessions per rank / in-flight commits per rank | 6 / 2 | 6 / 1 |
| Minimum KV tokens | 2048 | 2048 |
| Maximum prefill tokens | 16384 | 16384 |
| Maximum total tokens / running requests | 65536 / 8 | 65536 / 48 |
| Static memory fraction | 0.60 | 0.70 |
| Weight dtype | BF16 | BF16 |
| KV cache | Engine default | `fp8_e4m3` |
| Context length | Engine/config default | 16384 |
| Attention / linear-attention backend | Triton / Triton | Engine default |
| MoE / sampling backend | Triton / PyTorch | Triton / PyTorch |
| Retained streaming sessions | Disabled | Enabled |

Both scripts use explicit memory/KV budgets for single-node BF16 serving. Per-forward metadata caching across memory layers is enabled by default. Nano ordinary and slot readers share a bounded, CUDA-stream-keyed LRU cache of immutable query geometry across layers and forwards. This cache is independent of `MEMT_EXPERIMENT_METADATA_CACHE`, which controls token/slot metadata reuse within one forward; request IDs, live state-slot IDs, memory contents and layer parameters are not cached in query geometry. Slot reads and prefill/decode CUDA Graph replay require explicit opt-in; incompatible batches fall back to eager execution. These values are deployment examples; extra native SGLang arguments remain subject to startup validation.

### Launcher arguments

The launcher identifies the model (Nano or Flash) from the top-level `model_type` in `config.json` and validates its model-specific fields. There is no separate server-side model flag.

| Argument | Meaning |
| --- | --- |
| `--model-path` | Required local HF checkpoint directory. |
| `--max-sessions-per-rank` | Override `memt_runtime.max_sessions_per_rank`. |
| `--max-inflight-commits-per-rank` | Override `memt_runtime.max_inflight_commits_per_rank`. |
| `--min-kv-tokens` | Override `memt_runtime.min_kv_tokens`. |
| `--json-model-override-args` | Accepts only `{"memt_runtime": {...}}`; model structure comes from `config.json`. |
| `--state-directory` | Server-owned snapshot directory; required for export/import. |
| `--deterministic-communication` | Select fixed collective settings for reproducibility experiments; this does not guarantee whole-model determinism. |
| `--dry-run` | Validate the model/runtime configuration and print the launch identity and proposed SGLang arguments without loading SGLang or model weights. |

For example:

```bash
python -m memt_sglang.launch --model-path /models/MemT-1.0-Nano --dry-run
```

Dry-run does not validate SGLang arguments. Normal startup validates the actual `ServerArgs` returned by SGLang after parsing and post-processing, before launching the server. Configuration validation does not execute checkpoint Python model code.

Serving requires single-node unquantized BF16 safetensors. Chunked prefill (`--chunked-prefill-size -1`) and overlap are disabled by default; radix cache must remain disabled. Compatible prefill/decode shapes may use CUDA Graph replay; incompatible shapes use eager execution. `max_prefill_tokens` must be at least the checkpoint's `memt.chunk_tokens`.

Nano requires DP attention to be disabled. Flash enables DP attention and requires `tp_size == dp_size`, giving attention TP1. The launcher rejects `trust_remote_code`, `--config` merging and request header overrides. Additional unsupported execution modes are listed in [validation.py](memt_sglang/validation.py).

### Environment variables

| Variable | Meaning |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPUs used by the startup script; defaults are listed above. |
| `MEMT_PYTHON` | Python executable used by the startup scripts; default `python`. |
| `MEMT_PORT` | Startup script port; default `30000`. |
| `MEMT_STATE_DIRECTORY` | Startup script snapshot directory; default `./memt-state`. |
| `MEMT_API_KEY` | API key read by the CLI and supplied examples, sent as `Authorization: Bearer ...`. This does not configure the server's API key. |
| `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES` | Must remain disabled; enabling it is rejected at startup. |
| `MEMT_EXPERIMENT_METADATA_CACHE` | Defaults to `1`; reuse token/slot metadata across memory layers within one forward. Set to `0` to disable. |
| `MEMT_EXPERIMENT_SLOT_READ` | Set to `1` to read state slots directly; defaults to `0`. |
| `MEMT_EXPERIMENT_PREFILL_GRAPH` | Set to `1` to opt into prefill graphs; defaults to `0`. Nano supports native capture-gear padding while preserving valid-token counts and live request spans. Flash requires the explicit configuration and exact local-gear checks below. |
| `MEMT_EXPERIMENT_DECODE_GRAPH` | Set to `1` for decode graphs; defaults to `0`. |
| `MEMT_EXPERIMENT_PREFILL_OVERLAP` / `MEMT_EXPERIMENT_DECODE_OVERLAP` | Independently enable native phase overlap with `1`; both default to `0`. Commit boundaries synchronize. |
| `MEMT_EXPERIMENT_QUERY_CHUNKS` | Set to `1` and supply `--chunked-prefill-size` to chunk queries; the resolved per-rank budget must be ≥2048. Defaults to `0`. Complete commits remain one forward and may exceed this query budget. |
| `MEMT_EXPERIMENT_QUERY_COALESCE_MS` | Idle-boundary wait in ms: `0` disables it (default), otherwise `(0, 10]`. Supports Nano TP2 or Flash TP=DP=EP4/8. |
| `MEMT_EXPERIMENT_QUERY_COALESCE_REQUESTS` / `MEMT_EXPERIMENT_QUERY_COALESCE_TOKENS` | Release when either target is reached; defaults to 4 requests / 4096 input tokens when waiting is enabled. These are release thresholds, not batch limits. |

For example, enable the Nano feature combination:

```bash
MEMT_EXPERIMENT_METADATA_CACHE=1 MEMT_EXPERIMENT_SLOT_READ=1 \
MEMT_EXPERIMENT_PREFILL_GRAPH=1 MEMT_EXPERIMENT_DECODE_GRAPH=1 \
sh start_memt_nano.sh /models/MemT-1.0-Nano
```

Use `--cuda-graph-config` to select captured batch/token shapes. Flash supports explicit experimental prefill and decode graph requests on the pinned PyTorch 2.11.0 / SGLang 0.5.16 stack. The prefill experiment requires TP4 / DP4 / EP4, PP1, DP attention, A2A `none`, disabled overlap, unquantized BF16 weights and the pinned native compatibility checks. This is distinct from the TP8 script example above. Prefill capture gears default to `[512, 1024]` when requested; `[512, 1024, 4096]` must be selected explicitly. Prefill replay additionally requires all four ranks to have equal, nonzero original token counts, each rank's actual token count to match a captured local gear exactly, compatible metadata and agreement across ranks. Idle, uneven or unmatched batches fall back to eager execution. Both graph phases remain off by default; successful capture alone does not establish replay or numerical qualification. Dedicated GPU checks exercised local gears 512, 1024 and 4096 on all four ranks, with eager fallback for idle and uneven prefill batches. These checks do not establish batch invariance; see the validation results and limitations below.

All MemT memory blocks execute eagerly, including during breakable graph replay. The eager/opt-in comparison is a feature-combination comparison: eager uses metadata cache `0`, slot reads `0` and both graphs off; the opt-in combination explicitly enables metadata cache, slot reads and both graphs. Nano’s immutable query-geometry LRU remains shared by ordinary and slot readers in both configurations. Flash shares its single-token reader fast path between ordinary and slot reads.

Client concurrency, including c32, does not establish physical BS32 or per-rank token counts. Batch-shape-dependent numerical drift has been observed relative to single-request execution; graph/eager agreement for the same actual batch does not imply batch invariance. Neither BS32 nor deterministic communication guarantees identical logits across batch compositions.

Overlap and query chunking are optional experimental capabilities for community use; both default to off and neither guarantees a performance improvement. Phase overlap is independently selectable. Query chunks change native KV prefill grouping while reading the same published memory state. Complete commits are never split: defer them behind an existing prefill, or admit a whole commit exceeding the query chunk budget into an empty prefill, subject to native KV/SWA capacity checks. With Flash DP attention, SGLang divides the CLI chunk budget by DP size; `execution.query_chunk_per_rank` reports the resolved value in the server identity.

```bash
MEMT_EXPERIMENT_DECODE_OVERLAP=1 MEMT_EXPERIMENT_QUERY_CHUNKS=1 \
sh start_memt_nano.sh /models/MemT-1.0-Nano --chunked-prefill-size 8192
```

The launcher sets internal worker variables for the model (Nano or Flash), configuration identity and snapshot storage, together with the SGLang model package path and required kernel settings. These are managed by the launcher rather than supplied as deployment options.

### Validation setup and limits

The evaluated Nano checkpoint is `MemT-2.0-35B-A3B_2610161130`; these numbers do not characterize the separately distributed MemT-1.0-Nano weights. Flash uses MemT-1.0-Flash. Measurements use B300 SXM6 AC GPUs, BF16 weights, PyTorch 2.11.0+cu130 and SGLang 0.5.16. Nano runs on two GPUs with TP2/DP1/EP1; Flash uses four GPUs with TP4/DP4/EP4, attention TP1 and FP8 E4M3 KV. Both use PP1 and A2A `none`.

Eager sets metadata reuse and slot reads to `0` and disables both graphs. Best denotes the explicitly enabled combination configuration: it enables those two features and requests breakable prefill/decode graphs. Nano’s immutable query-geometry LRU is shared by both arms and both ordinary/slot paths; Flash’s single-token reader fast path is also shared by both reader paths. Nano prefill gears are `[512,1024,2048,4096,8192,16384]`, decode gears `[1,2,4,8,16,32]`; Flash uses prefill `[512,1024,4096]`, decode `[1,2,4,8]`. Shared server settings are `max_total_tokens=131072`, `max_running_requests=32`, `max_prefill_tokens=16384` and `context_length=32768`, with 40 sessions and two in-flight commits per rank. Static memory fractions are 0.6/0.8 for Nano/Flash. Fixed collective settings use `NCCL_ALGO=allreduce:tree` and disable custom all-reduce and FlashInfer all-reduce fusion; this is not full native deterministic inference.

Precision uses 64 cases per model: 16 real questions and 16 boundary input lengths, each with empty and populated memory. Populated QA memory contains the same complete selected conversation history within each model (15188 Nano tokens; 14686 Flash tokens). Every candidate follows its model's fresh eager-BS1 token prefix for 128 output positions. Relative L2 is computed independently for each complete vocabulary vector as `||candidate - golden||₂ / ||golden||₂`, then averaged over all positions; Pearson is averaged in the same way. The first-32 and first-128 windows include every vector, including threshold exceedances. Physical BS32 is checked independently of client concurrency. Internal controls retain eager-BS32, repeated BS1, slot-off and historical-main comparisons. Slot-off disables direct slot reads while keeping memory available.

In this Flash QA64 matrix, every prefill falls back to eager execution; graph arms actually use decode graphs. Prefill replay at local gears 512/1024/4096 was exercised by a separate Flash prefill replay/fallback diagnostic, outside the precision table’s QA64 coverage.

| Model / configuration | Mean relative L2, first 32 | Mean Pearson, first 32 | Mean relative L2, first 128 | Mean Pearson, first 128 |
| --- | ---: | ---: | ---: | ---: |
| Nano eager BS1 (golden) | 0.000000000 | 1.000000000 | 0.000000000 | 1.000000000 |
| Nano best BS1 | 0.000000000 | 1.000000000 | 0.000000000 | 1.000000000 |
| Nano best BS32 | 0.049405459 | 0.996819092 | 0.056124086 | 0.995009996 |
| Flash eager BS1 (golden) | 0.000000000 | 1.000000000 | 0.000000000 | 1.000000000 |
| Flash best BS1 | 0.000000000 | 1.000000000 | 0.000000000 | 1.000000000 |
| Flash best BS32 | 0.048546978 | 0.997824768 | 0.051201860 | 0.996930751 |

Both models’ BS32 results fail the original per-position relative L2 < 0.04 and Pearson ≥ 0.999 criteria against their eager-BS1 golden. Within this QA64 matrix, best, slot-off and eager are bitwise equal at the same actual batch size; the batch-shape drift is also present in eager. Artifact validation does not establish BS32 accuracy acceptance.

Nano padding was separately checked at 511/512/513 tokens with empty and populated memory and mixed-length BS2/BS3 batches. Actual graph admission selected 512/512/1024 respectively; mixed groups exercised gears 512/1024/2048. Across two rounds, MemT and the same-weight pure-backbone control each passed matched-batch eager/graph and cross-round repeatability checks, totaling 17,408 bitwise-equal full-vocabulary vector comparisons per model. These controls do not establish batch invariance. Nano retains the native maximum 2× padding factor: with minimum gear 512, inputs below 256 tokens fall back.

The separate Flash diagnostic exercised balanced all-rank prefill replay at local gears 512/1024/4096 and eager fallback for idle/uneven batches. All 96 prefill vectors matched the same-batch eager reference bitwise. Uneven-batch decode retained small differences (maximum relative L2 0.000250868; minimum Pearson 0.999999919), within the diagnostic tolerance. This does not extend the QA64 table to untested execution shapes.

### Client latency

The hardware, checkpoints, parallel configuration, graph gears and capacity settings are the same as the validation setup above. Nano uses BF16 KV and FP32 Mamba state; Flash uses FP8 E4M3 KV. Nano attention/linear attention use Triton; Flash uses dsv4 attention. Both use `moe_runner_backend=triton`, `sampling_backend=pytorch`, `tc_compiler=eager`, FCFS scheduling, disabled radix caching and streaming sessions with a one-token stream interval. The supplied scheduling conservativeness is 1.0, resolved to 0.3 by native Flash DP attention; Nano remains 1.0. Greedy sampling uses temperature 0, with seeds 20261020 for Nano and 511801400 for Flash.

Each query has 512 input tokens, native EOS and at most 256 output tokens. Empty and populated memory are measured separately. Populated performance memory is prepared from a 2048-token commit and imported into a fresh session before timing; it differs from the full-history precision workload. State preparation, session setup and import are outside the query timing interval. Overlap, query chunking, coalescing and diagnostic instrumentation are disabled.

Each condition uses four sequential warmup requests at client concurrency 1, or two successive waves of 32 fresh-session warmup requests at concurrency 32. It then runs three rounds of 32 measured requests, retaining every measured request and outlier. Client concurrency does not specify the physical batch size. The complete matrix contains 544 warmup and 1536 measured requests.

The table reports arithmetic means across the 96 measured requests per condition, in milliseconds. TTFT is submission to the first valid token. Each request's TPOT is the time between its first and last valid token divided by its actual output count minus one; those per-request values are then averaged. A new token in the terminal frame counts, while duplicate terminal tokens and waiting for DONE do not extend TPOT. Actual output lengths are reported because native EOS can stop before 256 tokens; TPOT is undefined for fewer than two output tokens.

Flash best requests both prefill and decode graphs. Startup capture and the separate replay/fallback diagnostic were verified, but formal timing has no per-forward graph counters. A lone request leaves other DP ranks idle and therefore falls back for prefill; admission at concurrency 32 depends on native scheduling, and its replay/fallback frequency was not measured. This is a combined feature comparison, not a graph-only ablation.

**Nano**

| Configuration | Client concurrency | Memory | Output tokens (min / mean / max) | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | --- | ---: | ---: | ---: |
| eager | 1 | empty | 79 / 79.00 / 79 | 80.79 | 64.12 |
| eager | 1 | nonempty | 134 / 134.00 / 134 | 81.32 | 64.53 |
| eager | 32 | empty | 103 / 103.00 / 103 | 221.08 | 70.18 |
| eager | 32 | nonempty | 141 / 141.00 / 141 | 314.74 | 68.98 |
| best | 1 | empty | 79 / 79.00 / 79 | 34.82 | 7.50 |
| best | 1 | nonempty | 134 / 134.00 / 134 | 35.14 | 7.40 |
| best | 32 | empty | 79 / 102.75 / 103 | 263.02 | 9.71 |
| best | 32 | nonempty | 112 / 140.70 / 141 | 207.50 | 10.29 |

**Flash**

| Configuration | Client concurrency | Memory | Output tokens (min / mean / max) | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | --- | ---: | ---: | ---: |
| eager | 1 | empty | 256 / 256.00 / 256 | 140.99 | 120.77 |
| eager | 1 | nonempty | 50 / 204.50 / 256 | 140.46 | 120.81 |
| eager | 32 | empty | 256 / 256.00 / 256 | 644.96 | 122.85 |
| eager | 32 | nonempty | 14 / 193.05 / 256 | 581.56 | 125.03 |
| best | 1 | empty | 256 / 256.00 / 256 | 149.58 | 30.13 |
| best | 1 | nonempty | 50 / 204.50 / 256 | 151.67 | 30.79 |
| best | 32 | empty | 256 / 256.00 / 256 | 563.41 | 36.91 |
| best | 32 | nonempty | 14 / 153.12 / 256 | 635.25 | 38.55 |

All measured requests produce at least two output tokens, so TPOT is defined for every row. TPOT is lower for best in this matrix, but TTFT does not improve in every condition. Actual output lengths differ in some concurrency-32 comparisons under native EOS and native scheduling; these are end-to-end workload measurements, not fixed-output or graph-only speedups. All three formal rounds and their outliers are retained. At concurrency 32, Nano eager/nonempty round-mean TTFT is 203.09/211.71/529.43 ms, and Nano best/empty is 210.45/200.87/377.73 ms. These later-round elevations remain included; the timing data do not establish their cause.

## Python client

Use a local Nano or Flash checkpoint matching the server. Load the tokenizer with `AutoTokenizer` for text/token conversion, then call `load_chat_format(..., tokenizer=tokenizer)` to get `chat_format` for formatting messages and `parser` for interpreting model output, including reasoning and tool calls. Pass both to `MemTSession` and call `chat()` to converse.

Memory reading is enabled by default. `enable_memory=True` saves unsaved past and current turns to latent local memory after the reply completes, then clears their plaintext conversation context. This also works on the first turn; disable reading separately with `set_memory_read(False)`.

Runnable examples:

```bash
python examples/generate.py /models/MemT-1.0-Nano
python examples/document_memory.py /models/MemT-1.0-Nano
python examples/commit_history.py /models/MemT-1.0-Nano
python examples/generate.py /models/MemT-1.0-Flash
```

For Flash, pass `tokenizer_type="deepseek_v4"` to `AutoTokenizer.from_pretrained()`.

- [generate.py](examples/generate.py): two ordinary chat turns with memory reading and automatic history writes disabled; the second turn uses the conversation transcript.
- [document_memory.py](examples/document_memory.py): ingest a document and ask a question using that memory.
- [commit_history.py](examples/commit_history.py): commit complete turns, save a snapshot and restore it into a fresh session.

### Conversation with memory

```python
import os
from transformers import AutoTokenizer
from memt_sglang.client import (
    MemTSession,
    NativeClient,
    load_chat_format,
)

model_path = "/models/MemT-1.0-Nano"
tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False
)
# chat_format builds inputs; parser extracts reasoning and tool calls.
chat_format, parser = load_chat_format(
    model_path,
    tokenizer=tokenizer,
    chat_template_kwargs={"enable_thinking": False},
)
client = NativeClient("http://127.0.0.1:30000", api_key=os.environ.get("MEMT_API_KEY"))
session = MemTSession(client, chat_format, parser)

answer = session.chat(
    "I have a three-month-old kitten named memmy who loves eating "
    "small shrimp and is excellent at remembering things."
)
print(answer)

# After replying, save the kitten-description and current game Q&As to latent
# local memory and clear their plaintext conversation context.
answer = session.chat(
    "What is a simple game I can play with my kitten?",
    enable_memory=True,
)
print(answer)

# Answer using the latent local memory of the kitten-description Q&A.
print(
    session.chat("What are my kitten's name, age, favorite food, and special trait?")
)
```

`chat()` returns text by default, as in HF. With `return_dict=True` (or configured tools), it returns `answer`, completion fields and the native response metadata and `output_ids`. As in HF, generated tokens including thinking, EOS and truncated outputs are retained for replay. `generation_completed` and `turn_completed` distinguish completed generations from length-limited outputs and pending tools.

Automatic commits wait for the final answer of a tool chain, which counts as one logical turn. A truncated current reply is not automatically committed; it remains available for a later completed turn with `enable_memory=True`, `commit_history()` or `save_state()`. On success, `memory_updated=True` and `committed_turns` includes the current logical turn. Clearing conversation context removes committed turns from subsequent prompts, not from `responses` or caller-owned logs.

The following table shows successful request sequences. Each control action is a `POST /memt/control`; queries and commits use `POST /generate`.

| Python call | HTTP sequence |
| --- | --- |
| `MemTSession(...)` | `GET /memt/capabilities` → `create` with `memory_read: true`. |
| `set_memory_read(enabled)` | `acquire` → `memory_read` → `release`. |
| `chat(question)` | `acquire` → query → `release`. |
| `chat(question, enable_memory=True)` | When the reply completes without pending tools: `acquire` → query → `begin` → one commit per unsaved turn, including the current turn → `publish` → `release`. The first turn also starts a transaction. |

`chat(..., enable_thinking=False)` disables thinking for that turn. Without an override, native HF template defaults apply. `load_chat_format()` explicitly executes Flash's trusted local `encoding_dsv4.py`; the server uses its installed tokenizer class without executing checkpoint Python model code. The client and server check the HF model/tokenizer identity before creating the session.

### Save and restore

Continue the preceding example:

```python
# Save local memory, including any unsaved conversation turns.
snapshot = session.save_state("memmy_memory.safetensors")
session.close()

with MemTSession(client, chat_format, parser, state=snapshot["path"]) as restored:
    print(len(restored.messages))  # 0: the transcript is not restored.
    print(restored.chat("How old is my kitten?"))
```

| Python call | HTTP sequence |
| --- | --- |
| `commit_history()` | `acquire` → `begin` → one commit per retained turn → `publish` → `release`; no requests when history is empty. |
| `save_state(path=None, overwrite=False)` | `acquire` → `begin` → replay pending turns → `export` candidate → `publish` → `release`. |
| `close()` | `acquire` → `close`. |
| `load_state(path)` | `acquire` → `begin` → `import` → `publish` with `memory_read: true` → `release`. |

`save_state()` commits remaining history and exports that candidate before adopting it, matching HF save behavior. Saving is rejected while tool results are pending. Snapshots live on the server in `--state-directory`; `close()` releases the session without saving it automatically. Import replaces memory, clears local messages and enables memory reading. Use a fresh session when restoring a conversation's memory.

An explicit snapshot name must be a simple `.safetensors` filename. Existing files are rejected unless `overwrite=True`. Omitting `path` allocates a new ID without overwriting an existing file. Export/import requires matching configuration identity and native encoding; see [snapshot format](#memory-snapshots).

### Standard chat

Create a fresh session and disable memory reading and automatic writes.

```python
import os
from transformers import AutoTokenizer
from memt_sglang.client import (
    MemTSession,
    NativeClient,
    load_chat_format,
)

model_path = "/models/MemT-1.0-Nano"
tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True, trust_remote_code=False
)
# chat_format builds inputs; parser extracts reasoning and tool calls.
chat_format, parser = load_chat_format(
    model_path,
    tokenizer=tokenizer,
    chat_template_kwargs={"enable_thinking": False},
)
client = NativeClient("http://127.0.0.1:30000", api_key=os.environ.get("MEMT_API_KEY"))

with MemTSession(client, chat_format, parser) as chat:
    chat.set_memory_read(False)
    print(chat.chat("Type 'I love MemT' exactly.", enable_memory=False))
    print(
        chat.chat("Translate your previous answer into Chinese.", enable_memory=False)
    )
```

### Document ingestion and write boundaries

Using the client, chat format and parser above:

```python
with MemTSession(client, chat_format, parser) as document_session:
    document_session.ingest(
        "I have a three-month-old kitten named memmy who loves eating "
        "small shrimp and is excellent at remembering things."
    )
    print(
        document_session.chat(
            "What are my kitten's name, age, favorite food, and special trait?"
        )
    )
```

Memory write events use the checkpoint's `memt.chunk_tokens` (2048 in both released models), advertised as `chunk_tokens` by the server. This is the internal write-event size, not the maximum length of a commit request.

Each complete Q&A is submitted in one prefill and keeps its read snapshot and native context across internal write events. Its encoded length must fit the server's `max_prefill_tokens`. `ingest(document)` first commits any pending complete history, then submits the document as separate `chunk_tokens` chunks in the same transaction, advancing the read snapshot and resetting native context between document chunks. Successful ingestion clears local history.

### Caller-executed tools

Pass HF-style function schemas as `tools` when constructing the session. A result with `requires_action=True` includes parsed `tool_calls`; execute those functions in caller code, then provide every result with `submit_tool_results()`. All tool rounds and exact generated tokens remain one replayable turn. A new chat, commit, ingestion or save is rejected until the pending tool results are supplied.

Given the application's `tools` schemas and an `execute_tool(call)` function that returns text, create a separate tool session and handle successive rounds until no tool calls remain:

```python
with MemTSession(client, chat_format, parser, tools=tools) as tool_session:
    result = tool_session.chat(
        "Look up the requested item.", enable_thinking=False, return_dict=True
    )
    while result["requires_action"]:
        results = [
            {"tool_call_id": call["id"], "content": execute_tool(call)}
            for call in result["tool_calls"]
        ]
        result = tool_session.submit_tool_results(results)
    print(result["answer"])
```

### Other client methods and interactive use

| API | Purpose |
| --- | --- |
| `MemTSession(client, chat_format, parser, *, session=None, system_prompt=None, tools=None, state=None)` | Open a session with an optional ID and system prompt. |
| `load_chat_format(path, *, tokenizer, chat_template_kwargs=None)` | Return `(chat_format, parser)`: input templates/native encoding and reasoning/function-call parsing. |
| `session.submit_tool_results(results)` | Continue the pending turn with exactly one text result per tool call; tools execute in caller code. |
| `session.messages` | Detached messages from the uncompressed conversation. |
| `session.status()` | Inspect session state, uncertainty and the number of uncommitted turns. |
| `session.recover()` | Reconcile an uncertain operation using its transaction receipt. |
| `session.reset()` | Clear session memory and local conversation state. |
| `NativeClient(base_url, *, api_key=None, timeout=300)` | Lower-level HTTP client exposing `query`, `commit`, `commit_turns` and `control`; timeout is in seconds. |
| `MemTClientError` | Includes `uncertain`, and may carry `stage`, `transaction` and a completed `answer`. |

For interactive use:

```bash
python -m memt_sglang.client --tokenizer /models/MemT-1.0-Nano --model memt-nano
python -m memt_sglang.client --tokenizer /models/MemT-1.0-Flash \
  --model memt-flash --deepseek-encoder /models/MemT-1.0-Flash/encoding_dsv4.py
```

Use `--url URL` for another server and `--enable-memory` to commit unsaved past and current turns after each completed answer. The CLI supports `/read on|off`, `/commit`, `/save [name.safetensors]`, `/load ID`, `/status`, `/reset` and `/quit`. Set `MEMT_API_KEY` when the server requires a Bearer API key.

## HTTP reference

### Routes

| Route | Purpose |
| --- | --- |
| `GET /memt/identity` | Model (`memt-nano` or `memt-flash`), configuration SHA-256, optional checkpoint ID, SGLang version and effective `memt_runtime`. |
| `GET /memt/capabilities` | Protocol `memt-http-v1`, server epoch, snapshot format `memt-memory-v1`, `chunk_tokens`, `max_prefill_tokens`, streaming format and KV-retention support. |
| `POST /memt/control` | Sessions, leases, transactions and snapshots. |
| `POST /generate` with a MemT RID | Query or commit tokens to session memory. |

Ordinary SGLang requests without a MemT RID do not access session memory. Live weight updates, cache administration and LoRA load/unload routes are disabled and return 409.

### Sessions, leases and transactions

- **Session:** an independent memory state. Session IDs and RID nonces use `[A-Za-z0-9_.-]{1,160}`. Both HTTP `create` and Python `MemTSession` default to `memory_read: true`.
- **Lease:** exclusive access to a session. Queries and state-changing operations require a valid lease. HTTP `ttl` accepts 1–3600 seconds and defaults to 300; the Python client requests 3600 seconds for its operation lease.
- **Transaction:** a caller-named group of memory writes. Writes become visible after `publish`; `abort` discards them together.
- **Epoch:** a new identity generated at every server start. Send it as `epoch` in control requests and `memt_epoch` in generation requests to reject stale sessions after a restart. The Python client carries it automatically.
- **RID:** `memt|<session>|<operation>|<nonce>`, where the operation is `query` or `commit:<token_count>`. Use a unique nonce for each request. Any RID starting with `memt` is reserved for validation, including malformed IDs.

### Control actions

Requests use `{"action": ..., "session": ..., "lease": ..., "epoch": ..., ...}`. `create`, `acquire` and `inspect` do not require a lease on the target session; other actions do. Copying a source session requires its lease.

| Action | Additional fields | Purpose |
| --- | --- | --- |
| `create` | Optional `memory_read`, `source` + `source_lease`, `retain_kv` | Create memory, optionally copied from a source session. `retain_kv` requires `--enable-streaming-session`. |
| `acquire` / `renew` | Optional `ttl` | Obtain or extend a lease. |
| `release` | — | Reconcile outstanding work and release the lease. |
| `begin` / `abort` | `transaction` | Start or discard a transaction. |
| `publish` | `transaction`; optional `memory_read` | Publish writes and optionally set the read policy. |
| `memory_read` | `enabled` | Change the read policy while the session is idle. |
| `export` | `encoding`; optional `file`, `transaction`, `overwrite` | Export published memory or a transaction candidate. Overwriting requires `overwrite: true`. |
| `import` | `transaction`, `file`, `encoding` | Load a snapshot into a transaction; `publish` makes it visible. |
| `inspect` / `status` | — | Inspect session state / transaction status and commit receipts. |
| `recover` | Optional `ttl` | Reconcile uncertain work and renew the lease. |
| `cancel` | `request_id` | Cancel the query identified by its RID. |
| `kv_reset` | — | Clear retained native context while preserving memory. |
| `reset` / `close` | — | Clear memory / delete the session. |

### Generation requests

A commit request, after acquiring lease `L` and beginning transaction `t1` in server epoch `E`:

```json
{
  "rid": "memt|s1|commit:3|n1",
  "input_ids": [101, 102, 103],
  "sampling_params": {"max_new_tokens": 0},
  "memt_lease": "L",
  "memt_transaction": "t1",
  "memt_epoch": "E"
}
```

A query after publishing the transaction:

```json
{
  "rid": "memt|s1|query|n2",
  "input_ids": [101, 104, 105],
  "sampling_params": {"max_new_tokens": 512},
  "memt_lease": "L",
  "memt_epoch": "E",
  "stream": false
}
```

- Send one nonempty list of token IDs in `input_ids`, at most `max_prefill_tokens` long. Batched MemT requests, `text`, input embeddings and multimodal inputs are unsupported.
- For `commit:<token_count>`, the count must equal the `input_ids` length. Set `max_new_tokens` to 0 and include `memt_transaction`. A commit may span multiple `chunk_tokens` write events; it is not limited to 2048 tokens.
- `sampling_params.max_new_tokens` must be an explicit nonnegative integer, `sampling_params.n` must be 1, and `ignore_eos` must remain false. Commit requests do not return output scores or hidden states.
- Only queries support `stream: true`. SSE frames contain cumulative text; successful completion includes `finish_reason` and ends with `data: [DONE]`.
- Retained-KV queries must extend the exact acknowledged token prefix. The service manages the native session; a supplied `session_params` may only be `{"id": "<session>"}` on a query with KV retention enabled.

### Errors and recovery

MemT middleware errors use `{"error": {"type": "invalid_memory_request", "message": "..."}}`.

| HTTP status | Meaning |
| --- | --- |
| 400 | Invalid request parameters. |
| 409 | Session conflict, such as an invalid lease, busy session or epoch mismatch. |
| 429 | Session or concurrent-commit capacity exhausted. |
| 502 | Native generation returned an incomplete or invalid success response. |
| 503 | Worker communication failed or acknowledgement timed out; the affected session may be uncertain. |

For `MemTClientError.uncertain=True`, inspect status and recover before further writes; do not blindly replay a write. If the client cannot reconcile a transaction receipt, create a new session and load a confirmed snapshot. A server restart changes the epoch and requires a new session with explicit snapshot import.

## Checkpoint and snapshot formats

### Model configuration

The checkpoint's `config.json` identifies the model:

| Model | `model_type` | `architectures` | Configuration class |
| --- | --- | --- | --- |
| MemT-1.0-Nano | `memt_qwen3_5_moe` | `["MemTQwen3_5MoeForCausalLM"]` | `MemTQwen3_5MoeConfig` |
| MemT-1.0-Flash | `memt_deepseek_v4` | `["MemTDeepseekV4ForCausalLM"]` | `MemTDeepseekV4Config` |

Nano stores the backbone in `backbone_configs` with `model_type: "qwen3_5_moe"` and a nested `text_config`. Flash uses a flat HF configuration and must not contain `backbone_configs`. Both store memory parameters in `memt`. Sizes are read from `config.json`; validation pins only the fields that select the memory algorithm and the unquantized BF16 serving path, and loading checks the memory geometry against the native attention layers. `memt_checkpoint_id` is optional; Nano requires a nonempty string when it is present. The serving identity always includes the exact `config.json` SHA-256.

The optional top-level `memt_runtime` contains only `max_sessions_per_rank`, `max_inflight_commits_per_rank` and `min_kv_tokens`, all positive integers. The launcher merges defaults, checkpoint values, JSON runtime overrides and explicit capacity arguments in that order. Without overrides, each field defaults to 1; the supplied scripts set the deployment values shown above.

Weights must be unquantized BF16 safetensors. The server uses its installed configuration and model classes without executing checkpoint Python model code.

### Memory snapshots

Snapshots are server-side `.safetensors` files under `--state-directory`. Each memory layer has one finite FP32 tensor named `memory.<layer>`, with the exact shape required by the model.

Metadata contains `format=memt-memory-v1` and `identity=<HF Session identity>`. The identity covers backbone configuration, memory configuration, native encoding, chat template and tokenizer vocabulary. Initialized files contain `memory.<layer>` FP32 tensors with shape `[1, heads, K, V]`; empty HF memories contain no tensors. Identity, layer set, dtype and shape are checked before import. Transfer the file into the server's `--state-directory` to load an HF snapshot by name; server exports can be loaded by HF with the same model/tokenizer. Older SGLang files using `memory.layers.<layer>` and `model`/`encoding` metadata are incompatible with this format. Snapshots contain published memory only, without the conversation transcript or native KV cache.

## Licensing

Original package code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Bundled third-party code retains its original license and copyright notices:

| Component | License |
| --- | --- |
| MemT SGLang integration, client and Nano / Flash memory implementations | [PolyForm Noncommercial 1.0.0](LICENSE) |
| Flash memory kernels adapted from flash-linear-attention (`memt_sglang/models/deepseek/ops/_triton/intra.py`, `wy.py`) | Original FLA code: [MIT](memt_sglang/_vendor/fla/FLA_LICENSE); MemT modifications: [PolyForm Noncommercial 1.0.0](LICENSE). |
| Vendored flash-linear-attention 0.5.2 subset (`memt_sglang/_vendor/fla`) | [MIT](memt_sglang/_vendor/fla/FLA_LICENSE), copyright 2023–2026 Songlin Yang, Yu Zhang and Zhiyuan Li. |

SGLang, PyTorch, Transformers, FlashInfer and sglang-kernel are installed dependencies governed by their own licenses. Model weights, tokenizer assets and chat templates are distributed separately under the terms provided with those releases. The repository's paper license does not apply to this software package.
