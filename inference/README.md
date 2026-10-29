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

Both scripts use eager execution and explicit memory/KV budgets for single-node BF16 serving. These values are examples, not fixed parallelism requirements; extra native SGLang arguments override script values subject to the startup checks below.

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

Serving requires single-node eager execution with unquantized BF16 safetensors. Chunked prefill is disabled (`--chunked-prefill-size -1`), and radix cache, overlap scheduling, and both prefill and decode CUDA graphs must remain disabled. `max_prefill_tokens` must be at least the checkpoint's `memt.chunk_tokens`.

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

The launcher sets internal worker variables for the model (Nano or Flash), configuration identity and snapshot storage, together with the SGLang model package path and required kernel settings. These are managed by the launcher rather than supplied as deployment options.

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

Automatic commits wait for the final answer of a tool chain, which counts as one logical turn. A truncated current reply is not automatically committed; it remains available for a later completed turn with `enable_memory=True`, `commit_history()` or `save_state()`. On success, `memory_updated=True` and `committed_turns` include the current logical turn. Clearing conversation context removes committed turns from subsequent prompts, not from `responses` or caller-owned logs.

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
