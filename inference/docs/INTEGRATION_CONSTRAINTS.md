# Integration constraints and configuration

These checks describe the current adapter implementation. They are not a list
of hardware or benchmark configurations qualified by the authors. Supporting a
new execution layout requires adapting its ownership, metadata and state
semantics, rather than deleting the startup check alone.

| Checked configuration | Implementation reason / work required to extend it |
| --- | --- |
| Radix cache disabled | Query metadata is planned against explicit session state; generic prefix-cache reuse does not encode the published memory version. Add state-aware cache keys/invalidation before enabling it. |
| PP=1, attention CP=1, DCP=1, single node | Pool allocation and receipt/publication cover the current local world and complete memory layer set. Partitioned layers/tokens and remote ownership need corresponding control and collective protocols. |
| One tokenizer worker | The HTTP control service owns leases and transaction metadata in one manager. Multiple independent managers need a shared authoritative owner. |
| Native BF16/auto dtype, safetensors/auto loading, HF/auto config, SGLang model implementation | Loader and memory injection implement these model/weight layouts. Other formats, quantizers and model implementations require explicit loader and projection adapters. |
| Flash attention TP=1, DP size equals TP world size | Each attention-DP rank uses full memory heads. Attention TP requires rank-local projection/head reads and state layout adaptation. Nano uses its native attention-TP layout with no attention DP. |
| Flash KV auto/FP8 E4M3; Nano KV auto/BF16 | These are the implemented native attention/state layouts. Alternative KV encodings require matching backend and recurrence validation. |
| No speculative decoding | A replay row represents one real query token. Draft/verify paths need speculative state ownership and rollback, and variable-width replay planning. |
| No disaggregated serving, Ray, gRPC modes | Control hooks, tagged native requests and cancellation use the local HTTP/tokenizer-to-scheduler path. Other transports and split workers need equivalent routing and retirement acknowledgements. |
| No encoder-only, embedding, language-only or skipped tokenizer modes | The public API and loader implement text generation and acknowledged input/output token ingestion, not independent encoder/embedding protocols. |
| No LoRA or remote-code loading | The registered model loader and memory modules have a fixed weight identity. Dynamic adapter/model code needs memory/snapshot identity and attachment rules. |
| No hidden-state return or attention-DP LM head | Request and logits row ownership currently follows the normal generation output path. Alternate outputs/distribution require coordinated row metadata. |
| No prefill CP/DSA CP, two-batch or single-batch overlap | Token partitioning and concurrent forward lifecycles are not implemented by the memory planner; ordered phase overlap is a separate adapter, not these native pipelines. |
| No whole-model torch.compile | Python hooks perform session planning, injection and publication. Tensor operators compile independently. A native whole-model trace needs graph breaks or a dedicated tensor-only integration. |
| No hierarchical cache | Generic KV offload/cache invalidation does not yet own MemT published and candidate state. |
| PDMux with memory graph unsupported | Current slot and residual replay buffers are shared on one ordered stream. Concurrent replays require independent buffers and lifetime tracking. |
| Query chunk budget positive and at least one native page | Native admission rounds intermediate boundaries to pages. A sub-page budget cannot advance a long query. Atomic writes still execute whole requests. |
| Maximum prefill covers one memory write event | Atomic commits must fit their complete write event; query chunk size is a separate scheduling budget. |
| Model path/runtime override identity, request header overrides disabled | Control capabilities and state codecs must describe the actual loaded model, not a per-request replacement of its configuration. |
| Config-file merging unsupported | The launcher validates the effective model/capacity policy from its explicit arguments. Supporting another merge source requires resolving and validating the final values once. |

## Defaults and optional features

Fast arithmetic is the default numerical policy. It does not by itself activate
all graph, compilation or scheduling optimizations. Use the graph example in
VALIDATION.md for the measured Flash layout; choose capacity for the available
memory. A portable default profile will only promote paths with the relevant
functional, numerical and asynchronous lifetime evidence.

`MEMT_NUMERICS=strict` requests the optional model-specific arithmetic profile.
Its fixed mHC partition remains configurable; 64 is the released reference
default, not an exclusive supported model shape. Native kernels still enforce
their actual geometry. Different partitions require numerical validation and
must not inherit evidence from the default partition.

Startup scripts provide overridable deployment examples. GPU counts, session
budgets, graph buckets, SWA ratios and timeout defaults are not shape/hardware
allowlists or universal performance recommendations. Stable C4 selection only
requires its actual CUDA/kernel/API capabilities; qualification of individual
GPUs belongs in validation evidence.

The bundled dependency patch contains capture ownership, stable selection and
optional deterministic arithmetic in distinct hunks. Runtime hooks install once
before worker execution and delegate unrelated requests to native methods.
Pinned internal methods are an explicit compatibility boundary; an SGLang
upgrade needs a reviewed adaptation and reproduction, not merely a version
string change. The core package does not include the independent profiling or
continuous-commit experiment tools.
