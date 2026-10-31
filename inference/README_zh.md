# MemT SGLang

[English](README.md) | [简体中文](README_zh.md)

`memt-sglang` 为 MemT-1.0 模型提供基于 SGLang 0.5.16 的原生会话记忆。Python 导入命名空间为 `memt_sglang`。

| 模型 | 骨干模型 | 部署配置示例 | 启动脚本 |
| --- | --- | --- | --- |
| MemT-1.0-Nano | Qwen3.6-35B-A3B | 2 张 GPU，TP2 / EP1 / DP1 | [start_memt_nano.sh](start_memt_nano.sh) |
| MemT-1.0-Flash | DeepSeek-V4-Flash | 8 张 GPU，TP8 / EP8 / DP8，attention TP1 | [start_memt_flash.sh](start_memt_flash.sh) |

这些脚本提供可运行的起始配置，尚未经过深入优化；请根据硬件和工作负载调整并行方式、token 预算和并发数。Nano 骨干的 HF 配置标识为 `qwen3_5_moe`。

- [安装](#安装)
- [启动服务](#启动服务)
- [Python 客户端](#python-客户端)
- [HTTP 接口](#http-接口)
- [Checkpoint 与快照格式](#checkpoint-与快照格式)
- [许可证](#许可证)

### Flash 记忆 decode 图的额外要求

`MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH=1` 把记忆读取路径纳入 decode CUDA 图。它是本包中
唯一一项带来 decode 提速的改动，同时也是前置条件最多的开关，因此这里给出**一整段可复制**的
配置，而不是分散说明。

**overlay 是独立交付物。** 下面每一项能力都通过显式的 API 版本常量探测，而不是 `hasattr`，
所以 overlay 缺失时会在启动阶段失败，不会静默降级。把 overlay 放到 site-packages 之前：

```bash
export PYTHONPATH=/path/to/sglang-flash-release-v2:$PYTHONPATH
```

#### 完整开关集合

```bash
# 必需：打开图，并保留它捕获的读取路径。
export MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH=1
export MEMT_EXPERIMENT_SLOT_READ=1

# 必需：replay 一致性。捕获只记录一次设备地址与控制流，之后每次 replay 重放同一批节点，
# 因此任何"在 replay 时可能选中不同 KV 行"的因素都必须在捕获前固定。
export MEMT_EXPERIMENT_C4_STABLE_TOPK=1               # top-k 平局打破稳定
export MEMT_EXPERIMENT_PADDED_DECODE_QUERY=1          # decode 几何定长
export MEMT_EXPERIMENT_REQUIRE_NATIVE_KV_CAPACITY=1   # KV 容量未被裁剪
export SGLANG_DSV4_PIN_DECODE_INTERMEDIATES=1         # 保留被捕获的张量所有者
export SGLANG_DSV4_DETERMINISTIC_PROFILE=1            # 见下方说明

# overlay 默认值；一并列出，因为同一个校验会读取它们。
export SGLANG_OPT_USE_TILELANG_MHC_PRE=1
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
```

启动时解码图需为 breakable：

```bash
--cuda-graph-config '{"prefill":{"backend":"disabled"},"decode":{"backend":"breakable","bs":[1,2,3,4,7,8,16,24,32],"max_bs":32}}'
```

overlap 调度与 PDMux 必须关闭；开任一者在启动期即被拒绝。

#### 为什么只设 `SGLANG_DSV4_PIN_DECODE_INTERMEDIATES` 不够

overlay 里计算的是

```python
_pin_decode_graph_owners = (
    SGLANG_DSV4_DETERMINISTIC_PROFILE and SGLANG_DSV4_PIN_DECODE_INTERMEDIATES
)
```

**两个都设**才真正保留被捕获的张量。本包同时要求两者，报错会点名缺的是哪一个。
只设后者会得到一张能捕获、能 replay、不报错的图，但它读的是自己从未保留的张量存储——
实测该配置下最大 case 的 target-logprob p99 高于 0.1，而正确 pin 之后为 0.000724。

#### 成功信号

每个 rank 的日志应出现一行：

```
MemT memory decode captured: local buckets=[...], graph segments=[1], native tensor owners=N
```

**每个 bucket 都是 `graph segments=[1]` 才是关键。** 出现 `2` 或更大说明图被截断
（最可能是捕获期间重新进入了某个 Python 回调），只有部分被录制。**这不是服务器会报错的状态，
它只是变慢**，所以必须靠看日志确认。

#### 失败信息对照

| 报错 | 原因 |
| --- | --- |
| `... requires the separately supplied SGLang stable exact top-k API v1` | `PYTHONPATH` 里没有 overlay，或 overlay 版本过旧 |
| `... requires these switches to be set exactly; missing or wrong: <names>` | 上面某一行未设或设为 `0`；报错会点名 |
| `Memory decode graph requires breakable decode and SLOT_READ=1` | `--cuda-graph-config` 的 decode backend 是 `disabled`，或 `SLOT_READ` 未开 |
| `Memory decode graph requires overlap scheduling disabled` | 未传 `--disable-overlap-schedule`，或开了 overlap |
| `MemT stable C4 profile currently validates CUDA SM103 only` | 非 SM103（B300 级别）设备。kernel 本身没有架构分支，这道门是**验证范围限制**，不是硬件需求 |

数值可复现性是独立、由使用者显式选择的档位：

| `MEMT_NUMERICS` | 保证什么 | Nano 上实测代价 |
| --- | --- | --- |
| `fast`（默认） | 结果正确；同一请求的答案可能随「批里还有什么」而变 | 基准 |
| `strict` | 并发 32 下同一请求跨轮可复现 | TPOT +7.7%（C1）~ +16.6%（C32） |

依据：同一冻结初态、greedy 生成 256 token、10 轮。并发 32 时 `strict` 复现
288/288 对请求，`fast` 为 256/288（88.9%）；并发 1 时两者都是 288/288。
选择 `fast` 却同时传入 `--enable-deterministic-inference` 或任一 `SGLANG_DSV4_*`
profile 开关时报错，而不是静默降级。

## 安装

服务端部署需要 Linux、NVIDIA CUDA 和 Python 3.11 或更新版本。在选定的环境中，从本目录安装：

```bash
uv pip install '.[server]'
```

仅作为客户端连接已有服务时：

```bash
uv pip install '.[client]'
```

`server` 可选依赖固定使用 PyTorch 2.11.0、SGLang 0.5.16、Transformers 5.12.1、flashinfer-python[cu13] 0.6.14 和 sglang-kernel 0.4.5。`client` 可选依赖需要 Transformers，不会安装 PyTorch。包内包含 flash-linear-attention 0.5.2 的部分源码，采用 MIT 许可证。

模型权重和 tokenizer 文件通过 Hugging Face 分发：

- MemT-1.0-Nano：<https://huggingface.co/MemTensor/MemT-1.0-Nano>
- MemT-1.0-Flash：<https://huggingface.co/MemTensor/MemT-1.0-Flash>

## 启动服务

使用完整的本地 MemT checkpoint 目录，其中应包含配套的 BF16 safetensors 权重、配置和 tokenizer 文件：

```bash
sh start_memt_nano.sh /models/MemT-1.0-Nano
sh start_memt_flash.sh /models/MemT-1.0-Flash
```

按需要服务的模型运行其中一条命令。两个脚本均调用 `python -m memt_sglang.launch`，默认监听 `127.0.0.1:30000`，记忆快照保存在 `./memt-state`。额外的 launcher 参数追加到命令末尾，例如：

```bash
sh start_memt_flash.sh /models/MemT-1.0-Flash \
  --port 30001 --state-directory /data/memt-state
```

### 推荐组合（实测，不是声明）

每一行都在冻结初态上跑了 10 轮，并发 1 与 32，512 token 查询，TPOT 单位 ms/token。
行标题写明实际启用的开关，不用 "best" 这类笼统标签。

| 模型 | 启用的特性 | 数值档位 | C1 TPOT | C32 TPOT | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| Nano | metadata cache、slot read、decode graph | `strict` | 7.48 | 10.98 | 推荐 |
| Nano | 同上 | `fast` | 7.48 | 10.98 | 相对另一个基准快 7.7%/16.6%，见上方档位表 |
| Nano | 同上 + prefill graph | `strict` | 9.04 | 11.59 | **不推荐**：慢 21% |
| Flash | metadata cache、slot read、decode graph | `strict` | 28.12 | 32.48 | 推荐 |
| Flash | 同上 | `fast` | 16.60 | 22.58 | 快 41%/31%，但先看可复现率 |
| Flash | 同上 + prefill graph | `strict` | 28.92 | 36.70 | **不推荐**：慢 3%/13% |

两条需要直说的结论：

- **prefill 图在我们这套配置下没有收益。** Nano 与 Flash 都确认入图（服务日志有捕获记录）、
  输出逐 token 一致，但都更慢。Flash 尤其明显：只有某个 rank 的 token 数恰好等于捕获的
  gear 时才入图，并发流量几乎不可能命中。
- **Flash 用 `fast` 放弃的不是末几位小数，而是并发 32 下的可复现性**：`strict` 下
  100% 的请求对可复现，`fast` 下只有 38~42%，首个分歧出现在第 4 个 token。

**没有实测所以不在表内**：overlap、query chunking、query coalescing、partial commit。
在有人真正跑过之前，这些按「不支持」对待。

### 脚本配置示例

| 配置项 | Nano | Flash |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1` | `0,1,2,3,4,5,6,7` |
| TP / EP / DP | 2 / 1 / 1 | 8 / 8 / 8 |
| 每 rank 会话数 / 并发提交数 | 6 / 2 | 6 / 1 |
| 最小 KV token 数 | 2048 | 2048 |
| 最大 prefill token 数 | 16384 | 16384 |
| 最大总 token 数 / 运行请求数 | 65536 / 8 | 65536 / 48 |
| 静态显存比例 | 0.60 | 0.70 |
| 权重 dtype | BF16 | BF16 |
| KV cache | 引擎默认值 | `fp8_e4m3` |
| 上下文长度 | 引擎／配置默认值 | 16384 |
| Attention / linear-attention 后端 | Triton / Triton | 引擎默认值 |
| MoE / sampling 后端 | Triton / PyTorch | Triton / PyTorch |
| 保留流式会话 KV | 关闭 | 开启 |

两个脚本均显式设置记忆和 KV 预算，用于单节点 BF16 服务。默认启用每次 forward 内跨层复用的 metadata 缓存。Nano 只读查询（包括默认 packed 路径）复用同一 CUDA stream 上不可变的 query plan，实时 state-slot ID 仍逐次 forward 更新。Flash grouped K/V normalization 复用 GPU epsilon 常量，保留原 FP32 精度（FP64 输入使用 FP64）。slot read 和 prefill/decode CUDA Graph 按下述环境变量显式开启。CUDA Graph 未命中兼容形状时回退 eager。这些值是部署示例，不是固定的并行度要求；额外的 SGLang 原生参数可以覆盖脚本值，但仍须通过下述启动检查。

实测范围、数值检查及剩余实现开销见[推理优化与算法成本边界](docs/optimization_boundaries_zh.md)。

各阶段时间占比、profiler 扰动与后续优化优先级见[推理阶段 Profiling 分析](docs/phase_profile_analysis_zh.md)。

编译器融合、实时 slot graph 接入与验收见[编译与 Decode CUDA Graph](docs/compile_decode_graph_zh.md)；[性能基准工具](benchmarks/memory_decode_graph/README.md)保留请求原始计时。

### Launcher 参数

Launcher 根据 `config.json` 顶层的 `model_type` 识别模型（Nano 或 Flash），并校验对应模型的配置字段。服务端不另设模型参数。

| 参数 | 含义 |
| --- | --- |
| `--model-path` | 必填，本地 HF checkpoint 目录。 |
| `--max-sessions-per-rank` | 覆盖 `memt_runtime.max_sessions_per_rank`。 |
| `--max-inflight-commits-per-rank` | 覆盖 `memt_runtime.max_inflight_commits_per_rank`。 |
| `--min-kv-tokens` | 覆盖 `memt_runtime.min_kv_tokens`。 |
| `--json-model-override-args` | 只接受 `{"memt_runtime": {...}}`；模型结构来自 `config.json`。 |
| `--state-directory` | 服务端快照目录；导出／导入快照时必须配置。 |
| `--deterministic-communication` | 为可复现实验选择固定的集合通信设置；不保证整个模型的确定性。 |
| `--dry-run` | 校验模型和运行时配置，打印启动身份信息及拟传入 SGLang 的参数，不加载 SGLang 或模型权重。 |

例如：

```bash
python -m memt_sglang.launch --model-path /models/MemT-1.0-Nano --dry-run
```

Dry-run 不校验 SGLang 参数。正常启动会在 SGLang 完成参数解析和后处理后，对实际生成的 `ServerArgs` 进行校验，通过后才启动服务。配置校验不会执行 checkpoint 中的 Python 模型代码。

服务使用单节点未量化的 BF16 safetensors 权重。默认关闭 chunked prefill（`--chunked-prefill-size -1`）与 overlap，radix cache 必须关闭；兼容的 prefill/decode shape 可以使用 CUDA Graph，不兼容时使用 eager 路径。`max_prefill_tokens` 不得小于 checkpoint 的 `memt.chunk_tokens`。

Nano 必须关闭 DP attention。Flash 开启 DP attention，并要求 `tp_size == dp_size`，使 attention TP 为 1。Launcher 拒绝 `trust_remote_code`、`--config` 配置合并和 request header overrides。其他不支持的执行模式见 [validation.py](memt_sglang/validation.py)。

### 环境变量

| 变量 | 含义 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | 启动脚本使用的 GPU，默认值见上表。 |
| `MEMT_PYTHON` | 启动脚本使用的 Python 可执行文件，默认 `python`。 |
| `MEMT_PORT` | 启动脚本的服务端口，默认 `30000`。 |
| `MEMT_STATE_DIRECTORY` | 启动脚本的快照目录，默认 `./memt-state`。 |
| `MEMT_API_KEY` | CLI 和示例读取的 API key，以 `Authorization: Bearer ...` 发送；该变量不配置服务端的 API key。 |
| `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES` | 必须保持关闭，开启时会拒绝启动。 |
| `MEMT_EXPERIMENT_METADATA_CACHE` | 默认 `1`，在一次 forward 内跨记忆层复用 token/slot 元数据；设为 `0` 关闭。 |
| `MEMT_EXPERIMENT_SLOT_READ` | 设为 `1` 启用按 slot 直接读取，默认 `0`。 |
| `MEMT_EXPERIMENT_PREFILL_GRAPH` | Nano 可设为 `1` 启用 token 数精确匹配的 prefill graph，默认 `0`。 |
| `MEMT_EXPERIMENT_DECODE_GRAPH` | 设为 `1` 启用 decode graph，默认 `0`。 |
| `MEMT_EXPERIMENT_NANO_COMPILE` | 设为 `1` 启用保留精度转换的 Nano 输出归一化融合和启动预热，默认 `0`；Flash 保留原有编译函数。 |
| `MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH` | 设为 `1` 将只读 memory 纳入 breakable decode graph；依赖 slot read，拒绝 overlap/PDMux，默认 `0`。 |
| `MEMT_EXPERIMENT_COMMIT_PLAN_CACHE` | 设为 `1` 复用 eager writer 的不可变几何元数据；缓存有界、按 stream 隔离，不含会话或状态内容，默认 `0`；不支持 commit graph capture。 |
| `MEMT_EXPERIMENT_PREFILL_OVERLAP` / `MEMT_EXPERIMENT_DECODE_OVERLAP` | 分别设为 `1` 启用对应阶段的 native overlap，默认均为 `0`；commit 边界同步。 |
| `MEMT_EXPERIMENT_QUERY_CHUNKS` | 设为 `1` 允许 query chunk，并显式传入 `--chunked-prefill-size`；解析后每 rank 预算须 ≥2048，默认 `0`。完整 commit 仍一次 forward 执行，可超过 query 预算。 |
| `MEMT_EXPERIMENT_QUERY_COALESCE_MS` | 空闲边界主动等待上限，单位 ms，默认 `0`（关闭），启用时须在 `(0, 10]`；只支持 Nano TP2 或 Flash TP=DP=EP4/8。 |
| `MEMT_EXPERIMENT_QUERY_COALESCE_REQUESTS` / `MEMT_EXPERIMENT_QUERY_COALESCE_TOKENS` | 达到任一阈值立即放行；启用等待时默认为 4 请求／4096 输入 tokens，非 batch 上限。 |

例如，启用 Nano 的四项优化：

```bash
MEMT_EXPERIMENT_METADATA_CACHE=1 MEMT_EXPERIMENT_SLOT_READ=1 \
MEMT_EXPERIMENT_PREFILL_GRAPH=1 MEMT_EXPERIMENT_DECODE_GRAPH=1 \
sh start_memt_nano.sh /models/MemT-1.0-Nano
```

可用 `--cuda-graph-config` 指定需要捕获的 batch/token 形状。Flash 的实验配置启用 metadata、slot read 和 decode graph，prefill graph 保持关闭；历史严格 logits 对比未通过，不应视为已完成数值验收。

Overlap 与 query chunk 是供社区按负载选择的实验功能，默认关闭，不保证性能正收益。两阶段 overlap 可独立开启；query chunk 改变 native KV prefill 分组，不改变 query 读取的已发布 memory state。完整 commit 不拆分：已有 prefill 时延后；空 prefill 可执行超过 query chunk 预算的完整 commit，仍受 native KV／SWA 容量限制。Flash DP attention 会将 CLI chunk 预算除以 DP 度数，identity 中 `execution.query_chunk_per_rank` 记录解析后的每 rank 值。

```bash
MEMT_EXPERIMENT_DECODE_OVERLAP=1 MEMT_EXPERIMENT_QUERY_CHUNKS=1 \
sh start_memt_nano.sh /models/MemT-1.0-Nano --chunked-prefill-size 8192
```

Launcher 会设置向 worker 传递模型（Nano 或 Flash）、配置身份和快照目录的内部变量，以及 SGLang 模型包路径和必要的内核设置。这些变量由 launcher 管理，不作为部署配置项。

## Python 客户端

使用与服务端匹配的本地 Nano 或 Flash checkpoint。通过 `AutoTokenizer` 加载负责文本与 token 转换的 tokenizer，再调用 `load_chat_format(..., tokenizer=tokenizer)`，得到负责消息格式化的 `chat_format` 和负责解析模型输出、Thinking 及工具调用的 `parser`。将二者传入 `MemTSession`，通过 `chat()` 对话。

记忆读取默认开启。`enable_memory=True` 会在当前回答完成后，将尚未保存的历史及当前逻辑轮写入潜在局部记忆，并清除这些轮次的推理明文上下文；首轮即可使用。需要关闭读取时，单独调用 `set_memory_read(False)`。

可运行的示例：

```bash
python examples/generate.py /models/MemT-1.0-Nano
python examples/document_memory.py /models/MemT-1.0-Nano
python examples/commit_history.py /models/MemT-1.0-Nano
python examples/generate.py /models/MemT-1.0-Flash
```

Flash 在 `AutoTokenizer.from_pretrained()` 中传入 `tokenizer_type="deepseek_v4"`。

- [generate.py](examples/generate.py)：关闭记忆读取和自动历史写入，进行两轮普通对话；第二轮使用保留的对话记录。
- [document_memory.py](examples/document_memory.py)：写入文档，随后利用该记忆回答问题。
- [commit_history.py](examples/commit_history.py)：提交完整问答，保存快照并恢复到新会话。

### 带记忆的对话

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
# chat_format 负责输入模板和编码；parser 负责解析 Thinking 和工具调用。
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

# 回答后，将介绍小猫和当前游戏两轮问答保存到潜在局部记忆，并清除其推理明文上下文。
answer = session.chat(
    "What is a simple game I can play with my kitten?",
    enable_memory=True,
)
print(answer)

# 使用潜在局部记忆中保存的小猫信息回答。
print(
    session.chat("What are my kitten's name, age, favorite food, and special trait?")
)
```

`chat()` 与 HF 一样默认返回回答文本；传入 `return_dict=True` 或配置工具时，返回 `answer`、完成状态，以及原生响应元数据和 `output_ids`。与 HF 一致，回放会保留实际生成的 Thinking、EOS 和截断输出 token；`generation_completed` 与 `turn_completed` 区分生成完成、长度截断和待处理工具调用。

工具链的自动提交会等待最终回答完成，整条链计为一个逻辑轮。当前回答截断时不自动提交，保留供后续完整轮开启 `enable_memory=True`、`commit_history()` 或 `save_state()` 提交。成功时 `memory_updated=True`，`committed_turns` 包含当前逻辑轮。清除明文上下文是指后续提示不再拼接已提交轮次，不会删除 `responses` 或调用方日志。

下表列出成功执行时的请求顺序。每个 control 动作均为一次 `POST /memt/control`；query 和 commit 使用 `POST /generate`。

| Python 调用 | HTTP 请求顺序 |
| --- | --- |
| `MemTSession(...)` | `GET /memt/capabilities` → `create`，其中 `memory_read: true`。 |
| `set_memory_read(enabled)` | `acquire` → `memory_read` → `release`。 |
| `chat(question)` | `acquire` → query → `release`。 |
| `chat(question, enable_memory=True)` | 回答完成且无待执行工具时：`acquire` → query → `begin` → 尚未保存的历史及当前轮各提交一次 commit → `publish` → `release`；首轮也会开启事务。 |

`chat(..., enable_thinking=False)` 为本轮关闭 Thinking；未覆盖时使用原生 HF 模板默认值。`load_chat_format()` 会显式执行 Flash 可信目录下的 `encoding_dsv4.py`；服务端使用已安装的 tokenizer 类，不执行 checkpoint 中的 Python 模型代码。创建会话前会核对客户端与服务端的 HF 模型/tokenizer 身份。

### 保存与恢复

接着上面的示例：

```python
# 保存局部记忆，包括所有尚未保存的对话轮次。
snapshot = session.save_state("memmy_memory.safetensors")
session.close()

with MemTSession(client, chat_format, parser, state=snapshot["path"]) as restored:
    print(len(restored.messages))  # 0：不会恢复对话记录。
    print(restored.chat("How old is my kitten?"))
```

| Python 调用 | HTTP 请求顺序 |
| --- | --- |
| `commit_history()` | `acquire` → `begin` → 每轮保留的 turn 各提交一次 commit → `publish` → `release`；历史为空时不发请求。 |
| `save_state(path=None, overwrite=False)` | `acquire` → `begin` → 回放待提交 turn → `export` 候选状态 → `publish` → `release`。 |
| `close()` | `acquire` → `close`。 |
| `load_state(path)` | `acquire` → `begin` → `import` → `publish`，其中 `memory_read: true` → `release`。 |

`save_state()` 会提交剩余历史，先导出候选状态，再采用该状态，与 HF 保存语义一致。工具结果尚未补齐时禁止保存。快照位于服务端的 `--state-directory`；`close()` 释放会话，不会自动保存。导入会替换记忆、清空本地消息，并自动开启记忆读取。恢复对话记忆时使用新会话。

显式指定的快照名必须是简单的 `.safetensors` 文件名；默认拒绝覆盖，需通过 `overwrite=True` 显式允许。不传 `path` 时会分配新的 ID，且不会覆盖已有文件。导出／导入要求配置身份和原生编码一致，详见[记忆快照](#记忆快照)。

### 标准对话

新建会话，关闭记忆读取和自动写入。

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
# chat_format 负责输入模板和编码；parser 负责解析 Thinking 和工具调用。
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

### 文档写入与提交边界

使用前面创建的客户端、chat format 和 parser：

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

记忆写入事件使用 checkpoint 的 `memt.chunk_tokens`，两个已发布模型均为 2048，服务端通过 `chunk_tokens` 字段公布该值。它表示内部写入事件的大小，不是单次 commit 请求的最大长度。

每轮完整问答在一次 prefill 中提交，跨内部写入事件时保留相同的读取快照和原生上下文；编码后的长度必须不超过服务端的 `max_prefill_tokens`。`ingest(document)` 先提交尚未写入的完整历史，再在同一事务中将文档按 `chunk_tokens` 分块提交；文档块之间推进读取快照并重置原生上下文。写入成功后清空本地历史。

### 调用方执行工具

创建会话时通过 `tools` 传入 HF 风格的函数 schema。返回 `requires_action=True` 时，`tool_calls` 包含已解析的工具请求；调用方执行函数，并通过 `submit_tool_results()` 一次性提供所有结果。多轮工具续接和实际生成 token 保留在同一条可回放的 turn 中。工具结果未补齐时，禁止开始新对话、提交历史、写入文档或保存。

以下示例假设应用已定义 `tools` schema 和返回文本的 `execute_tool(call)`。使用独立的工具会话，并持续处理，直到没有待执行的工具调用：

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

### 其他客户端方法与交互使用

| API | 用途 |
| --- | --- |
| `MemTSession(client, chat_format, parser, *, session=None, system_prompt=None, tools=None, state=None)` | 创建会话，可指定会话 ID 和 system prompt。 |
| `load_chat_format(path, *, tokenizer, chat_template_kwargs=None)` | 返回 `(chat_format, parser)`，分别负责输入模板／原生编码，以及 reasoning／function-call 解析。 |
| `session.submit_tool_results(results)` | 每个工具调用提供且仅提供一个文本结果，继续同一 turn；工具由调用方执行。 |
| `session.messages` | 未压缩对话的消息副本。 |
| `session.status()` | 查看会话状态、是否 uncertain，以及未提交的问答轮数。 |
| `session.recover()` | 根据事务回执确认 uncertain 操作的结果。 |
| `session.reset()` | 清空会话记忆和本地对话状态。 |
| `NativeClient(base_url, *, api_key=None, timeout=300)` | 底层 HTTP 客户端，提供 `query`、`commit`、`commit_turns` 和 `control`；超时单位为秒。 |
| `MemTClientError` | 包含 `uncertain` 属性，也可能携带 `stage`、`transaction` 和已完成的 `answer`。 |

交互式客户端：

```bash
python -m memt_sglang.client --tokenizer /models/MemT-1.0-Nano --model memt-nano
python -m memt_sglang.client --tokenizer /models/MemT-1.0-Flash \
  --model memt-flash --deepseek-encoder /models/MemT-1.0-Flash/encoding_dsv4.py
```

通过 `--url URL` 连接其他服务；使用 `--enable-memory` 可在每次回答完成后提交尚未保存的历史及当前逻辑轮。CLI 支持 `/read on|off`、`/commit`、`/save [name.safetensors]`、`/load ID`、`/status`、`/reset` 和 `/quit`。服务端要求 Bearer API key 时，设置 `MEMT_API_KEY`。

## HTTP 接口

### 路由

| 路由 | 用途 |
| --- | --- |
| `GET /memt/identity` | 模型（`memt-nano` 或 `memt-flash`）、配置 SHA-256、可选 checkpoint ID、SGLang 版本和生效的 `memt_runtime`。 |
| `GET /memt/capabilities` | 协议 `memt-http-v1`、服务 epoch、快照格式 `memt-memory-v1`、`chunk_tokens`、`max_prefill_tokens`、流式格式和 KV 保留支持。 |
| `POST /memt/control` | 会话、租约、事务和快照。 |
| 带 MemT RID 的 `POST /generate` | 提问，或将 token 提交到会话记忆。 |

不带 MemT RID 的普通 SGLang 请求不会访问会话记忆。在线权重更新、缓存管理及 LoRA 加载／卸载路由被禁用，返回 409。

### 会话、租约与事务

- **会话（session）**：一份独立的记忆状态。会话 ID 和 RID nonce 使用 `[A-Za-z0-9_.-]{1,160}`。HTTP `create` 和 Python `MemTSession` 均默认使用 `memory_read: true`。
- **租约（lease）**：会话的独占访问凭证。提问和修改状态的操作需要有效租约。HTTP `ttl` 范围为 1–3600 秒，默认 300 秒；Python 客户端为每次操作申请 3600 秒租约。
- **事务（transaction）**：调用方命名的一组记忆写入。`publish` 后写入才可见，`abort` 会整体放弃。
- **Epoch**：每次服务启动生成的新身份。control 请求使用 `epoch`，生成请求使用 `memt_epoch`，以便在服务重启后拒绝旧会话。Python 客户端会自动携带它。
- **RID**：格式为 `memt|<session>|<operation>|<nonce>`，operation 为 `query` 或 `commit:<token_count>`。每个请求使用唯一 nonce。所有以 `memt` 开头的 RID 都保留给 MemT 校验，包括格式错误的 RID。

### Control 动作

请求格式为 `{"action": ..., "session": ..., "lease": ..., "epoch": ..., ...}`。`create`、`acquire` 和 `inspect` 不需要目标会话的租约，其他动作需要。复制来源会话时必须持有来源会话的租约。

| 动作 | 额外字段 | 用途 |
| --- | --- | --- |
| `create` | 可选 `memory_read`、`source` + `source_lease`、`retain_kv` | 创建记忆，可从来源会话复制；`retain_kv` 要求启用 `--enable-streaming-session`。 |
| `acquire` / `renew` | 可选 `ttl` | 获取或续期租约。 |
| `release` | — | 确认未完成操作的结果并释放租约。 |
| `begin` / `abort` | `transaction` | 开启或放弃事务。 |
| `publish` | `transaction`；可选 `memory_read` | 发布写入，并可同时设置读取策略。 |
| `memory_read` | `enabled` | 在会话空闲时改变读取策略。 |
| `export` | `encoding`；可选 `file`、`transaction`、`overwrite` | 导出已发布的记忆或事务候选状态；覆盖需指定 `overwrite: true`。 |
| `import` | `transaction`、`file`、`encoding` | 将快照加载到事务中，`publish` 后生效。 |
| `inspect` / `status` | — | 查看会话状态／事务状态和提交回执。 |
| `recover` | 可选 `ttl` | 确认 uncertain 操作的结果，并续期租约。 |
| `cancel` | `request_id` | 按 RID 取消提问。 |
| `kv_reset` | — | 清空保留的原生上下文，同时保留记忆。 |
| `reset` / `close` | — | 清空记忆／删除会话。 |

### 生成请求

在服务 epoch `E` 中取得租约 `L`，并开启事务 `t1` 后，提交一次 commit 请求：

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

发布事务后提问：

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

- `input_ids` 必须是单个非空 token ID 列表，长度不超过 `max_prefill_tokens`。不支持批量 MemT 请求、`text`、输入 embedding 或多模态输入。
- `commit:<token_count>` 中的计数必须等于 `input_ids` 长度。`max_new_tokens` 必须为 0，并携带 `memt_transaction`。一次 commit 可以跨多个 `chunk_tokens` 写入事件，不限于 2048 token。
- `sampling_params.max_new_tokens` 必须显式设置为非负整数，`sampling_params.n` 必须为 1，`ignore_eos` 必须保持 false。Commit 请求不返回输出分数或隐藏状态。
- 只有 query 支持 `stream: true`。SSE 每帧包含累计全文；成功结束时提供 `finish_reason`，并以 `data: [DONE]` 结束。
- 保留 KV 的 query 必须在已确认的完整 token 前缀上追加内容。服务端管理原生会话；只有启用 KV 保留的 query 可以传入 `session_params`，且其值只能是 `{"id": "<session>"}`。

### 错误与恢复

MemT 中间件错误格式为 `{"error": {"type": "invalid_memory_request", "message": "..."}}`。

| HTTP 状态码 | 含义 |
| --- | --- |
| 400 | 请求参数错误。 |
| 409 | 会话状态冲突，例如租约无效、会话忙或 epoch 不匹配。 |
| 429 | 会话数或并发提交数达到上限。 |
| 502 | 原生生成返回了不完整或无效的成功响应。 |
| 503 | Worker 通信失败或确认超时；相关会话可能进入 uncertain 状态。 |

`MemTClientError.uncertain=True` 时，应先检查状态并恢复，再继续写入，不要盲目重放写请求。如果客户端无法确认事务回执，应新建会话并加载已确认的快照。服务重启会改变 epoch，需要新建会话并显式导入快照。

## Checkpoint 与快照格式

### 模型配置

Checkpoint 的 `config.json` 用以下字段标识模型：

| 模型 | `model_type` | `architectures` | 配置类 |
| --- | --- | --- | --- |
| MemT-1.0-Nano | `memt_qwen3_5_moe` | `["MemTQwen3_5MoeForCausalLM"]` | `MemTQwen3_5MoeConfig` |
| MemT-1.0-Flash | `memt_deepseek_v4` | `["MemTDeepseekV4ForCausalLM"]` | `MemTDeepseekV4Config` |

Nano 将骨干配置放在 `backbone_configs` 中，使用 `model_type: "qwen3_5_moe"`，文本配置位于其嵌套的 `text_config`。Flash 使用扁平 HF 配置，不能包含 `backbone_configs`。两个模型的记忆参数均放在 `memt` 中。尺寸类参数直接从 `config.json` 读取；校验只固定选择记忆算法的字段和未量化 BF16 推理路径，加载时还会核对记忆维度与原生 attention 层是否一致。`memt_checkpoint_id` 为可选字段；Nano 在提供该字段时要求它是非空字符串。服务身份始终包含完整 `config.json` 的 SHA-256。

顶层可选的 `memt_runtime` 只包含 `max_sessions_per_rank`、`max_inflight_commits_per_rank` 和 `min_kv_tokens`，均为正整数。Launcher 按默认值、checkpoint 配置、JSON runtime 覆盖、显式容量参数的顺序合并配置。未覆盖时每个字段默认为 1；启动脚本使用前文表格中的部署值。

权重必须是未量化的 BF16 safetensors。服务端使用已安装的配置类和模型类，不执行 checkpoint 中的 Python 模型代码。

### 记忆快照

快照是位于服务端 `--state-directory` 下的 `.safetensors` 文件。每个记忆层对应一个名为 `memory.<layer>` 的 FP32 张量，所有数值必须有限，形状必须与模型要求完全一致。

元数据包含 `format=memt-memory-v1` 和 `identity=<HF Session identity>`；身份覆盖骨干配置、记忆配置、原生编码、聊天模板和 tokenizer 词表。已初始化文件使用 `memory.<layer>` 键、FP32 类型和 `[1, heads, K, V]` 形状；空 HF 记忆不包含张量。导入前检查身份、层集合、类型和形状。将 HF 文件传入服务端 `--state-directory` 后可按文件名加载；服务端导出文件也可由相同模型/tokenizer 的 HF Session 加载。旧 SGLang 的 `memory.layers.<layer>` 及 `model`/`encoding` 元数据格式与此格式不兼容。快照只包含已发布记忆，不包含对话记录或原生 KV cache。

## 数值验证

记忆解码图改变的是记忆读取 kernel **何时发射**，不是**算什么**。这一点是测出来的，不是声称的。

### 设置

两臂同树、同冻结记忆快照、同 prompt、greedy 解码。唯一差别是 `MEMT_EXPERIMENT_MEMORY_DECODE_GRAPH`。上文列出的五个回放一致性开关**两臂都设**，所以对比隔离的是图本身，没有混入稳定 top-k 的开销。

两臂确实不同——这是从服务端日志核实的，不是假设：

| | 图关 | 图开 |
| --- | --- | --- |
| `Capture target decode CUDA graph` 行数 | 0 | 8 |
| `graph segments=[...]` | 无 | 6 次捕获，**全部 `[1]`**（单段） |

### 前 128 个 decode step 的 next-token logits

logits 以 float32 存储（用 bf16 存会把要测的差异本身量化掉），在完整的 129,280 词表上比较。

| 指标 | bs=1（118 步） | bs=32（4 个 DP rank × 128 步） |
| --- | --- | --- |
| L2 相对误差 `‖on−off‖₂/‖off‖₂` | **0.000e+00**（最大 0） | 每个 rank **0.000e+00** |
| Pearson 相关系数 | **1.000000**（最小 1.0） | 每个 rank **1.000000** |
| top-1 一致 | **118/118** | 每 rank **128/128** |

这里的 `0` 是**逐位相同**，不是"误差很小"。这是应有结果：同权重、同输入、同确定性路径——图只改变发射调度。

边界写明：本项覆盖 Flash 的 bs=1 与 bs=32 两个 bucket，只覆盖 decode。其余 bucket 是各自独立的图，未逐一对比（其捕获日志同样是 `segments=[1]`）。Nano 的图是另一套实现，未测。

### LocoMo 端到端

原评测链驱动 vLLM；本次只换引擎，交给官方离线打分器的产物格式不变，因此打分器原样运行。

| | 图关 | 图开 |
| --- | --- | --- |
| 实例数 | 1540 / 1540 | 1540 / 1540 |
| **聚合类别分** | **2.9714** | **2.9714** |
| 聚合 F1 | 0.26042 | 0.25987 |
| 输出质量门 | passed | passed |

类别分逐位相同。F1 差 5.5e-4（0.21%）——这是"每答案最多 512 token"的生成文本差异，不是短程数值分叉，因为上面 128 步 logits 是逐位相同的。两臂都恰好有 1/1540 道题触到长度上限，说明都没有被截断主导。

**跑这套评测时发现的上游缺陷。** 开 `--enable-deterministic-inference` 时，SGLang 按**整个 step** 决定采样路径（`is_all_greedy = all(top_k <= 1 for r in reqs)`）。commit 请求不生成 token，但仍携带采样参数；若它保留服务端默认值，任何"commit + query"混合的 step 都会变成非 greedy，而非 greedy 的 gumbel 路径在 `murmur_hash32` 里断言 `seed.shape == positions.shape`，直接打死调度器。本包的规避方式：commit 也使用 greedy 参数。这是上游行为，不是这些补丁引入的，值得反馈给上游。


## 许可证

本包原创代码采用 [PolyForm Noncommercial License 1.0.0](LICENSE)。随包提供的第三方代码保留原有许可证和版权声明：

| 组件 | 许可证 |
| --- | --- |
| MemT SGLang 集成、客户端以及 Nano / Flash 记忆实现 | [PolyForm Noncommercial 1.0.0](LICENSE) |
| 基于 flash-linear-attention 修改的 Flash 记忆 kernel（`memt_sglang/models/deepseek/ops/_triton/intra.py`、`wy.py`） | FLA 原始代码：[MIT](memt_sglang/_vendor/fla/FLA_LICENSE)；MemT 的修改部分：[PolyForm Noncommercial 1.0.0](LICENSE)。 |
| 随包提供的 flash-linear-attention 0.5.2 源码子集（`memt_sglang/_vendor/fla`） | [MIT](memt_sglang/_vendor/fla/FLA_LICENSE)，版权归 2023–2026 Songlin Yang、Yu Zhang 和 Zhiyuan Li 所有。 |

SGLang、PyTorch、Transformers、FlashInfer 和 sglang-kernel 为安装依赖，遵循各自的许可证。模型权重、tokenizer 文件和 chat template 单独分发，遵循相应发布物提供的条款。仓库的论文许可证不适用于本软件包。
