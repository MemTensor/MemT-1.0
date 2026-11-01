# PR7 交付整理与后续优化工作记录

## 目标与基线

工作基线为 PR7 `7a0566b2cad855165329d4f5efac16e4036185b0`，PR4 父提交为
`61f56f53fdcce80a0d27117d3e624ca767723663`。原分支及历史失败、性能数据均保留。
本分支默认采用 fast 计算策略，确定性为显式选择；图的内存生命周期保证独立于二者。
工程完成和团队业务质量验收分别记录。

## 当前状态

- 已复现旧代码 fast 与 Flash 图校验互相冲突，已解除确定性 profile 的图前置要求。
- 图请求默认解析 slot reader、padded geometry 和参考 stable C4；显式冲突仍报错。
- 默认不限制 GPU compute capability；保留内核能力检查以及用户主动配置的 allowlist。
- 实际 pinned runner 已按完整 bucket 最大值分配输入，未发现降序假设导致的线上失败；
  adapter 改为明确使用 runner.max_num_token，补任意捕获顺序回归。
- 旧 README 的类别编号“质量指标”和未证实的 owner 因果解释需要删除。
- 完整代码 GPU 验收、拆分、依赖复现及时间线实验尚未完成。

## 工作顺序与验收

1. 配置和生命周期修复：CPU 回归覆盖 fast/strict、缺失实际 owner、rank、slot 回收、
   非空状态、捕获顺序；保留真实不支持布局的错误。
2. 从 pinned SGLang 源码生成可应用补丁；在独立目录验证，不依赖实验机绝对路径。
3. 将 Flash prefill 实验和 global/tail 调度保存在独立分支/补丁中，整理 PR7 核心交付。
   attention TP/head-local read 另行适配，不能直接删除 DP 实现限制。
4. 英中文档、启动配置、CI；最终代码 lint/format/CPU、两模型 GPU smoke 和 Flash
   非空 S 的 graph/eager 数值比较。已有快档验收仅适用于原独立修复版本。
5. 独占复测关键性能点，确认整理未回退，再开始时间线研究。
6. 每项性能修改单独版本：瓶颈证据→修改→数值回归→无 profiler 独占整体复测。
7. 报告、可复现补丁、源代码和原始证据哈希归档，最终清理任务 GPU 进程。

## 时间线实验设计

Nano/Flash 分别测生成、长摄入、Agent。生成使用实际提交产生的非空 S 和每请求
不同的 prompt，至少覆盖 C1/C32；长摄入含混合长度和并发查询；Agent 含真实
query→工具等待→commit/publish→query，分开记录工具等待及 GPU 服务时间。

固定输入、权重、状态、并行布局和容量，记录实际 native/memory 图命中与回退。
先跑无 profiler 基线，再用相同配置独占采样。Nsight Systems 观察 CPU/CUDA/NCCL
和跨 stream 依赖；必要时用 PyTorch profiler 辅助归属，用 Nsight Compute 做少量
代表 kernel 的带宽/计算采样。诊断数据不作为正式性能数字。

分类：骨干非专家计算、专家计算、MemT 读取、MemT 写入、状态拷贝/转换、通信、
CPU 调度。报告 GPU 活跃区间、关键路径以及调度空隙；重叠区间不重复相加，
NCCL kernel 时长不直接等于可消除等待。公布分类规则、未归属时间和 profiler 扰动。

优先研究：投影/归一化融合、FP32→BF16 状态转换及 gather 复制、rank 同步和安全
调度重叠。只有整体 TPOT、TTFT 或吞吐改善且数值回归通过，才计入优化收益。
剩余架构成本按实测字节量、kernel 数、依赖链和资源利用率分析；不从单点平均
带宽利用率推导“吞吐已经到头”。

## 已建立的代码检查点

`1c19620` 保存配置/生命周期修复、依赖补丁、文档及规范整理的完整版本。
`archive/pr7-complete-experiments-20261102` 保留 Flash prefill 实验实现和其测试。
正式分支随后移除该实验的 AST 改写、固定 DP4/输入长度门禁和诊断注入入口；
保留 eager Flash prefill 与 breakable decode 的组合。移出的测试随实验保留，
不是删除失败测试以取得通过。Core CPU 当前 282 passed，8 subtests passed。
七个 SGLang 依赖文件从公开基础提交应用补丁后，逐文件哈希与之前独立验收的
固定 overlay 全部一致。正式 adapter 的 GPU 回归仍待执行。
