# Cruise

**简体中文** | [English](README_EN.md)

**通过设备驻留的解码 Epoch，消除逐 Token 的 Host 往返。**

Cruise 是面向昇腾 NPU 的研究原型。它利用 DataFlow Device UDF，将 LLM
解码过程中对延迟最敏感的内层控制循环下沉到设备侧。Host 仍然负责请求准入、
全局调度、公平性与故障恢复；固定批次完成准入后，设备可以连续执行一个有界的
decoder epoch，在设备侧更新 Paged-KV 状态、执行 greedy sampling，并在遇到 EOS
或达到 epoch 上限后再返回 Host。

Cruise 探索的是单次图重放之上的控制边界：图执行优化消除了单次模型执行内部的
逐算子下发开销，而 Cruise 进一步消除有界解码迭代之间反复发生的 Host 协调。

> Cruise 目前是实验性系统研究原型，并非生产级推理服务。现阶段支持范围经过
> 明确约束，不应将实验结果外推到未验证的工作负载。

## 系统架构

![Cruise 系统架构](docs/images/cruise-architecture-zh.svg)

当前主线实现来自 **Attempt 74**，已经具备以下能力：

- 面向固定形状 resident epoch 的 vLLM V1 scheduler contract；
- 由 DataFlow 持有模型与 KV 状态的专用 worker；
- EngineCore 到 native sidecar 的执行路径，每个 epoch 只进行一次请求与响应；
- B=4 的完整 Qwen2.5-7B decoder step，以及设备侧 greedy sampling；
- 持久化 Paged-KV、active-row mask、block table、slot mapping、row generation，
  以及跨 epoch 的安全行复用；
- 根据请求剩余预算，在 K=1、2、4、8 中选择有界 epoch 长度；
- 设备侧 EOS/最大步数终止，并将逐请求执行账目返回 vLLM；
- 执行前合法性校验与输入状态保持不变的 fallback 状态码；
- 最小化的 8 输入/2 输出 Host-UDF ABI；早期版本为 10 输入/10 输出，内部
  decoder ABI 仍保持 9 输入/4 输出不变；
- 对日志大小、marker-protected scratch、根盘剩余空间和清理流程的存储保护。

已经通过的多 epoch cohort 为 `A -> [A,B] -> [A,C]`：请求 A 始终保留原有 row
和 KV 状态；B 使用第二个 row；B 离开后，C 通过新的 generation 安全复用该 row。
Attempt 74 的严格实验门槛参见 [PROTOCOL.md](PROTOCOL.md)。

## 当前证据

Attempt 74 的 CANN 8.5.1 正式实验采用 `old -> new -> new -> old`，包含 30 个 old
和 30 个 new B=4/K=2 epoch。60 个样本均通过 token、请求状态、调用次数和时间窗
校验。实际 DataFlow API 边界 payload 如下：

| ABI | Feed/epoch | Fetch/epoch | 总计/epoch |
|---|---:|---:|---:|
| old 10/10 | 58,720,516 B | 78,184,928 B | 136,905,444 B |
| new 8/2 | 260 B | 368 B | 628 B |

每个 epoch 的 payload 减少 136,904,816 B。这里的数值来自实际
`FeedDataFlowGraph`/`FetchDataFlowGraph` tensor 的 `Tensor::GetSize()`，与声明
ledger 精确一致，并非由声明常量替代实测值。

在同一批 60 个稳态窗口中，扩展 tracer 覆盖的 `rtMemcpy`/`rtsMemcpy` API 调用数
均为 0；每个独立进程的 1,745 条 runtime memcpy 和 23 条 Mbuf 记录全部发生在
测量窗口外的启动期。CANN 8.5.1 的 application `msprof` 无法初始化该 resident
sidecar，因此上述 DataFlow payload **不是** PCIe、HCCS 或 DMA 物理链路字节，
本项目不作物理链路流量下降的声明。

30+30 样本的 Host 控制 wall time 中位数从 212.208 ms 降至 59.951 ms，对应
3.54x；Python CPU time 从 2.368 ms 降至 1.045 ms，对应 2.27x。完整结果、独立
验证范围与证据哈希见
[`evidence/ATTEMPT74-CANN851-R5.md`](evidence/ATTEMPT74-CANN851-R5.md)。较早的
G4 K-sweep 结果仍保留在
[`history/attempts/g4/G4-STATUS-20260724.md`](history/attempts/g4/G4-STATUS-20260724.md)。

## 支持边界

当前经过验证的环境与工作负载为：

- Ascend 910B2，以及经过验证的 CANN 8.5.1/9.0.0 DataFlow Device UDF 路径；
- Qwen2.5-7B-Instruct，TP=1、PP=1；
- vLLM V1 同步调度；
- one-token prompt 后进入 decode；
- 一个静态 B=4 图，通过 inactive-row mask 支持不足四路的批次；
- greedy sampling、有界 epoch，以及每个 row 固定两个 block 的 KV 布局。

Cruise 尚未证明通用 prefill、continuous batching、任意 sampling、speculative
decoding、preemption、cancellation、LoRA、TP/PP、多卡协调或 API server 性能。
这些是后续研究门槛，而不是可以忽略的兼容性假设。

## 仓库结构

| 路径 | 用途 |
|---|---|
| `src/vllm_ascend_resident_epoch/` | vLLM scheduler、worker、contract 与 backend 接入 |
| `controller/` | 当前 Device UDF 控制器 |
| `controller-old/` | 为受控 ABI 对照保留的旧版控制器 |
| `native/` | sidecar、bridge、AIR relocation 与 DataFlow/runtime tracing |
| `config/` | DataFlow 与 graph 配置模板 |
| `tests/` | 源码约束与集成单元测试 |
| `storage_guard/` | 根盘空间、scratch、日志与清理保护 |
| `experiments/synthetic-p0/` | 最初的 synthetic feasibility 实验 |
| `history/attempts/` | Attempt 41-73 以及 G4 阶段的纯源码快照 |
| `docs/` | 项目演进与仓库存储策略 |
| `scripts/audit_repository.py` | 大文件、生成物、主机名与敏感信息审计 |

## 本地检查

以下轻量 contract 与结果验证测试不依赖 NPU、PyTorch 或 vLLM：

```bash
python -m pip install pytest
python -m pytest -q \
  tests/test_abi_measurement.py \
  tests/test_contract.py \
  tests/test_engine_core_result_verifier.py \
  tests/test_multi_epoch_result_verifier.py
python scripts/audit_repository.py
python verify_minimal_abi_source.py . \
  --baseline-source history/attempts/vllm-integration-attempt73-multi-epoch-cohort
```

该子集目前包含 36 项测试。完整的 52 项测试还需要冻结版本的 PyTorch、vLLM 和
vLLM-Ascend 环境；native 执行还需要实验协议指定的 Ascend/DataFlow 工具链、
decoder AIR 与外部权重。模型生成物和原始测量数据不会存入本仓库。

## 复现硬件实验

`run_attempt74.sh` 是版本化的实验驱动脚本。它从干净 Git checkout 启动，通过
`CRUISE_*` 环境变量接收机器相关的外部资产路径，将生成物暂存到带 marker 的
`/dev/shm` scratch，在加载模型前检查 NPU 与存储 readiness，并且只在根盘保留
带 SHA256 manifest 的精简证据。CANN 8.5.1 正式运行后 scratch 已自动清理。

## 开发者预览运维入口

产品化路线的 M0 首批实现已引入机器可读兼容矩阵、严格 JSON 运行配置和统一 CLI：

```bash
cruise smoke
cruise doctor --mode npu \
  --profile attempt74-910b2-cann851-r5 --device 7
cruise doctor --mode runtime --config /etc/cruise/cruise.json
```

安装、配置、生命周期和故障边界分别见
[INSTALLATION.md](docs/INSTALLATION.md)、
[CONFIGURATION.md](docs/CONFIGURATION.md)、
[OPERATIONS.md](docs/OPERATIONS.md) 与
[COMPATIBILITY.md](docs/COMPATIBILITY.md)。这些入口目前用于 Developer Preview
和 EngineCore 资格验证；通用 API server 仍属于 M1，尚未被宣称为可用能力。

## 项目演进

源码档案记录了项目从 synthetic recurrence、完整 decoder 到 vLLM 集成的完整
演进过程，简要阶段划分参见
[docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md)。历史目录只用于保存研究来源，
不作为当前维护的 release branch。

暂定论文题目：

> **Cruise: Eliminating Per-Token Host Round Trips with Device-Resident Decode Epochs**
