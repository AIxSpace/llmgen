# llmgen

用 Zig 从零实现的大模型推理引擎，直接加载 GGUF（Qwen3-0.6B BF16），CPU / CUDA 双后端，带连续批处理、分页 KV Cache 和 CUDA Graph。

## 特性

- **GGUF 解析**：只读头部，权重按需从磁盘经 pinned 暂存区拷到显存
- **模型 → 计算图**：`ModelFactory` 把 Qwen3（RMSNorm / GQA / RoPE / SwiGLU）构建成计算图，图调度器按显存预算把层分配到设备
- **双后端**：CPU 算子（`src/backend/cpu`）和 CUDA 算子（`src/backend/cuda`：bf16x8 向量化、GEMV、FlashAttention-2 prefill、flash-decoding、GPU 采样）
- **服务引擎**：continuous batching，支持 `prefill_first` / `chunked` 两种调度策略，按块分配 KV，KV 不够时抢占
- **CUDA Graph**：按批大小分桶录图，decode 阶段回放
- **性能观测**：TTFT / TPOT / ITL、按 kernel 统计的 profiler、带宽和算力利用率

## 构建与运行

依赖 Zig 0.15.2（最低 0.15.1）；GPU 版本另外需要 CUDA Toolkit，默认 `-Dcuda-arch=sm_86`（RTX 3090）。

```bash
export LLMGEN_MODEL=/path/to/Qwen3-0.6B-BF16.gguf   # 或者用 --model 指定

make run      ARGS="chat 你好 64"        # CPU
make run-cuda ARGS="chat 你好 64"        # GPU
make run-cuda ARGS="chat -i"             # 交互式多轮对话
make test / make test-cuda               # 单元测试
```

不传 prompt 时用对拍基线的 prompt；greedy 输出的最后一行是 `token ids`，可以和基线 diff，检查优化前后结果是否一致：

```bash
make run-cuda 2>&1 | grep '^token ids' > ~/llmgen-golden.txt
make run-cuda 2>&1 | grep '^token ids' | diff - ~/llmgen-golden.txt && echo 一致
```

常用选项（完整列表见 `src/cli.zig`）：`--temp`、`--policy prefill_first|chunked`、`--chunk`、`--max-num-seqs`、`--max-model-len`、`--gpu-util`、`--kv-blocks`、`--eager`（不使用 CUDA Graph）、`--sync-each`（每个 kernel 后同步，用来定位出错的 kernel）、`--single`（单请求路径，配合 `LLMGEN_METRICS=1` / `LLMGEN_PROFILE=1` 打点）。

> GPU 上有其他进程占显存时，按 `--gpu-util` 算出的 KV 可能不够放一条序列（报 `KvCacheSmallerThanOneSequence`），这时用 `--kv-blocks 512` 固定 KV 块数。

## 压测

```bash
make workloads                                        # 生成 bench/workloads/*.json
make run-cuda ARGS="bench bench/workloads/s1.json"    # 输出和 nano-vllm 同口径的 JSON
make bench                                            # bench/run.sh：跑 5 次，取中位数
```

## 目录

```
src/
├── llmgen.zig cli.zig        入口、命令行
├── gguf.zig tokenizer.zig    GGUF 解析、BPE 分词
├── models/ module/           Qwen3 模型定义、层模块（linear / norm / rope / attention ...）
├── graph/                    计算图、图调度（设备分配、显存预算）
├── memory/                   内存规划、池化、分页 KV
├── executor.zig              前向执行、CUDA Graph 录制和回放
├── runner/                   服务引擎、批调度、块管理
├── backend/{cpu,cuda}/       算子实现
├── profiler/ bench/          性能统计、压测记录
bench/   压测脚本和历史结果      docs/   设计文档和教程      testdata/   算子对拍数据（来自 PyTorch）
```

## 文档

- [推理主流程](docs/flow.md) · [GGUF](docs/gguf.md) · [模型映射](docs/model.md) · [计算图](docs/graph.md) · [图调度](docs/scheduler.md) · [执行器](docs/executor.md)
- [分词器](docs/tokenizer.md) · [内存与 KV Cache](docs/memory.md) · [设备派发](docs/device.md) · [算子](docs/ops.md) · [连续批处理](docs/batching.html)
- [性能指标实战](docs/perf-metrics.html) · [性能优化教程：一步步优化到 serving](docs/llmgen-serving-unfused/index.html)
