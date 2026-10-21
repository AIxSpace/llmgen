# 推流主流程

- DeviceManager - 读取设备信息
- GGUFParser - 解析 gguf 格式的模型
- ModelFactory - 模型映射到 Qwen3
- Qwen3Model.buildComputeGraph - 构建计算图
- Scheduler - 资源调度 - 内存管理 MemoryManager
- MemoryManager.loadWeights - 加载模型权重
- Executor.init
- Tokenizer.encode - 编码 prompt
- generate
    - prefill
    - decode
- Batching