# 模型构造

- 工厂加载 - 初始化的时候，默认注册模型映射创建关系
    - init -> registerModel
- 实例化各自的模型类 - createFromArch ｜ createFromGGUF
- 动态映射 - 使用 vtable 来实现 interface 接口定义
- 搭建模型结构 - 参考 Genllm 模型结构文档
    - 声明输出 Tensor
    - rope - cache
    - embedding
    - deocde layer
    - qwen3 - attention
        - rms norm
        - qkv
        - reshape + permute
        - rope
        - attention
        - permute + reshape
    - qwen3 mlp

## 模型结构
    