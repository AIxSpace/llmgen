# Tensor

## 字段描述

### 基本信息

- name - 名称
- dims - 维度
- strides - 跨度

### 计算参数

- src - 参与计算的原始参数
- op_type - 算子计算类型
- op_params = 参与计算的附属参数 (一般是常量配置进来的)
    
### 数据相关

- data - 内存或显存数据
- data_type - 数据类型
- offset - 内存位置便宜
    - gguf 中的数据起始偏移位置
    - memory block 中的 offset 对应

### 模型相关

- layer_id - 大模型层编号
- device_type - 分配的设备类型