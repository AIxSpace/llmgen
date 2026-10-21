# 内存显存管理

## MemoryPool - 内存池

## DevicePools

- 权重池
- 激活池
- kv cache 池

## MemoryManager - 管理所有内存

- devices_map - 每个 device_id 对应的内存池 DevicePools
- layer_device_map - 每层用到的设备

## PagedAttentionManager - 注意力分页内存管理

- pool: MemoryPool - 对应的设备分配好的内存池
- layers - 按层划分的内存池
    - k_pool - BlockPool
    - v_pool - BlockPool
    - page_table