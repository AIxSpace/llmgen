# 计算图构建

- 读取 gguf_info 信息到 config
- 搭建模型结构 - 输出 out_tensor
- reverseBfsCollect - 反向广度优先遍历，找出所有的 tensor 节点
    - visited 数组用来记录是否访问过
    - queue 用来记录当前访问节点，方便查找在一个节点
    - all_tensor 记录所有节点
- toplogic Sort - 拓扑排序，把收集到的节点按执行顺序排列，并分层 (degree)
    - 先统计 parent children degree 关系
    - 入读为 0 作为第一层节点
    - 确认计算顺序
- 计算图导出对比模型结构