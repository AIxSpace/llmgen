# BPE 分词器

- 本质上是一种压缩算法
- 字符两两配对
- 找到出现频率最高的组成新的 token
- 为每种 token 分配一个编号
- 最终组成词表
- encode - 从 text -> token -> token ids
- decode -> token ids 反查出 text