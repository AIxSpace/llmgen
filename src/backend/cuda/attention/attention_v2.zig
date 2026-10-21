/// decode split-K 的分段长度: 批越小分段越细, 保证 grid = nkv × S × parts 足够铺满 82 个 SM.
pub fn partSize(s: usize) usize {
    return if (s >= 64) 512 else if (s >= 16) 256 else if (s >= 4) 128 else 64;
}
pub fn maxParts(s: usize, max_model_len: usize) usize {
    const ps = partSize(s);
    return (max_model_len + ps - 1) / ps;
}