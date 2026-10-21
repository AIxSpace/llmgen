#define PAGE_BLOCK_SIZE 16 // 对照 paged.zig 的 PAGE_BLOCK_SIZE

// 页表上传
static void upload_page_table(const int32_t* h_page_k_ids, const int32_t* h_page_v_ids, int32_t num_blocks, int32_t** d_page_k_ids, int32_t** d_page_v_ids) {
    cudaMalloc(d_page_k_ids, num_blocks * sizeof(int32_t));
    cudaMalloc(d_page_v_ids, num_blocks * sizeof(int32_t));
    cudaMemcpy(*d_page_k_ids, h_page_k_ids, num_blocks * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_page_v_ids, h_page_v_ids, num_blocks * sizeof(int32_t), cudaMemcpyHostToDevice);
}