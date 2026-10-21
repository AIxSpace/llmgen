pub extern fn cuda_malloc(bytes: usize) ?*anyopaque;
pub extern fn cuda_free(device_data_ptr: ?*anyopaque) void;

pub extern fn cuda_host_alloc(bytes: usize) ?*anyopaque;
pub extern fn cuda_host_free(host: ?*anyopaque) void;

pub extern fn cuda_h2d(dst_dev: ?*anyopaque, src_host: ?*const anyopaque, bytes: usize) void;
pub extern fn cuda_d2h(dst_host: ?*anyopaque, src_dev: ?*const anyopaque, bytes: usize) void;
pub extern fn cuda_d2d(dst_dev: ?*anyopaque, src_dev: ?*const anyopaque, bytes: usize) void;
pub extern fn cuda_h2d_async(dst_dev: ?*anyopaque, src_host: ?*const anyopaque, bytes: usize) void;
pub extern fn cuda_d2h_async(dst_host: ?*anyopaque, src_dev: ?*const anyopaque, bytes: usize) void;
