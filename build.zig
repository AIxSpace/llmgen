const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "llmgen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const have_cuda_option = b.option(bool, "cuda", "Enable real CUDA backend (Linux+NVCC+GPU)") orelse false;
    const nvcc_path_option = b.option([]const u8, "nvcc", "Path to nvcc") orelse "nvcc";
    const cuda_home_option = b.option([]const u8, "cuda-home", "CUDA toolkit home") orelse "/usr/local/cuda";
    const cuda_arch_option = b.option([]const u8, "cuda-arch", "nvcc -arch") orelse "sm_86";

    const build_options = b.addOptions();
    build_options.addOption(bool, "have_cuda", have_cuda_option);
    exe.root_module.addOptions("build_options", build_options);
    if (have_cuda_option) {
        cudalize(b, exe.root_module, .{
            .nvcc = nvcc_path_option,
            .home = cuda_home_option,
            .arch = cuda_arch_option,
        });
    }

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    // zig build run -- chat 你好 64
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_exe_tests.step);

    // 供 LLDB 调试用：编译测试二进制并安装到 zig-out/bin/debug-unit-tests（不执行）
    const test_filter = b.option([]const u8, "test-filter", "只编译名字包含该字符串的 test");
    const filters: []const []const u8 = if (test_filter) |f|
        b.allocator.dupe([]const u8, &.{f}) catch @panic("OOM")
    else
        &.{};

    const debug_tests = b.addTest(.{
        .name = "debug-unit-tests",
        .root_module = exe.root_module,
        .filters = filters,
    });
    const install_debug_tests = b.addInstallArtifact(debug_tests, .{});
    const debug_test_step = b.step("debug-test", "Build test binary for debugging");
    debug_test_step.dependOn(&install_debug_tests.step);
}

const CudaConfig = struct {
    nvcc: []const u8,
    home: []const u8,
    arch: []const u8,
};

const CUDA_SRC = [_][]const u8{
    "cuda_context.cu",
    "embedding/embedding.cu",
    "memory/memcpy.cu",
    "memory/memory.cu",
    "linear/linear.cu",
    "norm/norm_final.cu",
    "arithmetic/add_v1.cu",
    "arithmetic/mul_v1.cu",
    "activation/silu_v1.cu",
    "softmax/softmax.cu",
    "shape/shape.cu",
    "rope/rope_v1.cu",
    "attention/attention.cu",
    "sample/sample.cu",
};

// use nvcc compile .o file
fn cudalize(b: *std.Build, mod: *std.Build.Module, cuda_config: CudaConfig) void {
    for (CUDA_SRC) |src| {
        const nvcc_build_cmd = b.addSystemCommand(&.{ cuda_config.nvcc, "-c", "-O3", "-std=c++17" });
        nvcc_build_cmd.addArg(b.fmt("-arch={s}", .{cuda_config.arch}));
        nvcc_build_cmd.addArgs(&.{ "-Xcompiler", "-fPIC" });
        nvcc_build_cmd.addArg(b.fmt("-I{s}/include", .{cuda_config.home}));
        nvcc_build_cmd.addFileArg(b.path(b.fmt("src/backend/cuda/{s}", .{src})));
        nvcc_build_cmd.addArg("-o");
        mod.addObjectFile(nvcc_build_cmd.addOutputFileArg(b.fmt("{s}.o", .{src})));
    }
    mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{cuda_config.home}) });
    for ([_][]const u8{ "cudart", "cublas", "stdc++", "m", "pthread", "dl", "rt" }) |lib| {
        mod.linkSystemLibrary(lib, .{});
    }
    mod.link_libc = true;
}
