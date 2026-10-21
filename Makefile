LLMGEN_MODEL ?= /home/deploy/qwen/qwen306b/models/Qwen3-0.6B-BF16.gguf
CUDA_HOME ?= /usr/local/cuda
CUDA_FLAGS = -Dcuda=true -Dnvcc=$(CUDA_HOME)/bin/nvcc -Dcuda-home=$(CUDA_HOME)
export LLMGEN_MODEL

.PHONY: build build-cuda run run-cuda test test-cuda bench workloads compare plot clean

build:
	zig build -Doptimize=ReleaseFast
build-cuda:
	zig build $(CUDA_FLAGS) -Doptimize=ReleaseFast

# CPU 推理:make run ARGS="chat 你好 64"
run:
	zig build run -Doptimize=ReleaseFast -- $(ARGS)
# GPU 推理:make run-cuda ARGS="chat 你好 64"
run-cuda:
	LD_LIBRARY_PATH=$(CUDA_HOME)/lib64 zig build run $(CUDA_FLAGS) -Doptimize=ReleaseFast -- $(ARGS)

test:
	zig build test
test-cuda:
	LD_LIBRARY_PATH=$(CUDA_HOME)/lib64 zig build test $(CUDA_FLAGS)

# 压测:make bench(bench/run.sh, 跑 5 次取中位数);按 workload 压测:make workloads && make run-cuda ARGS="bench bench/workloads/s1.json"
bench:
	bash bench/run.sh
workloads:
	python3 bench/gen_workload.py bench/workloads

# 和 nano-vllm 对比:同一份 workload 两个引擎交替跑 3 次, 结果在 bench/results/, 再画图
compare:
	bash bench/compare.sh 3
plot:
	python3 bench/plot.py bench/results docs/images/bench-compare.png

# 对拍(greedy 输出在每一步优化前后应完全相同):
#   make run-cuda 2>&1 | grep '^token ids' > ~/llmgen-golden.txt
#   make run-cuda 2>&1 | grep '^token ids' | diff - ~/llmgen-golden.txt && echo 一致
# 性能打点:LLMGEN_METRICS=1 LLMGEN_PROFILE=1 make run-cuda ARGS="chat --single"

clean:
	rm -rf .zig-cache zig-out
