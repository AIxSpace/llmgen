#!/usr/bin/env bash
# llmgen vs nano-vllm 同 workload 对比(RTX 3090 上跑)
# 用法: bash bench/compare.sh [重复次数, 默认 3] [workload 列表, 默认 "s1 s2 s3 s4 s5"]
# 结果: bench/results/<engine>-<workload>-<i>.json, 画图: python3 bench/plot.py bench/results
set -euo pipefail
cd "$(dirname "$0")/.."

reps="${1:-3}"
wls="${2:-s1 s2 s3 s4 s5}"
NANOVLLM_DIR="${NANOVLLM_DIR:-$HOME/nano-vllm}"
HF_MODEL="${HF_MODEL:-$HOME/huggingface/Qwen3-0.6B/}"
export LLMGEN_MODEL="${LLMGEN_MODEL:-/home/deploy/qwen/qwen306b/models/Qwen3-0.6B-BF16.gguf}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

out=bench/results
mkdir -p "$out"
[ -f bench/workloads/s1.json ] || python3 bench/gen_workload.py bench/workloads
zig build -Dcuda=true -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast

for wl in $wls; do
  for i in $(seq 1 "$reps"); do
    # 两个引擎交替跑, 抵消 GPU 温度 / 频率随时间的漂移
    echo "== $wl #$i llmgen"
    ./zig-out/bin/llmgen bench "bench/workloads/$wl.json" --out "$out/llmgen-$wl-$i.json" 2>&1 | grep '^\[bench\]'
    echo "== $wl #$i nano-vllm"
    "$NANOVLLM_DIR/.venv/bin/python" bench/nanovllm_bench.py "bench/workloads/$wl.json" \
      --model "$HF_MODEL" --out "$out/nano-vllm-$wl-$i.json" 2>&1 | grep '^\[bench\]'
  done
done
