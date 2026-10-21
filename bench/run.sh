#!/usr/bin/env bash
# 用法: bash bench/run.sh [次数, 默认 5]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"     # 不管在哪个目录运行, 都切到仓库根目录

runs="${1:-5}"
commit=$(git rev-parse --short HEAD)
git diff --quiet || commit="${commit}-dirty"   # 工作区有未提交改动时打标记
out="bench/${commit}"
mkdir -p "$out"

# 和 Makefile 里 run-cuda 保持一致
export LLMGEN_MODEL="${LLMGEN_MODEL:-/home/deploy/qwen/qwen306b/models/Qwen3-0.6B-BF16.gguf}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
zig build -Dcuda=true -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast

for i in $(seq 1 "$runs"); do
  echo "run ${i}/${runs} ..."
  LLMGEN_METRICS=1 \
  LLMGEN_METRICS_JSON="${out}/run${i}.json" \
  ./zig-out/bin/llmgen chat --single > /dev/null 2> "${out}/run${i}.log" || {
    echo "第 ${i} 次运行失败, 日志: ${out}/run${i}.log"; exit 1;
  }
done

python3 - "$out" <<'EOF'
import json, glob, sys, statistics as st
files = sorted(glob.glob(sys.argv[1] + "/*.json"))
if not files:
    sys.exit("没有生成任何 JSON, 检查 run*.log")
runs = [json.load(open(f)) for f in files]
print(f"{len(runs)} 次运行, 结果目录 {sys.argv[1]}")
for k in ["ttft_ms", "tpot_ms", "itl_p99_ms", "prefill_tps", "decode_tps"]:
    xs = [r[k] for r in runs]
    m = st.median(xs)
    print(f"{k:12s} median={m:9.2f}  min={min(xs):9.2f}  max={max(xs):9.2f}  波动=±{(max(xs)-min(xs))/2/m*100:4.1f}%")
EOF