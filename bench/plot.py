"""汇总 bench/compare.sh 的结果, 打印 Markdown 表格并画对比图.

用法: uv run --with matplotlib python bench/plot.py [results 目录, 默认 bench/results] [图片输出, 默认 docs/images/bench-compare.png]
每个 (引擎, workload) 取多次运行的中位数。
"""

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENGINES = [("llmgen", "llmgen", "#2a78d6"), ("nano-vllm", "nano-vllm", "#eb6834")]
WORKLOADS = [
    ("s1-single", "s1 单请求\n512→256"),
    ("s4-concurrent8", "s4 8 并发\n256~512→128~512"),
    ("s2-concurrent32", "s2 32 并发\n128~512→128~512"),
    ("s3-bench256", "s3 256 并发\n100~1024→100~1024"),
    ("s5-longctx", "s5 长上下文\n3584→512"),
]
METRICS = [  # (取值函数, 行标题, 单位, 越大越好)
    (lambda d: d["throughput_out"], "输出吞吐", "tok/s", True),
    (lambda d: d["ttft_ms"]["p50"], "TTFT p50", "ms", False),
    (lambda d: d["itl_ms"]["p50"], "ITL p50", "ms", False),
]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(results_dir):
    runs = defaultdict(list)
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        d = json.load(open(f))
        runs[(d["engine"], d["workload"])].append(d)
    return runs


def med(runs, key, fn):
    xs = [fn(d) for d in runs.get(key, [])]
    return statistics.median(xs) if xs else None


def table(runs):
    rows = [
        "| 场景 | 引擎 | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | TTFT p50 / p99 (ms) | ITL p50 / p99 (ms) | 墙钟 (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for wl, label in WORKLOADS:
        for eng, name, _ in ENGINES:
            k = (eng, wl)
            if k not in runs:
                continue
            g = lambda fn: med(runs, k, fn)  # noqa: E731
            rows.append(
                f"| {label.replace(chr(10), ' ')} | {name} | {g(lambda d: d['throughput_out']):,.0f} | "
                f"{g(lambda d: d['throughput_total']):,.0f} | "
                f"{g(lambda d: d['ttft_ms']['p50']):,.1f} / {g(lambda d: d['ttft_ms']['p99']):,.1f} | "
                f"{g(lambda d: d['itl_ms']['p50']):.2f} / {g(lambda d: d['itl_ms']['p99']):.2f} | "
                f"{g(lambda d: d['wall_s']):.2f} |"
            )
    return "\n".join(rows)


def fmt(v):
    return f"{v:,.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def plot(runs, out):
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    wls = [(wl, label) for wl, label in WORKLOADS if any((e, wl) in runs for e, _, _ in ENGINES)]
    fig, axes = plt.subplots(len(METRICS), len(wls), figsize=(3.0 * len(wls), 2.7 * len(METRICS)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for r, (fn, title, unit, higher_better) in enumerate(METRICS):
        for c, (wl, label) in enumerate(wls):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            vals = [med(runs, (e, wl), fn) for e, _, _ in ENGINES]
            for i, ((_, _, color), v) in enumerate(zip(ENGINES, vals)):
                if v is None:
                    continue
                ax.bar(i, v, width=0.62, color=color, edgecolor=SURFACE, linewidth=2)
                ax.text(i, v, fmt(v), ha="center", va="bottom", fontsize=9, color=INK)
            # 加速比:以 nano-vllm 为基准, 统一写成 "llmgen 好多少倍"
            if all(v is not None for v in vals):
                a, b = vals
                ratio = a / b if higher_better else b / a
                ax.set_title(f"llmgen ×{ratio:.2f}", fontsize=10, color=INK, fontweight="bold" if ratio >= 1 else "normal")
            ax.set_xticks([])
            ax.set_ylim(0, max(v for v in vals if v is not None) * 1.22)
            ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(INK2)
            if c == 0:
                ax.set_ylabel(f"{title} ({unit})\n{'越高越好' if higher_better else '越低越好'}", fontsize=10, color=INK)
            if r == len(METRICS) - 1:
                ax.set_xlabel(label, fontsize=9, color=INK2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in ENGINES]
    fig.legend(handles, [n for _, n, _ in ENGINES], loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("llmgen vs nano-vllm · Qwen3-0.6B BF16 · RTX 3090", fontsize=13, color=INK, y=1.045)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"图已保存: {out}")


if __name__ == "__main__":
    results = sys.argv[1] if len(sys.argv) > 1 else "bench/results"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/images/bench-compare.png"
    runs = load(results)
    print(table(runs))
    plot(runs, out)
