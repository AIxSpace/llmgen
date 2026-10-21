"""用 nano-vllm 跑 gen_workload.py 生成的 workload, 输出和 `llmgen bench` 同字段的 JSON.

用法(在 nano-vllm 的 venv 里): python nanovllm_bench.py <workload.json> [--model DIR] [--out PATH] [--eager]

口径与 src/llmgen.zig 的 bench / runRound 一致:
  - 预热: 第 0 条请求的前 8 个 token, 生成 64 个
  - 各轮依次跑, 轮内请求在 t0 同时提交; 每步 step() 返回后(token 已回到 host)记时间戳
  - TTFT = t0 到该请求首 token; ITL = 同一请求相邻 token 的间隔(所有请求的样本合在一起)
  - 吞吐 = 一轮的输出 token 数 ÷ 各轮墙钟的中位数
"""

import argparse
import json
import os
import resource
import time

import numpy as np
import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


def stats(xs):
    if not xs:
        return {"mean": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "n": int(a.size),
    }


def run_round(llm, reqs, ttft_ms, itl_ms):
    t0 = time.perf_counter()
    seqs = []
    for r in reqs:
        sp = SamplingParams(temperature=r["temperature"], max_tokens=r["max_tokens"], ignore_eos=r["ignore_eos"])
        seq = Sequence(r["prompt_ids"], sp)
        llm.scheduler.add(seq)
        seqs.append(seq)
    last_emit = [None] * len(seqs)
    emitted = [0] * len(seqs)
    n_steps = 0
    while not llm.is_finished():
        llm.step()
        n_steps += 1
        now = time.perf_counter()
        for k, s in enumerate(seqs):
            n = s.num_completion_tokens
            if n == emitted[k]:
                continue
            emitted[k] = n
            if last_emit[k] is None:
                ttft_ms.append((now - t0) * 1e3)
            else:
                itl_ms.append((now - last_emit[k]) * 1e3)
            last_emit[k] = now
    return time.perf_counter() - t0, n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    ap.add_argument("--out")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-batched-tokens", type=int, default=16384)
    ap.add_argument("--gpu-util", type=float, default=0.9)
    args = ap.parse_args()

    wl = json.load(open(args.workload))
    reqs0 = wl["rounds"][0]
    in_tok = sum(len(r["prompt_ids"]) for r in reqs0)
    out_tok = sum(r["max_tokens"] for r in reqs0)
    print(f"[bench] {wl['name']}: {len(reqs0)} 请求 × {len(wl['rounds'])} 轮, 输入 {in_tok} / 输出 {out_tok} token", flush=True)

    t_load = time.perf_counter()
    llm = LLM(
        args.model,
        enforce_eager=args.eager,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        gpu_memory_utilization=args.gpu_util,
    )
    load_s = time.perf_counter() - t_load

    llm.generate([reqs0[0]["prompt_ids"][:8]], SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=True), use_tqdm=False)

    ttft_ms, itl_ms, walls, steps = [], [], [], 0
    for i, reqs in enumerate(wl["rounds"]):
        wall, n = run_round(llm, reqs, ttft_ms, itl_ms)
        walls.append(wall)
        steps += n
        print(f"[bench] 第 {i + 1} 轮: {wall:.3f} s, {out_tok / wall:.1f} tok/s", flush=True)

    wall = float(np.median(walls))
    free, total = torch.cuda.mem_get_info()
    mb = 1024.0 * 1024.0
    cfg = llm.model_runner.config
    kv = llm.model_runner.kv_cache
    summary = {
        "engine": "nano-vllm-eager" if args.eager else "nano-vllm",
        "workload": wl["name"],
        "n_requests": len(reqs0),
        "repeat": len(wl["rounds"]),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "wall_s": wall,
        "throughput_out": out_tok / wall,
        "throughput_total": (out_tok + in_tok) / wall,
        "ttft_ms": stats(ttft_ms),
        "itl_ms": stats(itl_ms),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "device_used_mb": (total - free) / mb,
        "kv_cache_mb": kv.numel() * kv.element_size() / mb,
        "kv_tokens": cfg.num_kvcache_blocks * cfg.kvcache_block_size,
        "load_s": load_s,
        "steps": steps,
    }
    js = json.dumps(summary, indent=2, ensure_ascii=False)
    print(js)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)


if __name__ == "__main__":
    main()
