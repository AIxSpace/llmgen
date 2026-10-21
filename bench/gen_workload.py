"""生成压测 workload(module-serving(-unfused) 与 nano-vllm 共用同一份 JSON, 保证输入完全一致).

用法: python gen_workload.py <out_dir>

场景:
  s1  单请求: 输入 512 / 输出 256, 5 轮(每轮不同 prompt) → 单请求 TTFT / ITL / 单流吞吐
  s2  中并发: 32 条, 输入 128~512 / 输出 128~512      → 中等并发
  s3  nano-vllm bench.py 同款: 256 条, 输入 100~1024 / 输出 100~1024(随机数生成顺序与 bench.py 完全一致)
  s4  小并发: 8 条 × 3 轮, 输入 256~512 / 输出 128~512 → decode 批落在 CUDA Graph 桶 4/8(GEMM 的 M=4~8)
  s5  长上下文单请求: 输入 3584 / 输出 512, 3 轮 → decode 注意力成为大头(cuda-course 第 31/35 课 split-K 消融)
全部 temperature=0.6、ignore_eos=True(两边输出 token 数严格一致).

格式: {"name", "rounds": [[req, ...], ...]} —— 各轮依次跑, 轮内请求在 t0 同时提交.
每轮 prompt 都不同: nano-vllm 有 prefix cache(module-serving 没有), 同一 prompt 重复会让它命中缓存
(且 prompt 恰为 256 的整数倍时会触发 nano-vllm 的 flash-attn 空 query 报错), 不是同一份工作量.
"""

import json
import os
import sys
from random import randint, seed


def s1():
    seed(0)
    rounds = [
        [ {"prompt_ids": [randint(0, 10000) for _ in range(512)], "max_tokens": 256, "temperature": 0.6, "ignore_eos": True} ] for _ in range(5)
    ]
    return {"name": "s1-single", "rounds": rounds}


def s2():
    seed(1)
    reqs = []
    for _ in range(32):
        prompt = [randint(0, 10000) for _ in range(randint(128, 512))]
        reqs.append({"prompt_ids": prompt, "max_tokens": randint(128, 512), "temperature": 0.6, "ignore_eos": True})
    return {"name": "s2-concurrent32", "rounds": [reqs]}


def s3():
    # 与 nano-vllm bench.py 逐行一致: 先生成全部 prompt, 再生成全部 max_tokens
    seed(0)
    num_seqs, max_input_len, max_ouput_len = 256, 1024, 1024
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    max_tokens = [randint(100, max_ouput_len) for _ in range(num_seqs)]
    reqs = [{"prompt_ids": p, "max_tokens": m, "temperature": 0.6, "ignore_eos": True}
            for p, m in zip(prompt_token_ids, max_tokens)]
    return {"name": "s3-bench256", "rounds": [reqs]}


def s4():
    # 各轮长度相同(bench 按第 0 轮统计 token 数), 只有 prompt 内容不同
    seed(2)
    lens = [(randint(256, 512), randint(128, 512)) for _ in range(8)]
    rounds = [[{"prompt_ids": [randint(0, 10000) for _ in range(n_in)], "max_tokens": n_out,
                "temperature": 0.6, "ignore_eos": True} for n_in, n_out in lens] for _ in range(3)]
    return {"name": "s4-concurrent8", "rounds": rounds}


def s5():
    seed(3)
    rounds = [[{"prompt_ids": [randint(0, 10000) for _ in range(3584)], "max_tokens": 512,
                "temperature": 0.6, "ignore_eos": True}] for _ in range(3)]
    return {"name": "s5-longctx", "rounds": rounds}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "workloads"
    os.makedirs(out, exist_ok=True)
    for name, fn in (("s1", s1), ("s2", s2), ("s3", s3), ("s4", s4), ("s5", s5)):
        wl = fn()
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(wl, f)
        r0 = wl["rounds"][0]
        n_in = sum(len(r["prompt_ids"]) for r in r0)
        n_out = sum(r["max_tokens"] for r in r0)
        print(f"{name}: 每轮 {len(r0)} 请求 × {len(wl['rounds'])} 轮, 每轮输入 {n_in} / 输出 {n_out} token")
