"""Timing pilot: 20 GRPO steps per mode at FIXED N=32 (worst-case length), mixed k -> results/timing.json.

Each mode runs in its own subprocess (fresh vLLM engine). Training runs on every chunk even when all
advantages are 0 (`train_on_zero_adv`), so train time is measured even if the untrained policy never
gets a reward.

    python -m markov_rl.pilot_timing                      # learned m=100 + full (spec)
    python -m markov_rl.pilot_timing --modes learned full tail
    python -m markov_rl.pilot_timing --gen_only           # fallback: generation only (LOWER bound)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from .common import RESULTS, run_meta, write_json

PILOTS = {
    "learned": "pilot_learned_m100.yaml",
    "full": "pilot_full.yaml",
    "tail": "pilot_tail_m100.yaml",
}
RUN_STEPS = 300
SKIP = 2  # warm-up steps excluded from averages (CUDA graph capture, allocator growth)


def curriculum_scale(n_lo=4, n_start=8, n_max=32, n_fixed=32, steps=RUN_STEPS) -> float:
    """Mean of E[N]/n_fixed over a run whose max N ramps linearly from n_start to n_max, with
    N ~ U[n_lo, n_hi(t)]. Assumes step cost is linear in N (tokens per step ~ N)."""
    t = np.arange(steps)
    n_hi = n_start + (n_max - n_start) * t / (steps - 1)
    return float(np.mean((n_lo + n_hi) / 2 / n_fixed))


def summarize(name: str, gen_only: bool) -> dict:
    d = RESULTS / "grpo" / name
    logs = [json.loads(l) for l in open(d / "log.jsonl")]
    summ = json.loads((d / "summary.json").read_text())
    use = logs[SKIP:] if len(logs) > SKIP else logs

    def mean(k):
        v = [r[k] for r in use if r.get(k) is not None]
        return float(np.mean(v)) if v else None

    step_s = mean("step_time_s")
    scale = curriculum_scale()
    out = {
        "run": name,
        "steps_measured": len(use),
        "sec_per_step": step_s,
        "gen_sec_per_step": mean("gen_time_s"),
        "train_sec_per_step": mean("train_time_s"),
        "sync_sec_per_step": mean("sync_time_s"),
        "tokens_generated_per_step": mean("tokens_generated"),
        "train_tokens_per_step": mean("train_n_tokens"),
        "train_seqs_per_step": mean("train_n_seqs"),
        "mean_chunks": mean("mean_chunks"),
        "answered_frac": mean("answered_frac"),
        "reward_mean": mean("reward_mean"),
        "peak_context_tokens": max(r["peak_context"] for r in logs),
        "peak_torch_mem_gb": max((r["peak_gpu_mem_gb"] or 0) for r in logs),
        "peak_device_mem_gb": max((r.get("gpu_used_total_gb") or 0) for r in logs),
        "startup_s": summ.get("startup_s"),
        "gen_only_lower_bound": gen_only,
        "est_gpu_h_per_300_steps_fixedN32_upper": RUN_STEPS * step_s / 3600 + (summ.get("startup_s") or 0) / 3600,
        "curriculum_scale": scale,
        "est_gpu_h_per_300_steps_curriculum": RUN_STEPS * step_s * scale / 3600 + (summ.get("startup_s") or 0) / 3600,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["learned", "full"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--gen_only", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="extra key=value overrides for grpo")
    args = ap.parse_args()
    path = RESULTS / "timing.json"
    timing = json.loads(path.read_text()) if path.exists() else {"modes": {}}
    for mode in args.modes:
        name = f"pilot_{mode}" + ("_genonly" if args.gen_only else "")
        cmd = [sys.executable, "-m", "markov_rl.grpo", "--config", f"markov_rl/configs/{PILOTS[mode]}",
               "--set", f"name={name}", f"steps={args.steps}", f"gen_only={args.gen_only}", *args.extra]
        print(" ".join(cmd), flush=True)
        log_path = RESULTS / "grpo" / name / "log.jsonl"
        if log_path.exists():
            log_path.unlink()  # fresh log per pilot
        subprocess.run(cmd, check=True)
        timing["modes"][mode] = summarize(name, args.gen_only)
        timing.update(run_meta(vars(args)))
        timing["assumptions"] = [
            "Fixed N=32 for every program: worst-case length, so (a) is an upper bound for a 300-step run.",
            "(b) scales (a) by mean(E[N])/32 under a linear ramp of max N from 8 to 32 over 300 steps, N~U[4, maxN];"
            " assumes step cost linear in N. Failing rollouts that use all chunks regardless of N make (b) optimistic.",
            f"First {SKIP} steps excluded from averages; per-run startup (model load, vLLM init) added once.",
            "train_on_zero_adv=True: backward runs on every chunk even when advantages are 0 (no optimizer step then),"
            " so train time reflects a batch where every group has signal (upper bound on train cost).",
            "Cost assumed independent of m (pilot at m=100). Tail cost = learned cost unless a tail pilot was run.",
        ]
        write_json(path, timing)  # write after each mode so partial results survive
        print(json.dumps(timing["modes"][mode], indent=1), flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
