"""Build results/run_table.md (+ .json) from results/timing.json.

    python -m markov_rl.run_table --rate 1.20      # $/GPU-hour (vast.ai rate)
"""
from __future__ import annotations

import argparse
import json

from .common import RESULTS, write_json

# (tier, run, init ckpt, m values, seeds, cost key)
PLAN = [
    ("A", "full-context RL", "warmup_plain", ["n/a"], 2, "full"),
    ("A", "tail RL", "warmup_plain", [25, 100], 2, "tail"),
    ("A", "learned RL", "warmup_mem", [25, 100], 2, "learned"),
    ("B", "tail RL", "warmup_plain", [50, 200], 1, "tail"),
    ("B", "learned RL", "warmup_mem", [50, 200], 1, "learned"),
]
FINAL_EVAL_PROGRAMS = 500 * 4 * 3  # test set: 500 per (N, k), N in {8,16,32,64}, k in {4,8,16}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=1.20, help="$ per GPU-hour")
    args = ap.parse_args()
    timing = load(RESULTS / "timing.json")
    if not timing:
        raise SystemExit("results/timing.json missing: run `python -m markov_rl.pilot_timing` first")
    modes = timing["modes"]
    notes = []

    def cost(key: str, which: str) -> float:
        src = key if key in modes else "learned"
        if src != key:
            notes.append(f"{key} cost taken from the learned pilot (no {key} pilot run).")
        return modes[src][f"est_gpu_h_per_300_steps_{which}"]

    lower = any(m.get("gen_only_lower_bound") for m in modes.values())
    rows, tot = [], {"upper": 0.0, "curr": 0.0}
    lean = {"upper": 0.0, "curr": 0.0}
    for tier, run, init, ms, seeds, key in PLAN:
        n = len(ms) * seeds
        u, c = cost(key, "fixedN32_upper"), cost(key, "curriculum")
        rows.append({"tier": tier, "run": run, "init": init, "m": ms, "seeds": seeds, "n_runs": n,
                     "gpu_h_per_run_upper": u, "gpu_h_per_run_curriculum": c, "total_upper": u * n, "total_curriculum": c * n})
        tot["upper"] += u * n
        tot["curr"] += c * n
        if tier == "A":  # lean plan: 1 seed, m in {25, 100}
            n_lean = len(ms)
            lean["upper"] += u * n_lean
            lean["curr"] += c * n_lean

    # Shared costs (measured where available).
    sft = [load(RESULTS / f"sft_warmup_{n}.json") for n in ("plain", "mem")]
    sft_h = sum((s or {}).get("wall_clock_s", 0) for s in sft) / 3600
    pilot_h = sum(json.loads((RESULTS / "grpo" / m["run"] / "summary.json").read_text()).get("wall_clock_s", 0)
                  for m in modes.values() if (RESULTS / "grpo" / m["run"] / "summary.json").exists()) / 3600
    we = load(RESULTS / "warmup_eval.json")
    warm_eval_h = (we or {}).get("wall_clock_s", 0) / 3600
    n_rl = sum(r["n_runs"] for r in rows)
    final_eval_h = None
    if we:
        n_eps = sum(r["overall"]["n"] for r in we["runs"].values())
        sec_per_ep = sum(r["gen_time_s"] for r in we["runs"].values()) / max(1, n_eps)
        # Batched throughput from the warm-up eval (N<=32); x1.5 allowance for the longer N=64 cells.
        final_eval_h = 1.5 * sec_per_ep * FINAL_EVAL_PROGRAMS * n_rl / 3600
    shared = sft_h + pilot_h + warm_eval_h + (final_eval_h or 0)
    lean_eval = (final_eval_h or 0) * 5 / n_rl

    def money(h):
        return f"${h * args.rate:,.0f}"

    L = []
    L.append(f"# Run table (GPU-hours on 1x A100 80GB; ${args.rate:.2f}/GPU-h)\n")
    if lower:
        L.append("**WARNING: pilot timed generation only (GRPO update not timed) -> numbers are LOWER bounds.**\n")
    L.append("GPU-h/run: upper = 300 steps at fixed N=32; curriculum = scaled by a linear N ramp (see timing.json).\n")
    L.append("| Tier | Run | Init ckpt | m | Seeds | # runs | GPU-h/run (upper) | GPU-h/run (curriculum) | Total (upper) | Total (curriculum) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['tier']} | {r['run']} | {r['init']} | {', '.join(map(str, r['m']))} | {r['seeds']} | {r['n_runs']} | "
                 f"{r['gpu_h_per_run_upper']:.2f} | {r['gpu_h_per_run_curriculum']:.2f} | {r['total_upper']:.1f} | {r['total_curriculum']:.1f} |")
    L.append(f"| shared | warm-up SFT x2 (measured {sft_h:.2f}), pilot (measured {pilot_h:.2f}), warm-up eval (measured {warm_eval_h:.2f}), "
             f"final test eval of {n_rl} runs (est. {final_eval_h or 0:.1f}) | | | | | | | {shared:.1f} | {shared:.1f} |")
    gu, gc = tot["upper"] + shared, tot["curr"] + shared
    shared_lean = sft_h + pilot_h + warm_eval_h + lean_eval
    lu, lc = lean["upper"] + shared_lean, lean["curr"] + shared_lean
    L.append("")
    L.append("| Plan | RL runs | GPU-h (upper) | GPU-h (curriculum) | $ (upper) | $ (curriculum) |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| Grand total | {n_rl} | {gu:.1f} | {gc:.1f} | {money(gu)} | {money(gc)} |")
    L.append(f"| Lean (1 seed, m in {{25, 100}} + full) | 5 | {lu:.1f} | {lc:.1f} | {money(lu)} | {money(lc)} |")
    L.append("")
    L.append("- k is mixed within every run (k ~ U{4, 8, 16} per program); eval slices by k. No separate k sweep.")
    L.append("- Proposal said '6 main runs + 4 sweep runs'; the actual plan is "
             f"{n_rl} RL runs (2 seeds x [full + tail x2 m + learned x2 m] + 1 seed x [tail, learned] x m in {{50, 200}}).")
    for n in sorted(set(notes)):
        L.append(f"- {n}")
    md = "\n".join(L) + "\n"
    (RESULTS / "run_table.md").write_text(md)
    write_json(RESULTS / "run_table.json", {"rate": args.rate, "rows": rows, "shared_gpu_h": shared,
                                            "grand_total": {"upper": gu, "curriculum": gc},
                                            "lean_total": {"upper": lu, "curriculum": lc},
                                            "lower_bound": lower, "notes": sorted(set(notes))})
    print(md)


if __name__ == "__main__":
    main()
