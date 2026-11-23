"""Chunked evaluation (greedy) with metrics and 95% bootstrap CIs.

Milestone eval (warmup_plain@full, warmup_mem@full, untrained tail = warmup_plain@tail m=100):
    python -m markov_rl.eval --milestone --Ns 8,16,32 --n_per_cell 200

Single run:
    python -m markov_rl.eval --adapter checkpoints/warmup_plain --name warmup_plain --mode full
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from transformers import AutoTokenizer

from . import interp
from .common import DATA, MODEL_NAME, RESULTS, ROOT, pick_device, read_jsonl, run_meta, write_json, write_jsonl
from .env import EnvConfig, MarkovEnv, run_episodes


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------

def episode_lines(log: dict) -> list[str]:
    """Complete output lines. full: chunks concatenated (one continuous trace). Other modes: per chunk, the
    last line is dropped if the chunk was cut by the token budget (it is partial by construction)."""
    if log["mode"] == "full":
        return "".join(log["chunk_texts"]).splitlines()
    lines = []
    for text, fr in zip(log["chunk_texts"], log["finish_reasons"]):
        ls = text.splitlines()
        if fr == "length" and ls and not text.endswith("\n"):
            ls = ls[:-1]
        lines += ls
    return lines


def state_fidelity(text: str, prog: interp.Program, st: list[dict]) -> float | None:
    """Fraction of the k registers whose value in `text` equals the true value at the step the text refers
    to. Works for <mem> bodies (pc gives the step) and raw tail windows (last trace line gives the step).
    Registers not mentioned count as wrong. None if no reference step can be found."""
    regs, ref = {}, None
    body = text if text.startswith("<mem>") else None
    if text.startswith("pc="):
        body = f"<mem>{text}</mem>"
    for line in (body or text).splitlines():
        kind = interp.classify_line(line)
        if kind == "trace":
            tl = interp.parse_trace_line(line)
            regs[tl.dst], ref = tl.result, tl.step
        elif kind == "mem":
            pc, r = interp.parse_state_body(line.strip()[5:-6])
            regs.update(r)
            ref = pc - 1
    if ref is None:
        if body is not None:  # possibly a head-truncated <mem> body: "pc=9;a=07;b=6"
            mpc = re.match(r"pc=(\d+)", text)
            if not mpc:
                return None
            ref = int(mpc.group(1)) - 1
            for r, v in re.findall(r";([a-p])=(\d{2})(?=;|$)", text):
                regs[r] = int(v)
        else:
            return None
    if not 0 <= ref <= prog.N:
        return 0.0
    truth = st[ref]
    return sum(regs.get(r) == truth[r] for r in prog.regs) / prog.k


def episode_metrics(log: dict, prog: interp.Program) -> dict:
    st = interp.states(prog)
    lines = episode_lines(log)
    kinds = [interp.classify_line(l) for l in lines]
    has_answer = log["pred"] is not None
    format_ok = has_answer and all(k in ("trace", "mem", "answer", "empty") for k in kinds)

    arith_n = arith_ok = 0
    by_step: dict[int, interp.TraceLine] = {}
    mem_n = mem_exact = 0
    last_step = 0
    for line, kind in zip(lines, kinds):
        if kind == "trace":
            tl = interp.parse_trace_line(line)
            if tl.arith_ok is not None:
                arith_n += 1
                arith_ok += tl.arith_ok
            by_step[tl.step] = tl  # last occurrence wins
            last_step = tl.step
        elif kind == "mem" and log["mode"] == "full":
            pc, regs = interp.parse_state_body(line.strip()[5:-6])
            s = min(last_step, prog.N)
            mem_n += 1
            mem_exact += (pc == s + 1 and regs == st[s])
    trace_correct = sum(
        1 for i in range(1, prog.N + 1)
        if i in by_step and by_step[i].dst == prog.steps[i - 1].dst and by_step[i].result == st[i][by_step[i].dst]
    )
    # Fidelity of the carried state at every boundary; empty/unparseable state counts as 0.
    fid = [(state_fidelity(s, prog, st) or 0.0) if s else 0.0 for s in log["states"]]
    return {
        "correct": float(log["reward"]),
        "format_ok": float(format_ok),
        "arith_n": arith_n, "arith_ok": arith_ok,
        "trace_acc": trace_correct / prog.N,
        "mem_n": mem_n, "mem_exact": mem_exact,
        "state_fidelity": float(np.mean(fid)) if fid else None,
        "chunks_used": log["chunks_used"], "tokens_generated": log["tokens_generated"],
        "peak_context": log["peak_context"],
    }


def bootstrap_ci(x, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(ms: list[dict]) -> dict:
    acc = [m["correct"] for m in ms]
    an = sum(m["arith_n"] for m in ms)
    mn = sum(m["mem_n"] for m in ms)
    fid = [m["state_fidelity"] for m in ms if m["state_fidelity"] is not None]
    return {
        "n": len(ms),
        "exact_match": float(np.mean(acc)),
        "exact_match_ci95": bootstrap_ci(acc),
        "format_compliance": float(np.mean([m["format_ok"] for m in ms])),
        "arith_acc": (sum(m["arith_ok"] for m in ms) / an) if an else None,
        "trace_acc": float(np.mean([m["trace_acc"] for m in ms])),
        "mem_dumps": mn,
        "mem_exact": (sum(m["mem_exact"] for m in ms) / mn) if mn else None,
        "state_fidelity": float(np.mean(fid)) if fid else None,
        "mean_chunks": float(np.mean([m["chunks_used"] for m in ms])),
        "mean_tokens": float(np.mean([m["tokens_generated"] for m in ms])),
        "mean_peak_context": float(np.mean([m["peak_context"] for m in ms])),
    }


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def load_eval_rows(split: str, Ns, n_per_cell: int, ks=(4, 8, 16)) -> list[dict]:
    rows = read_jsonl(DATA / f"{split}.jsonl")
    cnt, out = defaultdict(int), []
    for r in rows:
        key = (r["N"], r["k"])
        if r["N"] in Ns and r["k"] in ks and cnt[key] < n_per_cell:
            cnt[key] += 1
            out.append(r)
    return out


def evaluate(generator, tok, rows: list[dict], env_cfg: EnvConfig, name: str, n_samples: int = 20) -> dict:
    envs = []
    for r in rows:
        e = MarkovEnv(env_cfg, tok)
        e.reset(r["program"], meta={"id": r["id"]})
        envs.append(e)
    timing = run_episodes(envs, generator, temperature=0.0)
    logs = [e.episode_log() for e in envs]
    ms = [episode_metrics(l, e.prog) for l, e in zip(logs, envs)]
    cells = defaultdict(list)
    for r, m in zip(rows, ms):
        cells[f"N{r['N']}_k{r['k']}"].append(m)
        cells[f"N{r['N']}"].append(m)
        cells[f"k{r['k']}"].append(m)
    res = {
        "name": name,
        "env": asdict(env_cfg),
        "overall": aggregate(ms),
        "cells": {k: aggregate(v) for k, v in sorted(cells.items())},
        "gen_time_s": timing["gen_time_s"],
    }
    write_jsonl(RESULTS / "eval" / f"samples_{name}.jsonl", logs[:: max(1, len(logs) // n_samples)][:n_samples])
    o = res["overall"]
    print(f"[{name}] EM={o['exact_match']:.3f} {tuple(round(x, 3) for x in o['exact_match_ci95'])} "
          f"format={o['format_compliance']:.3f} arith={o['arith_acc']} trace={o['trace_acc']:.3f} "
          f"mem_exact={o['mem_exact']} time={timing['gen_time_s']:.0f}s", flush=True)
    return res


MILESTONE_RUNS = [
    # (name, adapter, mode, m)
    ("warmup_plain@full", "checkpoints/warmup_plain", "full", 0),
    ("warmup_mem@full", "checkpoints/warmup_mem", "full", 0),
    ("warmup_plain@tail_m100", "checkpoints/warmup_plain", "tail", 100),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", action="store_true")
    ap.add_argument("--adapter")
    ap.add_argument("--name")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--C", type=int, default=None, help="default: chosen_C from results/tokenization.json")
    ap.add_argument("--max_chunks", type=int, default=12)
    ap.add_argument("--split", default="test")
    ap.add_argument("--Ns", default="8,16,32")
    ap.add_argument("--n_per_cell", type=int, default=200)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    C = args.C or json.loads((RESULTS / "tokenization.json").read_text())["chosen_C"]
    Ns = [int(x) for x in args.Ns.split(",")]
    rows = load_eval_rows(args.split, Ns, args.n_per_cell)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    runs = MILESTONE_RUNS if args.milestone else [(args.name or args.adapter, args.adapter, args.mode, args.m)]

    from .gen import make_generator

    gen = make_generator(args.backend, tok, MODEL_NAME, lora_path=None, device=pick_device(),
                         gpu_memory_utilization=args.gpu_mem)
    t0 = time.time()
    results = {}
    for name, adapter, mode, m in runs:
        gen.set_lora(ROOT / adapter if adapter else None)
        cfg = EnvConfig(mode=mode, C=C, m=m, max_chunks=args.max_chunks)
        results[name] = evaluate(gen, tok, rows, cfg, name.replace("@", "_"))
    out = {
        **run_meta(vars(args)),
        "split": args.split, "Ns": Ns, "n_per_cell": args.n_per_cell, "C": C,
        "wall_clock_s": time.time() - t0,
        "runs": results,
    }
    path = args.out or (RESULTS / ("warmup_eval.json" if args.milestone else f"eval/{runs[0][0].replace('@', '_').replace('/', '_')}.json"))
    write_json(path, out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
