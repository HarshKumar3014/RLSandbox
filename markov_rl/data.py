"""Datasets: fixed-seed dev/test sets and the two warm-up SFT sets (jsonl).

    python -m markov_rl.data            # writes data/{dev,test,warmup_plain,warmup_mem}.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random

from . import interp
from .common import DATA, RESULTS, write_jsonl

KS = (4, 8, 16)
EVAL_NS = (8, 16, 32, 64)
TRAIN_N = (4, 32)
SEEDS = {"dev": 1001, "test": 2002, "warmup": 3003}


def make_row(prog: interp.Program, pid: str) -> dict:
    text = interp.render_program(prog)
    return {
        "id": pid,
        "N": prog.N,
        "k": prog.k,
        "program": prog.to_dict(),
        "program_text": text,
        "prompt": interp.render_prompt(text),
        "answer": interp.answer_from(prog),
    }


def sample_train_program(rng: random.Random, n_lo: int, n_hi: int, ks=KS) -> interp.Program:
    """On-demand training program: N ~ U[n_lo, n_hi], k uniformly mixed."""
    return interp.sample_program(rng, rng.randint(n_lo, n_hi), rng.choice(ks))


def make_eval_set(split: str, n_per_cell: int, Ns=EVAL_NS, ks=KS) -> list[dict]:
    rows = []
    for N in Ns:
        for k in ks:
            # Independent stream per cell so cell contents don't depend on n_per_cell of other cells.
            rng = random.Random(f"{split}-{SEEDS[split]}-{N}-{k}")
            for i in range(n_per_cell):
                rows.append(make_row(interp.sample_program(rng, N, k), f"{split}-N{N}-k{k}-{i}"))
    return rows


def dump_gap_range(k: int, tokstats: dict) -> tuple[int, int]:
    """Steps between <mem> dumps in warmup_mem, chosen so a run of trace lines plus one dump fits in one
    chunk of C tokens (dumps land about as often as learned-mode chunk boundaries would)."""
    C = tokstats["chosen_C"]
    tpl = tokstats["tokens_per_trace_line"]["verbose"]["mean"]
    s_min = tokstats["s_min"][str(k)]
    hi = max(1, math.floor((C - s_min - tokstats["mem_tag_tokens"] - 1) / tpl))
    return max(1, math.ceil(hi / 2)), hi


def make_warmup(n: int, tokstats: dict, terse: bool = False) -> tuple[list[dict], list[dict]]:
    """Same programs for both sets. Plain: no <mem>. Mem: full-state dump every U[lo, hi] steps."""
    rng = random.Random(SEEDS["warmup"])
    plain, mem = [], []
    for i in range(n):
        prog = sample_train_program(rng, *TRAIN_N)
        row = make_row(prog, f"warmup-{i}")
        lo, hi = dump_gap_range(prog.k, tokstats)
        dumps, t = set(), 0
        while True:
            t += rng.randint(lo, hi)
            if t >= prog.N:
                break
            dumps.add(t)
        plain.append({**row, "completion": interp.render_trace(prog, terse=terse)})
        mem.append({**row, "completion": interp.render_trace(prog, terse=terse, dump_after=dumps),
                    "dumps": sorted(dumps)})
    return plain, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_warmup", type=int, default=5000)
    ap.add_argument("--n_dev", type=int, default=50)
    ap.add_argument("--n_test", type=int, default=500)
    args = ap.parse_args()
    tokstats = json.loads((RESULTS / "tokenization.json").read_text())
    write_jsonl(DATA / "dev.jsonl", make_eval_set("dev", args.n_dev))
    write_jsonl(DATA / "test.jsonl", make_eval_set("test", args.n_test))
    plain, mem = make_warmup(args.n_warmup, tokstats)
    write_jsonl(DATA / "warmup_plain.jsonl", plain)
    write_jsonl(DATA / "warmup_mem.jsonl", mem)
    n_dumps = sum(len(r["dumps"]) for r in mem)
    print(f"dev/test/warmup written to {DATA}; C={tokstats['chosen_C']}; "
          f"gap ranges { {k: dump_gap_range(k, tokstats) for k in KS} }; mean dumps/example {n_dumps / len(mem):.2f}")


if __name__ == "__main__":
    main()
