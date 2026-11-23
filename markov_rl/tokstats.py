"""Measure token costs with the real Qwen2.5 tokenizer -> results/tokenization.json.

Also picks the default chunk budget C (see `choose_C`).

    python -m markov_rl.tokstats
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics

from transformers import AutoTokenizer

from . import interp
from .common import MODEL_NAME, RESULTS, write_json

KS = (4, 8, 16)
NS = (8, 16, 32, 64)
CS = (64, 128, 160, 192, 224, 256)
MAX_CHUNKS = 12


def ntok(tok, s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))


def measure(tok, n_programs: int = 300, seed: int = 0) -> dict:
    rng = random.Random(seed)
    out: dict = {"model": MODEL_NAME}

    # Tokens per trace line (incl. trailing newline), by format, sampled over random programs.
    per_line = {"verbose": [], "terse": []}
    trace_tokens = {fmt_: {N: [] for N in NS} for fmt_ in per_line}
    program_tokens = {N: [] for N in NS}
    for _ in range(n_programs):
        for N in NS:
            prog = interp.sample_program(rng, N, rng.choice(KS))
            program_tokens[N].append(ntok(tok, interp.render_prompt(interp.render_program(prog))))
            for fmt_ in per_line:
                terse = fmt_ == "terse"
                trace = interp.render_trace(prog, terse=terse)
                trace_tokens[fmt_][N].append(ntok(tok, trace))
                if N == 32:
                    regs = interp.initial_regs(prog)
                    for i, s in enumerate(prog.steps, 1):
                        per_line[fmt_].append(ntok(tok, interp.render_trace_line(i, s, regs, terse) + "\n"))
                        regs[s.dst], _ = interp.apply_step(s, regs)
    out["tokens_per_trace_line"] = {
        f: {"mean": statistics.mean(v), "max": max(v), "min": min(v)} for f, v in per_line.items()
    }
    out["trace_tokens_mean"] = {f: {N: statistics.mean(v) for N, v in d.items()} for f, d in trace_tokens.items()}
    out["trace_tokens_max"] = {f: {N: max(v) for N, v in d.items()} for f, d in trace_tokens.items()}
    out["prompt_tokens_mean_empty_state"] = {N: statistics.mean(v) for N, v in program_tokens.items()}
    out["answer_line_tokens"] = ntok(tok, "\n" + interp.render_answer(52))

    # Minimal state size s_min (content only; tags not counted toward m). Worst case pc has 2 digits.
    s_min, tags = {}, ntok(tok, "<mem></mem>")
    for k in KS:
        sizes = []
        for _ in range(200):
            regs = {r: rng.randrange(100) for r in interp.REG_NAMES[:k]}
            sizes.append(ntok(tok, interp.render_state(rng.randint(10, 64), regs, k, tags=False)))
        s_min[k] = max(sizes)
    out["s_min"] = s_min
    out["mem_tag_tokens"] = tags

    # Chunks needed. full/tail: trace (+answer) split into C-token chunks.
    # learned: every non-final chunk must also hold a full-state dump (s_min + tags + newline).
    tpl = out["tokens_per_trace_line"]["verbose"]["mean"]
    chunks = {"full_or_tail": {}, "learned": {}}
    for C in CS:
        chunks["full_or_tail"][C] = {
            N: math.ceil((out["trace_tokens_max"]["verbose"][N] + out["answer_line_tokens"]) / C) for N in NS
        }
        chunks["learned"][C] = {}
        for k in KS:
            lines = math.floor((C - s_min[k] - tags - 1) / tpl)
            chunks["learned"][C][k] = {N: (math.ceil(N / lines) if lines > 0 else None) for N in NS}
    out["chunks_needed_verbose"] = chunks
    out["max_chunks"] = MAX_CHUNKS
    return out


def choose_C(stats: dict, max_chunks: int = MAX_CHUNKS) -> tuple[int, list[str]]:
    """Smallest C such that:
    (1) N=16 needs >= 2 chunks (memory is exercised at N=16),
    (2) N=64 fits within max_chunks-2 chunks in full/tail mode (slack),
    (3) N=64 fits within max_chunks in learned mode for every k, assuming a full-state dump per chunk.
    """
    log = []
    ch = stats["chunks_needed_verbose"]
    for C in CS:
        ft = ch["full_or_tail"][C]
        lr = ch["learned"][C]
        c1 = ft[16] >= 2
        c2 = ft[64] <= max_chunks - 2
        c3 = all(lr[k][64] is not None and lr[k][64] <= max_chunks for k in KS)
        log.append(f"C={C}: N16_chunks={ft[16]} (>=2:{c1}), N64_full/tail={ft[64]} (<= {max_chunks-2}:{c2}), "
                   f"N64_learned={{k:{ {k: lr[k][64] for k in KS} }}} (<= {max_chunks}:{c3})")
        if c1 and c2 and c3:
            return C, log
    raise RuntimeError("no C satisfies the constraints:\n" + "\n".join(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "tokenization.json"))
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    stats = measure(tok)
    C, log = choose_C(stats)
    stats["chosen_C"] = C
    stats["choose_C_log"] = log
    write_json(args.out, stats)
    print(json.dumps({k: stats[k] for k in ("tokens_per_trace_line", "s_min", "chosen_C")}, indent=1))
    print("\n".join(log))


if __name__ == "__main__":
    main()
