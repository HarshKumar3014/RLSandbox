"""Assemble results/milestone_results.md from the json results (facts only).

    python -m markov_rl.report
"""
from __future__ import annotations

import json

from .common import RESULTS


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def tokenization_section(t) -> list[str]:
    if not t:
        return ["- tokenization.json missing"]
    tpl = t["tokens_per_trace_line"]
    ch = t["chunks_needed_verbose"]
    L = [
        f"- Tokenizer: `{t['model']}` (digits are single tokens).",
        f"- Tokens per trace line incl. newline (N=32 programs): verbose mean {tpl['verbose']['mean']:.1f} "
        f"(min {tpl['verbose']['min']}, max {tpl['verbose']['max']}); terse mean {tpl['terse']['mean']:.1f}.",
        f"- Minimal state size s_min (content `pc=..;a=..;...`, worst case 2-digit pc; tags not counted): "
        f"k=4: {t['s_min']['4']}, k=8: {t['s_min']['8']}, k=16: {t['s_min']['16']} tokens "
        f"(proposal guessed 25/45/85). `<mem></mem>` tags = {t['mem_tag_tokens']} tokens.",
        "- Prompt tokens (program, empty state), mean: "
        + ", ".join(f"N={n}: {v:.0f}" for n, v in t["prompt_tokens_mean_empty_state"].items()),
        "- Verbose trace tokens (max over 300 programs): "
        + ", ".join(f"N={n}: {v}" for n, v in t["trace_tokens_max"]["verbose"].items()),
        "",
        "Chunks needed (verbose; full/tail = trace+answer split into C-token chunks; learned = every non-final chunk "
        "also holds a full-state dump):",
        "",
        "| C | full/tail N=8 | N=16 | N=32 | N=64 | learned k=4 N=64 | k=8 N=64 | k=16 N=32 | k=16 N=64 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for C, ft in ch["full_or_tail"].items():
        lr = ch["learned"][C]
        L.append(f"| {C} | {ft['8']} | {ft['16']} | {ft['32']} | {ft['64']} | {lr['4']['64']} | {lr['8']['64']} | "
                 f"{lr['16']['32']} | {lr['16']['64']} |")
    L += [
        "",
        f"- **Chosen C = {t['chosen_C']} (proposal: 256).** Rule: smallest C with (1) N=16 needs >= 2 chunks, "
        f"(2) N=64 fits in <= {t['max_chunks'] - 2} chunks in full/tail mode, (3) N=64 fits in <= {t['max_chunks']} chunks in "
        "learned mode for every k when each chunk also carries a full-state dump.",
        "- Why not 256: at ~14.5 tokens/line an N=16 trace (~250 tokens) fits in one 256-token chunk, so N=8 and N=16 would "
        "never cross a boundary and memory would not be exercised.",
        "- Why not 128: a full k=16 dump costs ~78 tokens, leaving ~50 tokens (~3 trace lines) per learned chunk, so learned "
        "mode needs 22 chunks for N=64 and 11 for N=32 at k=16 (budget 12). C=160 still needs 13 at k=16, N=64.",
        "- Side effect: tail m=200 >= C=192 means tail-200 keeps the entire previous chunk (still not full context).",
        "- N=8 never crosses a chunk boundary at C=192 (it is a no-memory control cell).",
    ]
    return L


def warmup_section(we, sft) -> list[str]:
    L = []
    if sft:
        for s in sft:
            if s:
                L.append(f"- SFT `{s['config']['name']}`: {s['n_examples']} examples, {s['steps']} steps, "
                         f"wall-clock {s['wall_clock_s'] / 60:.1f} min, final loss {s['final_loss']:.4f}, "
                         f"peak mem {s['peak_gpu_mem_gb'] or 0:.1f} GB.")
    if not we:
        return L + ["- warmup_eval.json missing (run `python -m markov_rl.eval --milestone`)"]
    L += [f"- Eval: {we['split']} set, {we['n_per_cell']} programs per (N, k) cell, N in {we['Ns']}, k in {{4, 8, 16}}, "
          f"greedy, C={we['C']}, max 12 chunks. Wall-clock {we['wall_clock_s'] / 60:.1f} min.", ""]
    L.append("| Run | N | Format compliance % | Per-step arith % | Trace acc % | Final EM % [95% CI] | <mem> exact % | Mean chunks |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, r in we["runs"].items():
        for key in [f"N{n}" for n in we["Ns"]] + ["overall"]:
            a = r["overall"] if key == "overall" else r["cells"][key]
            L.append(f"| {name} | {key.lstrip('N') if key != 'overall' else 'all'} | {pct(a['format_compliance'])} | "
                     f"{pct(a['arith_acc'])} | {pct(a['trace_acc'])} | {pct(a['exact_match'])} {ci(a['exact_match_ci95'])} | "
                     f"{pct(a['mem_exact'])} | {a['mean_chunks']:.2f} |")
    L += ["", "By k (all N pooled), final EM %:", "", "| Run | k=4 | k=8 | k=16 |", "|---|---|---|---|"]
    for name, r in we["runs"].items():
        L.append(f"| {name} | " + " | ".join(pct(r["cells"][f"k{k}"]["exact_match"]) for k in (4, 8, 16)) + " |")
    tail = we["runs"].get("warmup_plain@tail_m100")
    if tail:
        o = tail["overall"]
        L += ["", f"- Untrained tail baseline (warmup_plain, tail m=100): final EM {pct(o['exact_match'])}% "
                  f"{ci(o['exact_match_ci95'])}; state fidelity of the tail window {pct(o['state_fidelity'])}% (H1 row)."]
    L.append("- Per-step arith = result correct given the operand values the model itself wrote on that line "
             "(copy lines excluded: no arithmetic). Trace acc = fraction of the N steps whose written value equals the "
             "ground truth. <mem> exact = dump equals the true (pc, registers) after the preceding trace line.")
    return L


def timing_section(timing) -> list[str]:
    if not timing:
        return ["- timing.json missing (run `python -m markov_rl.pilot_timing`)"]
    L = ["| Pilot (N=32 fixed, mixed k, 32x8 rollouts/step) | s/step | gen s | train s | sync s | gen tokens/step | "
         "train tokens/step | mean chunks | answered % | reward | peak torch mem GB | peak device mem GB | GPU-h/300 steps (upper) | (curriculum) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, m in timing["modes"].items():
        f = lambda v, d=1: "n/a" if v is None else f"{v:.{d}f}"
        L.append(f"| {mode}{' (GEN ONLY: lower bound)' if m['gen_only_lower_bound'] else ''} | {f(m['sec_per_step'])} | "
                 f"{f(m['gen_sec_per_step'])} | {f(m['train_sec_per_step'])} | {f(m['sync_sec_per_step'])} | "
                 f"{f(m['tokens_generated_per_step'], 0)} | {f(m['train_tokens_per_step'], 0)} | {f(m['mean_chunks'], 2)} | "
                 f"{pct(m['answered_frac'])} | {f(m['reward_mean'], 3)} | {f(m['peak_torch_mem_gb'])} | {f(m['peak_device_mem_gb'])} | "
                 f"{f(m['est_gpu_h_per_300_steps_fixedN32_upper'], 2)} | {f(m['est_gpu_h_per_300_steps_curriculum'], 2)} |")
    L += [""] + [f"- {a}" for a in timing.get("assumptions", [])]
    return L


DESIGN = [
    "k is mixed within every run (k ~ U{4, 8, 16} per program); eval slices by k. No separate k sweep.",
    "Learned-mode truncation: the content of the LAST `<mem>` span of the chunk is cut to its FIRST m tokens "
    "(head kept). `pc` is always written first and registers in alphabetical order, so pc survives any m >= ~4. "
    "Tags are not counted toward m. Missing/unterminated/malformed span -> empty state.",
    "Learned mode ends a chunk at `</mem>` (MEM1-style: the model decides when to write memory and move on).",
    "Two warm-up checkpoints: `warmup_plain` (no `<mem>` at all) initializes full and tail runs; `warmup_mem` (same "
    "programs/traces + full-state dumps every U[lo, hi] steps, hi = lines that fit in one C-token chunk with a dump) "
    "initializes learned runs only. Training the tail model on dumps would pre-teach it to write state and confound "
    "tail vs learned (mentor's point).",
    "To tell tail and learned apart after RL, every boundary state is logged; state fidelity is computed for tail "
    "windows too (registers parsed from trace lines / any `<mem>` in the window vs. the true state at the referenced step).",
    "Full-context baseline is budget-matched: same C x max_chunks (192 x 12) generation budget, same warm-up recipe "
    "and the same number of GRPO steps. In full mode all previous outputs are kept after `Trace:` (exactly the SFT format).",
    "Oracle-state SFT is treated as a reference point, not an upper bound (it gets supervision RL never sees). Not run yet.",
    "Reward: 1 iff the final `Answer:` equals the true value; out of chunks = 0. Advantage normalized within each group "
    "of 8 and applied to every chunk of the rollout; groups with zero reward variance contribute no gradient and are skipped.",
    "GRPO: one optimizer step per GRPO step by default (ppo_epochs=1, num_minibatches=1), so pi_old = pi_theta and the "
    "clip is inactive; multiple minibatches/epochs and k3 KL to the warm-up adapter are implemented as flags.",
    "Prompt (SFT, env, eval): `Program:\\n<program>\\nState: <state>\\nTrace:\\n` (`State:` with nothing after it when empty). "
    "No chat template.",
    "Verbose trace lines (`1: c=c+a=12+07=19.`) keep per-step arithmetic trivial; terse format available via flag.",
    "Query register is one written in the last ceil(N/4) steps.",
]

CHANGES = [
    "C changed from 256 to 192 (see tokenization section).",
    "Run count: 14 RL runs (not 6 + 4). See run table.",
    "Measured s_min is 20/40/72 tokens for k=4/8/16 (proposal: ~25/45/85). For m=25 and k=4, the minimal state fits "
    "(20 <= 25); for k=8/16, m=25 is below s_min, so H2 predicts degradation there by construction.",
    "Neither warm-up checkpoint has ever seen a non-empty `State:` field (both are full-context traces, per plan). "
    "Resuming from a carried state is learned purely by RL. If learned-mode reward stays ~0 at small N in the first "
    "RL steps, the fallback is to add resumption examples to warmup_mem (prompt = program + true state at a random "
    "step, completion = rest of trace) - this would be a change to the plan.",
    "Eval reuses the same chunked env as RL (mode=full for the warm-up eval) so the warm-up numbers are directly "
    "comparable to RL numbers.",
    "Local dev machine has no CUDA GPU: all GPU numbers come from the vast.ai A100; CPU tests cover interpreter, env, "
    "metrics, SFT and the GRPO loop with the HF-generate backend.",
]


def main():
    t = load("tokenization.json")
    sft = [load(f"sft_warmup_{n}.json") for n in ("plain", "mem")]
    we = load("warmup_eval.json")
    timing = load("timing.json")
    rt = (RESULTS / "run_table.md").read_text() if (RESULTS / "run_table.md").exists() else "run_table.md missing"
    L = ["# Milestone results (auto-generated by `python -m markov_rl.report`; facts only)", ""]
    L += ["## 1. Tokenization and chunk size", ""] + tokenization_section(t) + [""]
    L += ["## 2. Warm-up SFT at full context", ""] + warmup_section(we, sft) + [""]
    L += ["## 3. Timing pilot", ""] + timing_section(timing) + [""]
    L += ["## 4. Run table", "", rt.replace("# Run table", "### Run table"), ""]
    L += ["## 5. Design decisions", ""] + [f"- {d}" for d in DESIGN] + [""]
    L += ["## 6. Changes from the proposal / things that broke", ""] + [f"- {c}" for c in CHANGES]
    if timing and any(m.get("gen_only_lower_bound") for m in timing["modes"].values()):
        L.append("- GRPO update was not timed (generation-only pilot): run-table numbers are lower bounds.")
    L += ["", "## 7. Dates", "", "- Milestone: TBD", "- Poster: TBD", "- Final report: TBD", ""]
    (RESULTS / "milestone_results.md").write_text("\n".join(L))
    print(f"wrote {RESULTS / 'milestone_results.md'}")


if __name__ == "__main__":
    main()
