# Task: build the milestone pipeline for our COMS 4705 final project (due tomorrow)

I've attached our project proposal PDF and our mentor's feedback. Read both first. Short version: we train Qwen2.5-0.5B-Instruct with RL (GRPO) to solve register-machine programs in fixed-size chunks. Between chunks the context is reset to the program plus a carried state capped at m tokens. We compare three memory modes: full (no reset), tail (keep the last m tokens of the chunk), and learned (keep the model's own `<mem>...</mem>` span, truncated to m).

Nothing is built yet. The milestone is due tomorrow. Our mentor wants to see:
1. The warm-up (SFT) model reliably following the format and doing the arithmetic right at full context.
2. Measured GPU-hours per RL run, and the resulting run table.
3. Dates (I'll fill these in myself).

Compute: a rented A100 80GB on vast.ai. If you're running on a machine without a GPU, write the code, run the CPU tests, and give me exact copy-paste commands for the vast.ai box (setup, then each job, with expected runtime). Get GPU work started as early as possible and write the remaining code while it runs.

## Priorities (strict order, don't gold-plate)

- **P0-a**: interpreter + data generator + unit tests
- **P0-b**: warm-up SFT (two checkpoints, see below) + eval script → milestone item 1
- **P0-c**: chunked environment + rollout generation + timing pilot → milestone item 2
- **P1**: minimal working GRPO loop (needed for a real pilot; if it isn't ready, time generation alone and report that as a lower bound)
- **P2, NOT tonight**: OpenEnv/Docker packaging, probe, corruption test, dense reward, oracle-state SFT. Keep the env API to `reset() / step(action) / state()` so packaging is easy later.

## Repo layout

```
markov_rl/
  interp.py        # program sampling, execution, trace/state rendering + parsing
  data.py          # warm-up sets, fixed-seed dev/test sets (jsonl)
  env.py           # MarkovEnv: modes full|tail|learned; C, m, max_chunks
  sft.py           # LoRA SFT (HF Transformers + PEFT)
  eval.py          # chunked eval with vLLM; metrics + bootstrap CIs
  grpo.py          # our own GRPO implementation (no TRL GRPOTrainer)
  pilot_timing.py  # timing pilot → results/timing.json
  run_table.py     # builds results/run_table.md from timing.json
  configs/*.yaml   # one config per run
  tests/
results/           # all metrics as json + a milestone_results.md summary
```

Pin versions in requirements.txt. Seed everything. Log every run's config + git hash into its results file.

## P0-a: Interpreter spec

- k registers named `a, b, c, ...` (k ∈ {4, 8, 16}). Every value is mod 100, rendered as two digits zero-padded (`07`).
- Ops (uniform mix): `x=y+z`, `x=y-z`, `x=y` (copy), `x=y*2`, `x=y*3`, `x=y+c` (c ∈ 1..9). Operands can be the same register.
- Program text:
  ```
  Start: a=07 b=33 c=12 d=00
  1: c=c+a
  2: b=b*2
  ...
  N: a=d+5
  Q: value of a after step N?
  ```
  The query register should be one written in the last ~25% of steps, so the question can't be answered by ignoring the program.
- **Trace line format (verbose, default)**: `1: c=c+a=12+07=19.` Operand values are shown so the per-step arithmetic stays trivial for a 0.5B model. Add a flag for the terse format `1: c=19.` too.
- State format: `<mem>pc=9;a=07;b=66;c=19;d=53</mem>`. `pc` = the next step to execute; registers always in alphabetical order; tags are NOT counted toward m.
- Final line: `Answer: 52`.
- Parsers for: trace lines, `<mem>` spans (take the LAST span; missing/malformed → empty state), the final answer.
- **Measure with the real Qwen2.5 tokenizer and write to results/tokenization.json**: tokens per trace line (both formats), minimal state size s_min for k=4/8/16, and the number of chunks needed for N ∈ {8, 16, 32, 64} at C ∈ {64, 128, 256}. Qwen splits digits into single tokens, so expect ~5 tokens per register.
- Tests: interpreter is correct against a hand-checked program; render→parse round-trips; resuming the interpreter from any intermediate state reproduces the same final answer (we'll use this later as an exact sufficiency check).

**Important check on chunk size**: the proposal says C=256 new tokens per chunk. If a step is ~15 tokens, a 256-token chunk covers ~17 steps, so N=8 and N=16 never cross a chunk boundary and memory is never exercised for them. Using tokenization.json, pick the default C so that N=16 needs at least 2 chunks and N=64 fits within 12 chunks with some slack (probably C=128). Make C a config value, and flag in milestone_results.md that this changed from the proposal and why.

## P0-a: Data

- Training programs are sampled on demand: N ∈ [4, 32], k uniformly mixed from {4, 8, 16} within every run (don't sweep k separately; eval slices by k).
- Dev set (small) and test set: fixed seed, 500 programs per (N, k) cell for N ∈ {8, 16, 32, 64} (N=64 is out-of-distribution length). For tonight's milestone eval, 200 per cell for N ∈ {8, 16, 32} is enough.
- Warm-up SFT sets, ~5K examples each, full-context interpreter traces (program → whole trace → `Answer: X`):
  - `warmup_plain`: no `<mem>` spans at all. Used to initialize the **full** and **tail** runs.
  - `warmup_mem`: same traces, but with a full-state `<mem>` dump inserted after a random number of steps (resampled each time, range set so dumps land about as often as chunk boundaries would). Used ONLY to initialize **learned** runs, so the 0.5B model knows the tag format. Training the tail model on dumps would pre-teach it to write state and confound tail vs learned, which is exactly what our mentor flagged.

## P0-b: Warm-up SFT

- Qwen/Qwen2.5-0.5B-Instruct, LoRA rank 64 on all linear layers, plain-text prompts (no chat template), loss on the completion only.
- Prompt format, used everywhere (SFT, env, eval):
  ```
  Program:
  <program text>
  State: <carried state, or empty>
  Trace:
  ```
- Train both checkpoints (`warmup_plain`, `warmup_mem`). Each should be well under an A100-hour; report actual wall-clock.

## P0-b: Eval (eval.py, vLLM, greedy)

At full context (mode=full, same chunk budget as the RL runs), for each checkpoint and each (N, k) cell:
- **Format compliance**: fraction of outputs where every trace line parses and an `Answer:` line exists.
- **Per-step arithmetic accuracy**: for each parsed line, is the result correct given the operand values the model itself wrote on that line? This isolates arithmetic from propagation errors.
- **Trace accuracy**: fraction of steps whose value matches the ground truth.
- **Final-answer exact match** with 95% bootstrap CIs.
- For `warmup_mem`: fraction of `<mem>` dumps that exactly match the true state.

Also run the **untrained tail baseline** (warmup_plain in tail mode, m=100). We expect it to break (H1), so a low number is fine. It's one row in the table.

Write results/warmup_eval.json and add a table to milestone_results.md.

## P0-c: Environment (env.py)

Config: `mode ∈ {full, tail, learned}`, `C`, `m ∈ {25, 50, 100, 200}`, `max_chunks=12`, `N`, `k`.

Each chunk, the policy sees the prompt (program + carried state) and generates up to C new tokens. Stop conditions:
- `Answer: X` line → episode ends.
- learned mode only: `</mem>` also ends the chunk (the model decides when to write its memory and move on, MEM1-style).
- C tokens reached.

The next state is:
- **full**: all previous chunk outputs concatenated (no reset). Same C and max_chunks budget as the other modes, so the full-context baseline is budget-matched.
- **tail**: the last m tokens of this chunk's output (token-level slice, then decode).
- **learned**: the content of the last `<mem>` span, truncated to its FIRST m tokens (head kept, since `pc` comes first). No valid span → empty state.

Reward: 1 if the final answer is exactly correct, else 0. Running out of chunks = 0.

Log per episode: chunks used, tokens generated, peak context length, and the carried state at every boundary (we compute state fidelity from it later, including for tail mode).

## P1: GRPO (grpo.py), our own implementation

- 32 programs × 8 rollouts per step. Advantages are normalized within each group of 8 and applied to every chunk of that rollout; each chunk is its own training sequence (loss only on the generated tokens). Clipped policy-gradient objective as in DeepSeekMath; no KL term is fine to start, but make it a flag.
- Rollouts with vLLM (LoRA enabled, reload the adapter each step via a new LoRARequest id), training with HF + PEFT on the same GPU. If vLLM gives trouble, fall back to HF `generate` but report how much slower it is.
- Curriculum: start at N ∈ [4, 8]; raise the max N by 4 when rolling training accuracy passes 70%, up to 32.
- Log per step: reward mean, current N range, generation time, training time, tokens generated.

## P0-c: Timing pilot (pilot_timing.py)

- Run 20 GRPO steps for each of: `learned, m=100` and `full`, at a FIXED N=32 and mixed k. That's worst-case length, so the estimate is an upper bound.
- Report seconds per step (split into generation vs training), tokens per step, and peak GPU memory.
- Estimate GPU-hours per 300-step run two ways: (a) fixed N=32, as an upper bound; (b) scaled by the curriculum, assuming a linear ramp of N. Write results/timing.json.
- If the GRPO loop isn't ready, time generation only (256 rollouts through the env at N=32) and mark it clearly as a lower bound.

## run_table.py

Build results/run_table.md from timing.json:

| Tier | Run | Init ckpt | m | Seeds | GPU-h/run | Total |
|---|---|---|---|---|---|---|
| A | full-context RL | warmup_plain | n/a | 2 | | |
| A | tail RL | warmup_plain | 25, 100 | 2 | | |
| A | learned RL | warmup_mem | 25, 100 | 2 | | |
| B | tail RL | warmup_plain | 50, 200 | 1 | | |
| B | learned RL | warmup_mem | 50, 200 | 1 | | |
| shared | warm-up SFT ×2, pilot, eval | | | | | |

Use a separate cost for full-context, since its context is longer. Include a grand total and a "lean" total (1 seed, m ∈ {25, 100}). At $X/hr (make it a parameter; the vast.ai rate goes here), show dollars too.

## milestone_results.md

Facts only, as bullets and tables. Don't write report prose; I'll write the report myself. Include:
- tokenization numbers and the chosen C (with the reason, if it changed from 256)
- the warm-up eval table + the untrained-tail number
- timing + the run table
- a "design decisions" list: k mixed within runs; head-truncation of `<mem>` with pc first and tags not counted; two warm-up checkpoints and why; full-context baseline budget-matched (same C × max_chunks, same warm-up and steps); oracle-state SFT treated as a reference point, not an upper bound (not run yet)
- anything that broke or that you had to change, and why

## Rules

- Never execute model-written text as code. The interpreter only ever runs programs we generated.
- Don't change the task design beyond what's listed here without telling me. If something in the proposal is infeasible, say so in milestone_results.md instead of quietly working around it.
- When you finish each priority level, give me a short status update: what runs, what's measured, what's next.
