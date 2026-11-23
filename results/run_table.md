# Run table (GPU-hours on 1x A100 80GB; $0.82/GPU-h)

GPU-h/run: upper = 300 steps at fixed N=32; curriculum = scaled by a linear N ramp (see timing.json).

| Tier | Run | Init ckpt | m | Seeds | # runs | GPU-h/run (upper) | GPU-h/run (curriculum) | Total (upper) | Total (curriculum) |
|---|---|---|---|---|---|---|---|---|---|
| A | full-context RL | warmup_plain | n/a | 2 | 2 | 5.38 | 2.04 | 10.8 | 4.1 |
| A | tail RL | warmup_plain | 25, 100 | 2 | 4 | 12.12 | 4.57 | 48.5 | 18.3 |
| A | learned RL | warmup_mem | 25, 100 | 2 | 4 | 12.12 | 4.57 | 48.5 | 18.3 |
| B | tail RL | warmup_plain | 50, 200 | 1 | 2 | 12.12 | 4.57 | 24.2 | 9.1 |
| B | learned RL | warmup_mem | 50, 200 | 1 | 2 | 12.12 | 4.57 | 24.2 | 9.1 |
| shared | warm-up SFT x2 (measured 0.18), pilot (measured 1.24), warm-up eval (measured 0.12), final test eval of 14 runs (est. 2.8) | | | | | | | 4.4 | 4.4 |

| Plan | RL runs | GPU-h (upper) | GPU-h (curriculum) | $ (upper) | $ (curriculum) |
|---|---|---|---|---|---|
| Grand total | 14 | 160.6 | 63.2 | $131 | $52 |
| Lean (1 seed, m in {25, 100} + full) | 5 | 56.4 | 22.9 | $46 | $19 |

- k is mixed within every run (k ~ U{4, 8, 16} per program); eval slices by k. No separate k sweep.
- Proposal said '6 main runs + 4 sweep runs'; the actual plan is 14 RL runs (2 seeds x [full + tail x2 m + learned x2 m] + 1 seed x [tail, learned] x m in {50, 200}).
- tail cost taken from the learned pilot (no tail pilot run).
