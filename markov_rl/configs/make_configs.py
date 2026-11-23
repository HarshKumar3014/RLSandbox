"""Regenerate one yaml per run: python markov_rl/configs/make_configs.py"""
from pathlib import Path

HERE = Path(__file__).parent
INIT = {"full": "checkpoints/warmup_plain", "tail": "checkpoints/warmup_plain", "learned": "checkpoints/warmup_mem"}
RUNS = []  # (tier, mode, m, seed)
for s in (0, 1):
    RUNS.append(("A", "full", 0, s))
    for mode in ("tail", "learned"):
        for m in (25, 100):
            RUNS.append(("A", mode, m, s))
for mode in ("tail", "learned"):
    for m in (50, 200):
        RUNS.append(("B", mode, m, 0))


def write(name, body):
    (HERE / f"{name}.yaml").write_text(body)


for tier, mode, m, s in RUNS:
    name = f"rl_{mode}_s{s}" if mode == "full" else f"rl_{mode}_m{m}_s{s}"
    write(name, f"# Tier {tier}\nbase: rl_base.yaml\nname: {name}\nmode: {mode}\nm: {m}\nseed: {s}\n"
                f"init_adapter: {INIT[mode]}\n")

for mode, m in (("learned", 100), ("full", 0), ("tail", 100)):
    name = f"pilot_{mode}_m{m}" if mode != "full" else "pilot_full"
    write(name, f"# Timing pilot: fixed N=32 (upper bound), mixed k.\nbase: rl_base.yaml\nname: {name}\nmode: {mode}\n"
                f"m: {m}\nseed: 0\ninit_adapter: {INIT[mode]}\nsteps: 20\nfixed_N: 32\ntrain_on_zero_adv: true\n"
                f"save_every: 100000\n")
print(f"{len(RUNS)} RL configs + 3 pilot configs written")
