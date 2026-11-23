"""GRPO for MarkovEnv (own implementation; no TRL).

Per step: P programs x G rollouts. Reward is binary final-answer correctness. Advantages are normalized
within each group of G and applied to every chunk of that rollout; each chunk is its own training sequence
(prompt = program + carried state, loss only on the generated tokens). Clipped objective (DeepSeekMath),
optional k3 KL to the warm-up adapter. Rollouts with vLLM (adapter reloaded every step via a new
LoRARequest id), training with HF + PEFT on the same GPU.

    python -m markov_rl.grpo --config markov_rl/configs/rl_learned_m100_s0.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import data as data_mod
from .common import CKPT, MODEL_NAME, RESULTS, ROOT, load_config, pick_device, run_meta, seed_everything, write_json
from .env import EnvConfig, MarkovEnv, run_episodes

DEFAULTS = {
    "name": "rl_debug",
    "model": MODEL_NAME,
    "init_adapter": "checkpoints/warmup_mem",
    "mode": "learned",
    "C": None,  # None -> chosen_C from results/tokenization.json
    "m": 100,
    "max_chunks": 12,
    "steps": 300,
    "programs_per_step": 32,
    "group_size": 8,
    "ks": [4, 8, 16],
    # curriculum: N ~ U[n_lo, n_hi]; n_hi += n_inc when rolling acc > threshold (up to n_max)
    "n_lo": 4, "n_hi": 8, "n_max": 32, "n_inc": 4, "curriculum_threshold": 0.7, "curriculum_window": 5,
    "fixed_N": None,  # e.g. 32 for the timing pilot (disables curriculum)
    # optimization
    "lr": 2.0e-5,
    "clip_eps": 0.2,
    "kl_coef": 0.0,  # >0 enables k3 KL to the init adapter
    "loss_agg": "seq_mean",  # seq_mean (DeepSeekMath) | token_mean
    "ppo_epochs": 1,
    "num_minibatches": 1,  # optimizer steps per epoch; 1 + 1 epoch => pi_old == pi_theta (ratio = 1)
    "max_grad_norm": 1.0,
    "micro_batch_tokens": 24576,  # total (prompt+gen) tokens per micro-batch
    "micro_batch_gen_tokens": 4096,  # generated tokens per micro-batch (bounds the fp32 logits)
    "grad_checkpointing": True,
    "train_on_zero_adv": False,  # pilot: True so training time is measured even if all rewards are equal
    "gen_only": False,  # pilot fallback: skip training (lower bound)
    "temperature": 1.0,
    "top_p": 1.0,
    # infra
    "backend": "vllm",
    "gpu_mem": 0.35,
    "max_model_len": 4096,
    "save_every": 50,
    "seed": 0,
}


# ---------------------------------------------------------------------------
# Model / logprobs
# ---------------------------------------------------------------------------

def load_policy(cfg: dict, device: str):
    from peft import PeftModel

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=dtype, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(ROOT / cfg["init_adapter"]), is_trainable=True)
    if cfg["kl_coef"] > 0:
        model.load_adapter(str(ROOT / cfg["init_adapter"]), adapter_name="ref", is_trainable=False)
        model.set_adapter("default")
    for mod in model.modules():  # deterministic policy for logprobs: no LoRA dropout
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if cfg["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device)


def trainable_params(model):
    return [p for n, p in model.named_parameters() if p.requires_grad and ".ref." not in n]


def micro_batches(seqs: list[dict], max_tokens: int, max_gen: int) -> list[list[int]]:
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["prompt_ids"]) + len(seqs[i]["gen_ids"]))
    out, cur, cur_T, cur_g = [], [], 0, 0
    for i in order:
        L = len(seqs[i]["prompt_ids"]) + len(seqs[i]["gen_ids"])
        g = len(seqs[i]["gen_ids"])
        T = max(cur_T, L)
        if cur and (T * (len(cur) + 1) > max_tokens or cur_g + g > max_gen):
            out.append(cur)
            cur, T, cur_g = [], L, 0
        cur.append(i)
        cur_T, cur_g = T, cur_g + g
    if cur:
        out.append(cur)
    return out


def gen_token_logprobs(model, seqs: list[dict], device: str) -> torch.Tensor:
    """Log-probs of the generated tokens, concatenated in sequence order -> [sum(len(gen))]."""
    T = max(len(s["prompt_ids"]) + len(s["gen_ids"]) for s in seqs)
    inp = torch.zeros((len(seqs), T), dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    pm = torch.zeros((len(seqs), T), dtype=torch.bool)  # position t predicts a generated token t+1
    for b, s in enumerate(seqs):
        ids = s["prompt_ids"] + s["gen_ids"]
        inp[b, :len(ids)] = torch.tensor(ids)
        att[b, :len(ids)] = 1
        pm[b, len(s["prompt_ids"]) - 1: len(ids) - 1] = True
    inp, att, pm = inp.to(device), att.to(device), pm.to(device)
    causal_lm = model.get_base_model()  # LoRA layers are injected in place
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with amp:
        h = causal_lm.model(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state
        hs = h[pm]
        logits = causal_lm.lm_head(hs).float()
    tgt = inp[:, 1:][pm[:, :-1]]
    return logits.gather(-1, tgt[:, None]).squeeze(-1) - torch.logsumexp(logits, dim=-1)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

class Curriculum:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.n_hi = cfg["n_hi"]
        self.window: list[float] = []

    def range(self) -> tuple[int, int]:
        if self.cfg["fixed_N"]:
            return self.cfg["fixed_N"], self.cfg["fixed_N"]
        return self.cfg["n_lo"], self.n_hi

    def update(self, acc: float) -> bool:
        if self.cfg["fixed_N"]:
            return False
        self.window = (self.window + [acc])[-self.cfg["curriculum_window"]:]
        if (len(self.window) == self.cfg["curriculum_window"] and np.mean(self.window) > self.cfg["curriculum_threshold"]
                and self.n_hi < self.cfg["n_max"]):
            self.n_hi = min(self.cfg["n_max"], self.n_hi + self.cfg["n_inc"])
            self.window = []
            return True
        return False


def collect(cfg, env_cfg, tok, gen, rng, curriculum) -> tuple[list[MarkovEnv], dict]:
    lo, hi = curriculum.range()
    envs = []
    for p in range(cfg["programs_per_step"]):
        prog = data_mod.sample_train_program(rng, lo, hi, ks=tuple(cfg["ks"]))
        for _ in range(cfg["group_size"]):
            e = MarkovEnv(env_cfg, tok)
            e.reset(prog, meta={"group": p})
            envs.append(e)
    t = run_episodes(envs, gen, temperature=cfg["temperature"], top_p=cfg["top_p"])
    return envs, t


def build_sequences(cfg, envs: list[MarkovEnv]) -> tuple[list[dict], dict]:
    G = cfg["group_size"]
    seqs, n_signal = [], 0
    for g in range(0, len(envs), G):
        grp = envs[g: g + G]
        r = np.array([e.reward for e in grp])
        adv = (r - r.mean()) / (r.std() + 1e-4)
        n_signal += float(r.std() > 0)
        for e, a in zip(grp, adv):
            if a == 0 and not cfg["train_on_zero_adv"]:
                continue
            for c in e.chunks:
                if c["gen_ids"]:
                    seqs.append({"prompt_ids": c["prompt_ids"], "gen_ids": c["gen_ids"], "adv": float(a)})
    return seqs, {"frac_groups_with_signal": n_signal / (len(envs) // G)}


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def grpo_update(cfg, model, opt, seqs: list[dict], device: str) -> dict:
    if not seqs:
        return {"loss": 0.0, "grad_norm": 0.0, "n_seqs": 0, "n_tokens": 0, "clip_frac": 0.0, "kl": 0.0}
    params = trainable_params(model)
    mbs_all = micro_batches(seqs, cfg["micro_batch_tokens"], cfg["micro_batch_gen_tokens"])
    need_old = cfg["ppo_epochs"] > 1 or cfg["num_minibatches"] > 1
    if need_old or cfg["kl_coef"] > 0:
        with torch.no_grad():
            for mb in mbs_all:
                batch = [seqs[i] for i in mb]
                if need_old:
                    lp = gen_token_logprobs(model, batch, device).split([len(s["gen_ids"]) for s in batch])
                    for s, l in zip(batch, lp):
                        s["old_logp"] = l.detach()
                if cfg["kl_coef"] > 0:
                    model.set_adapter("ref")
                    lp = gen_token_logprobs(model, batch, device).split([len(s["gen_ids"]) for s in batch])
                    model.set_adapter("default")
                    for s, l in zip(batch, lp):
                        s["ref_logp"] = l.detach()
    stats = {"loss": 0.0, "clip_frac": 0.0, "kl": 0.0, "grad_norm": 0.0}
    n_tok_total = sum(len(s["gen_ids"]) for s in seqs)
    eps = cfg["clip_eps"]
    no_signal = cfg["kl_coef"] == 0 and all(s["adv"] == 0 for s in seqs)
    n_opt = 0
    for _ in range(cfg["ppo_epochs"]):
        idx = list(range(len(seqs)))
        random.shuffle(idx)
        parts = [idx[i:: cfg["num_minibatches"]] for i in range(cfg["num_minibatches"])]
        for part in parts:
            sub = [seqs[i] for i in part]
            n_seq, n_tok = len(sub), sum(len(s["gen_ids"]) for s in sub)
            for mb in micro_batches(sub, cfg["micro_batch_tokens"], cfg["micro_batch_gen_tokens"]):
                batch = [sub[i] for i in mb]
                lens = [len(s["gen_ids"]) for s in batch]
                logp = gen_token_logprobs(model, batch, device)
                old = torch.cat([s["old_logp"] for s in batch]) if need_old else logp.detach()
                adv = torch.cat([torch.full((l,), s["adv"], device=logp.device) for s, l in zip(batch, lens)])
                ratio = torch.exp(logp - old)
                pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv)
                if cfg["kl_coef"] > 0:
                    ref = torch.cat([s["ref_logp"] for s in batch])
                    d = ref - logp
                    kl = torch.exp(d) - d - 1
                    pg = pg + cfg["kl_coef"] * kl
                    stats["kl"] += kl.sum().item() / n_tok_total
                if cfg["loss_agg"] == "seq_mean":
                    loss = torch.stack([x.mean() for x in pg.split(lens)]).sum() / n_seq
                else:
                    loss = pg.sum() / n_tok
                loss.backward()
                stats["loss"] += loss.item()
                stats["clip_frac"] += ((ratio - 1).abs() > eps).float().sum().item() / n_tok_total
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            if no_signal:  # pilot timing path: compute the full backward but don't let Adam
                opt.zero_grad(set_to_none=True)  # amplify numerical-noise gradients into a real step
                continue
            opt.step()
            opt.zero_grad(set_to_none=True)
            stats["grad_norm"] += gn.item()
            n_opt += 1
    stats["grad_norm"] /= max(1, n_opt)
    stats.update(n_seqs=len(seqs), n_tokens=n_tok_total, n_micro_batches=len(mbs_all))
    return stats


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> dict:
    t_init = time.time()
    seed_everything(cfg["seed"])
    if cfg["C"] is None:
        cfg["C"] = json.loads((RESULTS / "tokenization.json").read_text())["chosen_C"]
    env_cfg = EnvConfig(mode=cfg["mode"], C=cfg["C"], m=cfg["m"], max_chunks=cfg["max_chunks"])
    out_dir = RESULTS / "grpo" / cfg["name"]
    ck_dir = CKPT / cfg["name"]
    sync_dir = ck_dir / "_sync"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    device = pick_device()

    # vLLM first (before HF touches CUDA), then the HF policy.
    from .gen import HFGenerator, VLLMGenerator

    gen = None
    if cfg["backend"] == "vllm":
        gen = VLLMGenerator(cfg["model"], max_model_len=cfg["max_model_len"], gpu_memory_utilization=cfg["gpu_mem"],
                            seed=cfg["seed"])
        gen.set_lora(ROOT / cfg["init_adapter"])
    model = load_policy(cfg, device)
    if gen is None:
        gen = HFGenerator(tok, model=model, device=device, max_model_len=cfg["max_model_len"])
    opt = torch.optim.AdamW(trainable_params(model), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=0.0)

    meta = run_meta(cfg)
    write_json(out_dir / "config.json", meta)
    log_f = open(out_dir / "log.jsonl", "a")
    rng = random.Random(cfg["seed"] + 12345)
    random.seed(cfg["seed"])
    cur = Curriculum(cfg)
    t_start = time.time()
    for step in range(1, cfg["steps"] + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        n_range = cur.range()
        envs, gt = collect(cfg, env_cfg, tok, gen, rng, cur)
        rewards = np.array([e.reward for e in envs])
        seqs, sstats = build_sequences(cfg, envs)
        t1 = time.time()
        if cfg["gen_only"]:
            ustats = {"n_seqs": len(seqs), "n_tokens": sum(len(s["gen_ids"]) for s in seqs)}
        else:
            model.train()
            ustats = grpo_update(cfg, model, opt, seqs, device)
        train_time = time.time() - t1
        t2 = time.time()
        if not cfg["gen_only"] and cfg["backend"] == "vllm":
            path = sync_dir / f"step{step}"
            model.save_pretrained(path, selected_adapters=["default"])
            gen.set_lora(path)
            old = sync_dir / f"step{step - 2}"
            if old.exists():
                shutil.rmtree(old)
        sync_time = time.time() - t2
        raised = cur.update(float(rewards.mean()))
        acc_k = {f"acc_k{k}": float(np.mean([e.reward for e in envs if e.prog.k == k] or [float("nan")]))
                 for k in cfg["ks"]}
        rec = {
            "step": step,
            "reward_mean": float(rewards.mean()),
            **acc_k,
            "N_range": list(n_range),
            "curriculum_raised": raised,
            "gen_time_s": gt["gen_time_s"],
            "train_time_s": train_time,
            "sync_time_s": sync_time,
            "step_time_s": gt["gen_time_s"] + train_time + sync_time,
            "rounds": gt["rounds"],
            "tokens_generated": int(sum(sum(len(c["gen_ids"]) for c in e.chunks) for e in envs)),
            "mean_chunks": float(np.mean([len(e.chunks) for e in envs])),
            "peak_context": int(max(c["prompt_len"] + len(c["gen_ids"]) for e in envs for c in e.chunks)),
            "answered_frac": float(np.mean([e.pred is not None for e in envs])),
            **sstats,
            **{f"train_{k}": v for k, v in ustats.items()},
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
            # device-wide (includes the vLLM engine process and its KV-cache reservation)
            "gpu_used_total_gb": ((lambda f, t: (t - f) / 1e9)(*torch.cuda.mem_get_info())
                                  if device == "cuda" else None),
            "elapsed_s": time.time() - t_start,
        }
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if step == 1:  # one sample rollout for eyeballing
            (out_dir / "sample_rollout.json").write_text(json.dumps(envs[0].episode_log(), indent=1))
        if not cfg["gen_only"] and (step % cfg["save_every"] == 0 or step == cfg["steps"]):
            model.save_pretrained(ck_dir / f"step{step}", selected_adapters=["default"])
    log_f.close()
    summary = {**meta, "wall_clock_s": time.time() - t_init, "startup_s": t_start - t_init, "steps": cfg["steps"]}
    write_json(out_dir / "summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set, DEFAULTS)
    train(cfg)


if __name__ == "__main__":
    main()
