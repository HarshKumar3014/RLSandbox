"""Warm-up SFT: LoRA on Qwen2.5-0.5B-Instruct, plain-text prompts, loss on completion only.

    python -m markov_rl.sft --config markov_rl/configs/sft_plain.yaml
"""
from __future__ import annotations

import argparse
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import CKPT, MODEL_NAME, RESULTS, ROOT, load_config, pick_device, read_jsonl, run_meta, seed_everything, write_json

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULTS = {
    "name": "warmup_plain",
    "data": "data/warmup_plain.jsonl",
    "model": MODEL_NAME,
    "epochs": 3,
    "lr": 2e-4,
    "batch_size": 16,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "max_len": 1536,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "seed": 0,
    "max_steps": None,  # for smoke tests
    "n_examples": None,
    "log_every": 10,
}


def make_lora_model(model_name: str, cfg: dict, device: str):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="sdpa")
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():  # LoRA weights in fp32, base frozen in bf16
        if p.requires_grad:
            p.data = p.data.float()
    return model.to(device)


def encode(tok, rows: list[dict], max_len: int) -> list[tuple[list[int], int]]:
    """-> list of (ids, n_prompt_tokens). Prompt and completion tokenized separately (the prompt ends with
    a newline, which is a pre-tokenizer boundary for Qwen, so this matches joint tokenization)."""
    out, n_trunc = [], 0
    for r in rows:
        p = tok.encode(r["prompt"], add_special_tokens=False)
        c = tok.encode(r["completion"], add_special_tokens=False) + [tok.eos_token_id]
        ids = p + c
        if len(ids) > max_len:
            n_trunc += 1
            ids = ids[:max_len]
        out.append((ids, len(p)))
    if n_trunc:
        print(f"warning: truncated {n_trunc} examples to {max_len} tokens")
    return out


def batches(examples, batch_size: int, rng: random.Random):
    """Length-bucketed batches: sort within shuffled mega-chunks to cut padding."""
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    mega = batch_size * 50
    out = []
    for i in range(0, len(idx), mega):
        chunk = sorted(idx[i:i + mega], key=lambda j: len(examples[j][0]))
        out += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng.shuffle(out)
    return out


def collate(examples, ids_list, pad_id: int, device: str):
    T = max(len(examples[i][0]) for i in ids_list)
    inp = torch.full((len(ids_list), T), pad_id, dtype=torch.long)
    lab = torch.full((len(ids_list), T), -100, dtype=torch.long)
    att = torch.zeros((len(ids_list), T), dtype=torch.long)
    for b, i in enumerate(ids_list):
        ids, n_p = examples[i]
        inp[b, :len(ids)] = torch.tensor(ids)
        lab[b, n_p:len(ids)] = torch.tensor(ids[n_p:])
        att[b, :len(ids)] = 1
    return inp.to(device), att.to(device), lab.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set, DEFAULTS)
    seed_everything(cfg["seed"])
    device = pick_device()
    out_dir = CKPT / cfg["name"]
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    rows = read_jsonl(ROOT / cfg["data"])
    if cfg["n_examples"]:
        rows = rows[: cfg["n_examples"]]
    examples = encode(tok, rows, cfg["max_len"])
    n_tok = sum(len(e[0]) - e[1] for e in examples)
    print(f"{len(examples)} examples, {n_tok} completion tokens, device={device}")

    model = make_lora_model(cfg["model"], cfg, device)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.999))
    rng = random.Random(cfg["seed"])
    steps_per_epoch = math.ceil(len(examples) / cfg["batch_size"])
    total = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_ratio"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    model.train()
    t0, step, log = time.time(), 0, []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    done = False
    for epoch in range(cfg["epochs"]):
        for b in batches(examples, cfg["batch_size"], rng):
            inp, att, lab = collate(examples, b, pad, device)
            with amp:
                loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg["log_every"] == 0 or step == 1:
                rec = {"step": step, "epoch": epoch, "loss": loss.item(), "grad_norm": gn.item(),
                       "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
                log.append(rec)
                print(rec, flush=True)
            if step >= total:
                done = True
                break
        if done:
            break
    wall = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    res = {
        **run_meta(cfg),
        "checkpoint": str(out_dir),
        "steps": step,
        "n_examples": len(examples),
        "completion_tokens_per_epoch": n_tok,
        "wall_clock_s": wall,
        "gpu_hours": wall / 3600 if device == "cuda" else None,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "final_loss": log[-1]["loss"] if log else None,
        "loss_log": log,
    }
    write_json(RESULTS / f"sft_{cfg['name']}.json", res)
    print(f"saved {out_dir}; wall-clock {wall / 60:.1f} min")


if __name__ == "__main__":
    main()
