"""Generation backends behind one interface: vLLM (GPU, default) and HF `generate` (fallback / CPU tests).

generate(prompts=[token ids], max_tokens, stop=[str], temperature) -> [GenOut]
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# vLLM's engine runs in a child process. Forking after CUDA is initialized in the parent (HF policy,
# torch.cuda calls) fails with "Cannot re-initialize CUDA in forked subprocess", so always spawn.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch


@dataclass
class GenOut:
    text: str
    token_ids: list[int]
    prompt_len: int
    finish_reason: str  # "stop" (EOS or stop string) | "length"


class VLLMGenerator:
    def __init__(self, model: str, max_model_len: int = 4096, gpu_memory_utilization: float = 0.35,
                 seed: int = 0, max_lora_rank: int = 64, enable_lora: bool = True):
        from vllm import LLM

        self.llm = LLM(model=model, dtype="bfloat16", seed=seed, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True,
                       enable_lora=enable_lora, max_lora_rank=max_lora_rank, max_loras=1)
        self.max_model_len = max_model_len
        self.lora = None
        self._lora_id = 0

    def set_lora(self, path) -> None:
        """Point generation at a (new) adapter dir. A fresh id forces vLLM to reload from disk."""
        from vllm.lora.request import LoRARequest

        if path is None:
            self.lora = None
            return
        self._lora_id += 1
        self.lora = LoRARequest(f"adapter{self._lora_id}", self._lora_id, str(path))

    def generate(self, prompts: list[list[int]], max_tokens: int, stop: list[str] | None = None,
                 temperature: float = 0.0, top_p: float = 1.0) -> list[GenOut]:
        from vllm import SamplingParams

        # Respect the context limit (only matters for long full-context prompts).
        groups: dict[int, list[int]] = {}
        for i, p in enumerate(prompts):
            groups.setdefault(max(1, min(max_tokens, self.max_model_len - len(p))), []).append(i)
        results: list[GenOut | None] = [None] * len(prompts)
        for mt, idx in groups.items():
            sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=mt, stop=stop or None,
                                include_stop_str_in_output=True, skip_special_tokens=True)
            outs = self.llm.generate([{"prompt_token_ids": prompts[i]} for i in idx], sp,
                                     lora_request=self.lora, use_tqdm=False)
            for i, o in zip(idx, outs):
                c = o.outputs[0]
                results[i] = GenOut(c.text, list(c.token_ids), len(prompts[i]),
                                    "length" if c.finish_reason == "length" else "stop")
        return results


class HFGenerator:
    """Slow fallback. Pass `model` to share weights with a training PeftModel (GRPO without vLLM)."""

    def __init__(self, tokenizer, model=None, model_name: str | None = None, lora_path=None,
                 device: str = "cpu", micro_batch: int = 32, max_model_len: int = 4096):
        self.tok = tokenizer
        self.device = device
        self.micro_batch = micro_batch
        self.max_model_len = max_model_len
        if model is None:
            from transformers import AutoModelForCausalLM

            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
            if lora_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(lora_path)).to(device)
        self.model = model
        self.eos_ids = sorted({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")})
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def set_lora(self, path) -> None:
        if path is None:
            return
        from peft import PeftModel, set_peft_model_state_dict
        from peft.utils import load_peft_weights

        if isinstance(self.model, PeftModel):
            set_peft_model_state_dict(self.model, load_peft_weights(str(path), device=self.device))
        else:
            self.model = PeftModel.from_pretrained(self.model, str(path)).to(self.device)

    def _trim(self, ids: list[int], stop: list[str] | None) -> tuple[list[int], str]:
        for j, t in enumerate(ids):
            if t in self.eos_ids:
                return ids[: j + 1], "stop"
        if stop:
            text = self.tok.decode(ids, skip_special_tokens=True)
            if any(s in text for s in stop):
                for j in range(1, len(ids) + 1):
                    if any(s in self.tok.decode(ids[:j], skip_special_tokens=True) for s in stop):
                        return ids[:j], "stop"
        return ids, None

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_tokens: int, stop: list[str] | None = None,
                 temperature: float = 0.0, top_p: float = 1.0) -> list[GenOut]:
        was_training = self.model.training
        self.model.eval()
        out = []
        for s in range(0, len(prompts), self.micro_batch):
            batch = prompts[s: s + self.micro_batch]
            T = max(len(p) for p in batch)
            inp = torch.full((len(batch), T), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), T), dtype=torch.long)
            for b, p in enumerate(batch):  # left padding
                inp[b, T - len(p):] = torch.tensor(p)
                att[b, T - len(p):] = 1
            mt = max(1, min(max_tokens, self.max_model_len - T))
            kw = dict(do_sample=temperature > 0, max_new_tokens=mt, pad_token_id=self.pad_id,
                      eos_token_id=self.eos_ids)
            if temperature > 0:
                kw.update(temperature=temperature, top_p=top_p, top_k=0)
            else:
                kw.update(temperature=None, top_p=None, top_k=None)
            if stop:
                kw.update(stop_strings=stop, tokenizer=self.tok)
            gen = self.model.generate(input_ids=inp.to(self.device), attention_mask=att.to(self.device), **kw)
            for b, p in enumerate(batch):
                ids = gen[b, T:].tolist()
                ids, reason = self._trim(ids, stop)
                while ids and ids[-1] == self.pad_id and self.pad_id not in self.eos_ids:
                    ids.pop()
                reason = reason or ("length" if len(ids) >= mt else "stop")
                out.append(GenOut(self.tok.decode(ids, skip_special_tokens=True), ids, len(p), reason))
        if was_training:
            self.model.train()
        return out


def make_generator(backend: str, tokenizer, model_name: str, lora_path=None, device: str = "cuda",
                   max_model_len: int = 4096, gpu_memory_utilization: float = 0.35, seed: int = 0, model=None):
    if backend == "vllm":
        g = VLLMGenerator(model_name, max_model_len=max_model_len,
                          gpu_memory_utilization=gpu_memory_utilization, seed=seed)
        g.set_lora(lora_path)
        return g
    if backend == "hf":
        return HFGenerator(tokenizer, model=model, model_name=model_name, lora_path=lora_path, device=device,
                           max_model_len=max_model_len)
    raise ValueError(backend)
