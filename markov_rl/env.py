"""MarkovEnv: chunked reasoning with a bounded carried state.

OpenEnv-style API: reset(program) -> obs; step(action) -> (obs, reward, done, info); state() -> dict.

Each chunk the policy sees `render_prompt(program, carried_state)` and writes up to C new tokens.
Next carried state by mode:
  full    : no reset. All previous chunk outputs are kept, appended after `Trace:` (exactly the SFT
            format), under the same C x max_chunks budget as the other modes.
  tail    : last m tokens of this chunk's output (token-level slice, then decode).
  learned : content of the LAST <mem>...</mem> span of this chunk, truncated to its FIRST m tokens
            (head kept; pc comes first; tags not counted). No valid span -> empty state.
Chunk ends at EOS / C tokens; in learned mode also at `</mem>`. An `Answer:` line ends the episode.
Reward: 1 if the final answer is exactly right, else 0 (out of chunks -> 0).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import interp
from .gen import GenOut

MODES = ("full", "tail", "learned")


@dataclass
class EnvConfig:
    mode: str = "learned"
    C: int = 192
    m: int = 100
    max_chunks: int = 12

    def __post_init__(self):
        assert self.mode in MODES, self.mode

    @property
    def stop(self) -> list[str]:
        return ["</mem>"] if self.mode == "learned" else []


class MarkovEnv:
    def __init__(self, cfg: EnvConfig, tokenizer):
        self.cfg = cfg
        self.tok = tokenizer
        self.prog: interp.Program | None = None

    # -- OpenEnv API ---------------------------------------------------------
    def reset(self, program: interp.Program | dict, meta: dict | None = None) -> dict:
        if isinstance(program, dict):
            program = interp.Program.from_dict(program.get("program", program))
        self.prog = program
        self.meta = meta or {}
        self.program_text = interp.render_program(program)
        self.answer = interp.answer_from(program)
        self.base_ids = self._enc(interp.render_prompt(self.program_text))
        self.carried = ""  # state text shown in the prompt (tail/learned)
        self.chunks: list[dict] = []
        self.done = False
        self.reward = 0.0
        self.pred = None
        self._obs = self._make_obs()
        return self._obs

    def step(self, action: GenOut | str) -> tuple[dict | None, float, bool, dict]:
        assert self.prog is not None and not self.done, "call reset() first"
        if isinstance(action, str):  # convenience for manual use
            ids = self._enc(action)
            action = GenOut(action, ids, len(self._obs["prompt_ids"]),
                            "length" if len(ids) >= self.cfg.C else "stop")
        ids = list(action.token_ids)[: self.cfg.C]
        text = action.text
        state_before = self.carried
        self.chunks.append({
            "prompt_ids": self._obs["prompt_ids"],
            "gen_ids": ids,
            "text": text,
            "finish_reason": action.finish_reason,
            "state_before": state_before,
            "prompt_len": len(self._obs["prompt_ids"]),
        })
        pred = interp.parse_answer(text)
        if pred is not None:
            self.done, self.pred = True, pred
            self.reward = float(pred == self.answer)
        else:
            self.carried = self._next_state(text, ids)
            self.chunks[-1]["state_after"] = self.carried
            if len(self.chunks) >= self.cfg.max_chunks:
                self.done, self.reward = True, 0.0
        self._obs = None if self.done else self._make_obs()
        info = {"chunk": len(self.chunks), "answer": self.answer, "pred": self.pred}
        return self._obs, self.reward, self.done, info

    def state(self) -> dict:
        return {
            "config": asdict(self.cfg),
            "program_text": self.program_text if self.prog else None,
            "answer": self.answer if self.prog else None,
            "carried_state": self.carried if self.prog else None,
            "chunk_index": len(self.chunks) if self.prog else 0,
            "done": self.done if self.prog else False,
            "reward": self.reward if self.prog else 0.0,
        }

    # -- internals -------------------------------------------------------------
    @property
    def obs(self) -> dict | None:
        return self._obs

    def _enc(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def _make_obs(self) -> dict:
        if self.cfg.mode == "full":
            ids = self.base_ids + [t for c in self.chunks for t in c["gen_ids"] if t not in self._special_ids]
            prompt = None  # decode lazily if needed
        else:
            prompt = interp.render_prompt(self.program_text, self.carried)
            ids = self._enc(prompt)
        return {"prompt": prompt, "prompt_ids": ids, "chunk": len(self.chunks), "stop": self.cfg.stop}

    @property
    def _special_ids(self) -> set[int]:
        return set(self.tok.all_special_ids)

    def _next_state(self, text: str, ids: list[int]) -> str:
        m = self.cfg.m
        if self.cfg.mode == "full":
            return ""  # context kept via self.chunks
        if self.cfg.mode == "tail":
            ids = [t for t in ids if t not in self._special_ids]
            return self.tok.decode(ids[-m:]) if m > 0 else ""
        span = interp.last_mem_span(text)
        if span is None:
            return ""
        return self.tok.decode(self._enc(span)[:m])

    # -- logging ---------------------------------------------------------------
    def episode_log(self, with_ids: bool = False) -> dict:
        gen_tokens = sum(len(c["gen_ids"]) for c in self.chunks)
        log = {
            **self.meta,
            "N": self.prog.N, "k": self.prog.k, "mode": self.cfg.mode, "m": self.cfg.m, "C": self.cfg.C,
            "answer": self.answer, "pred": self.pred, "reward": self.reward,
            "chunks_used": len(self.chunks),
            "tokens_generated": gen_tokens,
            "peak_context": max((c["prompt_len"] + len(c["gen_ids"]) for c in self.chunks), default=0),
            "states": [c.get("state_after") for c in self.chunks[:-1]] if self.cfg.mode != "full" else [],
            "chunk_texts": [c["text"] for c in self.chunks],
            "finish_reasons": [c["finish_reason"] for c in self.chunks],
        }
        if with_ids:
            log["chunk_ids"] = [(c["prompt_ids"], c["gen_ids"]) for c in self.chunks]
        return log


def run_episodes(envs: list[MarkovEnv], generator, temperature: float = 0.0, top_p: float = 1.0) -> dict:
    """Step all envs chunk-synchronously, batching every active env into one generate call per round."""
    import time

    cfg = envs[0].cfg
    assert all(e.cfg == cfg for e in envs), "one EnvConfig per batch"
    t0, rounds = time.time(), 0
    active = [e for e in envs if not e.done]
    while active:
        outs = generator.generate([e.obs["prompt_ids"] for e in active], max_tokens=cfg.C, stop=cfg.stop,
                                  temperature=temperature, top_p=top_p)
        for e, o in zip(active, outs):
            e.step(o)
        active = [e for e in active if not e.done]
        rounds += 1
    return {"gen_time_s": time.time() - t0, "rounds": rounds}
