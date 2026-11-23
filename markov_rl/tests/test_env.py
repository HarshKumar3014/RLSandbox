"""Env + metrics tests with the real tokenizer and scripted (oracle) actions; no model needed."""
import random

import pytest

from markov_rl import interp
from markov_rl.env import EnvConfig, MarkovEnv
from markov_rl.eval import episode_metrics, state_fidelity
from markov_rl.gen import GenOut


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def act(tok, text, C, eos=False):
    ids = tok.encode(text, add_special_tokens=False)[:C]
    if eos:
        ids = ids + [tok.eos_token_id]
    return GenOut(tok.decode(ids, skip_special_tokens=True), ids, 0, "stop" if eos else "length")


def oracle_chunks(prog, lines_per_chunk, with_mem):
    """Correct trace split into chunks; learned-style chunks end with a <mem> dump."""
    st = interp.states(prog)
    regs = interp.initial_regs(prog)
    lines = []
    for i, s in enumerate(prog.steps, 1):
        lines.append(interp.render_trace_line(i, s, regs))
        regs[s.dst], _ = interp.apply_step(s, regs)
    chunks = []
    for c in range(0, prog.N, lines_per_chunk):
        body = "\n".join(lines[c:c + lines_per_chunk]) + "\n"
        end = min(c + lines_per_chunk, prog.N)
        if end == prog.N:
            body += interp.render_answer(st[-1][prog.query])
        elif with_mem:
            body += interp.render_state(end + 1, st[end], prog.k)
        chunks.append(body)
    return chunks


def test_learned_oracle_gets_reward(tok):
    rng = random.Random(0)
    prog = interp.sample_program(rng, 20, 8)
    env = MarkovEnv(EnvConfig(mode="learned", C=192, m=100), tok)
    obs = env.reset(prog)
    assert obs["stop"] == ["</mem>"]
    assert "State:\nTrace:\n" in tok.decode(obs["prompt_ids"])
    for text in oracle_chunks(prog, 5, with_mem=True):
        obs, r, done, info = env.step(act(tok, text, 192, eos=text.rstrip().startswith("Answer") or "Answer" in text))
        if not done:
            assert obs["prompt"].split("State: ")[1].startswith("pc=")
            assert "<mem>" not in env.carried  # tags stripped
    assert done and r == 1.0
    log = env.episode_log()
    assert log["chunks_used"] == 4 and len(log["states"]) == 3
    m = episode_metrics(log, prog)
    assert m["correct"] == 1.0 and m["format_ok"] == 1.0 and m["trace_acc"] == 1.0 and m["arith_ok"] == m["arith_n"]
    assert m["state_fidelity"] == 1.0


def test_learned_head_truncation_keeps_pc(tok):
    rng = random.Random(1)
    prog = interp.sample_program(rng, 10, 16)
    env = MarkovEnv(EnvConfig(mode="learned", C=192, m=25), tok)
    env.reset(prog)
    st = interp.states(prog)
    text = interp.render_trace_line(1, prog.steps[0], interp.initial_regs(prog)) + "\n" + interp.render_state(2, st[1], 16)
    env.step(act(tok, text, 192))
    assert env.carried.startswith("pc=2;a=")
    assert len(tok.encode(env.carried, add_special_tokens=False)) <= 25
    # truncated state still scores the registers it kept
    f = state_fidelity(env.carried, prog, st)
    assert 0 < f < 1


def test_learned_missing_mem_gives_empty_state(tok):
    rng = random.Random(2)
    prog = interp.sample_program(rng, 30, 4)
    env = MarkovEnv(EnvConfig(mode="learned", C=64, m=100), tok)
    env.reset(prog)
    env.step(act(tok, "1: a=b+c=01+02=03.\n2: <mem>pc=3;a=0", 64))  # unterminated span
    assert env.carried == ""


def test_tail_keeps_last_m_tokens(tok):
    rng = random.Random(3)
    prog = interp.sample_program(rng, 30, 4)
    env = MarkovEnv(EnvConfig(mode="tail", C=192, m=25), tok)
    env.reset(prog)
    a = act(tok, oracle_chunks(prog, 12, with_mem=False)[0], 192)
    env.step(a)
    assert env.carried == tok.decode(a.token_ids[-25:])


def test_full_mode_reconstructs_sft_format(tok):
    rng = random.Random(4)
    prog = interp.sample_program(rng, 24, 8)
    C = 64
    env = MarkovEnv(EnvConfig(mode="full", C=C, m=0, max_chunks=12), tok)
    obs = env.reset(prog)
    full = tok.encode(interp.render_trace(prog), add_special_tokens=False) + [tok.eos_token_id]
    pos, done = 0, False
    while not done:
        ids = full[pos:pos + C]
        pos += len(ids)
        obs, r, done, _ = env.step(GenOut(tok.decode(ids, skip_special_tokens=True), ids, 0,
                                          "stop" if pos >= len(full) else "length"))
        if not done:  # prompt == SFT prompt + everything generated so far
            want = tok.encode(interp.render_prompt(interp.render_program(prog)), add_special_tokens=False) + full[:pos]
            assert obs["prompt_ids"] == want
    assert r == 1.0
    m = episode_metrics(env.episode_log(), prog)
    assert m["format_ok"] == 1.0 and m["trace_acc"] == 1.0


def test_out_of_chunks_zero_reward(tok):
    rng = random.Random(5)
    prog = interp.sample_program(rng, 30, 4)
    env = MarkovEnv(EnvConfig(mode="tail", C=16, m=8, max_chunks=3), tok)
    env.reset(prog)
    done = False
    n = 0
    while not done:
        _, r, done, _ = env.step(act(tok, "1: a=b+c=01+02=03.\n", 16))
        n += 1
    assert n == 3 and r == 0.0


def test_mem_dump_exact_metric(tok):
    rng = random.Random(6)
    prog = interp.sample_program(rng, 20, 8)
    text = interp.render_trace(prog, dump_after={5, 12})
    env = MarkovEnv(EnvConfig(mode="full", C=4096, m=0), tok)
    env.reset(prog)
    env.step(act(tok, text, 4096, eos=True))
    m = episode_metrics(env.episode_log(), prog)
    assert m["mem_n"] == 2 and m["mem_exact"] == 2 and m["correct"] == 1.0
    bad = text.replace("pc=6;", "pc=7;")
    env.reset(prog)
    env.step(act(tok, bad, 4096, eos=True))
    m = episode_metrics(env.episode_log(), prog)
    assert m["mem_n"] == 2 and m["mem_exact"] == 1


def test_state_api(tok):
    env = MarkovEnv(EnvConfig(mode="tail", C=192, m=100), tok)
    prog = interp.sample_program(random.Random(7), 8, 4)
    env.reset(prog)
    s = env.state()
    assert s["chunk_index"] == 0 and not s["done"] and s["answer"] == interp.answer_from(prog)
    env.step("Answer: " + interp.fmt(interp.answer_from(prog)))
    assert env.state()["done"] and env.state()["reward"] == 1.0
