import random

import pytest

from markov_rl import interp
from markov_rl.interp import Program, Step


def hand_program() -> Program:
    # Hand-checked example (values mod 100).
    steps = [
        Step("c", "add", "c", "a"),     # c = 12+07 = 19
        Step("b", "mul2", "b"),         # b = 33*2 = 66
        Step("d", "sub", "c", "b"),     # d = 19-66 = -47 -> 53
        Step("a", "addc", "d", const=5),  # a = 53+5 = 58
        Step("b", "mul3", "b"),         # b = 66*3 = 198 -> 98
        Step("c", "copy", "a"),         # c = 58
        Step("d", "add", "d", "d"),     # d = 53+53 = 106 -> 06
    ]
    return Program(k=4, init=[7, 33, 12, 0], steps=steps, query="b")


def test_hand_checked_execution():
    p = hand_program()
    final = interp.run(p)
    assert final == {"a": 58, "b": 98, "c": 58, "d": 6}
    assert interp.answer_from(p) == 98
    s = interp.states(p)
    assert s[0] == {"a": 7, "b": 33, "c": 12, "d": 0}
    assert s[3] == {"a": 7, "b": 66, "c": 19, "d": 53}


def test_hand_checked_rendering():
    p = hand_program()
    assert interp.render_program(p).splitlines() == [
        "Start: a=07 b=33 c=12 d=00",
        "1: c=c+a", "2: b=b*2", "3: d=c-b", "4: a=d+5", "5: b=b*3", "6: c=a", "7: d=d+d",
        "Q: value of b after step 7?",
    ]
    assert interp.render_trace(p).splitlines() == [
        "1: c=c+a=12+07=19.", "2: b=b*2=33*2=66.", "3: d=c-b=19-66=53.", "4: a=d+5=53+5=58.",
        "5: b=b*3=66*3=98.", "6: c=a=58.", "7: d=d+d=53+53=06.", "Answer: 98",
    ]
    assert interp.render_trace(p, terse=True).splitlines()[:2] == ["1: c=19.", "2: b=66."]
    assert interp.render_state(4, interp.states(p)[3], 4) == "<mem>pc=4;a=07;b=66;c=19;d=53</mem>"
    assert interp.render_prompt("X") == "Program:\nX\nState:\nTrace:\n"
    assert interp.render_prompt("X", "pc=2;a=01") == "Program:\nX\nState: pc=2;a=01\nTrace:\n"


@pytest.mark.parametrize("terse", [False, True])
def test_trace_roundtrip(terse):
    rng = random.Random(0)
    for _ in range(200):
        p = interp.sample_program(rng, rng.randint(1, 40), rng.choice([4, 8, 16]))
        st = interp.states(p)
        lines = interp.render_trace(p, terse=terse).splitlines()
        assert len(lines) == p.N + 1
        for i, line in enumerate(lines[:-1], 1):
            tl = interp.parse_trace_line(line)
            assert tl is not None, line
            assert tl.step == i and tl.dst == p.steps[i - 1].dst
            assert tl.result == st[i][tl.dst]
            assert tl.arith_ok in (True, None)
            if not terse:
                assert tl.expr == p.steps[i - 1].expr()
        assert interp.parse_answer(lines[-1]) == interp.answer_from(p)


def test_state_roundtrip():
    rng = random.Random(1)
    for _ in range(200):
        k = rng.choice([4, 8, 16])
        regs = {r: rng.randrange(100) for r in interp.REG_NAMES[:k]}
        pc = rng.randint(1, 65)
        text = "1: a=05.\n" + interp.render_state(pc, regs, k) + "\n"
        assert interp.parse_mem(text) == (pc, regs)
        assert interp.classify_line(interp.render_state(pc, regs, k)) == "mem"


def test_parsers_on_bad_input():
    assert interp.parse_mem("no mem here") == (None, {})
    assert interp.parse_mem("<mem>pc=3;a=1x</mem>") == (None, {})
    assert interp.parse_mem("<mem>pc=3;a=01") == (None, {})  # unterminated
    assert interp.parse_mem("<mem>pc=2;a=01</mem> x <mem>pc=5;a=09</mem>") == (5, {"a": 9})  # last span
    assert interp.parse_answer("Answer: 07") == 7
    assert interp.parse_answer("Answer: 03\nAnswer: 44") == 44
    assert interp.parse_answer("no answer") is None
    assert interp.parse_trace_line("1: c=c+a=12+07=20.").arith_ok is False
    assert interp.parse_trace_line("1: c=c+a=12+07=19") is None  # missing dot
    assert interp.parse_trace_line("1: c=c+a=12-07=05.") is None  # operator mismatch
    assert interp.parse_trace_line("1: c=b*2=10*3=30.") is None
    assert interp.classify_line("garbage") == "bad"
    assert interp.classify_line("Answer: 5") == "answer"


def test_resume_from_any_state_gives_same_answer():
    """Exact sufficiency check: (pc, registers) at any boundary determines the answer."""
    rng = random.Random(2)
    for _ in range(300):
        p = interp.sample_program(rng, rng.randint(1, 64), rng.choice([4, 8, 16]))
        want = interp.answer_from(p)
        for i, regs in enumerate(interp.states(p)):
            assert interp.answer_from(p, start_pc=i + 1, regs=regs) == want
            # Also via render -> parse of the state string.
            pc, parsed = interp.parse_mem(interp.render_state(i + 1, regs, p.k))
            assert interp.answer_from(p, start_pc=pc, regs=parsed) == want


def test_query_in_last_quarter():
    rng = random.Random(3)
    for _ in range(500):
        N = rng.randint(1, 64)
        p = interp.sample_program(rng, N, 8)
        lo = N - max(1, -(-N // 4))
        assert p.query in {s.dst for s in p.steps[lo:]}


def test_program_dict_roundtrip():
    rng = random.Random(4)
    p = interp.sample_program(rng, 20, 16)
    q = Program.from_dict(p.to_dict())
    assert interp.render_program(q) == interp.render_program(p)
    assert interp.answer_from(q) == interp.answer_from(p)
