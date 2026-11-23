"""Register-machine interpreter: program sampling, execution, rendering and parsing.

Only programs produced by `sample_program` are ever executed. Model output is
parsed with regexes into plain data; it is never evaluated as code.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

REG_NAMES = "abcdefghijklmnop"  # supports k <= 16
MOD = 100
OPS = ("add", "sub", "copy", "mul2", "mul3", "addc")


@dataclass(frozen=True)
class Step:
    dst: str
    op: str
    src1: str
    src2: str | None = None  # register for add/sub
    const: int | None = None  # 1..9 for addc

    def expr(self) -> str:
        """Right-hand side as written in the program, e.g. `c+a`, `b*2`, `d+5`."""
        if self.op == "add":
            return f"{self.src1}+{self.src2}"
        if self.op == "sub":
            return f"{self.src1}-{self.src2}"
        if self.op == "copy":
            return self.src1
        if self.op == "mul2":
            return f"{self.src1}*2"
        if self.op == "mul3":
            return f"{self.src1}*3"
        if self.op == "addc":
            return f"{self.src1}+{self.const}"
        raise ValueError(self.op)


@dataclass
class Program:
    k: int
    init: list[int]
    steps: list[Step]
    query: str
    meta: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return len(self.steps)

    @property
    def regs(self) -> str:
        return REG_NAMES[: self.k]

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "init": list(self.init),
            "steps": [[s.dst, s.op, s.src1, s.src2, s.const] for s in self.steps],
            "query": self.query,
        }

    @staticmethod
    def from_dict(d: dict) -> "Program":
        steps = [Step(dst, op, s1, s2, c) for dst, op, s1, s2, c in d["steps"]]
        return Program(k=d["k"], init=list(d["init"]), steps=steps, query=d["query"])


# ---------------------------------------------------------------------------
# Sampling and execution
# ---------------------------------------------------------------------------

def sample_program(rng: random.Random, N: int, k: int) -> Program:
    regs = REG_NAMES[:k]
    init = [rng.randrange(MOD) for _ in range(k)]
    steps = []
    for _ in range(N):
        op = rng.choice(OPS)
        dst = rng.choice(regs)
        src1 = rng.choice(regs)
        src2 = rng.choice(regs) if op in ("add", "sub") else None
        const = rng.randint(1, 9) if op == "addc" else None
        steps.append(Step(dst, op, src1, src2, const))
    # Query a register written in the last ~25% of steps.
    lo = N - max(1, -(-N // 4))  # N - ceil(N/4)
    query = steps[rng.randrange(lo, N)].dst
    return Program(k=k, init=init, steps=steps, query=query)


def apply_step(step: Step, regs: dict[str, int]) -> tuple[int, tuple[int, ...]]:
    """Return (result, operand values) for one step. Does not mutate regs."""
    v1 = regs[step.src1]
    if step.op == "add":
        v2 = regs[step.src2]
        return (v1 + v2) % MOD, (v1, v2)
    if step.op == "sub":
        v2 = regs[step.src2]
        return (v1 - v2) % MOD, (v1, v2)
    if step.op == "copy":
        return v1, (v1,)
    if step.op == "mul2":
        return (v1 * 2) % MOD, (v1,)
    if step.op == "mul3":
        return (v1 * 3) % MOD, (v1,)
    if step.op == "addc":
        return (v1 + step.const) % MOD, (v1,)
    raise ValueError(step.op)


def initial_regs(prog: Program) -> dict[str, int]:
    return dict(zip(prog.regs, prog.init))


def run(prog: Program, start_pc: int = 1, regs: dict[str, int] | None = None) -> dict[str, int]:
    """Execute steps start_pc..N (1-based) from `regs` (default: Start line). Returns final regs."""
    regs = dict(initial_regs(prog) if regs is None else regs)
    for step in prog.steps[start_pc - 1:]:
        regs[step.dst], _ = apply_step(step, regs)
    return regs


def answer_from(prog: Program, start_pc: int = 1, regs: dict[str, int] | None = None) -> int:
    return run(prog, start_pc, regs)[prog.query]


def states(prog: Program) -> list[dict[str, int]]:
    """states[i] = registers after i steps (states[0] = Start). Next step to execute is pc=i+1."""
    regs = initial_regs(prog)
    out = [dict(regs)]
    for step in prog.steps:
        regs[step.dst], _ = apply_step(step, regs)
        out.append(dict(regs))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def fmt(v: int) -> str:
    return f"{v:02d}"


def render_program(prog: Program) -> str:
    lines = ["Start: " + " ".join(f"{r}={fmt(v)}" for r, v in zip(prog.regs, prog.init))]
    for i, s in enumerate(prog.steps, 1):
        lines.append(f"{i}: {s.dst}={s.expr()}")
    lines.append(f"Q: value of {prog.query} after step {prog.N}?")
    return "\n".join(lines)


def render_trace_line(i: int, step: Step, regs_before: dict[str, int], terse: bool = False) -> str:
    res, vals = apply_step(step, regs_before)
    if terse:
        return f"{i}: {step.dst}={fmt(res)}."
    if step.op == "copy":
        return f"{i}: {step.dst}={step.src1}={fmt(vals[0])}."
    if step.op in ("add", "sub"):
        sym = "+" if step.op == "add" else "-"
        return f"{i}: {step.dst}={step.expr()}={fmt(vals[0])}{sym}{fmt(vals[1])}={fmt(res)}."
    if step.op in ("mul2", "mul3"):
        return f"{i}: {step.dst}={step.expr()}={fmt(vals[0])}*{step.op[-1]}={fmt(res)}."
    if step.op == "addc":
        return f"{i}: {step.dst}={step.expr()}={fmt(vals[0])}+{step.const}={fmt(res)}."
    raise ValueError(step.op)


def render_state(pc: int, regs: dict[str, int], k: int | None = None, tags: bool = True) -> str:
    names = sorted(regs) if k is None else REG_NAMES[:k]
    body = f"pc={pc};" + ";".join(f"{r}={fmt(regs[r])}" for r in names)
    return f"<mem>{body}</mem>" if tags else body


def render_answer(v: int) -> str:
    return f"Answer: {fmt(v)}"


def render_trace(prog: Program, terse: bool = False, dump_after: set[int] | None = None) -> str:
    """Full-context trace. `dump_after` = step indices after which a full <mem> dump is written."""
    dump_after = dump_after or set()
    regs = initial_regs(prog)
    lines = []
    for i, s in enumerate(prog.steps, 1):
        lines.append(render_trace_line(i, s, regs, terse))
        regs[s.dst], _ = apply_step(s, regs)
        if i in dump_after and i < prog.N:
            lines.append(render_state(i + 1, regs, prog.k))
    lines.append(render_answer(regs[prog.query]))
    return "\n".join(lines)


def render_prompt(program_text: str, state: str = "") -> str:
    """Prompt format shared by SFT, env and eval."""
    state_line = f"State: {state}" if state else "State:"
    return f"Program:\n{program_text}\n{state_line}\nTrace:\n"


# ---------------------------------------------------------------------------
# Parsing (model output -> data; never executed)
# ---------------------------------------------------------------------------

_R = "[a-p]"
_V = r"\d{1,2}"
TRACE_RE = re.compile(
    rf"^(?P<step>\d+): (?P<dst>{_R})="
    rf"(?:"
    rf"(?P<s1>{_R})(?P<sym>[+\-])(?P<s2>{_R})=(?P<v1>{_V})(?P=sym)(?P<v2>{_V})=(?P<res_bin>{_V})"  # add/sub
    rf"|(?P<m1>{_R})\*(?P<mk>[23])=(?P<mv>{_V})\*(?P=mk)=(?P<res_mul>{_V})"  # mul
    rf"|(?P<c1>{_R})\+(?P<cc>[1-9])=(?P<cv>{_V})\+(?P=cc)=(?P<res_c>{_V})"  # addc
    rf"|(?P<y1>{_R})=(?P<res_copy>{_V})"  # copy
    rf"|(?P<res_terse>{_V})"  # terse
    rf")\.$"
)
MEM_RE = re.compile(r"<mem>(.*?)</mem>", re.S)
STATE_RE = re.compile(rf"^pc=(\d+)((?:;{_R}={_V})*)$")
ANSWER_RE = re.compile(r"Answer:\s*(\d{1,3})")


@dataclass
class TraceLine:
    step: int
    dst: str
    result: int
    expr: str | None  # e.g. "c+a"; None for terse
    operands: tuple[int, ...]  # values the model wrote
    arith_ok: bool | None  # result correct given the model's own operands; None if no arithmetic


def parse_trace_line(line: str) -> TraceLine | None:
    m = TRACE_RE.match(line.strip())
    if not m:
        return None
    g = m.groupdict()
    step, dst = int(g["step"]), g["dst"]
    if g["res_bin"] is not None:
        v1, v2, res = int(g["v1"]), int(g["v2"]), int(g["res_bin"])
        want = (v1 + v2) % MOD if g["sym"] == "+" else (v1 - v2) % MOD
        return TraceLine(step, dst, res, f"{g['s1']}{g['sym']}{g['s2']}", (v1, v2), res == want)
    if g["res_mul"] is not None:
        v, kk, res = int(g["mv"]), int(g["mk"]), int(g["res_mul"])
        return TraceLine(step, dst, res, f"{g['m1']}*{kk}", (v,), res == (v * kk) % MOD)
    if g["res_c"] is not None:
        v, c, res = int(g["cv"]), int(g["cc"]), int(g["res_c"])
        return TraceLine(step, dst, res, f"{g['c1']}+{c}", (v,), res == (v + c) % MOD)
    if g["res_copy"] is not None:
        res = int(g["res_copy"])
        return TraceLine(step, dst, res, g["y1"], (res,), None)
    return TraceLine(step, dst, int(g["res_terse"]), None, (), None)


def parse_state_body(body: str) -> tuple[int | None, dict[str, int]]:
    """`pc=9;a=07;b=66` -> (9, {'a': 7, 'b': 66}). Malformed -> (None, {})."""
    m = STATE_RE.match(body.strip())
    if not m:
        return None, {}
    regs = {}
    for item in m.group(2).split(";")[1:]:
        r, v = item.split("=")
        regs[r] = int(v)
    return int(m.group(1)), regs


def last_mem_span(text: str) -> str | None:
    """Content of the LAST complete <mem>...</mem> span, or None."""
    spans = MEM_RE.findall(text)
    return spans[-1] if spans else None


def parse_mem(text: str) -> tuple[int | None, dict[str, int]]:
    span = last_mem_span(text)
    return (None, {}) if span is None else parse_state_body(span)


def parse_answer(text: str) -> int | None:
    ms = ANSWER_RE.findall(text)
    return int(ms[-1]) if ms else None


def classify_line(line: str) -> str:
    """One of: trace, mem, answer, empty, bad."""
    s = line.strip()
    if not s:
        return "empty"
    if parse_trace_line(s):
        return "trace"
    if re.fullmatch(r"<mem>.*</mem>", s) and parse_state_body(s[5:-6])[0] is not None:
        return "mem"
    if re.fullmatch(r"Answer:\s*\d{1,3}", s):
        return "answer"
    return "bad"
