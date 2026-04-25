"""PTX instruction namespace.

Every function in this module emits exactly one PTX instruction.
No hidden scheduling, no lowering passes. Ten calls = ten instructions.

Usage (inside a @kernel function):
    from pyptx import ptx

    ptx.wgmma.mma_async(shape=(64,256,16), dtype_d=f32, ...)
    ptx.cp.async.bulk.tensor_2d(dst=sA[0], src=A.tma_desc(), ...)
    ptx.mbarrier.wait(bar[0], phase)
    ptx.raw("tcgen05.mma.cta_group::1 ...;")

    with ptx.if_(is_producer):
        ...
    with ptx.else_():
        ...
    for k in ptx.range_(0, K, BK):
        ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Generator

from pyptx._trace import get_ctx
from pyptx.ir.nodes import (
    AddressOperand,
    ImmediateOperand,
    Instruction,
    Label,
    LabelOperand,
    Predicate,
    RegisterOperand,
    VectorOperand,
)
from pyptx.reg import NegPred, PipeRef, Reg, RegArray
from pyptx.smem import MbarrierRef, SharedAlloc, SharedSlice
from pyptx.types import PtxType


# ===================================================================
# Operand conversion
# ===================================================================

def _op(val: Any) -> Any:
    """Convert a DSL value to an IR operand."""
    if isinstance(val, Reg):
        return RegisterOperand(val.name)
    if isinstance(val, int):
        return ImmediateOperand(str(val))
    if isinstance(val, float):
        return ImmediateOperand(str(val))
    if isinstance(val, str):
        # Strings starting with 0x, 0d, 0f, or - are immediates, not labels
        if val and (val[0].isdigit() or val[0] == '-' or val.startswith("0x") or val.startswith("0X")):
            return ImmediateOperand(val)
        return LabelOperand(val)
    if isinstance(val, list):
        return VectorOperand(tuple(_op(v) for v in val))
    if isinstance(val, tuple):
        from pyptx.ir.nodes import ParenthesizedOperand
        return ParenthesizedOperand(tuple(_op(v) for v in val))
    if isinstance(val, RegArray):
        return VectorOperand(tuple(RegisterOperand(val[i].name) for i in range(val.count)))
    if isinstance(val, SharedSlice):
        return RegisterOperand(val.name)  # simplified; full impl uses offset
    if isinstance(val, SharedAlloc):
        return RegisterOperand(val.name)
    if isinstance(val, MbarrierRef):
        return RegisterOperand(val.name)
    # TmaDescriptorHandle — imported lazily to avoid circular dep with kernel.py
    if type(val).__name__ == "TmaDescriptorHandle":
        return LabelOperand(val.name)
    if type(val).__name__ == "TensorSpec":
        return LabelOperand(val.name)
    if isinstance(val, PipeRef):
        from pyptx.ir.nodes import PipeOperand
        return PipeOperand(RegisterOperand(val.left.name), RegisterOperand(val.right.name))
    if isinstance(val, NegPred):
        from pyptx.ir.nodes import NegatedOperand
        return NegatedOperand(RegisterOperand(val.reg.name))
    if isinstance(val, RegisterOperand | ImmediateOperand | LabelOperand | VectorOperand | AddressOperand):
        return val
    raise TypeError(f"Cannot convert {type(val).__name__} to PTX operand: {val!r}")


def _pred(p: Reg | NegPred | None) -> Predicate | None:
    if p is None:
        return None
    if isinstance(p, NegPred):
        return Predicate(register=p.reg.name, negated=True)
    if isinstance(p, Reg):
        return Predicate(register=p.name, negated=False)
    raise TypeError(f"Expected Reg or NegPred, got {type(p).__name__}")


def _emit(
    opcode: str,
    modifiers: tuple[str, ...],
    operands: tuple,
    pred: Reg | NegPred | None = None,
) -> None:
    import sys

    from pyptx.spec.validate import (
        is_strict,
        validate_instruction,
        validate_or_raise,
    )

    ctx = get_ctx()
    inst = Instruction(
        opcode=opcode,
        modifiers=modifiers,
        operands=tuple(_op(o) for o in operands),
        predicate=_pred(pred),
    )

    # Trace-time validation.
    #
    # Strict-mode rules:
    #   * Typed wrappers (ptx.wgmma.mma_async, ptx.mbarrier.init, ...)
    #     get validated against the spec and raise on any error-severity
    #     issue.
    #   * The dot-chain escape hatch ``ptx.inst.<opcode>...(...)`` is the
    #     escape hatch within the escape hatch. Users opting into it have
    #     explicitly chosen to bypass the typed surface, so validation is
    #     advisory there: issues are still recorded on the trace context
    #     but never raised. Without this carve-out the corpus codegen
    #     pipeline (which always emits `ptx.inst.*` calls) would be
    #     unable to round-trip real-world PTX that uses modifiers our
    #     spec doesn't yet enumerate.
    #   * Unknown opcodes (no spec at all) only emit a Python warning and
    #     never raise.
    #
    # The escape-hatch path is detected by walking one frame up the
    # Python call stack and checking whether the caller is the
    # ``_GenericInst.__call__`` method defined later in this module.
    caller = sys._getframe(1)
    via_escape_hatch = (
        caller is not None
        and caller.f_code.co_name == "__call__"
        and caller.f_code.co_filename == __file__
    )

    if via_escape_hatch or not is_strict():
        issues = validate_instruction(inst)
    else:
        issues = validate_or_raise(inst)

    if issues:
        bucket = getattr(ctx, "validation_issues", None)
        if bucket is None:
            bucket = []
            try:
                ctx.validation_issues = bucket  # type: ignore[attr-defined]
            except AttributeError:
                bucket = None
        if bucket is not None:
            bucket.extend(issues)

    ctx.emit(inst)


# ===================================================================
# Special registers
# ===================================================================

class _DimReg:
    """Special register with .x/.y/.z (e.g. %tid)."""
    def __init__(self, base: str, dtype: PtxType | None = None) -> None:
        self._base = base
        from pyptx.types import u32
        self._dtype = dtype or u32

    def x(self) -> Reg:
        return Reg(f"%{self._base}.x", self._dtype)

    def y(self) -> Reg:
        return Reg(f"%{self._base}.y", self._dtype)

    def z(self) -> Reg:
        return Reg(f"%{self._base}.z", self._dtype)


class _Special:
    """ptx.special.tid.x(), ptx.special.laneid(), etc."""
    @property
    def tid(self) -> _DimReg:
        return _DimReg("tid")

    @property
    def ntid(self) -> _DimReg:
        return _DimReg("ntid")

    @property
    def ctaid(self) -> _DimReg:
        return _DimReg("ctaid")

    @property
    def nctaid(self) -> _DimReg:
        return _DimReg("nctaid")

    def laneid(self) -> Reg:
        from pyptx.types import u32
        return Reg("%laneid", u32)

    def warpid(self) -> Reg:
        from pyptx.types import u32
        return Reg("%warpid", u32)

    def clock(self) -> Reg:
        from pyptx.types import u32
        return Reg("%clock", u32)

special = _Special()


def sreg(name: str) -> Reg:
    """Reference any PTX special register by name.

    Usage:
        ptx.sreg("%cluster_ctarank")
        ptx.sreg("%clusterid.x")
        ptx.sreg("%smid")

    For common ones, prefer ptx.special.tid.x() etc.
    """
    from pyptx.types import u32
    return Reg(name, u32)


def loc(file_idx: int, line: int, col: int = 0) -> None:
    """Emit a .loc debug directive for source attribution.

    Usage: ptx.loc(1, 40, 0)  →  .loc 1 40 0
    """
    ctx = get_ctx()
    ctx.emit(Instruction(
        opcode=".loc",
        modifiers=(),
        operands=(
            ImmediateOperand(str(file_idx)),
            ImmediateOperand(str(line)),
            ImmediateOperand(str(col)),
        ),
    ))


def file_(file_idx: int, filename: str) -> None:
    """Emit a .file debug directive.

    Usage: ptx.file_(1, "kernel.py")  →  .file 1 "kernel.py"
    """
    ctx = get_ctx()
    ctx.emit(Instruction(
        opcode=".file",
        modifiers=(),
        operands=(
            ImmediateOperand(str(file_idx)),
            LabelOperand(f'"{filename}"'),
        ),
    ))


def pragma(value: str) -> None:
    """Emit a .pragma directive.

    Usage: ptx.pragma("nounroll")  →  .pragma "nounroll";
    """
    from pyptx.ir.nodes import PragmaDirective
    ctx = get_ctx()
    ctx.emit(PragmaDirective(value=value))


def var(
    state_space: str,
    dtype: PtxType,
    name: str,
    *,
    size: int | None = None,
    align: int | None = None,
    linking: str | None = None,
) -> str:
    """Declare a variable in any state space.

    Usage:
        ptx.var("shared", b8, "smem", size=49152, align=128)
        ptx.var("param", b32, "param0")
        ptx.var("global", f32, "output", size=1024, linking="visible")

    Returns the variable name (for use with ptx.addr()).
    """
    from pyptx.ir.nodes import VarDecl
    from pyptx.ir.types import ScalarType, StateSpace, LinkingDirective

    ctx = get_ctx()
    ss = StateSpace.from_ptx(f".{state_space.replace('__', '::')}")
    scalar_type = ScalarType.from_ptx(dtype.ptx)
    link = LinkingDirective.from_ptx(f".{linking}") if linking else None

    ctx.var_decls.append(VarDecl(
        state_space=ss,
        type=scalar_type,
        name=name,
        array_size=size,
        alignment=align,
        linking=link,
    ))
    return name


# Convenience top-level: ptx.ctaid.x() etc.
def ctaid_x() -> Reg:
    """Convenience alias for ``%ctaid.x``."""
    return special.ctaid.x()

def ctaid_y() -> Reg:
    """Convenience alias for ``%ctaid.y``."""
    return special.ctaid.y()


def global_ptrs(*params: Any) -> tuple[Reg, ...]:
    """Load kernel parameter pointers into fresh global-space registers.

    For each kernel parameter (typically a ``TensorSpec`` passed into the
    ``@kernel`` function body) this emits the canonical prologue pair::

        ld.param.u64     %rd_n, [param_name];
        cvta.to.global.u64 %rd_n, %rd_n;

    and returns a tuple of ``Reg`` objects — one global-space b64
    pointer per parameter. Kernels then write::

        px, pw, py = ptx.global_ptrs(X, W, Y)

    instead of six lines of boilerplate per invocation. Single
    parameter still returns a 1-tuple; call-site unpack with a
    trailing comma::

        (px,) = ptx.global_ptrs(X)
    """
    from pyptx.reg import scalar as _reg_scalar
    from pyptx.types import b64

    if not params:
        return ()

    out: list[Reg] = []
    for p in params:
        r = _reg_scalar(b64)
        # Go through the inst-style escape hatch so the ``cvta.to.global``
        # path stays on the advisory-validation branch (the spec's cvta
        # entry doesn't enumerate the ``.to`` direction yet; real kernels
        # have been using ``ptx.inst.cvta.to.global_.u64`` for this).
        inst.ld.param.u64(r, addr(p))
        inst.cvta.to.global_.u64(r, r)
        out.append(r)
    return tuple(out)


# ===================================================================
# Warp-scope reductions
# ===================================================================
#
# Canonical butterfly-shuffle reduction over a warp (or a power-of-two
# sub-group of a warp). Emits ``shfl.sync.bfly.b32`` at shrinking XOR
# lane masks and combines with the requested op. Matches the pattern
# every reduction kernel in this repo writes by hand.
#
# ``width`` is the reduction group size in lanes: 32 = full warp,
# 4 = the per-row group used by wgmma.m64nN output fragments, etc.
# Must be a power of two in ``[2, 32]``.
#
# ``val`` is a ``Reg`` holding a scalar in ``b32`` / ``u32`` / ``s32`` /
# ``f32``. For f32 we bit-cast through a b32 temp on each iteration
# (``shfl.sync.bfly`` only accepts b32/b64). The result is written
# back into ``val`` in place.


def _warp_reduce(val: "Reg", *, op: str, width: int) -> None:
    """In-place warp-scope butterfly reduction of ``val``.

    ``op`` is one of ``"sum"``, ``"max"``, ``"min"`` (for f32). For
    integer types, ``"sum"`` / ``"max"`` / ``"min"`` / ``"and"`` /
    ``"or"`` / ``"xor"`` all work but we don't enumerate them here —
    use ``reduce_sum`` / ``reduce_max`` wrappers.

    After this call every lane in the reduction group holds the
    reduction of all lanes' original values. Lanes outside the group
    are untouched relative to each other but share data across group
    boundaries (``shfl.sync.bfly`` respects ``width`` naturally via
    the ``c`` parameter — we pass ``width-1`` so lanes outside the
    group clamp to themselves).
    """
    if width not in (2, 4, 8, 16, 32):
        raise ValueError(
            f"warp reduce width must be a power of two in [2, 32], got {width}"
        )
    dtype = val.dtype
    is_float = dtype.name in ("f32", "tf32")
    if dtype.name not in ("b32", "u32", "s32", "f32", "tf32"):
        raise TypeError(
            f"warp reduce supports 32-bit scalars only (b32/u32/s32/f32), got {dtype.name}"
        )

    from pyptx.reg import scalar as _reg_scalar
    from pyptx.types import b32, f32 as f32_type

    mask = width >> 1
    while mask > 0:
        # shfl.sync.bfly.b32 tmp_b, src_b, mask, 31, -1
        #
        # c is always 0x1F (full-warp clamp); the sub-group semantics
        # come from the xor mask ``b`` alone — for ``b < width`` the
        # paired lanes stay within the same ``width``-lane group, so
        # there's no out-of-range source to clamp. This matches the
        # hand-rolled _warp_bfly_*_f32 pattern in the pre-sugar kernels.
        src_b = _reg_scalar(b32)
        inst.mov.b32(src_b, val)
        tmp_b = _reg_scalar(b32)
        inst.shfl.sync.bfly.b32(tmp_b, src_b, mask, 31, -1)
        if is_float:
            tmp_f = _reg_scalar(f32_type)
            inst.mov.b32(tmp_f, tmp_b)
            partner = tmp_f
        else:
            partner = tmp_b

        if op == "sum":
            inst.add.f32(val, val, partner) if is_float \
                else inst.add.u32(val, val, partner)
        elif op == "max":
            inst.max.f32(val, val, partner) if is_float \
                else inst.max.u32(val, val, partner)
        elif op == "min":
            inst.min.f32(val, val, partner) if is_float \
                else inst.min.u32(val, val, partner)
        else:
            raise ValueError(f"unsupported warp reduce op {op!r}")

        mask >>= 1


class _Warp:
    """``ptx.warp.reduce_sum(val)`` / ``reduce_max(val)`` / ``reduce_min(val)``
    — in-place warp-scope reductions.

    ``width`` is the reduction group size in lanes (default 32 = full
    warp). Pass ``width=4`` for the per-row reduction across the
    4-lane groups that share a row in the ``wgmma.m64nN`` output
    fragment layout — this is what Flash Attention's online softmax
    needs to turn its per-thread row_max into a full-row max.
    """

    def reduce_sum(self, val: "Reg", *, width: int = 32) -> None:
        _warp_reduce(val, op="sum", width=width)

    def reduce_max(self, val: "Reg", *, width: int = 32) -> None:
        _warp_reduce(val, op="max", width=width)

    def reduce_min(self, val: "Reg", *, width: int = 32) -> None:
        _warp_reduce(val, op="min", width=width)


warp = _Warp()


# ===================================================================
# Control flow
# ===================================================================

@contextmanager
def if_(pred_reg: Reg | NegPred) -> Generator[None, None, None]:
    """Conditional block. Emits one branch instruction.

    Usage:
        with ptx.if_(is_producer):
            ...   # body executes only if pred is true

        # Optional chained else:
        with ptx.if_(p):
            ...
        with ptx.else_():
            ...
    """
    ctx = get_ctx()
    else_lbl = ctx.fresh_label("else")
    end_lbl = ctx.fresh_label("endif")

    # @!pred bra $else (skip body when pred is false). If there's no
    # else_() following, we'll retarget this branch at exit time: the
    # $else label will coincide with $endif and the body falls straight
    # through.
    if isinstance(pred_reg, NegPred):
        ctx.emit(Instruction(
            opcode="bra",
            operands=(LabelOperand(else_lbl),),
            predicate=Predicate(register=pred_reg.reg.name, negated=False),
        ))
    else:
        ctx.emit(Instruction(
            opcode="bra",
            operands=(LabelOperand(else_lbl),),
            predicate=Predicate(register=pred_reg.name, negated=True),
        ))

    ctx._if_stack.append((else_lbl, end_lbl))
    yield

    # Emit `bra $endif; $else:; $endif:` — the final $endif: label is a
    # tentative fall-through target that else_() will rewrite if one
    # follows. Leaving it in place keeps `if_` without an `else_` well-
    # formed (no dangling bra-target).
    ctx.emit(Instruction(opcode="bra", operands=(LabelOperand(end_lbl),)))
    ctx.emit(Label(name=else_lbl))
    ctx.emit(Label(name=end_lbl))


@contextmanager
def else_() -> Generator[None, None, None]:
    """Else block — must follow an if_() block.

    Usage:
        with ptx.if_(pred):
            ...
        with ptx.else_():
            ...
    """
    ctx = get_ctx()
    if not ctx._if_stack:
        raise RuntimeError("else_() must follow an if_() block")
    _, end_lbl = ctx._if_stack.pop()

    # The matching if_() has already emitted a tentative $endif: label
    # as its last statement so a bare if is well-formed. We're about to
    # emit the real else body, so pop that tentative label: the else
    # body should run between $else: and $endif:, not after a premature
    # $endif:.
    if ctx.statements and isinstance(ctx.statements[-1], Label) \
            and ctx.statements[-1].name == end_lbl:
        ctx.statements.pop()

    yield

    ctx.emit(Label(name=end_lbl))


@contextmanager
def scope() -> Generator[None, None, None]:
    """Open a PTX ``{ }`` block scope.

    Register declarations inside the scope are emitted inline (block-local)
    rather than hoisted to the function top. This maps directly to PTX's
    nested ``{ ... }`` scoping, where ``.reg`` declarations are local to
    the enclosing braces.

    Usage::

        with ptx.scope():
            tmp = reg.scalar(b32, name="tmp")
            ptx.inst.mov.b32(tmp, 42)
        # tmp is out of scope here; the name can be reused in another scope
    """
    from pyptx.ir.nodes import RawLine
    ctx = get_ctx()
    ctx.emit(RawLine(text="{"))
    ctx._scope_depth += 1
    try:
        yield
    finally:
        ctx._scope_depth -= 1
        ctx.emit(RawLine(text="}"))


@contextmanager
def loop(label_name: str, *, pred: "Reg | NegPred | None" = None) -> Generator[None, None, None]:
    """Emit a PTX loop: ``label: ... @pred bra label;``

    The label is emitted on entry, and a conditional backward branch
    is emitted on exit. The body goes inside the ``with`` block.

    Usage::

        with ptx.loop("k_loop", pred=p[14]):
            # ... loop body ...
            # at the end, emits: @%p14 bra k_loop;

    For unconditional loops (persistent tile loops), omit pred::

        with ptx.loop("tile_loop"):
            # ... body ...
            # emits: bra.uni tile_loop;
    """
    ctx = get_ctx()
    ctx.emit(Label(name=label_name))
    yield
    if pred is not None:
        _emit("bra", (), (LabelOperand(label_name),), pred=pred)
    else:
        _emit("bra", (".uni",), (LabelOperand(label_name),))


class PipelineState:
    """Loop-carried stage cursor + phase bit for ring-buffered pipelines.

    ``advance()`` emits the common Hopper pattern:
    - compare cursor against ``n_stages``
    - flip the phase on wrap
    - return the wrapped stage index
    - update the loop-carried cursor in place
    """

    __slots__ = ("n_stages", "cursor", "phase")

    def __init__(
        self,
        n_stages: int,
        *,
        cursor: Reg | None = None,
        phase: Reg | None = None,
    ) -> None:
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import u32

        self.n_stages = n_stages
        self.cursor = cursor if cursor is not None else reg_scalar(u32, init=0)
        self.phase = phase if phase is not None else reg_scalar(u32, init=0)

    def advance(self) -> tuple[Reg, Reg]:
        """Advance the ring and return ``(stage, phase)``."""
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import u32

        wrap = self.cursor == self.n_stages
        flip = reg_scalar(u32)
        selp(u32, flip, 1, 0, wrap)
        self.phase ^= flip

        stage = reg_scalar(u32)
        selp(u32, stage, 0, self.cursor, wrap)
        inst.add.s32(self.cursor, stage, 1)
        return stage, self.phase


def pipeline(
    n_stages: int,
    *,
    cursor: Reg | None = None,
    phase: Reg | None = None,
) -> PipelineState:
    """Create a loop-carried pipeline stage/phase helper."""
    return PipelineState(n_stages, cursor=cursor, phase=phase)


@contextmanager
def expr() -> Generator[None, None, None]:
    """Capture a Python expression's PTX instructions into one CompoundExpr.

    All ``ptx.inst.*`` calls and Reg operator overloads inside the block
    are buffered, then emitted as a single :class:`CompoundExpr` IR node.
    Instructions execute in Python evaluation order (which IS the correct
    data-dependency order for expressions).

    Usage::

        with ptx.expr():
            rd[26] = ((r[192] - 8192) & 0x3FF80) >> 4 | CONST

    The PTX output is identical to writing the instructions individually.
    The benefit is a compact, readable Python source.
    """
    from pyptx.ir.compound import CompoundExpr

    ctx = get_ctx()
    # Save the current statement list and replace with a buffer
    saved = ctx.statements
    buffer: list = []
    ctx.statements = buffer
    try:
        yield
    finally:
        ctx.statements = saved
        # Wrap all buffered instructions into one CompoundExpr
        from pyptx.ir.nodes import Instruction
        instrs = tuple(s for s in buffer if isinstance(s, Instruction))
        non_instrs = [s for s in buffer if not isinstance(s, Instruction)]
        # Emit any non-instruction statements (reg decls from scope) first
        for s in non_instrs:
            ctx.emit(s)
        if instrs:
            ctx.emit(CompoundExpr(instructions=instrs))


def range_(start, stop: int, step: int = 1) -> Generator[Reg, None, None]:
    """Staged loop. Emits PTX branches for the loop structure.

    ``start`` can be a Python ``int`` or a ``Reg`` (for persistent
    kernel scheduling where the loop starts at ``ctaid.x``).

    Usage:
        for k in ptx.range_(0, K, BK):
            ...  # k is a Reg holding the loop variable

        # Persistent: start from ctaid.x
        for tile in ptx.range_(cta_id, total_tiles, NUM_SM):
            ...

    Emits:
        mov.s32 %rN, start;    (or mov.s32 %rN, %start_reg;)
        $loop:
        setp.ge.s32 %pN, %rN, stop;
        @%pN bra $endloop;
        ... body ...
        add.s32 %rN, %rN, step;
        bra $loop;
        $endloop:
    """
    from pyptx.types import s32
    from pyptx.ir.types import ScalarType

    ctx = get_ctx()

    # Allocate loop variable register
    loop_idx = ctx.alloc_reg_name("%ri")
    loop_var = f"%ri{loop_idx}"
    ctx.reg_decls.append(
        __import__("pyptx.ir.nodes", fromlist=["RegDecl"]).RegDecl(
            type=ScalarType.from_ptx(".s32"), name=loop_var
        )
    )

    loop_lbl = ctx.fresh_label("loop")
    end_lbl = ctx.fresh_label("endloop")

    # Allocate a pred for the exit condition
    from pyptx.reg import _ensure_pred_decl
    p_idx = ctx.alloc_reg_name("%p")
    _ensure_pred_decl(ctx, p_idx + 1)
    p_name = f"%p{p_idx}"

    # mov.s32 %rN, start  (or mov from Reg if start is runtime)
    if isinstance(start, Reg):
        ctx.emit(Instruction(
            opcode="mov", modifiers=(".s32",),
            operands=(RegisterOperand(loop_var), RegisterOperand(start.name)),
        ))
    else:
        ctx.emit(Instruction(
            opcode="mov", modifiers=(".s32",),
            operands=(RegisterOperand(loop_var), ImmediateOperand(str(start))),
        ))
    # $loop:
    ctx.emit(Label(name=loop_lbl))
    # setp.ge.s32 %pN, %rN, stop
    ctx.emit(Instruction(
        opcode="setp", modifiers=(".ge", ".s32"),
        operands=(RegisterOperand(p_name), RegisterOperand(loop_var), ImmediateOperand(str(stop))),
    ))
    # @%pN bra $endloop
    ctx.emit(Instruction(
        opcode="bra", operands=(LabelOperand(end_lbl),),
        predicate=Predicate(register=p_name, negated=False),
    ))

    k = Reg(loop_var, s32)
    yield k

    # add.s32 %rN, %rN, step
    ctx.emit(Instruction(
        opcode="add", modifiers=(".s32",),
        operands=(RegisterOperand(loop_var), RegisterOperand(loop_var), ImmediateOperand(str(step))),
    ))
    # bra $loop
    ctx.emit(Instruction(opcode="bra", operands=(LabelOperand(loop_lbl),)))
    # $endloop:
    ctx.emit(Label(name=end_lbl))


# ===================================================================
# Raw escape hatch
# ===================================================================

def raw(text: str) -> None:
    """Emit a raw PTX instruction string.

    Usage: ptx.raw("tcgen05.mma.cta_group::1.kind::f16 ...;")

    Parses the text and records the resulting IR instruction(s).
    """
    from pyptx.parser import parse as _parse

    ctx = get_ctx()
    wrapper = (
        f".version 8.5\n.target sm_90a\n.address_size 64\n"
        f".visible .entry _raw()\n{{\n\t{text}\n}}\n"
    )
    module = _parse(wrapper)
    func = module.directives[0]
    for stmt in func.body:
        ctx.emit(stmt)


# ===================================================================
# Instruction wrappers — each emits exactly one PTX instruction
# ===================================================================

# -- wgmma namespace -------------------------------------------------------

class _Wgmma:
    """ptx.wgmma.mma_async(...), ptx.wgmma.fence(), etc."""

    MASKED_DESC_B128 = 0x4000004000010000

    def mma_async(
        self,
        *,
        shape: tuple[int, int, int],
        dtype_d: PtxType,
        dtype_a: PtxType,
        dtype_b: PtxType,
        d: RegArray,
        a: Any,
        b: Any,
        scale_d: "Reg | bool | int" = False,
        scale_a: int = 1,
        scale_b: int = 1,
        trans_a: int = 0,
        trans_b: int = 0,
        a_k_offset: int = 0,
        b_k_offset: int = 0,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit ``wgmma.mma_async.sync.aligned.{shape}.{dtype_d}.{dtype_a}.{dtype_b}``.

        Operand layout for the dense ``.f32.bf16.bf16`` / ``.f32.f16.f16``
        form (from PTX ISA §9.7.14.5)::

            wgmma.mma_async ... d-vec, a-desc, b-desc, scale-d,
                                imm-scale-a, imm-scale-b,
                                imm-trans-a, imm-trans-b;

        - ``d``: register vector holding the accumulator.
        - ``a``, ``b``: u64 shared-memory descriptor registers (see
          :meth:`make_descriptor`).
        - ``scale_d``: ``.pred`` operand. ``False`` (default) means the
          instruction computes ``D = A * B`` (fresh accumulator, ignoring
          whatever was in D). ``True`` means ``D = D + A * B``
          (accumulate into existing D). You may also pass a Reg of
          dtype ``pred`` for runtime selection.
        - ``scale_a``, ``scale_b``: ``.s32`` immediates, must be 1 or -1.
          -1 negates the corresponding operand.
        - ``trans_a``, ``trans_b``: ``.s32`` immediates, 0 or 1.
          Transpose flag for A / B.
        """
        m, n, k = shape
        mods = (
            ".mma_async", ".sync", ".aligned",
            f".m{m}n{n}k{k}",
            dtype_d.ptx, dtype_a.ptx, dtype_b.ptx,
        )

        # If a/b are SharedAllocs carrying gmma_layout metadata, build
        # the descriptor automatically from their canonical layout.
        # Otherwise assume the caller already passed a u64 Reg
        # descriptor (the lower-level API path).
        a = self._resolve_operand(
            a, operand="A", default_shape=(m, k),
            dtype=dtype_a, default_major="K",
            addr_byte_offset=a_k_offset,
        )
        b = self._resolve_operand(
            b, operand="B", default_shape=(k, n),
            dtype=dtype_b, default_major="MN",
            addr_byte_offset=b_k_offset,
        )

        # Resolve scale_d into a .pred Reg. bool / int → allocate and
        # initialize a fresh pred; Reg → use as-is.
        if isinstance(scale_d, Reg):
            scale_d_reg = scale_d
        else:
            from pyptx.reg import scalar as reg_scalar
            from pyptx.types import pred as pred_type
            scale_d_reg = reg_scalar(pred_type)
            # setp.ne.b32 p, <1 or 0>, 0  sets p = (const != 0)
            # Equivalent shortcut: setp with immediate values.
            const = 1 if bool(scale_d) else 0
            # Use a scratch u32 register to hold the constant.
            from pyptx.types import u32
            tmp = reg_scalar(u32)
            inst.mov.u32(tmp, const)
            inst.setp.ne.b32(scale_d_reg, tmp, 0)

        operands: list[Any] = []
        # d is a register array OR a list of registers → vector operand
        if isinstance(d, list):
            operands.append(d)
        else:
            operands.append([d[i] for i in range(d.count)])
        operands.append(a)
        operands.append(b)
        operands.append(scale_d_reg)
        operands.append(scale_a)
        operands.append(scale_b)
        operands.append(trans_a)
        operands.append(trans_b)
        _emit("wgmma", mods, tuple(operands), pred=pred)

    def fence(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit wgmma.fence.sync.aligned;"""
        _emit("wgmma", (".fence", ".sync", ".aligned"), (), pred=pred)

    def commit_group(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit wgmma.commit_group.sync.aligned;"""
        _emit("wgmma", (".commit_group", ".sync", ".aligned"), (), pred=pred)

    def wait_group(self, n: int, *, pred: Reg | NegPred | None = None) -> None:
        """Emit wgmma.wait_group.sync.aligned N;"""
        _emit("wgmma", (".wait_group", ".sync", ".aligned"), (n,), pred=pred)

    # -- Shared memory descriptor builder ------------------------------------

    # Swizzle mode encoding for the wgmma shared memory descriptor.
    # These match the values expected in bits [63:62] of the u64 desc.
    SWIZZLE_NONE = 0
    SWIZZLE_128B = 1
    SWIZZLE_64B = 2
    SWIZZLE_32B = 3

    def _resolve_operand(
        self,
        val: Any,
        *,
        operand: str,
        default_shape: tuple[int, int],
        dtype: Any,
        default_major: str,
        addr_byte_offset: int = 0,
    ) -> Any:
        """Convert an A/B operand to a u64 descriptor Reg if the caller
        gave us enough info, otherwise pass it through unchanged.

        ``addr_byte_offset`` shifts the descriptor's start_addr field,
        enabling sub-tile access within a larger SMEM allocation (used
        by BK > 16 GEMMs that do multiple wgmma calls per K-tile).
        """
        if isinstance(val, Reg):
            return val

        from pyptx.smem import SharedAlloc, SharedSlice
        alloc = val.alloc if isinstance(val, SharedSlice) else val

        if isinstance(alloc, SharedAlloc) and alloc.gmma_layout is not None:
            layout = alloc.gmma_layout
            return self.make_descriptor(
                val,
                leading_byte_offset=layout.leading_byte_offset,
                stride_byte_offset=layout.stride_byte_offset,
                swizzle=layout.swizzle_code,
                addr_byte_offset=addr_byte_offset,
            )

        return val

    def auto_descriptor(
        self,
        smem_base: Any,
        *,
        dtype: Any,
        shape: tuple[int, int],
        major: str | int,
    ) -> "Reg":
        """Build a wgmma descriptor from a shape + dtype + major hint.

        Zero magic-number API: pyptx picks the right canonical GMMA
        layout (INTERLEAVE / B32 / B64 / B128) and derives the
        leading/stride byte offsets automatically.

        Args:
            smem_base: ``SharedAlloc`` / ``SharedSlice`` / ``Reg``
                pointing at the start of the shared memory tile. The
                caller is responsible for allocating the tile with a
                matching ``swizzle=...`` (see the returned layout for
                the right name) and, if the tile is TMA-loaded, for
                using the matching ``Layout.TMA_*B`` in the @kernel
                Tile spec — swizzle on write and swizzle on read must
                be the same or the data comes back permuted.
            dtype: pyptx element type (``bf16``, ``f16``, ``f32``...).
            shape: ``(M, K)`` for A or ``(K, N)`` for B.
            major: ``"K"`` (row-major A / col-major B) or ``"MN"``
                (col-major A / row-major B). Must match the
                ``trans_a`` / ``trans_b`` flags on the subsequent
                ``wgmma.mma_async`` call.

        Returns: a fresh ``Reg`` holding the 64-bit descriptor.
        """
        from pyptx.wgmma_layout import Major, pick_gmma_layout
        if isinstance(major, str):
            major_enum = Major.K if major.upper() == "K" else Major.MN
        else:
            major_enum = Major(major)
        m_or_n, k = int(shape[0]), int(shape[1])
        if major_enum == Major.MN:
            m_or_n = int(shape[1])  # B's N dim is shape[1]
            k = int(shape[0])
        layout = pick_gmma_layout(
            elem_bytes=max(dtype.bits // 8, 1),
            m_or_n=m_or_n,
            k=k,
            major=major_enum,
        )
        return self.make_descriptor(
            smem_base,
            leading_byte_offset=layout.leading_byte_offset,
            stride_byte_offset=layout.stride_byte_offset,
            swizzle=layout.swizzle_code,
        )

    def make_descriptor(
        self,
        smem_base: Any,
        *,
        leading_byte_offset: int,
        stride_byte_offset: int,
        swizzle: int = 0,
        base_offset: int = 0,
        addr_byte_offset: int = 0,
    ) -> "Reg":
        """Build a wgmma shared-memory descriptor in a fresh u64 register.

        wgmma.mma_async takes A and B as 64-bit descriptors that encode:
          bits [13:0]   start_addr      = shared_addr >> 4
          bits [29:16]  leading_offset  = leading_byte_offset >> 4
          bits [45:32]  stride_offset   = stride_byte_offset >> 4
          bits [51:49]  base_offset     (0 unless swizzle requires it)
          bits [63:62]  swizzle mode    (0=none, 1=128B, 2=64B, 3=32B)

        ``addr_byte_offset`` is added to the smem_base address before
        the start_addr field is computed. This is how sub-tile descriptors
        work for BK > 16 GEMMs: each of the 4 wgmma calls within a
        K-tile references a different 16-column slice of the A and B
        allocations by adding ``j * slice_bytes`` to the base.

        This helper emits PTX that computes the descriptor at kernel
        runtime by taking the shared memory base address (which is
        known to ptxas as a relocatable symbol) and OR-ing in the
        compile-time-constant leading/stride/base/swizzle fields.

        Args:
            smem_base: a ``SharedAlloc`` / ``SharedSlice`` / ``Reg`` holding
                (or naming) the shared memory base the descriptor refers to.
                If a ``SharedAlloc``/``SharedSlice`` is passed, we emit an
                extra ``mov.u64`` to lift the symbolic name into a register.
            leading_byte_offset: constant int — the leading dimension stride
                of the matrix tile in bytes. For a row-major 16x8 bf16
                tile this is 16 (one row).
            stride_byte_offset: constant int — the stride between "core
                matrices" in the tile. For a 16x8 bf16 tile that's split
                into two 8x8 core matrices vertically, this is 128
                (8 rows * 16 bytes/row).
            swizzle: one of ``SWIZZLE_{NONE,128B,64B,32B}``.
            base_offset: swizzle base offset (0-7); 0 for most uses.

        Returns: a fresh ``Reg`` of dtype ``u64`` holding the descriptor,
        usable directly as ``a=`` / ``b=`` to ``wgmma.mma_async``.
        """
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import u64

        if not (0 <= swizzle <= 3):
            raise ValueError(f"wgmma swizzle must be 0..3, got {swizzle}")
        if not (0 <= base_offset <= 7):
            raise ValueError(f"wgmma base_offset must be 0..7, got {base_offset}")

        # Step 1: get the shared memory base address into a u64 register.
        # For a SharedAlloc / SharedSlice, emit mov.u64 of the PTX
        # symbol. If the alloc has a byte_offset (dynamic SMEM mode),
        # add it so the descriptor points at the right sub-region.
        if isinstance(smem_base, Reg):
            # Ensure addr_reg is u64 (SMEM offsets may be u32)
            from pyptx.types import u64 as _u64_type
            if smem_base.dtype.bits < 64:
                addr_reg = reg_scalar(u64)
                inst.cvt.u64.u32(addr_reg, smem_base)
            else:
                addr_reg = smem_base
        else:
            addr_reg = reg_scalar(u64)
            name = _addr_base_name(smem_base)
            inst.mov.u64(addr_reg, name)
            # Dynamic SMEM offset: only for allocs named "dyn_smem".
            # Static allocs (smem_N) already have correct addresses
            # from their PTX symbol — adding byte_offset would
            # double-count.
            from pyptx.smem import SharedAlloc, SharedSlice
            alloc_obj = smem_base.alloc if isinstance(smem_base, SharedSlice) else smem_base
            if isinstance(alloc_obj, SharedAlloc) and alloc_obj.name == "dyn_smem" and alloc_obj.byte_offset > 0:
                off_reg = reg_scalar(u64)
                inst.mov.b64(off_reg, alloc_obj.byte_offset)
                inst.add.s64(addr_reg, addr_reg, off_reg)

        # Step 1b: apply sub-tile byte offset if given. Always use a
        # FRESH register so we don't mutate the caller's Reg in place
        # (which would cause cumulative drift when called in a loop).
        if addr_byte_offset != 0:
            shifted = reg_scalar(u64)
            inst.add.s64(shifted, addr_reg, addr_byte_offset)
            addr_reg = shifted

        # Step 2: compute the constant part of the descriptor.
        #
        # Bits [13:0]  are filled at runtime from addr_reg >> 4.
        # Bits [15:14] are reserved (zero).
        # Bits [29:16] encode leading_byte_offset >> 4.
        # Bits [31:30] are reserved.
        # Bits [45:32] encode stride_byte_offset >> 4.
        # Bits [48:46] are reserved.
        # Bits [51:49] encode base_offset.
        # Bits [61:52] are reserved.
        # Bits [63:62] encode swizzle.
        const_part: int = 0
        const_part |= ((leading_byte_offset >> 4) & 0x3FFF) << 16
        const_part |= ((stride_byte_offset >> 4) & 0x3FFF) << 32
        const_part |= (base_offset & 0x7) << 49
        const_part |= (swizzle & 0x3) << 62

        # Step 3: emit the PTX arithmetic:
        #   shr.u64 %desc, %addr, 4;
        #   and.b64 %desc, %desc, 0x3FFF;
        #   mov.b64 %c, const_part;
        #   or.b64  %desc, %desc, %c;
        desc = reg_scalar(u64)
        inst.shr.u64(desc, addr_reg, 4)
        inst.and_.b64(desc, desc, 0x3FFF)

        const_reg = reg_scalar(u64)
        inst.mov.b64(const_reg, const_part)
        inst.or_.b64(desc, desc, const_reg)
        return desc

    def masked_descriptor(
        self,
        smem_addr: Any,
        *,
        byte_offset: int = 0,
        mask: int = 0x3FF80,
        const_bits: int = MASKED_DESC_B128,
    ) -> "Reg":
        """Build a descriptor from a computed shared-memory address.

        This is the lower-level Hopper GEMM pattern used by handwritten
        kernels that derive descriptors from lane/stage-specific shared
        addresses:

          tmp  = smem_addr + byte_offset
          bits = tmp & mask
          idx  = bits >> 4
          desc = cvt.u64.u32(idx) | const_bits
        """
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import u32, u64

        if isinstance(smem_addr, Reg):
            base = smem_addr
        else:
            base = reg_scalar(u32)
            inst.mov.u32(base, _addr_base_name(smem_addr))

        tmp = base + byte_offset if byte_offset != 0 else base
        bits = tmp & mask
        idx = bits >> 4
        desc = reg_scalar(u64)
        inst.cvt.u64.u32(desc, idx)
        inst.or_.b64(desc, desc, const_bits)
        return desc


wgmma = _Wgmma()


# -- cp.async namespace ----------------------------------------------------

class _TmaTensorOp:
    """Callable object that also has a .store attribute.

    Usage:
        ptx.cp.async_.bulk.tensor_2d(dst=..., src=..., coord=..., mbar=...)      # load
        ptx.cp.async_.bulk.tensor_2d.store(dst=..., src=..., coord=...)          # store
    """

    def __init__(self, ndim: int) -> None:
        self._ndim = ndim

    def __call__(
        self,
        *,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...] | None = None,
        mbar: MbarrierRef | None = None,
        cta_group: int = 1,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit ``cp.async.bulk.tensor.Nd[.cta_group::K].shared::cluster.global.mbarrier::
        complete_tx::bytes[.multicast::cluster] [dst], [tensorMap, {c0, c1, ...}], [mbar][, mask];``

        - ``dst``: shared-memory destination (SharedSlice / SharedAlloc /
          Reg holding a shared address).
        - ``src``: TMA descriptor source — either a Reg loaded via
          ``ld.param.u64`` from a TMA desc param slot, a TensorSpec whose
          ``tma_desc()`` was called earlier in the trace, or a raw
          address literal/string.
        - ``coord``: N-tuple of tile coordinates (ints or Regs).
        - ``mbar``: MbarrierRef the TMA will arrive on.
        - ``cta_group``: 2 enables the Blackwell 2-SM cooperative load.
        - ``multicast_mask``: enables ``.multicast::cluster`` and supplies
          the u16 lane bitmask of participating CTAs.
        """
        if coord is None:
            raise ValueError(
                f"cp.async.bulk.tensor.{self._ndim}d requires coord=(c0, ...)"
            )
        if len(coord) != self._ndim:
            raise ValueError(
                f"cp.async.bulk.tensor.{self._ndim}d expects {self._ndim} "
                f"coordinates, got {len(coord)}"
            )
        if cta_group not in (1, 2):
            raise ValueError(f"cta_group must be 1 or 2, got {cta_group}")
        operands: list[Any] = [
            _make_address(dst),
            _make_tma_address(src, tuple(coord)),
        ]
        if mbar is not None:
            operands.append(_make_address(mbar))
        mods: list[str] = [".async", ".bulk", ".tensor", f".{self._ndim}d"]
        if cta_group == 2:
            mods.append(".cta_group::2")
        mods.extend([
            ".shared::cluster", ".global",
            ".mbarrier::complete_tx::bytes",
        ])
        if multicast_mask is not None:
            mods.append(".multicast::cluster")
            operands.append(multicast_mask)
        _emit("cp", tuple(mods), tuple(operands), pred=pred)

    def store(
        self,
        *,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...] | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit ``cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group
        [tensorMap, {c0, c1, ...}], [src];``

        - ``dst``: TMA descriptor for the destination tensor.
        - ``src``: shared-memory source (SharedSlice / SharedAlloc).
        - ``coord``: N-tuple of tile coordinates (ints or Regs).
        """
        if coord is None:
            raise ValueError(
                f"cp.async.bulk.tensor.{self._ndim}d.store requires coord=(c0, ...)"
            )
        if len(coord) != self._ndim:
            raise ValueError(
                f"cp.async.bulk.tensor.{self._ndim}d.store expects "
                f"{self._ndim} coordinates, got {len(coord)}"
            )
        operands: tuple[Any, ...] = (
            _make_tma_address(dst, tuple(coord)),
            _make_address(src),
        )
        mods = (
            ".async", ".bulk", ".tensor", f".{self._ndim}d",
            ".global", ".shared::cta",
            ".bulk_group",
        )
        _emit("cp", mods, operands, pred=pred)

    def shared_cta_global_tile(
        self,
        *,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...] | None = None,
        mbar: Any,
        cta_group: int = 1,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit ``cp.async.bulk.tensor.Nd.shared::cta.global.tile``.

        This is the Blackwell collective-TMA form used by Mosaic/Pallas for
        leader-tracked partitioned loads. The mbarrier operand is typically a
        cluster-mapped shared address produced by ``mapa.shared::cluster``.
        """
        if coord is None:
            raise ValueError(
                f"cp.async.bulk.tensor.{self._ndim}d.shared_cta_global_tile "
                "requires coord=(c0, ...)"
            )
        if len(coord) != self._ndim:
            raise ValueError(
                f"cp.async.bulk.tensor.{self._ndim}d.shared_cta_global_tile "
                f"expects {self._ndim} coordinates, got {len(coord)}"
            )
        if cta_group not in (1, 2):
            raise ValueError(f"cta_group must be 1 or 2, got {cta_group}")
        mods: list[str] = [
            ".async", ".bulk", ".tensor", f".{self._ndim}d",
            ".shared::cta", ".global", ".tile",
            ".mbarrier::complete_tx::bytes",
        ]
        if cta_group == 2:
            mods.append(".cta_group::2")
        operands: tuple[Any, ...] = (
            _make_address(dst),
            _make_tma_address(src, tuple(coord)),
            _make_address(mbar),
        )
        _emit("cp", tuple(mods), operands, pred=pred)


class _CpAsyncBulkTensor:
    """ptx.cp.async.bulk.tensor_2d(...) etc."""

    tensor_1d = _TmaTensorOp(1)
    tensor_2d = _TmaTensorOp(2)
    tensor_3d = _TmaTensorOp(3)
    tensor_4d = _TmaTensorOp(4)
    tensor_5d = _TmaTensorOp(5)

    # ----- TMA load tile_Nd family (global -> shared::cluster) -----

    def _tile_load(
        self,
        ndim: int,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        mbar: Any | None,
        cache_hint: Any | None,
        multicast_mask: Any | None,
        pred: Reg | NegPred | None,
        *,
        mode: str = "tile",
        cta_group: int = 1,
    ) -> None:
        """Internal: emit cp.async.bulk.tensor.{N}d[.cta_group::K].shared::cluster.global ... ;

        ``cta_group=2`` emits the 2-SM cooperative variant used by Blackwell
        collective MMA kernels: each CTA in the cluster issues the same
        instruction with the same coords, and the hardware distributes the
        load across both CTAs. The mbarrier arrive fires once per CTA via
        ``.shared::cluster`` and — when paired with ``multicast_mask`` —
        ``.multicast::cluster`` so both CTAs' mbars transition together.
        """
        if len(coord) != ndim:
            raise ValueError(
                f"cp.async.bulk.tensor.{ndim}d expects {ndim} coordinates, got {len(coord)}"
            )
        if cta_group not in (1, 2):
            raise ValueError(f"cta_group must be 1 or 2, got {cta_group}")
        mods: list[str] = [
            ".async", ".bulk", ".tensor",
            f".{ndim}d",
        ]
        if cta_group == 2:
            mods.append(".cta_group::2")
        mods.extend([
            ".shared::cluster", ".global",
            ".mbarrier::complete_tx::bytes",
        ])
        if mode != "tile":
            # im2col modifier comes between Nd and the state spaces in PTX, but
            # the parser captures all dot-modifiers as a flat tuple — preserve order.
            mods.append(f".{mode}")
        if multicast_mask is not None:
            mods.append(".multicast::cluster")
        if cache_hint is not None:
            mods.append(".L2::cache_hint")
        operands: list[Any] = [
            _make_tma_address(dst, coord),
            _make_address(src),
        ]
        if mbar is not None:
            operands.append(_make_address(mbar))
        if multicast_mask is not None:
            operands.append(multicast_mask)
        if cache_hint is not None:
            operands.append(cache_hint)
        _emit("cp", tuple(mods), tuple(operands), pred=pred)

    def tile_1d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes ... ;"""
        self._tile_load(1, dst, src, coord, mbar, cache_hint, multicast_mask, pred)

    def tile_2d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        cta_group: int = 1,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.2d[.cta_group::K].shared::cluster.global.mbarrier::complete_tx::bytes ... ;"""
        self._tile_load(2, dst, src, coord, mbar, cache_hint, multicast_mask, pred, cta_group=cta_group)

    def tile_3d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes ... ;"""
        self._tile_load(3, dst, src, coord, mbar, cache_hint, multicast_mask, pred)

    def tile_4d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes ... ;"""
        self._tile_load(4, dst, src, coord, mbar, cache_hint, multicast_mask, pred)

    def tile_5d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes ... ;"""
        self._tile_load(5, dst, src, coord, mbar, cache_hint, multicast_mask, pred)

    # ----- im2col variants (3d/4d/5d only — TMA spec) -----

    def im2col_3d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.3d.shared::cluster.global.im2col ... ;"""
        self._tile_load(3, dst, src, coord, mbar, cache_hint, multicast_mask, pred, mode="im2col")

    def im2col_4d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.4d.shared::cluster.global.im2col ... ;"""
        self._tile_load(4, dst, src, coord, mbar, cache_hint, multicast_mask, pred, mode="im2col")

    def im2col_5d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.5d.shared::cluster.global.im2col ... ;"""
        self._tile_load(5, dst, src, coord, mbar, cache_hint, multicast_mask, pred, mode="im2col")

    # ----- scatter / gather (Blackwell, 2d) -----

    def scatter4_2d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.2d.global.shared::cta.scatter4 ... ;"""
        if len(coord) != 2:
            raise ValueError(f"scatter4_2d expects 2 coordinates, got {len(coord)}")
        mods = (
            ".async", ".bulk", ".tensor", ".2d",
            ".global", ".shared::cta", ".scatter4",
        )
        operands: list[Any] = [_make_address(dst), _make_tma_address(src, coord)]
        if mbar is not None:
            operands.append(_make_address(mbar))
        _emit("cp", mods, tuple(operands), pred=pred)

    def gather4_2d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        mbar: Any | None = None,
        cache_hint: Any | None = None,
        multicast_mask: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.2d.shared::cluster.global.gather4 ... ;"""
        if len(coord) != 2:
            raise ValueError(f"gather4_2d expects 2 coordinates, got {len(coord)}")
        mods: list[str] = [
            ".async", ".bulk", ".tensor", ".2d",
            ".shared::cluster", ".global", ".gather4",
            ".mbarrier::complete_tx::bytes",
        ]
        if multicast_mask is not None:
            mods.append(".multicast::cluster")
        if cache_hint is not None:
            mods.append(".L2::cache_hint")
        operands: list[Any] = [
            _make_tma_address(dst, coord),
            _make_address(src),
        ]
        if mbar is not None:
            operands.append(_make_address(mbar))
        if multicast_mask is not None:
            operands.append(multicast_mask)
        if cache_hint is not None:
            operands.append(cache_hint)
        _emit("cp", tuple(mods), tuple(operands), pred=pred)

    # ----- TMA store store_Nd family (shared -> global) -----

    def _tile_store(
        self,
        ndim: int,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        cache_hint: Any | None,
        pred: Reg | NegPred | None,
    ) -> None:
        if len(coord) != ndim:
            raise ValueError(
                f"cp.async.bulk.tensor.{ndim}d store expects {ndim} coordinates, got {len(coord)}"
            )
        mods: list[str] = [
            ".async", ".bulk", ".tensor",
            f".{ndim}d",
            ".global", ".shared::cta",
        ]
        if cache_hint is not None:
            mods.append(".L2::cache_hint")
        operands: list[Any] = [
            _make_tma_address(dst, coord),
            _make_address(src),
        ]
        if cache_hint is not None:
            operands.append(cache_hint)
        _emit("cp", tuple(mods), tuple(operands), pred=pred)

    def store_1d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        cache_hint: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.1d.global.shared::cta ... ;"""
        self._tile_store(1, dst, src, coord, cache_hint, pred)

    def store_2d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        cache_hint: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.2d.global.shared::cta ... ;"""
        self._tile_store(2, dst, src, coord, cache_hint, pred)

    def store_3d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        cache_hint: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.3d.global.shared::cta ... ;"""
        self._tile_store(3, dst, src, coord, cache_hint, pred)

    def store_4d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        cache_hint: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.4d.global.shared::cta ... ;"""
        self._tile_store(4, dst, src, coord, cache_hint, pred)

    def store_5d(
        self,
        dst: Any,
        src: Any,
        coord: tuple[Any, ...],
        *,
        cache_hint: Any | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit cp.async.bulk.tensor.5d.global.shared::cta ... ;"""
        self._tile_store(5, dst, src, coord, cache_hint, pred)


# ---------- helpers used by the cp.async.bulk.tensor wrappers ----------

def _addr_base_name(val: Any) -> str:
    """Extract the base register/symbol name from a Reg/AddressOperand/str/etc."""
    if isinstance(val, AddressOperand):
        return val.base
    if isinstance(val, Reg):
        return val.name
    if isinstance(val, RegisterOperand):
        return val.name
    if isinstance(val, MbarrierRef | SharedAlloc | SharedSlice):
        return val.name
    if isinstance(val, str):
        return val
    # TensorSpec / TmaDescriptorHandle from pyptx.kernel — use duck typing
    # to avoid a circular import.
    if hasattr(val, "name") and isinstance(val.name, str):
        return val.name
    raise TypeError(f"Cannot get address base from {type(val).__name__}: {val!r}")


def _coord_text(coord: tuple[Any, ...]) -> str:
    """Render a TMA coordinate tuple to the inline ", {c0, c1, ...}" form."""
    parts: list[str] = []
    for c in coord:
        if isinstance(c, Reg):
            parts.append(c.name)
        elif isinstance(c, RegisterOperand):
            parts.append(c.name)
        elif isinstance(c, int):
            parts.append(str(c))
        elif isinstance(c, str):
            parts.append(c)
        elif isinstance(c, ImmediateOperand):
            parts.append(c.text)
        else:
            raise TypeError(f"Unsupported TMA coordinate {type(c).__name__}: {c!r}")
    return ", {" + ", ".join(parts) + "}"


def _make_tma_address(val: Any, coord: tuple[Any, ...]) -> AddressOperand:
    """Build an AddressOperand for [base, {c0, c1, ...}]."""
    return AddressOperand(base=_addr_base_name(val), offset=_coord_text(coord))


def _dyn_smem_addr_reg(byte_offset: int) -> Reg:
    """Emit instructions to compute a u32 register holding ``dyn_smem + byte_offset``.

    PTX shared-memory instructions like ``mbarrier.init`` and
    ``cp.async.bulk.tensor`` require a register operand for the
    address when using extern dynamic shared memory. This helper
    materializes the address in a fresh u32 register.
    """
    from pyptx._trace import get_ctx
    from pyptx.reg import scalar as _reg_scalar
    from pyptx.types import u32 as _u32
    ctx = get_ctx()
    addr = _reg_scalar(_u32)
    ctx.emit(Instruction(
        opcode="mov", modifiers=(".b32",),
        operands=(RegisterOperand(addr.name), RegisterOperand("dyn_smem")),
    ))
    if byte_offset != 0:
        ctx.emit(Instruction(
            opcode="add", modifiers=(".s32",),
            operands=(
                RegisterOperand(addr.name),
                RegisterOperand(addr.name),
                ImmediateOperand(str(byte_offset)),
            ),
        ))
    return addr


def _make_address(val: Any) -> AddressOperand:
    """Wrap a Reg / SharedAlloc / etc. in an AddressOperand if needed.

    For dynamic SMEM allocs (name == "dyn_smem"), emits register-based
    addressing: ``mov.b32 %r, dyn_smem; add.s32 %r, %r, offset;`` and
    returns ``[%r]``. This is required because PTX shared-memory
    instructions need a register address for extern dynamic SMEM.
    """
    if isinstance(val, AddressOperand):
        return val
    if hasattr(val, "address_operand") and callable(val.address_operand):
        return val.address_operand()
    from pyptx.smem import MbarrierRef, SharedAlloc, SharedSlice

    # Handle MbarrierRef with dynamic SMEM
    if isinstance(val, MbarrierRef) and val.name == "dyn_smem":
        addr_reg = _dyn_smem_addr_reg(val.byte_offset)
        return AddressOperand(base=addr_reg.name, offset=None)

    if isinstance(val, MbarrierRef) and val.byte_offset != 0:
        return AddressOperand(base=val.name, offset=str(val.byte_offset))

    # Handle SharedAlloc / SharedSlice with dynamic SMEM
    alloc_obj = val.alloc if isinstance(val, SharedSlice) else val
    if isinstance(alloc_obj, SharedAlloc) and alloc_obj.name == "dyn_smem":
        addr_reg = _dyn_smem_addr_reg(alloc_obj.byte_offset)
        return AddressOperand(base=addr_reg.name, offset=None)

    return AddressOperand(base=_addr_base_name(val), offset=None)


def _maybe_address(val: Any) -> Any:
    """Return an address operand for address-like values, else pass through raw."""
    if isinstance(val, AddressOperand):
        return val
    if hasattr(val, "address_operand") and callable(val.address_operand):
        return val.address_operand()
    from pyptx.smem import MbarrierRef, SharedAlloc, SharedSlice

    if isinstance(val, (MbarrierRef, SharedAlloc, SharedSlice)):
        return _make_address(val)
    return val


class _CpAsyncBulk:
    tensor_1d = _CpAsyncBulkTensor.tensor_1d
    tensor_2d = _CpAsyncBulkTensor.tensor_2d
    tensor_3d = _CpAsyncBulkTensor.tensor_3d
    tensor_4d = _CpAsyncBulkTensor.tensor_4d
    tensor_5d = _CpAsyncBulkTensor.tensor_5d

    @property
    def tensor(self) -> _CpAsyncBulkTensor:
        return _CpAsyncBulkTensor()

    def commit_group(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit ``cp.async.bulk.commit_group;``.

        Closes all preceding ``.bulk_group`` operations (e.g. TMA stores
        issued via ``ptx.cp.async_.bulk.tensor_2d.store(...)``) into a
        single commit group that can be waited on with
        ``wait_group``. Analogous to ``wgmma.commit_group``.
        """
        _emit("cp", (".async", ".bulk", ".commit_group"), (), pred=pred)

    def wait_group(
        self,
        n: int,
        *,
        read: bool = False,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit ``cp.async.bulk.wait_group{.read} N;``.

        Waits until at most N outstanding bulk commit groups remain. Use
        ``wait_group(0)`` after a TMA store to block until every
        bulk_group store has landed in global memory.

        ``read=True`` (``wait_group.read``) relaxes the wait — it only
        guarantees the outstanding writes are visible to subsequent
        reads, not that they are globally visible.
        """
        mods = [".async", ".bulk", ".wait_group"]
        if read:
            mods.append(".read")
        _emit("cp", tuple(mods), (ImmediateOperand(str(int(n))),), pred=pred)


class _CpAsync:
    @property
    def bulk(self) -> _CpAsyncBulk:
        return _CpAsyncBulk()


class _Cp:
    @property
    def async_(self) -> _CpAsync:
        return _CpAsync()

    # Convenience: ptx.cp.async.bulk.tensor_2d(...)
    @property
    def async_bulk(self) -> _CpAsyncBulk:
        return _CpAsyncBulk()


cp = _Cp()


# -- mbarrier namespace ----------------------------------------------------

def _materialize_u32_addr(base: Any, byte_offset: int = 0) -> Reg:
    """Materialize ``base + byte_offset`` into a fresh ``u32`` register."""
    from pyptx.reg import scalar as reg_scalar
    from pyptx.types import u32

    addr = reg_scalar(u32)
    if isinstance(base, Reg):
        if byte_offset == 0:
            inst.mov.b32(addr, base)
        else:
            inst.add.s32(addr, base, byte_offset)
        return addr

    inst.mov.u32(addr, _addr_base_name(base))
    if byte_offset != 0:
        inst.add.s32(addr, addr, byte_offset)
    return addr


class _BarrierSlot:
    """Single mbarrier slot rooted at a shared-memory base + byte offset."""

    __slots__ = ("_base", "_byte_offset", "_addr_reg")

    def __init__(self, base: Any, byte_offset: int) -> None:
        self._base = base
        self._byte_offset = byte_offset
        self._addr_reg: Reg | None = None

    @property
    def byte_offset(self) -> int:
        return self._byte_offset

    def address_operand(self) -> AddressOperand:
        if isinstance(self._base, Reg):
            return AddressOperand(base=self._base.name, offset=str(self._byte_offset))
        return AddressOperand(base=_addr_base_name(self._base), offset=str(self._byte_offset))

    def addr_reg(self) -> Reg:
        if self._addr_reg is not None:
            return self._addr_reg
        if isinstance(self._base, Reg):
            if self._byte_offset == 0:
                return self._base
            self._addr_reg = self._base + self._byte_offset
            return self._addr_reg
        self._addr_reg = _materialize_u32_addr(self._base, self._byte_offset)
        return self._addr_reg

    def init(self, count: int, *, pred: Reg | NegPred | None = None) -> None:
        mbarrier.init(self, count, pred=pred)

    def wait(self, phase: Reg | int, *, pred: Reg | NegPred | None = None) -> None:
        mbarrier.wait(self, phase, pred=pred)

    def arrive(self, *, pred: Reg | NegPred | None = None) -> Reg:
        return mbarrier.arrive(self, pred=pred)

    def arrive_expect_tx(
        self,
        tx_count: int | Reg,
        *,
        pred: Reg | NegPred | None = None,
    ) -> Reg:
        return mbarrier.arrive_expect_tx(self, tx_count, pred=pred)

    def arrive_remote(
        self,
        cta_id: Reg,
        count: Reg | int = 1,
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        cluster.arrive_remote(self.addr_reg(), cta_id, count, pred=pred)


class _DynamicBarrierSlot:
    """Barrier slot selected by a runtime stage index."""

    __slots__ = ("_base", "_array_offset", "_stage", "_addr_reg")

    def __init__(self, base: Any, array_offset: int, stage: Reg) -> None:
        self._base = base
        self._array_offset = array_offset
        self._stage = stage
        self._addr_reg: Reg | None = None

    def addr_reg(self) -> Reg:
        if self._addr_reg is not None:
            return self._addr_reg
        stage_bytes = self._stage << 3
        if self._array_offset != 0:
            stage_bytes += self._array_offset
        if isinstance(self._base, Reg):
            self._addr_reg = self._base + stage_bytes
        else:
            self._addr_reg = _materialize_u32_addr(self._base, 0) + stage_bytes
        return self._addr_reg

    def address_operand(self) -> AddressOperand:
        addr = self.addr_reg()
        return AddressOperand(base=addr.name, offset=None)

    def wait(self, phase: Reg | int, *, pred: Reg | NegPred | None = None) -> None:
        mbarrier.wait(self, phase, pred=pred)

    def arrive(self, *, pred: Reg | NegPred | None = None) -> Reg:
        return mbarrier.arrive(self, pred=pred)

    def arrive_expect_tx(
        self,
        tx_count: int | Reg,
        *,
        pred: Reg | NegPred | None = None,
    ) -> Reg:
        return mbarrier.arrive_expect_tx(self, tx_count, pred=pred)

    def arrive_remote(
        self,
        cta_id: Reg,
        count: Reg | int = 1,
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        cluster.arrive_remote(self.addr_reg(), cta_id, count, pred=pred)


class _BarrierArray:
    """Indexable mbarrier ring rooted at ``base + byte_offset``."""

    __slots__ = ("_base", "_byte_offset", "_count")

    def __init__(self, base: Any, byte_offset: int, count: int) -> None:
        self._base = base
        self._byte_offset = byte_offset
        self._count = count

    def __getitem__(self, idx: int) -> _BarrierSlot:
        if not 0 <= idx < self._count:
            raise IndexError(f"Barrier index {idx} out of range for count={self._count}")
        return _BarrierSlot(self._base, self._byte_offset + idx * 8)

    def at(self, stage: Reg) -> _DynamicBarrierSlot:
        return _DynamicBarrierSlot(self._base, self._byte_offset, stage)

    def init_all(self, count: int, *, pred: Reg | NegPred | None = None) -> None:
        """Initialize every slot in the array with the same arrive count."""
        for idx in range(self._count):
            self[idx].init(count, pred=pred)

    def arrive_remote_all(
        self,
        cta_id: Reg,
        count: Reg | int = 1,
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Issue the same remote arrive to every slot in the array."""
        for idx in range(self._count):
            self[idx].arrive_remote(cta_id, count, pred=pred)


class _Mbarrier:
    """ptx.mbarrier — Hopper mbarrier primitives.

    Contract for bracket wrapping: ``mbar`` arguments are always
    converted to address operands (``[mbar_0]``). State and predicate
    output registers are allocated inside the wrapper and returned from
    the ``arrive``/``arrive_expect_tx``/``try_wait`` calls so the caller
    can use them directly without boilerplate.
    """

    def init(
        self,
        mbar: Any,
        count: int,
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit ``mbarrier.init.shared::cta.b64 [mbar], count;``."""
        _emit(
            "mbarrier",
            (".init", ".shared::cta", ".b64"),
            (_make_address(mbar), ImmediateOperand(str(count))),
            pred=pred,
        )

    def inval(
        self,
        mbar: Any,
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit ``mbarrier.inval.shared.b64 [mbar];``."""
        _emit(
            "mbarrier",
            (".inval", ".shared", ".b64"),
            (_make_address(mbar),),
            pred=pred,
        )

    def arrive(
        self,
        mbar: Any,
        *,
        pred: Reg | NegPred | None = None,
    ) -> "Reg":
        """Emit ``mbarrier.arrive.shared.b64 state, [mbar];``.

        Returns the freshly allocated ``b64`` state register so callers
        can feed it to a subsequent wait if they need to. Users that
        don't care about the token can ignore the return value.
        """
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import b64
        state = reg_scalar(b64)
        _emit(
            "mbarrier",
            (".arrive", ".shared", ".b64"),
            (state, _make_address(mbar)),
            pred=pred,
        )
        return state

    def arrive_expect_tx(
        self,
        mbar: Any,
        tx_count: int | "Reg",
        *,
        pred: Reg | NegPred | None = None,
    ) -> "Reg":
        """Emit ``mbarrier.arrive.expect_tx.shared::cta.b64 state, [mbar], tx_count;``.

        Used by the thread that issues a TMA load: records the expected
        transaction-size so the mbarrier knows when the async bulk copy
        has fully completed. Returns the state register.
        """
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import b64
        state = reg_scalar(b64)
        _emit(
            "mbarrier",
            (".arrive", ".expect_tx", ".shared::cta", ".b64"),
            (state, _make_address(mbar), tx_count),
            pred=pred,
        )
        return state

    def expect_tx(
        self,
        mbar: Any,
        tx_count: int | "Reg",
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit ``mbarrier.expect_tx.shared.b64 [mbar], tx_count;``.

        The standalone form (no arrive, no state register output).
        """
        _emit(
            "mbarrier",
            (".expect_tx", ".shared", ".b64"),
            (_make_address(mbar), tx_count),
            pred=pred,
        )

    def try_wait(
        self,
        mbar: Any,
        phase: "Reg | int",
        *,
        parity: bool = True,
        pred: Reg | NegPred | None = None,
    ) -> "Reg":
        """Emit ``mbarrier.try_wait{.parity}.shared.b64 p, [mbar], phase;``.

        Returns the freshly allocated ``.pred`` register (``p``) that is
        true when the wait completed. The typical use is inside a busy
        loop that branches back to the try_wait label until ``p`` is true.

        When ``parity=True`` (the default) the instruction is the phase-
        bit flavor used with a single-bit phase register; use
        ``parity=False`` for the token-based form.
        """
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import pred as pred_type
        p = reg_scalar(pred_type)
        mods = [".try_wait"]
        if parity:
            mods.append(".parity")
        mods.extend((".shared", ".b64"))
        _emit(
            "mbarrier",
            tuple(mods),
            (p, _make_address(mbar), phase),
            pred=pred,
        )
        return p

    def wait(
        self,
        mbar: Any,
        phase: "Reg | int",
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit a ``mbarrier.try_wait.parity`` spin loop that blocks the
        calling thread until the barrier completes.

        Produces roughly::

            wait_loop:
                mbarrier.try_wait.parity.shared.b64 p, [mbar], phase;
                @!p bra wait_loop;

        If you need non-blocking behavior, call ``try_wait`` directly and
        branch on its return value.
        """
        ctx = get_ctx()
        loop_label = ctx.fresh_label("mbar_wait")
        # loop:
        from pyptx.ir.nodes import Label
        ctx.emit(Label(name=loop_label))
        p = self.try_wait(mbar, phase, parity=True, pred=pred)
        # @!p bra loop;
        bra(loop_label, pred=~p)

    def array(self, base: Any, byte_offset: int, count: int) -> _BarrierArray:
        """Create an indexable barrier array rooted at ``base + byte_offset``."""
        return _BarrierArray(base, byte_offset, count)


mbarrier = _Mbarrier()


# -- fence namespace -------------------------------------------------------

class _Fence:
    def proxy_async(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit fence.proxy.async;"""
        _emit("fence", (".proxy", ".async"), (), pred=pred)

    def proxy_async_shared_cta(
        self, *, pred: Reg | NegPred | None = None
    ) -> None:
        """Emit ``fence.proxy.async.shared::cta;``.

        The narrower-scope variant required to make an mbarrier init
        (which is a generic shared-memory write) visible to the TMA
        async proxy. Without this scope, the async proxy's view of
        the mbarrier can lag the thread that init'd it, so
        ``cp.async.bulk.tensor.*`` with ``.mbarrier::complete_tx::bytes``
        silently signals a stale barrier and the corresponding
        ``mbarrier.try_wait`` never completes.

        Canonical Hopper pattern for a TMA-loaded pipeline stage::

            mbarrier.init [bar], count;
            fence.proxy.async.shared::cta;
            mbarrier.arrive.expect_tx [bar], tx_bytes;
            cp.async.bulk.tensor.Nd.shared::cluster.global...;
            mbarrier.try_wait.parity [bar], phase;
        """
        _emit("fence", (".proxy", ".async", ".shared::cta"), (), pred=pred)

    def proxy_async_generic_acquire_shared_cluster(
        self, *, pred: Reg | NegPred | None = None
    ) -> None:
        """Emit the cluster-scope acquire fence used by Mosaic GPU collectives.

        PTX spelling::

            fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;

        Mosaic inserts this after waiting on a cluster-visible hand-off before
        reusing a collective TMA pipeline slot. It is narrower than a generic
        fence and pairs with Blackwell cluster-shared async-proxy state.
        """
        pred_prefix = ""
        if pred is not None:
            pred_prefix = f"@{pred.name} "
        raw(
            pred_prefix
            + "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;"
        )

    def mbarrier_init(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit fence.mbarrier_init.release.cluster;"""
        _emit("fence", (".mbarrier_init", ".release", ".cluster"), (), pred=pred)


fence = _Fence()


# -- stmatrix ---------------------------------------------------------------

def stmatrix(
    *,
    smem: Any,
    regs: RegArray | list,
    layout: str = "x4",
    trans: bool = False,
    shape: str = "m8n8",
    pred: Reg | NegPred | None = None,
) -> None:
    """Emit stmatrix.sync.aligned.{shape}.{count}[.trans].shared.b16.

    The ``layout`` kwarg accepts compound forms like "x4.trans" for
    backwards compatibility with raw PTX-style strings — they're split
    on dots into separate modifiers.

    Args:
        smem: destination shared-memory address
        regs: source registers (RegArray or list of Regs)
        layout: either just "x4" / "x2" / "x1", or a compound like
                "x4.trans" which auto-sets ``trans=True``
        trans: whether to emit the .trans modifier (default False)
        shape: tile shape, defaults to "m8n8"
        pred: optional predicate
    """
    # Split compound layout strings (e.g. "x4.trans")
    layout_parts = layout.split(".")
    count = layout_parts[0]
    if len(layout_parts) > 1 and "trans" in layout_parts[1:]:
        trans = True

    mods: list[str] = [".sync", ".aligned", f".{shape}", f".{count}"]
    if trans:
        mods.append(".trans")
    mods.extend([".shared::cta", ".b16"])

    if isinstance(regs, list):
        reg_operand = regs
    else:
        reg_operand = [regs[i] for i in range(regs.count)]
    _emit("stmatrix", tuple(mods), (smem, reg_operand), pred=pred)


def stmatrix_x4_trans_f32_bf16(
    *,
    frag: RegArray | list["Reg"],
    smem_base: Any,
    lane: "Reg",
    row_stride: int,
    tmp_bf16: list["Reg"] | None = None,
    tmp_pack: list["Reg"] | None = None,
) -> None:
    """Pack an f32 fragment to bf16 and store it via ``stmatrix.x4.trans``."""
    from pyptx.reg import scalar as reg_scalar
    from pyptx.types import b16, b32, u32

    regs = frag.regs() if isinstance(frag, RegArray) else list(frag)
    if len(regs) % 8 != 0:
        raise ValueError("ptx.stmatrix_x4_trans_f32_bf16 requires len(frag) % 8 == 0")

    rs = tmp_bf16 if tmp_bf16 is not None else [reg_scalar(b16) for _ in range(8)]
    pack = tmp_pack if tmp_pack is not None else [reg_scalar(b32) for _ in range(4)]
    if len(rs) != 8 or len(pack) != 4:
        raise ValueError("tmp_bf16 must have 8 regs and tmp_pack must have 4 regs")

    lane32 = lane & 31
    warp = lane >> 5
    tid_off = reg_scalar(u32)
    inst.mad.lo.s32(tid_off, warp, 16, 0)
    r1 = lane32 & 7
    inst.mad.lo.s32(tid_off, r1, row_stride, tid_off)
    r2 = lane32 >> 4
    inst.mad.lo.s32(tid_off, r2, row_stride * 8, tid_off)
    r3 = lane32 & 8
    inst.add.s32(tid_off, tid_off, r3)
    stm_base = smem_base + (tid_off << 1)

    group_stride_bytes = row_stride * 16 * 2
    for group in range(len(regs) // 8):
        for k in range(8):
            with scope():
                inst.cvt.rn.bf16.f32(rs[k], regs[group * 8 + k])
        for k in range(4):
            inst.mov.b32(pack[k], [rs[k * 2], rs[k * 2 + 1]])
        stmatrix(
            smem=addr(stm_base + (group * group_stride_bytes)),
            regs=pack,
            layout="x4.trans",
        )


# -- ldmatrix ---------------------------------------------------------------

def ldmatrix(
    *,
    dst: RegArray | list[Reg],
    src: Any,
    layout: str = "x4",
    trans: bool = False,
    pred: Reg | NegPred | None = None,
) -> None:
    """Emit ldmatrix.sync.aligned.{layout}[.trans].shared.b16."""
    mods: list[str] = [".sync", ".aligned", f".{layout}"]
    if trans:
        mods.append(".trans")
    mods.extend([".shared", ".b16"])
    if isinstance(dst, RegArray):
        dst_op = [dst[i] for i in range(dst.count)]
    else:
        dst_op = dst
    _emit("ldmatrix", tuple(mods), (dst_op, src), pred=pred)


# -- bar (barrier) ----------------------------------------------------------

class _Bar:
    def sync(
        self,
        n: Any = 0,
        count: Any | None = None,
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit bar.sync N;"""
        operands = (n,) if count is None else (n, count)
        _emit("bar", (".sync",), operands, pred=pred)


bar = _Bar()


class NamedBarrier:
    """Named CTA barrier with an optional participant count."""

    __slots__ = ("barrier_id", "count")

    def __init__(self, barrier_id: Any, count: Any | None = None) -> None:
        self.barrier_id = barrier_id
        self.count = count

    def sync(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit ``bar.sync`` using the stored barrier id and participant count."""
        bar.sync(self.barrier_id, self.count, pred=pred)


def named_barrier(barrier_id: Any, *, count: Any | None = None) -> NamedBarrier:
    """Create a named ``bar.sync`` wrapper."""
    return NamedBarrier(barrier_id, count)


# -- Convenience wrappers for common patterns --------------------------------


def setmaxnreg(count: int, *, inc: bool = True, pred: "Reg | NegPred | None" = None) -> None:
    """Emit ``setmaxnreg.{inc|dec}.sync.aligned.u32 count;``

    Used for warp specialization: consumers increase registers (inc=True),
    producers decrease (inc=False).
    """
    direction = ".inc" if inc else ".dec"
    _emit("setmaxnreg", (direction, ".sync", ".aligned", ".u32"), (count,), pred=pred)



def _setp_ne_zero(dst: "Reg", value: "Reg") -> None:
    """Emit ``setp.ne.{type} dst, value, 0;`` with sane integer compare type."""
    cmp_type = value.dtype.ptx
    if cmp_type == ".b32":
        cmp_type = ".u32"
    elif cmp_type == ".b64":
        cmp_type = ".u64"
    _emit("setp", (".ne", cmp_type), (dst, value, 0))


def kloop(
    total: int | "Reg",
    *,
    unroll: int,
    body: Callable[[], None],
    loop_label: str = "kloop",
) -> None:
    """Emit an unrolled counted loop with a peeled tail ladder."""
    if unroll <= 0 or (unroll & (unroll - 1)) != 0:
        raise ValueError("ptx.kloop requires a positive power-of-two unroll")

    from pyptx.reg import Reg, scalar as reg_scalar
    from pyptx.types import pred as pred_type

    if isinstance(total, int):
        for _ in range(total):
            body()
        return
    if not isinstance(total, Reg):
        raise TypeError(f"ptx.kloop total must be int or Reg, got {type(total).__name__}")

    remainder = total & (unroll - 1)
    rounded = total - remainder

    with if_(total >= unroll):
        main_left = reg_scalar(total.dtype)
        _emit("mov", (total.dtype.ptx,), (main_left, rounded))
        keep_going = reg_scalar(pred_type)
        _setp_ne_zero(keep_going, main_left)
        with loop(loop_label, pred=keep_going):
            for _ in range(unroll):
                body()
            main_left -= unroll
            _setp_ne_zero(keep_going, main_left)

    def emit_tail(index: int) -> None:
        if index >= unroll:
            return
        with if_(remainder != (index - 1)):
            body()
            emit_tail(index + 1)

    with if_(remainder != 0):
        body()
        emit_tail(2)


def selp(
    dtype: "PtxType",
    dst: "Reg",
    true_val: Any,
    false_val: Any,
    pred_reg: "Reg",
    *,
    pred: "Reg | NegPred | None" = None,
) -> None:
    """Emit ``selp.{type} dst, true_val, false_val, pred;``

    Ternary select: ``dst = pred ? true_val : false_val``.
    """
    mod = f".{dtype.ptx.lstrip('.')}" if hasattr(dtype, 'ptx') else f".{dtype}"
    _emit("selp", (mod,), (dst, true_val, false_val, pred_reg), pred=pred)


    # cluster operations are defined later alongside the existing _Cluster class


class _Tma:
    """High-level TMA load/store with 3D layout.

    Wraps ``cp.async.bulk.tensor.3d`` with the coordinate convention
    used by fast.cu: ``{0, row, col/64}`` for the 3D tiled layout.
    """

    def load_3d(
        self,
        dst: Any,
        src: Any,
        row: Any = None,
        col: Any = None,
        mbar: Any = None,
        coords: tuple[Any, ...] | None = None,
        *,
        pred: "Reg | NegPred | None" = None,
    ) -> None:
        """TMA 3D load: ``cp.async.bulk.tensor.3d.shared::cluster.global...``

        ``col`` is automatically divided by 64 for the 3D coordinate.
        """
        from pyptx.reg import scalar as _scalar, Reg
        from pyptx.types import u32 as _u32

        if coords is not None:
            coord = coords
        else:
            if row is None or col is None:
                raise TypeError("ptx.tma.load_3d requires row/col or coords=(...)")
            # Compute col/64 = col >> 6
            if isinstance(col, Reg):
                col_div = _scalar(_u32)
                _emit("shr", (".u32",), (col_div, col, ImmediateOperand("6")))
                col_coord = col_div
            elif isinstance(col, int):
                col_coord = ImmediateOperand(str(col // 64))
            else:
                col_coord = col
            coord = (ImmediateOperand("0"), row, col_coord)

        _emit(
            "cp",
            (".async", ".bulk", ".tensor", ".3d",
             ".shared::cluster", ".global",
             ".tile",
             ".mbarrier::complete_tx::bytes"),
            (_make_address(dst),
             AddressOperand(base=_addr_base_name(src), offset=_coord_text(coord)),
             _make_address(mbar)),
            pred=pred,
        )

    def load_3d_multicast(
        self,
        dst: Any,
        src: Any,
        row: Any = None,
        col: Any = None,
        mbar: Any = None,
        mask: Any = None,
        coords: tuple[Any, ...] | None = None,
        *,
        issuer: int | Reg | NegPred | None = None,
        pred: "Reg | NegPred | None" = None,
    ) -> None:
        """TMA 3D load with cluster multicast."""
        from pyptx.reg import scalar as _scalar, Reg
        from pyptx.types import u32 as _u32

        if issuer is not None:
            if pred is not None:
                raise ValueError("ptx.tma.load_3d_multicast accepts either issuer= or pred=, not both")
            pred = cluster.rank(issuer) if isinstance(issuer, int) else issuer

        if coords is not None:
            coord = coords
        else:
            if row is None or col is None:
                raise TypeError("ptx.tma.load_3d_multicast requires row/col or coords=(...)")
            if isinstance(col, Reg):
                col_div = _scalar(_u32)
                _emit("shr", (".u32",), (col_div, col, ImmediateOperand("6")))
                col_coord = col_div
            elif isinstance(col, int):
                col_coord = ImmediateOperand(str(col // 64))
            else:
                col_coord = col
            coord = (ImmediateOperand("0"), row, col_coord)

        _emit(
            "cp",
            (".async", ".bulk", ".tensor", ".3d",
             ".shared::cluster", ".global",
             ".tile",
             ".mbarrier::complete_tx::bytes",
             ".multicast::cluster"),
            (_make_address(dst),
             AddressOperand(base=_addr_base_name(src), offset=_coord_text(coord)),
             _make_address(mbar),
             mask),
            pred=pred,
        )

    def store_3d(
        self,
        dst: Any,
        src: Any,
        row: Any = None,
        col: Any = None,
        coords: tuple[Any, ...] | None = None,
        *,
        pred: "Reg | NegPred | None" = None,
    ) -> None:
        """TMA 3D store: ``cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group``"""
        from pyptx.reg import scalar as _scalar, Reg
        from pyptx.types import u32 as _u32

        if coords is not None:
            coord = coords
        else:
            if row is None or col is None:
                raise TypeError("ptx.tma.store_3d requires row/col or coords=(...)")
            if isinstance(col, Reg):
                col_div = _scalar(_u32)
                _emit("shr", (".u32",), (col_div, col, ImmediateOperand("6")))
                col_coord = col_div
            elif isinstance(col, int):
                col_coord = ImmediateOperand(str(col // 64))
            else:
                col_coord = col
            coord = (ImmediateOperand("0"), row, col_coord)

        _emit(
            "cp",
            (".async", ".bulk", ".tensor", ".3d",
             ".global", ".shared::cta", ".tile", ".bulk_group"),
            (AddressOperand(base=_addr_base_name(dst), offset=_coord_text(coord)),
             _make_address(src)),
            pred=pred,
        )


tma = _Tma()


# -- Generic instruction (dot-chain fallback) -------------------------------

class _GenericInst:
    """Fallback for instructions without a dedicated wrapper.

    Usage: ptx.inst.mov.b32(dst, src)
           ptx.inst.add.f32(d, a, b)
    """
    def __init__(self, opcode: str = "", modifiers: tuple[str, ...] = ()) -> None:
        self._opcode = opcode
        self._modifiers = modifiers

    def __getattr__(self, name: str) -> _GenericInst:
        # Strip leading underscore if the next char is a digit (for .2d, .128x256b, etc.)
        if name.startswith("_") and len(name) > 1 and name[1].isdigit():
            ptx_name = name[1:]
        # Strip trailing underscore for Python keywords (.global_ → .global)
        elif name.endswith("_") and not name.endswith("__"):
            ptx_name = name.rstrip("_")
        else:
            ptx_name = name
        # Convert double underscore to :: (.shared__cta → .shared::cta)
        ptx_name = ptx_name.replace("__", "::")
        if not self._opcode:
            return _GenericInst(opcode=ptx_name)
        return _GenericInst(self._opcode, self._modifiers + (f".{ptx_name}",))

    def __call__(self, *operands: Any, pred: Reg | NegPred | None = None) -> "Reg | None":
        """Emit the instruction and return the destination register (if any).

        Returning the dest Reg enables expression chaining in the sugar pass::

            _t = ptx.inst.shr.s32(r[3], r[88], 6)  # _t is r[3]
            ptx.inst.add.s32(r[10], _t, r[5])       # uses _t
        """
        _emit(self._opcode, self._modifiers, operands, pred=pred)
        # Return the first operand if it's a Reg (the destination)
        if operands and isinstance(operands[0], Reg):
            return operands[0]
        return None

    @staticmethod
    def _raw_emit(text: str) -> None:
        """Emit a raw PTX line (e.g. '{' or '}' for nested scopes)."""
        from pyptx._trace import get_ctx
        from pyptx.ir.nodes import RawLine
        ctx = get_ctx()
        ctx.emit(RawLine(text=text))


inst = _GenericInst()


class _Pipe:
    """Sequential instruction chain — same PTX order, fewer Python lines.

    Usage::

        ptx.pipe(r[192]) \\
            .add.s32(r[193], -8192) \\
            .and_.b32(r[194], 262016) \\
            .shr.u32(r[195], 4)

    Each step emits one PTX instruction: ``op dst, <prev_result>, operand``.
    Instructions execute in the exact order written — no reordering.
    The chain returns the final destination register.
    """

    __slots__ = ("_src",)

    def __init__(self, src: Reg) -> None:
        self._src = src

    def __getattr__(self, name: str) -> "_PipeStep":
        if name.startswith("_"):
            raise AttributeError(name)
        if name.endswith("_") and not name.endswith("__"):
            ptx_name = name.rstrip("_")
        else:
            ptx_name = name
        ptx_name = ptx_name.replace("__", "::")
        return _PipeStep(self._src, ptx_name, ())


class _PipeStep:
    """Accumulates dot-modifiers, then emits on call."""

    __slots__ = ("_src", "_opcode", "_modifiers")

    def __init__(self, src: Reg, opcode: str, modifiers: tuple[str, ...]) -> None:
        self._src = src
        self._opcode = opcode
        self._modifiers = modifiers

    def __getattr__(self, name: str) -> "_PipeStep":
        if name.startswith("_"):
            raise AttributeError(name)
        if name.endswith("_") and not name.endswith("__"):
            ptx_name = name.rstrip("_")
        else:
            ptx_name = name
        ptx_name = ptx_name.replace("__", "::")
        if not self._modifiers and not self._opcode:
            return _PipeStep(self._src, ptx_name, ())
        return _PipeStep(self._src, self._opcode, self._modifiers + (f".{ptx_name}",))

    def __call__(self, dst: Reg, *operands: Any) -> "_Pipe":
        """Emit: opcode.mods dst, self._src, *operands; then chain."""
        _emit(self._opcode, self._modifiers, (dst, self._src) + operands)
        return _Pipe(dst)


def pipe(src: Reg) -> _Pipe:
    """Start an instruction pipeline chain.

    Each chained call emits one PTX instruction in order, feeding the
    previous result as the first source operand. No instruction
    reordering — the PTX is identical to writing the calls separately.

    Usage::

        ptx.pipe(r[192]).add.s32(r[193], -8192).and_.b32(r[194], 262016).shr.u32(r[195], 4)
    """
    return _Pipe(src)


# -- Convenience: common instructions directly on the module ----------------

def mov(dtype: PtxType, dst: Reg, src: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit mov.{dtype} dst, src;"""
    _emit("mov", (dtype.ptx,), (dst, src), pred=pred)

def add(dtype: PtxType, dst: Reg, a: Any, b: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit add.{dtype} dst, a, b;"""
    _emit("add", (dtype.ptx,), (dst, a, b), pred=pred)

def ret(*, pred: Reg | NegPred | None = None) -> None:
    """Emit ret;"""
    _emit("ret", (), (), pred=pred)

def bra(label: str, *, pred: Reg | NegPred | None = None) -> None:
    """Emit bra label;"""
    _emit("bra", (), (label,), pred=pred)


def label(name: str) -> None:
    """Emit a label: label_name:"""
    ctx = get_ctx()
    ctx.emit(Label(name=name))


def addr(base: Any, offset: Any = None) -> AddressOperand:
    """Create an address operand: [base], [base+offset].

    Accepts anything ``_addr_base_name`` knows about: ``Reg``,
    ``RegisterOperand``, ``AddressOperand``, ``MbarrierRef``,
    ``SharedAlloc``, ``SharedSlice``, any ``TensorSpec`` /
    ``TmaDescriptorHandle`` (duck-typed via ``.name``), or a plain
    string.

    Usage:
        ptx.addr(rd[0])         → [%rd0]
        ptx.addr(rd[0], 16)     → [%rd0+16]
        ptx.addr("param0")      → [param0]
        ptx.addr(A)             → [A]   # where A is a kernel TensorSpec
    """
    base_str = _addr_base_name(base)
    off_str: str | None = None
    if offset is not None:
        if isinstance(offset, Reg):
            off_str = offset.name
        else:
            off_str = str(offset)
    return AddressOperand(base=base_str, offset=off_str)


def param(dtype: PtxType, name: str, dst: Reg | None = None) -> Reg:
    """Load or materialize a kernel parameter and return the destination reg.

    For scalar/raw scalar params this emits ``ld.param``.
    For raw aggregate params like ``b8.align64.array128 tma_A`` this emits
    the existing ``mov``-from-symbol pattern.
    """
    from pyptx.reg import scalar as reg_scalar

    ctx = get_ctx()
    out = dst if dst is not None else reg_scalar(dtype)
    raw_param_types = getattr(ctx, "raw_param_types", {}) or {}
    raw_type = raw_param_types.get(name)
    if raw_type is not None and any(part.startswith("array") for part in raw_type.split(".")):
        getattr(inst.mov, dtype.name)(out, name)
    else:
        getattr(inst.ld.param, dtype.name)(out, addr(name))
    return out


# ===================================================================
# tcgen05 namespace (Blackwell sm_100a)
# ===================================================================

_TCGEN05_KINDS = ("tf32", "f16", "i8", "f8f6f4", "mxf8f6f4", "mxf4", "mxf4nvf4")
_TCGEN05_COLLECTOR_A = ("discard", "lastuse", "fill", "use")


class _Tcgen05:
    """ptx.tcgen05.alloc/dealloc/mma/fence/wait/ld/st/cp/shift — Blackwell tensor-core ops."""

    BLACKWELL_MASKED_DESC_B128 = 0x4000404000010000

    _UMMA_MAJOR = {"K": 0, "MN": 1}
    _UMMA_SCALE_IN = {1: 0, -1: 1}
    _UMMA_C_FORMAT = {"f16": 0, "f32": 1, "s32": 2}
    _UMMA_F16_F32_FORMAT = {"f16": 0, "bf16": 1, "tf32": 2}
    _UMMA_LAYOUT_TYPE = {
        "NONE": 0,
        "128B_BASE32B": 1,
        "128B": 2,
        "64B": 4,
        "32B": 6,
    }

    def alloc(
        self,
        tmem_addr: Any,
        ncols: int | Reg,
        *,
        cta_group: int = 1,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit tcgen05.alloc.cta_group::N.sync.aligned.shared::cta.b32 [tmem_addr], ncols;"""
        mods = (
            ".alloc",
            f".cta_group::{cta_group}",
            ".sync", ".aligned",
            ".shared::cta", ".b32",
        )
        _emit("tcgen05", mods, (_make_address(tmem_addr), ncols), pred=pred)

    def dealloc(
        self,
        taddr: Any,
        ncols: int | Reg,
        *,
        cta_group: int = 1,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit tcgen05.dealloc.cta_group::N.sync.aligned.b32 taddr, ncols;"""
        mods = (
            ".dealloc",
            f".cta_group::{cta_group}",
            ".sync", ".aligned", ".b32",
        )
        _emit("tcgen05", mods, (taddr, ncols), pred=pred)

    def relinquish_alloc_permit(
        self,
        *,
        cta_group: int = 1,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit tcgen05.relinquish_alloc_permit.cta_group::N.sync.aligned;"""
        mods = (
            ".relinquish_alloc_permit",
            f".cta_group::{cta_group}",
            ".sync", ".aligned",
        )
        _emit("tcgen05", mods, (), pred=pred)

    def mma(
        self,
        d_tmem: Any,
        a_desc: Any,
        b_desc: Any,
        idesc: Any,
        *,
        cta_group: int = 1,
        kind: str = "f16",
        enable_input_d: bool | int | None = True,
        scale_d: Any | None = None,
        sparse: bool = False,
        ashift: bool = False,
        collector_a: str | None = None,
        a_is_tmem: bool = False,
        sparse_metadata: Any | None = None,
        pred_operand: Any | None = None,
        scale_c: int | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit a Blackwell ``tcgen05.mma`` instruction.

        The dense F16/BF16/TF32 forms accepted by PTX source use either::

            tcgen05.mma.cta_group::1.kind::f16
                [d_tmem], a_desc, b_desc, idesc, {mask0,mask1,mask2,mask3}, p;

        or the runtime-accumulate variant::

            tcgen05.mma.cta_group::1.kind::f16
                [d_tmem], a_desc, b_desc, idesc, {mask0,mask1,mask2,mask3}, p, SCALE_C;

        where ``p`` is the runtime accumulate/select-input-D flag,
        ``SCALE_C`` is a compile-time immediate, and the mask tuple is
        typically all zeros for dense CUTLASS/CuTe forms.
        Sparse variants insert ``[metadata]`` before ``idesc``.
        ``scale_d`` here is the accumulate/select-input-D source
        (CUTLASS calls it ``scaleC`` / ``accumulate``), not a PTX modifier.
        """
        if kind not in _TCGEN05_KINDS:
            raise ValueError(f"tcgen05.mma kind must be one of {_TCGEN05_KINDS}, got {kind!r}")
        if collector_a is not None and collector_a not in _TCGEN05_COLLECTOR_A:
            raise ValueError(
                f"tcgen05.mma collector_a must be one of {_TCGEN05_COLLECTOR_A}, got {collector_a!r}"
            )
        mods: list[str] = [
            ".mma",
            *([".sp"] if sparse else []),
            f".cta_group::{cta_group}",
            f".kind::{kind}",
            *([".ashift"] if ashift else []),
            *([f".collector::a::{collector_a}"] if collector_a else []),
        ]
        a_operand = _make_address(a_desc) if a_is_tmem else a_desc
        operands: list[Any] = [_make_address(d_tmem), a_operand, b_desc]
        if sparse_metadata is not None:
            operands.append(_make_address(sparse_metadata))
        operands.append(idesc)

        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import pred as pred_t, u32 as u32_t

        if isinstance(pred_operand, Reg):
            pred_reg = pred_operand
            if pred_reg.dtype != pred_t:
                pred_word = pred_reg
                pred_reg = reg_scalar(pred_t)
                inst.setp.ne.b32(pred_reg, pred_word, 0)
        elif pred_operand is not None:
            pred_reg = reg_scalar(pred_t)
            pred_word = reg_scalar(u32_t)
            inst.mov.u32(pred_word, 1 if bool(pred_operand) else 0)
            inst.setp.ne.b32(pred_reg, pred_word, 0)
        elif isinstance(scale_d, Reg):
            if scale_d.dtype == pred_t:
                pred_reg = scale_d
            else:
                pred_reg = reg_scalar(pred_t)
                inst.setp.ne.b32(pred_reg, scale_d, 0)
        else:
            pred_reg = reg_scalar(pred_t)
            pred_word = reg_scalar(u32_t)
            inst.mov.u32(pred_word, 1 if bool(scale_d) else 0)
            inst.setp.ne.b32(pred_reg, pred_word, 0)

        ctx = get_ctx()
        ptx_version = getattr(ctx, "ptx_version", None)
        use_dense_mask_tuple = bool(
            not sparse
            and ptx_version is not None
            and ptx_version >= (8, 8)
        )
        if use_dense_mask_tuple:
            zero_word = reg_scalar(u32_t)
            inst.mov.u32(zero_word, 0)
            # Dense disable-output-lane mask: 4 words for cta_group=1 (M≤128
            # covers 128 lanes), 8 words for cta_group=2 (M=256 covers 256
            # lanes). All zeros = no output lanes disabled.
            n_mask_words = 8 if cta_group == 2 else 4
            operands.append([zero_word] * n_mask_words)
        operands.append(pred_reg)
        if scale_c is not None:
            operands.append(int(scale_c))

        _emit("tcgen05", tuple(mods), tuple(operands), pred=pred)

    def fence_before_thread_sync(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit tcgen05.fence::before_thread_sync;"""
        _emit("tcgen05", (".fence::before_thread_sync",), (), pred=pred)

    def fence_after_thread_sync(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit tcgen05.fence::after_thread_sync;"""
        _emit("tcgen05", (".fence::after_thread_sync",), (), pred=pred)

    def commit(
        self,
        mbar: Any,
        *,
        cta_group: int = 1,
        multicast: bool = False,
        multicast_mask: Any | None = None,
        space: str = "cluster",
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit tcgen05.commit.cta_group::N.mbarrier::arrive::one[.multicast::cluster].shared::cluster.b64 [mbar][, mask];

        For ``cta_group=2`` the commit must arrive on every participating
        CTA's local mbarrier; pass ``multicast=True`` with ``multicast_mask``
        set to a u16 bitmask of the peer-CTA ranks to signal.
        """
        if space not in ("cta", "cluster"):
            raise ValueError(f"tcgen05.commit space must be 'cta' or 'cluster', got {space!r}")
        if multicast_mask is not None:
            multicast = True
        mods: list[str] = [
            ".commit",
            f".cta_group::{cta_group}",
            ".mbarrier::arrive::one",
        ]
        if multicast:
            mods.append(".multicast::cluster")
        mods.extend([f".shared::{space}", ".b64"])
        operands: list[Any] = [_make_address(mbar)]
        if multicast and multicast_mask is not None:
            operands.append(multicast_mask)
        _emit("tcgen05", tuple(mods), tuple(operands), pred=pred)

    def wait_ld(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit tcgen05.wait::ld.sync.aligned;"""
        _emit("tcgen05", (".wait::ld", ".sync", ".aligned"), (), pred=pred)

    def wait_st(self, *, pred: Reg | NegPred | None = None) -> None:
        """Emit tcgen05.wait::st.sync.aligned;"""
        _emit("tcgen05", (".wait::st", ".sync", ".aligned"), (), pred=pred)

    def ld(
        self,
        dst_regs: Any,
        taddr: Any,
        *,
        shape: str = "16x128b",
        count: int = 1,
        dtype: str = "b32",
        pack: bool = False,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit tcgen05.ld.sync.aligned.{shape}.x{count}[.pack::16b] dst, [taddr];"""
        mods: list[str] = [
            ".ld", ".sync", ".aligned",
            f".{shape}",
            f".x{count}",
            f".{dtype}",
        ]
        if pack:
            mods.append(".pack::16b")
        if isinstance(dst_regs, RegArray):
            dst_op: Any = [dst_regs[i] for i in range(dst_regs.count)]
        else:
            dst_op = dst_regs
        _emit("tcgen05", tuple(mods), (dst_op, _make_address(taddr)), pred=pred)

    def st(
        self,
        taddr: Any,
        src_regs: Any,
        *,
        shape: str = "16x128b",
        count: int = 1,
        dtype: str = "b32",
        unpack: bool = False,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit tcgen05.st.sync.aligned.{shape}.x{count}[.unpack::16b] [taddr], src;"""
        mods: list[str] = [
            ".st", ".sync", ".aligned",
            f".{shape}",
            f".x{count}",
            f".{dtype}",
        ]
        if unpack:
            mods.append(".unpack::16b")
        if isinstance(src_regs, RegArray):
            src_op: Any = [src_regs[i] for i in range(src_regs.count)]
        else:
            src_op = src_regs
        _emit("tcgen05", tuple(mods), (_make_address(taddr), src_op), pred=pred)

    def cp(
        self,
        taddr: Any,
        src: Any,
        *,
        cta_group: int = 1,
        size: str = "128x256b",
        src_is_addr: bool | None = None,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit tcgen05.cp.cta_group::N.{size} [taddr], [smem];"""
        mods = (
            ".cp",
            f".cta_group::{cta_group}",
            f".{size}",
        )
        src_operand = _make_address(src) if src_is_addr is True else _maybe_address(src) if src_is_addr is None else src
        _emit("tcgen05", mods, (_make_address(taddr), src_operand), pred=pred)

    def shift(
        self,
        taddr: Any,
        *,
        cta_group: int = 1,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Emit tcgen05.shift.cta_group::N.down [taddr];"""
        mods = (
            ".shift",
            f".cta_group::{cta_group}",
            ".down",
        )
        _emit("tcgen05", mods, (_make_address(taddr),), pred=pred)

    def make_instr_desc_f16bf16_f32(
        self,
        *,
        m: int = 128,
        n: int = 256,
        ab_dtype: str = "bf16",
        a_major: str = "K",
        b_major: str = "K",
        scale_a: int = 1,
        scale_b: int = 1,
        saturate: bool = False,
        sparse: bool = False,
        max_shift: int = 0,
    ) -> int:
        """Build the 32-bit Blackwell UMMA instruction descriptor.

        Mirrors CUTLASS/CuTe's ``UMMA::make_instr_desc`` for the common
        dense F16/BF16 -> F32 path used by the first Blackwell GEMM kernels.
        The PTX instruction consumes the upper 32 bits of ``idescE``; this
        helper returns that 32-bit descriptor value directly.
        """
        if m not in (64, 128, 256):
            raise ValueError(f"tcgen05.make_instr_desc_f16bf16_f32 m must be 64, 128, or 256; got {m}")
        if n % 8 != 0 or not (8 <= n <= 256):
            raise ValueError(f"tcgen05.make_instr_desc_f16bf16_f32 n must be a multiple of 8 between 8 and 256; got {n}")
        if a_major not in self._UMMA_MAJOR:
            raise ValueError(f"a_major must be one of {tuple(self._UMMA_MAJOR)}, got {a_major!r}")
        if b_major not in self._UMMA_MAJOR:
            raise ValueError(f"b_major must be one of {tuple(self._UMMA_MAJOR)}, got {b_major!r}")
        if scale_a not in self._UMMA_SCALE_IN:
            raise ValueError("scale_a must be 1 or -1")
        if scale_b not in self._UMMA_SCALE_IN:
            raise ValueError("scale_b must be 1 or -1")
        if ab_dtype not in self._UMMA_F16_F32_FORMAT:
            raise ValueError(
                f"ab_dtype must be one of {tuple(self._UMMA_F16_F32_FORMAT)}, got {ab_dtype!r}"
            )
        if not (0 <= max_shift <= 3):
            raise ValueError("max_shift must be in [0, 3]")

        ab_format = self._UMMA_F16_F32_FORMAT[ab_dtype]
        desc = 0
        desc |= (1 if sparse else 0) << 2
        desc |= (1 if saturate else 0) << 3
        desc |= self._UMMA_C_FORMAT["f32"] << 4
        desc |= ab_format << 7
        desc |= ab_format << 10
        desc |= self._UMMA_SCALE_IN[scale_a] << 13
        desc |= self._UMMA_SCALE_IN[scale_b] << 14
        desc |= self._UMMA_MAJOR[a_major] << 15
        desc |= self._UMMA_MAJOR[b_major] << 16
        desc |= (n >> 3) << 17
        desc |= (m >> 4) << 24
        desc |= max_shift << 30
        return desc

    def descriptor(
        self,
        smem_addr: Any,
        *,
        byte_offset: int = 0,
        stride_bytes: int,
        leading_bytes: int = 16,
        swizzle: str = "128B",
        version: int = 1,
        base_offset: int = 0,
        lbo_mode: int = 0,
    ) -> Reg:
        """Build a Blackwell UMMA shared-memory descriptor.

        This mirrors CUTLASS/CuTe's ``UMMA::SmemDescriptor`` encoding.
        ``stride_bytes`` and ``leading_bytes`` are byte offsets and must be
        multiples of 16 because PTX stores them without the low 4 bits.
        """
        if swizzle not in self._UMMA_LAYOUT_TYPE:
            raise ValueError(f"swizzle must be one of {tuple(self._UMMA_LAYOUT_TYPE)}, got {swizzle!r}")
        if stride_bytes % 16 != 0:
            raise ValueError("stride_bytes must be a multiple of 16")
        if leading_bytes % 16 != 0:
            raise ValueError("leading_bytes must be a multiple of 16")
        if not (0 <= version <= 0x3):
            raise ValueError("version must fit in 2 bits")
        if not (0 <= base_offset <= 0x7):
            raise ValueError("base_offset must fit in 3 bits")
        if not (0 <= lbo_mode <= 0x1):
            raise ValueError("lbo_mode must fit in 1 bit")

        const_bits = 0
        const_bits |= (leading_bytes >> 4) << 16
        const_bits |= (stride_bytes >> 4) << 32
        const_bits |= version << 46
        const_bits |= base_offset << 49
        const_bits |= lbo_mode << 52
        const_bits |= self._UMMA_LAYOUT_TYPE[swizzle] << 61

        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import b64

        raw = wgmma.masked_descriptor(
            smem_addr,
            byte_offset=byte_offset,
            mask=0x3FFF0,
            const_bits=const_bits,
        )
        desc = reg_scalar(b64)
        inst.mov.b64(desc, raw)
        return desc

    def masked_descriptor(
        self,
        smem_addr: Any,
        *,
        byte_offset: int = 0,
        mask: int = 0x3FFF0,
        const_bits: int = BLACKWELL_MASKED_DESC_B128,
    ) -> Reg:
        """Build a Blackwell shared-memory descriptor from a shared address.

        This mirrors the CUTLASS SM100 GEMM pattern:

          tmp  = smem_addr + byte_offset
          idx  = (tmp >> 4) & 0x3fff
          desc = cvt.u64.u32(idx) | 0x4000404000010000

        Prefer ``ptx.tcgen05.descriptor(...)`` for new code; this helper keeps
        the original fixed-B128 GEMM constant for backward compatibility.
        """
        if mask != 0x3FFF0 or const_bits != self.BLACKWELL_MASKED_DESC_B128:
            from pyptx.reg import scalar as reg_scalar
            from pyptx.types import b64

            raw = wgmma.masked_descriptor(
                smem_addr,
                byte_offset=byte_offset,
                mask=mask,
                const_bits=const_bits,
            )
            desc = reg_scalar(b64)
            inst.mov.b64(desc, raw)
            return desc

        return self.descriptor(
            smem_addr,
            byte_offset=byte_offset,
            stride_bytes=1024,
            leading_bytes=16,
            swizzle="128B",
        )


tcgen05 = _Tcgen05()


# ===================================================================
# setmaxnreg
# ===================================================================

def setmaxnreg_inc(reg_count: int, *, pred: Reg | NegPred | None = None) -> None:
    """Emit setmaxnreg.inc.sync.aligned.u32 N;"""
    _emit("setmaxnreg", (".inc", ".sync", ".aligned", ".u32"),
          (ImmediateOperand(str(reg_count)),), pred=pred)


def setmaxnreg_dec(reg_count: int, *, pred: Reg | NegPred | None = None) -> None:
    """Emit setmaxnreg.dec.sync.aligned.u32 N;"""
    _emit("setmaxnreg", (".dec", ".sync", ".aligned", ".u32"),
          (ImmediateOperand(str(reg_count)),), pred=pred)


# ===================================================================
# elect.sync
# ===================================================================

def elect_sync(dst: Reg, pred_out: Reg, membermask: int | Reg) -> None:
    """Emit elect.sync dst|pred, membermask;

    dst gets the leader lane index, pred_out gets the elected bit.
    """
    pipe = PipeRef(dst, pred_out)
    _emit("elect", (".sync",), (pipe, membermask))


# ===================================================================
# cluster namespace
# ===================================================================

class _Cluster:
    """ptx.cluster.arrive(), ptx.cluster.wait(), ptx.cluster.sync() — barrier.cluster.* helpers."""

    def arrive(self, *, aligned: bool = False, pred: Reg | NegPred | None = None) -> None:
        """Emit barrier.cluster.arrive[.aligned];"""
        mods: tuple[str, ...] = (".cluster", ".arrive")
        if aligned:
            mods = mods + (".aligned",)
        _emit("barrier", mods, (), pred=pred)

    def wait(self, *, aligned: bool = False, pred: Reg | NegPred | None = None) -> None:
        """Emit barrier.cluster.wait[.aligned];"""
        mods: tuple[str, ...] = (".cluster", ".wait")
        if aligned:
            mods = mods + (".aligned",)
        _emit("barrier", mods, (), pred=pred)

    def sync(self, *, aligned: bool = False, pred: Reg | NegPred | None = None) -> None:
        """Emit barrier.cluster.arrive + barrier.cluster.wait."""
        self.arrive(aligned=aligned, pred=pred)
        self.wait(aligned=aligned, pred=pred)

    def rank(self, cta_rank: int | Reg) -> Reg:
        """Return a predicate for ``%cluster_ctarank == cta_rank``."""
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import u32

        rank_reg = reg_scalar(u32)
        inst.mov.u32(rank_reg, sreg("%cluster_ctarank"))
        return rank_reg == cta_rank

    def map_shared_u32(
        self,
        bar_addr: Reg,
        cta_id: Reg | int,
        *,
        pred: Reg | NegPred | None = None,
    ) -> Reg:
        """Return ``mapa.shared::cluster.u32`` of ``bar_addr`` for ``cta_id``."""
        from pyptx.reg import scalar as _scalar
        from pyptx.types import b32 as _b32, u32 as _u32

        rem = _scalar(_b32)
        cta_val = cta_id
        if isinstance(cta_id, int):
            cta_reg = _scalar(_u32)
            inst.mov.u32(cta_reg, cta_id)
            cta_val = cta_reg
        _emit("mapa", (".shared::cluster", ".u32"), (rem, bar_addr, cta_val), pred=pred)
        return rem

    def arrive_multicast(
        self,
        bar_addr: Reg,
        mask: Any,
        count: "Reg | int" = 1,
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Arrive on the mbarrier at the same SMEM offset on every CTA in a
        cluster whose rank bit is set in ``mask`` (u16).

        Emits ``mbarrier.arrive.shared::cluster.multicast::cluster.b64 _,
        [bar_addr], count, mask`` in a single instruction — one arrive per
        target CTA. Replaces the common ``arrive_remote(peer) + arrive()``
        pair used for cross-CTA hand-off mbars. The variant isn't in the
        generated spec table yet, so we drop to ``raw`` emission.
        """
        from pyptx.ir.nodes import RawLine
        from pyptx.reg import scalar as _scalar_reg
        from pyptx.types import u32 as _u32

        # Normalize operands to their textual PTX form.
        addr_txt: str
        if hasattr(bar_addr, "name"):
            addr_txt = f"[{bar_addr.name}]"
        else:
            # Allow a raw string or AddressOperand-compatible value.
            addr_txt = f"[{bar_addr}]"
        mask_txt = mask.name if hasattr(mask, "name") else str(mask)
        count_txt = str(count) if isinstance(count, int) else count.name
        pred_prefix = ""
        if pred is not None:
            pred_prefix = f"@{pred.name} "
        get_ctx().emit(RawLine(
            text=(
                f"\t{pred_prefix}mbarrier.arrive.shared::cluster"
                f".multicast::cluster.b64 _, {addr_txt}, {count_txt}, {mask_txt};"
            )
        ))

    def arrive_remote(
        self,
        bar_addr: Reg,
        cta_id: Reg,
        count: "Reg | int" = 1,
        *,
        pred: Reg | NegPred | None = None,
    ) -> None:
        """Arrive on a barrier in a remote CTA within the cluster.

        Wraps the 3-instruction pattern::

            { mapa.shared::cluster.u32 remAddr, bar_addr, cta_id;
              mbarrier.arrive.shared::cluster.b64 _, [remAddr], count; }
        """
        from pyptx.ir.nodes import RawLine
        ctx = get_ctx()
        ctx.emit(RawLine(text="{"))
        ctx._scope_depth += 1
        from pyptx.reg import scalar as _scalar
        from pyptx.types import b32 as _b32
        rem = _scalar(_b32, name="remAddr32")
        _emit("mapa", (".shared::cluster", ".u32"), (rem, bar_addr, cta_id), pred=pred)
        _emit("mbarrier", (".arrive", ".shared::cluster", ".b64"),
              (LabelOperand("_"), AddressOperand(base="remAddr32"), count), pred=pred)
        ctx._scope_depth -= 1
        ctx.emit(RawLine(text="}"))


cluster = _Cluster()


class _Cvta:
    """Small helpers for common ``cvta`` conversions."""

    def param(self, src: Reg, dst: Reg | None = None) -> Reg:
        """Emit ``cvta.param.u64`` and return the destination register."""
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import b64

        out = dst if dst is not None else reg_scalar(b64)
        inst.cvta.param.u64(out, src)
        return out

    def to_global(self, src: Reg, dst: Reg | None = None) -> Reg:
        """Emit ``cvta.to.global.u64`` and return the destination register."""
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import b64

        out = dst if dst is not None else reg_scalar(b64)
        inst.cvta.to.global_.u64(out, src)
        return out


cvta = _Cvta()


# ===================================================================
# Common arithmetic / memory wrappers
# ===================================================================

def sub(dtype: PtxType, dst: Reg, a: Any, b: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit sub.{dtype} dst, a, b;"""
    _emit("sub", (dtype.ptx,), (dst, a, b), pred=pred)


def mul(
    dtype: PtxType,
    dst: Reg,
    a: Any,
    b: Any,
    *,
    mode: str | None = None,
    pred: Reg | NegPred | None = None,
) -> None:
    """Emit mul[.lo|.hi|.wide].{dtype} dst, a, b;"""
    if mode is not None and mode not in ("lo", "hi", "wide"):
        raise ValueError(f"mul mode must be one of lo/hi/wide, got {mode!r}")
    mods: tuple[str, ...] = ((f".{mode}",) if mode else ()) + (dtype.ptx,)
    _emit("mul", mods, (dst, a, b), pred=pred)


def mad(
    *args,
    mode: str = "lo",
    pred: Reg | NegPred | None = None,
):
    """Emit ``mad`` in either explicit-dst or expression style.

    Explicit-dst form:
        ``ptx.mad(s32, dst, a, b, c)``

    Expression form:
        ``dst = ptx.mad(a, b, c)``
    """
    from pyptx.reg import scalar as reg_scalar

    if mode not in ("lo", "hi", "wide"):
        raise ValueError(f"mad mode must be one of lo/hi/wide, got {mode!r}")

    if args and isinstance(args[0], PtxType):
        dtype, dst, a, b, c = args
        _emit("mad", (f".{mode}", dtype.ptx), (dst, a, b, c), pred=pred)
        return None

    if len(args) != 3:
        raise TypeError("ptx.mad expects either (dtype, dst, a, b, c) or (a, b, c)")

    a, b, c = args
    dtype = None
    for operand in (a, b, c):
        if isinstance(operand, Reg):
            dtype = operand.dtype
            break
    if dtype is None:
        raise TypeError("ptx.mad(a, b, c) needs at least one Reg operand to infer dtype")

    out = reg_scalar(dtype)
    mod_dtype = ".s32" if dtype.ptx in (".u32", ".s32", ".b32") else dtype.ptx
    _emit("mad", (f".{mode}", mod_dtype), (out, a, b, c), pred=pred)
    return out


def shl(dtype: PtxType, dst: Reg, a: Any, b: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit shl.{dtype} dst, a, b;"""
    _emit("shl", (dtype.ptx,), (dst, a, b), pred=pred)


def shr(dtype: PtxType, dst: Reg, a: Any, b: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit shr.{dtype} dst, a, b;"""
    _emit("shr", (dtype.ptx,), (dst, a, b), pred=pred)


def setp(
    cmp_op: str,
    dtype: PtxType,
    pred_out: Reg,
    a: Any,
    b: Any,
    *,
    pred_negate: Reg | NegPred | None = None,
) -> None:
    """Emit setp.{cmp_op}.{dtype} pred_out, a, b;"""
    if cmp_op not in ("lt", "le", "gt", "ge", "eq", "ne"):
        raise ValueError(
            f"setp cmp_op must be one of lt/le/gt/ge/eq/ne, got {cmp_op!r}"
        )
    _emit("setp", (f".{cmp_op}", dtype.ptx), (pred_out, a, b), pred=pred_negate)


def cvt(
    dst_type: PtxType,
    src_type: PtxType,
    dst: Reg,
    src: Any,
    *,
    rounding: str | None = None,
    ftz: bool = False,
    sat: bool = False,
    pred: Reg | NegPred | None = None,
) -> None:
    """Emit cvt[.rnd][.ftz][.sat].{dst_type}.{src_type} dst, src;"""
    mods: list[str] = []
    if rounding is not None:
        mods.append(f".{rounding}")
    if ftz:
        mods.append(".ftz")
    if sat:
        mods.append(".sat")
    mods.append(dst_type.ptx)
    mods.append(src_type.ptx)
    _emit("cvt", tuple(mods), (dst, src), pred=pred)


def ld(
    dtype: PtxType,
    dst: Reg,
    addr: Any,
    *,
    space: str = "global",
    cache: str | None = None,
    pred: Reg | NegPred | None = None,
) -> None:
    """Emit ld.{space}[.{cache}].{dtype} dst, [addr];"""
    if space not in ("global", "shared", "local", "param", "const", "generic"):
        raise ValueError(f"ld space must be one of global/shared/local/param/const, got {space!r}")
    mods: list[str] = []
    if space != "generic":
        mods.append(f".{space.replace('__', '::')}")
    if cache is not None:
        mods.append(f".{cache}")
    mods.append(dtype.ptx)
    _emit("ld", tuple(mods), (dst, _make_address(addr)), pred=pred)


def st(
    dtype: PtxType,
    addr: Any,
    src: Any,
    *,
    space: str = "global",
    cache: str | None = None,
    pred: Reg | NegPred | None = None,
) -> None:
    """Emit st.{space}[.{cache}].{dtype} [addr], src;"""
    if space not in ("global", "shared", "local", "param", "const", "generic"):
        raise ValueError(f"st space must be one of global/shared/local/param/const, got {space!r}")
    mods: list[str] = []
    if space != "generic":
        mods.append(f".{space.replace('__', '::')}")
    if cache is not None:
        mods.append(f".{cache}")
    mods.append(dtype.ptx)
    _emit("st", tuple(mods), (_make_address(addr), src), pred=pred)


def and_(dtype: PtxType, dst: Reg, a: Any, b: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit and.{dtype} dst, a, b;"""
    _emit("and", (dtype.ptx,), (dst, a, b), pred=pred)


def or_(dtype: PtxType, dst: Reg, a: Any, b: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit or.{dtype} dst, a, b;"""
    _emit("or", (dtype.ptx,), (dst, a, b), pred=pred)


def xor_(dtype: PtxType, dst: Reg, a: Any, b: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit xor.{dtype} dst, a, b;"""
    _emit("xor", (dtype.ptx,), (dst, a, b), pred=pred)


def not_(dtype: PtxType, dst: Reg, src: Any, *, pred: Reg | NegPred | None = None) -> None:
    """Emit not.{dtype} dst, src;"""
    _emit("not", (dtype.ptx,), (dst, src), pred=pred)


# ===================================================================
# Codegen companions: IR Instruction -> typed wrapper Python call.
#
# Each function inspects inst.modifiers and returns a Python source
# string for the equivalent typed wrapper call, OR None if this
# specific variant is not handled by the wrapper.
# ===================================================================

def _strip_dot(m: str) -> str:
    return m[1:] if m.startswith(".") else m


def _pred_suffix(inst: "Instruction", cg: Any) -> str:
    if inst.predicate is None:
        return ""
    p_ref = cg.reg_ref(inst.predicate.register)
    if inst.predicate.negated:
        return f", pred=~{p_ref}"
    return f", pred={p_ref}"


def _ptx_type_pyname(mod: str) -> str | None:
    """'.f32' -> 'f32' if it's a known PtxType, else None."""
    name = _strip_dot(mod)
    from pyptx.codegen.codegen import _TYPE_IMPORTS  # local import to avoid cycle
    return _TYPE_IMPORTS.get(name)


# ----- wgmma -----

def _codegen_wgmma_mma_async(inst: Instruction, cg: Any) -> str | None:
    mods = inst.modifiers
    if not (len(mods) >= 7 and mods[0] == ".mma_async"
            and mods[1] == ".sync" and mods[2] == ".aligned"):
        return None
    shape_mod = mods[3]  # .mNxNyNz
    if not shape_mod.startswith(".m"):
        return None
    try:
        rest = shape_mod[2:]
        m_str, rest2 = rest.split("n", 1)
        n_str, k_str = rest2.split("k", 1)
        m, n, k = int(m_str), int(n_str), int(k_str)
    except (ValueError, IndexError):
        return None
    dtype_d = _ptx_type_pyname(mods[4])
    dtype_a = _ptx_type_pyname(mods[5])
    dtype_b = _ptx_type_pyname(mods[6])
    if not (dtype_d and dtype_a and dtype_b):
        return None
    if len(inst.operands) < 5:
        return None
    d_op = cg.operand(inst.operands[0])
    a_op = cg.operand(inst.operands[1])
    b_op = cg.operand(inst.operands[2])
    sd = cg.operand(inst.operands[3])
    sa = cg.operand(inst.operands[4])
    sb = cg.operand(inst.operands[5])
    return (
        f"ptx.wgmma.mma_async(shape=({m}, {n}, {k}), "
        f"dtype_d={dtype_d}, dtype_a={dtype_a}, dtype_b={dtype_b}, "
        f"d={d_op}, a={a_op}, b={b_op}, "
        f"scale_d={sd}, scale_a={sa}, scale_b={sb}{_pred_suffix(inst, cg)})"
    )


def _codegen_wgmma_fence(inst: Instruction, cg: Any) -> str | None:
    if inst.modifiers == (".fence", ".sync", ".aligned") and not inst.operands:
        return f"ptx.wgmma.fence({_pred_suffix(inst, cg).lstrip(', ')})"
    return None


def _codegen_wgmma_commit_group(inst: Instruction, cg: Any) -> str | None:
    if inst.modifiers == (".commit_group", ".sync", ".aligned") and not inst.operands:
        return f"ptx.wgmma.commit_group({_pred_suffix(inst, cg).lstrip(', ')})"
    return None


def _codegen_wgmma_wait_group(inst: Instruction, cg: Any) -> str | None:
    if inst.modifiers == (".wait_group", ".sync", ".aligned") and len(inst.operands) == 1:
        n_op = cg.operand(inst.operands[0])
        return f"ptx.wgmma.wait_group({n_op}{_pred_suffix(inst, cg)})"
    return None


# ----- cp.async.bulk.tensor -----

def _codegen_cp_async_bulk_tensor(inst: Instruction, cg: Any) -> str | None:
    """Match cp.async.bulk.tensor.Nd loads (global -> shared)."""
    mods = inst.modifiers
    if inst.opcode != "cp":
        return None
    if not (len(mods) >= 6 and mods[0] == ".async" and mods[1] == ".bulk"
            and mods[2] == ".tensor"):
        return None
    nd_mod = mods[3]
    if not (nd_mod.endswith("d") and nd_mod[1:-1].isdigit()):
        return None
    ndim = int(nd_mod[1:-1])
    # Load = .shared::cluster + .global; store = .global + .shared::cta
    if ".shared::cluster" not in mods or ".global" not in mods:
        return None
    is_im2col = ".im2col" in mods
    is_gather4 = ".gather4" in mods
    if not inst.operands:
        return None
    dst_addr = inst.operands[0]
    if not isinstance(dst_addr, AddressOperand) or dst_addr.offset is None:
        return None
    # Coordinates are encoded in offset like ", {%r0, %r1}"
    off = dst_addr.offset.strip()
    if not (off.startswith(",") or off.startswith("{")):
        return None
    inside = off.lstrip(",").strip()
    if not (inside.startswith("{") and inside.endswith("}")):
        return None
    coord_strs = [c.strip() for c in inside[1:-1].split(",") if c.strip()]
    coord_pys = [cg.reg_ref(c) if c.startswith("%") else c for c in coord_strs]
    coord_tuple = "(" + ", ".join(coord_pys) + ("," if len(coord_pys) == 1 else "") + ")"
    dst_py = cg.reg_ref(dst_addr.base) if dst_addr.base.startswith("%") else f'"{dst_addr.base}"'
    # src is operand 1
    src_py = cg.operand(inst.operands[1])
    extras: list[str] = []
    op_idx = 2
    if op_idx < len(inst.operands) and ".mbarrier::complete_tx::bytes" in mods:
        extras.append(f"mbar={cg.operand(inst.operands[op_idx])}")
        op_idx += 1
    if ".multicast::cluster" in mods and op_idx < len(inst.operands):
        extras.append(f"multicast_mask={cg.operand(inst.operands[op_idx])}")
        op_idx += 1
    if ".L2::cache_hint" in mods and op_idx < len(inst.operands):
        extras.append(f"cache_hint={cg.operand(inst.operands[op_idx])}")
        op_idx += 1
    extras_str = (", " + ", ".join(extras)) if extras else ""
    if is_gather4:
        method = "gather4_2d"
    elif is_im2col:
        method = f"im2col_{ndim}d"
    else:
        method = f"tile_{ndim}d"
    return (
        f"ptx.cp.async_.bulk.tensor.{method}({dst_py}, {src_py}, {coord_tuple}"
        f"{extras_str}{_pred_suffix(inst, cg)})"
    )


def _codegen_cp_async_bulk_tensor_store(inst: Instruction, cg: Any) -> str | None:
    """Match cp.async.bulk.tensor.Nd stores (shared::cta -> global)."""
    mods = inst.modifiers
    if inst.opcode != "cp":
        return None
    if not (len(mods) >= 6 and mods[0] == ".async" and mods[1] == ".bulk"
            and mods[2] == ".tensor"):
        return None
    nd_mod = mods[3]
    if not (nd_mod.endswith("d") and nd_mod[1:-1].isdigit()):
        return None
    ndim = int(nd_mod[1:-1])
    if ".global" not in mods or ".shared::cta" not in mods:
        return None
    if ".shared::cluster" in mods:  # that's a load
        return None
    is_scatter = ".scatter4" in mods
    if not inst.operands or len(inst.operands) < 2:
        return None
    if is_scatter:
        return None
    dst_addr = inst.operands[0]
    src_op = inst.operands[1]
    if not isinstance(dst_addr, AddressOperand) or dst_addr.offset is None:
        return None
    off = dst_addr.offset.strip()
    inside = off.lstrip(",").strip()
    if not (inside.startswith("{") and inside.endswith("}")):
        return None
    coord_strs = [c.strip() for c in inside[1:-1].split(",") if c.strip()]
    coord_pys = [cg.reg_ref(c) if c.startswith("%") else c for c in coord_strs]
    coord_tuple = "(" + ", ".join(coord_pys) + ("," if len(coord_pys) == 1 else "") + ")"
    dst_py = cg.reg_ref(dst_addr.base) if dst_addr.base.startswith("%") else f'"{dst_addr.base}"'
    src_py = cg.operand(src_op)
    extras: list[str] = []
    if ".L2::cache_hint" in mods and len(inst.operands) >= 3:
        extras.append(f"cache_hint={cg.operand(inst.operands[2])}")
    extras_str = (", " + ", ".join(extras)) if extras else ""
    return (
        f"ptx.cp.async_.bulk.tensor.store_{ndim}d({dst_py}, {src_py}, {coord_tuple}"
        f"{extras_str}{_pred_suffix(inst, cg)})"
    )


# ----- tcgen05 -----

def _cta_group_from_mods(mods: tuple[str, ...]) -> int | None:
    for m in mods:
        if m.startswith(".cta_group::"):
            try:
                return int(m.split("::", 1)[1])
            except ValueError:
                return None
    return None


def _codegen_tcgen05_alloc(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".alloc" not in inst.modifiers:
        return None
    if ".dealloc" in inst.modifiers or ".relinquish_alloc_permit" in inst.modifiers:
        return None
    cg_n = _cta_group_from_mods(inst.modifiers)
    if cg_n is None or len(inst.operands) < 2:
        return None
    return (
        f"ptx.tcgen05.alloc({cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, cta_group={cg_n}{_pred_suffix(inst, cg)})"
    )


def _codegen_tcgen05_dealloc(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".dealloc" not in inst.modifiers:
        return None
    cg_n = _cta_group_from_mods(inst.modifiers)
    if cg_n is None or len(inst.operands) < 2:
        return None
    return (
        f"ptx.tcgen05.dealloc({cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, cta_group={cg_n}{_pred_suffix(inst, cg)})"
    )


def _codegen_tcgen05_relinquish(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".relinquish_alloc_permit" not in inst.modifiers:
        return None
    cg_n = _cta_group_from_mods(inst.modifiers)
    if cg_n is None:
        return None
    return f"ptx.tcgen05.relinquish_alloc_permit(cta_group={cg_n}{_pred_suffix(inst, cg)})"


def _codegen_tcgen05_fence(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05":
        return None
    if inst.modifiers == (".fence::before_thread_sync",):
        return f"ptx.tcgen05.fence_before_thread_sync({_pred_suffix(inst, cg).lstrip(', ')})"
    if inst.modifiers == (".fence::after_thread_sync",):
        return f"ptx.tcgen05.fence_after_thread_sync({_pred_suffix(inst, cg).lstrip(', ')})"
    return None


def _codegen_tcgen05_wait(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05":
        return None
    if inst.modifiers == (".wait::ld", ".sync", ".aligned"):
        return f"ptx.tcgen05.wait_ld({_pred_suffix(inst, cg).lstrip(', ')})"
    if inst.modifiers == (".wait::st", ".sync", ".aligned"):
        return f"ptx.tcgen05.wait_st({_pred_suffix(inst, cg).lstrip(', ')})"
    return None


def _codegen_tcgen05_commit(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".commit" not in inst.modifiers:
        return None
    cg_n = _cta_group_from_mods(inst.modifiers)
    if cg_n is None or not inst.operands:
        return None
    multicast = ".multicast::cluster" in inst.modifiers
    extra = ", multicast=True" if multicast else ""
    if ".shared::cta" in inst.modifiers:
        extra += ", space=\"cta\""
    return (
        f"ptx.tcgen05.commit({cg.operand(inst.operands[0])}, "
        f"cta_group={cg_n}{extra}{_pred_suffix(inst, cg)})"
    )


def _codegen_tcgen05_shift(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".shift" not in inst.modifiers:
        return None
    cg_n = _cta_group_from_mods(inst.modifiers)
    if cg_n is None or not inst.operands:
        return None
    return (
        f"ptx.tcgen05.shift({cg.operand(inst.operands[0])}, "
        f"cta_group={cg_n}{_pred_suffix(inst, cg)})"
    )


def _codegen_tcgen05_cp(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".cp" not in inst.modifiers:
        return None
    cg_n = _cta_group_from_mods(inst.modifiers)
    if cg_n is None or len(inst.operands) < 2:
        return None
    size = None
    for m in inst.modifiers:
        if m == ".cp" or m.startswith(".cta_group::"):
            continue
        size = _strip_dot(m)
        break
    if size is None:
        return None
    return (
        f"ptx.tcgen05.cp({cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, cta_group={cg_n}, size=\"{size}\""
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_tcgen05_ld(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".ld" not in inst.modifiers:
        return None
    if len(inst.operands) < 2:
        return None
    shape = None
    count = None
    dtype = None
    pack = ".pack::16b" in inst.modifiers
    for m in inst.modifiers:
        s = _strip_dot(m)
        if "x" in s and not s.startswith("x") and any(ch.isdigit() for ch in s):
            shape = s
        elif s.startswith("x") and s[1:].isdigit():
            count = int(s[1:])
        elif s == "b32":
            dtype = s
    if shape is None or count is None:
        return None
    extra = ", pack=True" if pack else ""
    if dtype and dtype != "b32":
        extra += f', dtype="{dtype}"'
    return (
        f"ptx.tcgen05.ld({cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, shape=\"{shape}\", count={count}{extra}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_tcgen05_st(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".st" not in inst.modifiers:
        return None
    if len(inst.operands) < 2:
        return None
    shape = None
    count = None
    dtype = None
    unpack = ".unpack::16b" in inst.modifiers
    for m in inst.modifiers:
        s = _strip_dot(m)
        if "x" in s and not s.startswith("x") and any(ch.isdigit() for ch in s):
            shape = s
        elif s.startswith("x") and s[1:].isdigit():
            count = int(s[1:])
        elif s == "b32":
            dtype = s
    if shape is None or count is None:
        return None
    extra = ", unpack=True" if unpack else ""
    if dtype and dtype != "b32":
        extra += f', dtype="{dtype}"'
    return (
        f"ptx.tcgen05.st({cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, shape=\"{shape}\", count={count}{extra}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_tcgen05_mma(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "tcgen05" or ".mma" not in inst.modifiers:
        return None
    cg_n = _cta_group_from_mods(inst.modifiers)
    if cg_n is None:
        return None
    kind = None
    sparse = ".sp" in inst.modifiers
    ashift = ".ashift" in inst.modifiers
    collector_a = None
    for m in inst.modifiers:
        if m.startswith(".kind::"):
            kind = m.split("::", 1)[1]
            break
    for m in inst.modifiers:
        if m.startswith(".collector::a::"):
            collector_a = m.split("::")[-1]
            break
    if kind is None:
        return None
    if len(inst.operands) < 6:
        return None
    d = cg.operand(inst.operands[0])
    a = cg.operand(inst.operands[1])
    b = cg.operand(inst.operands[2])
    extras = []
    if sparse:
        extras.append("sparse=True")
    if ashift:
        extras.append("ashift=True")
    if collector_a:
        extras.append(f'collector_a="{collector_a}"')

    idx = 3
    if sparse:
        if len(inst.operands) < 6:
            return None
        extras.append(f"sparse_metadata={cg.operand(inst.operands[idx])}")
        idx += 1

    idesc = cg.operand(inst.operands[idx])
    idx += 1

    if len(inst.operands) <= idx:
        return None
    from pyptx.ir.nodes import VectorOperand
    if isinstance(inst.operands[idx], VectorOperand):
        idx += 1
        if len(inst.operands) <= idx:
            return None
    extras.append(f"pred_operand={cg.operand(inst.operands[idx])}")
    idx += 1
    if len(inst.operands) > idx:
        extras.append(f"enable_input_d={cg.operand(inst.operands[idx])}")

    extra = (", " + ", ".join(extras)) if extras else ""
    return (
        f"ptx.tcgen05.mma({d}, {a}, {b}, {idesc}, cta_group={cg_n}, "
        f"kind=\"{kind}\"{extra}{_pred_suffix(inst, cg)})"
    )


# ----- setmaxnreg -----

def _codegen_setmaxnreg(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "setmaxnreg" or len(inst.operands) != 1:
        return None
    if ".inc" in inst.modifiers:
        return f"ptx.setmaxnreg_inc({cg.operand(inst.operands[0])}{_pred_suffix(inst, cg)})"
    if ".dec" in inst.modifiers:
        return f"ptx.setmaxnreg_dec({cg.operand(inst.operands[0])}{_pred_suffix(inst, cg)})"
    return None


# ----- elect.sync -----

def _codegen_elect_sync(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "elect" or inst.modifiers != (".sync",):
        return None
    if len(inst.operands) != 2:
        return None
    pipe = inst.operands[0]
    from pyptx.ir.nodes import PipeOperand as _PO
    if not isinstance(pipe, _PO):
        return None
    if not (isinstance(pipe.left, RegisterOperand) and isinstance(pipe.right, RegisterOperand)):
        return None
    dst = cg.reg_ref(pipe.left.name)
    pred_out = cg.reg_ref(pipe.right.name)
    membermask = cg.operand(inst.operands[1])
    return f"ptx.elect_sync({dst}, {pred_out}, {membermask})"


# ----- barrier.cluster (cluster wrapper) -----

def _codegen_barrier_cluster(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "barrier" or ".cluster" not in inst.modifiers:
        return None
    aligned = ".aligned" in inst.modifiers
    pred_str = _pred_suffix(inst, cg)
    parts: list[str] = []
    if aligned:
        parts.append("aligned=True")
    if pred_str:
        parts.append(pred_str.lstrip(", "))
    args = ", ".join(parts)
    if ".arrive" in inst.modifiers:
        return f"ptx.cluster.arrive({args})"
    if ".wait" in inst.modifiers:
        return f"ptx.cluster.wait({args})"
    return None


# ----- mbarrier -----

def _codegen_mbarrier_init(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "mbarrier" or ".init" not in inst.modifiers:
        return None
    if len(inst.operands) != 2:
        return None
    return (
        f"ptx.mbarrier.init({cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}{_pred_suffix(inst, cg)})"
    )


def _codegen_mbarrier_arrive(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "mbarrier" or ".arrive" not in inst.modifiers:
        return None
    if ".expect_tx" in inst.modifiers:
        return None
    if len(inst.operands) != 1:
        return None
    return f"ptx.mbarrier.arrive({cg.operand(inst.operands[0])}{_pred_suffix(inst, cg)})"


def _codegen_mbarrier_wait(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "mbarrier":
        return None
    if ".wait" not in inst.modifiers and ".try_wait" not in inst.modifiers:
        return None
    if len(inst.operands) != 2:
        return None
    return (
        f"ptx.mbarrier.wait({cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}{_pred_suffix(inst, cg)})"
    )


def _codegen_mbarrier_try_wait(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "mbarrier" or ".try_wait" not in inst.modifiers:
        return None
    if len(inst.operands) != 2:
        return None
    return (
        f"ptx.mbarrier.try_wait({cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}{_pred_suffix(inst, cg)})"
    )


def _codegen_mbarrier_arrive_expect_tx(inst: Instruction, cg: Any) -> str | None:
    # Wrapper does not implement this variant; reserved for future expansion.
    return None


# ----- fence -----

def _codegen_fence_proxy_async(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "fence":
        return None
    if inst.modifiers == (".proxy", ".async") and not inst.operands:
        return f"ptx.fence.proxy_async({_pred_suffix(inst, cg).lstrip(', ')})"
    return None


def _codegen_fence_mbarrier_init(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "fence":
        return None
    if inst.modifiers == (".mbarrier_init", ".release", ".cluster") and not inst.operands:
        return f"ptx.fence.mbarrier_init({_pred_suffix(inst, cg).lstrip(', ')})"
    return None


# ----- bar.sync -----

def _codegen_bar_sync(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "bar" or inst.modifiers != (".sync",):
        return None
    if len(inst.operands) == 0:
        return f"ptx.bar.sync({_pred_suffix(inst, cg).lstrip(', ')})"
    if len(inst.operands) == 1:
        return f"ptx.bar.sync({cg.operand(inst.operands[0])}{_pred_suffix(inst, cg)})"
    return None


# ----- stmatrix / ldmatrix -----

def _codegen_stmatrix(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "stmatrix":
        return None
    mods = inst.modifiers
    if mods[:2] != (".sync", ".aligned"):
        return None
    layout = None
    for m in mods[2:]:
        s = _strip_dot(m)
        if s.startswith("x") and s[1:].isdigit():
            layout = s
            break
    if layout is None or len(inst.operands) < 2:
        return None
    smem_op = cg.operand(inst.operands[0])
    regs_op = cg.operand(inst.operands[1])
    return (
        f"ptx.stmatrix(smem={smem_op}, regs={regs_op}, layout=\"{layout}\""
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_ldmatrix(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "ldmatrix":
        return None
    mods = inst.modifiers
    if mods[:2] != (".sync", ".aligned"):
        return None
    layout = None
    for m in mods[2:]:
        s = _strip_dot(m)
        if s.startswith("x") and s[1:].isdigit():
            layout = s
            break
    if layout is None or len(inst.operands) < 2:
        return None
    trans = ".trans" in mods
    dst_op = cg.operand(inst.operands[0])
    src_op = cg.operand(inst.operands[1])
    extra = ", trans=True" if trans else ""
    return (
        f"ptx.ldmatrix(dst={dst_op}, src={src_op}, layout=\"{layout}\"{extra}"
        f"{_pred_suffix(inst, cg)})"
    )


# ----- mov / add / sub / mul / mad / shl / shr / setp / cvt / ld / st / and / or / xor / not / ret / bra -----

def _codegen_mov(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "mov" or len(inst.modifiers) != 1:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None or len(inst.operands) != 2:
        return None
    return (
        f"ptx.mov({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}{_pred_suffix(inst, cg)})"
    )


def _codegen_add(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "add" or len(inst.modifiers) != 1:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None or len(inst.operands) != 3:
        return None
    return (
        f"ptx.add({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_sub(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "sub" or len(inst.modifiers) != 1:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None or len(inst.operands) != 3:
        return None
    return (
        f"ptx.sub({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_mul(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "mul" or len(inst.operands) != 3:
        return None
    mods = inst.modifiers
    mode = None
    dtype_mod = None
    for m in mods:
        s = _strip_dot(m)
        if s in ("lo", "hi", "wide"):
            mode = s
        else:
            dtype_mod = m
    if dtype_mod is None:
        return None
    dtype = _ptx_type_pyname(dtype_mod)
    if dtype is None:
        return None
    extra = f", mode=\"{mode}\"" if mode else ""
    return (
        f"ptx.mul({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}{extra}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_mad(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "mad" or len(inst.operands) != 4 or len(inst.modifiers) != 2:
        return None
    mode_mod, dtype_mod = inst.modifiers
    mode = _strip_dot(mode_mod)
    if mode not in ("lo", "hi", "wide"):
        return None
    dtype = _ptx_type_pyname(dtype_mod)
    if dtype is None:
        return None
    return (
        f"ptx.mad({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}, "
        f"{cg.operand(inst.operands[3])}, mode=\"{mode}\""
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_shl(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "shl" or len(inst.modifiers) != 1 or len(inst.operands) != 3:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None:
        return None
    return (
        f"ptx.shl({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_shr(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "shr" or len(inst.modifiers) != 1 or len(inst.operands) != 3:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None:
        return None
    return (
        f"ptx.shr({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_setp(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "setp" or len(inst.modifiers) != 2 or len(inst.operands) != 3:
        return None
    cmp_mod, dtype_mod = inst.modifiers
    cmp_op = _strip_dot(cmp_mod)
    if cmp_op not in ("lt", "le", "gt", "ge", "eq", "ne"):
        return None
    dtype = _ptx_type_pyname(dtype_mod)
    if dtype is None:
        return None
    return (
        f"ptx.setp(\"{cmp_op}\", {dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_cvt(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "cvt" or len(inst.operands) != 2:
        return None
    mods = list(inst.modifiers)
    if len(mods) < 2:
        return None
    src_mod = mods[-1]
    dst_mod = mods[-2]
    dst_t = _ptx_type_pyname(dst_mod)
    src_t = _ptx_type_pyname(src_mod)
    if not (dst_t and src_t):
        return None
    extras = mods[:-2]
    rounding = None
    ftz = False
    sat = False
    for m in extras:
        s = _strip_dot(m)
        if s == "ftz":
            ftz = True
        elif s == "sat":
            sat = True
        else:
            rounding = s
    extra_kwargs: list[str] = []
    if rounding is not None:
        extra_kwargs.append(f"rounding=\"{rounding}\"")
    if ftz:
        extra_kwargs.append("ftz=True")
    if sat:
        extra_kwargs.append("sat=True")
    extra_str = (", " + ", ".join(extra_kwargs)) if extra_kwargs else ""
    return (
        f"ptx.cvt({dst_t}, {src_t}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}{extra_str}{_pred_suffix(inst, cg)})"
    )


_SPACE_NAMES = ("global", "shared", "local", "param", "const")


def _codegen_ld(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "ld" or len(inst.operands) != 2:
        return None
    mods = inst.modifiers
    if not mods:
        return None
    space = None
    cache = None
    dtype_mod = None
    for m in mods:
        s = _strip_dot(m).split("::", 1)[0]
        if s in _SPACE_NAMES and space is None:
            space = s
        elif _ptx_type_pyname(m):
            dtype_mod = m
        else:
            cache = _strip_dot(m)
    if dtype_mod is None or space is None:
        return None
    dtype = _ptx_type_pyname(dtype_mod)
    if dtype is None:
        return None
    extras: list[str] = []
    if space != "global":
        extras.append(f"space=\"{space}\"")
    if cache is not None:
        extras.append(f"cache=\"{cache}\"")
    extra_str = (", " + ", ".join(extras)) if extras else ""
    return (
        f"ptx.ld({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}{extra_str}{_pred_suffix(inst, cg)})"
    )


def _codegen_st(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "st" or len(inst.operands) != 2:
        return None
    mods = inst.modifiers
    if not mods:
        return None
    space = None
    cache = None
    dtype_mod = None
    for m in mods:
        s = _strip_dot(m).split("::", 1)[0]
        if s in _SPACE_NAMES and space is None:
            space = s
        elif _ptx_type_pyname(m):
            dtype_mod = m
        else:
            cache = _strip_dot(m)
    if dtype_mod is None or space is None:
        return None
    dtype = _ptx_type_pyname(dtype_mod)
    if dtype is None:
        return None
    extras: list[str] = []
    if space != "global":
        extras.append(f"space=\"{space}\"")
    if cache is not None:
        extras.append(f"cache=\"{cache}\"")
    extra_str = (", " + ", ".join(extras)) if extras else ""
    return (
        f"ptx.st({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}{extra_str}{_pred_suffix(inst, cg)})"
    )


def _codegen_and(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "and" or len(inst.modifiers) != 1 or len(inst.operands) != 3:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None:
        return None
    return (
        f"ptx.and_({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_or(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "or" or len(inst.modifiers) != 1 or len(inst.operands) != 3:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None:
        return None
    return (
        f"ptx.or_({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_xor(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "xor" or len(inst.modifiers) != 1 or len(inst.operands) != 3:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None:
        return None
    return (
        f"ptx.xor_({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}, {cg.operand(inst.operands[2])}"
        f"{_pred_suffix(inst, cg)})"
    )


def _codegen_not(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "not" or len(inst.modifiers) != 1 or len(inst.operands) != 2:
        return None
    dtype = _ptx_type_pyname(inst.modifiers[0])
    if dtype is None:
        return None
    return (
        f"ptx.not_({dtype}, {cg.operand(inst.operands[0])}, "
        f"{cg.operand(inst.operands[1])}{_pred_suffix(inst, cg)})"
    )


def _codegen_ret(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "ret" or inst.modifiers or inst.operands:
        return None
    return f"ptx.ret({_pred_suffix(inst, cg).lstrip(', ')})"


def _codegen_bra(inst: Instruction, cg: Any) -> str | None:
    if inst.opcode != "bra" or inst.modifiers or len(inst.operands) != 1:
        return None
    target = cg.operand(inst.operands[0])
    return f"ptx.bra({target}{_pred_suffix(inst, cg)})"


# ===================================================================
# Codegen registry
# ===================================================================

from typing import Callable as _Callable  # noqa: E402

TYPED_WRAPPER_CODEGEN: dict[str, "list[_Callable]"] = {
    "wgmma": [
        _codegen_wgmma_mma_async,
        _codegen_wgmma_fence,
        _codegen_wgmma_commit_group,
        _codegen_wgmma_wait_group,
    ],
    "cp": [
        _codegen_cp_async_bulk_tensor,
        _codegen_cp_async_bulk_tensor_store,
    ],
    "tcgen05": [
        _codegen_tcgen05_alloc,
        _codegen_tcgen05_dealloc,
        _codegen_tcgen05_relinquish,
        _codegen_tcgen05_fence,
        _codegen_tcgen05_wait,
        _codegen_tcgen05_commit,
        _codegen_tcgen05_shift,
        _codegen_tcgen05_cp,
        _codegen_tcgen05_ld,
        _codegen_tcgen05_st,
        _codegen_tcgen05_mma,
    ],
    "setmaxnreg": [_codegen_setmaxnreg],
    "elect": [_codegen_elect_sync],
    "barrier": [_codegen_barrier_cluster],
    "mbarrier": [
        _codegen_mbarrier_init,
        _codegen_mbarrier_arrive,
        _codegen_mbarrier_wait,
        _codegen_mbarrier_try_wait,
        _codegen_mbarrier_arrive_expect_tx,
    ],
    "fence": [
        _codegen_fence_proxy_async,
        _codegen_fence_mbarrier_init,
    ],
    "bar": [_codegen_bar_sync],
    "stmatrix": [_codegen_stmatrix],
    "ldmatrix": [_codegen_ldmatrix],
    "mov": [_codegen_mov],
    "add": [_codegen_add],
    "sub": [_codegen_sub],
    "mul": [_codegen_mul],
    "mad": [_codegen_mad],
    "shl": [_codegen_shl],
    "shr": [_codegen_shr],
    "setp": [_codegen_setp],
    "cvt": [_codegen_cvt],
    "ld": [_codegen_ld],
    "st": [_codegen_st],
    "and": [_codegen_and],
    "or": [_codegen_or],
    "xor": [_codegen_xor],
    "not": [_codegen_not],
    "ret": [_codegen_ret],
    "bra": [_codegen_bra],
}


# ===================================================================
# Validator overloads for the new typed wrappers.
#
# The base spec table only knows about a subset of tcgen05 modifier
# variants. Register additional overloads here so the new wrappers
# emit instructions that pass validate_or_raise(). This does NOT
# modify pyptx/spec/* — it only calls the public register_overload()
# entry point at module load time.
# ===================================================================

def _register_wrapper_overloads() -> None:
    from pyptx.spec.ptx import InstructionSpec, ModifierGroup
    from pyptx.spec.validate import register_overload

    # cp.async.bulk.tensor extended (im2col / scatter4 / gather4)
    register_overload(InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",)),
            ModifierGroup("tensor", (".tensor",)),
            ModifierGroup("dim", (".1d", ".2d", ".3d", ".4d", ".5d")),
            ModifierGroup("dst", (".global", ".shared::cta", ".shared::cluster")),
            ModifierGroup("src", (".global", ".shared::cta", ".shared::cluster")),
            ModifierGroup("mode", (".im2col", ".im2col_w", ".im2col_w_128",
                                   ".tile_gather4", ".tile_scatter4",
                                   ".gather4", ".scatter4")),
            ModifierGroup("completion", (
                ".mbarrier::complete_tx::bytes",
                ".bulk_group",
            )),
            ModifierGroup("multicast", (".multicast::cluster",)),
            ModifierGroup("cache_hint", (".L2::cache_hint",)),
        ),
        operand_pattern="varies",
        min_operands=2,
        max_operands=10,
        description="cp.async.bulk.tensor with im2col/gather/scatter (Hopper/Blackwell)",
        since_version=(8, 0),
        arch="sm_90",
    ))

    # tcgen05.fence::before_thread_sync / .fence::after_thread_sync
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup(
                "op",
                (".fence::before_thread_sync", ".fence::after_thread_sync"),
                required=True,
            ),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="tcgen05 thread-sync fences (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # tcgen05.wait::ld / .wait::st
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (".wait::ld", ".wait::st"), required=True),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="tcgen05 ld/st wait barriers (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # tcgen05.ld / .st with shape and count modifiers
    _TCGEN05_SHAPES = (".16x32bx2", ".16x64b", ".16x128b", ".16x256b", ".32x32b")
    _TCGEN05_NUMS = (".x1", ".x2", ".x4", ".x8", ".x16", ".x32", ".x64", ".x128")
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (".ld", ".st"), required=True),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("shape", _TCGEN05_SHAPES),
            ModifierGroup("num", _TCGEN05_NUMS),
            ModifierGroup("type", (".b32",)),
            ModifierGroup("pack", (".pack::16b", ".unpack::16b")),
        ),
        operand_pattern="dst|[taddr], [taddr]|src",
        min_operands=2,
        max_operands=2,
        description="tcgen05.ld / tcgen05.st (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # tcgen05.alloc / .dealloc / .relinquish_alloc_permit
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup(
                "op",
                (".alloc", ".dealloc", ".relinquish_alloc_permit"),
                required=True,
            ),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("space", (".shared::cta",)),
            ModifierGroup("type", (".b32",)),
        ),
        operand_pattern="varies",
        min_operands=0,
        max_operands=2,
        description="tcgen05 alloc/dealloc/relinquish (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # tcgen05.commit (with mbarrier::arrive::one and optional multicast)
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (".commit",), required=True),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("completion", (".mbarrier::arrive::one",)),
            ModifierGroup("multicast", (".multicast::cluster",)),
            ModifierGroup("space", (".shared::cta", ".shared::cluster")),
            ModifierGroup("type", (".b64",)),
        ),
        operand_pattern="[mbar]",
        min_operands=1,
        max_operands=1,
        description="tcgen05.commit (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # tcgen05.shift
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (".shift",), required=True),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("direction", (".down",)),
        ),
        operand_pattern="[taddr]",
        min_operands=1,
        max_operands=1,
        description="tcgen05.shift (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # tcgen05.cp with size variants
    _TCGEN05_CP_SIZES = (
        ".128x256b", ".4x256b", ".128x128b",
        ".64x128b.warpx2::02_13", ".64x128b.warpx2::01_23",
        ".32x128b.warpx4",
        ".128x256b.b8x16.b4x16_p64", ".128x256b.b8x16.b6x16_p32",
        ".128x128b.b8x16.b4x16_p64", ".128x128b.b8x16.b6x16_p32",
        ".64x128b.warpx2::02_13.b8x16.b4x16_p64",
        ".64x128b.warpx2::02_13.b8x16.b6x16_p32",
        ".64x128b.warpx2::01_23.b8x16.b4x16_p64",
        ".64x128b.warpx2::01_23.b8x16.b6x16_p32",
        ".32x128b.warpx4.b8x16.b4x16_p64",
        ".32x128b.warpx4.b8x16.b6x16_p32",
        ".4x256b.b8x16.b4x16_p64", ".4x256b.b8x16.b6x16_p32",
    )
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (".cp",), required=True),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("size", _TCGEN05_CP_SIZES),
        ),
        operand_pattern="[taddr], [smem]",
        min_operands=2,
        max_operands=2,
        description="tcgen05.cp (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))

    # tcgen05.mma with kind::, optional sparse metadata, predicate, and
    # optional trailing scale immediate.
    register_overload(InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (".mma",), required=True),
            ModifierGroup("sparse", (".sp",)),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("kind", (
                ".kind::tf32", ".kind::f16", ".kind::i8",
                ".kind::f8f6f4", ".kind::mxf8f6f4",
                ".kind::mxf4", ".kind::mxf4nvf4",
            )),
            ModifierGroup("ashift", (".ashift",)),
            ModifierGroup("collector", (
                ".collector::a::discard",
                ".collector::a::lastuse",
                ".collector::a::fill",
                ".collector::a::use",
            )),
            ModifierGroup("block_scale", (".block_scale",)),
        ),
        operand_pattern="[d_tmem], a_desc, b_desc[, metadata], idesc, pred[, scale]",
        min_operands=5,
        max_operands=7,
        description="tcgen05.mma (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ))


_register_wrapper_overloads()
