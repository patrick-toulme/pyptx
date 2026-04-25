"""PTX text → pyptx Python code generator.

The killer onramp: take any existing PTX kernel, get working pyptx code.

Usage:
    from pyptx.codegen import ptx_to_python
    python_code = ptx_to_python(open("kernel.ptx").read())
    print(python_code)

Or from the command line:
    python -m pyptx.codegen kernel.ptx
"""

from __future__ import annotations

from pyptx.ir.nodes import (
    AddressOperand,
    BlankLine,
    Block,
    Comment,
    Function,
    ImmediateOperand,
    Instruction,
    Label,
    LabelOperand,
    Module,
    NegatedOperand,
    Operand,
    ParenthesizedOperand,
    PipeOperand,
    PragmaDirective,
    RawLine,
    RegDecl,
    RegisterOperand,
    Statement,
    VarDecl,
    VectorOperand,
)
from pyptx.ir.types import StateSpace
from pyptx.parser import parse


# Python keywords that need trailing underscore when used as attributes
_PYTHON_KEYWORDS = frozenset({
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
    "or", "pass", "raise", "return", "try", "while", "with", "yield",
})

# Special PTX registers → pyptx API calls
_SPECIAL_REGS: dict[str, str] = {
    "%tid.x": "ptx.special.tid.x()",
    "%tid.y": "ptx.special.tid.y()",
    "%tid.z": "ptx.special.tid.z()",
    "%ntid.x": "ptx.special.ntid.x()",
    "%ntid.y": "ptx.special.ntid.y()",
    "%ntid.z": "ptx.special.ntid.z()",
    "%ctaid.x": "ptx.special.ctaid.x()",
    "%ctaid.y": "ptx.special.ctaid.y()",
    "%ctaid.z": "ptx.special.ctaid.z()",
    "%nctaid.x": "ptx.special.nctaid.x()",
    "%nctaid.y": "ptx.special.nctaid.y()",
    "%nctaid.z": "ptx.special.nctaid.z()",
    "%laneid": "ptx.special.laneid()",
    "%warpid": "ptx.special.warpid()",
    "%clock": "ptx.special.clock()",
    "%clock64": "ptx.special.clock64()",
}

# PTX special registers that don't need a wrapper — emitted via ptx.sreg()
_SREG_NAMES = frozenset({
    "%cluster_ctarank", "%cluster_nctarank",
    "%clusterid.x", "%clusterid.y", "%clusterid.z", "%clusterid.w",
    "%nclusterid.x", "%nclusterid.y", "%nclusterid.z", "%nclusterid.w",
    "%cluster_ctaid.x", "%cluster_ctaid.y", "%cluster_ctaid.z",
    "%cluster_nctaid.x", "%cluster_nctaid.y", "%cluster_nctaid.z",
    "%is_explicit_cluster",
    "%smid", "%nsmid", "%gridid",
    "%lanemask_eq", "%lanemask_le", "%lanemask_lt", "%lanemask_ge", "%lanemask_gt",
    "%pm0", "%pm1", "%pm2", "%pm3",
    "%envreg0", "%envreg1", "%envreg2", "%envreg3", "%envreg4",
    "%envreg5", "%envreg6", "%envreg7", "%envreg8", "%envreg9",
    "%envreg10", "%envreg11", "%envreg12", "%envreg13", "%envreg14", "%envreg15",
    "%envreg16", "%envreg17", "%envreg18", "%envreg19", "%envreg20",
    "%envreg21", "%envreg22", "%envreg23", "%envreg24", "%envreg25",
    "%envreg26", "%envreg27", "%envreg28", "%envreg29", "%envreg30", "%envreg31",
    "%total_smem_size", "%dynamic_smem_size",
})

# Type name → pyptx.types import name
_TYPE_IMPORTS: dict[str, str] = {
    "b8": "b8", "b16": "b16", "b32": "b32", "b64": "b64", "b128": "b128",
    "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64",
    "s8": "s8", "s16": "s16", "s32": "s32", "s64": "s64",
    "f16": "f16", "f16x2": "f16x2", "bf16": "bf16", "bf16x2": "bf16x2",
    "tf32": "tf32", "f32": "f32", "f64": "f64",
    "e4m3": "e4m3", "e5m2": "e5m2", "pred": "pred",
}


def ptx_to_python(source: str, *, sugar: bool = False, kernel_name: str = "matmul_kernel") -> str:
    """Convert PTX source text to pyptx Python code.

    Parses the PTX, then generates Python code using the pyptx DSL API.
    The generated code can be executed directly.

    Args:
        source: PTX source text.
        sugar: If True, apply the sugar pass to clean up names, labels,
            and recognize high-level patterns (address chains, loops).
        kernel_name: Name to use for the kernel function when sugar=True.
    """
    module = parse(source)
    if sugar:
        from pyptx.codegen.sugar import apply_sugar
        module = apply_sugar(module, kernel_name=kernel_name)
    return ir_to_python(module, sugar=sugar)


def ir_to_python(module: Module, *, sugar: bool = False) -> str:
    """Convert an IR Module to pyptx Python code."""
    return _emit_module(module, sugar=sugar)


# ---------------------------------------------------------------------------
# Code generation internals
# ---------------------------------------------------------------------------

class _CodeGen:
    """Stateful code generator that tracks register mappings."""

    def __init__(self, *, sugar: bool = False) -> None:
        self._lines: list[str] = []
        self._indent = ""
        self._reg_arrays: dict[str, str] = {}  # "%r" → "r"
        self._reg_singles: dict[str, str] = {}  # "%p0" → "p0"
        self._types_needed: set[str] = set()
        self._var_names: dict[str, str] = {}
        self._all_pyvars: set[str] = set()  # all Python var names ever used (for uniqueness)
        self.sugar: bool = sugar
        self._label_annotations: dict[str, str] = {}  # label → comment to emit before it

    def line(self, text: str) -> None:
        self._lines.append(f"{self._indent}{text}")

    def blank(self) -> None:
        self._lines.append("")

    def indent(self) -> None:
        self._indent += "    "

    def dedent(self) -> None:
        self._indent = self._indent[:-4]

    def result(self) -> str:
        return "\n".join(self._lines) + "\n"

    def reg_ref(self, name: str) -> str:
        """Convert a PTX register name to a Python expression.

        %r5 → r[5], %tid.x → ptx.special.tid.x(), etc.
        """
        if name in _SPECIAL_REGS:
            return _SPECIAL_REGS[name]

        # Known special registers → ptx.sreg("%name")
        if name in _SREG_NAMES:
            return f'ptx.sreg("{name}")'

        for base, pyvar in self._reg_arrays.items():
            if name.startswith(base) and len(name) > len(base):
                suffix = name[len(base):]
                if suffix.isdigit():
                    return f"{pyvar}[{suffix}]"

        if name in self._reg_singles:
            return self._reg_singles[name]

        # Fallback: unknown register (likely a special register or %-prefixed name)
        # Emit via ptx.sreg() which accepts any register name
        return f'ptx.sreg("{name}")'

    def operand(self, op: Operand) -> str:
        """Render an operand as a Python expression."""
        match op:
            case RegisterOperand(name=name):
                return self.reg_ref(name)
            case ImmediateOperand(text=text):
                # Preserve hex/special format by wrapping non-decimal in quotes
                # so it passes through as ImmediateOperand text, not Python int
                if text.startswith("0x") or text.startswith("0X") or text.startswith("0d") or text.startswith("0f"):
                    return f'"{text}"'
                return text
            case LabelOperand(name=name):
                # Check if this "label" is actually a register reference
                # (PTX allows plain-name registers like "exit_predicate")
                if name in self._reg_singles:
                    return self._reg_singles[name]
                for base, pyvar in self._reg_arrays.items():
                    if name.startswith(base) and len(name) > len(base):
                        suffix = name[len(base):]
                        if suffix.isdigit():
                            return f"{pyvar}[{suffix}]"
                return f'"{name}"'
            case VectorOperand(elements=elems):
                inner = ", ".join(self.operand(e) for e in elems)
                return f"[{inner}]"
            case AddressOperand(base=base, offset=offset):
                base_py = self.reg_ref(base) if base.startswith("%") else f'"{base}"'
                if offset is not None:
                    # Complex offsets (TMA coords) — use raw string
                    if "," in str(offset) or "{" in str(offset):
                        return f'ptx.addr({base_py}, "{offset}")'
                    return f"ptx.addr({base_py}, {offset})"
                return f"ptx.addr({base_py})"
            case ParenthesizedOperand(elements=elems):
                inner = ", ".join(self.operand(e) for e in elems)
                if len(elems) == 1:
                    return f"({inner},)"
                return f"({inner})"
            case NegatedOperand(operand=inner):
                return f"~{self.operand(inner)}"
            case PipeOperand(left=left, right=right):
                return f"{self.operand(left)} | {self.operand(right)}"
            case _:
                return repr(op)

    def modifier_chain(self, modifiers: tuple[str, ...]) -> str:
        """Convert modifiers to dot-chain attribute access.

        Falls back to ptx.raw() for modifiers that can't be valid Python
        attribute names (e.g., starting with a digit like .128x256b).
        """
        parts = []
        for mod in modifiers:
            name = mod.lstrip(".")
            name = name.replace("::", "__")
            if name in _PYTHON_KEYWORDS:
                name += "_"
            # Digit-starting modifiers (.2d, .128x256b) → prefix with _
            if name and name[0].isdigit():
                name = "_" + name
            parts.append(f".{name}")
        return "".join(parts)


def _emit_module(module: Module, *, sugar: bool = False) -> str:
    cg = _CodeGen(sugar=sugar)
    arch = module.target.targets[0] if module.target.targets else "sm_90a"
    version = (module.version.major, module.version.minor)

    functions = [d for d in module.directives if isinstance(d, Function)]

    # Check for module-level extern .shared referenced by the kernel.
    # There may be multiple extern .shared decls (one per C++ namespace);
    # we need the one actually used by our function(s).
    all_extern_smem: set[str] = set()
    for d in module.directives:
        if isinstance(d, VarDecl) and d.state_space == StateSpace.SHARED:
            if d.linking is not None and d.linking.value == "extern":
                all_extern_smem.add(d.name)

    # Find which extern smem names are actually referenced in function bodies
    extern_smem_name: str | None = None
    if all_extern_smem and functions:
        from pyptx.emitter import emit as _emit_module_text
        # Quick check: emit the function body and scan for references
        func_text = _emit_module_text(module)
        for name in all_extern_smem:
            clean = name.rstrip("[]")
            # Check if the name appears in instructions (not just the decl)
            # by looking for it after the { line
            body_start = func_text.find("{")
            if body_start >= 0 and clean in func_text[body_start:]:
                extern_smem_name = name
                break
        if extern_smem_name is None and all_extern_smem:
            # Fallback: use the last one (usually the kernel's own namespace)
            extern_smem_name = sorted(all_extern_smem)[-1]

    for func in functions:
        _collect_types(func, cg)

    # Imports
    cg.line("from pyptx import kernel, reg, smem, ptx")
    if cg._types_needed:
        sorted_types = sorted(cg._types_needed)
        cg.line(f"from pyptx.types import {', '.join(sorted_types)}")
    cg.blank()

    for i, func in enumerate(functions):
        if i > 0:
            cg.blank()
            cg.blank()
        _emit_function(func, arch, version, cg, extern_smem_name=extern_smem_name)

    return cg.result()


def _collect_types(func: Function, cg: _CodeGen) -> None:
    def scan(stmts):
        for stmt in stmts:
            if isinstance(stmt, RegDecl):
                tn = stmt.type.value
                if tn in _TYPE_IMPORTS:
                    cg._types_needed.add(_TYPE_IMPORTS[tn])
            elif isinstance(stmt, VarDecl):
                tn = stmt.type.value
                if tn in _TYPE_IMPORTS:
                    cg._types_needed.add(_TYPE_IMPORTS[tn])
            elif isinstance(stmt, Instruction):
                # Typed wrappers (ptx.mov(u64, ...), ptx.add(f32, ...))
                # use type names as arguments, so scan instruction modifiers
                # for any .u64, .f32, etc. that might be emitted as names.
                for mod in stmt.modifiers:
                    name = mod.lstrip(".")
                    if name in _TYPE_IMPORTS:
                        cg._types_needed.add(_TYPE_IMPORTS[name])
            elif isinstance(stmt, Block):
                scan(stmt.body)
    scan(func.body)


def _pyvar_for_reg(base: str) -> str:
    clean = base.lstrip("%")
    if clean in _PYTHON_KEYWORDS:
        clean += "_"
    return clean


def _emit_function(
    func: Function,
    arch: str,
    version: tuple[int, int],
    cg: _CodeGen,
    *,
    extern_smem_name: str | None = None,
) -> None:
    from pyptx.ir.nodes import Param, FunctionDirective

    # Reset per-function register tracking (each function has its own scope)
    cg._reg_arrays = {}
    cg._reg_singles = {}
    cg._var_names = {}
    cg._all_pyvars = set()

    # Build @kernel kwargs
    kwargs: list[str] = [f'arch="{arch}"', f"version={version}"]

    # Extract raw_params from the function's param list
    if func.params:
        param_tuples: list[str] = []
        for p in func.params:
            # Build the type string: "u64.ptr.global" or "u32"
            # For array params like ".align 64 .b8 name[128]", encode
            # as "b8.align64.array128"
            type_str = p.type.value
            if p.ptr_state_space is not None:
                type_str += f".ptr.{p.ptr_state_space.value}"
            if p.alignment is not None:
                type_str += f".align{p.alignment}"
            if p.ptr_alignment is not None and p.ptr_state_space is not None:
                type_str += f".palign{p.ptr_alignment}"
            if p.array_size is not None:
                type_str += f".array{p.array_size}"
            param_tuples.append(f'("{type_str}", "{p.name}")')
        kwargs.append(f"raw_params=[{', '.join(param_tuples)}]")

    if extern_smem_name:
        # Strip trailing [] from array-style names
        clean_name = extern_smem_name.rstrip("[]")
        kwargs.append(f'extern_smem="{clean_name}"')

    # Extract function directives as raw_directives list
    if func.directives:
        dir_strs: list[str] = []
        for d in func.directives:
            if isinstance(d, FunctionDirective):
                if d.values:
                    vals = ", ".join(str(v) for v in d.values)
                    dir_strs.append(f'("{d.name}", ({vals},))')
                else:
                    dir_strs.append(f'("{d.name}", ())')
        if dir_strs:
            kwargs.append(f"raw_directives=[{', '.join(dir_strs)}]")

    cg.line(f"@kernel({', '.join(kwargs)})")
    cg.line(f"def {func.name}():")
    cg.indent()

    # When sugar is enabled, analyze the body for structural annotations
    if cg.sugar:
        from pyptx.codegen.sugar import _find_loops, _find_warp_split
        cg._label_annotations.update(_find_loops(func.body))
        cg._label_annotations.update(_find_warp_split(func.body))

    # Emit body in order, flattening nested blocks but preserving statement order
    # (declarations stay inline at their original position)
    _emit_body_stmts(func.body, cg, sugar=cg.sugar)

    cg.dedent()


def _is_mbarrier_wait_block(stmt) -> tuple | None:
    """Detect the mbarrier try_wait spin-loop pattern in a Block:

        { .reg .pred P1; L0: mbarrier.try_wait.parity ... P1, [addr], phase;
          @!P1 bra L0; @P1 bra L1; L1: }

    Returns (addr_operand, phase_operand) if matched, else None.
    """
    if not isinstance(stmt, Block):
        return None
    body = stmt.body
    # Need at least: RegDecl + Label + Instruction(try_wait) + bra + bra/label
    instrs = [s for s in body if isinstance(s, Instruction)]
    if len(instrs) < 2:
        return None
    # Find the try_wait instruction
    for s in instrs:
        if s.opcode == "mbarrier" and any("try_wait" in m for m in s.modifiers):
            # Operands: P1, [addr], phase  (or similar)
            if len(s.operands) >= 3:
                return (s.operands[1], s.operands[2])  # addr, phase
            elif len(s.operands) >= 2:
                return (s.operands[1], None)
    return None


def _is_wgmma_block(stmt) -> tuple | None:
    """Detect a wgmma.mma_async inside a Block (inline asm wrapper).

    Returns (opcode_mods, operands) if matched, else None.
    """
    if not isinstance(stmt, Block):
        return None
    instrs = [s for s in stmt.body if isinstance(s, Instruction)]
    for s in instrs:
        if s.opcode == "wgmma" and any("mma_async" in m for m in s.modifiers):
            return (s.modifiers, s.operands)
    return None


def _get_def_reg(inst: "Instruction") -> str | None:
    """Get the register defined by this instruction (first operand for most PTX)."""
    if not inst.operands:
        return None
    op = inst.operands[0]
    if isinstance(op, RegisterOperand):
        return op.name
    return None


def _get_use_regs(inst: "Instruction") -> set[str]:
    """Get all registers read by this instruction (operands 1+, plus predicate)."""
    uses = set()
    for op in inst.operands[1:]:
        if isinstance(op, RegisterOperand):
            uses.add(op.name)
        elif isinstance(op, AddressOperand):
            if op.base.startswith("%"):
                uses.add(op.base)
        elif isinstance(op, VectorOperand):
            for e in op.elements:
                if isinstance(e, RegisterOperand):
                    uses.add(e.name)
    if inst.predicate:
        uses.add(inst.predicate.register)
    return uses


def _analyze_temp_chains(body) -> tuple[set[str], dict[str, tuple[int, "Instruction"]], set[int]]:
    """Find temp registers (defined once, used once) for chain folding.

    Returns:
        temp_regs: set of register names that are temps
        temp_defs: map reg_name → (body_index, defining Instruction)
        skip_indices: body indices to skip (the temp's defining instruction)
    """
    from collections import Counter

    # Count defs and uses for each register
    defs: dict[str, list[tuple[int, Instruction]]] = {}
    use_counts: Counter = Counter()

    for i, s in enumerate(body):
        if not isinstance(s, Instruction):
            continue
        d = _get_def_reg(s)
        if d:
            defs.setdefault(d, []).append((i, s))
        for u in _get_use_regs(s):
            use_counts[u] += 1

    # A temp is: defined exactly once AND used exactly once AND
    # the defining instruction is a pure computation (not a load/store/branch)
    FOLDABLE_OPCODES = frozenset({
        "add", "sub", "mul", "shl", "shr", "and", "or", "xor",
        "not", "neg", "mad", "bfe",
        # cvt excluded: width changes (u32→u64) must stay as explicit instructions
    })

    temp_regs = set()
    temp_defs_map: dict[str, tuple[int, Instruction]] = {}
    skip_indices: set[int] = set()

    # First pass: identify candidates
    candidates = set()
    for reg_name, def_list in defs.items():
        if len(def_list) != 1:
            continue
        idx, inst = def_list[0]
        if use_counts[reg_name] != 1:
            continue
        if inst.opcode not in FOLDABLE_OPCODES:
            continue
        if reg_name.startswith("%f"):
            continue
        candidates.add(reg_name)

    # Build a set of "dangerous" indices — positions right after barriers/fences
    BARRIER_OPCODES = frozenset({"bar", "barrier", "mbarrier", "fence", "cp"})
    dangerous_positions: set[int] = set()
    for i, s in enumerate(body):
        if isinstance(s, Instruction) and s.opcode in BARRIER_OPCODES:
            # Mark the next 5 positions as dangerous
            for j in range(i + 1, min(len(body), i + 6)):
                dangerous_positions.add(j)

    # Remove candidates at dangerous positions
    safe_candidates = set()
    for reg_name in candidates:
        idx, _ = defs[reg_name][0]
        if idx not in dangerous_positions:
            safe_candidates.add(reg_name)
    candidates = safe_candidates

    # Second pass: only skip if the consumer is also foldable or a chain endpoint
    # (i.e., the consumer uses the folded expression, not the raw register)
    # A temp is safe to skip if its single consumer either:
    # 1. Is also a temp (chain continues)
    # 2. Has at least one temp source and is a non-temp destination (endpoint)
    for reg_name in candidates:
        idx, inst = defs[reg_name][0]
        # Find the consumer instruction
        consumer_ok = False
        consumer_idx = None
        for i, s in enumerate(body):
            if not isinstance(s, Instruction):
                continue
            for op in s.operands[1:]:
                if isinstance(op, RegisterOperand) and op.name == reg_name:
                    consumer_dst = _get_def_reg(s)
                    if consumer_dst in candidates:
                        consumer_ok = True
                    elif s.opcode in FOLDABLE_OPCODES or s.opcode in ("or", "and", "add", "sub", "shl", "shr", "xor"):
                        consumer_ok = True
                    consumer_idx = i
                    break
            if consumer_ok:
                break

        # Only safe to skip if there are NO non-temp instructions between
        # the temp's definition and its consumer. Otherwise, skipping
        # would reorder instructions past non-temps.
        if consumer_ok and consumer_idx is not None:
            safe_to_skip = True
            for j in range(idx + 1, consumer_idx):
                s = body[j]
                if isinstance(s, Instruction):
                    j_dst = _get_def_reg(s)
                    if j_dst not in candidates:
                        # Non-temp instruction between def and consumer
                        safe_to_skip = False
                        break
            if not safe_to_skip:
                consumer_ok = False

        if consumer_ok:
            temp_regs.add(reg_name)
            temp_defs_map[reg_name] = (idx, inst)
            skip_indices.add(idx)

    return temp_regs, temp_defs_map, skip_indices


def _fold_expression(reg_name: str, cg: "_CodeGen", depth: int = 0) -> str:
    """Recursively fold a temp register into a nested ptx.inst.* call.

    Since ptx.inst.__call__ now returns the destination Reg, we can
    nest calls: ptx.inst.and_.b32(r[194], ptx.inst.add.s32(r[193], r[192], -8192), 262016)

    This preserves the EXACT instruction modifiers (.s32 vs .u32) from
    the original PTX — unlike Python operator overloading which may use
    different signedness.
    """
    if depth > 6 or reg_name not in cg._temp_regs:
        return cg.reg_ref(reg_name)

    _, inst = cg._temp_defs[reg_name]

    # cvt: don't fold through — width conversions must remain as instructions
    if inst.opcode == "cvt":
        return cg.reg_ref(reg_name)

    # Build a nested ptx.inst.* call
    dst_ref = cg.reg_ref(reg_name)
    opcode_py = inst.opcode
    if opcode_py in _PYTHON_KEYWORDS:
        opcode_py += "_"
    chain = f"ptx.inst.{opcode_py}{cg.modifier_chain(inst.modifiers)}"

    # Recursively fold source operands
    folded_srcs = [_fold_operand(op, cg, depth + 1) for op in inst.operands[1:]]

    return f"{chain}({dst_ref}, {', '.join(folded_srcs)})"


def _fold_operand(op, cg: "_CodeGen", depth: int = 0) -> str:
    """Fold an operand, recursively inlining temps."""
    if isinstance(op, RegisterOperand):
        if op.name in cg._temp_regs:
            return _fold_expression(op.name, cg, depth)
        return cg.reg_ref(op.name)
    if isinstance(op, ImmediateOperand):
        # Show hex for large constants (likely descriptor bit patterns)
        try:
            val = int(op.text)
            if abs(val) > 0xFFFF:
                return f"0x{val & 0xFFFFFFFFFFFFFFFF:X}"
            return op.text
        except ValueError:
            return op.text
    return cg.operand(op)


def _instruction_signature(stmt) -> str | None:
    """Return a canonical string key for an instruction's shape (opcode+modifiers+operand_types).

    Two instructions with the same signature differ only in their operand
    register indices / immediate values — they're structurally identical.
    """
    if isinstance(stmt, Instruction):
        op_types = []
        for op in stmt.operands:
            op_types.append(type(op).__name__)
        pred = "P" if stmt.predicate else ""
        return f"{pred}{stmt.opcode}{''.join(stmt.modifiers)}({','.join(op_types)})"
    if isinstance(stmt, Block):
        # Signature of the block's contents
        sigs = [_instruction_signature(s) for s in stmt.body if _instruction_signature(s)]
        return "BLOCK[" + ";".join(sigs) + "]"
    return None


def _find_repeated_groups(body, min_group_size: int = 3, min_repeats: int = 4):
    """Find repeated blocks of N identical instruction signatures.

    Skips non-instruction statements (comments, blank lines) when
    building the signature list, but maps back to original indices.

    Returns a list of (start_index, group_size, repeat_count) tuples
    where indices and sizes refer to the ORIGINAL body (including skipped stmts).
    """
    # Build a list of (original_index, signature) for instructional stmts only
    indexed_sigs: list[tuple[int, str]] = []
    for i, s in enumerate(body):
        sig = _instruction_signature(s)
        if sig is not None:
            indexed_sigs.append((i, sig))

    if not indexed_sigs:
        return []

    n = len(indexed_sigs)
    results = []
    used = set()

    for gsize in range(min(15, n // min_repeats), min_group_size - 1, -1):
        si = 0
        while si <= n - gsize * min_repeats:
            if indexed_sigs[si][0] in used:
                si += 1
                continue

            pattern = tuple(indexed_sigs[si + j][1] for j in range(gsize))
            count = 1
            pos = si + gsize
            while pos + gsize <= n:
                match = all(indexed_sigs[pos + j][1] == pattern[j] for j in range(gsize))
                if not match:
                    break
                count += 1
                pos += gsize

            if count >= min_repeats:
                # Map back to original body indices
                orig_start = indexed_sigs[si][0]
                orig_end = indexed_sigs[si + gsize * count - 1][0]
                # Group size in original body = span of one repeat
                one_repeat_end = indexed_sigs[si + gsize - 1][0]
                orig_gsize = one_repeat_end - orig_start + 1
                results.append((orig_start, orig_gsize, count))
                for k in range(si, si + gsize * count):
                    used.add(indexed_sigs[k][0])
                si += gsize * count
            else:
                si += 1

    return sorted(results)


def _emit_body_stmts(body, cg: _CodeGen, *, sugar: bool = False) -> None:
    """Emit all body statements in order.

    When sugar=True, recognizes patterns and emits sugar:
    - mbarrier try_wait spin-loops → ptx.mbarrier.wait(addr, phase)
    - wgmma blocks → compact ptx.wgmma.mma_async(...) calls
    - Repeated instruction groups → Python for loops
    """
    # Pre-pass: find repeated groups if sugar is enabled
    repeated_groups: dict[int, tuple[int, int]] = {}  # start_idx → (group_size, count)
    if sugar:
        for start, gsize, count in _find_repeated_groups(body, min_group_size=5, min_repeats=2):
            repeated_groups[start] = (gsize, count)

    # Pre-pass: use-def analysis for address chain folding
    # Find registers defined exactly once and used exactly once (temps)
    if sugar:
        cg._temp_regs, cg._temp_defs, cg._skip_indices = _analyze_temp_chains(body)
    else:
        cg._temp_regs = set()
        cg._temp_defs = {}
        cg._skip_indices = set()

    # Pre-pass: detect loops (label + ... + bra back to label)
    loop_info: dict[int, tuple[int, str | None]] = {}  # label_idx → (bra_idx, pred_reg_or_None)
    if sugar:
        label_positions: dict[str, int] = {}
        for i, s in enumerate(body):
            if isinstance(s, Label):
                label_positions[s.name] = i
        for i, s in enumerate(body):
            if isinstance(s, Instruction) and s.opcode == "bra":
                for op in s.operands:
                    if isinstance(op, LabelOperand) and op.name in label_positions:
                        lbl_idx = label_positions[op.name]
                        if lbl_idx < i:  # backward branch = loop
                            pred_name = None
                            if s.predicate:
                                pred_name = s.predicate.register
                                if s.predicate.negated:
                                    pred_name = "!" + pred_name
                            elif ".uni" in s.modifiers:
                                pred_name = None  # unconditional
                            loop_info[lbl_idx] = (i, pred_name)

    idx = 0
    while idx < len(body):
        # Group adjacent temp chains into ptx.expr() blocks.
        # All instructions stay in order — ptx.expr() just captures them
        # into one CompoundExpr IR node for visual grouping.
        if sugar and idx in cg._skip_indices:
            in_group = any(
                g_start <= idx < g_start + g_size * g_count
                for g_start, (g_size, g_count) in repeated_groups.items()
            )
            if not in_group:
                # Find the full chain: consecutive temps + their endpoint
                chain_end = idx
                while chain_end < len(body) and chain_end in cg._skip_indices:
                    chain_end += 1
                # Include the endpoint (non-temp that consumes the last temp)
                if chain_end < len(body) and isinstance(body[chain_end], Instruction):
                    has_temp_src = any(
                        isinstance(op, RegisterOperand) and op.name in cg._temp_regs
                        for op in body[chain_end].operands[1:]
                    )
                    if has_temp_src:
                        chain_end += 1
                # Emit as ptx.expr() block
                if chain_end - idx >= 3:  # only group 3+ instruction chains
                    dst = _get_def_reg(body[chain_end - 1]) if isinstance(body[chain_end - 1], Instruction) else None
                    dst_ref = cg.reg_ref(dst) if dst else "?"
                    # Show the folded expression as a comment
                    folded_srcs = [_fold_operand(op, cg) for op in body[chain_end-1].operands[1:]] if isinstance(body[chain_end-1], Instruction) else []
                    cg.line(f"with ptx.expr():  # {dst_ref}")
                    cg.indent()
                    for j in range(idx, chain_end):
                        _emit_body_stmts_single(body[j], cg, sugar=False)
                    cg.dedent()
                    idx = chain_end
                    continue
                # Short chain: emit normally
                _emit_body_stmts_single(body[idx], cg, sugar=sugar)
                idx += 1
                continue

        # Check if this index starts a repeated group
        if sugar and idx in repeated_groups:
            gsize, count = repeated_groups[idx]
            _emit_parameterized_loop(body, idx, gsize, count, cg)
            idx += gsize * count
            continue

        # Check if this label starts a loop
        if sugar and idx in loop_info:
            bra_idx, pred_name = loop_info[idx]
            label_name = body[idx].name
            if pred_name:
                pred_ref = cg.reg_ref(pred_name.lstrip("!"))
                if pred_name.startswith("!"):
                    pred_ref = f"~{pred_ref}"
                cg.line(f'with ptx.loop("{label_name}", pred={pred_ref}):')
            else:
                cg.line(f'with ptx.loop("{label_name}"):')
            cg.indent()
            # Recursively emit loop body — this picks up nested loops
            loop_body = tuple(body[idx + 1 : bra_idx])
            _emit_body_stmts(loop_body, cg, sugar=sugar)
            cg.dedent()
            idx = bra_idx + 1
            continue

        _emit_body_stmts_single(body[idx], cg, sugar=sugar)
        idx += 1


def _extract_operand_value(op) -> int | None:
    """Extract a numeric value from a register index or immediate."""
    if isinstance(op, RegisterOperand):
        # %r325 → 325, %f5882 → 5882
        name = op.name.lstrip("%")
        # Find trailing digits
        i = len(name)
        while i > 0 and name[i-1].isdigit():
            i -= 1
        if i < len(name):
            return int(name[i:])
    if isinstance(op, ImmediateOperand):
        try:
            return int(op.text)
        except ValueError:
            pass
    return None


def _parameterize_operand(values: list[int | None], loop_var: str = "_i") -> str | None:
    """Given a list of numeric values across N iterations, return an expression.

    E.g. [147456, 149760, 152064, ...] → "147456 + _i * 2304"
    E.g. [5889, 5889, 5889, ...] → "5889" (constant)
    Returns None if no clean pattern is found.
    """
    if not values or any(v is None for v in values):
        return None
    vals = [v for v in values if v is not None]
    if len(set(vals)) == 1:
        return str(vals[0])  # constant
    if len(vals) < 2:
        return None
    stride = vals[1] - vals[0]
    if stride == 0:
        return str(vals[0])
    # Check arithmetic progression
    for i in range(2, len(vals)):
        if vals[i] - vals[i-1] != stride:
            return None  # not a clean AP
    base = vals[0]
    if stride == 1:
        return f"{base} + {loop_var}" if base else loop_var
    elif stride == -1:
        return f"{base} - {loop_var}" if base else f"-{loop_var}"
    elif stride > 0:
        return f"{base} + {loop_var} * {stride}"
    else:
        return f"{base} - {loop_var} * {-stride}"


def _emit_parameterized_loop(
    body: tuple, start: int, gsize: int, count: int, cg: _CodeGen,
) -> None:
    """Emit a for loop with parameterized register indices.

    Compares operands across iterations to find arithmetic progressions,
    then emits the loop body with expressions like `base + _i * stride`.
    """
    # Collect the instructional statements for each iteration
    iterations: list[list] = []
    for it in range(count):
        iter_stmts = []
        for j in range(gsize):
            s = body[start + it * gsize + j]
            if _instruction_signature(s) is not None:
                iter_stmts.append(s)
        iterations.append(iter_stmts)

    if not iterations or not iterations[0]:
        # Fallback: emit without parameterization
        cg.line(f"# Repeated block: {count}x")
        for j in range(gsize):
            _emit_body_stmts_single(body[start + j], cg, sugar=True)
        return

    n_instrs = len(iterations[0])
    cg.line(f"for _i in range({count}):  # {n_instrs} instructions × {count} iterations")
    cg.indent()

    # For each instruction position, parameterize the operands
    for inst_idx in range(n_instrs):
        first = iterations[0][inst_idx]
        if not isinstance(first, (Instruction, Block)):
            _emit_body_stmts_single(first, cg, sugar=True)
            continue

        # For Blocks (like cvt in scope), just emit the first instance
        # with a comment — parameterizing inside blocks is too complex
        if isinstance(first, Block):
            _emit_body_stmts_single(first, cg, sugar=True)
            continue

        # Compare operands across iterations
        param_ops: list[str] = []
        for op_idx, op in enumerate(first.operands):
            values = []
            for it in range(count):
                if inst_idx < len(iterations[it]):
                    inst = iterations[it][inst_idx]
                    if isinstance(inst, Instruction) and op_idx < len(inst.operands):
                        values.append(_extract_operand_value(inst.operands[op_idx]))
                    else:
                        values.append(None)
                else:
                    values.append(None)

            expr = _parameterize_operand(values)
            if expr is not None and any(v != values[0] for v in values if v is not None):
                # Varying operand — use parameterized expression
                # Need to reconstruct the operand string with the expression
                base_str = cg.operand(op)
                # Replace the numeric part with the expression
                if isinstance(op, RegisterOperand):
                    prefix = op.name.lstrip("%")
                    i = len(prefix)
                    while i > 0 and prefix[i-1].isdigit():
                        i -= 1
                    reg_prefix = prefix[:i]
                    # Find which array this register belongs to
                    for arr_name, pyvar in cg._reg_arrays.items():
                        if arr_name == f"%{reg_prefix}":
                            param_ops.append(f"{pyvar}[{expr}]")
                            break
                    else:
                        param_ops.append(base_str)
                elif isinstance(op, ImmediateOperand):
                    param_ops.append(expr)
                else:
                    param_ops.append(base_str)
            else:
                param_ops.append(cg.operand(op))

        # Emit the parameterized instruction
        pred_str = ""
        if first.predicate is not None:
            p_ref = cg.reg_ref(first.predicate.register)
            pred_str = f", pred={'~' if first.predicate.negated else ''}{p_ref}"

        opcode_py = first.opcode
        if opcode_py in _PYTHON_KEYWORDS:
            opcode_py += "_"
        chain = f"ptx.inst.{opcode_py}{cg.modifier_chain(first.modifiers)}"
        args = ", ".join(param_ops)
        if pred_str:
            cg.line(f"{chain}({args}{pred_str})")
        else:
            cg.line(f"{chain}({args})")

    cg.dedent()


def _collect_chain(endpoint: "Instruction", cg: "_CodeGen") -> list[tuple[str, "Instruction"]]:
    """Walk backward through the temp chain from an endpoint instruction.

    Returns a list of (reg_name, instruction) tuples in EXECUTION ORDER
    (first to last), ending with the endpoint itself.
    """
    chain: list[tuple[str | None, "Instruction"]] = []

    def _walk(inst: "Instruction") -> None:
        """Recursively collect temp sources."""
        for op in inst.operands[1:]:
            if isinstance(op, RegisterOperand) and op.name in cg._temp_regs:
                _, temp_inst = cg._temp_defs[op.name]
                _walk(temp_inst)  # recurse into sources first (execution order)
                chain.append((op.name, temp_inst))

    _walk(endpoint)
    dst = _get_def_reg(endpoint)
    chain.append((dst, endpoint))
    return chain


def _emit_pipe_chain(chain: list[tuple[str, "Instruction"]], cg: "_CodeGen") -> None:
    """Emit a chain of instructions as a ptx.pipe() call."""
    if not chain:
        return

    # First instruction: find the non-temp source to start the pipe
    first_name, first_inst = chain[0]
    # The pipe source is the first non-temp source operand of the first instruction
    src_op = None
    for op in first_inst.operands[1:]:
        if isinstance(op, RegisterOperand) and op.name not in cg._temp_regs:
            src_op = cg.reg_ref(op.name)
            break
        elif isinstance(op, ImmediateOperand):
            src_op = cg.operand(op)
            break
    if src_op is None:
        src_op = cg.operand(first_inst.operands[1]) if len(first_inst.operands) > 1 else "?"

    parts = [f"ptx.pipe({src_op})"]
    for reg_name, inst in chain:
        opcode_py = inst.opcode
        if opcode_py in _PYTHON_KEYWORDS:
            opcode_py += "_"
        mods = cg.modifier_chain(inst.modifiers)
        dst_ref = cg.reg_ref(reg_name) if reg_name else "?"

        # Collect non-pipe operands (skip dst and the piped source)
        extra_ops = []
        piped_src_found = False
        for i, op in enumerate(inst.operands):
            if i == 0:
                continue  # skip dst
            op_str = cg.operand(op)
            if not piped_src_found and isinstance(op, RegisterOperand) and op.name in cg._temp_regs:
                piped_src_found = True
                continue  # this is the piped-through source, skip
            if not piped_src_found and i == 1:
                piped_src_found = True
                continue  # first source is piped
            extra_ops.append(op_str)

        extra = ", ".join(extra_ops)
        if extra:
            parts.append(f".{opcode_py}{mods}({dst_ref}, {extra})")
        else:
            parts.append(f".{opcode_py}{mods}({dst_ref})")

    line = " \\\n        ".join(parts)
    cg.line(line)


def _emit_body_stmts_single(stmt, cg: _CodeGen, *, sugar: bool = False) -> None:
    """Emit a single body statement."""
    if isinstance(stmt, RegDecl):
        _emit_reg_decl(stmt, cg)
    elif isinstance(stmt, VarDecl):
        _emit_var_decl(stmt, cg)
    elif isinstance(stmt, Instruction):
        if sugar:
            if stmt.opcode == "wgmma" and any("fence" in m for m in stmt.modifiers):
                cg.blank()
                cg.line("# --- wgmma compute ---")
            elif stmt.opcode == "setmaxnreg":
                cg.blank()
                if ".inc" in stmt.modifiers:
                    cg.line("# === CONSUMER WARP GROUP ===")
                elif ".dec" in stmt.modifiers:
                    cg.line("# === PRODUCER WARP GROUP ===")
            elif stmt.opcode == "barrier" and any("cluster" in m for m in stmt.modifiers):
                cg.line("# --- cluster barrier ---")

            # Fold address chains: skip temp instructions, emit chain endpoint
            # as assignment expression using Reg operator overloading.
            # The operators emit the same PTX instructions internally.
            # RegArray.__setitem__ emits a final mov to the target slot.
            if cg._temp_regs and _get_def_reg(stmt) and len(stmt.operands) >= 2:
                dst = _get_def_reg(stmt)
                has_temp_src = any(
                    isinstance(op, RegisterOperand) and op.name in cg._temp_regs
                    for op in stmt.operands[1:]
                )
                if has_temp_src and dst not in cg._temp_regs:
                    dst_ref = cg.reg_ref(dst)
                    folded_srcs = [_fold_operand(op, cg) for op in stmt.operands[1:]]
                    OP_MAP = {
                        ("or", ".b64"): "|", ("or", ".b32"): "|",
                        ("and", ".b32"): "&", ("and", ".b64"): "&",
                        ("add", ".s32"): "+", ("add", ".s64"): "+",
                        ("add", ".u32"): "+",
                        ("sub", ".s32"): "-", ("sub", ".u32"): "-",
                        ("shl", ".b32"): "<<", ("shl", ".b64"): "<<",
                        ("shr", ".u32"): ">>", ("shr", ".s32"): ">>",
                        ("xor", ".b32"): "^", ("xor", ".b64"): "^",
                    }
                    op_sym = None
                    for (opc, mod), sym in OP_MAP.items():
                        if stmt.opcode == opc and mod in stmt.modifiers:
                            op_sym = sym
                            break
                    if op_sym and len(folded_srcs) == 2:
                        cg.line(f"# {dst_ref} = ({folded_srcs[0]} {op_sym} {folded_srcs[1]})")

        _emit_instruction(stmt, cg)
    elif isinstance(stmt, Label):
        if sugar and stmt.name in cg._label_annotations:
            cg.blank()
            cg.line(cg._label_annotations[stmt.name])
        cg.line(f'ptx.label("{stmt.name}")')
    elif isinstance(stmt, PragmaDirective):
        cg.line(f'ptx.pragma("{stmt.value}")')
    elif isinstance(stmt, Block):
        if sugar:
            mbar_match = _is_mbarrier_wait_block(stmt)
            if mbar_match is not None:
                addr_op, phase_op = mbar_match
                addr_str = cg.operand(addr_op)
                phase_str = cg.operand(phase_op) if phase_op is not None else "0"
                cg.line(f"ptx.mbarrier.wait({addr_str}, {phase_str})")
                return

            wgmma_match = _is_wgmma_block(stmt)
            if wgmma_match is not None:
                _emit_wgmma_sugar(wgmma_match[0], wgmma_match[1], cg)
                return

        cg.line("with ptx.scope():")
        cg.indent()
        saved_arrays = dict(cg._reg_arrays)
        saved_singles = dict(cg._reg_singles)
        _emit_body_stmts(stmt.body, cg, sugar=sugar)
        cg._reg_arrays = saved_arrays
        cg._reg_singles = saved_singles
        cg.dedent()
    # Skip Comment, BlankLine, RawLine — they're formatting/metadata


def _emit_wgmma_sugar(modifiers: tuple[str, ...], operands: tuple, cg: _CodeGen) -> None:
    """Emit a compact wgmma.mma_async call.

    Instead of listing 128 individual register operands, group them
    as a slice of the accumulator array: acc[start:end].
    """
    # Parse shape from modifiers: .mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
    shape_str = ""
    dtype_d = dtype_a = dtype_b = ""
    for m in modifiers:
        m = m.lstrip(".")
        if m.startswith("m") and "n" in m and "k" in m:
            shape_str = m  # e.g. "m64n256k16"
        elif m in ("f32", "f16", "bf16", "tf32", "e4m3", "e5m2"):
            if not dtype_d:
                dtype_d = m
            elif not dtype_a:
                dtype_a = m
            else:
                dtype_b = m

    # Operands: [d_regs...], desc_a, desc_b, scale_d, imm1, imm2, imm3, imm4
    # First operand is a VectorOperand (the accumulator registers)
    from pyptx.ir.nodes import VectorOperand, RegisterOperand
    if operands and isinstance(operands[0], VectorOperand):
        d_vec = operands[0]
        n_regs = len(d_vec.elements)
        # Try to express as acc[start:end] if they're contiguous from an array
        first = d_vec.elements[0] if d_vec.elements else None
        if isinstance(first, RegisterOperand):
            first_ref = cg.reg_ref(first.name)
            # Check if it looks like f[N] from an array
            if "[" in first_ref:
                base = first_ref.split("[")[0]
                last = d_vec.elements[-1]
                if isinstance(last, RegisterOperand):
                    last_ref = cg.reg_ref(last.name)
                    if "[" in last_ref and last_ref.split("[")[0] == base:
                        first_idx = int(first_ref.split("[")[1].rstrip("]"))
                        last_idx = int(last_ref.split("[")[1].rstrip("]"))
                        d_str = f"{base}[{min(first_idx, last_idx)}:{max(first_idx, last_idx)+1}]"
                    else:
                        d_str = first_ref
                else:
                    d_str = first_ref
            else:
                d_str = first_ref
        else:
            d_str = "acc"
    else:
        d_str = "acc"

    # Remaining operands: desc_a, desc_b, scale_d, imm...
    rest = [cg.operand(op) for op in operands[1:]]
    desc_a = rest[0] if len(rest) > 0 else "?"
    desc_b = rest[1] if len(rest) > 1 else "?"
    scale_d = rest[2] if len(rest) > 2 else "1"

    # Emit as a compact list comprehension if regs are contiguous
    if "[" in d_str and ":" in d_str:
        # d_str is like "f[5762:5890]" — emit as list(f[i] for i in range(...))
        base = d_str.split("[")[0]
        range_part = d_str.split("[")[1].rstrip("]")
        lo, hi = range_part.split(":")
        d_emit = f"[{base}[i] for i in range({lo}, {hi})]"
    else:
        d_emit = d_str
    cg.line(f"# wgmma {shape_str} {dtype_d}.{dtype_a}.{dtype_b} ({n_regs} regs)")
    cg.line(f"ptx.inst.wgmma.mma_async.sync.aligned.{shape_str}.{dtype_d}.{dtype_a}.{dtype_b}("
            f"{d_emit}, {desc_a}, {desc_b}, {', '.join(rest[2:])})")


def _emit_reg_decl(rd: RegDecl, cg: _CodeGen) -> None:
    type_name = _TYPE_IMPORTS.get(rd.type.value, rd.type.value)
    base_pyvar = _pyvar_for_reg(rd.name)

    # Shadowing in nested blocks: use a fresh Python name if the outer
    # scope already has this register.
    pyvar = base_pyvar
    if rd.name in cg._reg_arrays or rd.name in cg._reg_singles:
        # Find a unique suffix
        i = 1
        while f"{base_pyvar}_{i}" in cg._all_pyvars:
            i += 1
        pyvar = f"{base_pyvar}_{i}"
    cg._all_pyvars.add(pyvar)

    if rd.count is not None:
        cg.line(f'{pyvar} = reg.array({type_name}, {rd.count}, name="{rd.name}")')
        cg._reg_arrays[rd.name] = pyvar
    else:
        cg.line(f'{pyvar} = reg.scalar({type_name}, name="{rd.name}")')
        cg._reg_singles[rd.name] = pyvar


def _emit_var_decl(vd: VarDecl, cg: _CodeGen) -> None:
    # Emit all VarDecls via ptx.var(), which supports any state space.
    # This covers .shared, .global, .local, .const, .param.
    type_name = _TYPE_IMPORTS.get(vd.type.value, vd.type.value)
    state_space_name = vd.state_space.value.replace("::", "__")
    # Avoid shadowing the smem module import
    pyvar = f"s_{vd.name}" if vd.name == "smem" else vd.name

    kwargs = [f'"{state_space_name}"', type_name, f'"{vd.name}"']
    if vd.array_size is not None:
        kwargs.append(f"size={vd.array_size}")
    if vd.alignment is not None:
        kwargs.append(f"align={vd.alignment}")
    if vd.linking is not None:
        kwargs.append(f'linking="{vd.linking.value}"')

    cg.line(f'ptx.var({", ".join(kwargs)})')
    cg._var_names[vd.name] = pyvar


def _emit_instruction(inst: Instruction, cg: _CodeGen) -> None:
    from pyptx.emitter.emitter import _emit_operand as emit_ir_operand

    operand_strs = [cg.operand(op) for op in inst.operands]

    pred_str = ""
    if inst.predicate is not None:
        p_ref = cg.reg_ref(inst.predicate.register)
        if inst.predicate.negated:
            pred_str = f", pred=~{p_ref}"
        else:
            pred_str = f", pred={p_ref}"

    # .loc and .file are debug directives — skip them in transpiled output.
    # They're not needed for execution and .loc's comma-separated format
    # doesn't match PTX's space-separated syntax.
    if inst.opcode in (".loc", ".file"):
        return

    # ret
    if inst.opcode == "ret" and not inst.operands:
        cg.line(f"ptx.ret({pred_str.lstrip(', ')})" if pred_str else "ptx.ret()")
        return

    # bra (only the no-modifier case; with modifiers falls through to generic)
    if inst.opcode == "bra" and len(inst.operands) == 1 and not inst.modifiers:
        target = cg.operand(inst.operands[0])
        cg.line(f"ptx.bra({target}{pred_str})" if pred_str else f"ptx.bra({target})")
        return

    # Sugar rewrites: replace verbose ptx.inst.* with high-level DSL calls
    if cg.sugar:
        # mov.u32(dst, tid.x) → ptx.inst.mov.u32(dst, ptx.special.tid.x())
        # Keep as ptx.inst.mov but use the readable special reg names
        if inst.opcode == "mov" and ".u32" in inst.modifiers and len(inst.operands) == 2:
            src = inst.operands[1]
            if isinstance(src, RegisterOperand):
                if src.name in _SPECIAL_REGS:
                    cg.line(f"ptx.inst.mov.u32({operand_strs[0]}, {_SPECIAL_REGS[src.name]})")
                    return
                if src.name in _SREG_NAMES:
                    cg.line(f'ptx.inst.mov.u32({operand_strs[0]}, ptx.sreg("{src.name}"))')
                    return

        # Operator sugar for simple arithmetic on b32/u32 regs:
        #   add.s32(dst, a, b) → dst = a + b
        #   shl.b32(dst, a, N) → dst = a << N
        #   and.b32(dst, a, M) → dst = a & M
        #   etc.
        _ARITH_OPS = {
            ("add", ".s32"): "+", ("add", ".u32"): "+",
            ("sub", ".s32"): "-", ("sub", ".u32"): "-",
            ("shl", ".b32"): "<<",
            ("and", ".b32"): "&",
            ("xor", ".b32"): "^",
        }
        for (opc, mod), sym in _ARITH_OPS.items():
            if inst.opcode == opc and mod in inst.modifiers and len(inst.operands) == 3:
                dst_str = operand_strs[0]
                cg.line(f"{dst_str} = ({operand_strs[1]} {sym} {operand_strs[2]}){pred_str}")
                return

        # bar.sync(N) → ptx.bar.sync(N)
        if inst.opcode == "bar" and ".sync" in inst.modifiers:
            cg.line(f"ptx.bar.sync({operand_strs[0]}{pred_str})")
            return

        # setmaxnreg → ptx.inst.setmaxnreg(N)
        if inst.opcode == "setmaxnreg" and len(inst.operands) == 1:
            cg.line(f"ptx.inst.setmaxnreg{cg.modifier_chain(inst.modifiers)}({operand_strs[0]})")
            return

    # General: ptx.inst.opcode.mod1.mod2(operands)
    opcode_py = inst.opcode
    # Opcodes starting with '.' (e.g. .loc, .file) or special chars → ptx.raw
    has_invalid_opcode = (
        not opcode_py
        or opcode_py[0].isdigit()
        or opcode_py.startswith(".")
        or opcode_py in ("{", "}")
        or not opcode_py.replace("_", "").isalnum()
    )
    if opcode_py in _PYTHON_KEYWORDS:
        opcode_py += "_"
    if has_invalid_opcode:
        # Fall back to ptx.raw for opcodes that can't be Python identifiers
        raw_parts = []
        if inst.predicate:
            neg = "!" if inst.predicate.negated else ""
            raw_parts.append(f"@{neg}{inst.predicate.register} ")
        raw_parts.append(inst.opcode + "".join(inst.modifiers))
        if inst.operands:
            op_strs_raw = [emit_ir_operand(op) for op in inst.operands]
            raw_parts.append(" " + ", ".join(op_strs_raw))
        raw_parts.append(";")
        cg.line(f'ptx.raw(\'{"".join(raw_parts)}\')')
        return

    chain = f"ptx.inst.{opcode_py}{cg.modifier_chain(inst.modifiers)}"
    args = ", ".join(operand_strs)
    if pred_str and not args:
        # Strip leading ", " from pred_str when there are no operands
        cg.line(f"{chain}({pred_str.lstrip(', ')})")
    else:
        cg.line(f"{chain}({args}{pred_str})")
