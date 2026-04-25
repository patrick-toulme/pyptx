# How It Works

This page is a walkthrough of what happens when you write:

```python
from pyptx import kernel, reg, ptx
from pyptx.types import f32, u32

@kernel(arch="sm_90a")
def tiny():
    x = reg.scalar(u32)
    ptx.inst.mov.u32(x, ptx.special.tid.x())
    ptx.ret()

tiny()   # triggers trace + emit + driver JIT on first call
```

Between "tiny()" on line 8 and "cuLaunchKernel" a few microseconds
later, the function body traverses five compiler stages. There is no
intermediate optimizer between your Python and the PTX — one call maps
to one instruction by construction. The machinery in this page is
what enforces that.

## The Five Stages

```text
         ┌───────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌────────┐
Python ─▶│ trace │─▶│ optimize │─▶│ assemble │─▶│ emit │─▶│ driver │─▶ SASS
         └───────┘  └──────────┘  └──────────┘  └──────┘  └────────┘
```

| Stage | Input | Output | Source |
| --- | --- | --- | --- |
| **trace** | Python function body | `list[Statement]` | `pyptx/_trace.py` |
| **optimize** | `list[Statement]` | `list[Statement]` | `pyptx/ir/optimize.py` |
| **assemble** | statements + decls | `ir.Module` | `pyptx/kernel.py` |
| **emit** | `ir.Module` | PTX source text | `pyptx/emitter/` |
| **driver JIT** | PTX text | `CUfunction` | `cuModuleLoadData` via `cuda-python` |

And one inverse direction, used by the transpiler and round-trip tests:

```text
PTX text  ──▶  parse  ──▶  ir.Module   (pyptx/parser/)
```

Emitter (stage 4) and parser are inverse functions over the IR. The
entire `tests/corpus/` — 218+ real PTX files — round-trips
byte-identically through `parse → emit`.

The rest of this page walks each stage.

---

## Stage 1: Tracing

The `@kernel` decorator doesn't compile anything at decoration time.
It stores the Python function and the arg specs, returns a callable
wrapper, and waits. On the first call, the wrapper:

1. Binds symbolic shape dims against the actual tensor shapes.
2. Opens a `TraceContext`.
3. Calls the Python function body. Every `ptx.*` and `reg.*` call
   records an IR node into the context.
4. Takes the accumulated IR and hands it to stages 2–5.

The whole tracing machinery lives in `pyptx/_trace.py` — 105 lines.

### TraceContext

```python
class TraceContext:
    def __init__(self, *, ptx_version: tuple[int, int] | None = None) -> None:
        self.reg_decls: list[RegDecl] = []
        self.var_decls: list[VarDecl] = []
        self.statements: list[Statement] = []
        self.ptx_version: tuple[int, int] | None = ptx_version
        self._label_counter = 0
        self._reg_counter: dict[str, int] = {}
        self._if_stack: list[tuple[str, str]] = []
        self.dyn_smem_offset: int = 0
        self.force_dynamic_smem: bool = False
        self._scope_depth: int = 0
```

Three buffers:

- **`reg_decls`** — register declarations (`.reg .u32 %r<16>;`).
  Hoisted to the top of the function body at emit time.
- **`var_decls`** — variable declarations (`.shared .align 128 .b8 ...;`).
  Also hoisted.
- **`statements`** — the instructions, labels, and inner scope blocks
  in emit order.

The split between `reg_decls` and `statements` exists because PTX
requires declarations before any instruction that uses them. Rather
than thread this requirement into every DSL call, the tracer just
parks decls in a separate bucket and concatenates at emit time.

### Thread-local activation

The context is stored in Python's `threading.local`:

```python
_local = threading.local()

@contextmanager
def trace_scope(*, ptx_version=None):
    ctx = TraceContext(ptx_version=ptx_version)
    old = getattr(_local, "ctx", None)
    _local.ctx = ctx
    try:
        yield ctx
    finally:
        _local.ctx = old
```

`get_ctx()` retrieves the current context; if none exists, every
`ptx.*` call raises with a clear "call this inside a `@kernel`"
message.

This is why you can't call `ptx.inst.mov.u32(...)` at module import
time — there's no active trace context, and the call errors
immediately rather than silently producing unused IR.

### How `ptx.inst.mov.u32(...)` becomes an `Instruction` node

Each call like:

```python
ptx.inst.mov.u32(x, ptx.special.tid.x())
```

resolves to a dispatch function in `pyptx/ptx.py` that:

1. Pulls the active `TraceContext` via `get_ctx()`.
2. Builds an `Instruction(opcode="mov", modifiers=(".u32",), operands=(...))`.
3. Calls `ctx.emit(stmt)` to append it to `ctx.statements`.

The `operands` tuple is constructed from the Python arguments —
`Reg` objects become `RegisterOperand`, Python ints become
`ImmediateOperand`, `ptx.addr(...)` calls become `AddressOperand`,
and so on. The mapping is 1:1 — one Python call records one
`Instruction` node.

`reg.scalar(u32)` is slightly different: it records a `RegDecl` via
`ctx.emit_reg_decl(...)` and returns a `Reg` wrapper that knows its
register name. Subsequent uses of that `Reg` reference the same name.

### Scopes: `ptx.scope()` and `ptx.expr()`

Two special blocks modify tracing behavior:

- **`with ptx.scope():`** increments `_scope_depth`. While depth > 0,
  `emit_reg_decl` routes into the statement list instead of the
  hoisted `reg_decls` — so declarations inside the scope stay local
  to the `{ ... }` block. This is how block-local register allocation
  works.
- **`with ptx.expr():`** collects all instructions emitted inside the
  block into a single `CompoundExpr` node (more on this below). Used
  by the transpiler's `--sugar` pass to group temp chains that came
  from one Python expression.

### Control flow primitives

Python-level control flow is traced:

- `ptx.if_(pred)`/`ptx.else_()`/(close of the `with`): emit
  `setp`/`bra`/`label` triples around the body, using
  `fresh_label("If")` / `fresh_label("End")` for the targets.
- `ptx.loop("name", pred=...)`: emit a labeled backward branch.
- `ptx.range_(n)`: emit an unrolled loop. Python-side — the body
  gets traced `n` times, each with its own register state.

Python `for i in range(...):` with a constant `n` is handled at the
**Python level**, not the PTX level — the loop body is traced `n`
times, and the IR has no loop construct. This is how `for g in range(8):`
in an epilogue unrolls to 8 copies of the store sequence.

---

## Stage 2: Optimize

After the trace finishes, the accumulated `statements` run through
one semantics-preserving pass: **copy propagation**. Source is in
`pyptx/ir/optimize.py`, 150 lines.

### Why it exists

The DSL lets you write:

```python
x = reg.scalar(u32)
y = reg.scalar(u32)
x = y   # NOT a Python rebinding — triggers RegArray.__setitem__
```

or more commonly, inside `reg.array`:

```python
acc = reg.array(f32, 32)
acc[5] = some_expression   # emits mov.b32 acc[5], <expr_result>
```

Operator overloading (`+`, `*`, `&`, `>>`) creates a fresh temp
register for the result, and the `__setitem__` on the array emits a
`mov.b32 acc[5], %fresh_temp`. That extra `mov` is wasted — you
could have just written the expression's result directly into
`acc[5]`.

Copy propagation removes it:

1. Scan for `mov.bN %dst, %src` where `%src` is a fresh temp.
2. Verify `%src` is defined once and used only by this mov.
3. Rename `%src` → `%dst` in the definition, delete the mov, delete
   the `.reg` declaration for `%src`.

Result: the PTX is identical to what you'd get from writing
`ptx.inst.*(acc[5], ...)` directly — no extra register, no extra mov.

### The only pass

Copy propagation is currently the **only** post-trace pass. There is no:

- Instruction scheduling (order is fixed by Python evaluation).
- Dead code elimination (you're expected to not emit dead code).
- Constant folding beyond what's visible to Python.
- Register allocation (the DSL allocates, your hand).

The bet is that the user is writing PTX and knows what they want; the
compiler shouldn't second-guess. Copy propagation is the narrow
exception because `RegArray.__setitem__` genuinely emits an
instruction the user didn't ask for.

---

## Stage 3: Assemble The Module

The traced body goes into an `ir.Function`, which goes into an
`ir.Module`. This happens in `pyptx/kernel.py`:

```python
module = Module(
    version=Version(8, 4),
    target=Target(("sm_90a",)),
    address_size=AddressSize(64),
    directives=(
        # ...any smem var decls...
        Function(
            is_entry=True,
            name="tiny",
            params=(...),          # built from in_specs/out_specs
            body=body_statements,
            directives=(
                FunctionDirective("maxntid", (128, 1, 1)),
                # ...other hints...
            ),
        ),
    ),
)
```

The `@kernel` decorator's kwargs (`block`, `grid`, `smem`, `arch`)
become directives attached to the function. Dynamic SMEM > 48 KB
gets a `.extern .shared .align 128 .b8 dyn_smem[];` variable
declaration and flips a bit that the launch shim reads to call
`cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, N)` before
launching.

---

## The IR

Before stage 4, a word on what the IR actually looks like.

### Frozen dataclasses, tuple-valued collections

Every IR node is a `@dataclass(frozen=True)`. Collections inside
nodes are tuples, not lists. This means:

- **Nodes are immutable.** You can't mutate a parsed kernel — you
  rebuild it with `dataclasses.replace(...)`.
- **Nodes are hashable by structure.** `ir1 == ir2` is a deep
  structural comparison; `hash(ir1)` works.
- **The IR is a value type.** Equality is "same shape, same fields,"
  not pointer identity.

The value-type design is load-bearing for round-trip testing: parse
a kernel, emit it, parse the emit, compare the IRs — if they're
structurally equal, the round-trip is lossless. The entire
`tests/corpus/` (218+ real PTX files) is validated this way.

### The node hierarchy

Core statement nodes (things that appear in a function body):

- **`Instruction`** — one PTX instruction.
  `opcode="mov"`, `modifiers=(".b32",)`, `operands=(dst, src)`,
  optional `predicate`.
- **`Label`** — a branch target.
- **`RegDecl`** / **`VarDecl`** — declarations.
- **`PragmaDirective`** — `.pragma "..."`.
- **`Comment`** / **`BlankLine`** — preserved for formatting.
- **`RawLine`** — the escape hatch. When the parser can't
  structurally parse a line (very rare — new ISA features, odd
  formatting), it captures raw text. The emitter emits the text
  verbatim. The parser never crashes.
- **`Block`** — a nested `{ ... }` scope.
- **`IntrinsicScope`** — a DSL-only wrapper around instructions
  emitted by an `@ptx.intrinsic` decorated function. Rendered as
  `BEGIN/END` comments in the emitted PTX so inspection tools can
  see which high-level call produced which instructions.

And operand nodes (things that appear in `Instruction.operands`):

- **`RegisterOperand`** — `%r0`.
- **`ImmediateOperand`** — `42` / `0xFF` / `0d3FF0000000000000`.
  Stored as raw text so float literal precision is preserved exactly.
- **`LabelOperand`** — a label used as a branch target.
- **`VectorOperand`** — `{%r0, %r1, %r2, %r3}` (v4 loads/stores).
- **`AddressOperand`** — `[base]` or `[base+offset]`.
- **`ParenthesizedOperand`** — `(op1, op2, ...)` used in call returns.
- **`NegatedOperand`** — `!%p0` for logical negation.
- **`PipeOperand`** — `%p0|%p1`, the dual predicate output of `setp`.

### FormattingInfo: the round-trip secret

Each statement can carry a `FormattingInfo` with `indent`,
`trailing`, `blank_lines_before`, `preceding_comments`, and `raw_line`.

The parser fills these in when it reads source. The emitter uses
them exactly when present — same indent, same trailing whitespace,
same blank lines before this statement. When absent (trace output),
the emitter uses sensible defaults (4-space indent, one statement
per line).

This is why parser → emitter is byte-identical on the corpus.
Without `FormattingInfo`, round-trip would be semantically correct
but not byte-identical (one-vs-four spaces of indent, reordered
whitespace, etc.). The corpus contains real CUTLASS, Triton,
DeepGEMM, TK output — irregular formatting is the rule, not the
exception.

### CompoundExpr: ptx.expr() groups

```python
@dataclass(frozen=True)
class CompoundExpr:
    instructions: tuple[Instruction, ...]
```

Not in the `Statement` union; handled via duck-typed `instructions`
attribute in the emitter. Represents a group of instructions traced
from a single Python expression like:

```python
with ptx.expr():
    rd[26] = ((r[192] - 8192) & 0x3FF80) >> 4 | CONST
```

The emitted PTX is identical — `CompoundExpr` is cosmetic grouping
only. The transpiler's `--sugar` pass produces these to re-group
long temp chains that came from one high-level expression in the
original source.

---

## Stage 4: Emit

The IR → text path lives in `pyptx/emitter/emitter.py`, 405 lines.
Structurally it's a visitor over `Module` → `Directive` → `Function`
→ `Statement` → `Operand`, with one `_emit_*` function per node
type.

The top-level entry:

```python
def emit(module: Module) -> str:
    if module.raw_source is not None:
        return module.raw_source     # parsed-from-source shortcut
    parts = []
    # ...header + directives...
    return "\n".join(parts) + "\n"
```

The `raw_source` shortcut is a round-trip optimization: if you parsed
a module and didn't modify it, emit just returns the original text.
Only when you construct or modify IR does the emitter actually walk
the tree.

### Per-statement emission

For `Instruction`:

```
    [predicate] opcode[modifiers] operand0, operand1, ...;
```

The emitter concatenates `opcode + modifiers` without spaces
(`mov.b32`), then comma-separates operands, then adds the trailing
`;`. If `FormattingInfo` specifies a leading indent or a trailing
comment, those are reproduced.

For `RegDecl`:

```
    .reg .type name;           # single register
    .reg .type name<count>;    # register array
```

For a `Label`, it's the label name followed by `:` at the
appropriate indent.

For an `IntrinsicScope`, the emitter wraps the inner statements in
`// BEGIN name(args_repr)` and `// END name(args_repr)` comments.
The enclosed instructions are emitted normally — the comments are
for humans and tooling, not PTX semantics.

### Why the emitter is simple

PTX is an assembly language — instruction, modifiers, operands,
terminating semicolon. There's no nesting, no expression grammar,
no type propagation. A visitor with one case per IR node type
covers everything. The ~400 lines of `emitter.py` handle real
CUTLASS and DeepGEMM output with no special cases.

---

## Stage 5: Driver JIT

The emitted PTX text goes to NVIDIA's driver via `cuModuleLoadData`:

```python
# pyptx/jax_support.py
module = cuda.cuModuleLoadData(ptx_bytes)
fn = cuda.cuModuleGetFunction(module, entry_name)
```

The driver JITs PTX → SASS (NVIDIA's real machine code) at load
time. The result is cached by `(ptx_string, arch)` so repeat calls
don't retrigger the JIT.

No `ptxas` required at install. No CUDA toolkit required beyond the
driver. The `cuda-python` package provides the binding; the driver
itself ships with the GPU.

Launch is then `cuLaunchKernel(fn, grid_x, grid_y, grid_z, block_x,
block_y, block_z, smem_bytes, stream, args, ...)`. JAX and PyTorch
route through the tiny C++ shim at `pyptx/_shim/pyptx_launch.cc` so
the call is issued on the correct stream that the framework is
sequencing on.

---

## The Reverse Direction: Parser

`pyptx/parser/` turns PTX text back into IR. Three modules:

- **`tokens.py`** — token types (77 lines).
- **`lexer.py`** — source text → stream of tokens (327 lines).
- **`parser.py`** — tokens → IR (1246 lines, recursive descent).

The parser is **opcode-agnostic**. It doesn't know `mov` vs
`wgmma.mma_async` — it parses the universal structure:

```
    [@predicate] opcode.modifier.modifier operand, operand, operand;
```

and produces an `Instruction` node with the right fields. This is
why new ISA features (Blackwell `tcgen05.*`, future Thor instructions)
parse correctly without any parser changes — they're just another
opcode with modifiers.

When the parser hits something it can't structurally parse (unusual
directive, inline asm with quirky escaping), it captures a `RawLine`
and moves on. The emitter emits `RawLine` verbatim. The kernel is
still valid IR; you just can't structurally modify that particular
line without reparsing it yourself.

### Byte-identical round-trip

The test `tests/test_roundtrip.py` runs:

```python
for path in corpus_files:  # 218+ real PTX files
    text = path.read_text()
    ir = parse(text)
    emitted = emit(ir)
    assert emitted == text   # byte-for-byte
```

This passes for CUTLASS kernels, Triton output, DeepGEMM, fast.cu,
ThunderKittens examples, and the Mamba-SSM kernels. The combination
of `FormattingInfo` preservation + `raw_source` fallback +
`RawLine` escape hatch is what makes it possible.

The transpiler (`pyptx/codegen/`) depends on this round-trip
property: it parses PTX into IR, runs rewriting passes (name
demangling, loop raising, expression grouping), and emits executable
Python. Every PTX kernel the transpiler accepts is one that survives
round-trip.

---

## IntrinsicScope and @ptx.intrinsic

A small DSL surface detail worth knowing. `@ptx.intrinsic` wraps a
function that emits multiple instructions; the trace captures those
instructions into an `IntrinsicScope`:

```python
@ptx.intrinsic
def reduce_sum(reg_in):
    # ...a dozen shfl.bfly.sync + add.f32 instructions...
    pass

# In a kernel:
ptx.warp.reduce_sum(sum_sq)   # emits IntrinsicScope(name="reduce_sum", ...)
```

In the emitted PTX, this shows up as:

```
// BEGIN reduce_sum(%f5)
    shfl.bfly.sync.b32 ...
    add.f32 ...
    ...
// END reduce_sum(%f5)
```

The comments are for humans reading the PTX. The parser sees them as
comments and discards the intrinsic grouping — a round-trip produces
the same instructions, just without the scope wrapper. That's fine
because `IntrinsicScope` is a construction-time concept, not a
semantic one.

---

## Spec Validation

There's a small companion system that isn't part of the compiler
proper but prevents a class of user errors before trace even runs:
`pyptx/spec/`. It holds a declarative description of the PTX ISA —
which modifiers combine, what operand types each opcode takes, how
many destinations vs sources — and validates `ptx.inst.*` calls
against that spec.

When you write:

```python
ptx.inst.mov.u32(x, y, z)   # ← three operands, mov takes two
```

the spec validator catches it at trace time with a message like
"mov.u32 expects 2 operands, got 3," instead of producing broken PTX
that fails at `cuModuleLoadData` with a harder-to-debug error.

The spec is in `pyptx/spec/ptx.py` — 930 lines of data. The
validator (`validate.py`, 660 lines) is called from the `ptx.inst.*`
dispatch.

---

## Why This Design

Five design decisions, in order of how much they matter:

1. **Frozen-dataclass IR with tuple-valued collections.** Makes the
   IR a value type: hashable, comparable by structure, immutable.
   Round-trip testing is `a == b`, not a custom walker. Rewrites use
   `dataclasses.replace` — no mutation accidents.
2. **FormattingInfo on every statement.** What makes byte-identical
   round-trip possible on real-world kernels with idiosyncratic
   formatting. Cheap — it's a pointer-sized field on each node that
   most code ignores.
3. **Opcode-agnostic parser.** New ISA features parse for free. The
   parser doesn't know about `tcgen05.mma` or any future instruction;
   it just parses "opcode.modifiers operands;" and the IR holds the
   strings. The validator (spec) knows the semantics, but the IR
   layer is ISA-blind.
4. **One Python call = one Instruction node.** No lowering, no
   scheduler, no optimizer between trace and emit (except the one
   copy-propagation pass that removes __setitem__ movs). The user
   controls instruction order; the compiler respects it.
5. **Driver JIT, not ptxas.** No CUDA toolkit at install time. PTX
   strings go to `cuModuleLoadData`; the driver produces SASS. Every
   supported CUDA driver can load every PTX the emitter produces.

The whole compiler — tracer + IR + optimizer + emitter + parser —
is under 3000 lines of Python. The IR alone is 350 lines; most of
the line count is in the parser (recursive descent over the full
PTX grammar) and the ISA spec (data tables). There is no code
generator in the conventional sense; emit is a visitor that
stringifies already-complete instructions.

## What To Read Next

- [PTX Namespace](guides/ptx-namespace.md) — reference for every
  DSL call that appears in stage 1 (trace).
- [Transpiler](transpiler.md) — the parser + emitter combined into
  a PTX → Python converter.
- [Philosophy](philosophy.md) — the "why" of "one call = one
  instruction," restated at a higher level.
- `pyptx/_trace.py`, `pyptx/ir/nodes.py`, `pyptx/emitter/emitter.py`
  — the source is ~900 lines total and readable end-to-end.
