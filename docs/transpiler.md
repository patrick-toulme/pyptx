# Transpiler

`pyptx` is a real PTX-to-Python transpiler. Feed it compiled PTX from
`nvcc`, Triton, Pallas, or any other source; get back Python that uses
the same `@kernel` / `reg` / `smem` / `ptx` namespaces you would write
by hand, and **round-trips byte-identical**.

That matters for three workflows:

- port an existing PTX kernel into pyptx
- study how a production kernel maps onto the DSL
- iterate on a kernel with a safe PTX baseline you can diff against

The maintained **815 TFLOPS** Hopper GEMM in `examples/hopper/gemm_highperf_hopper.py`
is exactly this workflow, applied to [fast.cu's kernel12](https://github.com/pranjalssh/fast.cu):
compile → extract PTX → transpile with `--sugar` → hand-clean → ship.

## The pipeline

```text
           ┌───────┐  ┌─────────┐  ┌───────┐  ┌──────┐
PTX text ─▶│ parse │─▶│ codegen │─▶│ trace │─▶│ emit │─▶ PTX text
           └───────┘  └─────────┘  └───────┘  └──────┘
```

| Stage | Input | Output | Source |
| --- | --- | --- | --- |
| **parse** | PTX text | `ir.Module` | `pyptx/parser/` (opcode-agnostic recursive descent) |
| **sugar** *(optional)* | `ir.Module` | rewritten `ir.Module` | `pyptx/codegen/sugar.py` — name demangling, loop raising, mbarrier spin-loop collapse, expression grouping |
| **codegen** | `ir.Module` | Python (pyptx DSL) | `pyptx/codegen/codegen.py` |
| **trace** | Python | `ir.Module` | `pyptx/_trace.py` (see [How It Works](how-it-works.md)) |
| **emit** | `ir.Module` | PTX text | `pyptx/emitter/` |

The round-trip invariant: **parse → emit gives back the input bytes
exactly**, verified on 218+ real kernels in `tests/corpus/`.

## Quick start

```bash
python -m pyptx.codegen kernel.ptx --sugar --name my_kernel > my_kernel.py
python my_kernel.py   # runs
```

Or from Python:

```python
from pathlib import Path
from pyptx.codegen import ptx_to_python

source = Path("kernel.ptx").read_text()
python_source = ptx_to_python(source, sugar=True, kernel_name="my_kernel")
print(python_source)
```

## A real before/after

Here's a fragment from fast.cu's kernel12, the WGMMA K-loop body. This
is what comes out of `nvcc -arch=sm_90a -ptx fast.cu` (one of four
sub-k WGMMA calls):

=== "Raw PTX (as compiled)"

    ```ptx
    $L__BB13_2:
        mov.u64 %rd283, 0;
        shl.b32 %r421, %r420, 13;
        add.s32 %r422, %r421, %r419;
        shl.b32 %r423, %r422, 1;
        mov.b64 %rd284, %_ZN3M124smemE;
        add.s64 %rd285, %rd284, %r423;
        add.s64 %rd286, %rd285, -8192;
        and.b64 %rd287, %rd286, 262016;
        shr.u64 %rd288, %rd287, 4;
        or.b64 %rd289, %rd288, 4611686293305344000;
        wgmma.fence.sync.aligned;
        wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16
            {%f0, %f1, %f2, ..., %f127},
            %rd289, %rd300, %p21, 1, 1, 0, 0;
        ...
    ```

=== "After `--sugar`"

    ```python
    with ptx.loop("consumer_pair_loop"):
        stage, phase = consumer_pipe.advance()
        full.at(stage).wait(phase)

        ptx.inst.wgmma.fence.sync.aligned()
        a_base = smem_base + (((stage << 13) + lane_mask) << 1)
        a_desc = ptx.wgmma.masked_descriptor(
            a_base, byte_offset=-8192, mask=262016,
        )
        for sub_k in range(4):
            b_desc = ptx.wgmma.masked_descriptor(
                b_base,
                byte_offset=b_offsets[sub_k],
                mask=b_masks[sub_k],
            )
            ptx.inst.wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16(
                acc_frag, a_desc, b_desc,
                0 if (first_scale_zero and sub_k == 0) else 1,
                1, 1, 0, 0,
            )
        ptx.inst.wgmma.commit_group.sync.aligned()
        ptx.inst.wgmma.wait_group.sync.aligned(0)
    ```

The sugar pass did real work here:

- `_ZN3M124smemE` → `smem` (demangled).
- The spin-loop over `mbarrier.try_wait` (not shown) collapsed into
  `full.at(stage).wait(phase)`.
- The four near-identical WGMMA blocks collapsed into a `range(4)`
  loop with a `b_offsets` table.
- The address arithmetic for the A descriptor stayed inline — because
  reordering it past the WGMMA fence would be a correctness bug, and
  the sugar pass refuses to move instructions across barriers.

The result is **byte-identical PTX** when re-emitted. The
transpiled-and-sugared kernel is 665 lines of pyptx Python (down from
~1070 raw). Same PTX, same 815 TFLOPS.

## What sugar does

The rewrite passes, in order:

| Pass | Effect |
| --- | --- |
| Name demangling | `_ZN3M124smemE` → `smem`; mangled params → `M, N, K, tma_A` |
| Label shortening | `$L__BB13_2` → `BB2` |
| Loop raising | `label + ... + bra(label)` → `with ptx.loop(label, pred=p):` |
| Mbarrier collapse | spin-loop block → `ptx.mbarrier.wait(addr, phase)` |
| WGMMA compact | 128 register operands → `[f[i] for i in range(lo, hi)]` |
| Parameterized loops | repeated instruction blocks → `for _i in range(N)` with `base + _i*stride` |
| Expression grouping | temp chains → `with ptx.expr():` blocks with `# rd[26] = ...` comments |

None of the passes reorder instructions across barriers, WGMMA fences,
or mbarrier arrives. PTX instruction order matters for memory
consistency on Hopper, and the transpiler refuses to touch it.

## Parse / emit without codegen

If you want to work one layer below the Python DSL:

```python
from pathlib import Path
from pyptx.parser import parse
from pyptx.emitter import emit

source = Path("kernel.ptx").read_text()
module = parse(source)
assert emit(module) == source   # round-trip invariant
```

This is the path for tooling, normalization passes, or structural
tests that don't need to generate Python.

## Round-trip testing

The transpiler is backed by real round-trip tests:

- `tests/test_codegen.py` — codegen → trace → emit
- `tests/test_pipeline.py` — single-file pipeline
- `tests/test_pipeline_full.py` — multi-file pipeline
- `tests/test_external_corpus.py` — 218+ corpus files

All of these assert byte-identical or structural equivalence across
the round-trip. If you break one of them, the PR doesn't land.

## The usual workflow

```bash
# 1. Get the PTX out of whatever produced it.
cuobjdump -ptx kernel.cubin > kernel.ptx
# or: nvcc -arch=sm_90a -ptx kernel.cu

# 2. Transpile to pyptx.
python -m pyptx.codegen kernel.ptx --sugar --name my_kernel > my_kernel.py

# 3. Run it. It should work out of the box.
python my_kernel.py

# 4. Hand-clean the pieces you want to own. Keep diffing emitted PTX.
diff <(python -c "from my_kernel import k; print(k.ptx())") kernel.ptx
```

Step 4 is where the value is. The transpile gives you a correct,
editable, emitting-identical-PTX baseline; refactoring from there is
much safer than starting from C++ templates or PTX text.
