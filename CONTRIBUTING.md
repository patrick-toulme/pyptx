# Contributing to pyptx

Thanks for wanting to help. pyptx is early; the structure is deliberately
narrow and the bar for new surface area is high. This file covers the
mechanics.

## Dev setup

```bash
git clone https://github.com/patrick-toulme/pyptx
cd pyptx
pip install -e ".[all,dev,docs]"
```

On Debian/Ubuntu with a PEP 668 system Python, add `--break-system-packages`
or use a venv.

Build the C++ launch shim once (needed for JAX/PyTorch execution paths):

```bash
cd pyptx/_shim && bash build.sh
```

Install `ninja` so the PyTorch C++ extension JIT-compiles on first launch
(drops dispatch overhead from ~34 µs to ~14 µs):

```bash
pip install ninja
```

## Running the tests

```bash
# Full suite (excludes external PTX corpus if not checked out).
pytest tests/ -q

# CPU-only subset (matches CI).
pytest tests/ --ignore=tests/test_external_corpus.py --ignore=tests/test_gpu_execution.py -q

# A single kernel's round-trip test.
pytest tests/test_roundtrip.py -q -k rms_norm
```

The GPU tests (`tests/test_gpu_execution.py`, `tests/test_torch_dispatch.py`,
`tests/test_jax_integration.py`) only run if an H100-class GPU is visible.

## The invariant that must not break

**Round-trip round-tripping the PTX corpus must stay byte-identical.**

`tests/corpus/` holds 218+ real PTX files from CUTLASS, Triton, fast.cu,
DeepGEMM, ThunderKittens, and the LLVM test suite. Every PR runs
`tests/test_roundtrip.py`, which:

1. Parses each file into the pyptx IR
2. Re-emits PTX from the IR
3. Asserts the output equals the input byte-for-byte

If you touch the parser, the IR, the emitter, or any instruction spec,
this is the test that catches silent regressions. **Don't skip it.**

If you need to add a corpus file, place it in `tests/corpus/` and add a
one-line comment at the top citing its source. Files that fail round-trip
get filed as parser bugs, not as corpus exceptions.

## How to add a kernel

1. Put the file in `examples/hopper/` (or the appropriate `sm_XX` subdir).
2. Start the module docstring with **one headline sentence** that
   includes the measured perf number on H100 — this becomes the hero
   line on the docs site.
3. Follow the existing pattern in `examples/hopper/rms_norm.py`:
   - `build_<name>(...)` factory that returns a `@kernel`-decorated callable
   - `<name>_ref(...)` reference implementation for correctness checking
   - `_run_jax_case` and `_run_torch_case` test functions
   - `main()` that iterates sizes and prints `OK`/`FAIL`
4. Add a benchmark entry in `benchmarks/bench_final.py` with a realistic
   shape.
5. Regenerate the example docs:
   ```bash
   PYTHONPATH=. python3 docs/scripts/generate_docs.py
   ```

## How to extend the DSL

The DSL has four lanes:

- `pyptx/ptx.py` — instruction wrappers + structured control flow
- `pyptx/reg.py` — register allocation + `Reg` operator overloads
- `pyptx/smem.py` — shared-memory allocs, swizzles, mbarriers
- `pyptx/spec/` — declarative ISA spec + validation

The **contract is: one DSL call should map to exactly one PTX
instruction** (sugar helpers that expand to a well-known canonical
sequence are OK — e.g. `ptx.warp.reduce_sum` expands to the standard
`shfl.sync.bfly` butterfly — but these should be few, named clearly,
and documented).

Before adding a helper, check whether the existing primitives can
express the pattern inline. If they can, prefer that.

## Style

- No ruff/black enforced auto-format — we follow the surrounding code.
- Type hints on public functions.
- Docstrings: Google style. The first paragraph becomes the docs summary.
- Line width ≤ 100.
- No emojis in code, no marketing prose in docstrings.

## Submitting

1. Fork → feature branch → PR against `main`.
2. CI must pass on all four Python versions (3.10–3.13).
3. If your change affects perf, include a before/after from
   `benchmarks/bench_final.py` or `benchmarks/bench_hopper_gemm.py` in
   the PR description.
4. Squash-merge is fine; please keep the final commit message
   explaining *why*, not just *what*.
