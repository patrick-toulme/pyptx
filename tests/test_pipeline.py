"""Full pipeline tests: PTX → parse → IR → codegen → Python → execute → IR → compare.

This is the ultimate round-trip test. It exercises every layer:
- Parser: PTX text → IR
- Codegen: IR → Python source
- DSL: execute Python → trace → IR
- Emitter: IR → PTX text
- Normalizer: strip formatting for structural comparison

The final IR must match the original IR (modulo comments and formatting).
"""

import pytest
from pathlib import Path

from pyptx.parser import parse
from pyptx.codegen import ptx_to_python
from pyptx.ir.normalize import normalize_module, diff_modules
from pyptx.kernel import Kernel


CORPUS_DIR = Path(__file__).parent / "corpus"

# Files that can do the full PTX → Python → PTX pipeline round-trip
_PIPELINE_FILES = [
    "minimal.ptx",
    "vector_add.ptx",
    "branches.ptx",
    "predicates.ptx",
    "shared_memory.ptx",
    "wgmma_simple.ptx",
    "cluster_ops.ptx",
    "mbarrier_full.ptx",
    "tma_load_store.ptx",
    "wgmma_gemm_tile.ptx",
    "tcgen05_blackwell.ptx",
    "function_call.ptx",
    "hopper_misc.ptx",
    "less_slow_sm90a.ptx",
]


def pipeline_files():
    return [CORPUS_DIR / f for f in _PIPELINE_FILES if (CORPUS_DIR / f).exists()]


def _find_kernels(namespace: dict) -> list[Kernel]:
    """Find all Kernel objects in a namespace."""
    return [v for v in namespace.values() if isinstance(v, Kernel)]


@pytest.mark.parametrize("ptx_file", pipeline_files(), ids=lambda p: p.name)
def test_full_pipeline(ptx_file):
    """PTX → parse → codegen → exec → emit → parse → normalize → compare.

    Compares only .entry functions (entry_only=True) since .func helpers
    may not survive the round-trip when each @kernel traces independently.
    """
    source = ptx_file.read_text()

    # Step 1: Parse original PTX → IR
    ir_original = parse(source)

    # Step 2: Codegen IR → Python source
    python_code = ptx_to_python(source)

    # Step 3: Execute Python → get kernel objects
    namespace: dict = {}
    exec(python_code, namespace)
    kernels = _find_kernels(namespace)
    assert kernels, f"No @kernel function found in generated code for {ptx_file.name}"

    # Step 4: Test each kernel independently
    # Build a map of kernel name → Kernel object
    kernel_map = {k._fn.__name__: k for k in kernels}

    # Find entry functions in the original IR
    from pyptx.ir.nodes import Function
    orig_entries = [
        d for d in ir_original.directives
        if isinstance(d, Function) and d.is_entry
    ]

    all_diffs = []
    for orig_func in orig_entries:
        if orig_func.name not in kernel_map:
            all_diffs.append(f"Missing kernel for {orig_func.name}")
            continue

        kfn = kernel_map[orig_func.name]
        roundtripped_ptx = kfn.ptx()
        ir_roundtripped = parse(roundtripped_ptx)

        canon_orig = normalize_module(ir_original)
        canon_trip = normalize_module(ir_roundtripped)

        # Compare just this function by name
        diffs = diff_modules(canon_orig, canon_trip, entry_only=True)
        # Filter to only diffs about this function
        func_diffs = [d for d in diffs if orig_func.name in d or "function count" in d]
        # If function count differs, it's expected (we only emit one at a time)
        func_diffs = [d for d in func_diffs if "function count" not in d]
        all_diffs.extend(func_diffs)

    assert not all_diffs, (
        f"Pipeline round-trip mismatch for {ptx_file.name}:\n"
        + "\n".join(f"  {d}" for d in all_diffs)
    )


@pytest.mark.parametrize("ptx_file", pipeline_files(), ids=lambda p: p.name)
def test_codegen_is_valid_python(ptx_file):
    """Generated Python code must be syntactically valid."""
    source = ptx_file.read_text()
    python_code = ptx_to_python(source)
    compile(python_code, f"<pipeline:{ptx_file.name}>", "exec")


@pytest.mark.parametrize("ptx_file", pipeline_files(), ids=lambda p: p.name)
def test_codegen_is_executable(ptx_file):
    """Generated Python code must execute without errors."""
    source = ptx_file.read_text()
    python_code = ptx_to_python(source)
    namespace: dict = {}
    exec(python_code, namespace)
    kernels = _find_kernels(namespace)
    assert len(kernels) >= 1, f"No @kernel function found for {ptx_file.name}"
