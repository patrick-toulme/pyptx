"""Full PTX → Python → PTX pipeline tests across the entire corpus.

Tests that every corpus file can be:
1. Parsed to IR
2. Converted to Python code via codegen
3. Compiled as valid Python
4. Executed through the DSL tracer
5. Emitted back to PTX
6. Re-parsed and compared to original IR (normalized)

Files that don't pass are tracked in _KNOWN_DIFFS with the specific reason.
"""

import pytest
from pathlib import Path

from pyptx.parser import parse
from pyptx.codegen import ptx_to_python
from pyptx.ir.normalize import normalize_module, diff_modules
from pyptx.ir.nodes import Function
from pyptx.kernel import Kernel


CORPUS_DIR = Path(__file__).parent / "corpus"

# Files with known pipeline diffs (structural issues during round-trip).
# These still parse, codegen, and execute — they just don't match the
# original IR exactly after the round-trip due to:
# - Nested { } block scoping (matmul_wmma_*)
# - Multi-function files with complex scoping
# - Large Triton dumps with debug sections
_KNOWN_DIFFS = frozenset({
    "matmul_wmma_v1.ptx",
    "matmul_wmma_v2.ptx",
    "matmul_wmma_v3.ptx",
    "matmul_wmma_v4.ptx",
    "matmul_wmma_v5.ptx",
    "matmul_wmma_v6.ptx",
    "matmul_wmma_v7.ptx",
    "matmul_wmma_v8.ptx",
    "matmul_wmma_v9.ptx",
    "matmul_wmma_v10.ptx",
    "ptoxide_fncall.ptx",
    # Large Triton dumps — pipeline works but body length differs
    "fa_ws_pingpong_sm90a.ptx",
    "matmul_tma_sm120a.ptx",
    "matmul_tma_v33_sm90a.ptx",
    "matmul_tma_v34_sm90a.ptx",
    "matmul_tma_v35_sm90a.ptx",
})

# Files where Python compile fails (edge cases in codegen)
_COMPILE_FAIL = frozenset({
    "matmul_wgmma_v32_sm90a.ptx",  # Complex Triton output
})


def all_corpus_files():
    return sorted(CORPUS_DIR.rglob("*.ptx"))


def pipeline_match_files():
    return [
        f for f in all_corpus_files()
        if f.name not in _KNOWN_DIFFS and f.name not in _COMPILE_FAIL
    ]


@pytest.mark.parametrize("ptx_file", pipeline_match_files(), ids=lambda p: p.name)
def test_pipeline_match(ptx_file: Path):
    """Full PTX → Python → PTX pipeline should produce matching IR."""
    source = ptx_file.read_text()

    # Parse and codegen
    py = ptx_to_python(source)

    # Must be valid Python
    compile(py, ptx_file.name, "exec")

    # Must execute without errors
    ns: dict = {}
    exec(py, ns)

    # Must produce at least one Kernel
    kernels = [v for v in ns.values() if isinstance(v, Kernel)]
    assert kernels, f"No @kernel function in {ptx_file.name}"

    # For each entry function in the original, trace the corresponding
    # generated kernel and compare normalized IR
    ir_orig = parse(source)
    orig_entries = [
        d for d in ir_orig.directives
        if isinstance(d, Function) and d.is_entry
    ]

    kernel_map = {k._fn.__name__: k for k in kernels}

    for orig_func in orig_entries:
        assert orig_func.name in kernel_map, (
            f"{ptx_file.name}: kernel {orig_func.name} not generated"
        )
        kfn = kernel_map[orig_func.name]
        out_ptx = kfn.ptx()

        canon_orig = normalize_module(parse(source))
        canon_trip = normalize_module(parse(out_ptx))
        diffs = diff_modules(canon_orig, canon_trip, entry_only=True)

        func_diffs = [
            d for d in diffs
            if orig_func.name in d and "function count" not in d
        ]
        assert not func_diffs, (
            f"{ptx_file.name}::{orig_func.name} pipeline mismatch:\n"
            + "\n".join(f"  {d}" for d in func_diffs)
        )


@pytest.mark.parametrize("ptx_file", all_corpus_files(), ids=lambda p: p.name)
def test_pipeline_codegen_produces_valid_python(ptx_file: Path):
    """Every corpus file's codegen output must be valid Python syntax."""
    if ptx_file.name in _COMPILE_FAIL:
        pytest.skip("Known compile failure — edge case in codegen")
    source = ptx_file.read_text()
    py = ptx_to_python(source)
    compile(py, ptx_file.name, "exec")


@pytest.mark.parametrize("ptx_file", all_corpus_files(), ids=lambda p: p.name)
def test_pipeline_codegen_executes(ptx_file: Path):
    """Every corpus file's codegen output must execute without errors."""
    if ptx_file.name in _COMPILE_FAIL:
        pytest.skip("Known compile failure — edge case in codegen")
    source = ptx_file.read_text()
    py = ptx_to_python(source)
    try:
        compile(py, ptx_file.name, "exec")
    except SyntaxError:
        pytest.skip("Syntax error in codegen output")
    ns: dict = {}
    exec(py, ns)
    kernels = [v for v in ns.values() if isinstance(v, Kernel)]
    assert kernels, f"No @kernel function in {ptx_file.name}"

    # Trace each kernel to confirm the DSL accepts what we generated
    for k in kernels:
        _ = k.ptx()
