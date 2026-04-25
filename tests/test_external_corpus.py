"""Tests for external PTX corpus files (real-world PTX from public repos).

Sources:
- ashvardanian/less_slow.cpp (Apache-2.0): hand-written Hopper PTX with wgmma
- unixpickle/learn-ptx: 44 hand-written PTX files (wmma, cp.async, etc.)
- gvilums/ptoxide (Apache-2.0/MIT): compiler-generated PTX
- llvm/ (Apache-2.0 with LLVM Exception): synthesized from LLVM NVPTX test
  CHECK lines — covers wgmma, TMA, mbarrier, tcgen05, cluster, etc.
"""

import pytest
from pathlib import Path

from pyptx.parser import parse
from pyptx.emitter import emit


EXTERNAL_DIR = Path(__file__).parent / "corpus" / "external"


def external_files() -> list[Path]:
    if not EXTERNAL_DIR.exists():
        return []
    # Recursively include subdirectories (e.g. llvm/)
    return sorted(EXTERNAL_DIR.rglob("*.ptx"))


@pytest.mark.parametrize("ptx_file", external_files(), ids=lambda p: p.name)
def test_parse(ptx_file):
    """External PTX files must parse without errors."""
    source = ptx_file.read_text()
    module = parse(source)
    assert len(module.directives) > 0


@pytest.mark.parametrize("ptx_file", external_files(), ids=lambda p: p.name)
def test_roundtrip(ptx_file):
    """External PTX files should round-trip: parse → emit → identical."""
    source = ptx_file.read_text()
    module = parse(source)
    output = emit(module)
    assert output == source, f"Round-trip mismatch for {ptx_file.name}"
