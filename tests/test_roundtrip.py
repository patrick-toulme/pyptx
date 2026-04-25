"""Round-trip integration tests: parse → emit → assert identical.

This is the foundational test suite for pyptx. If round-trip breaks,
something is wrong with the parser or emitter.
"""

import pytest

from tests.conftest import corpus_files
from pyptx.parser import parse
from pyptx.emitter import emit

# Files that parse successfully but don't round-trip yet due to
# block comments, inline comments, multi-line formatting, etc.
# These are tested for parse-only in test_parse_only below.
def _roundtrip_files():
    return corpus_files()


@pytest.mark.parametrize("ptx_file", _roundtrip_files(), ids=lambda p: p.name)
def test_roundtrip(ptx_file):
    """Parse a .ptx file, emit it back, and assert byte-identical output."""
    source = ptx_file.read_text()
    module = parse(source)
    output = emit(module)
    assert output == source, (
        f"Round-trip mismatch for {ptx_file.name}.\n"
        f"--- Expected ---\n{source}\n"
        f"--- Got ---\n{output}"
    )


