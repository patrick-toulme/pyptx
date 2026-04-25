"""Shared test fixtures."""

from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"


def corpus_files() -> list[Path]:
    """Return all .ptx files in the test corpus, sorted by name."""
    return sorted(CORPUS_DIR.glob("*.ptx"))
