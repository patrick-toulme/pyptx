"""Compare structural markers between the current Hopper flash kernel and a
transpiled CUTLASS Hopper FMHA reference.

This is not a correctness or performance benchmark. It is a cheap shape check
for the development loop while porting an FA3-style kernel into handwritten
pyptx.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "examples" / "hopper" / "experimental" / "flash_attention_hopper.py"
REFERENCE = ROOT / "benchmarks" / "cutlass_hopper_fmha_reference.py"


PATTERNS = (
    "setmaxnreg",
    "elect.sync",
    "mbarrier.init",
    "mbarrier.try_wait",
    "mbarrier.arrive.expect_tx",
    "cp.async_.bulk.tensor._3d",
    "cp.async_.bulk.tensor._4d",
    "cp.async_.bulk.tensor_2d",
    "wgmma.mma_async",
    "PersistentTileScheduler",
)


def count_patterns(text: str) -> dict[str, int]:
    return {pattern: text.count(pattern) for pattern in PATTERNS}


def main() -> None:
    current_text = CURRENT.read_text()
    ref_text = REFERENCE.read_text()
    current = count_patterns(current_text)
    reference = count_patterns(ref_text)

    print("pattern".ljust(30), "current".rjust(8), "reference".rjust(10), "delta".rjust(8))
    for pattern in PATTERNS:
        cur = current[pattern]
        ref = reference[pattern]
        print(pattern.ljust(30), str(cur).rjust(8), str(ref).rjust(10), str(cur - ref).rjust(8))


if __name__ == "__main__":
    main()
