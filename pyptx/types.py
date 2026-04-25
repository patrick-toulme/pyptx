"""PTX scalar type descriptors.

The public API of this module is the set of singleton :class:`PtxType`
instances such as ``u32``, ``bf16``, ``f32``, and ``pred``.

These type objects are used throughout the DSL:

```python
from pyptx.types import bf16, f32, u32, pred

acc = reg.array(f32, 64)
sA = smem.alloc(bf16, (STAGES, BM, BK))
tid = reg.from_(ptx.special.tid.x(), u32)
p = reg.scalar(pred)
```

The type singletons are intentionally lightweight. They mostly serve as
an explicit bridge between Python code and PTX type spelling.
"""

from __future__ import annotations


class PtxType:
    """A PTX scalar type.

    Singleton instances (bf16, f32, etc.) are the public API.
    """

    __slots__ = ("name", "bits")

    def __init__(self, name: str, bits: int) -> None:
        self.name = name
        self.bits = bits

    @property
    def ptx(self) -> str:
        """PTX text form with leading dot: '.f32'."""
        return f".{self.name}"

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PtxType):
            return self.name == other.name
        return NotImplemented


# -- Bit types ---------------------------------------------------------------
b8 = PtxType("b8", 8)
b16 = PtxType("b16", 16)
b32 = PtxType("b32", 32)
b64 = PtxType("b64", 64)
b128 = PtxType("b128", 128)

# -- Unsigned integers -------------------------------------------------------
u8 = PtxType("u8", 8)
u16 = PtxType("u16", 16)
u32 = PtxType("u32", 32)
u64 = PtxType("u64", 64)

# -- Signed integers ---------------------------------------------------------
s8 = PtxType("s8", 8)
s16 = PtxType("s16", 16)
s32 = PtxType("s32", 32)
s64 = PtxType("s64", 64)

# -- Floating point ----------------------------------------------------------
f16 = PtxType("f16", 16)
f16x2 = PtxType("f16x2", 32)
bf16 = PtxType("bf16", 16)
bf16x2 = PtxType("bf16x2", 32)
tf32 = PtxType("tf32", 32)
f32 = PtxType("f32", 32)
f64 = PtxType("f64", 64)

# -- Alternate FP formats (Hopper/Blackwell) ---------------------------------
e4m3 = PtxType("e4m3", 8)
e5m2 = PtxType("e5m2", 8)

# -- Predicate --------------------------------------------------------------
pred = PtxType("pred", 1)

# -- Lookup by name ----------------------------------------------------------
_BY_NAME: dict[str, PtxType] = {t.name: t for t in [
    b8, b16, b32, b64, b128,
    u8, u16, u32, u64,
    s8, s16, s32, s64,
    f16, f16x2, bf16, bf16x2, tf32, f32, f64,
    e4m3, e5m2,
    pred,
]}


def from_name(name: str) -> PtxType:
    """Look up a PtxType by name (with or without leading dot)."""
    raw = name.lstrip(".")
    t = _BY_NAME.get(raw)
    if t is None:
        raise ValueError(f"Unknown PTX type: {name!r}")
    return t
