"""Tensor boundary specifications for :func:`pyptx.kernel`.

The :class:`Tile` and :class:`Layout` APIs describe the contract between
the Python runtime world and the traced PTX kernel:

- logical tensor shape
- dtype
- layout
- optional TMA box shape metadata

These specs are used by runtime integrations to:

- resolve symbolic dimensions such as ``"M"`` or ``"K"``
- validate shapes and dtypes at call time
- synthesize TMA descriptors when required

Example:

```python
from pyptx import Tile, Layout, kernel
from pyptx.types import bf16, f32

@kernel(
    in_specs=(
        Tile("M", "K", bf16, Layout.ROW),
        Tile("K", "N", bf16, Layout.COL),
    ),
    out_specs=(Tile("M", "N", f32, Layout.ROW),),
    grid=lambda M, N, K: (M // 128, N // 256),
    block=(128, 1, 1),
    arch="sm_90a",
)
def gemm(A, B, C): ...
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from pyptx.types import PtxType


class Layout(Enum):
    """Memory layout for a tile.

    ROW   — row-major (C order)
    COL   — column-major (Fortran order)
    TMA_128B — TMA 128-byte swizzle (Hopper)
    TMA_64B  — TMA 64-byte swizzle
    TMA_32B  — TMA 32-byte swizzle
    INTERLEAVED — CUTLASS interleaved layout
    """

    ROW = "row"
    COL = "col"
    TMA_128B = "tma_128b"
    TMA_64B = "tma_64b"
    TMA_32B = "tma_32b"
    INTERLEAVED = "interleaved"


# A dimension can be either an integer (concrete) or a string (symbolic).
Dim = Union[int, str]


@dataclass(frozen=True)
class Tile:
    """A tensor spec: shape + dtype + layout.

    Shape dimensions can be integers (concrete) or strings (symbolic variables
    that get bound to real integers at call time). For example:

        Tile("M", "K", bf16, Layout.ROW)

    describes an MxK bfloat16 row-major tile where M and K are unknown at
    decoration time but will be bound when the kernel is called with a
    concrete JAX array.

    ``tma_box`` is the per-TMA-load box shape when this tile is consumed via
    ``cp.async.bulk.tensor``. Defaults to the full tensor (one load reads
    everything). K-loop kernels that use ``Tile.wgmma_a(..., tile_k=...)``
    set this implicitly so each TMA load brings in exactly one K slice.

    ``tma_rank`` selects the descriptor encoding used by the runtime when the
    kernel body calls ``tensor.tma_desc()``:

    - ``2``: normal rank-2 descriptor
    - ``3``: Hopper-style reshaped rank-3 descriptor used by the high-perf
      handwritten GEMM examples

    ``tma_padding`` only matters for ``tma_rank=3`` and requests the padded
    innermost store box used by the Hopper GEMM epilogue.
    """

    # Store shape as a tuple of Dim. Constructed via __init__ that accepts
    # variable positional args for ergonomics: Tile("M", "K", bf16, Layout.ROW)
    shape: tuple[Dim, ...]
    dtype: PtxType
    layout: Layout = Layout.ROW
    tma_box: tuple[Dim, ...] | None = None
    tma_rank: int = 2
    tma_padding: bool = False

    def __init__(
        self,
        *dims_and_dtype: Dim | PtxType | Layout,
        dtype: PtxType | None = None,
        layout: Layout = Layout.ROW,
        tma_box: tuple[Dim, ...] | None = None,
        tma_rank: int = 2,
        tma_padding: bool = False,
    ) -> None:
        """Accept Tile("M", "K", bf16, Layout.ROW) or Tile("M", "K", dtype=bf16).

        Positional args are: dim1, dim2, ..., dimN, [dtype], [layout]
        where dtype and layout can also be passed as keyword args.
        """
        parsed_shape: list[Dim] = []
        parsed_dtype = dtype
        parsed_layout = layout

        for arg in dims_and_dtype:
            if isinstance(arg, PtxType):
                if parsed_dtype is not None and parsed_dtype is not arg:
                    raise ValueError(
                        f"Tile got two dtypes: {parsed_dtype} and {arg}"
                    )
                parsed_dtype = arg
            elif isinstance(arg, Layout):
                parsed_layout = arg
            elif isinstance(arg, (int, str)):
                parsed_shape.append(arg)
            else:
                raise TypeError(
                    f"Tile arg must be int, str, PtxType, or Layout; got "
                    f"{type(arg).__name__}: {arg!r}"
                )

        if parsed_dtype is None:
            raise ValueError("Tile requires a dtype (e.g. bf16, f32)")
        if not parsed_shape:
            raise ValueError("Tile requires at least one dimension")

        # __init__ on a frozen dataclass needs object.__setattr__
        object.__setattr__(self, "shape", tuple(parsed_shape))
        object.__setattr__(self, "dtype", parsed_dtype)
        object.__setattr__(self, "layout", parsed_layout)
        object.__setattr__(self, "tma_box", tma_box)
        object.__setattr__(self, "tma_rank", tma_rank)
        object.__setattr__(self, "tma_padding", tma_padding)

    @property
    def rank(self) -> int:
        """Number of dimensions in the tile shape."""
        return len(self.shape)

    @property
    def symbolic_dims(self) -> tuple[str, ...]:
        """Return the symbolic dim names in this tile."""
        return tuple(d for d in self.shape if isinstance(d, str))

    def resolve_shape(self, env: dict[str, int]) -> tuple[int, ...]:
        """Resolve symbolic dims using a name -> int environment."""
        resolved: list[int] = []
        for d in self.shape:
            if isinstance(d, int):
                resolved.append(d)
            else:
                if d not in env:
                    raise KeyError(
                        f"Tile shape variable {d!r} not bound in env. "
                        f"Available: {sorted(env)}"
                    )
                resolved.append(env[d])
        return tuple(resolved)

    def matches(self, array_shape: tuple[int, ...], array_dtype_name: str) -> bool:
        """Check if a concrete JAX array matches this tile's structure.

        Does NOT check symbolic dims — use extract_env for that.
        """
        if len(array_shape) != self.rank:
            return False
        if array_dtype_name != self.dtype.name and array_dtype_name != self._numpy_dtype_name():
            return False
        for spec_dim, arr_dim in zip(self.shape, array_shape):
            if isinstance(spec_dim, int) and spec_dim != arr_dim:
                return False
        return True

    def extract_env(self, array_shape: tuple[int, ...]) -> dict[str, int]:
        """Extract the {symbolic_dim: concrete_int} env from an array shape."""
        env: dict[str, int] = {}
        if len(array_shape) != self.rank:
            raise ValueError(
                f"Tile rank mismatch: tile has {self.rank} dims, array has "
                f"{len(array_shape)} dims"
            )
        for spec_dim, arr_dim in zip(self.shape, array_shape):
            if isinstance(spec_dim, str):
                env[spec_dim] = arr_dim
            elif spec_dim != arr_dim:
                raise ValueError(
                    f"Tile dim mismatch: expected {spec_dim}, got {arr_dim}"
                )
        return env

    # ------------------------------------------------------------------
    # wgmma operand shortcuts
    # ------------------------------------------------------------------

    @classmethod
    def wgmma_a(
        cls,
        m: "Dim",
        k: "Dim",
        dtype: PtxType,
        *,
        tile_m: "Dim | None" = None,
        tile_k: "Dim | None" = None,
    ) -> "Tile":
        """Tile for a wgmma A operand (row-major MxK, K-major).

        Picks the matching ``Layout.TMA_*B`` automatically based on the
        **per-TMA-load K width**, defaulting to the full tensor K. Pass
        ``tile_k`` explicitly to describe a K-loop kernel that loads a
        narrower slice per iteration, and ``tile_m`` to describe a
        multi-CTA kernel where each CTA loads a narrower M slice::

            # Whole-K load (one TMA, one wgmma step)
            Tile.wgmma_a(64, 16, bf16)

            # K=16 slices of a K=64 tensor (four TMA loads, four wgmma)
            Tile.wgmma_a(64, 64, bf16, tile_k=16)

            # Multi-CTA: 2048xK tensor, each CTA loads a 64xK slice
            Tile.wgmma_a(2048, 64, bf16, tile_m=64, tile_k=16)

        The TMA descriptor's box is ``(tile_m, tile_k)`` — defaulting to
        ``(M, tile_k)`` when ``tile_m`` is omitted. The user's shared
        memory allocation should be sized to match that box, e.g.
        ``smem.wgmma_tile(bf16, (tile_m, tile_k), major="K")``.

        Symbolic dims are supported — if ``k`` (or ``tile_k``) is a
        ``str``, the layout defaults to ``Layout.ROW`` and the runtime
        side is expected to resolve.
        """
        layout_k = tile_k if tile_k is not None else k
        if tile_m is not None or tile_k is not None:
            tma_box = (
                tile_m if tile_m is not None else m,
                tile_k if tile_k is not None else k,
            )
        else:
            tma_box = None
        return cls(
            m, k,
            dtype=dtype,
            layout=_wgmma_layout_for("A", layout_k, dtype),
            tma_box=tma_box,
        )

    @classmethod
    def wgmma_b(
        cls,
        k: "Dim",
        n: "Dim",
        dtype: PtxType,
        *,
        tile_k: "Dim | None" = None,
        tile_n: "Dim | None" = None,
    ) -> "Tile":
        """Tile for a wgmma B operand (row-major KxN, MN-major).

        Same idea as :meth:`wgmma_a`. ``tile_n`` slices N (per-CTA col
        tile) and ``tile_k`` slices K (per-iteration K slice). The TMA
        descriptor's box is ``(tile_k, tile_n)`` — when either is
        omitted the full tensor dim is used.
        """
        layout_n = tile_n if tile_n is not None else n
        if tile_k is not None or tile_n is not None:
            tma_box = (
                tile_k if tile_k is not None else k,
                tile_n if tile_n is not None else n,
            )
        else:
            tma_box = None
        return cls(
            k, n,
            dtype=dtype,
            layout=_wgmma_layout_for("B", layout_n, dtype),
            tma_box=tma_box,
        )


def _wgmma_layout_for(operand: str, inner_dim: "Dim", dtype: PtxType) -> Layout:
    """Map a wgmma operand's inner dimension to the right TMA Layout.

    For operand A (K-major), inner dim is K. For operand B (MN-major),
    inner dim is N. Row width in bytes = inner_dim * dtype.bytes, and
    that maps to a canonical GMMA layout class — see
    ``pyptx.wgmma_layout.pick_gmma_layout``.
    """
    if isinstance(inner_dim, str):
        # Symbolic dim; fall back to ROW and let the runtime side
        # figure out the concrete layout once the shape is bound.
        return Layout.ROW

    elem_bytes = max(dtype.bits // 8, 1)
    row_bytes = int(inner_dim) * elem_bytes
    # For rows wider than 128 bytes (e.g. N=128 bf16 = 256 bytes),
    # clamp to SWIZZLE_128B — CUTLASS does the same. The TMA loads
    # 128-byte stripes and the swizzle wraps within each stripe.
    if row_bytes >= 128:
        return Layout.TMA_128B
    try:
        return {
            16: Layout.ROW,  # INTERLEAVE = no swizzle
            32: Layout.TMA_32B,
            64: Layout.TMA_64B,
        }[row_bytes]
    except KeyError:
        raise ValueError(
            f"pyptx: wgmma_{operand.lower()}({inner_dim}, {dtype}) has row "
            f"width {row_bytes} bytes, which is not one of the canonical "
            f"GMMA widths {{16, 32, 64, 128}}."
        )

    def _numpy_dtype_name(self) -> str:
        """Map pyptx PtxType names to numpy/JAX dtype names."""
        return _PTX_TO_NUMPY_DTYPE.get(self.dtype.name, self.dtype.name)

    def __repr__(self) -> str:
        dims = ", ".join(repr(d) for d in self.shape)
        return f"Tile({dims}, {self.dtype}, {self.layout.name})"


# Map pyptx type names to their numpy/JAX dtype equivalents.
_PTX_TO_NUMPY_DTYPE: dict[str, str] = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "s8": "int8",
    "s16": "int16",
    "s32": "int32",
    "s64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "b8": "uint8",
    "b16": "uint16",
    "b32": "uint32",
    "b64": "uint64",
    "pred": "bool_",
}


def unify_envs(envs: list[dict[str, int]]) -> dict[str, int]:
    """Merge multiple {dim: int} envs; error on conflicts.

    If Tile("M", "K") and Tile("K", "N") are both inputs, the K dim must
    agree between them — this function catches mismatches.
    """
    merged: dict[str, int] = {}
    for env in envs:
        for k, v in env.items():
            if k in merged and merged[k] != v:
                raise ValueError(
                    f"Tile dim {k!r} has conflicting values: "
                    f"{merged[k]} vs {v}"
                )
            merged[k] = v
    return merged
