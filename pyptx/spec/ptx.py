"""Declarative PTX ISA specification — the 'tablegen' for pyptx.

This module is the single source of truth for PTX instruction definitions.
The parser does NOT depend on this — it is opcode-agnostic. This spec is
used for:

  1. Validation: checking that an IR Instruction has legal modifiers/operands
  2. DSL generation: typed builders for each instruction
  3. Documentation and coverage tracking

The spec is built incrementally. Missing instructions do not block parsing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModifierGroup:
    """A named group of modifier options for an instruction.

    Modifiers within a group are typically mutually exclusive.
    The group's position in the InstructionSpec's modifier_groups tuple
    reflects the expected order in PTX text.
    """

    name: str
    options: tuple[str, ...]
    required: bool = False


@dataclass(frozen=True)
class InstructionSpec:
    """Declarative specification for a PTX instruction or instruction family.

    Attributes:
        opcode: The base opcode string (e.g. 'mov', 'ld', 'wgmma').
        modifier_groups: Ordered modifier slots. Each group defines the legal
            options for one modifier position.
        operand_pattern: Human-readable operand pattern (e.g. 'd, a, b').
            Used for documentation and operand count validation.
        min_operands: Minimum number of operands.
        max_operands: Maximum number of operands.
        description: One-line description of the instruction.
        since_version: (major, minor) PTX version that introduced it.
        arch: Required architecture (e.g. 'sm_90a'), or None if universal.
    """

    opcode: str
    modifier_groups: tuple[ModifierGroup, ...] = ()
    operand_pattern: str = ""
    min_operands: int = 0
    max_operands: int = 0
    description: str = ""
    since_version: tuple[int, int] = (1, 0)
    arch: str | None = None


# ---------------------------------------------------------------------------
# Type modifier groups (reused across many instructions)
# ---------------------------------------------------------------------------

_INT_TYPES = (".u8", ".u16", ".u32", ".u64", ".s8", ".s16", ".s32", ".s64")
_FLOAT_TYPES = (".f16", ".f16x2", ".bf16", ".bf16x2", ".tf32", ".f32", ".f64")
_BIT_TYPES = (".b8", ".b16", ".b32", ".b64", ".b128")
_FP8_TYPES = (".e4m3", ".e5m2")
_ALL_TYPES = _INT_TYPES + _FLOAT_TYPES + _BIT_TYPES + _FP8_TYPES + (".pred",)

_TYPE = ModifierGroup("type", _ALL_TYPES, required=True)
_INT_TYPE = ModifierGroup("type", _INT_TYPES, required=True)
_FLOAT_TYPE = ModifierGroup("type", _FLOAT_TYPES, required=True)
_BIT_TYPE = ModifierGroup("type", _BIT_TYPES, required=True)

_SCOPE = ModifierGroup("scope", (".cta", ".gpu", ".sys"))
_SPACE = ModifierGroup(
    "space",
    (".global", ".shared", ".shared::cta", ".shared::cluster", ".local", ".const", ".param"),
)
_CACHE = ModifierGroup("cache", (".ca", ".cg", ".cs", ".lu", ".cv", ".nc"))
_VEC = ModifierGroup("vector", (".v2", ".v4"))

# ---------------------------------------------------------------------------
# Instruction definitions
# ---------------------------------------------------------------------------

INSTRUCTIONS: dict[str, InstructionSpec] = {}


def _register(*specs: InstructionSpec) -> None:
    for spec in specs:
        INSTRUCTIONS[spec.opcode] = spec


# -- Data movement ----------------------------------------------------------

_register(
    InstructionSpec(
        opcode="mov",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Move register to register",
    ),
    InstructionSpec(
        opcode="ld",
        modifier_groups=(_SPACE, _CACHE, _VEC, _TYPE),
        operand_pattern="d, [a]",
        min_operands=2,
        max_operands=2,
        description="Load from memory",
    ),
    InstructionSpec(
        opcode="st",
        modifier_groups=(_SPACE, _CACHE, _VEC, _TYPE),
        operand_pattern="[a], b",
        min_operands=2,
        max_operands=2,
        description="Store to memory",
    ),
    InstructionSpec(
        opcode="cvt",
        modifier_groups=(
            ModifierGroup("rounding", (".rn", ".rz", ".rm", ".rp", ".rni", ".rzi", ".rmi", ".rpi")),
            ModifierGroup("ftz", (".ftz",)),
            ModifierGroup("sat", (".sat",)),
            ModifierGroup("dst_type", _ALL_TYPES, required=True),
            ModifierGroup("src_type", _ALL_TYPES, required=True),
        ),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Convert between types",
    ),
    InstructionSpec(
        opcode="cvta",
        modifier_groups=(
            _SPACE,
            ModifierGroup("size", (".u32", ".u64"), required=True),
        ),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Convert address between generic and explicit space",
    ),
)

# -- Arithmetic -------------------------------------------------------------

_register(
    InstructionSpec(
        opcode="add",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Add two values",
    ),
    InstructionSpec(
        opcode="sub",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Subtract two values",
    ),
    InstructionSpec(
        opcode="mul",
        modifier_groups=(
            ModifierGroup("mode", (".lo", ".hi", ".wide")),
            _TYPE,
        ),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Multiply two values",
    ),
    InstructionSpec(
        opcode="mad",
        modifier_groups=(
            ModifierGroup("mode", (".lo", ".hi", ".wide")),
            _TYPE,
        ),
        operand_pattern="d, a, b, c",
        min_operands=4,
        max_operands=4,
        description="Multiply-add",
    ),
    InstructionSpec(
        opcode="fma",
        modifier_groups=(
            ModifierGroup("rounding", (".rn", ".rz", ".rm", ".rp")),
            ModifierGroup("ftz", (".ftz",)),
            ModifierGroup("sat", (".sat",)),
            _FLOAT_TYPE,
        ),
        operand_pattern="d, a, b, c",
        min_operands=4,
        max_operands=4,
        description="Fused multiply-add",
    ),
    InstructionSpec(
        opcode="div",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Divide",
    ),
    InstructionSpec(
        opcode="rem",
        modifier_groups=(_INT_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Remainder",
    ),
    InstructionSpec(
        opcode="abs",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Absolute value",
    ),
    InstructionSpec(
        opcode="neg",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Negate",
    ),
    InstructionSpec(
        opcode="min",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Minimum of two values",
    ),
    InstructionSpec(
        opcode="max",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Maximum of two values",
    ),
)

# -- Bitwise / logic --------------------------------------------------------

_register(
    InstructionSpec(
        opcode="and",
        modifier_groups=(_BIT_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Bitwise AND",
    ),
    InstructionSpec(
        opcode="or",
        modifier_groups=(_BIT_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Bitwise OR",
    ),
    InstructionSpec(
        opcode="xor",
        modifier_groups=(_BIT_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Bitwise XOR",
    ),
    InstructionSpec(
        opcode="not",
        modifier_groups=(_BIT_TYPE,),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Bitwise NOT",
    ),
    InstructionSpec(
        opcode="shl",
        modifier_groups=(_BIT_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Shift left",
    ),
    InstructionSpec(
        opcode="shr",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Shift right",
    ),
)

# -- Comparison / predicate -------------------------------------------------

_register(
    InstructionSpec(
        opcode="setp",
        modifier_groups=(
            ModifierGroup("cmp", (".eq", ".ne", ".lt", ".le", ".gt", ".ge",
                                   ".lo", ".ls", ".hi", ".hs",
                                   ".equ", ".neu", ".ltu", ".leu", ".gtu", ".geu",
                                   ".num", ".nan"), required=True),
            ModifierGroup("ftz", (".ftz",)),
            _TYPE,
        ),
        operand_pattern="p[|q], a, b",
        min_operands=3,
        max_operands=3,
        description="Set predicate from comparison",
    ),
    InstructionSpec(
        opcode="selp",
        modifier_groups=(_TYPE,),
        operand_pattern="d, a, b, c",
        min_operands=4,
        max_operands=4,
        description="Select between values based on predicate",
    ),
)

# -- Control flow -----------------------------------------------------------

_register(
    InstructionSpec(
        opcode="bra",
        modifier_groups=(ModifierGroup("uni", (".uni",)),),
        operand_pattern="target",
        min_operands=1,
        max_operands=1,
        description="Branch",
    ),
    InstructionSpec(
        opcode="call",
        modifier_groups=(ModifierGroup("uni", (".uni",)),),
        operand_pattern="(ret), func, (args)",
        min_operands=1,
        max_operands=10,
        description="Call function",
    ),
    InstructionSpec(
        opcode="ret",
        modifier_groups=(ModifierGroup("uni", (".uni",)),),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Return from function",
    ),
    InstructionSpec(
        opcode="exit",
        modifier_groups=(),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Exit thread",
    ),
)

# -- Synchronization --------------------------------------------------------

_register(
    InstructionSpec(
        opcode="bar",
        modifier_groups=(
            ModifierGroup("op", (".sync", ".arrive", ".red"), required=True),
        ),
        operand_pattern="a[, b]",
        min_operands=1,
        max_operands=3,
        description="Barrier synchronization",
    ),
    InstructionSpec(
        opcode="barrier",
        modifier_groups=(
            ModifierGroup("op", (".sync", ".arrive"), required=True),
            ModifierGroup("scope", (".cta", ".cluster")),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=1,
        description="Named barrier",
    ),
    InstructionSpec(
        opcode="membar",
        modifier_groups=(
            ModifierGroup("level", (".cta", ".gl", ".sys"), required=True),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Memory barrier",
    ),
    InstructionSpec(
        opcode="fence",
        modifier_groups=(
            ModifierGroup("op", (".proxy", ".mbarrier_init", ".sc", ".acq_rel")),
            ModifierGroup("scope", (".cta", ".gpu", ".sys", ".cluster")),
            ModifierGroup("proxy", (".alias", ".async", ".tensormap::generic")),
            ModifierGroup("space", (".shared", ".shared::cta", ".shared::cluster", ".global")),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Memory fence",
    ),
)

# -- Atomic operations ------------------------------------------------------

_register(
    InstructionSpec(
        opcode="atom",
        modifier_groups=(
            _SPACE,
            _SCOPE,
            ModifierGroup("op", (".add", ".min", ".max", ".inc", ".dec",
                                  ".and", ".or", ".xor", ".exch", ".cas"), required=True),
            ModifierGroup("ftz", (".noftz",)),
            _TYPE,
        ),
        operand_pattern="d, [a], b[, c]",
        min_operands=3,
        max_operands=4,
        description="Atomic read-modify-write",
    ),
    InstructionSpec(
        opcode="red",
        modifier_groups=(
            _SPACE,
            _SCOPE,
            ModifierGroup("op", (".add", ".min", ".max", ".inc", ".dec",
                                  ".and", ".or", ".xor"), required=True),
            _TYPE,
        ),
        operand_pattern="[a], b",
        min_operands=2,
        max_operands=2,
        description="Reduction (atomic without return)",
    ),
)

# =========================================================================
# HOPPER (sm_90a, PTX 7.8 / 8.0+)
# =========================================================================

# -- wgmma: Warpgroup Matrix Multiply-Accumulate ---------------------------

# Generate all valid wgmma shapes: M always 64, N in steps of 8, K varies by type
_WGMMA_SHAPES_K16 = tuple(f".m64n{n}k16" for n in range(8, 257, 8))
_WGMMA_SHAPES_K8 = tuple(f".m64n{n}k8" for n in range(8, 257, 8))
_WGMMA_SHAPES_K32 = tuple(f".m64n{n}k32" for n in range(8, 257, 8))
_WGMMA_SHAPES_K256 = tuple(f".m64n{n}k256" for n in (8, 16, 24, 32, 48, 64, 80, 88, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 248, 256))
_WGMMA_ALL_SHAPES = _WGMMA_SHAPES_K8 + _WGMMA_SHAPES_K16 + _WGMMA_SHAPES_K32 + _WGMMA_SHAPES_K256

_WGMMA_DTYPES_D = (".f16", ".f32", ".s32")
_WGMMA_DTYPES_AB = (".f16", ".bf16", ".tf32", ".e4m3", ".e5m2", ".u8", ".s8", ".b1")

_register(
    InstructionSpec(
        opcode="wgmma",
        modifier_groups=(
            ModifierGroup("op", (".mma_async", ".fence", ".commit_group", ".wait_group"), required=True),
            ModifierGroup("sp", (".sp",)),  # sparse variant
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("satfinite", (".satfinite",)),
            ModifierGroup("shape", _WGMMA_ALL_SHAPES),
            ModifierGroup("dtype_d", _WGMMA_DTYPES_D),
            ModifierGroup("dtype_a", _WGMMA_DTYPES_AB),
            ModifierGroup("dtype_b", _WGMMA_DTYPES_AB),
        ),
        operand_pattern="d, a_desc, b_desc, scale_d, imm_a, imm_b [, trans_a, trans_b]",
        min_operands=0,
        max_operands=128,
        description="Warpgroup-level matrix multiply-accumulate (Hopper)",
        since_version=(8, 0),
        arch="sm_90a",
    ),
)

# -- TMA: Tensor Memory Access (cp.async.bulk) -----------------------------

_register(
    InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("op", (".async",), required=True),
            ModifierGroup("bulk", (".bulk",)),
            ModifierGroup("tensor", (".tensor",)),
            ModifierGroup("dim", (".1d", ".2d", ".3d", ".4d", ".5d")),
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("tensor_kind", (".tile", ".im2col", ".im2col_no_offs", ".gather4")),
            ModifierGroup("dst", (".global", ".shared::cta", ".shared::cluster")),
            ModifierGroup("src", (".global", ".shared::cta", ".shared::cluster")),
            ModifierGroup("completion", (
                ".mbarrier::complete_tx::bytes",
                ".bulk_group",
            )),
            ModifierGroup("multicast", (".multicast::cluster",)),
            ModifierGroup("cache_hint", (".L2::cache_hint",)),
        ),
        operand_pattern="[dst], [src], size|tensorCoords, [mbar] [, ctaMask] [, policy]",
        min_operands=2,
        max_operands=10,
        description="Asynchronous bulk copy / TMA tensor load-store (Hopper)",
        since_version=(8, 0),
        arch="sm_90",
    ),
    # cp.reduce.async.bulk.tensor (TMA with reduction)
    InstructionSpec(
        opcode="cp",
        modifier_groups=(
            ModifierGroup("reduce", (".reduce",), required=True),
            ModifierGroup("op_async", (".async",), required=True),
            ModifierGroup("bulk_tensor", (".bulk",)),
            ModifierGroup("tensor", (".tensor",)),
            ModifierGroup("dim", (".1d", ".2d", ".3d", ".4d", ".5d")),
            ModifierGroup("dst", (".global",)),
            ModifierGroup("src", (".shared::cta",)),
            ModifierGroup("red_op", (".add", ".min", ".max", ".inc", ".dec", ".and", ".or", ".xor"), required=True),
            ModifierGroup("completion", (".bulk_group",)),
        ),
        operand_pattern="[tensorMap, coords], [srcMem]",
        min_operands=2,
        max_operands=6,
        description="TMA reduction (shared → global with op)",
        since_version=(8, 0),
        arch="sm_90",
    ),
)

# -- MBarrier: Memory Barrier Objects --------------------------------------

_register(
    InstructionSpec(
        opcode="mbarrier",
        modifier_groups=(
            ModifierGroup("op", (
                ".init", ".inval",
                ".arrive", ".arrive_drop",
                ".test_wait", ".try_wait",
                ".pending_count",
                ".expect_tx", ".complete_tx",
            ), required=True),
            ModifierGroup("arrive_variant", (".expect_tx", ".noComplete")),
            ModifierGroup("sem", (".acquire", ".release", ".relaxed")),
            ModifierGroup("scope", (".cta", ".cluster")),
            ModifierGroup("space", (".shared", ".shared::cta", ".shared::cluster")),
            ModifierGroup("type", (".b64",)),
        ),
        operand_pattern="varies by op",
        min_operands=0,
        max_operands=4,
        description="Memory barrier object operations (Ampere+, extended on Hopper)",
        since_version=(7, 0),
        arch="sm_80",
    ),
)

# -- Matrix load/store (shared memory) ------------------------------------

_register(
    InstructionSpec(
        opcode="ldmatrix",
        modifier_groups=(
            ModifierGroup("sync", (".sync",), required=True),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("shape", (".m8n8",)),
            ModifierGroup("count", (".x1", ".x2", ".x4"), required=True),
            ModifierGroup("trans", (".trans",)),
            ModifierGroup("space", (".shared", ".shared::cta")),
            ModifierGroup("type", (".b16",), required=True),
        ),
        operand_pattern="{d}, [addr]",
        min_operands=2,
        max_operands=2,
        description="Warp-level matrix load from shared memory",
        since_version=(6, 5),
        arch="sm_75",
    ),
    InstructionSpec(
        opcode="stmatrix",
        modifier_groups=(
            ModifierGroup("sync", (".sync",), required=True),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("shape", (".m8n8",)),
            ModifierGroup("count", (".x1", ".x2", ".x4"), required=True),
            ModifierGroup("trans", (".trans",)),
            ModifierGroup("space", (".shared", ".shared::cta")),
            ModifierGroup("type", (".b16",), required=True),
        ),
        operand_pattern="[addr], {d}",
        min_operands=2,
        max_operands=2,
        description="Warp-level matrix store to shared memory (Hopper)",
        since_version=(7, 8),
        arch="sm_90",
    ),
)

# -- Cluster operations (Hopper, PTX 7.8+) --------------------------------

_register(
    InstructionSpec(
        opcode="barrier",
        modifier_groups=(
            ModifierGroup("op", (".sync", ".arrive"), required=True),
            ModifierGroup("scope", (".cta", ".cluster")),
            ModifierGroup("aligned", (".aligned",)),
        ),
        operand_pattern="[, count]",
        min_operands=0,
        max_operands=1,
        description="Named barrier / cluster barrier",
        since_version=(7, 8),
    ),
    InstructionSpec(
        opcode="mapa",
        modifier_groups=(
            ModifierGroup("space", (".shared::cluster",)),
            ModifierGroup("type", (".u32", ".u64"), required=True),
        ),
        operand_pattern="d, a, b",
        min_operands=3,
        max_operands=3,
        description="Map shared address into target CTA address space (Hopper)",
        since_version=(7, 8),
        arch="sm_90",
    ),
    InstructionSpec(
        opcode="getctarank",
        modifier_groups=(
            ModifierGroup("space", (".shared::cluster",)),
            ModifierGroup("type", (".u32", ".u64"), required=True),
        ),
        operand_pattern="d, a",
        min_operands=2,
        max_operands=2,
        description="Get CTA rank for a shared memory address (Hopper)",
        since_version=(7, 8),
        arch="sm_90",
    ),
)

# -- Hopper fence extensions -----------------------------------------------

_register(
    InstructionSpec(
        opcode="fence",
        modifier_groups=(
            ModifierGroup("op", (".proxy", ".mbarrier_init", ".sc", ".acq_rel")),
            ModifierGroup("scope", (".cta", ".gpu", ".sys", ".cluster")),
            ModifierGroup("proxy", (
                ".alias", ".async",
                ".tensormap::generic",
            )),
            # Address-space narrowing for proxy fences. PTX syntax
            # places these as a separate modifier after the proxy kind:
            # ``fence.proxy.async.shared::cta``.
            ModifierGroup("space", (".shared", ".shared::cta", ".shared::cluster", ".global")),
            ModifierGroup("sem", (".release", ".acquire")),
        ),
        operand_pattern="[addr, size]",
        min_operands=0,
        max_operands=2,
        description="Memory fence (extended with proxy.async, mbarrier_init on Hopper)",
        since_version=(7, 0),
    ),
)

# -- Hopper-specific: setmaxnreg, elect.sync, griddepcontrol ---------------

_register(
    InstructionSpec(
        opcode="setmaxnreg",
        modifier_groups=(
            ModifierGroup("action", (".inc", ".dec"), required=True),
            ModifierGroup("sync", (".sync",), required=True),
            ModifierGroup("aligned", (".aligned",), required=True),
            ModifierGroup("type", (".u32",), required=True),
        ),
        operand_pattern="regCount",
        min_operands=1,
        max_operands=1,
        description="Dynamically adjust max register count for warpgroup (Hopper)",
        since_version=(8, 0),
        arch="sm_90a",
    ),
    InstructionSpec(
        opcode="elect",
        modifier_groups=(
            ModifierGroup("sync", (".sync",), required=True),
        ),
        operand_pattern="d|p, membermask",
        min_operands=2,
        max_operands=2,
        description="Elect a leader thread from active threads (Hopper)",
        since_version=(8, 0),
        arch="sm_90",
    ),
    InstructionSpec(
        opcode="griddepcontrol",
        modifier_groups=(
            ModifierGroup("action", (".launch_dependents", ".wait"), required=True),
        ),
        operand_pattern="",
        min_operands=0,
        max_operands=0,
        description="Control dependent grid execution (Hopper)",
        since_version=(7, 8),
        arch="sm_90",
    ),
)

# -- Hopper async: red.async, st.async -------------------------------------

_register(
    InstructionSpec(
        opcode="red",
        modifier_groups=(
            ModifierGroup("async", (".async",)),
            ModifierGroup("sem", (".relaxed",)),
            ModifierGroup("scope_red", (".cluster",)),
            _SPACE,
            _SCOPE,
            ModifierGroup("op", (".add", ".min", ".max", ".inc", ".dec",
                                  ".and", ".or", ".xor"), required=True),
            ModifierGroup("completion", (".mbarrier::complete_tx::bytes",)),
            _TYPE,
        ),
        operand_pattern="[a], b [, [mbar]]",
        min_operands=2,
        max_operands=3,
        description="Reduction (atomic without return, with async variant on Hopper)",
        since_version=(1, 0),
    ),
    InstructionSpec(
        opcode="st",
        modifier_groups=(
            ModifierGroup("async", (".async",)),
            ModifierGroup("weak", (".weak",)),
            _SPACE,
            _VEC,
            ModifierGroup("completion", (".mbarrier::complete_tx::bytes",)),
            _TYPE,
        ),
        operand_pattern="[a], b [, [mbar]]",
        min_operands=2,
        max_operands=3,
        description="Store (with async variant on Hopper)",
        since_version=(1, 0),
    ),
)

# =========================================================================
# BLACKWELL (sm_100a, PTX 8.6 / 8.7+)
# =========================================================================

# -- tcgen05: Fifth-generation Tensor Core operations ----------------------

_register(
    # tcgen05.mma — the main MMA instruction
    InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (
                ".mma", ".ld", ".st", ".cp", ".shift",
                ".fence", ".commit", ".wait",
                ".alloc", ".dealloc", ".relinquish_alloc_permit",
            ), required=True),
            ModifierGroup("sp", (".sp",)),  # sparse variant
            ModifierGroup("cta_group", (".cta_group::1", ".cta_group::2")),
            ModifierGroup("kind", (
                ".kind::tf32", ".kind::f16", ".kind::i8",
                ".kind::f8f6f4", ".kind::mxf8f6f4", ".kind::mxf4", ".kind::mxf4nvf4",
            )),
            ModifierGroup("block_scale", (".block_scale",)),
            ModifierGroup("scale_vec_size", (".scale_vec_size::2X", ".scale_vec_size::4X")),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
            ModifierGroup("shape", (
                ".16x64b", ".16x128b", ".16x256b", ".32x32b",
            )),
            ModifierGroup("num", (
                ".x1", ".x2", ".x4", ".x8", ".x16", ".x32", ".x64", ".x128",
            )),
            ModifierGroup("direction", (".down",)),  # for tcgen05.shift
            ModifierGroup("space", (".shared::cta", ".shared::cluster")),
            ModifierGroup("type", (".b32", ".b64")),
            ModifierGroup("completion", (
                ".mbarrier::arrive::one",
                ".multicast::cluster",
            )),
        ),
        operand_pattern="varies by op (tmem addr, descriptors, mbar, regs)",
        min_operands=0,
        max_operands=128,
        description="Fifth-generation tensor core operations (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ),
)

# -- Blackwell: clusterlaunchcontrol ---------------------------------------

_register(
    InstructionSpec(
        opcode="clusterlaunchcontrol",
        modifier_groups=(
            ModifierGroup("op", (".try_cancel",), required=True),
            ModifierGroup("is_canceled", (".is_canceled",)),
        ),
        operand_pattern="result, [mbar]",
        min_operands=2,
        max_operands=2,
        description="Cluster launch control (Blackwell)",
        since_version=(8, 7),
        arch="sm_100a",
    ),
)

# -- Special registers / misc ----------------------------------------------

_register(
    InstructionSpec(
        opcode="shfl",
        modifier_groups=(
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("mode", (".up", ".down", ".bfly", ".idx"), required=True),
            _BIT_TYPE,
        ),
        operand_pattern="d|p, a, b, c, membermask",
        min_operands=4,
        max_operands=5,
        description="Warp shuffle",
    ),
    InstructionSpec(
        opcode="vote",
        modifier_groups=(
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("mode", (".all", ".any", ".uni", ".ballot"), required=True),
            _BIT_TYPE,
        ),
        operand_pattern="d, p, membermask",
        min_operands=2,
        max_operands=3,
        description="Warp vote",
    ),
    InstructionSpec(
        opcode="match",
        modifier_groups=(
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("mode", (".any", ".all"), required=True),
            _BIT_TYPE,
        ),
        operand_pattern="d, a, membermask",
        min_operands=2,
        max_operands=3,
        description="Warp match",
    ),
    InstructionSpec(
        opcode="redux",
        modifier_groups=(
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("op", (".add", ".min", ".max", ".and", ".or", ".xor"), required=True),
            _TYPE,
        ),
        operand_pattern="d, a, membermask",
        min_operands=2,
        max_operands=3,
        description="Warp reduction",
    ),
)

# -- Conversions / special --------------------------------------------------

_register(
    InstructionSpec(
        opcode="prmt",
        modifier_groups=(
            ModifierGroup("type", (".b32",), required=True),
            ModifierGroup("mode", (".f4e", ".b4e", ".rc8", ".ecl", ".ecr", ".rc16")),
        ),
        operand_pattern="d, a, b, c",
        min_operands=4,
        max_operands=4,
        description="Permute bytes",
    ),
    InstructionSpec(
        opcode="lop3",
        modifier_groups=(
            ModifierGroup("type", (".b32",), required=True),
        ),
        operand_pattern="d, a, b, c, immLut",
        min_operands=5,
        max_operands=5,
        description="Arbitrary 3-input logic operation",
    ),
)

# -- Blackwell (sm_100a) instructions --------------------------------------

_register(
    InstructionSpec(
        opcode="tcgen05",
        modifier_groups=(
            ModifierGroup("op", (".mma", ".cp", ".fence", ".commit", ".wait"), required=True),
            ModifierGroup("sync", (".sync",)),
            ModifierGroup("aligned", (".aligned",)),
        ),
        operand_pattern="varies",
        min_operands=0,
        max_operands=128,
        description="Fifth-generation tensor core operations (Blackwell)",
        since_version=(8, 6),
        arch="sm_100a",
    ),
)
