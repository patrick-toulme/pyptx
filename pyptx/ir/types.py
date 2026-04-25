"""Enumerations for the PTX type system, state spaces, and linking directives."""

from __future__ import annotations

from enum import Enum


class ScalarType(Enum):
    """PTX scalar types.

    Values are the PTX text representation (without the leading dot).
    """

    B8 = "b8"
    B16 = "b16"
    B32 = "b32"
    B64 = "b64"
    B128 = "b128"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"
    S8 = "s8"
    S16 = "s16"
    S32 = "s32"
    S64 = "s64"
    F16 = "f16"
    F16X2 = "f16x2"
    BF16 = "bf16"
    BF16X2 = "bf16x2"
    TF32 = "tf32"
    F32 = "f32"
    F64 = "f64"
    E4M3 = "e4m3"
    E5M2 = "e5m2"
    PRED = "pred"

    @property
    def ptx(self) -> str:
        """Return the PTX text form with leading dot, e.g. '.b32'."""
        return f".{self.value}"

    @classmethod
    def from_ptx(cls, text: str) -> ScalarType:
        """Parse from PTX text (with or without leading dot).

        Raises ValueError if not a known type.
        """
        raw = text.lstrip(".")
        return cls(raw)


class StateSpace(Enum):
    """PTX state spaces."""

    REG = "reg"
    SREG = "sreg"
    CONST = "const"
    GLOBAL = "global"
    LOCAL = "local"
    PARAM = "param"
    SHARED = "shared"
    SHARED_CTA = "shared::cta"
    SHARED_CLUSTER = "shared::cluster"

    @property
    def ptx(self) -> str:
        """Return the PTX text form with leading dot, e.g. '.shared::cta'."""
        return f".{self.value}"

    @classmethod
    def from_ptx(cls, text: str) -> StateSpace:
        """Parse from PTX text (with or without leading dot)."""
        raw = text.lstrip(".")
        return cls(raw)


class LinkingDirective(Enum):
    """PTX linking directives."""

    VISIBLE = "visible"
    EXTERN = "extern"
    WEAK = "weak"
    COMMON = "common"

    @property
    def ptx(self) -> str:
        """Return the PTX text form with leading dot, e.g. '.visible'."""
        return f".{self.value}"

    @classmethod
    def from_ptx(cls, text: str) -> LinkingDirective:
        """Parse from PTX text (with or without leading dot)."""
        raw = text.lstrip(".")
        return cls(raw)
