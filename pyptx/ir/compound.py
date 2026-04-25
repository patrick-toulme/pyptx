"""CompoundExpr IR node — a sequence of instructions emitted as one unit.

When the user writes::

    with ptx.expr():
        rd[26] = ((r[192] - 8192) & 0x3FF80) >> 4 | CONST

The Reg operators (``+``, ``&``, ``>>``, ``|``) emit PTX instructions
as normal, but they're captured into a single CompoundExpr node instead
of being emitted as separate statements. The instructions execute in
Python evaluation order, which IS the correct data-dependency order.

The emitter renders each instruction in the CompoundExpr on its own
line — identical PTX output. The benefit is purely in the Python source.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyptx.ir.nodes import Instruction


@dataclass(frozen=True)
class CompoundExpr:
    """A group of instructions traced from a single Python expression.

    Emitted identically to individual instructions — this is a cosmetic
    grouping that preserves instruction order from Python evaluation.

    Not part of the Statement union type — handled via duck typing
    in the emitter (checks for ``instructions`` attribute).
    """

    instructions: tuple[Instruction, ...]
