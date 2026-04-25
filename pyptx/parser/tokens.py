"""Token types for the PTX lexer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenKind(Enum):
    """PTX token types."""

    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()

    # Names
    IDENTIFIER = auto()
    REGISTER = auto()

    # Directives (anything starting with '.' — type suffixes, state spaces,
    # keywords like .version/.entry/.func, including ::qualified like .shared::cta)
    DIRECTIVE = auto()

    # Punctuation
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    SEMICOLON = auto()
    COLON = auto()
    AT = auto()
    BANG = auto()
    PLUS = auto()
    MINUS = auto()
    PIPE = auto()
    LESS = auto()
    GREATER = auto()
    EQUALS = auto()
    STAR = auto()
    SLASH = auto()
    TILDE = auto()
    AMPERSAND = auto()

    # Whitespace / comments
    NEWLINE = auto()
    COMMENT = auto()

    # End of file
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """A single token from the PTX source.

    Attributes:
        kind: The token type.
        text: Exact source text of the token.
        line: 1-based line number.
        col: 1-based column number.
        leading_whitespace: Whitespace characters before this token
            (spaces and tabs, not newlines). Used by the parser to
            reconstruct FormattingInfo for round-trip fidelity.
    """

    kind: TokenKind
    text: str
    line: int
    col: int
    leading_whitespace: str = ""

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.text!r}, {self.line}:{self.col})"
