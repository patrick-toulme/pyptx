"""PTX lexer: source text → token stream.

Context-free tokenizer. The parser is responsible for distinguishing
opcodes from identifiers, and type suffixes from directives.
"""

from __future__ import annotations

from pyptx.parser.tokens import Token, TokenKind


class LexError(Exception):
    """Raised when the lexer encounters an unrecognizable character."""

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Lex error at {line}:{col}: {message}")


def tokenize(source: str) -> list[Token]:
    """Tokenize PTX source text into a list of tokens.

    The token stream includes NEWLINE and COMMENT tokens (needed for
    formatting preservation) and ends with an EOF token.
    """
    lexer = _Lexer(source)
    return lexer.tokenize()


class _Lexer:
    """Internal lexer state machine."""

    def __init__(self, source: str) -> None:
        self._src = source
        self._pos = 0
        self._line = 1
        self._col = 1

    # -- Character access ---------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        i = self._pos + offset
        if i >= len(self._src):
            return "\0"
        return self._src[i]

    def _advance(self) -> str:
        ch = self._src[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    def _at_end(self) -> bool:
        return self._pos >= len(self._src)

    # -- Tokenization -------------------------------------------------------

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []

        while not self._at_end():
            ws = self._consume_horizontal_whitespace()
            if self._at_end():
                break

            ch = self._peek()

            # Newline
            if ch == "\n":
                line, col = self._line, self._col
                self._advance()
                tokens.append(Token(TokenKind.NEWLINE, "\n", line, col, ws))
                continue

            # Carriage return (handle \r\n)
            if ch == "\r":
                self._advance()
                if self._peek() == "\n":
                    self._advance()
                continue

            # Block comments: /* ... */
            if ch == "/" and self._peek(1) == "*":
                tokens.append(self._lex_block_comment(ws))
                continue

            # Line comments: // ...
            if ch == "/" and self._peek(1) == "/":
                tokens.append(self._lex_comment(ws))
                continue

            # String literals
            if ch == '"':
                tokens.append(self._lex_string(ws))
                continue

            # Register names (%...)
            if ch == "%":
                tokens.append(self._lex_register(ws))
                continue

            # Directives (.word, .1d, .128x256b, .shared::cta, etc.)
            if ch == "." and (self._peek(1).isalpha() or self._peek(1).isdigit()):
                tokens.append(self._lex_directive(ws))
                continue

            # Numbers (digits, or - followed by digit when appropriate)
            if ch.isdigit():
                tokens.append(self._lex_number(ws))
                continue

            # Identifiers (letters, _, $)
            if ch.isalpha() or ch == "_" or ch == "$":
                tokens.append(self._lex_identifier(ws))
                continue

            # Single-character punctuation
            tokens.append(self._lex_punctuation(ws))

        # EOF
        tokens.append(Token(TokenKind.EOF, "", self._line, self._col, ""))
        return tokens

    # -- Whitespace ---------------------------------------------------------

    def _consume_horizontal_whitespace(self) -> str:
        start = self._pos
        while not self._at_end() and self._peek() in (" ", "\t"):
            self._advance()
        return self._src[start : self._pos]

    # -- Comments -----------------------------------------------------------

    def _lex_comment(self, ws: str) -> Token:
        line, col = self._line, self._col
        start = self._pos
        # Consume // and everything until end of line
        while not self._at_end() and self._peek() != "\n":
            self._advance()
        text = self._src[start : self._pos]
        return Token(TokenKind.COMMENT, text, line, col, ws)

    def _lex_block_comment(self, ws: str) -> Token:
        line, col = self._line, self._col
        start = self._pos
        self._advance()  # /
        self._advance()  # *
        while not self._at_end():
            if self._peek() == "*" and self._peek(1) == "/":
                self._advance()  # *
                self._advance()  # /
                break
            self._advance()
        text = self._src[start : self._pos]
        return Token(TokenKind.COMMENT, text, line, col, ws)

    # -- Strings ------------------------------------------------------------

    def _lex_string(self, ws: str) -> Token:
        line, col = self._line, self._col
        start = self._pos
        self._advance()  # opening "
        while not self._at_end():
            ch = self._peek()
            if ch == "\\":
                self._advance()  # skip escape
                if not self._at_end():
                    self._advance()
            elif ch == '"':
                self._advance()  # closing "
                break
            else:
                self._advance()
        text = self._src[start : self._pos]
        return Token(TokenKind.STRING, text, line, col, ws)

    # -- Registers ----------------------------------------------------------

    def _lex_register(self, ws: str) -> Token:
        line, col = self._line, self._col
        start = self._pos
        self._advance()  # %
        # Consume alphanumeric and dots (for %tid.x, %ntid.y, etc.)
        while not self._at_end():
            ch = self._peek()
            if ch.isalnum() or ch == "_":
                self._advance()
            elif ch == "." and self._peek(1).isalpha():
                # Part of special register like %tid.x
                self._advance()
            else:
                break
        text = self._src[start : self._pos]
        return Token(TokenKind.REGISTER, text, line, col, ws)

    # -- Directives ---------------------------------------------------------

    def _lex_directive(self, ws: str) -> Token:
        line, col = self._line, self._col
        start = self._pos
        self._advance()  # .
        # Consume the directive word
        while not self._at_end() and (self._peek().isalnum() or self._peek() == "_"):
            self._advance()
        # Handle :: qualifiers, possibly chained
        # e.g., .shared::cta, .mbarrier::complete_tx::bytes, .fence::before_thread_sync
        while self._peek() == ":" and self._peek(1) == ":":
            self._advance()  # first :
            self._advance()  # second :
            while not self._at_end() and (
                self._peek().isalnum() or self._peek() == "_"
            ):
                self._advance()
        text = self._src[start : self._pos]
        return Token(TokenKind.DIRECTIVE, text, line, col, ws)

    # -- Numbers ------------------------------------------------------------

    def _lex_number(self, ws: str) -> Token:
        line, col = self._line, self._col
        start = self._pos

        # Hex: 0x / 0X
        if self._peek() == "0" and self._peek(1) in ("x", "X"):
            self._advance()  # 0
            self._advance()  # x
            while not self._at_end() and self._is_hex_digit(self._peek()):
                self._advance()
            text = self._src[start : self._pos]
            return Token(TokenKind.INTEGER, text, line, col, ws)

        # Double-as-hex: 0d / 0D
        if self._peek() == "0" and self._peek(1) in ("d", "D"):
            self._advance()  # 0
            self._advance()  # d
            while not self._at_end() and self._is_hex_digit(self._peek()):
                self._advance()
            text = self._src[start : self._pos]
            return Token(TokenKind.FLOAT, text, line, col, ws)

        # Float-as-hex: 0f / 0F (followed by hex digits)
        if self._peek() == "0" and self._peek(1) in ("f", "F") and self._is_hex_digit(
            self._peek(2)
        ):
            self._advance()  # 0
            self._advance()  # f
            while not self._at_end() and self._is_hex_digit(self._peek()):
                self._advance()
            text = self._src[start : self._pos]
            return Token(TokenKind.FLOAT, text, line, col, ws)

        # Decimal integer or float
        while not self._at_end() and self._peek().isdigit():
            self._advance()

        # Check for decimal point
        if self._peek() == "." and self._peek(1).isdigit():
            self._advance()  # .
            while not self._at_end() and self._peek().isdigit():
                self._advance()
            # Optional exponent
            if self._peek() in ("e", "E"):
                self._advance()
                if self._peek() in ("+", "-"):
                    self._advance()
                while not self._at_end() and self._peek().isdigit():
                    self._advance()
            text = self._src[start : self._pos]
            return Token(TokenKind.FLOAT, text, line, col, ws)

        text = self._src[start : self._pos]
        return Token(TokenKind.INTEGER, text, line, col, ws)

    @staticmethod
    def _is_hex_digit(ch: str) -> bool:
        return ch in "0123456789abcdefABCDEF"

    # -- Identifiers --------------------------------------------------------

    def _lex_identifier(self, ws: str) -> Token:
        line, col = self._line, self._col
        start = self._pos
        while not self._at_end() and (
            self._peek().isalnum() or self._peek() in ("_", "$")
        ):
            self._advance()
        text = self._src[start : self._pos]
        return Token(TokenKind.IDENTIFIER, text, line, col, ws)

    # -- Punctuation --------------------------------------------------------

    _PUNCT: dict[str, TokenKind] = {
        "{": TokenKind.LBRACE,
        "}": TokenKind.RBRACE,
        "(": TokenKind.LPAREN,
        ")": TokenKind.RPAREN,
        "[": TokenKind.LBRACKET,
        "]": TokenKind.RBRACKET,
        ",": TokenKind.COMMA,
        ";": TokenKind.SEMICOLON,
        ":": TokenKind.COLON,
        "@": TokenKind.AT,
        "!": TokenKind.BANG,
        "+": TokenKind.PLUS,
        "-": TokenKind.MINUS,
        "|": TokenKind.PIPE,
        "*": TokenKind.STAR,
        "/": TokenKind.SLASH,
        "~": TokenKind.TILDE,
        "&": TokenKind.AMPERSAND,
        "<": TokenKind.LESS,
        ">": TokenKind.GREATER,
        "=": TokenKind.EQUALS,
    }

    def _lex_punctuation(self, ws: str) -> Token:
        line, col = self._line, self._col
        ch = self._advance()
        kind = self._PUNCT.get(ch)
        if kind is None:
            raise LexError(f"Unexpected character: {ch!r}", line, col)
        return Token(kind, ch, line, col, ws)
