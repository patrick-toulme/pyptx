"""Tests for the PTX lexer."""

from pyptx.parser.lexer import tokenize
from pyptx.parser.tokens import TokenKind


def _kinds(source: str) -> list[TokenKind]:
    """Tokenize and return just the token kinds (excluding EOF)."""
    return [t.kind for t in tokenize(source) if t.kind != TokenKind.EOF]


def _texts(source: str) -> list[str]:
    """Tokenize and return just the token texts (excluding EOF/NEWLINE)."""
    return [
        t.text
        for t in tokenize(source)
        if t.kind not in (TokenKind.EOF, TokenKind.NEWLINE)
    ]


class TestDirectives:
    def test_version_directive(self):
        tokens = _texts(".version 8.5")
        assert tokens == [".version", "8.5"]

    def test_target_directive(self):
        tokens = _texts(".target sm_90a")
        assert tokens == [".target", "sm_90a"]

    def test_shared_cta(self):
        tokens = _texts(".shared::cta")
        assert tokens == [".shared::cta"]

    def test_shared_cluster(self):
        tokens = _texts(".shared::cluster")
        assert tokens == [".shared::cluster"]

    def test_type_suffix(self):
        tokens = _texts(".b32")
        assert tokens == [".b32"]


class TestRegisters:
    def test_simple_register(self):
        tokens = _texts("%r0")
        assert tokens == ["%r0"]

    def test_special_register(self):
        tokens = _texts("%tid.x")
        assert tokens == ["%tid.x"]

    def test_predicate_register(self):
        tokens = _texts("%p0")
        assert tokens == ["%p0"]

    def test_64bit_register(self):
        tokens = _texts("%rd1")
        assert tokens == ["%rd1"]


class TestNumbers:
    def test_decimal_integer(self):
        toks = tokenize("42")
        assert toks[0].kind == TokenKind.INTEGER
        assert toks[0].text == "42"

    def test_hex_integer(self):
        toks = tokenize("0xFF")
        assert toks[0].kind == TokenKind.INTEGER
        assert toks[0].text == "0xFF"

    def test_double_hex_float(self):
        toks = tokenize("0d3FF0000000000000")
        assert toks[0].kind == TokenKind.FLOAT
        assert toks[0].text == "0d3FF0000000000000"

    def test_float_hex(self):
        toks = tokenize("0f3F800000")
        assert toks[0].kind == TokenKind.FLOAT
        assert toks[0].text == "0f3F800000"

    def test_decimal_float(self):
        toks = tokenize("1.5")
        assert toks[0].kind == TokenKind.FLOAT
        assert toks[0].text == "1.5"


class TestPunctuation:
    def test_braces(self):
        kinds = _kinds("{}")
        assert kinds == [TokenKind.LBRACE, TokenKind.RBRACE]

    def test_brackets(self):
        kinds = _kinds("[]")
        assert kinds == [TokenKind.LBRACKET, TokenKind.RBRACKET]

    def test_parens(self):
        kinds = _kinds("()")
        assert kinds == [TokenKind.LPAREN, TokenKind.RPAREN]

    def test_at_bang(self):
        kinds = _kinds("@!")
        assert kinds == [TokenKind.AT, TokenKind.BANG]

    def test_semicolon(self):
        kinds = _kinds(";")
        assert kinds == [TokenKind.SEMICOLON]

    def test_pipe(self):
        kinds = _kinds("|")
        assert kinds == [TokenKind.PIPE]

    def test_angle_brackets(self):
        kinds = _kinds("<>")
        assert kinds == [TokenKind.LESS, TokenKind.GREATER]


class TestPredicatedInstruction:
    def test_predicate_tokens(self):
        texts = _texts("@%p0 bra DONE;")
        assert texts == ["@", "%p0", "bra", "DONE", ";"]

    def test_negated_predicate_tokens(self):
        texts = _texts("@!%p0 mov.b32 %r1, 0;")
        assert texts == ["@", "!", "%p0", "mov", ".b32", "%r1", ",", "0", ";"]


class TestAddressExpression:
    def test_register_with_offset(self):
        texts = _texts("[%rd0+16]")
        assert texts == ["[", "%rd0", "+", "16", "]"]


class TestComments:
    def test_line_comment(self):
        toks = tokenize("mov.b32 %r0, %r1; // copy register")
        comment_toks = [t for t in toks if t.kind == TokenKind.COMMENT]
        assert len(comment_toks) == 1
        assert comment_toks[0].text == "// copy register"


class TestStrings:
    def test_pragma_string(self):
        toks = tokenize('"nounroll"')
        assert toks[0].kind == TokenKind.STRING
        assert toks[0].text == '"nounroll"'


class TestWhitespace:
    def test_leading_whitespace_preserved(self):
        toks = tokenize("\tmov.b32 %r0, %r1;")
        # First non-whitespace token should have tab as leading_whitespace
        mov_tok = next(t for t in toks if t.text == "mov")
        assert mov_tok.leading_whitespace == "\t"

    def test_newlines_are_tokens(self):
        toks = tokenize("a\nb\n")
        nl_count = sum(1 for t in toks if t.kind == TokenKind.NEWLINE)
        assert nl_count == 2


class TestVectorOperand:
    def test_vector_tokens(self):
        texts = _texts("{%r0, %r1, %r2, %r3}")
        assert texts == ["{", "%r0", ",", "%r1", ",", "%r2", ",", "%r3", "}"]


class TestRegDecl:
    def test_reg_with_range(self):
        texts = _texts(".reg .b32 %r<100>;")
        assert texts == [".reg", ".b32", "%r", "<", "100", ">", ";"]
