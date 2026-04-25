"""PTX recursive-descent parser: token stream → IR.

The parser is opcode-agnostic — it does not maintain a table of PTX opcodes.
It parses the general syntactic form:

    [@predicate] identifier[.modifier]* operand_list ;

for any instruction. Per-instruction validation (legal modifiers, operand
counts) is handled separately by pyptx.spec.validate.
"""

from __future__ import annotations

from pyptx.ir.nodes import (
    AddressOperand,
    AddressSize,
    Directive,
    FormattingInfo,
    Function,
    FunctionDirective,
    ImmediateOperand,
    Instruction,
    Label,
    LabelOperand,
    Module,
    NegatedOperand,
    Operand,
    ParenthesizedOperand,
    Param,
    PipeOperand,
    Predicate,
    PragmaDirective,
    RegDecl,
    RegisterOperand,
    Statement,
    Target,
    VarDecl,
    VectorOperand,
    Version,
)
from pyptx.ir.types import LinkingDirective, ScalarType, StateSpace
from pyptx.parser.lexer import tokenize
from pyptx.parser.tokens import Token, TokenKind


class ParseError(Exception):
    """Raised when the parser encounters invalid syntax."""

    def __init__(self, message: str, line: int, col: int) -> None:
        self.line = line
        self.col = col
        super().__init__(f"Parse error at {line}:{col}: {message}")


def parse(source: str) -> Module:
    """Parse PTX source text into an IR Module."""
    tokens = tokenize(source)
    parser = _Parser(tokens, source)
    return parser.parse_module()


class _Parser:
    """Recursive-descent parser for PTX."""

    def __init__(self, tokens: list[Token], source: str = "") -> None:
        self._tokens = tokens
        self._pos = 0
        self._pending_stmts: list[Statement] = []
        self._source = source
        self._source_lines = source.split("\n") if source else []
        self._collected_comments: list[str] = []

    # -- Cursor utilities ---------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _peek_kind(self) -> TokenKind:
        return self._tokens[self._pos].kind

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: TokenKind, text: str | None = None) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise ParseError(
                f"Expected {kind.name}, got {tok.kind.name} ({tok.text!r})",
                tok.line,
                tok.col,
            )
        if text is not None and tok.text != text:
            raise ParseError(
                f"Expected {text!r}, got {tok.text!r}", tok.line, tok.col
            )
        return self._advance()

    def _match(self, kind: TokenKind, text: str | None = None) -> Token | None:
        tok = self._peek()
        if tok.kind != kind:
            return None
        if text is not None and tok.text != text:
            return None
        return self._advance()

    def _skip_newlines_and_comments(self) -> int:
        """Skip newline and comment tokens. Return count of blank lines.

        Collected comments and blank lines are stored in
        self._collected_comments for the next statement to pick up.
        """
        blank_lines = 0
        prev_was_newline = False
        while self._peek_kind() in (TokenKind.NEWLINE, TokenKind.COMMENT):
            tok = self._peek()
            if tok.kind == TokenKind.NEWLINE:
                if prev_was_newline:
                    blank_lines += 1
                    self._collected_comments.append("")  # blank line
                prev_was_newline = True
            else:
                # Comment token — collect it with its whitespace
                comment_text = tok.leading_whitespace + tok.text
                self._collected_comments.append(comment_text)
                prev_was_newline = False
            self._advance()
        return blank_lines

    def _take_comments(self) -> tuple[str, ...]:
        """Take and clear collected comments."""
        comments = tuple(self._collected_comments)
        self._collected_comments.clear()
        return comments

    def _error(self, message: str) -> ParseError:
        tok = self._peek()
        return ParseError(message, tok.line, tok.col)

    def _consume_trailing_comment_as_raw_header_part(self) -> None:
        """Consume a trailing comment on the same line (for header round-trip)."""
        # The raw_header captures this from source lines, so just skip the token
        if self._peek_kind() == TokenKind.COMMENT:
            self._advance()

    # -- Module -------------------------------------------------------------

    def parse_module(self) -> Module:
        from pyptx.ir.nodes import BlankLine, Comment as CommentNode

        # Collect leading comments/blank lines before .version
        directives_before: list[Directive] = []
        prev_was_newline = False
        while self._peek_kind() in (TokenKind.NEWLINE, TokenKind.COMMENT):
            tok = self._peek()
            if tok.kind == TokenKind.NEWLINE:
                if prev_was_newline:
                    directives_before.append(BlankLine())
                prev_was_newline = True
            else:
                directives_before.append(CommentNode(text=tok.leading_whitespace + tok.text))
                prev_was_newline = False
            self._advance()

        version = self._parse_version()
        # Capture inline comment after version
        self._consume_trailing_comment_as_raw_header_part()
        self._skip_newlines_and_comments()
        target = self._parse_target()
        self._consume_trailing_comment_as_raw_header_part()
        self._skip_newlines_and_comments()
        address_size = self._parse_address_size()
        self._consume_trailing_comment_as_raw_header_part()

        # Build raw header from source (to preserve inline comments on header lines)
        raw_header: str | None = None
        if self._source_lines:
            # Find the lines that contain version/target/address_size
            # Use the source lines for lossless round-trip of the header
            version_line = version.major  # placeholder; compute from tokens
            header_start = 1
            # Find lines with .version, .target, .address_size
            for i, line in enumerate(self._source_lines):
                stripped = line.strip()
                if stripped.startswith('.version'):
                    header_start = i + 1
                    break
            header_end = header_start
            for i in range(header_start - 1, len(self._source_lines)):
                stripped = self._source_lines[i].strip()
                if stripped.startswith('.address_size'):
                    header_end = i + 1
                    break
            raw_header = "\n".join(self._source_lines[header_start - 1 : header_end])

        # Parse module-level directives with comments/blank lines as IR nodes
        directives: list[Directive] = list(directives_before)
        prev_was_newline = False
        while self._peek_kind() != TokenKind.EOF:
            tok = self._peek()
            if tok.kind == TokenKind.NEWLINE:
                if prev_was_newline:
                    directives.append(BlankLine())
                prev_was_newline = True
                self._advance()
                continue
            if tok.kind == TokenKind.COMMENT:
                directives.append(CommentNode(text=tok.leading_whitespace + tok.text))
                self._advance()
                prev_was_newline = False
                continue
            prev_was_newline = False
            try:
                directives.append(self._parse_top_level_directive())
            except ParseError:
                # Fault tolerance: capture as RawLine
                from pyptx.ir.nodes import RawLine
                raw_parts = [tok.leading_whitespace]
                while not self._at_end() and self._peek_kind() not in (
                    TokenKind.NEWLINE, TokenKind.EOF,
                ):
                    t = self._advance()
                    raw_parts.append(t.leading_whitespace)
                    raw_parts.append(t.text)
                directives.append(RawLine(text="".join(raw_parts)))

        return Module(
            version=version,
            target=target,
            address_size=address_size,
            directives=tuple(directives),
            raw_header=raw_header,
            raw_source=self._source if self._source else None,
        )

    def _parse_version(self) -> Version:
        self._expect(TokenKind.DIRECTIVE, ".version")
        self._skip_newlines_and_comments()
        # Version is lexed as a FLOAT token (e.g. "8.5") or INTEGER + DIRECTIVE
        tok = self._peek()
        if tok.kind == TokenKind.FLOAT:
            self._advance()
            parts = tok.text.split(".")
            return Version(major=int(parts[0]), minor=int(parts[1]))
        elif tok.kind == TokenKind.INTEGER:
            major_tok = self._advance()
            # The .N part will be lexed as a DIRECTIVE
            minor_tok = self._expect(TokenKind.DIRECTIVE)
            minor_str = minor_tok.text.lstrip(".")
            return Version(major=int(major_tok.text), minor=int(minor_str))
        else:
            raise self._error(f"Expected version number, got {tok.kind.name}")

    def _parse_target(self) -> Target:
        self._expect(TokenKind.DIRECTIVE, ".target")
        self._skip_newlines_and_comments()
        targets: list[str] = []
        # Target name is an identifier (e.g. sm_90a)
        tok = self._expect(TokenKind.IDENTIFIER)
        targets.append(tok.text)
        while self._match(TokenKind.COMMA):
            self._skip_newlines_and_comments()
            tok = self._expect(TokenKind.IDENTIFIER)
            targets.append(tok.text)
        return Target(targets=tuple(targets))

    def _parse_address_size(self) -> AddressSize:
        self._expect(TokenKind.DIRECTIVE, ".address_size")
        self._skip_newlines_and_comments()
        tok = self._expect(TokenKind.INTEGER)
        return AddressSize(size=int(tok.text))

    # -- Top-level directives -----------------------------------------------

    def _parse_top_level_directive(self) -> Directive:
        tok = self._peek()

        # Check for linking directives (.visible, .extern, .weak, .common)
        if tok.kind == TokenKind.DIRECTIVE and tok.text in (
            ".visible",
            ".extern",
            ".weak",
            ".common",
        ):
            return self._parse_function_or_global_with_linking()

        # .entry or .func
        if tok.kind == TokenKind.DIRECTIVE and tok.text in (".entry", ".func"):
            return self._parse_function(linking=None)

        # Module-level variable declaration (.global, .shared, .const)
        if tok.kind == TokenKind.DIRECTIVE and tok.text in (
            ".global",
            ".shared",
            ".const",
            ".local",
        ):
            return self._parse_var_decl_at(indent="")

        # .pragma
        if tok.kind == TokenKind.DIRECTIVE and tok.text == ".pragma":
            return self._parse_pragma(indent="")

        # .file, .section, .loc — debug info directives at module level
        if tok.kind == TokenKind.DIRECTIVE and tok.text in _SPACE_SEPARATED_DIRECTIVES:
            start_line = tok.line
            self._advance()
            return self._parse_space_separated_directive(
                tok.text, "", start_line, None,
            )

        # Module-level data directives: .b8 1, .u32 0, etc. (used in debug sections)
        if tok.kind == TokenKind.DIRECTIVE and tok.text in _DATA_DIRECTIVES:
            start_line = tok.line
            self._advance()
            return self._parse_space_separated_directive(
                tok.text, "", start_line, None,
            )

        # Module-level { } data blocks (used inside .section blocks)
        if tok.kind == TokenKind.LBRACE:
            start_line = tok.line
            self._advance()
            return Instruction(
                opcode="{",
                modifiers=(),
                operands=(),
                predicate=None,
                formatting=FormattingInfo(
                    indent=tok.leading_whitespace,
                    raw_line=self._extract_source_lines(start_line, start_line) if self._source_lines else None,
                ),
            )
        if tok.kind == TokenKind.RBRACE:
            start_line = tok.line
            self._advance()
            return Instruction(
                opcode="}",
                modifiers=(),
                operands=(),
                predicate=None,
                formatting=FormattingInfo(
                    indent=tok.leading_whitespace,
                    raw_line=self._extract_source_lines(start_line, start_line) if self._source_lines else None,
                ),
            )

        raise self._error(f"Unexpected directive at module level: {tok.text!r}")

    def _parse_function_or_global_with_linking(self) -> Directive:
        start_line = self._peek().line
        linking_tok = self._advance()
        linking = LinkingDirective.from_ptx(linking_tok.text)
        self._skip_newlines_and_comments()

        tok = self._peek()
        if tok.kind == TokenKind.DIRECTIVE and tok.text in (".entry", ".func"):
            return self._parse_function(linking=linking)

        # Global variable with linking directive
        if tok.kind == TokenKind.DIRECTIVE and tok.text in (
            ".global",
            ".shared",
            ".const",
            ".local",
        ):
            vd = self._parse_var_decl_at(indent="")
            # Override raw_line to include the linking directive
            raw_line = None
            if self._source_lines and vd.formatting:
                raw_line = self._extract_source_lines(start_line, start_line)
            preceding = vd.formatting.preceding_comments if vd.formatting else ()
            return VarDecl(
                state_space=vd.state_space,
                type=vd.type,
                name=vd.name,
                array_size=vd.array_size,
                alignment=vd.alignment,
                initializer=vd.initializer,
                linking=linking,
                formatting=FormattingInfo(
                    preceding_comments=preceding,
                    raw_line=raw_line,
                ),
            )

        raise self._error(
            f"Expected .entry, .func, or state space after {linking_tok.text}"
        )

    # -- Functions ----------------------------------------------------------

    def _parse_function(self, linking: LinkingDirective | None) -> Function:
        preceding = self._take_comments()

        func_tok = self._advance()  # .entry or .func
        is_entry = func_tok.text == ".entry"
        self._skip_newlines_and_comments()

        # For .func, check for return parameters: .func (ret_params) name(...)
        return_params: tuple[Param, ...] | None = None
        if not is_entry and self._peek_kind() == TokenKind.LPAREN:
            return_params = self._parse_param_list()
            self._skip_newlines_and_comments()

        # Function name
        name_tok = self._expect(TokenKind.IDENTIFIER)
        name = name_tok.text
        self._skip_newlines_and_comments()

        # Parameters
        params: tuple[Param, ...] = ()
        if self._peek_kind() == TokenKind.LPAREN:
            params = self._parse_param_list()
        self._skip_newlines_and_comments()

        # Function directives (.maxnreg, .maxntid, etc.)
        func_directives: list[FunctionDirective] = []
        while (
            self._peek_kind() == TokenKind.DIRECTIVE
            and self._peek().text in _FUNCTION_DIRECTIVE_NAMES
        ):
            func_directives.append(self._parse_function_directive())
            self._skip_newlines_and_comments()

        # Body
        body: tuple[Statement, ...] = ()
        if self._peek_kind() == TokenKind.LBRACE:
            body = self._parse_function_body()

        return Function(
            is_entry=is_entry,
            name=name,
            params=params,
            return_params=return_params,
            body=body,
            linking=linking,
            directives=tuple(func_directives),
            formatting=FormattingInfo(preceding_comments=preceding) if preceding else None,
        )

    def _parse_param_list(self) -> tuple[Param, ...]:
        self._expect(TokenKind.LPAREN)
        self._skip_newlines_and_comments()

        params: list[Param] = []
        if self._peek_kind() != TokenKind.RPAREN:
            params.append(self._parse_param())
            self._skip_newlines_and_comments()
            while self._match(TokenKind.COMMA):
                self._skip_newlines_and_comments()
                params.append(self._parse_param())
                self._skip_newlines_and_comments()

        self._expect(TokenKind.RPAREN)
        return tuple(params)

    def _parse_param(self) -> Param:
        # .param [.align N] .type [.ptr .space [.align N]] name[array]
        # OR for .func params: .reg .type name
        tok = self._peek()
        if tok.kind != TokenKind.DIRECTIVE or tok.text not in (".param", ".reg"):
            raise self._error(f"Expected '.param' or '.reg', got {tok.text!r}")
        state_space_tok = self._advance()
        state_space = StateSpace.from_ptx(state_space_tok.text)
        self._skip_newlines_and_comments()

        alignment: int | None = None
        if self._peek_kind() == TokenKind.DIRECTIVE and self._peek().text == ".align":
            self._advance()
            self._skip_newlines_and_comments()
            alignment = int(self._expect(TokenKind.INTEGER).text)
            self._skip_newlines_and_comments()

        type_tok = self._expect(TokenKind.DIRECTIVE)
        scalar_type = ScalarType.from_ptx(type_tok.text)
        self._skip_newlines_and_comments()

        # Optional .ptr qualifier
        ptr_state_space: StateSpace | None = None
        ptr_alignment: int | None = None
        if self._peek_kind() == TokenKind.DIRECTIVE and self._peek().text == ".ptr":
            self._advance()
            self._skip_newlines_and_comments()
            # State space
            ss_tok = self._expect(TokenKind.DIRECTIVE)
            ptr_state_space = StateSpace.from_ptx(ss_tok.text)
            self._skip_newlines_and_comments()
            # Optional .align
            if (
                self._peek_kind() == TokenKind.DIRECTIVE
                and self._peek().text == ".align"
            ):
                self._advance()
                self._skip_newlines_and_comments()
                ptr_alignment = int(self._expect(TokenKind.INTEGER).text)
                self._skip_newlines_and_comments()

        # Parameter name
        name_tok = self._expect(TokenKind.IDENTIFIER)
        name = name_tok.text

        # Optional array size
        array_size: int | None = None
        if self._peek_kind() == TokenKind.LBRACKET:
            self._advance()
            size_tok = self._expect(TokenKind.INTEGER)
            array_size = int(size_tok.text)
            self._expect(TokenKind.RBRACKET)

        return Param(
            state_space=state_space,
            type=scalar_type,
            name=name,
            array_size=array_size,
            alignment=alignment,
            ptr_state_space=ptr_state_space,
            ptr_alignment=ptr_alignment,
        )

    def _parse_function_directive(self) -> FunctionDirective:
        tok = self._advance()  # e.g. .maxnreg
        name = tok.text.lstrip(".")
        self._skip_newlines_and_comments()
        values: list[int | str] = []
        # Parse comma-separated values
        val_tok = self._peek()
        if val_tok.kind == TokenKind.INTEGER:
            values.append(int(self._advance().text))
            while self._match(TokenKind.COMMA):
                self._skip_newlines_and_comments()
                values.append(int(self._expect(TokenKind.INTEGER).text))
        return FunctionDirective(name=name, values=tuple(values))

    def _parse_function_body(self) -> tuple[Statement, ...]:
        self._expect(TokenKind.LBRACE)
        return self._parse_body_until(TokenKind.RBRACE)

    def _parse_body_until(self, end_token: TokenKind) -> tuple[Statement, ...]:
        """Parse statements, comments, and blank lines until end_token."""
        from pyptx.ir.nodes import BlankLine, Block, Comment, RawLine

        statements: list[Statement] = []
        prev_was_newline = False

        while self._peek_kind() != end_token and self._peek_kind() != TokenKind.EOF:
            tok = self._peek()

            # Newline → track blank lines
            if tok.kind == TokenKind.NEWLINE:
                if prev_was_newline:
                    statements.append(BlankLine())
                prev_was_newline = True
                self._advance()
                continue

            # Comment → Comment IR node
            if tok.kind == TokenKind.COMMENT:
                statements.append(Comment(text=tok.leading_whitespace + tok.text))
                self._advance()
                prev_was_newline = False
                continue

            prev_was_newline = False

            # Nested block: { ... }
            if tok.kind == TokenKind.LBRACE:
                block_indent = tok.leading_whitespace
                self._advance()  # consume {
                inner = self._parse_body_until(TokenKind.RBRACE)
                # _parse_body_until already consumed the closing }
                statements.append(Block(
                    body=inner,
                    formatting=FormattingInfo(indent=block_indent),
                ))
                continue

            # Regular statement — wrap in try/except for fault tolerance
            try:
                stmt = self._parse_statement()
                statements.append(stmt)
                if self._pending_stmts:
                    statements.extend(self._pending_stmts)
                    self._pending_stmts.clear()
            except ParseError:
                # Fault tolerance: capture the rest of the line as RawLine
                raw_parts = [tok.leading_whitespace]
                while not self._at_end() and self._peek_kind() not in (
                    TokenKind.NEWLINE, TokenKind.EOF,
                ):
                    t = self._advance()
                    raw_parts.append(t.leading_whitespace)
                    raw_parts.append(t.text)
                statements.append(RawLine(text="".join(raw_parts)))

        self._expect(end_token)
        return tuple(statements)

    def _at_end(self) -> bool:
        return self._pos >= len(self._tokens) or self._peek_kind() == TokenKind.EOF

    # -- Statements ---------------------------------------------------------

    def _parse_statement(self) -> Statement:
        tok = self._peek()
        indent = tok.leading_whitespace

        # Nested scope block: { ... } — parse contents recursively
        if tok.kind == TokenKind.LBRACE:
            return self._parse_nested_block(indent)

        # Predicated instruction: @pred ...
        if tok.kind == TokenKind.AT:
            return self._parse_instruction_with_indent(indent)

        # Directive-starting statement
        if tok.kind == TokenKind.DIRECTIVE:
            return self._parse_directive_statement(indent)

        # Identifier: could be a label (name:) or instruction (opcode ...)
        if tok.kind == TokenKind.IDENTIFIER:
            return self._parse_label_or_instruction(indent)

        raise self._error(
            f"Expected statement, got {tok.kind.name} ({tok.text!r})"
        )

    def _parse_directive_statement(self, indent: str) -> Statement:
        tok = self._peek()

        # Register declaration
        if tok.text == ".reg":
            decls = self._parse_reg_decl(indent)
            if len(decls) == 1:
                return decls[0]
            # Multi-register: queue extras, return first
            self._pending_stmts.extend(decls[1:])
            return decls[0]

        # Variable declarations (.shared, .global, .local, .const, .param)
        if tok.text in (".shared", ".global", ".local", ".const", ".param",
                        ".shared::cta", ".shared::cluster"):
            return self._parse_var_decl_at(indent)

        # Pragma
        if tok.text == ".pragma":
            return self._parse_pragma(indent)

        # Otherwise treat as instruction (some directives like .loc are instructions)
        return self._parse_instruction_with_indent(indent)

    def _parse_label_or_instruction(self, indent: str) -> Statement:
        # Look ahead: if IDENTIFIER followed by COLON, it's a label
        if self._peek_kind() == TokenKind.IDENTIFIER:
            # Save position for backtracking
            saved_pos = self._pos
            ident_tok = self._advance()
            self._skip_newlines_and_comments()
            if self._peek_kind() == TokenKind.COLON:
                self._advance()  # consume :
                preceding = self._take_comments()
                return Label(
                    name=ident_tok.text,
                    formatting=FormattingInfo(
                        indent=indent,
                        preceding_comments=preceding,
                    ),
                )
            # Not a label, backtrack
            self._pos = saved_pos

        return self._parse_instruction_with_indent(indent)

    # -- Register declarations ----------------------------------------------

    def _parse_reg_decl(self, indent: str) -> list[RegDecl]:
        start_line = self._peek().line
        self._advance()  # .reg
        self._skip_newlines_and_comments()

        type_tok = self._expect(TokenKind.DIRECTIVE)
        scalar_type = ScalarType.from_ptx(type_tok.text)
        self._skip_newlines_and_comments()

        # Register name (can be %r or plain identifier like accum)
        tok = self._peek()
        if tok.kind in (TokenKind.REGISTER, TokenKind.IDENTIFIER):
            name_tok = self._advance()
        else:
            raise self._error(f"Expected register name, got {tok.kind.name} ({tok.text!r})")
        name = name_tok.text

        # Optional range: <count>
        count: int | None = None
        if self._peek_kind() == TokenKind.LESS:
            self._advance()  # <
            count_tok = self._expect(TokenKind.INTEGER)
            count = int(count_tok.text)
            self._expect(TokenKind.GREATER)  # >

        # Handle comma-separated multi-register declarations:
        # .reg .b64 desc_a, desc_b;
        names: list[tuple[str, int | None]] = [(name, count)]
        while self._peek_kind() == TokenKind.COMMA:
            self._advance()
            self._skip_newlines_and_comments()
            tok = self._peek()
            if tok.kind in (TokenKind.REGISTER, TokenKind.IDENTIFIER):
                extra_name = self._advance().text
            else:
                raise self._error(f"Expected register name, got {tok.kind.name}")
            extra_count: int | None = None
            if self._peek_kind() == TokenKind.LESS:
                self._advance()
                extra_count = int(self._expect(TokenKind.INTEGER).text)
                self._expect(TokenKind.GREATER)
            names.append((extra_name, extra_count))

        semi_tok = self._expect(TokenKind.SEMICOLON)
        end_line = semi_tok.line

        # Capture raw source line for lossless round-trip
        raw_line = self._extract_source_lines(start_line, end_line) if self._source_lines else None
        preceding = self._take_comments()

        # Always return a single RegDecl with raw_line for perfect round-trip
        # (multi-reg decls are preserved via raw_line)
        return [RegDecl(
            type=scalar_type,
            name=name,
            count=count,
            formatting=FormattingInfo(
                indent=indent,
                preceding_comments=preceding,
                raw_line=raw_line,
            ),
        )]

    # -- Variable declarations ----------------------------------------------

    def _parse_var_decl_at(self, indent: str) -> VarDecl:
        start_line = self._peek().line
        ss_tok = self._advance()  # state space directive
        state_space = StateSpace.from_ptx(ss_tok.text)
        self._skip_newlines_and_comments()

        alignment: int | None = None
        if self._peek_kind() == TokenKind.DIRECTIVE and self._peek().text == ".align":
            self._advance()
            self._skip_newlines_and_comments()
            alignment = int(self._expect(TokenKind.INTEGER).text)
            self._skip_newlines_and_comments()

        type_tok = self._expect(TokenKind.DIRECTIVE)
        scalar_type = ScalarType.from_ptx(type_tok.text)
        self._skip_newlines_and_comments()

        name_tok = self._expect(TokenKind.IDENTIFIER)
        name = name_tok.text

        # Array dimensions: name[N] or name[N][M] etc. (empty [] is also valid)
        array_dims: list[int | str] = []
        has_empty = False
        while self._peek_kind() == TokenKind.LBRACKET:
            self._advance()
            if self._peek_kind() == TokenKind.RBRACKET:
                # Empty brackets: name[]
                has_empty = True
                array_dims.append("")
            else:
                size_tok = self._expect(TokenKind.INTEGER)
                array_dims.append(int(size_tok.text))
            self._expect(TokenKind.RBRACKET)

        array_size: int | None = None
        if len(array_dims) == 1:
            if isinstance(array_dims[0], int):
                array_size = array_dims[0]
            else:
                # Empty brackets — encode in name
                name = name + "[]"
                array_size = None
        elif len(array_dims) > 1:
            # Multi-dimensional: encode all dims in name for lossless round-trip
            dim_str = "".join(
                f"[{d}]" if isinstance(d, int) else "[]" for d in array_dims
            )
            name = name + dim_str
            array_size = None

        # Optional initializer: = {val1, val2, ...} or = scalar
        initializer: tuple[str, ...] | None = None
        if self._peek_kind() == TokenKind.EQUALS:
            self._advance()
            self._skip_newlines_and_comments()
            init_parts: list[str] = []
            if self._peek_kind() == TokenKind.LBRACE:
                # {v1, v2, ...}
                self._advance()
                while self._peek_kind() != TokenKind.RBRACE:
                    if self._peek_kind() in (TokenKind.NEWLINE, TokenKind.COMMENT):
                        self._advance()
                        continue
                    if self._peek_kind() == TokenKind.COMMA:
                        self._advance()
                        continue
                    tok = self._advance()
                    init_parts.append(tok.text)
                self._expect(TokenKind.RBRACE)
            else:
                # Scalar initializer
                init_parts.append(self._advance().text)
            initializer = tuple(init_parts)

        semi_tok = self._expect(TokenKind.SEMICOLON)
        end_line = semi_tok.line

        raw_line = self._extract_source_lines(start_line, end_line) if self._source_lines else None
        preceding = self._take_comments()

        return VarDecl(
            state_space=state_space,
            type=scalar_type,
            name=name,
            array_size=array_size,
            alignment=alignment,
            initializer=initializer,
            formatting=FormattingInfo(
                indent=indent,
                preceding_comments=preceding,
                raw_line=raw_line,
            ),
        )

    # -- Pragma -------------------------------------------------------------

    def _parse_pragma(self, indent: str) -> PragmaDirective:
        self._advance()  # .pragma
        self._skip_newlines_and_comments()
        str_tok = self._expect(TokenKind.STRING)
        # Strip surrounding quotes
        value = str_tok.text[1:-1]
        self._expect(TokenKind.SEMICOLON)
        return PragmaDirective(
            value=value, formatting=FormattingInfo(indent=indent)
        )

    # -- Source extraction helpers -------------------------------------------

    def _extract_source_lines(self, start_line: int, end_line: int) -> str:
        """Extract raw source text for lines [start_line, end_line] (1-based)."""
        if not self._source_lines:
            return ""
        lines = self._source_lines[start_line - 1 : end_line]
        return "\n".join(lines)

    # -- Instructions -------------------------------------------------------

    def _parse_instruction_with_indent(self, indent: str) -> Instruction:
        start_line = self._peek().line

        # Optional predicate
        predicate: Predicate | None = None
        if self._peek_kind() == TokenKind.AT:
            predicate = self._parse_predicate()

        # Opcode (identifier or directive like .loc)
        tok = self._peek()
        if tok.kind == TokenKind.IDENTIFIER:
            opcode = self._advance().text
        elif tok.kind == TokenKind.DIRECTIVE:
            # Some PTX "instructions" start with a dot (e.g., .loc for debug)
            opcode = self._advance().text

            # Special handling for debug/data directives: space-separated
            # operands, no semicolon (.loc, .file, .section)
            if opcode in _SPACE_SEPARATED_DIRECTIVES:
                return self._parse_space_separated_directive(
                    opcode, indent, start_line, predicate,
                )
        else:
            raise self._error(
                f"Expected opcode, got {tok.kind.name} ({tok.text!r})"
            )

        # Modifiers: consume following .XXX directive tokens
        modifiers: list[str] = []
        while self._peek_kind() == TokenKind.DIRECTIVE:
            modifiers.append(self._advance().text)

        # Skip newlines/comments between modifiers and operands (multi-line instructions)
        self._skip_newlines_and_comments()

        # Operands (may be empty, e.g., "ret;")
        operands: tuple[Operand, ...] = ()
        if self._peek_kind() != TokenKind.SEMICOLON:
            operands = self._parse_operand_list()

        self._skip_newlines_and_comments()
        semi_tok = self._expect(TokenKind.SEMICOLON)
        end_line = semi_tok.line

        # Capture trailing comment on the same line as the semicolon
        trailing = ""
        if self._peek_kind() == TokenKind.COMMENT and self._peek().line == end_line:
            trailing = self._peek().leading_whitespace + self._peek().text
            self._advance()

        # Always capture raw source for lossless round-trip
        raw_line: str | None = None
        if self._source_lines:
            raw_line = self._extract_source_lines(start_line, end_line)

        # Collect preceding comments
        preceding = self._take_comments()

        return Instruction(
            opcode=opcode,
            modifiers=tuple(modifiers),
            operands=operands,
            predicate=predicate,
            formatting=FormattingInfo(
                indent=indent,
                trailing=trailing,
                preceding_comments=preceding,
                raw_line=raw_line,
            ),
        )

    def _parse_space_separated_directive(
        self,
        opcode: str,
        indent: str,
        start_line: int,
        predicate: Predicate | None,
    ) -> Instruction:
        """Parse directives like .loc with space-separated operands, no semicolon.

        Format: .loc file_idx line col
                .file N "filename"
        Terminated by newline (optionally followed by inline comment).
        """
        operands: list[Operand] = []
        # Consume tokens until NEWLINE or EOF
        while self._peek_kind() not in (TokenKind.NEWLINE, TokenKind.EOF, TokenKind.COMMENT):
            tok = self._peek()
            if tok.kind == TokenKind.INTEGER:
                operands.append(ImmediateOperand(self._advance().text))
            elif tok.kind == TokenKind.STRING:
                # .file 1 "name.py"
                text = self._advance().text
                operands.append(LabelOperand(text))
            elif tok.kind == TokenKind.IDENTIFIER:
                operands.append(LabelOperand(self._advance().text))
            elif tok.kind == TokenKind.DIRECTIVE:
                # .section .debug_info
                operands.append(LabelOperand(self._advance().text))
            elif tok.kind == TokenKind.MINUS:
                self._advance()
                num = self._expect(TokenKind.INTEGER).text
                operands.append(ImmediateOperand(f"-{num}"))
            else:
                break

        end_line = self._peek().line if self._peek_kind() != TokenKind.EOF else start_line

        # Capture trailing comment
        trailing = ""
        if self._peek_kind() == TokenKind.COMMENT:
            trailing = self._peek().leading_whitespace + self._peek().text
            self._advance()

        raw_line: str | None = None
        if self._source_lines:
            raw_line = self._extract_source_lines(start_line, end_line)

        preceding = self._take_comments()

        return Instruction(
            opcode=opcode,
            modifiers=(),
            operands=tuple(operands),
            predicate=predicate,
            formatting=FormattingInfo(
                indent=indent,
                trailing=trailing,
                preceding_comments=preceding,
                raw_line=raw_line,
            ),
        )

    def _parse_predicate(self) -> Predicate:
        self._expect(TokenKind.AT)
        negated = False
        if self._peek_kind() == TokenKind.BANG:
            self._advance()
            negated = True
        # Predicate register can be %p0 or a plain name like exit_predicate
        tok = self._peek()
        if tok.kind in (TokenKind.REGISTER, TokenKind.IDENTIFIER):
            reg_tok = self._advance()
        else:
            raise self._error(f"Expected predicate register, got {tok.kind.name} ({tok.text!r})")
        return Predicate(register=reg_tok.text, negated=negated)

    # -- Operands -----------------------------------------------------------

    def _parse_operand_list(self) -> tuple[Operand, ...]:
        operands: list[Operand] = []
        self._skip_newlines_and_comments()
        operands.append(self._parse_operand_with_pipe())
        while self._match(TokenKind.COMMA):
            self._skip_newlines_and_comments()
            operands.append(self._parse_operand_with_pipe())
        return tuple(operands)

    def _parse_operand_with_pipe(self) -> Operand:
        """Parse an operand, handling the | operator for dual predicates."""
        left = self._parse_single_operand()
        if self._peek_kind() == TokenKind.PIPE:
            self._advance()
            right = self._parse_single_operand()
            return PipeOperand(left=left, right=right)
        return left

    def _parse_single_operand(self) -> Operand:
        tok = self._peek()

        # Negated operand: !%p0
        if tok.kind == TokenKind.BANG:
            self._advance()
            inner = self._parse_single_operand()
            return NegatedOperand(operand=inner)

        # Vector operand: {%r0, %r1, ...}
        if tok.kind == TokenKind.LBRACE:
            return self._parse_vector_operand()

        # Parenthesized: either call-style (op1, op2) or constant expression ((128 >> 4) << 16)
        if tok.kind == TokenKind.LPAREN:
            return self._parse_paren_or_const_expr()

        # Address operand: [base+offset]
        if tok.kind == TokenKind.LBRACKET:
            return self._parse_address_operand()

        # Register
        if tok.kind == TokenKind.REGISTER:
            self._advance()
            return RegisterOperand(name=tok.text)

        # Numeric literals
        if tok.kind in (TokenKind.INTEGER, TokenKind.FLOAT):
            self._advance()
            return ImmediateOperand(text=tok.text)

        # Negative immediate: -N
        if tok.kind == TokenKind.MINUS:
            self._advance()
            num_tok = self._peek()
            if num_tok.kind in (TokenKind.INTEGER, TokenKind.FLOAT):
                self._advance()
                return ImmediateOperand(text=f"-{num_tok.text}")
            raise self._error(f"Expected number after '-', got {num_tok.text!r}")

        # Identifier (label reference, symbol, etc.)
        if tok.kind == TokenKind.IDENTIFIER:
            self._advance()
            return LabelOperand(name=tok.text)

        raise self._error(
            f"Expected operand, got {tok.kind.name} ({tok.text!r})"
        )

    def _parse_vector_operand(self) -> VectorOperand:
        self._expect(TokenKind.LBRACE)
        self._skip_newlines_and_comments()
        elements: list[Operand] = []
        if self._peek_kind() != TokenKind.RBRACE:
            elements.append(self._parse_single_operand())
            while self._match(TokenKind.COMMA):
                self._skip_newlines_and_comments()
                elements.append(self._parse_single_operand())
        self._skip_newlines_and_comments()
        self._expect(TokenKind.RBRACE)
        return VectorOperand(elements=tuple(elements))

    def _parse_paren_or_const_expr(self) -> ParenthesizedOperand | ImmediateOperand:
        """Parse either (op1, op2) for call or ((128 >> 4) << 16) as const expr."""
        # Peek ahead to decide: if we see operators like >> << inside,
        # treat the whole thing as a constant expression → ImmediateOperand
        saved_pos = self._pos
        depth = 0
        is_const_expr = False
        scan_pos = self._pos
        while scan_pos < len(self._tokens):
            tok = self._tokens[scan_pos]
            if tok.kind == TokenKind.LPAREN:
                depth += 1
            elif tok.kind == TokenKind.RPAREN:
                depth -= 1
                if depth == 0:
                    break
            elif tok.kind == TokenKind.GREATER and scan_pos + 1 < len(self._tokens) and self._tokens[scan_pos + 1].kind == TokenKind.GREATER:
                is_const_expr = True
                break
            elif tok.kind == TokenKind.LESS and scan_pos + 1 < len(self._tokens) and self._tokens[scan_pos + 1].kind == TokenKind.LESS:
                is_const_expr = True
                break
            scan_pos += 1

        if is_const_expr:
            # Collect everything between matching parens as raw text
            return self._parse_const_expr()
        return self._parse_parenthesized_operand()

    def _parse_const_expr(self) -> ImmediateOperand:
        """Parse a constant expression like ((128 >> 4) << 16) as raw text."""
        start_tok = self._peek()
        parts: list[str] = []
        depth = 0
        while True:
            tok = self._peek()
            if tok.kind == TokenKind.LPAREN:
                depth += 1
            elif tok.kind == TokenKind.RPAREN:
                depth -= 1
            parts.append(tok.leading_whitespace)
            parts.append(tok.text)
            self._advance()
            if depth == 0:
                break
            if tok.kind == TokenKind.EOF:
                raise self._error("Unexpected EOF in constant expression")
        return ImmediateOperand(text="".join(parts).strip())

    def _parse_parenthesized_operand(self) -> ParenthesizedOperand:
        self._expect(TokenKind.LPAREN)
        self._skip_newlines_and_comments()
        elements: list[Operand] = []
        if self._peek_kind() != TokenKind.RPAREN:
            elements.append(self._parse_operand_with_pipe())
            self._skip_newlines_and_comments()
            while self._match(TokenKind.COMMA):
                self._skip_newlines_and_comments()
                elements.append(self._parse_operand_with_pipe())
                self._skip_newlines_and_comments()
        self._expect(TokenKind.RPAREN)
        return ParenthesizedOperand(elements=tuple(elements))

    def _parse_address_operand(self) -> AddressOperand:
        self._expect(TokenKind.LBRACKET)

        # Base: register or identifier
        tok = self._peek()
        if tok.kind == TokenKind.REGISTER:
            base = self._advance().text
        elif tok.kind == TokenKind.IDENTIFIER:
            base = self._advance().text
        else:
            raise self._error(
                f"Expected register or symbol in address, got {tok.text!r}"
            )

        # Offset: everything between base and ]
        # Handles simple (+N, +reg), complex (+N*M, *3+128), and TMA ([base, {coords}])
        offset: str | None = None
        if self._peek_kind() != TokenKind.RBRACKET:
            # Collect until ] (including any commas for TMA syntax)
            raw_parts: list[str] = []
            depth = 0
            while True:
                tok = self._peek()
                if tok.kind == TokenKind.RBRACKET and depth == 0:
                    break
                if tok.kind == TokenKind.EOF:
                    raise self._error("Unexpected EOF in address expression")
                if tok.kind == TokenKind.LBRACE:
                    depth += 1
                elif tok.kind == TokenKind.RBRACE:
                    depth -= 1
                raw_parts.append(tok.leading_whitespace)
                raw_parts.append(tok.text)
                self._advance()
            raw = "".join(raw_parts)
            # Strip leading '+' for simple offsets so they round-trip as "4" not "+4"
            stripped = raw.lstrip()
            if stripped.startswith("+"):
                stripped = stripped[1:].lstrip()
            offset = stripped if stripped else None

        self._expect(TokenKind.RBRACKET)
        return AddressOperand(base=base, offset=offset)

    def _collect_until_rbracket(self) -> str:
        """Collect raw text for complex address sub-expressions (e.g. {%r0, %r1})."""
        parts: list[str] = []
        depth = 0
        while True:
            tok = self._peek()
            if tok.kind == TokenKind.RBRACKET and depth == 0:
                break
            if tok.kind == TokenKind.COMMA and depth == 0:
                break
            if tok.kind == TokenKind.EOF:
                raise self._error("Unexpected EOF in address expression")
            if tok.kind == TokenKind.LBRACE:
                depth += 1
            elif tok.kind == TokenKind.RBRACE:
                depth -= 1
            # Include the token's leading whitespace for fidelity
            parts.append(tok.leading_whitespace)
            parts.append(tok.text)
            self._advance()
        return "".join(parts)


# Function directive names we recognize
_FUNCTION_DIRECTIVE_NAMES = frozenset({
    ".maxnreg",
    ".maxntid",
    ".reqntid",
    ".minnctapersm",
    ".maxnctapersm",
    ".noreturn",
    ".pragma",
    ".explicitcluster",
    ".reqnctapercluster",
    ".cluster_dim",
})

# Directives with space-separated operands (no comma, no semicolon required)
_SPACE_SEPARATED_DIRECTIVES = frozenset({
    ".loc",       # .loc file_idx line col
    ".file",      # .file N "name"
    ".section",   # .section .debug_info
})

# Module-level data directives: .b8 1, .b16 42, .b32 0, .b64 0, .u32 1
_DATA_DIRECTIVES = frozenset({
    ".b8", ".b16", ".b32", ".b64",
    ".u8", ".u16", ".u32", ".u64",
    ".s8", ".s16", ".s32", ".s64",
    ".f32", ".f64",
})
