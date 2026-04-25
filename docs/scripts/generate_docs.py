#!/usr/bin/env python3
"""Generate API and examples pages for the docs site.

The API pages are emitted as plain Markdown so they render both on the
docs site and directly on GitHub. That means we do not rely on
``mkdocstrings`` directives in the checked-in ``docs/api/*.md`` files.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
API_DIR = DOCS / "api"
EXAMPLES_DIR = DOCS / "examples"
PYPTX_DIR = ROOT / "pyptx"
EXAMPLE_SRC_DIR = ROOT / "examples"


def _module_doc(path: Path) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return ast.get_docstring(module) or ""


def _summary(path: Path) -> str:
    doc = _module_doc(path)
    parts = [part.strip() for part in doc.split("\n\n") if part.strip()]
    return parts[0] if parts else "No module docstring yet."


def _title_from_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _doc_block(doc: str | None) -> list[str]:
    if not doc:
        return ["No docstring yet.", ""]
    return [textwrap.dedent(doc).strip(), ""]


def _format_signature(name: str, obj: Any, *, bound: bool = False) -> str | None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    if bound and params and params[0].name in {"self", "cls"}:
        sig = sig.replace(parameters=params[1:])
    return f"{name}{sig}"


def _render_constant(value: Any) -> str:
    rendered = repr(value)
    if len(rendered) > 120:
        rendered = rendered[:117] + "..."
    return rendered


def _slug(name: str) -> str:
    return name.lower().replace("_", "-")


def _top_level_names(module: Any, source: Path) -> list[str]:
    explicit = getattr(module, "__all__", None)
    if explicit:
        return [name for name in explicit if hasattr(module, name)]

    tree = ast.parse(source.read_text(encoding="utf-8"))
    names: list[str] = []
    seen: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_") and node.name not in seen and hasattr(module, node.name):
                names.append(node.name)
                seen.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if not name.startswith("_") and name not in seen and hasattr(module, name):
                        names.append(name)
                        seen.add(name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if not name.startswith("_") and name not in seen and hasattr(module, name):
                names.append(name)
                seen.add(name)
    return names


def _member_specs(obj: Any, *, namespace: bool) -> list[tuple[str, str, Any, str | None, bool]]:
    owner = obj.__class__ if namespace else obj
    members: list[tuple[str, str, Any, str | None, bool]] = []
    for name, raw in owner.__dict__.items():
        if name.startswith("_"):
            continue

        if isinstance(raw, property):
            members.append((name, "property", None, inspect.getdoc(raw), False))
            continue

        if isinstance(raw, staticmethod):
            fn = raw.__func__
            members.append((name, "staticmethod", fn, inspect.getdoc(fn), False))
            continue

        if isinstance(raw, classmethod):
            fn = raw.__func__
            members.append((name, "classmethod", fn, inspect.getdoc(fn), True))
            continue

        if inspect.isfunction(raw):
            members.append((name, "method", raw, inspect.getdoc(raw), True))
            continue

        value = getattr(obj if namespace else owner, name, raw)
        if inspect.isclass(value):
            members.append((name, "class", value, inspect.getdoc(value), False))
        else:
            members.append((name, "attribute", value, inspect.getdoc(value), False))
    return members


def _render_members(lines: list[str], title: str, obj: Any, *, namespace: bool) -> None:
    members = _member_specs(obj, namespace=namespace)
    if not members:
        return

    lines.extend([f"### {title}", ""])
    for name, kind, value, doc, bound in members:
        heading = f"#### `{name}`"
        if kind in {"method", "staticmethod", "classmethod"} and value is not None:
            sig = _format_signature(name, value, bound=bound)
            if sig:
                heading = f"#### `{sig}`"
        lines.extend([heading, "", f"- Kind: `{kind}`", ""])
        if kind == "attribute" and value is not None:
            lines.extend([f"- Value: `{_render_constant(value)}`", ""])
        lines.extend(_doc_block(doc))


def _render_symbol(lines: list[str], module_name: str, name: str, value: Any) -> None:
    lines.extend([f'<a id="{_slug(name)}"></a>', "", f"## `{name}`", ""])

    if inspect.isfunction(value):
        sig = _format_signature(name, value)
        lines.extend([f"- Kind: `function`", ""])
        if sig:
            lines.extend(["```python", sig, "```", ""])
        lines.extend(_doc_block(inspect.getdoc(value)))
        return

    if inspect.isclass(value):
        sig = _format_signature(name, value)
        lines.extend([f"- Kind: `class`", ""])
        if sig:
            lines.extend(["```python", f"class {sig}", "```", ""])
        lines.extend(_doc_block(inspect.getdoc(value)))
        _render_members(lines, "Members", value, namespace=False)
        return

    if inspect.ismodule(value):
        lines.extend([f"- Kind: `module`", "", f"- Target: `{value.__name__}`", ""])
        lines.extend(_doc_block(inspect.getdoc(value)))
        return

    lines.extend([f"- Kind: `namespace`", "", f"- Type: `{type(value).__name__}`", ""])
    lines.extend(_doc_block(inspect.getdoc(value)))
    _render_members(lines, "Members", value, namespace=True)


def generate_api_pages() -> list[str]:
    modules = ["pyptx"]
    modules += [
        f"pyptx.{path.stem}"
        for path in sorted(PYPTX_DIR.glob("*.py"))
        if path.name != "__init__.py" and not path.name.startswith("_")
    ]

    index_lines = [
        "# API Reference",
        "",
        "These pages are generated from the `pyptx/` package and module docstrings.",
        "",
    ]

    for module_name in modules:
        if module_name == "pyptx":
            source = PYPTX_DIR / "__init__.py"
            slug = "pyptx"
        else:
            slug = module_name.split(".")[-1]
            source = PYPTX_DIR / f"{slug}.py"

        module = importlib.import_module(module_name)
        doc = _module_doc(source)
        summary = _summary(source)
        public_names = _top_level_names(module, source)

        index_lines.append(f"- [`{module_name}`]({slug}.md): {summary}")

        lines = [
            f"# `{module_name}`",
            "",
            "> This page is generated from source docstrings and public symbols.",
            "",
        ]
        lines.extend(_doc_block(doc))

        if public_names:
            lines.extend(["## Public API", ""])
            for name in public_names:
                lines.append(f"- [`{name}`](#{name.lower().replace('_', '-')})")
            lines.append("")

        for name in public_names:
            _render_symbol(lines, module_name, name, getattr(module, name))

        _write(API_DIR / f"{slug}.md", "\n".join(lines).rstrip() + "\n")

    _write(API_DIR / "index.md", "\n".join(index_lines) + "\n")
    return modules


def _render_example_page(path: Path, title: str) -> str:
    """Render one example page. Uses the full module docstring as the
    narrative (not just the first paragraph), adds a source-link
    header, and puts the full source under a collapsible section so
    the narrative stays scannable."""
    source = path.read_text(encoding="utf-8")
    full_doc = _module_doc(path).strip()
    rel_path = path.relative_to(ROOT).as_posix()
    github_url = f"https://github.com/patrick-toulme/pyptx/blob/dev/{rel_path}"

    parts = [
        f"# {title}",
        "",
        f"[:material-github: View on GitHub]({github_url}){{ .md-button }} ",
        f"[:material-file-code: `{rel_path}`]({github_url}){{ .md-button }}",
        "",
    ]
    if full_doc:
        parts.extend(["## Overview", "", full_doc, ""])
    parts.extend([
        "## Source",
        "",
        '??? example "Full source"',
        "",
    ])
    # Indent the code block by 4 spaces so it nests inside the ??? block.
    for line in ["```python", *source.rstrip().splitlines(), "```"]:
        parts.append("    " + line if line else "")
    parts.append("")
    return "\n".join(parts)


def generate_example_pages() -> list[str]:
    examples = [
        path for path in sorted(EXAMPLE_SRC_DIR.rglob("*.py"))
        if path.name != "__init__.py"
    ]
    for path in examples:
        rel = path.relative_to(EXAMPLE_SRC_DIR)
        slug = rel.with_suffix("")
        title = " / ".join(_title_from_name(part) for part in slug.parts)
        page = _render_example_page(path, title)
        _write(EXAMPLES_DIR / f"{slug.as_posix()}.md", page)

    # examples/index.md is hand-authored as a gallery with hero cards
    # (see docs/examples/index.md). Do not overwrite it.
    return [path.relative_to(EXAMPLE_SRC_DIR).with_suffix("").as_posix() for path in examples]


def main() -> None:
    modules = generate_api_pages()
    examples = generate_example_pages()
    print(f"generated {len(modules)} api pages and {len(examples)} example pages")


if __name__ == "__main__":
    main()
