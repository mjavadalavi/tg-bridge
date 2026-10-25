"""AST scan for module-level mutable state in the package (DESIGN section 2, globals policy).

Rule. At module level (also inside module-level if, try and with blocks, never inside def or class):
- every assignment binds a name in UPPERCASE (a leading underscore and dunder names are allowed) to an immutable value:
  a literal, a tuple of allowed values, an arithmetic expression over them, a plain name or attribute used as an alias,
  a typing subscript, or a call from IMMUTABLE_CALLS (frozenset, MappingProxyType, re.compile, timedelta, object, pure
  os.path helpers) or a frozen dataclass of the package (a dataclass(frozen=True) class defined in the scanned tree);
- a CamelCase name bound to a type expression (a subscript, a `|` union or a plain reference) is a type alias and is allowed;
- a name that holds a mutable module value is not an alias target (no `X = _CACHE` after `_CACHE = {}`);
- no function declares `global`, because that rebinds module state at run time.
Anything else (a list, dict or set, a call such as Lock() or dict(), a comprehension, a lambda) is a violation.

Run from V3: python3 -I tests/check_globals.py [package_dir]. It prints one line per violation and exits 1 when there is any.
"""

import ast
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "tg_bridge")
UPPERCASE_NAME = re.compile(r"^(?:__[a-z_]+__|_?[A-Z][A-Z0-9_]*)$")
CAMEL_NAME = re.compile(r"^_?[A-Z][A-Za-z0-9]*$")
IMMUTABLE_CALLS = frozenset({
    "frozenset", "compile", "re.compile", "NewType", "TypeVar", "ParamSpec", "TypeAliasType", "MappingProxyType",
    "timedelta", "object", "os.path.join", "os.path.dirname", "os.path.basename", "os.path.realpath",
    "os.path.abspath", "os.path.expanduser",
})
CONTAINER_BLOCKS = (ast.If, ast.Try, ast.With)


def _dotted(node: ast.expr) -> str:
    """The dotted name of a call target (os.path.join), or '' when it is not a plain name chain."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    return ""


def frozen_dataclass_names(tree: ast.Module) -> frozenset[str]:
    """The classes of one module decorated @dataclass(frozen=True) (with or without other keywords)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or _dotted(decorator.func) not in ("dataclass", "dataclasses.dataclass"):
                continue
            if any(kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                   for kw in decorator.keywords):
                names.add(node.name)
    return frozenset(names)


def _immutable(node: ast.expr, mutable_names: frozenset[str], frozen: frozenset[str]) -> bool:
    """True when the value can never be changed in place."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id not in mutable_names
    if isinstance(node, ast.Attribute):
        return True
    if isinstance(node, ast.Subscript):
        return _immutable(node.value, mutable_names, frozen)
    if isinstance(node, ast.Tuple):
        return all(_immutable(item, mutable_names, frozen) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        return _immutable(node.operand, mutable_names, frozen)
    if isinstance(node, ast.BinOp):
        return _immutable(node.left, mutable_names, frozen) and _immutable(node.right, mutable_names, frozen)
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        return name in IMMUTABLE_CALLS or name.rsplit(".", 1)[-1] in frozen
    return False


def _type_expression(node: ast.expr) -> bool:
    """A value that names a type (a type alias), not a value."""
    return isinstance(node, (ast.Subscript, ast.Name, ast.Attribute)) or (
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
    )


def _module_assignments(body: Sequence[ast.stmt]) -> Iterable[ast.Assign | ast.AnnAssign]:
    """Every assignment that runs at import time: the module body, descending into if/try/with, never into def or class."""
    for stmt in body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            yield stmt
        elif isinstance(stmt, CONTAINER_BLOCKS):
            blocks = [stmt.body, getattr(stmt, "orelse", []), getattr(stmt, "finalbody", [])]
            blocks += [handler.body for handler in getattr(stmt, "handlers", [])]
            for block in blocks:
                yield from _module_assignments(block)


def _targets(stmt: ast.Assign | ast.AnnAssign) -> list[ast.expr]:
    return list(stmt.targets) if isinstance(stmt, ast.Assign) else [stmt.target]


def check_source(source: str, filename: str = "<source>", frozen: frozenset[str] = frozenset()) -> list[str]:
    """The violations in one module's source, as 'file:line: message' lines. frozen names the package's frozen
    dataclasses (see scan); a call to one of them is an immutable value."""
    tree = ast.parse(source, filename=filename)
    problems: list[str] = []
    mutable: set[str] = set()
    for stmt in _module_assignments(tree.body):
        if stmt.value is None:
            continue
        for target in _targets(stmt):
            if not isinstance(target, ast.Name):
                problems.append(f"{filename}:{stmt.lineno}: module-level target is not a plain name")
                continue
            alias = CAMEL_NAME.match(target.id) is not None and _type_expression(stmt.value)
            if not (UPPERCASE_NAME.match(target.id) or alias):
                problems.append(f"{filename}:{stmt.lineno}: module-level name {target.id!r} is not UPPERCASE")
            if not _immutable(stmt.value, frozenset(mutable), frozen):
                problems.append(f"{filename}:{stmt.lineno}: {target.id!r} is bound to a mutable value")
                mutable.add(target.id)
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            problems.append(f"{filename}:{node.lineno}: 'global {', '.join(node.names)}' rebinds module state")
    return problems


def _python_files(package_dir: str) -> Mapping[str, str]:
    """Relative path -> source text of every .py file under package_dir (__pycache__ skipped), in path order."""
    sources: dict[str, str] = {}
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(root, name)
                with open(path, encoding="utf-8") as handle:
                    sources[os.path.relpath(path, package_dir)] = handle.read()
    return sources


def scan(package_dir: str = PACKAGE_DIR) -> list[str]:
    """Every violation under package_dir, in path order. [] when clean. Frozen dataclasses are collected across the
    whole tree first, so a value built by one module's frozen dataclass counts as immutable in every module."""
    sources = _python_files(package_dir)
    frozen: set[str] = set()
    for source in sources.values():
        frozen |= frozen_dataclass_names(ast.parse(source))
    problems: list[str] = []
    for relative, source in sources.items():
        problems.extend(check_source(source, relative, frozenset(frozen)))
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    problems = scan(args[0] if args else PACKAGE_DIR)
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
