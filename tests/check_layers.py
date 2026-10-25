"""Structure audit of the package (DESIGN section 1 dependency direction, section 1 size rule).

Two checks over tg_bridge/**/*.py, both AST based:

1. Import direction. Every import, including ones inside a function, is resolved to a dotted module name of the
   package (relative imports included) and classified by its unit (the first name after tg_bridge). A unit may
   import its own modules, the leaf units (config, paths, clock, logging_ and the package root, which only holds
   __version__), and the units listed in ALLOWED. Any other import is a violation. The matrix is the design's:
   cli -> app -> {daemon, bot, core, feed, agents, store, telegram}; bot -> core; core -> {agents.base, agents.events,
   agents.registry, feed.format, store, telegram}; feed -> {agents.events, telegram}; agents and store and telegram
   import only themselves (plus leaves).
2. Cycles. The module graph of internal imports must be acyclic. A cycle is reported as its module list.
3. Size. Every package module must stay under MAX_LINES lines (design hard ceiling 450).

Run from V3: python3 -I tests/check_layers.py [package_dir]. It prints one line per violation and exits 1 when there
is any.
"""

import ast
import os
import sys
from collections.abc import Mapping, Sequence

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "tg_bridge")
MAX_LINES = 450
LEAVES = frozenset({"config", "paths", "clock", "logging_", ""})
ALLOWED: Mapping[str, frozenset[str]] = {
    "__main__": frozenset({"cli"}),
    "cli": frozenset({"app", "daemon", "telegram"}),
    "app": frozenset({"daemon", "bot", "core", "feed", "agents", "store", "telegram"}),
    "daemon": frozenset({"bot", "core", "feed", "agents", "store", "telegram"}),
    "bot": frozenset({"core", "feed.format", "agents", "store", "telegram"}),
    "core": frozenset({"agents.base", "agents.events", "agents.registry", "feed.format", "store", "telegram"}),
    "feed": frozenset({"agents.events", "telegram"}),
    "agents": frozenset(),
    "store": frozenset(),
    "telegram": frozenset(),
}


def module_name(relative: str) -> str:
    """'agents/codex/session.py' -> 'agents.codex.session'; '__init__.py' of a package names the package."""
    parts = relative[:-3].split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def unit_of(dotted: str) -> str:
    """The unit of a dotted name inside the package: 'agents.base' -> 'agents'; the root -> ''."""
    return dotted.split(".", 1)[0]


def _package_of(dotted: str, is_init: bool) -> str:
    if is_init:
        return dotted
    return dotted.rsplit(".", 1)[0] if "." in dotted else ""


def resolve(node: ast.AST, dotted: str, is_init: bool, modules: frozenset[str]) -> list[str]:
    """The internal dotted module names one import statement reaches. Names of non-module imports are dropped."""
    found: list[str] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "tg_bridge" or alias.name.startswith("tg_bridge."):
                found.append(alias.name[len("tg_bridge"):].lstrip("."))
        return found
    if not isinstance(node, ast.ImportFrom):
        return found
    if node.level == 0:
        if not node.module or not (node.module == "tg_bridge" or node.module.startswith("tg_bridge.")):
            return found
        base = node.module[len("tg_bridge"):].lstrip(".")
    else:
        package = _package_of(dotted, is_init).split(".") if _package_of(dotted, is_init) else []
        up = node.level - 1
        if up > len(package):
            return found
        anchor = package[: len(package) - up]
        base = ".".join(anchor + ([node.module] if node.module else []))
    found.append(base)
    for alias in node.names:
        candidate = ".".join(part for part in (base, alias.name) if part)
        if candidate in modules:
            found.append(candidate)
    return found


def internal_imports(tree: ast.Module, dotted: str, is_init: bool, modules: frozenset[str]) -> list[tuple[int, str]]:
    """(line, target) for every import in the module, nested ones too; target is a dotted name ('' is the root)."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for target in resolve(node, dotted, is_init, modules):
                out.append((node.lineno, target))
    return out


def direction_problem(unit: str, target: str) -> str | None:
    """None when unit may import target, else a message."""
    target_unit = unit_of(target)
    if target_unit in LEAVES or unit == target_unit:
        return None
    for prefix in ALLOWED.get(unit, frozenset()):
        if target == prefix or target.startswith(prefix + "."):
            return None
    return f"unit {unit!r} may not import {target!r}"


def cycles(graph: Mapping[str, set[str]]) -> list[list[str]]:
    """Every strongly connected component with more than one module, or a module importing itself."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[list[str]] = []
    counter = [0]

    def visit(start: str) -> None:
        work = [(start, iter(sorted(graph.get(start, ()))))]
        index[start] = low[start] = counter[0]
        counter[0] += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in graph:
                    continue
                if child not in index:
                    index[child] = low[child] = counter[0]
                    counter[0] += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1 or node in graph.get(node, ()):
                    result.append(sorted(component))

    for name in sorted(graph):
        if name not in index:
            visit(name)
    return result


def _python_files(package_dir: str) -> dict[str, str]:
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
    """Every violation under package_dir, as 'file:line: message' lines, in path order. [] when clean."""
    sources = _python_files(package_dir)
    modules = frozenset(module_name(relative) for relative in sources)
    problems: list[str] = []
    graph: dict[str, set[str]] = {name: set() for name in modules}
    for relative, source in sources.items():
        dotted = module_name(relative)
        is_init = os.path.basename(relative) == "__init__.py"
        tree = ast.parse(source, filename=relative)
        if len(source.splitlines()) > MAX_LINES:
            problems.append(f"{relative}:1: {len(source.splitlines())} lines (limit {MAX_LINES})")
        unit = unit_of(dotted)
        for line, target in internal_imports(tree, dotted, is_init, modules):
            if target in modules:
                graph[dotted].add(target)
            reason = direction_problem(unit, target)
            if reason:
                problems.append(f"{relative}:{line}: {reason}")
    for component in cycles(graph):
        problems.append(f"import cycle: {' -> '.join(component)}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    problems = scan(args[0] if args else PACKAGE_DIR)
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
