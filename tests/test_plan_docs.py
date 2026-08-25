"""The plan docs must describe the code that exists.

The plans in docs/superpowers/plans/ embed code under a `# path/to/file.py`
first line. That code is an ABBREVIATED proposal, never a copy of the file,
so this cannot compare them line by line. What it does compare is the part
a reader acts on: every function and class the plan declares must exist in
the named file with the same signature.

One-directional on purpose. The plan is an excerpt, so the real file may
declare anything the plan omits; what it may not do is contradict it. A
signature that has silently moved on is worse than an omission -- someone
reads the plan, believes it, and writes a call that does not typecheck.

Blocks must be valid Python. The "Existing interfaces you consume" sections
are stubs (`def f(a, b) -> T: ...`) for exactly this reason: a block that
cannot be parsed cannot be checked, and a checker that skips what it cannot
parse quietly stops guarding the blocks most likely to rot.
"""

import ast
import glob
import re

import pytest

PLAN_GLOB = "docs/superpowers/plans/*.md"
BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)
LABEL_RE = re.compile(r"#\s*((?:dhvani|tests|scripts)/[\w/]+\.py)$")


def _blocks():
    """(doc, line_no, path, source) for every file-labelled section.

    Sections, not blocks: the "Existing interfaces you consume" listings put
    several files in ONE fence, each introduced by its own `# path.py` line.
    Attributing a whole fence to whichever file happened to be named first
    would check router and store declarations against pipeline.py and report
    failures that are purely the checker's confusion.

    A fence with no `# path.py` line at all is prose or shell and is skipped.
    """
    for doc in sorted(glob.glob(PLAN_GLOB)):
        text = open(doc, encoding="utf-8").read()
        for match in BLOCK_RE.finditer(text):
            base = text[:match.start()].count("\n") + 2
            path, start, buf = None, 0, []
            for offset, line in enumerate(match.group(1).split("\n")):
                label = LABEL_RE.match(line.strip())
                if label:
                    if path:
                        yield doc, base + start, path, "\n".join(buf)
                    path, start, buf = label.group(1), offset, []
                elif path is not None:
                    buf.append(line)
            if path:
                yield doc, base + start, path, "\n".join(buf)


def _arg_names(node):
    """Parameter names plus whether each carries a default.

    Names and default-presence, not default VALUES: a changed default is a
    behaviour question this file has no way to judge, while a renamed,
    added, or removed parameter always breaks a caller written from the doc.
    """
    args = node.args
    positional = args.posonlyargs + args.args
    first_default = len(positional) - len(args.defaults)
    out = [(a.arg, i >= first_default) for i, a in enumerate(positional)]
    if args.vararg:
        out.append(("*" + args.vararg.arg, False))
    out += [(a.arg, args.kw_defaults[i] is not None)
            for i, a in enumerate(args.kwonlyargs)]
    if args.kwarg:
        out.append(("**" + args.kwarg.arg, False))
    return out


def _signatures(tree):
    """qualname -> parameters, for module-level and class-level defs only.

    Nested helpers are deliberately out of scope: they are not a contract
    anyone can call from outside the file.
    """
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = _arg_names(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}.{sub.name}"] = _arg_names(sub)
    return found


def test_the_harness_actually_finds_the_plan_code():
    """Anti-vacuity. If the fence or label pattern ever stops matching, every
    check below passes over an empty list and this file guards nothing."""
    found = list(_blocks())
    assert len(found) >= 50, f"only {len(found)} labelled blocks found"
    assert len({doc for doc, _, _, _ in found}) >= 3, "expected all three plans"


def test_every_labelled_block_names_a_file_that_exists():
    missing = [f"{doc}:{line} -> {path}"
               for doc, line, path, _ in _blocks()
               if not glob.glob(path)]
    assert not missing, "plan references files that no longer exist:\n" + "\n".join(missing)


def test_every_labelled_block_is_parseable_python():
    """A block that cannot be parsed cannot be signature-checked. Interface
    summaries therefore use stub bodies (`-> T: ...`) rather than a prose
    notation, so they are covered like everything else."""
    broken = []
    for doc, line, path, src in _blocks():
        try:
            ast.parse(src)
        except SyntaxError as exc:
            broken.append(f"{doc}:{line} ({path}) line {exc.lineno}: {exc.msg}")
    assert not broken, (
        "plan code blocks must be valid Python so they can be checked; "
        "write interface listings as stubs, e.g. `def f(a, b) -> T: ...`:\n"
        + "\n".join(broken)
    )


def test_plan_signatures_match_the_code():
    problems = []
    for doc, line, path, src in _blocks():
        try:
            declared = _signatures(ast.parse(src))
        except SyntaxError:
            continue  # reported by test_every_labelled_block_is_parseable_python
        actual = _signatures(ast.parse(open(path, encoding="utf-8").read()))
        for name, params in declared.items():
            if name not in actual:
                problems.append(
                    f"{doc}:{line}\n"
                    f"    declares {path}::{name}, which does not exist"
                )
            elif actual[name] != params:
                problems.append(
                    f"{doc}:{line}\n"
                    f"    {path}::{name}\n"
                    f"      plan: {[p[0] for p in params]}\n"
                    f"      code: {[p[0] for p in actual[name]]}"
                )
    assert not problems, (
        f"{len(problems)} plan signature(s) no longer match the code:\n\n"
        + "\n".join(problems)
    )
