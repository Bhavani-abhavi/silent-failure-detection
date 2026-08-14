"""Enforces two architectural boundaries.

1. **`drift_core` is domain-agnostic.** The core must not know it is running
   on credit data. If a new domain would require changing it, that is a
   design failure and it goes in the findings log.

2. **`estimation` cannot reach labels.** This one is not cosmetic. The
   label-free estimators are validated by comparing their output against
   matured labels, and an estimator with an import path to
   `domains.finance.lending_club.default_label_within_horizon` could consume
   the answer it is being graded on. Nothing at runtime would reveal it — the
   estimates would simply be excellent — so the guard has to be structural.

The import-linter contracts in pyproject.toml cover both in CI. These tests
duplicate them deliberately so `pytest` alone catches a violation.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIFT_CORE = ROOT / "drift_core"
ESTIMATION = ROOT / "estimation"
FORBIDDEN_ROOTS = {
    "domains", "pipeline", "reports", "model", "estimation", "backtest",
}
ESTIMATION_FORBIDDEN_ROOTS = {"domains", "backtest", "reports"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "module_path",
    sorted(DRIFT_CORE.glob("*.py")),
    ids=lambda p: p.name,
)
def test_drift_core_imports_nothing_domain_specific(module_path):
    violations = _imported_roots(module_path) & FORBIDDEN_ROOTS
    assert not violations, (
        f"{module_path.name} imports {sorted(violations)}. drift_core must "
        f"stay domain-agnostic — if a domain needs something from the core, "
        f"the core grows a general parameter, not a domain import."
    )


@pytest.mark.parametrize(
    "module_path",
    sorted(ESTIMATION.glob("*.py")),
    ids=lambda p: p.name,
)
def test_estimation_cannot_reach_labels(module_path):
    """A label-free estimator that can import a label source is not one.

    Reference labels are legitimate and are *passed in* as arrays by the
    caller — that is the production reality, where history is labelled and
    the current window is not. What must not exist is a path by which this
    package could fetch labels itself.
    """
    violations = _imported_roots(module_path) & ESTIMATION_FORBIDDEN_ROOTS
    assert not violations, (
        f"{module_path.name} imports {sorted(violations)}. An estimator that "
        f"can reach the labels it is scored against invalidates component B, "
        f"and the failure would look like unusually good results."
    )


def test_backtest_is_the_only_place_truth_and_estimates_meet():
    """The scoring lives downstream of both, by construction.

    `estimation` must not import `backtest`; the dependency runs the other
    way. If it were ever reversed, an estimator could read the onset
    definition it is being timed against.
    """
    for module_path in sorted(ESTIMATION.glob("*.py")):
        assert "backtest" not in _imported_roots(module_path)


BANNED_VOCABULARY = [
    "readmission", "mimic", "icd", "diagnosis", "patient",
    "lending_club", "fico", "loan_amnt", "borrower", "default_rate",
    "conversion", "click_through", "sku", "cart",
]


def _code_identifiers_and_literals(path: Path) -> set[str]:
    """Names and string literals that are actually part of the code.

    Docstrings and comments are deliberately excluded. The core's docstrings
    *should* name domain concepts — they explain what the core refuses to
    know about, and "no column semantics, no 'readmission'" is exactly the
    kind of sentence that belongs there. What must not appear is a domain
    term in a real identifier or a runtime string.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    # Every bare string-expression statement is documentation: module,
    # class, and function docstrings, plus the attribute docstrings used
    # after enum members in types.py. None are runtime values, so exclude
    # them by node identity (comparing text fails — ast.get_docstring
    # dedents, so the cleaned string no longer matches the raw node).
    doc_node_ids = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in doc_node_ids:
                found.add(node.value)
    return {item.lower() for item in found}


@pytest.mark.parametrize(
    "module_path",
    sorted(DRIFT_CORE.glob("*.py")),
    ids=lambda p: p.name,
)
def test_drift_core_has_no_domain_vocabulary(module_path):
    """Catches the subtler failure: no domain imports, but hardcoded column
    names or domain concepts leaking into the supposedly generic core."""
    code_text = _code_identifiers_and_literals(module_path)
    offenders = [
        term for term in BANNED_VOCABULARY if any(term in item for item in code_text)
    ]
    assert not offenders, (
        f"domain vocabulary in {module_path.name} code (not docstrings): "
        f"{offenders}"
    )
