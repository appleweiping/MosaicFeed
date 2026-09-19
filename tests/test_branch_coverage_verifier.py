from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

validate_branch_totals = cast(
    Callable[[dict[str, object]], str],
    runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "verify_branch_coverage.py"))[
        "validate_branch_totals"
    ],
)


def test_branch_gate_uses_covered_arcs_not_combined_coverage() -> None:
    assert validate_branch_totals({"num_branches": 10, "covered_branches": 9}).endswith("90.00%")
    with pytest.raises(ValueError, match="required: 90%"):
        validate_branch_totals({"num_branches": 10, "covered_branches": 8, "percent_covered": 95.0})


@pytest.mark.parametrize(
    "totals",
    [
        {},
        {"num_branches": 0, "covered_branches": 0},
        {"num_branches": True, "covered_branches": 1},
        {"num_branches": 10, "covered_branches": 11},
        {"num_branches": 10, "covered_branches": False},
    ],
)
def test_branch_gate_rejects_unusable_coverage_data(totals: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid or missing"):
        validate_branch_totals(totals)
