"""Enforce the roadmap's branch-only gate, not coverage.py's combined percentage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from coverage import Coverage

MINIMUM_BRANCH_PERCENT = 90


def validate_branch_totals(totals: dict[str, object]) -> str:
    branches = totals.get("num_branches")
    covered = totals.get("covered_branches")
    if (
        isinstance(branches, bool)
        or not isinstance(branches, int)
        or branches <= 0
        or isinstance(covered, bool)
        or not isinstance(covered, int)
        or not 0 <= covered <= branches
    ):
        raise ValueError("coverage report has invalid or missing branch totals")
    percentage = 100 * covered / branches
    result = f"pure branch coverage: {covered}/{branches} = {percentage:.2f}%"
    if 100 * covered < MINIMUM_BRANCH_PERCENT * branches:
        raise ValueError(f"{result}; required: {MINIMUM_BRANCH_PERCENT}%")
    return result


def main() -> None:
    coverage = Coverage()
    coverage.load()
    with tempfile.TemporaryDirectory(prefix="mosaicfeed-branch-coverage-") as directory:
        report_path = Path(directory) / "coverage.json"
        coverage.json_report(outfile=str(report_path))
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("totals"), dict):
        raise ValueError("coverage report does not have a totals object")
    print(validate_branch_totals(payload["totals"]))


if __name__ == "__main__":
    main()
