"""NPY header bounds, encoding and noncompletion contract."""

from typing import Any, Dict

import pytest

from npy_header_cases import CASES, run_case


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_numpy_header_contract(case: Dict[str, Any]) -> None:
    run_case(case)
