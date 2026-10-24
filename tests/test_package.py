"""Smoke test: the package installs and imports."""

import student_performance


def test_package_imports() -> None:
    assert student_performance.__version__ == "0.1.0"
