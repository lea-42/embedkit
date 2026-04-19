import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", default=False, help="Run live tests that make real API calls")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: marks tests as requiring real API calls (run with --live)")
