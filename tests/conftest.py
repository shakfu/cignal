from collections.abc import Callable
from pathlib import Path

import pytest
from support import FIXTURES, Case

from cignal.build import REFERENCE_FLAGS, build_driver
from cignal.compiler import Target

Driver = Callable[[Case, str], Path]


@pytest.fixture(scope="session")
def driver(tmp_path_factory: pytest.TempPathFactory) -> Driver:
    """Build (once per session) the driver for a case's kernel. `variant` is "float" or "double"."""
    cache: dict[tuple[str, str, str], Path] = {}

    def get(case: Case, variant: str) -> Path:
        key = (case.fixture, case.kernel, variant)
        if key not in cache:
            out = tmp_path_factory.mktemp(f"{case.kernel}_{variant}")
            flags = REFERENCE_FLAGS if variant == "double" else ()
            cache[key] = build_driver(FIXTURES / case.fixture, out, case.kernel, Target(f32_ctype=variant), flags)
        return cache[key]

    return get
