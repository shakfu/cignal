import os
from collections.abc import Callable
from pathlib import Path

import pytest
from support import FIXTURES, Case

import cignal.kernel
from cignal.build import REFERENCE_FLAGS, build_driver, find_pd_include, find_sc_include
from cignal.compiler import Target

Driver = Callable[[Case, str], Path]
cignal.kernel.CHECK_I32 = True  # the reference fails where a stored i32 would wrap in C
THIRDPARTY = Path(__file__).parents[1] / "thirdparty"

# Fall back to the repository's copy of the host headers. Test modules call the finders at import, after this.
for finder, var, sub in ((find_pd_include, "PD_INCLUDE", "pd"), (find_sc_include, "SC_INCLUDE", "supercollider")):
    if finder() is None:
        os.environ[var] = str(THIRDPARTY / sub)
        finder.cache_clear()


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
