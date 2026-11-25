"""`cignal fetch-headers`: pinned downloads into the user cache, verified against `thirdparty/`."""

import dataclasses
import hashlib
import shutil
from pathlib import Path

import pytest
from conftest import THIRDPARTY

from cignal.build import find_sc_include
from cignal.cli import main
from cignal.headers import HOSTS, PD_HEADERS, SC_HEADERS, FetchError, Headers, fetch, headers_dir, is_fetched

SUBDIRS = {PD_HEADERS.name: "pd", SC_HEADERS.name: "supercollider"}


def mirror(headers: Headers, root: Path) -> Headers:
    """`headers` served from a file:// copy of `thirdparty/`, laid out as upstream."""
    for local, (remote, _) in headers.files.items():
        (root / remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(THIRDPARTY / SUBDIRS[headers.name] / local, root / remote)
    return dataclasses.replace(headers, base_url=root.as_uri() + "/")


@pytest.fixture(autouse=True)
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache" / "cignal"


@pytest.mark.parametrize("headers", [PD_HEADERS, SC_HEADERS], ids=lambda h: h.name)
def test_manifest_matches_thirdparty(headers: Headers) -> None:
    root = THIRDPARTY / SUBDIRS[headers.name]
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    assert files == set(headers.files)
    for local, (_, digest) in headers.files.items():
        assert hashlib.sha256((root / local).read_bytes()).hexdigest() == digest, local


@pytest.mark.parametrize("headers", [PD_HEADERS, SC_HEADERS], ids=lambda h: h.name)
def test_fetch_installs_verified_files(tmp_path: Path, cache: Path, headers: Headers) -> None:
    dest = fetch(mirror(headers, tmp_path / "mirror"))
    assert dest == cache / "headers" / headers.name
    assert is_fetched(headers)
    # A second fetch finds the cache complete and downloads nothing.
    assert fetch(dataclasses.replace(headers, base_url=(tmp_path / "gone").as_uri() + "/")) == dest


def test_hash_mismatch_installs_nothing(tmp_path: Path) -> None:
    served = mirror(SC_HEADERS, tmp_path / "mirror")
    with (tmp_path / "mirror/include/plugin_interface/SC_PlugIn.h").open("a") as f:
        f.write("/* tampered */\n")
    with pytest.raises(FetchError, match="SC_PlugIn.h: SHA-256"):
        fetch(served)
    assert not headers_dir(SC_HEADERS).exists()
    assert list(headers_dir(SC_HEADERS).parent.iterdir()) == []  # the staging directory is gone


def test_missing_file_is_a_fetch_error(tmp_path: Path) -> None:
    served = mirror(PD_HEADERS, tmp_path / "mirror")
    (tmp_path / "mirror/src/m_pd.h").unlink()
    with pytest.raises(FetchError, match="m_pd.h"):
        fetch(served)
    assert not headers_dir(PD_HEADERS).exists()


def test_a_concurrent_fetch_that_finishes_first_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other fetch installs between our `rmtree` and `replace`, so the rename meets a non-empty directory."""
    served = mirror(PD_HEADERS, tmp_path / "mirror")
    replace = Path.replace

    def racing(self: Path, target: Path) -> Path:
        shutil.copytree(self, target)
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", racing)
    assert fetch(served) == headers_dir(PD_HEADERS)
    assert is_fetched(PD_HEADERS)


def test_a_failed_install_is_a_fetch_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served = mirror(PD_HEADERS, tmp_path / "mirror")

    def failing(self: Path, target: Path) -> Path:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "replace", failing)
    with pytest.raises(FetchError, match="Permission denied"):
        fetch(served)


def test_cli_fetches_every_host_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key, headers in HOSTS.items():
        monkeypatch.setitem(HOSTS, key, mirror(headers, tmp_path / "mirror"))
    assert main(["fetch-headers"]) == 0
    assert capsys.readouterr().out.split() == [str(headers_dir(PD_HEADERS)), str(headers_dir(SC_HEADERS))]


def test_cli_reports_fetch_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(HOSTS, "pd", dataclasses.replace(PD_HEADERS, base_url=(tmp_path / "gone").as_uri() + "/"))
    assert main(["fetch-headers", "pd"]) == 1


def test_finder_uses_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if any(Path(p, "plugin_interface/SC_PlugIn.h").is_file() for p in ("/usr/include/SuperCollider",
                                                                      "/usr/local/include/SuperCollider")):  # fmt: skip
        pytest.skip("system SuperCollider headers take precedence")
    try:
        with monkeypatch.context() as m:
            m.delenv("SC_INCLUDE", raising=False)  # conftest may have pointed it at thirdparty/
            find_sc_include.cache_clear()
            assert find_sc_include() is None
            fetch(mirror(SC_HEADERS, tmp_path / "mirror"))
            find_sc_include.cache_clear()
            assert find_sc_include() == headers_dir(SC_HEADERS)
    finally:
        find_sc_include.cache_clear()  # later tests resolve again with conftest's environment
