"""Download the Pd and SuperCollider plugin headers into the user cache, pinned by commit and SHA-256.

cignal does not ship them: the SuperCollider headers are GPL-2.0-or-later. `thirdparty/` in the repository
holds the same files for tests. To update a host, replace its `thirdparty/` files and the pins here together:
`tests/test_headers.py` checks that they agree.
"""

# ruff: noqa: E501  (the pinned hash table: one file per line)

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


class FetchError(Exception):
    """A download failed, or a file did not match its pinned SHA-256."""


@dataclass(frozen=True)
class Headers:
    name: str  # cache directory name
    base_url: str  # raw file URL at the pinned commit
    files: dict[str, tuple[str, str]]  # path in the cache -> (path upstream, SHA-256)


PD_HEADERS = Headers(
    "pd-0.57-0",
    "https://raw.githubusercontent.com/pure-data/pure-data/f3497d41dd28d4ec1906a0601ff4c0a661c3e5d9/",
    {
        "LICENSE.txt": ("LICENSE.txt", "a552fb28d0392201422eea6811ef8a744da754adfa732da1779167ce01b19489"),
        "m_pd.h": ("src/m_pd.h", "9ad0a74e9ef2d46a4a8c28b3e59b6558a43afefa98455f55cb7af842141746f8"),
    },
)

SC_HEADERS = Headers(
    "supercollider-3.14.1",
    "https://raw.githubusercontent.com/supercollider/supercollider/426edf6d8742e1cc3bd85b51ca0c4e595d37a903/",
    {
        "COPYING": ("COPYING", "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"),
        "common/SC_Alloca.h": ("include/common/SC_Alloca.h", "021a9d3084ee069ffcf2ae166a05607b8c77702da08088fcd4b01cb375465ae8"),
        "common/SC_BoundsMacros.h": ("include/common/SC_BoundsMacros.h", "d92ed72a31bbd4231c091774f38560de0661d88c29c51db908e03d4db4fcaac8"),
        "common/SC_Endian.h": ("include/common/SC_Endian.h", "a0afdd6572fc27bca599d9b132aa9ac54d825acfa1eb25b0b3bee017853ccf8d"),
        "common/SC_Export.h": ("include/common/SC_Export.h", "4a643c3cd1dd3965d8b5910f67b44e040ddb9b65e16491580883c4e33d4eb947"),
        "common/SC_Reply.h": ("include/common/SC_Reply.h", "421cfc8309c306a0a4d9b1115205922c77765ce512cc4c6223084027777e7a4a"),
        "common/SC_StringBuffer.h": ("include/common/SC_StringBuffer.h", "85578d8fb07f4a4430337c0d47f243f830613bc1df42c42328eafd97a504ab80"),
        "common/SC_Types.h": ("include/common/SC_Types.h", "8239fe5b77ec4909d15d743d9c312662d87274f6a20675863e19033be2f8773b"),
        "common/SC_fftlib.h": ("include/common/SC_fftlib.h", "1c425ec4962cfbab034c8cb22c87f47bd5c0be1e2673086de77f55577e3517ce"),
        "common/clz.h": ("include/common/clz.h", "cb7491dc3183fe08d28e888d6649c8c0b0a3435790f5547eb6f0fe1890e53f23"),
        "common/function_attributes.h": ("include/common/function_attributes.h", "8acaea16a78b847e5eb4c404a60a9a3ea17fdcc022ec3d7bc2cfa8c72e738f1b"),
        "common/wintime.h": ("include/common/wintime.h", "69e51830c39ceebaa75013b99e6777053d60b2f9542734766c558db47eab8c35"),
        "plugin_interface/FFT_UGens.h": ("include/plugin_interface/FFT_UGens.h", "9631f60edd20d72f1831ac4680300fc3a260ddcdb3b2f0a0bb58fc4c1bc1e50d"),
        "plugin_interface/Hash.h": ("include/plugin_interface/Hash.h", "c12b9895895241f8eda206e58c116e909ec1d13f7cde637bece1b5ab0ade6fa3"),
        "plugin_interface/SCComplex.h": ("include/plugin_interface/SCComplex.h", "c5e4e6e4be667176940d2b00029d18791e16c76d920117255b6656ce470beb59"),
        "plugin_interface/SC_BelaScope.h": ("include/plugin_interface/SC_BelaScope.h", "ff5b5ad51a923b94af050a917aab9a47b6f4cd3e0d7e0757bf356c80437ffbc0"),
        "plugin_interface/SC_BufGen.h": ("include/plugin_interface/SC_BufGen.h", "fb5297b496424202097d3fb5e3e39cb0c2d478930ecdeb56604c00d789c4f1ef"),
        "plugin_interface/SC_Complex.h": ("include/plugin_interface/SC_Complex.h", "7791dc85d8026eb6a665bd686ff21a91cf6beaaa3949ad54f444007893837a2f"),
        "plugin_interface/SC_Constants.h": ("include/plugin_interface/SC_Constants.h", "07cce20204b8b6580662abf976fd0a57f888b8d4612b77e13c5002121db08e7e"),
        "plugin_interface/SC_DemandUnit.h": ("include/plugin_interface/SC_DemandUnit.h", "7cd7643e58366a1095e7afd77c420b40e6d90c44baa548f3b84d5d2b271372ca"),
        "plugin_interface/SC_FifoMsg.h": ("include/plugin_interface/SC_FifoMsg.h", "66e289e95323049ca527662e1e23c22062f5f58da60f0384d275cec44137c423"),
        "plugin_interface/SC_Graph.h": ("include/plugin_interface/SC_Graph.h", "b0c2394639e47285a63c9e2e9e1b72beeb300f9728a3f1d964f13b3eb94e1ecc"),
        "plugin_interface/SC_InlineBinaryOp.h": ("include/plugin_interface/SC_InlineBinaryOp.h", "d825c9701dbf0ff23ca89154561ecb5f558f7a99db1036f76457520715d613ba"),
        "plugin_interface/SC_InlineUnaryOp.h": ("include/plugin_interface/SC_InlineUnaryOp.h", "ae9b928de6681008b73a3db9a449dda52635b11b10b406d3df5dbccc8f3652b2"),
        "plugin_interface/SC_InterfaceTable.h": ("include/plugin_interface/SC_InterfaceTable.h", "244a85cd45769a591c108f61b099cab69494127749f1ecfd04489016392f54c6"),
        "plugin_interface/SC_Node.h": ("include/plugin_interface/SC_Node.h", "7c71216b956c9687a906d85bf49fc2242f07b455de166482293019800e1a6cb4"),
        "plugin_interface/SC_PlugIn.h": ("include/plugin_interface/SC_PlugIn.h", "ba38dd929ccd8d472dfc475199a98cd03db902392c9780cefc70c53655057750"),
        "plugin_interface/SC_PlugIn.hpp": ("include/plugin_interface/SC_PlugIn.hpp", "ab0d5a5d49a6fd736c80d4d9c80f78affa65940fc042be7f968f3fe35b1ced0d"),
        "plugin_interface/SC_RGen.h": ("include/plugin_interface/SC_RGen.h", "19c68e1b6c34dcde7515fe5df70e5d0c9d89c07474513526a4e55b0cc390f3fc"),
        "plugin_interface/SC_Rate.h": ("include/plugin_interface/SC_Rate.h", "da696482c7e13c2103f74c023edab33f4bde1a20051dc8cafe9a1fd4729c605b"),
        "plugin_interface/SC_SndBuf.h": ("include/plugin_interface/SC_SndBuf.h", "76e83e5f6ee93922ae69c6c83d95c966e7cb17f6b44223b7812695c4065c0a5a"),
        "plugin_interface/SC_Unit.h": ("include/plugin_interface/SC_Unit.h", "aaafcb4a9613a6255b776b95b5c6c81b9b7a6a38440325d25d1590e58f2c920c"),
        "plugin_interface/SC_Wire.h": ("include/plugin_interface/SC_Wire.h", "dfaa713ed8b5dd81dc8120731373df03c113e39490d43c47986037c1463c84e3"),
        "plugin_interface/SC_World.h": ("include/plugin_interface/SC_World.h", "7f212d9117ee15726dca10bd4145e7984cffce6f36475197a4409f9182557132"),
        "plugin_interface/SC_sndfile_stub.h": ("include/plugin_interface/SC_sndfile_stub.h", "46a948c9f53a0284b86c01a86a9822da16daa964ac6dcdc07efb223ecec3bc6a"),
        "plugin_interface/Unroll.h": ("include/plugin_interface/Unroll.h", "d1b3eb4e1e06a9c84f187c068b3ef1e9e9b9db62fbfc0d43cd6d38f14cd04edf"),
        "plugin_interface/sc_msg_iter.h": ("include/plugin_interface/sc_msg_iter.h", "8903a4b3c7f465e92a5c9f7269e2734a254f07ca7aef3512a4e57980bc59e235"),
    },
)  # fmt: skip

HOSTS = {"pd": PD_HEADERS, "sc": SC_HEADERS}


def cache_dir() -> Path:
    """`$XDG_CACHE_HOME/cignal`, else `~/Library/Caches/cignal` on macOS and `~/.cache/cignal` elsewhere."""
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg) / "cignal"
    return Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache") / "cignal"


def headers_dir(headers: Headers) -> Path:
    return cache_dir() / "headers" / headers.name


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_fetched(headers: Headers) -> bool:
    """Every file is in the cache and matches its pinned hash."""
    dest = headers_dir(headers)
    return all((dest / p).is_file() and _sha256((dest / p).read_bytes()) == h for p, (_, h) in headers.files.items())


def fetch(headers: Headers) -> Path:
    """Download `headers` into the cache, unless already there, and return the directory.

    Files are verified and staged in a temporary directory, which then replaces the cache entry whole, so a failed
    download leaves no partial set.
    """
    dest = headers_dir(headers)
    if is_fetched(headers):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / headers.name
        for local, (remote, expected) in headers.files.items():
            url = headers.base_url + remote
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read()
            except (urllib.error.URLError, OSError) as exc:
                raise FetchError(f"{url}: {exc}") from exc
            if (actual := _sha256(data)) != expected:
                raise FetchError(f"{url}: SHA-256 {actual}, expected {expected}")
            (staged / local).parent.mkdir(parents=True, exist_ok=True)
            (staged / local).write_bytes(data)
        try:
            if dest.exists():
                shutil.rmtree(dest)
            staged.replace(dest)
        except OSError as exc:
            if not is_fetched(headers):  # else a concurrent fetch installed the same files first
                raise FetchError(f"{dest}: {exc}") from exc
    return dest


__all__ = ["HOSTS", "PD_HEADERS", "SC_HEADERS", "FetchError", "Headers", "cache_dir", "fetch", "headers_dir"]
