"""Write generated C, render host shims, and invoke the C compiler."""

import functools
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from string import Template

from .compiler import CKernel, Target, compile_kernel
from .headers import PD_HEADERS, SC_HEADERS, headers_dir

STRICT_FLAGS = ("-std=c99", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wdouble-promotion", "-Werror")
# Reference builds must not fuse a*b+c, or they stop matching CPython's rounding.
REFERENCE_FLAGS = ("-ffp-contract=off",)


class BuildError(Exception):
    """The C compiler failed. The message includes its stderr."""


def cc() -> str:
    return os.environ.get("CC", "cc")


def cxx() -> str:
    return os.environ.get("CXX", "c++")


def write_kernel(kernel: CKernel, out_dir: Path) -> tuple[Path, Path]:
    """Write the kernel's header and source into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = out_dir / kernel.header_name
    source = header.with_suffix(".c")
    header.write_text(kernel.header)
    source.write_text(kernel.source)
    return header, source


def pd_object_name(kernel: CKernel) -> str:
    """Default Pd object name. It can collide with a Pd built-in (`biquad~`), which then wins silently."""
    return kernel.name.lower() + "~"


def sc_ugen_name(kernel: CKernel) -> str:
    """Default SuperCollider UGen name. The prefix keeps it clear of core UGens such as `Delay1`."""
    return "Cignal" + kernel.name


def render_shim(shim: str, kernel: CKernel, pd_name: str | None = None, sc_name: str | None = None) -> str:
    """Fill a shim template (`driver`, `pd` or `sc`) for `kernel`."""
    filename = "sc.cpp.in" if shim == "sc" else f"{shim}.c.in"
    template = resources.files("cignal.shims").joinpath(filename).read_text()
    pd_name = pd_name or pd_object_name(kernel)
    if not pd_name.endswith("~") or not pd_name[:-1].replace("_", "a").isalnum():
        raise BuildError(f"Pd object name {pd_name!r} must be an identifier followed by `~`")
    sc_name = sc_name or sc_ugen_name(kernel)
    if not (sc_name.isidentifier() and sc_name[0].isupper() and sc_name.isascii()):
        raise BuildError(f"SuperCollider UGen name {sc_name!r} must be an ASCII identifier starting with a capital")
    ftz = resources.files("cignal.shims").joinpath("ftz.h.in").read_text()
    return Template(template).substitute(
        ftz=ftz,
        sc_name=sc_name,
        name=kernel.c_name or kernel.name,
        header=kernel.header_name,
        sample=kernel.sample_ctype,
        n_inputs=kernel.n_inputs,
        n_outputs=kernel.n_outputs,
        n_params=len(kernel.params),
        n_events=len(kernel.events),
        max_event_args=max([1, *(len(types) for _, types in kernel.events)]),
        sc_event_cmds="".join(
            f'    DefineUnitCmd("{sc_name}", "{e}", (UnitCmdFunc)&{sc_name}_event<{k}>);\n'
            for k, (e, _) in enumerate(kernel.events)
        ),
        pd_name=pd_name,
        c_prefix=pd_name.replace("~", "_tilde"),
    )


def run_cc(args: Sequence[str | Path], compiler: str | None = None) -> None:
    command = [compiler or cc(), *map(str, args)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(f"{' '.join(command)}\n{proc.stderr}")


def compile_object(source: Path, flags: Sequence[str] = STRICT_FLAGS) -> Path:
    """Compile one C file to an object next to it."""
    obj = source.with_suffix(".o")
    run_cc([*flags, "-c", source, "-o", obj])
    return obj


def build_driver(
    path: str | Path,
    out_dir: Path,
    kernel: str | None = None,
    target: Target | None = None,
    extra_flags: Sequence[str] = (),
    compiled: CKernel | None = None,
) -> Path:
    """Compile `kernel` and the standalone driver into an executable. `compiled` skips recompiling."""
    ck = compiled or compile_kernel(path, kernel, target)
    _, source = write_kernel(ck, out_dir)
    driver = out_dir / f"{ck.header_name[:-2]}_driver.c"
    driver.write_text(render_shim("driver", ck))
    exe = out_dir / f"{ck.header_name[:-2]}_driver"
    run_cc([*STRICT_FLAGS, *extra_flags, "-I", out_dir, source, driver, "-o", exe, "-lm"])
    return exe


def pd_extension() -> str:
    return ".pd_darwin" if sys.platform == "darwin" else ".pd_linux"


def _mac_apps(pattern: str) -> list[Path]:
    """Bundles in /Applications up to two folders below it, newest name first. A recursive glob took 10 s."""
    root = Path("/Applications")
    return sorted((app for depth in ("", "*/", "*/*/") for app in root.glob(depth + pattern)), reverse=True)


@functools.cache
def find_pd_include() -> Path | None:
    """Locate `m_pd.h`: `$PD_INCLUDE`, then standard install locations, then `cignal fetch-headers`."""
    candidates = [os.environ.get("PD_INCLUDE", ""), "/usr/include/pd", "/usr/local/include/pd"]
    candidates += [str(app / "Contents/Resources/src") for app in _mac_apps("Pd*.app")]
    candidates.append(str(headers_dir(PD_HEADERS)))
    return next((Path(c) for c in candidates if c and (Path(c) / "m_pd.h").is_file()), None)


@functools.cache
def find_pd_binary() -> Path | None:
    """Locate the `pd` executable: `$PD`, then `PATH`, then macOS application bundles."""
    candidates = [os.environ.get("PD", ""), shutil.which("pd") or ""]
    candidates += [str(app / "Contents/Resources/bin/pd") for app in _mac_apps("Pd*.app")]
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


def build_pd_external(
    path: str | Path,
    out_dir: Path,
    kernel: str | None = None,
    pd_include: Path | None = None,
    pd_name: str | None = None,
    extra_flags: Sequence[str] = (),
    compiled: CKernel | None = None,
) -> Path:
    """Compile `kernel` as a Pd external. Returns the path of the loadable binary."""
    include = pd_include or find_pd_include()
    if include is None:
        raise BuildError("m_pd.h not found; run `cignal fetch-headers pd`, set PD_INCLUDE or pass --pd-include")
    ck = compiled or compile_kernel(path, kernel)
    _, source = write_kernel(ck, out_dir)
    pd_name = pd_name or pd_object_name(ck)
    shim = out_dir / f"{pd_name.replace('~', '_tilde')}.c"
    shim.write_text(render_shim("pd", ck, pd_name))
    binary = out_dir / f"{pd_name}{pd_extension()}"
    link = ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]
    run_cc(["-std=c99", "-O2", "-fPIC", "-Wall", *extra_flags, "-I", include, "-I", out_dir, source, shim, *link,
            "-o", binary])  # fmt: skip
    return binary


def sc_extension() -> str:
    return ".scx" if sys.platform == "darwin" else ".so"


@functools.cache
def find_sc_include() -> Path | None:
    """Locate the SuperCollider headers (the directory holding `plugin_interface/` and `common/`).

    `$SC_INCLUDE`, then standard install locations, then `cignal fetch-headers`.
    """
    candidates = [os.environ.get("SC_INCLUDE", ""), "/usr/include/SuperCollider", "/usr/local/include/SuperCollider"]
    candidates.append(str(headers_dir(SC_HEADERS)))
    return next((Path(c) for c in candidates if c and (Path(c) / "plugin_interface/SC_PlugIn.h").is_file()), None)


@functools.cache
def find_scsynth() -> Path | None:
    """Locate `scsynth`: `$SCSYNTH`, then `PATH`, then the macOS application bundle."""
    candidates = [os.environ.get("SCSYNTH", ""), shutil.which("scsynth") or ""]
    candidates += [str(app / "Contents/Resources/scsynth") for app in _mac_apps("SuperCollider*.app")]
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


def sc_class_file(kernel: CKernel, sc_name: str) -> str:
    """The sclang class that makes the UGen usable from SuperCollider code."""
    inputs = [f"in{k}" for k in range(kernel.n_inputs)]
    params = [f"{name} = {default!r}" for name, _, _, default in kernel.params]
    args = ", ".join([*inputs, *(name for name, *_ in kernel.params)])
    signature = ", ".join([*inputs, *params])
    base = "MultiOutUGen" if kernel.n_outputs > 1 else "UGen"
    call = ", ".join(["'audio'", *([args] if args else [])])
    lines = [
        f"// Generated by cignal for kernel {kernel.name}. Do not edit.",
        f"{sc_name} : {base} {{",
        f"    *ar {{ |{signature}|" if signature else "    *ar {",
        f"        ^this.multiNew({call})",
        "    }",
    ]
    if kernel.n_outputs > 1:
        lines += ["    init { |... theInputs|", "        inputs = theInputs;",
                  f"        ^this.initOutputs({kernel.n_outputs}, rate)", "    }"]  # fmt: skip
    return "\n".join([*lines, "}", ""])


def build_sc_plugin(
    path: str | Path,
    out_dir: Path,
    kernel: str | None = None,
    sc_include: Path | None = None,
    sc_name: str | None = None,
    extra_flags: Sequence[str] = (),
    compiled: CKernel | None = None,
) -> Path:
    """Compile `kernel` as a SuperCollider UGen plugin, and write its sclang class file next to it.

    The kernel is compiled as C and the shim as C++: the generated C uses compound literals, which C++ lacks.
    """
    include = sc_include or find_sc_include()
    if include is None:
        raise BuildError(
            "SuperCollider headers not found; run `cignal fetch-headers sc`, set SC_INCLUDE or pass --sc-include"
        )
    ck = compiled or compile_kernel(path, kernel)
    _, source = write_kernel(ck, out_dir)
    sc_name = sc_name or sc_ugen_name(ck)
    shim = out_dir / f"{sc_name}.cpp"
    shim.write_text(render_shim("sc", ck, sc_name=sc_name))
    (out_dir / f"{sc_name}.sc").write_text(sc_class_file(ck, sc_name))
    obj = out_dir / f"{source.stem}.o"
    run_cc(["-std=c99", "-O2", "-fPIC", "-Wall", *extra_flags, "-c", source, "-o", obj])
    binary = out_dir / f"{sc_name}{sc_extension()}"
    link = ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]
    headers: list[str | Path] = ["-I", include / "plugin_interface", "-I", include / "common", "-I", out_dir]
    run_cc(["-std=c++17", "-O2", "-fPIC", "-Wall", *extra_flags, *headers, shim, obj, *link, "-o", binary],
           compiler=cxx())  # fmt: skip
    return binary
