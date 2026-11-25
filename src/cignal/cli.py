"""Command-line interface: `cignal check`, `cignal build` and `cignal fetch-headers`."""

import argparse
import sys
from pathlib import Path

from .build import BuildError, build_driver, build_pd_external, build_sc_plugin, write_kernel, write_manifest
from .compiler import compile_kernel
from .diagnostics import CompileError, Diagnostic, diagnostics_to_json
from .headers import HOSTS, FetchError, fetch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cignal", description="Compile Python DSP kernels to C99.")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="validate a kernel without writing output")
    check.add_argument("file", type=Path)
    check.add_argument("--kernel")
    check.add_argument("--format", choices=("text", "json"), default="text")

    build = commands.add_parser("build", help="write C for a kernel, and optionally build a shim")
    build.add_argument("file", type=Path)
    build.add_argument("--kernel")
    build.add_argument("-o", "--output", type=Path, default=Path("build"))
    build.add_argument("--shim", choices=("none", "driver", "pd", "sc"), default="none")
    build.add_argument("--pd-include", type=Path, help="directory containing m_pd.h")
    build.add_argument("--pd-name", help="Pd object name (default: lowercased kernel name + `~`)")
    build.add_argument("--sc-include", type=Path, help="SuperCollider header directory (holds plugin_interface/)")
    build.add_argument("--sc-name", help="SuperCollider UGen name (default: Cignal + kernel name)")
    build.add_argument("--prefix", default="", help="prefix for every generated C identifier, e.g. `mylib_`")
    build.add_argument("--manifest", action="store_true", help="also write <stem>.cignal.json, the kernel API as JSON")

    fetch_headers = commands.add_parser("fetch-headers", help="download host plugin headers into the user cache")
    fetch_headers.add_argument("hosts", nargs="*", choices=sorted(HOSTS), help="default: all")

    args = parser.parse_args(argv)
    if args.command == "build":
        for flag, shim in [("pd_include", "pd"), ("pd_name", "pd"), ("sc_include", "sc"), ("sc_name", "sc")]:
            if getattr(args, flag) is not None and args.shim != shim:
                parser.error(f"--{flag.replace('_', '-')} needs --shim {shim}")
    if args.command == "fetch-headers":
        try:
            for host in args.hosts or sorted(HOSTS):
                print(fetch(HOSTS[host]))
        except FetchError as exc:
            print(f"cignal: fetch failed: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        if args.command == "check":
            ck = compile_kernel(args.file, args.kernel)
            if args.format == "json":
                print(diagnostics_to_json(ck.warnings))
            else:
                _report(ck.warnings, args.file)
            return 0
        ck = compile_kernel(args.file, args.kernel, prefix=args.prefix)
        _report(ck.warnings, args.file)
        if args.shim == "driver":
            print(build_driver(args.file, args.output, compiled=ck))
        elif args.shim == "pd":
            print(build_pd_external(args.file, args.output, pd_include=args.pd_include, pd_name=args.pd_name,
                                    compiled=ck))  # fmt: skip
        elif args.shim == "sc":
            print(build_sc_plugin(args.file, args.output, sc_include=args.sc_include, sc_name=args.sc_name,
                                  compiled=ck))  # fmt: skip
        else:
            for written in write_kernel(ck, args.output):
                print(written)
        if args.manifest:
            print(write_manifest(ck, args.output))
        return 0
    except CompileError as exc:
        if args.command == "check" and args.format == "json":
            print(diagnostics_to_json(exc.diagnostics))
        else:
            print(exc, file=sys.stderr)
        return 1
    except (BuildError, ValueError) as exc:
        print(f"cignal: build failed:\n{exc}", file=sys.stderr)
        return 1


def _report(warnings: list[Diagnostic], path: Path) -> None:
    for warning in warnings:
        print(warning.format(str(path)), file=sys.stderr)


__all__ = ["main"]
