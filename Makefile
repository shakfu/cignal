.PHONY: test test-linux linux-image lint type-check bench

test:
	uv run pytest

lint:
	uv run ruff check src tests

type-check:
	uv run mypy

bench:
	uv run python bench/run.py

# Linux (GCC, glibc libm, Debian's Pd) in an Apple `container` VM: `make test-linux ARCH=amd64` runs
# x86-64 under Rosetta. The source is copied in, so the macOS .venv is never used.
ARCH ?= arm64
LINUX_IMAGE = cignal-linux-$(ARCH)

linux-image:
	container build --arch $(ARCH) -t $(LINUX_IMAGE) -f ci/linux.Containerfile ci

test-linux:
	container run --rm --arch $(ARCH) -v "$(CURDIR)":/src:ro $(LINUX_IMAGE) sh -c '\
		cp -r /src/src /src/tests /src/pyproject.toml /src/uv.lock /src/README.md /src/LICENSE /work/ && \
		uv sync -q && uv run pytest -p no:cacheprovider'
