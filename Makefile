.PHONY: all sync build rebuild test lint lint-check format format-check \
        typecheck qa clean distclean wheel sdist dist check publish-test \
        publish bench bench-layout linux-image test-linux upgrade coverage coverage-html release help

# Default target
all: build

# Sync environment (initial setup, installs dependencies + package)
sync:
	@uv sync

# Sync the editable install (pure Python: edits take effect immediately)
build:
	@uv sync

# Alias for build
rebuild: build

# Run tests
test:
	@uv run pytest tests/ -v

# Lint with ruff (applies fixes)
lint:
	@uv run ruff check --fix src/ tests/

# Lint with ruff (check only, no fixes)
lint-check:
	@uv run ruff check src/ tests/

# Format with ruff
format:
	@uv run ruff format src/ tests/

# Check formatting without modifying files
format-check:
	@uv run ruff format --check src/ tests/

# Type check with mypy
typecheck:
	@uv run mypy src/cignal

# Run a full quality assurance check (non-mutating; mirrors CI)
qa: lint-check format-check typecheck test

# Build wheel
wheel:
	@uv build --wheel

# Build source distribution
sdist:
	@uv build --sdist

# Check distributions with twine
check:
	@uv run twine check dist/*

# Build both wheel and sdist
dist: wheel sdist check

# Publish to TestPyPI
publish-test: check
	@uv run twine upload --repository testpypi dist/*

# Publish to PyPI
publish: check
	@uv run twine upload dist/*

# Upgrade all dependencies
upgrade:
	@uv lock --upgrade
	@uv sync

bench:
	@uv run python bench/run.py

# Both layouts of every Array[K, N] kernel, alternating (docs/dev/soa.md); `CC=gcc make bench-layout`
bench-layout:
	@uv run python bench/layout.py

# Linux (GCC, glibc libm, Debian's Pd) in an Apple `container` VM: `make test-linux ARCH=amd64` runs
# x86-64 under Rosetta. The source is copied in, so the macOS .venv is never used.
ARCH ?= arm64
LINUX_IMAGE = cignal-linux-$(ARCH)

linux-image:
	@container build --arch $(ARCH) -t $(LINUX_IMAGE) -f ci/linux.Containerfile ci

test-linux:
	@container run --rm --arch $(ARCH) -v "$(CURDIR)":/src:ro $(LINUX_IMAGE) sh -c '\
		cp -r /src/src /src/tests /src/thirdparty /src/pyproject.toml /src/uv.lock /src/README.md /src/LICENSE /work/ && \
		uv sync -q && uv run pytest -p no:cacheprovider'

# Run tests with coverage
coverage:
	@uv run pytest tests/ -v --cov=src/cignal --cov-report=term-missing

# Generate HTML coverage report
coverage-html:
	@uv run pytest tests/ -v --cov=src/cignal --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# Create a release (bump version, tag, push)
release:
	@echo "Current version: $$(grep '^version' pyproject.toml | head -1)"
	@read -p "New version: " version; \
	sed "s/^version = .*/version = \"$$version\"/" pyproject.toml > pyproject.toml.tmp && mv pyproject.toml.tmp pyproject.toml; \
	git add pyproject.toml; \
	git commit -m "Bump version to $$version"; \
	git tag -a "v$$version" -m "Release $$version"; \
	echo "Tagged v$$version. Run 'git push && git push --tags' to publish."

# Clean build artifacts
clean:
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf src/*.egg-info/
	@rm -rf .pytest_cache/
	@find . -not -path './.venv/*' -name "*.so" -delete
	@find . -not -path './.venv/*' -name "*.pyd" -delete
	@find . -not -path './.venv/*' -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean everything, including the resolved environment
distclean: clean
	@rm -rf .venv/ .mypy_cache/ .ruff_cache/

# Show help
help:
	@echo "Available targets:"
	@echo "  all          - Sync the environment (default)"
	@echo "  sync         - Sync environment (initial setup)"
	@echo "  build        - Sync the editable install"
	@echo "  rebuild      - Alias for build"
	@echo "  test         - Run tests"
	@echo "  lint         - Lint with ruff (applies fixes)"
	@echo "  lint-check   - Lint with ruff (check only)"
	@echo "  format       - Format with ruff"
	@echo "  format-check - Check formatting without modifying files"
	@echo "  typecheck    - Type check with mypy"
	@echo "  qa           - Run full quality assurance (non-mutating: lint-check, format-check, typecheck, test)"
	@echo "  wheel        - Build wheel distribution"
	@echo "  sdist        - Build source distribution"
	@echo "  dist         - Build both wheel and sdist"
	@echo "  check        - Check distributions with twine"
	@echo "  publish-test - Publish to TestPyPI"
	@echo "  publish      - Publish to PyPI"
	@echo "  upgrade      - Upgrade all dependencies"
	@echo "  bench        - Benchmark performance"
	@echo "  linux-image  - Use apple container to build linux test image"
	@echo "  test-linux   - Test linux platform with container"	
	@echo "  coverage     - Run tests with coverage"
	@echo "  coverage-html - Generate HTML coverage report"
	@echo "  release      - Bump version, tag, and prepare release"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Remove all generated files"
	@echo "  help         - Show this help message"
