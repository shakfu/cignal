# Linux test environment: GCC, glibc libm, Pd, SuperCollider's scsynth and plugin headers, Python 3.13, uv.
# Build and run with `make linux-image test-linux [ARCH=amd64]`.
FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libc6-dev make puredata-core puredata-dev \
       supercollider-server supercollider-dev supercollider-language python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
# Debian's sclang is the Qt build; these let it start without a display.
ENV QT_QPA_PLATFORM=offscreen QTWEBENGINE_DISABLE_SANDBOX=1 QT_QUICK_BACKEND=software LIBGL_ALWAYS_SOFTWARE=1 \
    QTWEBENGINE_CHROMIUM_FLAGS="--no-sandbox --disable-gpu"
WORKDIR /work
