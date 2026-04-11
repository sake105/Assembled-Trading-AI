# syntax=docker/dockerfile:1.6
#
# Assembled-Trading-AI container image (Sprint 4 / Plan C25).
#
# Multi-stage build:
#   - builder: installs pinned deps into a virtualenv at /opt/venv
#   - runtime: copies the venv and application source into a slim image
#
# Design notes
# ------------
# * No secrets are baked in. API keys and broker credentials must be
#   injected at run time via env vars or a mounted .env file that lives
#   outside the image.
# * The runtime stage drops privileges to an unprivileged ``trader``
#   user.
# * The container defaults to the paper-trading scheduler entrypoint but
#   accepts an override command at run time.
# * Both ``requirements.lock`` (pinned) and ``pyproject.toml`` are
#   copied so editable installs still pick up the project metadata.

# ----------------------------------------------------------------------
# Stage 1: builder
# ----------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System packages needed to build wheels for scipy/numpy on slim base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy lockfile first for better layer caching.
COPY requirements.lock ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.lock

# Copy project metadata + source and install the package itself.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-deps .

# ----------------------------------------------------------------------
# Stage 2: runtime
# ----------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    ASSEMBLED_OUTPUT_DIR=/app/output

# Minimal runtime libraries for numpy/scipy wheels that link against BLAS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libopenblas0 \
        liblapack3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 trader

WORKDIR /app

# Copy the prepared virtualenv and the application layout.
COPY --from=builder /opt/venv /opt/venv
COPY --chown=trader:trader src/        ./src/
COPY --chown=trader:trader scripts/    ./scripts/
COPY --chown=trader:trader configs/    ./configs/
COPY --chown=trader:trader pyproject.toml ./

# Writable output dir (volume mount recommended).
RUN mkdir -p /app/output /app/data && chown -R trader:trader /app/output /app/data

USER trader

# Simple health probe: verify the package imports cleanly.
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import assembled_core" || exit 1

# Default entrypoint: paper-trading scheduler. Override at run time for
# ad-hoc commands (pytest, scripts, interactive shells).
ENTRYPOINT ["python"]
CMD ["scripts/paper_trading_scheduler.py"]
