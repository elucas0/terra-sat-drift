# --- Build Stage ---
FROM ghcr.io/astral-sh/uv:latest AS builder

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy files using absolute paths
COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

# Run sync by explicitly pointing to the project directory
RUN uv sync --frozen --no-install-project --no-dev --project /app

# Copy source code to absolute path
COPY . /app

# Final sync
RUN uv sync --frozen --no-dev --project /app


# --- Runtime Stage ---
FROM python:3.11-slim-bookworm

# Copy the source code to an absolute path
COPY --from=builder /app /app

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Run using the absolute path to the module
CMD ["python", "-m", "terra_sat_drift"]