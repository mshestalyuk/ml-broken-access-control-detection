# Scanner image: same environment locally and in CI. Model is mounted, not baked in.
FROM python:3.14-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY data ./data
RUN uv sync --locked --no-dev

RUN useradd --create-home scanner && git config --system --add safe.directory '*'
USER scanner

ENTRYPOINT ["bacdetect"]
CMD ["--help"]
