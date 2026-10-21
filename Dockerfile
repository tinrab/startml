# syntax=docker/dockerfile:1
# Reproducible training image. PyTorch wheels bundle CUDA, so a plain Python base is enough;
# the host only needs the NVIDIA driver + container toolkit.
#   docker build -t startml .
#   docker run --rm --gpus all --ipc=host -v ./data:/app/data -v ./outputs:/app/outputs startml train trainer=gpu

FROM python:3.14-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["train"]
