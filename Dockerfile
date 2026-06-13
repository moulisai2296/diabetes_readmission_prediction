# Serving image for the readmission-risk API.
# Build context expects artifacts/ to exist (run: python -m src.api.export_model).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1) Dependencies first (cached unless the lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Application code + serving artifacts + governance reports (viewed at /governance)
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY governance/ ./governance/

ENV PATH="/app/.venv/bin:$PATH" \
    ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000
# Container-level liveness check hitting the app's own endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
