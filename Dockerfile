# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build the React console (Vite single-page app).
# ---------------------------------------------------------------------------
FROM node:22-slim AS webbuild
WORKDIR /web
# BASE_PATH controls the SPA's public sub-path. Default "/" (served at site root,
# e.g. local Docker at localhost:8080). Behind a reverse proxy under a sub-path,
# pass BASE_PATH=/agents/ so assets, router basename, and API calls carry the prefix.
ARG BASE_PATH=/
ENV BASE_PATH=${BASE_PATH}
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — Python runtime that serves the API and the built console.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRACEGUARD_ARTIFACT_DIR=/data/runs \
    TRACEGUARD_WEB_DIR=/app/web \
    TRACEGUARD_PAPER_PDF=/app/usenixsecurity2026.pdf \
    TRACEGUARD_HOST=0.0.0.0 \
    TRACEGUARD_PORT=8080

WORKDIR /app

# tpm2-tools lets the in-VM app read the guest vTPM (NV index 0x01400001) for real
# Azure SEV-SNP/TDX attestation. It is inert off Azure: attestation.py probes for a
# vTPM device first and falls back to the honest "simulated" verdict when absent.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tpm2-tools ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY usenixsecurity2026.pdf ./usenixsecurity2026.pdf
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 artifact \
    && mkdir -p /data/runs /data/runtime \
    && chown -R artifact:artifact /data

# The built single-page console (index.html + hashed assets/).
COPY --from=webbuild /web/dist /app/web

USER artifact
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=12s --retries=5 \
  CMD python -c "import json,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8080/api/health',timeout=2); raise SystemExit(0 if r.status == 200 and json.load(r).get('status') in {'ok','healthy'} else 1)"

CMD ["traceguard", "serve", "--host", "0.0.0.0", "--port", "8080"]
