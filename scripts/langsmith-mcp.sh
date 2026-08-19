#!/usr/bin/env bash
# Launch the official LangSmith MCP server (stdio) with creds from .env.
# Do not print secrets.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi
if [ -z "${LANGSMITH_API_KEY:-}" ] && [ -n "${LANGCHAIN_API_KEY:-}" ]; then
  export LANGSMITH_API_KEY="$LANGCHAIN_API_KEY"
fi
if [ -z "${LANGSMITH_PROJECT:-}" ] && [ -n "${LANGCHAIN_PROJECT:-}" ]; then
  export LANGSMITH_PROJECT="$LANGCHAIN_PROJECT"
fi
if [ -z "${LANGSMITH_API_KEY:-}" ]; then
  echo "LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is not set in $ROOT/.env" >&2
  exit 1
fi
UVX="$(command -v uvx || true)"
if [ -z "$UVX" ]; then
  for p in /opt/homebrew/bin/uvx "$HOME/.local/bin/uvx"; do
    [ -x "$p" ] && UVX="$p" && break
  done
fi
if [ -z "$UVX" ]; then
  echo "uvx not found. Install with: brew install uv" >&2
  exit 1
fi
exec "$UVX" langsmith-mcp-server "$@"
