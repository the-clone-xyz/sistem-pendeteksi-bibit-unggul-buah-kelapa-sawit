#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8501}"
LOG_DIR="${CODESPACE_SERVICE_LOG_DIR:-/tmp/sawit-codespace-services}"
STREAMLIT_LOG="$LOG_DIR/streamlit.log"
CLOUDFLARED_LOG="$LOG_DIR/cloudflared.log"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-$HOME/.local/bin/cloudflared}"
LOCAL_ENV_FILE="${CLOUDFLARE_TUNNEL_ENV_FILE:-$ROOT_DIR/.env}"
LOCK_DIR="/tmp/sawit-codespace-services.lock"

mkdir -p "$LOG_DIR" "$(dirname "$CLOUDFLARED_BIN")"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Codespace services startup is already running."
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

is_running() {
  local pattern="$1"

  if command -v pgrep >/dev/null 2>&1; then
    pgrep -f "$pattern" >/dev/null 2>&1
    return $?
  fi

  ps -ef | grep -E "$pattern" | grep -v grep >/dev/null 2>&1
}

read_token_from_env_file() {
  local env_file="$1"
  local line
  local token

  [ -f "$env_file" ] || return 0
  line="$(
    grep -E '^(export[[:space:]]+)?CLOUDFLARE_TUNNEL_TOKEN=' "$env_file" 2>/dev/null \
      | tail -n 1
  )"
  [ -n "$line" ] || return 0

  line="${line#export }"
  token="${line#CLOUDFLARE_TUNNEL_TOKEN=}"
  token="${token%\"}"
  token="${token#\"}"
  token="${token%\'}"
  token="${token#\'}"
  printf '%s' "$token"
}

start_streamlit() {
  if is_running "[s]treamlit run app.py.*--server.port ${APP_PORT}"; then
    echo "Streamlit already running on port ${APP_PORT}."
    return 0
  fi

  (
    cd "$ROOT_DIR" || exit 1
    nohup python3 -m streamlit run app.py \
      --server.address "$APP_HOST" \
      --server.port "$APP_PORT" \
      >> "$STREAMLIT_LOG" 2>&1 &
  )

  echo "Started Streamlit on ${APP_HOST}:${APP_PORT}. Log: ${STREAMLIT_LOG}"
}

cloudflared_download_url() {
  case "$(uname -m)" in
    x86_64|amd64)
      printf '%s' "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
      ;;
    aarch64|arm64)
      printf '%s' "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
      ;;
    *)
      return 1
      ;;
  esac
}

ensure_cloudflared() {
  local url
  local tmp_path

  if [ -x "$CLOUDFLARED_BIN" ]; then
    return 0
  fi

  url="$(cloudflared_download_url)" || {
    echo "Unsupported CPU architecture for cloudflared: $(uname -m)" >&2
    return 1
  }

  tmp_path="${CLOUDFLARED_BIN}.tmp"
  echo "Downloading cloudflared to ${CLOUDFLARED_BIN}..."
  curl -L --fail --show-error --silent -o "$tmp_path" "$url" || return 1
  chmod +x "$tmp_path"
  mv "$tmp_path" "$CLOUDFLARED_BIN"
}

start_cloudflare_tunnel() {
  local token="${CLOUDFLARE_TUNNEL_TOKEN:-}"

  if is_running "[c]loudflared tunnel.*run"; then
    echo "Cloudflare tunnel already running."
    return 0
  fi

  if [ -z "$token" ]; then
    token="$(read_token_from_env_file "$LOCAL_ENV_FILE")"
  fi

  if [ -z "$token" ]; then
    echo "Cloudflare tunnel skipped: set CLOUDFLARE_TUNNEL_TOKEN as a Codespaces secret or in ${LOCAL_ENV_FILE}."
    return 0
  fi

  ensure_cloudflared || {
    echo "Cloudflare tunnel skipped: failed to install cloudflared." >&2
    return 1
  }

  (
    cd "$ROOT_DIR" || exit 1
    TUNNEL_TOKEN="$token" nohup "$CLOUDFLARED_BIN" tunnel --no-autoupdate run \
      >> "$CLOUDFLARED_LOG" 2>&1 &
  )

  echo "Started Cloudflare tunnel. Log: ${CLOUDFLARED_LOG}"
}

start_streamlit
start_cloudflare_tunnel
