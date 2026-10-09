#!/usr/bin/env bash
set -euo pipefail
: "${TS_AUTHKEY:?set TS_AUTHKEY}"
: "${APP_USER:?set APP_USER}"
: "${APP_PASS:?set APP_PASS}"

export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p "$HF_HOME"

echo "[pod] installing deps..."
pip install -q -r requirements.txt

echo "[pod] installing tailscale..."
if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

echo "[pod] joining tailnet..."
tailscale up --authkey="$TS_AUTHKEY" --hostname=nudify

echo "[pod] enabling private https..."
tailscale serve --bg 8000

echo "[pod] starting app..."
pkill -f "run.py" 2>/dev/null || true
nohup python run.py --host 127.0.0.1 --port 8000 --warm > /workspace/app.log 2>&1 &
sleep 3
tail -n 5 /workspace/app.log || true
echo
tailscale serve status || true