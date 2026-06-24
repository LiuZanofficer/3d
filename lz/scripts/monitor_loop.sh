#!/bin/bash
# Monitor loop for E4 training: every 30 minutes append a snapshot.
# Usage:
#   nohup bash monitor_loop.sh /root/autodl-tmp/lz_outputs/E4_v2 50 1800 \
#     > /root/autodl-tmp/lz_outputs/E4_v2/monitor.log 2>&1 &
#
# Args:
#   $1 = run-dir (default /root/autodl-tmp/lz_outputs/E4_v2)
#   $2 = max-epochs (default 50)
#   $3 = interval seconds (default 1800 = 30 min)

RUN_DIR="${1:-/root/autodl-tmp/lz_outputs/E4_v2}"
MAX_EPOCHS="${2:-50}"
INTERVAL="${3:-1800}"

PYTHON="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/llz/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRINTER="${SCRIPT_DIR}/print_status.py"
STATUS_FILE="${RUN_DIR}/status.md"

mkdir -p "$RUN_DIR"
echo "# E4 monitor — run-dir=${RUN_DIR}, every ${INTERVAL}s, max_epochs=${MAX_EPOCHS}" > "$STATUS_FILE"
echo "# Started at $(date '+%Y-%m-%d %H:%M:%S')" >> "$STATUS_FILE"
echo >> "$STATUS_FILE"

while true; do
    {
        "$PYTHON" "$PRINTER" --run-dir "$RUN_DIR" --max-epochs "$MAX_EPOCHS"
    } >> "$STATUS_FILE" 2>&1 || true

    # If training has finished (no Lightning python process), exit the loop.
    if ! pgrep -f "src/train.py.*${RUN_DIR}" > /dev/null 2>&1; then
        echo "==========  monitor: training process gone, exiting ==========" >> "$STATUS_FILE"
        break
    fi

    sleep "$INTERVAL"
done
