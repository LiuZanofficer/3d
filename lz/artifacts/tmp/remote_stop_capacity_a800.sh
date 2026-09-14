#!/usr/bin/env bash
set -e
OUT_FILE=/tmp/sc_capacity_current_out_a800.txt
PID_FILE=/tmp/sc_capacity_current_pid_a800.txt
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  kill -15 "$PID" 2>/dev/null || true
  sleep 3
  kill -9 "$PID" 2>/dev/null || true
fi
pkill -f "sc_full_capacity_check_run_a800_" 2>/dev/null || true
sleep 2
ps -eo pid,etime,cmd | grep -E 'sc_capacity_pipeline.py|src/train.py .*sc_memory_check|src/eval.py .*sc_memory_check' | grep -v grep || echo STOPPED
if [ -f "$OUT_FILE" ]; then
  echo "LAST_OUT=$(cat "$OUT_FILE")"
fi
