#!/usr/bin/env bash
set +e
for p in $(pgrep -f "[a]800_throughput_tune.py"); do kill -TERM "$p"; done
for p in $(pgrep -f "[s]rc/train.py .*a800_throughput_tune_20260221_154239"); do kill -TERM "$p"; done
sleep 2
for p in $(pgrep -f "[a]800_throughput_tune.py"); do kill -KILL "$p"; done
for p in $(pgrep -f "[s]rc/train.py .*a800_throughput_tune_20260221_154239"); do kill -KILL "$p"; done
ps -eo pid,ppid,cmd | grep -e a800_throughput_tune.py -e src/train.py | grep -v grep || true
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
