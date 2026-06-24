#!/usr/bin/env bash
# 一行连 AutoDL — 使用工作目录下的 SSH 配置
# Usage: bash connect.sh [extra-args]    # 进交互式 shell
#        bash connect.sh "nvidia-smi"    # 远程跑一条命令
exec ssh -F "$(dirname "$0")/.ssh/config" autodl "$@"
