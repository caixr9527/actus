#!/usr/bin/env bash
set -euo pipefail

# 本地开发模式启动（热重载）
exec uvicorn app.main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-23140}" --reload
