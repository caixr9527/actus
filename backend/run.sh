#!/bin/bash

exec uvicorn app.main:app --host 0.0.0.0 --port 23140 --timeout-graceful-shutdown 5