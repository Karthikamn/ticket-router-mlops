#!/usr/bin/env bash
set -e
mlflow server \
  --backend-store-uri sqlite:////mlflow/db/mlflow.db \
  --default-artifact-root /mlflow/artifacts \
  --host 0
