#!/bin/bash
# Start the interactive Labkit-SAI server as an engine of the LEXIS/EXA4MIND
# AI Inference Service (persistent GPU job; the service renews it while in use).
#
# Inference-service template parameters:
#   $1  HTTPPort   port the service tunnels to on the compute node
#   $2  model ID   QA/ensemble pipeline that must be available (e.g. labkit-pool-v1)
#   $3  ImagePath  path to labkit-sai.sif
# Environment:
#   LABKIT_TOKEN_FILE  file with the server's X-API-Token (project storage, mode 600; required)
#   LABKIT_BUNDLES     QA bundle directory (default: the image's qa-bundles)
#   LABKIT_RECORD_DIR  optional: store interactive training records here
#   LABKIT_MODELS      optional: folder with fine-tuned segmenters (e.g. cellpose.<id>-ft from a train job)
#   LABKIT_CACHE       writable model-weight cache in project storage (default: \$HOME/.cache/labkit-sai)
#
# Startup loads every model of the pipeline's pool first (LABKIT_WARMUP), so the first
# user request does not pay the cold start. /metrics answers without the token (liveness
# only). All other routes need X-API-Token.
set -euo pipefail
PORT="${1:?HTTPPort}"; MODEL_ID="${2:?model ID}"; SIF="${3:?ImagePath}"
: "${LABKIT_TOKEN_FILE:?Set LABKIT_TOKEN_FILE}"
command -v apptainer >/dev/null 2>&1 || ml apptainer

cache="${LABKIT_CACHE:-$HOME/.cache/labkit-sai}"; mkdir -p "$cache"
# The token stays in a file (never on a command line visible to other users of the node).
binds=(--bind "$LABKIT_TOKEN_FILE:/mnt/secret/token:ro" --bind "$cache:/home/labkit/.cache")
envs=(--env LABKIT_TOKEN_FILE=/mnt/secret/token --env LABKIT_PUBLIC_METRICS=1
      --env LABKIT_QA_DEVICE=cuda --env LABKIT_REQUIRE_PIPELINE="$MODEL_ID" --env LABKIT_WARMUP=1)
if [ -n "${LABKIT_BUNDLES:-}" ]; then
    binds+=(--bind "$LABKIT_BUNDLES:/mnt/bundles:ro"); envs+=(--env LABKIT_QA_BUNDLES=/mnt/bundles)
fi
if [ -n "${LABKIT_MODELS:-}" ]; then
    binds+=(--bind "$LABKIT_MODELS:/mnt/models:ro"); envs+=(--env LABKIT_EXTRA_MODELS_DIR=/mnt/models)
fi
if [ -n "${LABKIT_RECORD_DIR:-}" ]; then
    mkdir -p "$LABKIT_RECORD_DIR"
    binds+=(--bind "$LABKIT_RECORD_DIR:/mnt/records"); envs+=(--env LABKIT_RECORD_DIR=/mnt/records)
fi

home="$(mktemp -d)"
exec apptainer run --nv --cleanenv -H "$home" --pwd /app/server-pytorch "${binds[@]}" "${envs[@]}" "$SIF" \
    python -m uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
