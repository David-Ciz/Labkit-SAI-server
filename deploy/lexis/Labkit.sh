#!/bin/bash
# Labkit-SAI batch job for LEXIS AI Training / HEAppE GenericCommandTemplate.
#
# Set this script as the computation block's userScriptPath. HEAppE's generic.sh
# expands the template parameters and sources this file inside the job directory,
# which holds ./input (the block's input datasets) and ./output (the output dataset).
#
# Environment variables (block "Environment Variables" / hyperparameters):
#   MODE            train | segment                       (default: segment)
#   LABKIT_SIF      path to labkit-sai.sif in project storage (required)
#   METHODS         comma-separated pool, e.g. cellpose.original,sam2.original-large
#   PIPELINE_ID     segment: bundle to apply; train: name of the new bundle
#   BASE_BUNDLE     train: bundle providing QA initialisation (default 10-corrected-v2)
#   RAW_ENCODING    preserve | minmax_uint8 (use minmax_uint8 for 16-bit/float images)
#   FLAG_BELOW      segment: QA flag threshold (default 0.5)
#   MAX_CELLS       segment: cells per image, 0 = all
#   EXTRA_ARGS      anything else for labkit_job.py
#   LABKIT_CACHE    writable model-weight cache in project storage (default ./.labkit-cache);
#                   pre-populate it on a login node if compute nodes have no internet access
set -euo pipefail

: "${LABKIT_SIF:?Set LABKIT_SIF to the labkit-sai.sif path in project storage}"
command -v apptainer >/dev/null 2>&1 || ml apptainer

mkdir -p ./output
args=(--mode "${MODE:-segment}" --input-root /mnt/input --output-root /mnt/output)
[ -n "${METHODS:-}" ] && args+=(--methods "$METHODS")
[ -n "${PIPELINE_ID:-}" ] && args+=(--pipeline-id "$PIPELINE_ID")
[ -n "${BASE_BUNDLE:-}" ] && args+=(--base-bundle "$BASE_BUNDLE")
[ -n "${RAW_ENCODING:-}" ] && args+=(--raw-encoding "$RAW_ENCODING")
[ -n "${FLAG_BELOW:-}" ] && args+=(--flag-below "$FLAG_BELOW")
[ -n "${MAX_CELLS:-}" ] && args+=(--max-cells "$MAX_CELLS")

cache="${LABKIT_CACHE:-$PWD/.labkit-cache}"; mkdir -p "$cache"
binds=(--bind "$PWD/output:/mnt/output" --bind "$cache:/home/labkit/.cache")
[ -d ./input ] && binds+=(--bind "$PWD/input:/mnt/input:ro")

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
apptainer run --nv --cleanenv -H "$PWD" --pwd /app/server-pytorch "${binds[@]}" "$LABKIT_SIF" \
    python /app/deploy/lexis/labkit_job.py "${args[@]}" ${EXTRA_ARGS:-}
