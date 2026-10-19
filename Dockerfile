# syntax=docker/dockerfile:1
ARG PIXI_VERSION=0.81.0
FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS build
ARG BACKEND=pytorch
WORKDIR /app
COPY LICENSE README.md ./
COPY server-${BACKEND}/pixi.toml server-${BACKEND}/pixi.lock ./server-${BACKEND}/
WORKDIR /app/server-${BACKEND}
# Installation happens without a GPU; CUDA is supplied by the host at runtime.
RUN --mount=type=cache,target=/root/.cache/rattler \
    --mount=type=cache,target=/root/.cache/uv \
    CONDA_OVERRIDE_CUDA=12.9 pixi install --frozen --platform linux-64-cuda
RUN CONDA_OVERRIDE_CUDA=12.9 pixi shell-hook --frozen --shell bash > /app/activate.sh

FROM ubuntu:24.04 AS runtime
ARG BACKEND=pytorch
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/labkit \
    XDG_CACHE_HOME=/home/labkit/.cache \
    HF_HOME=/home/labkit/.cache/huggingface \
    TORCH_HOME=/home/labkit/.cache/torch \
    CELLPOSE_LOCAL_MODELS_PATH=/home/labkit/.cache/cellpose \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 labkit \
    && mkdir -p /home/labkit/.cache /app/server-MODELS \
    && chown -R labkit:labkit /home/labkit
COPY --from=build /app/server-${BACKEND}/.pixi /app/server-${BACKEND}/.pixi
COPY --from=build /app/activate.sh /app/activate.sh
COPY server-pytorch/*.py /app/server-pytorch/
COPY server-tensorflow/*.py /app/server-tensorflow/
COPY segmentation-wrappers/ /app/segmentation-wrappers/
COPY qa_ensemble/ /app/qa_ensemble/
COPY LICENSE /app/LICENSE
COPY --chmod=0755 docker/entrypoint.sh /app/entrypoint.sh
COPY docker/healthcheck.py /app/healthcheck.py
WORKDIR /app/server-${BACKEND}
ENV INSTANSEG_BIOIMAGEIO_PATH=/home/labkit/.cache/instanseg \
    KERAS_HOME=/home/labkit/.cache/keras \
    KMP_AFFINITY=disabled
USER labkit
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /app/entrypoint.sh python /app/healthcheck.py
ENTRYPOINT ["/app/entrypoint.sh"]
# One worker: model selection and the cached model are process-local.
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
