# Labkit-SAI segmentation server

FastAPI servers for PyTorch (Cellpose, InstanSeg, Mask R-CNN, SAM2) and
TensorFlow (StarDist). All API routes require an `X-API-Token` header.

## Docker

The image uses each backend's committed Pixi environment. These environments
are Linux x86-64 with CUDA libraries, so Compose selects `linux/amd64`.
On Apple Silicon, Docker uses emulation and inference runs on CPU; the Mac GPU
is not used. The first build downloads a large machine-learning environment.
The image disables OpenMP affinity (`KMP_AFFINITY=disabled`) to avoid an affinity
assertion during PyTorch startup under emulation.

Create an API token before starting (existing tokens can be reused):

```sh
# Run from the repository root. Skip this if secret.txt already exists.
openssl rand -hex 32 > server-pytorch/secret.txt
docker compose up -d --build --wait pytorch
```

PyTorch is available at `http://127.0.0.1:8000`. The token and models are mounted
read-only and excluded from the image. The service runs as UID 10001; the mounted
files must be readable by that user. Downloaded model caches persist in a named
volume. Models may download their official weights on first use, requiring
internet access. The empty `original.*.model` files in `server-MODELS` advertise
the official models; preserve them alongside any custom models.

Check the authenticated API:

```sh
curl --fail -H "X-API-Token: $(cat server-pytorch/secret.txt)" \
  http://127.0.0.1:8000/segmentation_2D/list_available_methods
docker compose logs -f pytorch
docker compose down
```

For StarDist on a native Linux x86-64 host, create its token and start the optional
TensorFlow service. Its image builds, but startup did not complete under x86
emulation on the development Mac; TensorFlow runtime is not verified there.

```sh
openssl rand -hex 32 > server-tensorflow/secret.txt
docker compose --profile tensorflow up -d --build --wait
```

TensorFlow uses `http://127.0.0.1:8001` and its own token and cache volume.
Both servers use one Uvicorn worker because each keeps the active model in memory.

## NVIDIA GPU host

Use a Linux NVIDIA host configured with the NVIDIA Container Toolkit and a driver
compatible with the CUDA libraries in the selected backend's Pixi lockfile.
The current PyTorch environment contains CUDA 12.6; TensorFlow contains CUDA 12.9.

```sh
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build --wait pytorch
# Include StarDist as well:
docker compose -f compose.yaml -f compose.gpu.yaml --profile tensorflow up -d --build --wait
```

The GPU override reserves one NVIDIA GPU per service. See
[Docker's GPU setup documentation](https://docs.docker.com/compose/how-tos/gpu-support/).
The build installs the locked environment without requiring a GPU; the runtime
does not invoke Pixi or need to resolve dependencies.

The API binds to localhost on the host by default. To serve other machines,
set `LABKIT_BIND_ADDRESS=0.0.0.0` in `.env` and provide HTTPS through your reverse
proxy. Ports can be changed with `LABKIT_PYTORCH_PORT` and `LABKIT_TENSORFLOW_PORT`.

Build an image directly:

```sh
docker build --platform linux/amd64 --build-arg BACKEND=pytorch -t labkit-sai-pytorch:local .
# Use BACKEND=tensorflow for StarDist.
```

## Verification

The PyTorch container was built and checked on Docker Desktop on Apple Silicon:
authenticated health check, rejection of unauthenticated requests, model discovery,
binary data round-trip, and the `fakenet.square` binary segmentation endpoint all
passed. Neural model inference and NVIDIA GPU execution require separate checks
on the target host. Both CPU and GPU Compose configurations passed validation.
