# Native macOS test server

Local testing uses Apple Silicon Python and PyTorch MPS, outside Docker. The
Linux CUDA container and HPC deployment remain separate verification targets.

## Start and stop

From the server checkout:

```sh
.venv/bin/python scripts/macos_server.py start
.venv/bin/python scripts/macos_server.py status
.venv/bin/python scripts/macos_server.py stop
```

From the `second-brain` directory, use the executable terminal entry points:

```sh
labkit-evaluation/start-server
labkit-evaluation/stop-server
```

The older `.command` shortcuts remain available for Finder. Startup checks the
authenticated API.

Use `http://127.0.0.1:8003` in the existing **Labkit Cellpose QA** client with access
key **`1234`**. The native launcher supplies this localhost test key through
`LABKIT_LOCAL_TEST_KEY`; the existing `server-pytorch/secret.txt` remains the
default for remote/container deployments. Restart the native server after updating
from the earlier shared-key setup.
Logs are in `runtime/macos-8003.log`. The server binds only to localhost.

Start with `cellpose.original` alone for interactive checking. In **QA + ensemble**,
draw around the central cell, request model/target outputs, run, inspect the preview,
then **Apply result to selected label**. The normal SAI controller puts masks into
labels first; **Submit to segmentation** separately transfers labels to segmentation.

## Devices and model files

The native launcher explicitly requests `mps` for segmentation and frozen QA/ensemble.
It fails on unavailable MPS instead of silently switching the whole model to CPU.
Automatic MPS operator fallback is disabled. `/runtime` and QA capabilities report
the selected devices; requests also record each method's device in their manifest.

An explicit per-model override can be stored in ignored `runtime/macos-devices.json`,
for example `{"maskrcnnv1COCO": "cpu"}` if a future backend needs it. No override is
currently required by the verified seven-model request.

The native launcher uses ignored `runtime/qa-bundles/` when present, otherwise
`qa-bundles/`. The current native registry advertises `corrected-v2` first and
preserves the earlier experimental pipeline. Close and reopen an older QA dialog
to refresh its list, then select `corrected-v2`. The corrected export and adapter
match research scores exactly; see the vault's
`outputs/labkit-corrected-bundle-2026-11-04/README.md` for verification and the
remaining accuracy limitation. Other deployments retain their configured bundle root.
Pretrained segmentation weights are in ignored `runtime-cache/macos/`, initially
copied from Docker. Model advertisements remain in `server-MODELS/`.
InstanSeg's native package is 0.1.2 (0.1.0 was unavailable for this Python); its
official model resolved to version 0.1.1. Do not treat its results as an exact
reproduction of the earlier Linux environment.

## Installation and verification

`scripts/setup_macos.py` creates an isolated Python 3.12 `.venv`, installs
`requirements-macos.txt`, and installs official SAM2 source at a fixed commit with
the CUDA extension disabled. It can be bootstrapped with the vault's existing Python.
`requirements-macos.lock.txt` records the installed environment. The vault environment
is unchanged. Run `scripts/prepare_macos_files.py` once to reuse local demo files
and Docker model caches.

Verification on 27 October:

- 26 server checks passed; `pip check` found no broken requirements.
- Authenticated native Cellpose + frozen QA/ranking returned a nonempty 256×256 mask
  in 4.99 seconds on the first checked request.
- All seven demo methods executed on MPS. Six target masks were accepted; InstanSeg's
  output missed the explicit foreground point and was correctly excluded. QA,
  ranking and fusion completed, returning a nonempty mask. Full request: 286.51 s.
- These are runtime checks on an existing development example, not new accuracy or
  speedup claims. The returned masks and manifests are in
  `labkit-evaluation/native-macos/` in the vault.

The previous Docker test container is stopped, preserved, and still owns its cached
weights. Port 8003 now belongs to the native server. David separately confirmed
the native QA-dialog preview/apply workflow on 4 November. Corrected-bundle
integration subsequently passed 29 server tests, exact saved-table parity and
a real four-model MPS request. Linux GPU/HPC verification remains open.
