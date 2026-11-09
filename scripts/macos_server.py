"""Start/status/stop the native, localhost-only Labkit server."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
LOCAL_TEST_KEY = "1234"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=["start", "status", "stop"])
parser.add_argument("--port", type=int, default=8003)
args = parser.parse_args()
runtime = root / "runtime"
runtime.mkdir(exist_ok=True)
receipt = runtime / f"macos-{args.port}.json"


def running():
    if not receipt.exists():
        return None
    info = json.loads(receipt.read_text())
    try:
        os.kill(info["pid"], 0)
        return info
    except ProcessLookupError:
        return None


info = running()
if args.action == "stop":
    if info:
        # Confirm the recorded PID still belongs to our server before signalling it.
        command = subprocess.check_output(["ps", "-p", str(info["pid"]), "-o", "command="], text=True)
        if "uvicorn server:app" not in command or f"--port {args.port}" not in command:
            raise SystemExit("Recorded PID no longer belongs to this server; left untouched")
        for side in info.get("side", []):
            try:
                if "side_server.py" in subprocess.check_output(["ps", "-p", str(side["pid"]), "-o", "command="], text=True):
                    os.kill(side["pid"], signal.SIGTERM)
                    print("Side server stopped:", side["group"], side["pid"])
            except (subprocess.CalledProcessError, ProcessLookupError):
                pass
        os.kill(info["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                os.kill(info["pid"], 0)
            except ProcessLookupError:
                receipt.unlink(missing_ok=True)
                print("Native server stopped:", info["pid"])
                break
            time.sleep(.1)
        else:
            raise SystemExit("Native server is still stopping; wait before starting again")
    else:
        print("Native server is not running on port", args.port)
elif args.action == "status":
    print(json.dumps(info or {"running": False, "port": args.port}, indent=2))
elif info:
    print("Already running:", json.dumps(info))
else:
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"Port {args.port} is already occupied; existing service left untouched")
    cache = root / "runtime-cache/macos"
    native_bundles = root / "runtime/qa-bundles"
    if not native_bundles.is_dir():
        native_bundles = root / "qa-bundles"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "LABKIT_DEVICE": "mps", "LABKIT_QA_DEVICE": "mps",
           "LABKIT_LOCAL_TEST_KEY": LOCAL_TEST_KEY,
           "LABKIT_QA_BUNDLES": str(native_bundles), "LABKIT_MODELS_DIR": str(root / "server-MODELS"),
           "HF_HOME": str(cache / "huggingface"), "TORCH_HOME": str(cache / "torch"),
           "CELLPOSE_LOCAL_MODELS_PATH": str(cache / "cellpose"),
           "INSTANSEG_BIOIMAGEIO_PATH": str(cache / "instanseg"), "MPLCONFIGDIR": str(cache / "matplotlib"),
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    # Explicit per-model overrides may be selected after compatibility testing.
    config = root / "runtime/macos-devices.json"
    if config.exists():
        for model, device in json.loads(config.read_text()).items():
            env["LABKIT_DEVICE_" + model.upper()] = device
    # Optional side servers for models in separate environments (server-extra/side_server.py).
    # runtime/side-backends.json: {"microsam": {"python": "/path/env/bin/python", "port": 8011}, ...}
    side = []
    side_config = runtime / "side-backends.json"
    if side_config.exists():
        import secrets
        token_file = runtime / "side-token.txt"
        token_file.write_text(secrets.token_hex(16))
        token_file.chmod(0o600)
        side_env = {k: v for k, v in env.items() if not k.startswith("LABKIT_")}
        for group, spec in json.loads(side_config.read_text()).items():
            with (runtime / f"side-{group}.log").open("a") as log:
                proc = subprocess.Popen([spec["python"], str(root / "server-extra/side_server.py"), group,
                                         "--port", str(spec["port"]), "--token-file", str(token_file)],
                                        cwd=root / "server-extra", env=side_env, stdout=log, stderr=log, start_new_session=True)
            side.append({"group": group, "pid": proc.pid, "url": f"http://127.0.0.1:{spec['port']}"})
        env["LABKIT_SIDE_BACKENDS"] = ",".join(f"{s['group']}={s['url']}" for s in side)
        env["LABKIT_SIDE_TOKEN_FILE"] = str(token_file)
    logfile = runtime / f"macos-{args.port}.log"
    with logfile.open("a") as log:
        process = subprocess.Popen([str(root / ".venv/bin/python"), "-m", "uvicorn", "server:app",
                                    "--host", "127.0.0.1", "--port", str(args.port), "--workers", "1"],
                                   cwd=root / "server-pytorch", env=env, stdout=log, stderr=log,
                                   start_new_session=True)
    info = {"pid": process.pid, "url": f"http://127.0.0.1:{args.port}", "log": str(logfile), "side": side}
    receipt.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    import requests
    headers = {"X-API-Token": LOCAL_TEST_KEY}
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"Server startup failed; see {logfile}")
        try:
            response = requests.get(info["url"] + "/runtime", headers=headers, timeout=2)
            if response.ok:
                print("Ready:", json.dumps(response.json()))
                break
        except requests.RequestException:
            pass
        time.sleep(.5)
    else:
        raise SystemExit(f"Server is still starting; see {logfile}")
