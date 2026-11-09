"""Reuse the local demo credential, frozen bundle and downloaded model weights."""
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
vault = root.parents[1]
secret = root / "server-pytorch/secret.txt"
if not secret.exists():
    shutil.copy2(vault / "outputs/labkit-qa-ensemble-2026-10-19/local-trial/secret.txt", secret)
    secret.chmod(0o600)
bundles = root / "qa-bundles"
if not bundles.exists():
    bundles.symlink_to(vault / "outputs/labkit-qa-ensemble-2026-10-22/demo-package/qa-bundles", target_is_directory=True)
cache = root / "runtime-cache/macos"
cache.mkdir(parents=True, exist_ok=True)
sources = ["cellpose", "torch", "instanseg"]
for name in sources:
    dest = cache / name
    if not dest.exists():
        subprocess.run(["docker", "cp", f"labkit-sai-target-validation:/home/labkit/.cache/{name}", str(dest)], check=True)
    print("Cache ready:", name)
hub = cache / "huggingface/hub"
hub.mkdir(parents=True, exist_ok=True)
for size in ["tiny", "small", "base-plus", "large"]:
    name = "models--facebook--sam2.1-hiera-" + size
    dest = hub / name
    if not dest.exists():
        subprocess.run(["docker", "cp", f"labkit-sai-target-validation:/home/labkit/.cache/huggingface/hub/{name}", str(dest)], check=True)
    print("Cache ready:", name)
