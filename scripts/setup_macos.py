"""Bootstrap the isolated native runtime; execute with the vault's Python."""
from pathlib import Path
import os
import subprocess

root = Path(__file__).resolve().parents[1]
python = root / ".venv/bin/python"
if not python.exists():
    subprocess.run(["/opt/homebrew/bin/python3.12", "-m", "venv", str(root / ".venv")], check=True)
pip = [str(python), "-m", "pip", "--disable-pip-version-check"]
subprocess.run([*pip, "install", "--retries", "0", "--timeout", "20", "-r", str(root / "requirements-macos.txt")], check=True)
# Official SAM2 source. CUDA extensions are unnecessary on Metal; torch is already installed.
env = {**os.environ, "SAM2_BUILD_CUDA": "0"}
subprocess.run([*pip, "install", "--retries", "0", "--timeout", "20", "--no-build-isolation",
                "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"], env=env, check=True)
