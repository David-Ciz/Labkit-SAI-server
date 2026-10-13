"""Check the authenticated API without printing its token."""
from pathlib import Path
from urllib.request import Request, urlopen

token = (Path.cwd() / "secret.txt").read_text(encoding="utf-8").strip()
request = Request("http://127.0.0.1:8000/", headers={"X-API-Token": token})
with urlopen(request, timeout=5) as response:
    if response.status != 200:
        raise SystemExit(1)
