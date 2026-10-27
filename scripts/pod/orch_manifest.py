"""Manifest of the archives sent to Orchestron with scripts/upload_artifacts.sh (termination rule, bootstrap.sh).

Usage: orch_manifest.py <manifest.json> [<dir>=<rc> ...]
Lists every file in each directory with its size and SHA-256 and the exit code of the upload of that directory.
"complete" is true only if at least one file is listed and every upload exited 0; the exit code is 1 otherwise. Under Orchestron the server
verifies the SHA-256 of multi-part files; this manifest records what was sent so the deposit can be checked.
"""
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(dirs: dict) -> dict:
    files, dir_rc = [], {}
    for d, rc in dirs.items():
        dir_rc[d] = rc
        for f in sorted(Path(d).glob("*")):
            if f.is_file() and not f.name.startswith("."):
                files.append({"name": f.name, "dir": d, "size": f.stat().st_size, "sha256": sha256(f),
                              "uploaded": rc == 0})
    return {"channel": "orchestron upload_artifacts.sh", "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "directories": dir_rc, "files": files, "missing": [f["name"] for f in files if not f["uploaded"]],
            "complete": bool(files) and all(rc == 0 for rc in dir_rc.values())}  # never complete when empty


def main(argv) -> int:
    out, pairs = Path(argv[0]), argv[1:]
    dirs = {}
    for kv in pairs:
        d, _, rc = kv.rpartition("=")
        dirs[d] = int(rc)
    m = build(dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1))
    print(f"orchestron manifest: {len(m['files'])} file(s), complete={m['complete']}")
    return 0 if m["complete"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
