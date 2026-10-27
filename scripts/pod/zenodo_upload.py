"""Upload pilot archives to the Paper I Zenodo draft (23534235). Draft only: never publishes.

The token is read from ZENODO_TOKEN (RunPod secret) and never printed.

Usage: zenodo_upload.py <archive dir> [manifest path]

Every ``*.tar`` in the archive directory is uploaded as ``pilot_p1_<name>``:
- a file already in the draft bucket with the same MD5 is skipped (re-running the upload is safe);
- each file is retried with exponential backoff (ZENODO_RETRIES attempts, default 5);
- after the loop the deposition is read again and every archive is checked against its local MD5;
- ZENODO_MANIFEST.json (default: <archive dir>/../results/ZENODO_MANIFEST.json) records the outcome per file,
  so that it is pushed with the status branch;
- the exit code is 1 if any archive is missing or mismatched, so that the pod is never terminated on a partial
  upload (docs/HANDOVER_TRACKD_P1.md, Section 5, items 1-2).
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEP = os.environ.get("ZENODO_DEPOSITION", "23534235")
API = "https://zenodo.org/api/deposit/depositions/" + DEP
PREFIX = "pilot_p1_"


def req(url, method="GET", data=None, headers=None, timeout=600):
    h = {"Authorization": "Bearer " + os.environ["ZENODO_TOKEN"]}
    h.update(headers or {})
    r = urllib.request.Request(url, data=data, method=method, headers=h)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body) if body else {}


def md5_of(path: Path) -> str:
    m = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            m.update(blk)
    return m.hexdigest()


def remote_files(dep: dict) -> dict:
    """filename -> bare MD5 hex of the files in the deposition (Zenodo reports 'md5:<hex>' or '<hex>')."""
    return {f["filename"]: str(f.get("checksum") or "").split(":")[-1] for f in dep.get("files", [])}


def with_retry(fn, what: str, retries: int, delay: float, sleep=time.sleep):
    """Call fn() up to ``retries`` times, doubling the delay after each failure; re-raise the last error.
    HTTP 4xx other than 408/429 is not retried (authorisation or request errors do not heal)."""
    err = None
    for i in range(max(1, retries)):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err = e
        print(f"{what}: attempt {i + 1}/{retries} failed ({type(err).__name__}: {err})", flush=True)
        if i + 1 < retries:
            sleep(delay * 2 ** i)
    raise err


def put_file(bucket: str, key: str, path: Path) -> dict:
    """Streamed PUT of one file into the draft bucket (a new file handle per attempt)."""
    with open(path, "rb") as f:
        return req(f"{bucket}/{key}", "PUT", f, {"Content-Type": "application/octet-stream",
                                                  "Content-Length": str(path.stat().st_size)}, timeout=3600)


def main(folder, manifest=None, retries=None, delay=None, sleep=time.sleep) -> int:
    folder = Path(folder)
    manifest = Path(manifest) if manifest else folder.parent / "results" / "ZENODO_MANIFEST.json"
    retries = int(retries or os.environ.get("ZENODO_RETRIES", "5"))
    delay = float(delay if delay is not None else os.environ.get("ZENODO_RETRY_DELAY", "10"))
    dep = with_retry(lambda: req(API), "read deposition", retries, delay, sleep)
    if dep.get("submitted"):
        raise SystemExit("deposition already published; refusing to modify")
    bucket = dep["links"]["bucket"]
    have = remote_files(dep)
    rows = {}
    for p in sorted(folder.glob("*.tar")):
        key = PREFIX + p.name
        row = {"local_file": p.name, "bytes": p.stat().st_size, "md5": md5_of(p)}
        if have.get(key) == row["md5"]:
            row["action"] = "skipped_already_present"
        else:
            try:
                r = with_retry(lambda: put_file(bucket, key, p), f"upload {key}", retries, delay, sleep)
                row["action"] = "uploaded"
                row["md5_returned"] = str(r.get("checksum") or "").split(":")[-1] or None
            except Exception as e:
                row["action"] = "failed"
                row["error"] = f"{type(e).__name__}: {e}"
        rows[key] = row
        print(key, row["action"], row["bytes"], flush=True)
    # completeness check against a fresh read of the deposition, not against the PUT responses
    try:
        final = remote_files(with_retry(lambda: req(API), "re-read deposition", retries, delay, sleep))
    except Exception as e:
        final = {}
        print(f"completeness check could not read the deposition: {type(e).__name__}: {e}", flush=True)
    for key, row in rows.items():
        row["verified"] = final.get(key) == row["md5"]
    missing = sorted(k for k, r in rows.items() if not r["verified"])
    out = {"deposition": DEP, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pod": os.environ.get("RUNPOD_POD_ID", "local"), "n_archives": len(rows),
           "n_verified": len(rows) - len(missing), "complete": not missing, "missing": missing, "files": rows}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(out, indent=1))
    print(f"zenodo manifest: {out['n_verified']}/{out['n_archives']} verified; complete={out['complete']}",
          flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
