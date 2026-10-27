#!/usr/bin/env bash
# Upload trained-model artefacts from a RunPod pod to Orchestron for the Zenodo deposit.
# Usage: scripts/upload_artifacts.sh <output_dir>
# Uses ORCH_UPLOAD_URL and ORCH_UPLOAD_TOKEN, which Orchestron sets on the pod at launch.
# Upload: final and best checkpoints, config files, metrics JSON. Not intermediate checkpoints.
# Files above 90 MB are sent in 90 MB parts (Cloudflare limits each request to 100 MB);
# the server joins them and verifies the SHA-256 checksum.
set -euo pipefail
dir="${1:?output directory required}"
: "${ORCH_UPLOAD_URL:?not running under Orchestron}"
: "${ORCH_UPLOAD_TOKEN:?not running under Orchestron}"
PART=$((90 * 1024 * 1024))
put() {  # put <file> <query>
  curl -fsS --retry 5 --retry-delay 10 --retry-all-errors -X PUT \
       -H "X-Upload-Token: ${ORCH_UPLOAD_TOKEN}" --upload-file "$1" "$2" > /dev/null
}
shopt -s nullglob
count=0
for f in "$dir"/*; do
  [ -f "$f" ] || continue
  name="$(basename "$f")"
  case "$name" in .*|*.tmp|*.log.partial) continue;; esac
  size=$(stat -c %s "$f")
  if [ "$size" -le "$PART" ]; then
    echo "uploading $name ($size bytes)"
    put "$f" "${ORCH_UPLOAD_URL}/${name}"
  else
    sum=$(sha256sum "$f" | cut -d' ' -f1)
    tmp=$(mktemp -d)
    split -b "$PART" -d -a 4 "$f" "$tmp/p"
    parts=$(ls "$tmp" | wc -l)
    echo "uploading $name ($size bytes) in $parts parts"
    i=1
    for p in "$tmp"/p*; do
      put "$p" "${ORCH_UPLOAD_URL}/${name}?part=${i}&parts=${parts}&sha256=${sum}"
      i=$((i + 1))
    done
    rm -rf "$tmp"
  fi
  count=$((count + 1))
done
echo "uploaded ${count} file(s)"
