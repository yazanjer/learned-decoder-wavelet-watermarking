#!/usr/bin/env bash
# Push a results directory to an orphan branch (force-push of that results branch only; never main or wip).
# Files over 95 MB are skipped (they belong on the Zenodo draft). Usage: push_results.sh <dir> <branch>
set -uo pipefail
SRC="$1"; BR="$2"
case "$BR" in r/*) ;; *) echo "refusing to push to non-results branch $BR"; exit 1;; esac
[ -d "$SRC" ] || exit 0
T=$(mktemp -d)
cd "$T" && git init -q && git checkout -q --orphan "$BR"
rsync -a --max-size=95m "$SRC"/ ./
git add -A && git -c user.name="Yazan Aljeroudi" -c user.email="yazan.aljeroudi@gmail.com" \
  commit -q -m "pilot results $BR $(date -u +%FT%TZ) pod ${RUNPOD_POD_ID:-local}"
ok=1
for i in 1 2 3; do
  if git push -q -f "https://x-access-token@github.com/${GH_REPO}.git" "HEAD:refs/heads/$BR"; then ok=0; break; fi
  sleep 10
done
rm -rf "$T"
# exit status 1 when every push attempt failed (callers record it; d-2026-10-27-i6b, required change 3)
exit $ok
