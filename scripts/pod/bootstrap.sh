#!/usr/bin/env bash
# Pod entry point for the Paper I pilot. Environment: STEPS (e.g. "data g2 g1" or "data g3"), MAX_HOURS,
# GH_TOKEN, GH_REPO, optional ZENODO_TOKEN, optional METHOD_TIMEOUT_S.
# Stage I-2/I-2b additions (all optional, defaults keep the pilot behaviour):
#   STATUS_BRANCH  results branch for the stop record (default r/pilot-p1-status)
#   STEP_TIMEOUTS  per-step wall-clock limits in seconds, e.g. "data:900 i2:8640 i2b:2700"; a stage that reaches
#                  its limit is killed and the next stage starts, so each stage stays within its own cap
#   step "tests"   runs the unit tests in the pod; a failure stops the pod (no compute outside RunPod)
#   step "zenodo"  uploads the archives written so far to the Zenodo draft (draft only)
#   FATAL_STEPS    steps whose failure stops the pod at once, e.g. "i5_inputs" (stage I-5: a missing input archive
#                  stops the pod before the test gate and the paid work; STOP_REASON "<step>_failed")
# Safeguards (docs/PILOT_P1.md, Section 11): hard stop at MAX_HOURS, idle-GPU stop after 15 min in phase
# 'train', self-stop when the job ends. The controlling agent terminates the stopped pod and keeps a ledger.
# Termination rule (docs/HANDOVER_TRACKD_P1.md, Section 5, item 2): a pod is terminated only after
# results/ZENODO_MANIFEST.json on the status branch reports "complete": true (or no archives were written).
# If STOP_REASON ends in "_upload_incomplete" the pod is only stopped: its disk keeps the archives, and only the
# upload is re-run (STEPS="zenodo") or the owner decides. The I-4 seed 2-4 checkpoints were lost by terminating
# before this check. The status branch is pushed after every step so long runs can be monitored.
# Under Orchestron (ORCH_UPLOAD_URL set) the archives and results are also sent with scripts/upload_artifacts.sh;
# results/ORCH_UPLOAD_MANIFEST.json must then report "complete": true before the pod is terminated.
set -uo pipefail
WORK=${WORK:-/workspace}; export WORK
REPO="$WORK/repo"; T="$WORK/trackd"; mkdir -p "$T/results"
STATUS_BRANCH=${STATUS_BRANCH:-r/pilot-p1-status}
stop_pod() { echo "$1" > "$T/results/STOP_REASON"; bash "$REPO/scripts/pod/push_results.sh" "$T/results" "$STATUS_BRANCH"; runpodctl stop pod "$RUNPOD_POD_ID"; }

# 1. hard wall-clock stop
( sleep "$(python3 -c "print(int(float('${MAX_HOURS:-1}')*3600))")"; stop_pod "max_hours" ) &
# 2. idle watchdog: GPU utilisation < 5 % for 15 minutes while the phase is 'train'
( idle=0; while sleep 5; do
    ph=$(cat "$T/phase" 2>/dev/null || echo prep)
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "$ph" = "train" ] && [ "${u:-100}" -lt 5 ]; then idle=$((idle+5)); else idle=0; fi
    if [ "$idle" -ge 900 ]; then stop_pod "idle_gpu"; fi
  done ) &

export GIT_ASKPASS="$WORK/askpass.sh"
echo "started $(date -u +%FT%TZ) pod $RUNPOD_POD_ID steps=$STEPS max_hours=$MAX_HOURS" > "$T/results/POD_LOG"
apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y aria2 unzip rsync >/dev/null 2>&1
pip -q install -e "$REPO" kornia natsort opencv-python-headless pywavelets pytest statsmodels pandas >/dev/null 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv >> "$T/results/POD_LOG" 2>&1
python -c "import torch,torchvision,kornia;print('torch',torch.__version__,'tv',torchvision.__version__,'kornia',kornia.__version__)" >> "$T/results/POD_LOG" 2>&1
cd "$REPO"
EXTRA=""
[ -n "${METHOD_TIMEOUT_S:-}" ] && EXTRA="method_timeout_s=$METHOD_TIMEOUT_S"
step_timeout() { for kv in ${STEP_TIMEOUTS:-}; do [ "${kv%%:*}" = "$1" ] && { echo "${kv#*:}"; return; }; done; echo 0; }
ZEN_DONE=0
UPLOAD_OK=1
zenodo_upload() {  # zenodo_upload <timeout s>: retries, skips verified files, writes the manifest; rc 1 if incomplete
  timeout --kill-after=30 "$1" python "$REPO/scripts/pod/zenodo_upload.py" "$T/archive" \
    "$T/results/ZENODO_MANIFEST.json" >> "$T/results/POD_LOG" 2>&1
}
for s in $STEPS; do
  echo "step $s start $(date -u +%FT%TZ)" >> "$T/results/POD_LOG"
  if [ "$s" = "tests" ]; then
    python -m pytest -q tests > "$T/results/TESTS_LOG" 2>&1; rc=$?
    tail -3 "$T/results/TESTS_LOG" >> "$T/results/POD_LOG"
    if [ $rc -ne 0 ]; then echo "step tests FAILED rc=$rc" >> "$T/results/POD_LOG"; stop_pod "tests_failed"; sleep infinity; fi
  elif [ "$s" = "zenodo" ]; then
    if [ -n "${ZENODO_TOKEN:-}" ] && ls "$T/archive"/*.tar >/dev/null 2>&1; then
      if zenodo_upload "$(step_timeout zenodo | sed 's/^0$/900/')"; then
        mkdir -p "$T/archive_uploaded"; mv "$T/archive"/*.tar "$T/archive_uploaded"/
      else
        echo "step zenodo INCOMPLETE (see ZENODO_MANIFEST.json); archives kept on the pod disk" >> "$T/results/POD_LOG"
      fi
    fi
    ZEN_DONE=1
  else
    to=$(step_timeout "$s")
    if [ "$to" != "0" ]; then
      timeout --kill-after=60 "$to" python -m trackd_wm.pilot "$s" work="$T" $EXTRA >> "$T/results/POD_LOG" 2>&1
    else
      python -m trackd_wm.pilot "$s" work="$T" $EXTRA >> "$T/results/POD_LOG" 2>&1
    fi
    rc=$?
    [ $rc -ne 0 ] && echo "step $s FAILED rc=$rc (124 = stage time limit)" >> "$T/results/POD_LOG"
    if [ $rc -ne 0 ] && [[ " ${FATAL_STEPS:-} " == *" $s "* ]]; then
      stop_pod "${s}_failed"; sleep infinity
    fi
  fi
  echo "step $s end $(date -u +%FT%TZ)" >> "$T/results/POD_LOG"
  bash "$REPO/scripts/pod/push_results.sh" "$T/results" "$STATUS_BRANCH"   # status after every step
done
if ls "$T/archive"/*.tar >/dev/null 2>&1; then
  if [ -n "${ZENODO_TOKEN:-}" ]; then
    zenodo_upload "${ZENODO_FINAL_TIMEOUT:-1800}" || UPLOAD_OK=0
  elif [ -z "${ORCH_UPLOAD_URL:-}" ]; then
    UPLOAD_OK=0; echo "archives present but no upload channel (ZENODO_TOKEN / ORCH_UPLOAD_URL)" >> "$T/results/POD_LOG"
  fi
fi
if [ -n "${ORCH_UPLOAD_URL:-}" ]; then
  # archives first, then results/ORCH_UPLOAD_MANIFEST.json (name, size, SHA-256 and upload rc of every archive),
  # then the results directory; "complete": false keeps the pod from being terminated (termination rule above)
  ARC_RC=""
  for d in "$T/archive" "$T/archive_uploaded"; do
    [ -d "$d" ] || continue
    bash "$REPO/scripts/upload_artifacts.sh" "$d" >> "$T/results/POD_LOG" 2>&1; rc=$?
    ARC_RC="$ARC_RC $d=$rc"
    [ $rc -ne 0 ] && { UPLOAD_OK=0; echo "upload_artifacts.sh failed for $d" >> "$T/results/POD_LOG"; }
  done
  # result subdirectories (e.g. results/i6_public): uploaded and listed in the manifest (d-2026-10-27-i6b)
  for d in "$T/results"/*/; do
    [ -d "$d" ] || continue
    d="${d%/}"
    bash "$REPO/scripts/upload_artifacts.sh" "$d" >> "$T/results/POD_LOG" 2>&1; rc=$?
    ARC_RC="$ARC_RC $d=$rc"
    [ $rc -ne 0 ] && { UPLOAD_OK=0; echo "upload_artifacts.sh failed for $d" >> "$T/results/POD_LOG"; }
  done
  python3 "$REPO/scripts/pod/orch_manifest.py" "$T/results/ORCH_UPLOAD_MANIFEST.json" $ARC_RC >> "$T/results/POD_LOG" 2>&1 \
    || UPLOAD_OK=0
  bash "$REPO/scripts/upload_artifacts.sh" "$T/results" >> "$T/results/POD_LOG" 2>&1 \
    || { UPLOAD_OK=0; echo "upload_artifacts.sh failed for $T/results" >> "$T/results/POD_LOG"; }
fi
# a step that could not preserve its outputs leaves results/UPLOAD_INCOMPLETE (trackd_wm.i6.preserve)
[ -f "$T/results/UPLOAD_INCOMPLETE" ] && { UPLOAD_OK=0; cat "$T/results/UPLOAD_INCOMPLETE" >> "$T/results/POD_LOG"; }
if [ "$UPLOAD_OK" = 1 ]; then stop_pod "completed"; else stop_pod "completed_upload_incomplete"; fi
