#!/usr/bin/env bash
# One shared lock for every model load, benchmark and server start on the box.
# Usage: gpu-lock.sh <job-name> <command...>
#
# Rules we ended up with after several days of many concurrent jobs:
#   - Every GPU or model-loading command runs inside exactly one hold; never nest holds.
#   - Release between runs. A child that must keep the lock gets fd 9 passed down explicitly.
#   - Downloads, compiles that do not touch the GPU, and analysis run OUTSIDE the lock.
#   - At most ONE queued waiter per job. flock(1) is not FIFO: when one job queued ten
#     waiters, wake-up order was effectively random and other jobs starved for hours.
#   - Speed samples prewarm their model files inside the same hold (scripts/prewarm.sh).
set -euo pipefail
(( $# >= 2 )) || { echo "usage: $0 <job-name> <command...>" >&2; exit 2; }
job=$1; shift
dir=${GPU_LOCK_DIR:-/tmp/gpu-lock}
mkdir -p "$dir"
log="$dir/gpu-lock.log"
exec 9>"$dir/gpu.lock"
echo "$(date -u +%FT%TZ) WAIT $job: $*" | cut -c1-300 >> "$log"
flock 9
echo "$(date -u +%FT%TZ) HOLD $job pid=$$" >> "$log"
trap 'rc=$?; echo "$(date -u +%FT%TZ) FREE $job rc=$rc" >> "$log"' EXIT
"$@"
