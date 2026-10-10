#!/usr/bin/env bash
# Read every model file to EOF so the OS page cache holds it before the server starts.
# Run it inside the same GPU-lock hold as the server (see gpu-lock.sh), immediately before
# launch. It records Cached from /proc/meminfo before and after so a run can show its state.
#
# Why: llama.cpp mmaps GGUF weights. Our two models (DeepSeek-V4.1-Flash MXFP4, 473 GiB, and
# GLM-5.3-Flash Q4_K_M, 188 GiB) do not fit in 503 GiB of RAM together, so alternating runs
# evicted each other's expert pages and decode read NVMe. That produced 30-40% swings in
# decode speed between otherwise identical runs. This is not a residency guarantee: there is
# no mlock, and nothing stops later eviction.
#
# Usage: prewarm.sh FILE... (for split GGUFs, pass every shard, plus any drafter)
set -euo pipefail
(( $# > 0 )) || { echo "usage: $0 FILE..." >&2; exit 2; }
cached() { awk '/^Cached:/ {print $2}' /proc/meminfo; }
before=$(cached)
start=$(date +%s.%N)
total=0
for f in "$@"; do
    [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
    size=$(stat -c %s "$f")
    # Pipe through cat: GNU wc -c on a regular file only seeks, it does not read.
    read_bytes=$(cat -- "$f" | wc -c)
    [[ "$read_bytes" == "$size" ]] || { echo "short read: $f ($read_bytes of $size)" >&2; exit 1; }
    total=$(( total + size ))
done
end=$(date +%s.%N)
after=$(cached)
printf '{"files": %d, "bytes": %d, "seconds": %.3f, "cached_kib_before": %d, "cached_kib_after": %d}\n' \
    "$#" "$total" "$(awk -v a="$start" -v b="$end" 'BEGIN {print b - a}')" "$before" "$after"
