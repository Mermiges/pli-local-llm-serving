#!/usr/bin/env bash
# GLM-5.3-Flash Q4_K_M (AesSedai) on 4x RTX 3090 + EPYC 7302P, upstream llama.cpp.
# A16 placement (see glm53-a16-ub2048.sh) plus the model's built-in MTP head as a 2-token drafter.
# Use for prompts up to ~30k tokens. At 64k it is SLOWER than no drafter.
#
# Measured (warm, cool DIMMs, medians of fresh processes, 400 greedy tokens), 2026-10-11:
#   2k prompt:  prefill 168.94 tok/s, decode 18.58 tok/s (3), draft acceptance 72%
#   30k prompt: prefill 174.84 tok/s, decode 18.68 tok/s (3), draft acceptance 81-86%
#   64k prompt: prefill 149.65 tok/s, decode 15.72 tok/s (2)  <- no drafter: 159.58 / 16.59
# Draft n=1 and n=3 were also measured (3 processes per cell); n=2 was best at 30k.
#
# Engine: ggml-org/llama.cpp at 3d65c90d04d337e88f2b1f7f0061f40a5324e662 (2026-10-08),
#   Release, CUDA 12.6, sm_86. No source changes.
set -euo pipefail
: "${MODELS:?set MODELS to the directory holding the model folders}"
: "${LLAMA_GLM:?set LLAMA_GLM to the llama.cpp checkout with build/bin}"
: "${GPUS:?set GPUS to four comma-separated GPU UUIDs in the intended CUDA0..3 order}"

export CUDA_VISIBLE_DEVICES="$GPUS"
OT='^blk\.(7|8|9|11)\.ffn_(gate|up|down)_exps\.weight$=CUDA0,'\
'^blk\.(14|18|19|22)\.ffn_(gate|up|down)_exps\.weight$=CUDA1,'\
'^blk\.(31|33|34|35)\.ffn_(gate|up|down)_exps\.weight$=CUDA2,'\
'^blk\.(36|38|39|42)\.ffn_(gate|up|down)_exps\.weight$=CUDA3,'\
'^blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight$=CPU'

exec "$LLAMA_GLM/build/bin/llama-server" \
  -m "$MODELS/glm-5.3-flash-q4_k_m/GLM-5.3-Flash-Q4_K_M-00001-of-00006.gguf" \
  -ngl 99 -ts 1,1,1,1 -ot "$OT" \
  -t 16 -tb 16 -b 2048 -ub 2048 \
  -fa on -ctk f16 -ctv f16 --no-warmup \
  -c 65536 -np 1 --no-context-shift --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1 \
  --host 127.0.0.1 --port "${PORT:-8470}"
