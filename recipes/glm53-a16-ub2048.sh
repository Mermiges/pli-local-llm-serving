#!/usr/bin/env bash
# GLM-5.3-Flash Q4_K_M (AesSedai) on 4x RTX 3090 + EPYC 7302P, upstream llama.cpp.
# "A16" placement: 16 whole routed-expert layers on the GPUs (4 per card), the other
# 26 routed layers in system RAM. Batch/ubatch 2048, 16 threads, F16 KV, ctx 65536.
#
# Measured (warm, median of 3 fresh processes, 400 greedy tokens):
#   2k prompt:  prefill 184.09 tok/s, decode 17.49 tok/s
#   30k prompt: prefill 169.48 tok/s, decode 12.42 tok/s
# Numerical gate vs all-experts-on-CPU at the same ubatch (2x8192 tokens):
#   mean KLD 0.001431, same top-1 99.158%, PPL 2.861159 vs 2.860488.
#
# Engine: ggml-org/llama.cpp at 3d65c90d04d337e88f2b1f7f0061f40a5324e662 (2026-10-08),
#   Release, CUDA 12.6, sm_86. No source changes.
# Device order matters: the -ot layer lists were picked from measured routing mass and
#   assume CUDA0..CUDA3 = the four cards in CUDA_VISIBLE_DEVICES order. On our rig CUDA3
#   is the x8 card; it also ends up holding the output head.
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
  --host 127.0.0.1 --port "${PORT:-8470}"
