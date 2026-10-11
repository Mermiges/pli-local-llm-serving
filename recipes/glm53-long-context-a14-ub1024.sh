#!/usr/bin/env bash
# GLM-5.3-Flash Q4_K_M (AesSedai) on 4x RTX 3090 + EPYC 7302P, upstream llama.cpp, 292k context.
# A16 cannot allocate this context (the ubatch-2048 compute buffer needs ~4.5 GB on CUDA0).
# A14 at ubatch 1024 is the largest placement that keeps a 1,536 MiB per-card reserve
# (at ubatch 2048 the largest was A11): 14 whole routed-expert layers on the GPUs, the other
# 28 in system RAM.
#
# Measured (warm, cool DIMMs, single fresh processes, 400 greedy tokens), 2026-10-11:
#   2k prompt:  prefill 100.50 tok/s, decode 16.75 tok/s
#   30k prompt: prefill  99.06 tok/s, decode 16.54 tok/s
#   with MTP n=2 (add the --spec-* line below): 2k 94.29 / 20.10, 30k 97.90 / 18.19
#   (MTP n=2 was slower than no drafter at 64k on the A16 configuration; we did not measure
#   it beyond 30k here.)
# Processed a 251,136-token prompt + 400 tokens with the reserve intact; found a planted fact at
# 50% depth in 129,030- and 260,102-token prompts. 512k and 1M contexts do not fit.
# Numerical: ubatch-1024 PPL 2.858379 (+0.09% vs 2.8559); same-ubatch placement KLD 0.0 against
# all-experts-on-CPU (batched path only; see README "Numerical gates").
#
# Engine: ggml-org/llama.cpp at 3d65c90d04d337e88f2b1f7f0061f40a5324e662 (2026-10-08). No changes.
set -euo pipefail
: "${MODELS:?set MODELS to the directory holding the model folders}"
: "${LLAMA_GLM:?set LLAMA_GLM to the llama.cpp checkout with build/bin}"
: "${GPUS:?set GPUS to four comma-separated GPU UUIDs in the intended CUDA0..3 order}"

export CUDA_VISIBLE_DEVICES="$GPUS"
OT='^blk\.(7|8|11)\.ffn_(gate|up|down)_exps\.weight$=CUDA0,'\
'^blk\.(14|18|19|22)\.ffn_(gate|up|down)_exps\.weight$=CUDA1,'\
'^blk\.(31|33|34|35)\.ffn_(gate|up|down)_exps\.weight$=CUDA2,'\
'^blk\.(36|38|42)\.ffn_(gate|up|down)_exps\.weight$=CUDA3,'\
'^blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight$=CPU'

exec "$LLAMA_GLM/build/bin/llama-server" \
  -m "$MODELS/glm-5.3-flash-q4_k_m/GLM-5.3-Flash-Q4_K_M-00001-of-00006.gguf" \
  -ngl 99 -ts 1,1,1,1 -ot "$OT" \
  -t 16 -tb 16 -b 2048 -ub 1024 \
  -fa on -ctk f16 -ctv f16 --no-warmup \
  -c 292096 -np 1 --no-context-shift --jinja \
  --host 127.0.0.1 --port "${PORT:-8470}"
  # optional MTP drafter: --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1
