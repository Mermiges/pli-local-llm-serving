#!/usr/bin/env bash
# GLM-5.3-Flash Q4_K_M + DFlash2 drafter (Anbeeld/GLM-5.3-Flash-DFlash2-GGUF, Q4_K_M),
# same A16 placement as glm53-a16-ub2048.sh, draft n_max 4. Context is 32768, not 65536:
# with the drafter loaded, 65536 did not leave our 1536 MiB per-card reserve.
#
# Measured (warm, 400 greedy tokens):
#   2k prompt:  prefill 156.90 tok/s, decode 18.67 tok/s, draft acceptance 59.7% (median of 3, 2026-10-10)
#   30k prompt: prefill 173.76 tok/s, decode 20.43 tok/s (median of 3, cool DIMMs, 2026-10-11)
#   That is about +19% decode vs the cool A16 baseline at 30k (17.18). The 30k runs decoded at
#   20.57 / 20.43 / 18.68 tok/s and the slowest produced different greedy tokens: outputs are not
#   repeatable run to run. Does not fit ctx 65536 with a 1,536 MiB per-card reserve.
# Target-only PPL on the same build (drafter inactive): 2.8609 (+0.18% vs 2.8559 reference).
#
# Engine: ggml-org/llama.cpp at 3d65c90d04d337e88f2b1f7f0061f40a5324e662 plus the
#   glm5-next t_layer_inp hook (patches/glm5-next-dflash-layer-inp.patch; the same change is
#   proposed upstream in ggml-org/llama.cpp PR #30268 for issue #30093). Without the hook the
#   server aborts on the first request.
# -devd must include the device that holds the target's output head (CUDA3 here): the
#   drafter reuses the target's output.weight. With -devd CUDA0 the server aborted at
#   context creation ("pre-allocated tensor (output.weight) in a buffer (CUDA3) that cannot
#   run the operation"). We leave CUDA0 out because it has the least free VRAM.
# Drafter licence: CC BY-NC-ND 4.0 (non-commercial). Check it before using this in production.
set -euo pipefail
: "${MODELS:?}"; : "${LLAMA_GLM_DFLASH:?llama.cpp checkout with the glm5-next hook}"; : "${GPUS:?}"

export CUDA_VISIBLE_DEVICES="$GPUS"
OT='^blk\.(7|8|9|11)\.ffn_(gate|up|down)_exps\.weight$=CUDA0,'\
'^blk\.(14|18|19|22)\.ffn_(gate|up|down)_exps\.weight$=CUDA1,'\
'^blk\.(31|33|34|35)\.ffn_(gate|up|down)_exps\.weight$=CUDA2,'\
'^blk\.(36|38|39|42)\.ffn_(gate|up|down)_exps\.weight$=CUDA3,'\
'^blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight$=CPU'

exec "$LLAMA_GLM_DFLASH/build/bin/llama-server" \
  -m "$MODELS/glm-5.3-flash-q4_k_m/GLM-5.3-Flash-Q4_K_M-00001-of-00006.gguf" \
  -ngl 99 -ts 1,1,1,1 -ot "$OT" \
  -t 16 -tb 16 -b 2048 -ub 2048 \
  -fa on -ctk f16 -ctv f16 --no-warmup --fit off \
  -c 32768 -np 1 --no-context-shift --jinja \
  --spec-type draft-dflash \
  -md "$MODELS/glm-5.3-flash-dflash2-gguf/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf" \
  --spec-draft-n-max 4 -devd CUDA1,CUDA2,CUDA3 -ngld 99 \
  --host 127.0.0.1 --port "${PORT:-8480}"
