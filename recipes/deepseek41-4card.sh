#!/usr/bin/env bash
# DeepSeek-V4.1-Flash MXFP4 GGUF (10 shards, smalinin/DeepSeek-V4.1-Flash-GGUF) on 4x RTX 3090.
# Engine: vcruz305/llama.cpp branch runtime/deepseek41 (5210c7c5e) plus our local commits:
#   - chunked indexer scoring so ctx 327680 runs at ubatch 2048 ("CHUNK3", c214eb64a)
#   - patches/c2-dsml-parser.patch (V4.1 DSML tool-call parsing, no turn-start checkpoints)
#   - patches/xid31-mul-mat-id-padding.patch (fixes a deterministic CUDA illegal memory access)
#
# Layout: -ts 19.5,21.5,0,0 puts layers 0-19 on CUDA0 and 20-39 + output on CUDA1. Layer 20
# owns the compressed-KV and index-key caches that layers 21-39 read; keeping them on one card
# avoided a cross-card copy that halved 30k decode on an earlier -ts 1,1 layout.
# Eight routed-expert layers on GPU (29 -> CUDA1, 39 -> CUDA0, 36-38 -> CUDA2, 33-35 -> CUDA3),
# all other experts and the Engram tables in RAM.
#
# Measured on this layout (page-cache state not controlled; see results/deepseek41-speed.csv):
#   CHUNK3 build: prefill 111.6 tok/s @2k, 107.6 @30k, 88.0 @64k; decode 10.4-11.8 tok/s
#     (single fresh-process samples, 96- and 400-token outputs)
#   CHUNK3 + Xid 31 fix: prefill 111-117 tok/s @2k/30k; decode 15.8 @2k, 13.4-13.5 @30k
#     (two samples each, 64-token outputs)
# The C2 parser change only touches chat parsing; its outputs matched the CHUNK3 build on every
# completed comparison we ran.
set -euo pipefail
: "${MODELS:?}"; : "${LLAMA_DS41:?llama.cpp (vcruz305 fork + patches) checkout with build/bin}"; : "${GPUS:?}"

export CUDA_VISIBLE_DEVICES="$GPUS"
OT='.*engram_embd.*=CPU,'\
'blk\.29\.ffn_.*_exps.*=CUDA1,blk\.39\.ffn_.*_exps.*=CUDA0,'\
'blk\.(36|37|38)\.ffn_.*_exps.*=CUDA2,blk\.(33|34|35)\.ffn_.*_exps.*=CUDA3,'\
'.*ffn_.*_exps.*=CPU'

exec "$LLAMA_DS41/build/bin/llama-server" \
  -m "$MODELS/deepseek-v4.1-flash-mxfp4/DeepSeek-V4.1-Flash-MXFP4-00001-of-00010.gguf" \
  -ngl 999 -ts 19.5,21.5,0,0 -ot "$OT" \
  -c 327680 -np 1 -fa on -fit off -t 16 -tb 16 -b 2048 -ub 2048 -sm layer -mg 0 \
  --host 127.0.0.1 --port "${PORT:-8420}"
