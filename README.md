# pli-local-llm-serving

PLI (ProprietaryLegal.ai) builds legal-workflow tools that run large language models on its
own hardware. This repository publishes the serving configurations, test method and measured
results from getting two large mixture-of-experts models to run well on one workstation-class
server with four consumer GPUs.

Everything in the results tables was measured by us on the machine described below. Where we
mention numbers published by others, we say so.

## The machine

| Part | Detail |
|---|---|
| CPU | AMD EPYC 7302P, 16 cores / 32 threads, Zen 2, AVX2 (no AVX-512) |
| RAM | 8 x 64 GB DDR4-3200, all eight channels (503 GiB usable) |
| GPUs | 4 x NVIDIA RTX 3090 24 GB, PCIe Gen4 (three x16, one x8), no NVLink |
| Storage | NVMe; models are memory-mapped from it |
| Software | Ubuntu 24.04, NVIDIA driver 580.126.20, CUDA toolkit 12.6.85, GCC 13.3 |

## The models

| Model | Quantization | Source | Size on disk |
|---|---|---|---|
| DeepSeek-V4.1-Flash | MXFP4 GGUF, 10 shards | `smalinin/DeepSeek-V4.1-Flash-GGUF` | 473 GiB |
| GLM-5.3-Flash | Q4_K_M GGUF, 6 shards | `AesSedai/GLM-5.3-Flash-GGUF` | 188 GiB |
| GLM-5.3-Flash DFlash2 drafter (optional) | Q4_K_M GGUF | `Anbeeld/GLM-5.3-Flash-DFlash2-GGUF` (CC BY-NC-ND 4.0) | 0.7 GB |

Neither model fits in GPU memory. In every configuration below, the attention and shared
weights sit on the GPUs and most routed experts stay in system RAM; the question is which
experts to put on the GPUs and how to batch.

## Best configurations

### GLM-5.3-Flash: upstream llama.cpp, "A16" placement

Engine: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at
`3d65c90d04d337e88f2b1f7f0061f40a5324e662` (2026-10-08), no source changes. Full script:
[`recipes/glm53-a16-ub2048.sh`](recipes/glm53-a16-ub2048.sh).

```bash
CUDA_VISIBLE_DEVICES=$GPU0,$GPU1,$GPU2,$GPU3 \
llama-server -m $MODELS/glm-5.3-flash-q4_k_m/GLM-5.3-Flash-Q4_K_M-00001-of-00006.gguf \
  -ngl 99 -ts 1,1,1,1 \
  -ot '^blk\.(7|8|9|11)\.ffn_(gate|up|down)_exps\.weight$=CUDA0,^blk\.(14|18|19|22)\.ffn_(gate|up|down)_exps\.weight$=CUDA1,^blk\.(31|33|34|35)\.ffn_(gate|up|down)_exps\.weight$=CUDA2,^blk\.(36|38|39|42)\.ffn_(gate|up|down)_exps\.weight$=CUDA3,^blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight$=CPU' \
  -t 16 -tb 16 -b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 --no-warmup \
  -c 65536 -np 1 --no-context-shift --jinja --host 127.0.0.1 --port 8470
```

- **A16** = 16 whole routed-expert layers on the GPUs, four per card, chosen by measured routing
  mass on two prompt profiles. Each routed layer is 4.08 GiB, so four layers add about 16.3 GiB
  per card. The other 26 routed layers stay in RAM. The `-ot` GPU rules must come before the
  CPU catch-all, and `--n-cpu-moe` must not be used with them (it takes precedence).
- **ubatch 2048** roughly triples prefill against ubatch 512 (56 to about 175 tok/s).
- **16 threads** = one per physical core. 8, 12 and 24 threads all decoded slower in single samples (12.0, 12.7 and 12.2 vs 16.9 tok/s).
- **F16 KV.** We did not accept q8_0 KV.

Warm medians (three fresh processes each, 400 greedy output tokens):

| Prompt | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 2,000 tokens | 184.09 | 17.49 |
| 30,000 tokens | 169.48 | 12.42 |

### GLM-5.3-Flash: A16 + DFlash2 drafter (fastest decode we measured)

Same engine plus an 11-line hook so `glm5-next` exposes the layer inputs DFlash needs
([`patches/glm5-next-dflash-layer-inp.patch`](patches/glm5-next-dflash-layer-inp.patch); the same
change is proposed upstream as ggml-org/llama.cpp PR #30268 for issue #30093). Full script:
[`recipes/glm53-a16-ub2048-dflash2.sh`](recipes/glm53-a16-ub2048-dflash2.sh).

```bash
  ... same -m/-ngl/-ts/-ot/-t/-tb/-b/-ub/-fa/-ctk/-ctv/--no-warmup as above ... \
  --fit off -c 32768 \
  --spec-type draft-dflash -md $MODELS/glm-5.3-flash-dflash2-gguf/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf \
  --spec-draft-n-max 4 -devd CUDA1,CUDA2,CUDA3 -ngld 99
```

| Prompt | Prefill tok/s | Decode tok/s | Draft acceptance |
|---:|---:|---:|---:|
| 2,000 tokens | 156.90 | 18.67 | 59.7% |
| 30,000 tokens | 173.36 | 20.33 | 66.7–71.4% |

Context is 32,768 here because 65,536 plus the drafter did not leave our 1,536 MiB per-card
safety reserve. `-devd` must include the card holding the target's output head (CUDA3 here),
because the drafter reuses it. The drafter is licensed CC BY-NC-ND 4.0 (non-commercial), so we
treat this configuration as research-only.

### DeepSeek-V4.1-Flash: vcruz305 fork + our fixes

Engine: [vcruz305/llama.cpp](https://github.com/vcruz305/llama.cpp) branch `runtime/deepseek41`
(`5210c7c5e`), plus:

- our chunked indexer scoring, which lets context 327,680 run at ubatch 2048 (not yet published
  here; see "Not included");
- [`patches/c2-dsml-parser.patch`](patches/c2-dsml-parser.patch): V4.1 writes its tool-call tags
  as `<｜DSML｜ calls>` (with a space and the block named `calls`), so the V4 parser returned V4.1
  tool calls as plain text. The patch parses them, adds the V4.1 template and tests, and (in its
  second commit) removes turn-start checkpoint delimiters, which changed outputs;
- [`patches/xid31-mul-mat-id-padding.patch`](patches/xid31-mul-mat-id-padding.patch): fixes a
  deterministic CUDA illegal memory access (Xid 31) in `MUL_MAT_ID`, with a reproducer test.

Full script: [`recipes/deepseek41-4card.sh`](recipes/deepseek41-4card.sh).

```bash
CUDA_VISIBLE_DEVICES=$GPU0,$GPU1,$GPU2,$GPU3 \
llama-server -m $MODELS/deepseek-v4.1-flash-mxfp4/DeepSeek-V4.1-Flash-MXFP4-00001-of-00010.gguf \
  -ngl 999 -ts 19.5,21.5,0,0 \
  -ot '.*engram_embd.*=CPU,blk\.29\.ffn_.*_exps.*=CUDA1,blk\.39\.ffn_.*_exps.*=CUDA0,blk\.(36|37|38)\.ffn_.*_exps.*=CUDA2,blk\.(33|34|35)\.ffn_.*_exps.*=CUDA3,.*ffn_.*_exps.*=CPU' \
  -c 327680 -np 1 -fa on -fit off -t 16 -tb 16 -b 2048 -ub 2048 -sm layer -mg 0 \
  --host 127.0.0.1 --port 8420
```

`-ts 19.5,21.5,0,0` puts layers 0–19 on CUDA0 and 20–39 on CUDA1, keeping layer 20 (which
owns the caches that layers 21–39 read) with its readers. Measured prefill was 111.6 / 107.6 /
88.0 tok/s at 2k / 30k / 64k prompts and decode 10.4–15.8 tok/s depending on build and run.
These DeepSeek speeds were taken before we controlled the page cache (see below), so treat
them as indicative. Details: [`results/results.md`](results/results.md).

## Results

- [`results/results.md`](results/results.md): all tables, with what each number means.
- [`results/glm53-speed.csv`](results/glm53-speed.csv),
  [`results/glm53-numerical.csv`](results/glm53-numerical.csv),
  [`results/deepseek41-speed.csv`](results/deepseek41-speed.csv): raw rows.
- [`docs/technical-notes.md`](docs/technical-notes.md): what we learned, in more detail.

## Method

**Speed.** Each sample is a fresh `llama-server` process with one request, `cache_prompt=false`,
a frozen prompt of exactly 2,000 or 30,000 tokens, and 400 greedy output tokens (temperature 0,
top_k 1, seed 0, `ignore_eos`). Rates come from the server's own timings. A median is over three
fresh processes.

**"Warm" means prewarmed page cache.** Before each server start, inside the same lock hold,
every model file is read to EOF ([`scripts/prewarm.sh`](scripts/prewarm.sh)) and `Cached` is
recorded before and after. We found this necessary because the two models (473 GiB + 188 GiB)
do not fit together in 503 GiB of RAM: llama.cpp memory-maps the weights, and alternating jobs
evicted each other's experts, which swung decode by 30–40% between identical runs. Numbers
measured before we adopted this rule are labelled "uncontrolled".

**Numerical gates.** Every placement, cache or engine change is checked on two independent
8,192-token chunks (8,190 scored positions, full-vocabulary logits) against a reference that
computes all routed experts on the CPU:

- mean KL divergence ≤ 0.005, same top-1 token ≥ 98%, perplexity within ±0.5%, at ubatch ≥ 64;
- the reference must use the **same ubatch** as the candidate. Changing only the ubatch from 512
  to 2048 moves mean KLD to 0.012 and top-1 to 96.4% in upstream llama.cpp while perplexity
  moves only 0.18%, so a ubatch change is judged by perplexity;
- below ubatch 64 (where expert caches usually engage) KLD has a floor of about 0.0115 even with
  the feature off. There we compare ON with the same build OFF at the same ubatch and require
  KLD(ON, OFF) ≤ F, top-1 within one point of OFF, perplexity within ±0.5%, and logged cache
  hits, where F is the OFF build's own KLD against the CPU reference.

**Prompt cache.** On the DeepSeek fork, a long request answered from a warm prompt cache (29k
of 31k tokens reused) produced different text from the same request processed cold, on two
different builds alike. We therefore compare builds cold-to-cold, or warm-to-warm with the
same request history.

**One shared GPU lock.** Every model load and benchmark runs inside one `flock` hold
([`scripts/gpu-lock.sh`](scripts/gpu-lock.sh)), with at most one queued waiter per job
(`flock` is not FIFO) and with downloads and CPU-only work kept outside the lock.

## Patches

| File | Applies to | What it does |
|---|---|---|
| `patches/xid31-mul-mat-id-padding.patch` | vcruz305/llama.cpp `runtime/deepseek41` (any tree with ggml before ggml-org #29953) | Reproducer test, then the padding fix for the MUL_MAT_ID MMQ over-read |
| `patches/c2-dsml-parser.patch` | vcruz305/llama.cpp `runtime/deepseek41` snapshot `5210c7c5e` | V4.1 DSML tool-call parsing; the head of ggml-org PR #28696 already contains an equivalent parser |
| `patches/glm5-next-dflash-layer-inp.patch` | ggml-org/llama.cpp before PR #30268 | Fills `t_layer_inp` in `glm5-next` so DFlash drafters work |
| `patches/club3090-v16-glm5-pretokenizer.patch` | noonghunna/llama.cpp `stack/club3090-moecachev1.6` | What we changed to load mainline GGUFs in that fork: architecture string `glm5next` → `glm5-next`, and the `glm5` pre-tokenizer mapped to the existing GLM-4 path. club-3090 itself rewrites the GGUF metadata instead (their PR #1605), which also adds a key this patch does not |

## Rejected or not adopted

| Option | Result | Why not |
|---|---|---|
| neurall/llama.cpp expert cache (builds on ggml-org PR #27861) | cache ON KLD 0.0111 vs OFF 0.0115 at ubatch 8 | Dropped before speed tests; no defect shown |
| noonghunna/llama.cpp club-3090 moe-cache v1.6 | ubatch 8 cache passed the floor rule; serving at ubatch 512 failed (KLD 0.0122, top-1 96.6%, PPL +0.51%); with A16 no combination kept the VRAM reserve | Serving numerical shift, no VRAM room with A16 |
| glm-flash-lite, EXL3 3.05 bpw, one GPU | 2k prefill 230, decode 13.3 tok/s; PPL +5.8%; two fresh runs diverge at token 86 | Quality loss; not repeatable |
| ExLlamaV3 4.05 bpw, four GPUs | 2k prefill 90–151, decode 8.9 tok/s; PPL −0.5%; runs diverge at token 41 | Slower than llama.cpp; not repeatable |
| GSQ-RCO 3.5-bit GGUF | PPL +5.8%, KLD 0.106, top-1 90.5% | Quality loss |
| DeepSeek server "append fast path" (ours) | cached and cold outputs diverge at token 13 | Changes outputs |
| DeepSeek turn-start message delimiters (ours) | restored checkpoint differs from fresh prefill | Changes outputs |
| DeepSeek speculative decoding (DSpark drafter) | 8/8 strict token-ID comparisons failed | Changes outputs; no admitted gain |

More rows and the numbers behind them: [`results/results.md`](results/results.md).

## Not included

- Legal evaluation material. We also ran an internal legal-task battery; its prompts,
  documents and model outputs are private. Only aggregate pass/fail counts appear in the results.
- Our DeepSeek chunked-indexer engine commits. They sit on top of a third-party runtime patch
  whose licence we have not confirmed.
- Model weights.

## Upstream reports

We are reporting the reproducible problems we hit to the projects concerned, with logs and
patches. Links will be added here once they are posted.

## Licence

MIT for the scripts, recipes, documentation and results in this repository; see
[`LICENSE`](LICENSE) for the scope, including patches and third-party model licences.
