# Results

All numbers below were measured by us on one machine (see the main README for the rig) between
2026-10-07 and 2026-10-10, unless a row says "reported". Raw rows are in the CSV files next to
this page:

- `glm53-speed.csv`: GLM-5.3-Flash speed rows
- `glm53-numerical.csv`: GLM-5.3-Flash numerical gates (KLD, top-1 agreement, perplexity)
- `deepseek41-speed.csv`: DeepSeek-V4.1-Flash speed rows

## How to read the speed numbers

- **Fresh process.** Each sample starts a new `llama-server`, sends one request with
  `cache_prompt=false`, and stops the server. No prompt cache is reused.
- **Warm (prewarmed).** Before the server starts, every model file (all GGUF shards, plus the
  drafter if one is used) is read to the end inside the same lock hold, so the OS page cache
  holds the weights. We record `Cached` from `/proc/meminfo` before and after. There is no
  `mlock` and no `drop_caches`. Prewarm time is not counted in the token rates.
- **Uncontrolled.** Measured before we adopted the prewarm rule. The page cache may have held
  some, all or none of the weights. Treat these as historical; they are not comparable with the
  warm rows.
- **Request.** A frozen prompt of exactly 2,000 or 30,000 tokens. 400 output tokens, greedy
  (temperature 0, top_k 1, seed 0, repeat penalty 1), `ignore_eos=true`.
- **Rates.** Prefill = prompt tokens / prompt time; decode = generated tokens / generation time,
  both as reported by the server's own timings.
- **Medians** are over three fresh processes. A single sample is labelled as such.

## GLM-5.3-Flash Q4_K_M, 4x RTX 3090, warm

| Configuration | ctx | Prompt | Prefill tok/s | Decode tok/s | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| A16, ub 2048, t16, F16 KV (baseline) | 65,536 | 2,000 | 184.09 | 17.49 | — |
| A16, ub 2048, t16, F16 KV (baseline) | 65,536 | 30,000 | 169.48 | 12.42 | — |
| A16 + DFlash2 n_max 4 | 32,768 | 2,000 | 156.90 | 18.67 | 59.7% |
| A16 + DFlash2 n_max 4 | 32,768 | 30,000 | 173.36 | **20.33** | 66.7–71.4% |
| A16 + built-in MTP n=1 (single sample) | 65,536 | 2,000 | 169.75 | 18.68 | 82.2% |

DFlash2 against the baseline: decode +6.8% and prefill −14.8% at 2k; decode +63.7% and prefill
+2.3% at 30k. These compare whole configurations (the DFlash build carries the glm5-next hook
and runs at ctx 32,768), so they do not isolate the drafter's effect. The MTP comparison
(n = 1, 2, 3 at 2k and 30k, three samples each) was not finished; only the single n=1 sample
above exists under the warm protocol.

Thread count (single warm samples at 2k, same baseline otherwise): t8 12.0, t12 12.7,
**t16 16.9**, t24 12.2 tok/s decode. The CPU has 16 cores / 32 threads.

## GLM-5.3-Flash numerical gates

Method: two independent 8,192-token chunks of a fixed English text, 8,190 scored positions,
full-vocabulary logits. Each candidate is compared with all routed experts computed on the CPU.

| Candidate | vs reference | Mean KLD | Same top-1 | PPL | Result |
|---|---|---:|---:|---:|---|
| A16 ub 512 | CPU experts ub 512 | 0.000017 | 99.963% | 2.855923 (ref 2.8559) | PASS |
| A16 ub 2048 | CPU experts ub 2048 | 0.001431 | 99.158% | 2.861159 (ref 2.860488) | PASS |
| A16 ub 2048 | CPU experts ub 512 | 0.012230 | 96.410% | 2.860921 (+0.18%) | diagnostic only: ubatch change |
| DFlash build, target only | published reference | — | — | 2.8609 (+0.18%) | PASS (PPL) |
| neurall fork, cache ON, ub 8 | CPU experts ub 512 | 0.0111 | 96.65% | +0.54% | not admitted (see notes) |
| neurall fork, cache OFF, ub 8 | CPU experts ub 512 | 0.0115 | — | — | small-ubatch floor |
| club v1.6, cache ON, ub 8 | club cache OFF, ub 8 | 0.010214 | 96.618% | 2.864015 vs 2.8673 | PASS (floor 0.011744) |
| club v1.6 serving, ub 512 | CPU experts ub 512 | 0.012201 | 96.557% | 2.870447 (+0.51%) | FAIL |
| GSQ-RCO 3.5-bit GGUF | Q4_K_M | 0.105732 | 90.537% | 3.021915 (+5.81%) | FAIL |
| EXL3 3.05 bpw | (different engine; PPL only) | — | — | 3.022792 (+5.84%) | quality loss |
| EXL3 4.05 bpw | (different engine; PPL only) | — | — | 2.841710 (−0.50%) | close to Q4_K_M |

Notes:

- Changing only the ubatch (512 to 2048) moves mean KLD by about 0.012 and top-1 agreement to
  about 96.4% in upstream llama.cpp, with perplexity unchanged within 0.2%. A placement or
  cache feature must therefore be compared with a reference at the **same** ubatch; a change
  of ubatch is judged by perplexity.
- At ubatch below 64 the same effect sets a floor near KLD 0.0115. For small-ubatch features
  we compare ON with the same build OFF at the same ubatch, and require
  KLD(ON, OFF) ≤ F, where F = KLD(OFF at that ubatch, CPU reference at ub 512).
- The neurall row was measured before that rule existed. Its ON result (0.0111) is no worse
  than its own OFF result (0.0115), so the data show no cache-specific error. We dropped it
  for time, not for a demonstrated defect, and measured no speed for it.

## Rejected or not adopted

| Option | What we measured | Why not adopted |
|---|---|---|
| neurall/llama.cpp expert cache (builds on ggml-org PR #27861) | KLD above at ub 8; cache hit rate 43% | Dropped before speed tests; no defect shown |
| noonghunna/llama.cpp club-3090 moe-cache v1.6 | ub 8 cache admitted; serving at ub 512 failed the gate; cache-only 2k 36.5 / 14.4 tok/s (uncontrolled); with A16 every combination left less than our 1,536 MiB per-card reserve (1,327–1,523 MiB) | Serving-batch numerical shift and no VRAM room with A16 |
| glm-flash-lite, EXL3 3.05 bpw, one GPU | 2k prefill 230 / decode 13.3 tok/s; PPL +5.84%; two fresh runs diverge at output token 86 | Quality loss; not repeatable |
| ExLlamaV3 4.05 bpw, four GPUs, 28 MoE layers on CPU | 2k prefill 90–151 / decode 8.9 tok/s; PPL −0.50%; two fresh runs diverge at token 41 | Slower than llama.cpp A16; not repeatable |
| GSQ-RCO 3.5-bit GGUF | PPL +5.81%, KLD 0.106, top-1 90.5% | Quality loss |
| DFlash2 n_max 7 (all experts on CPU) | 2k decode 7.0–7.1 tok/s | Slower than n_max 4 |
| DeepSeek server "append fast path" (ours) | cached vs cold outputs diverged at generated token 13 (top-logit margin 0.20; up to 0.82 logprob difference on the shared top-10) | Changes outputs |
| DeepSeek V4.1 turn-start "message delimiters" (ours) | restored-checkpoint output not identical to fresh prefill | Changes outputs |
| DeepSeek speculative decoding with a DSpark drafter | 8 of 8 strict token-ID comparisons failed | Changes outputs; no admitted speed gain |
| DeepSeek static hot-expert cache (ours) | single-token path relative NLL error 5.94e-4 (limit 1e-4), deterministic | Numerical defect in our code |
| DeepSeek anonymous THP (`--load-mode none --no-host --no-repack`) | decode +41% at 2k / +34% at 30k; startup 37 s → 330 s; prefill slower | Startup and prefill cost outweigh decode gain as a default |

## DeepSeek-V4.1-Flash MXFP4, 4x RTX 3090

Speeds were not taken under the prewarm protocol; see `deepseek41-speed.csv`.

| Build | Prompt | Prefill tok/s | Decode tok/s | Samples |
|---|---:|---:|---:|---|
| CHUNK3 | 2,000 | 111.6 | 11.5 | 1 (96 tokens) |
| CHUNK3 | 30,000 | 107.6 | 10.4 | 1 (96 tokens) |
| CHUNK3 | 64,000 | 88.0 | 10.8 | 1 (96 tokens) |
| CHUNK3 + Xid 31 fix | 2,000 | 111.3 / 117.3 | 15.8 / 15.8 | 2 (64 tokens) |
| CHUNK3 + Xid 31 fix | 30,000 | 112.3 / 112.0 | 13.4 / 13.5 | 2 (64 tokens) |

Long context: on the CHUNK3 engine with experimental compact sparse attention, decode at
250,003 tokens of context was 10.38 tok/s versus 6.87 for the dense path (+51%; +43% with the
order of the runs reversed), with identical output token IDs. A needle at 50% depth of a
250,184-token prompt was found. These were measured before the prewarm rule.

Correctness checks on the DeepSeek engine changes:

- Chunked indexer: two-chunk 8k perplexity 2.0441, NLL equal to the unchunked engine within
  2.2e-15 relative; ordered top-k of three indexer layers identical (829,440 entries each).
- Xid 31 fix: new `test-backend-ops` case faults on the unfixed build (4/4) and passes on the
  fixed build (4/4); full `MUL_MAT_ID` suite 882/882; the crashing request sequence completes;
  11/11 exact-output controls pass.
- C2 parser change: on an internal task battery, every completed comparison against the
  unchanged build matched (all 23 long-context QA outputs identical at ~122k and ~252k-token
  prompts; recall 20/20 and 18/20 respectively for both builds; tool selection strict 12/16 and
  acceptable 16/16 for both).

The task battery's prompts and documents are private and are not published.
