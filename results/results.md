# Results

All numbers below were measured by us on one machine (see the main README for the rig) between
2026-10-07 and 2026-10-11, unless a row says "reported". Raw rows are in the CSV files next to
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

## Memory temperature matters: read this before comparing any speed

On 2026-10-11 we found that this machine's DDR4 slows down when its hottest DIMMs pass about
70 °C. The same GLM configuration, same prompt, same CPU clock (3,000 MHz), measured twice an
hour apart:

| Max DIMM temperature during the run | Prefill tok/s (2k) | Decode tok/s (2k) |
|---|---:|---:|
| 73–75 °C | 167.24 | 12.66 |
| 61–64 °C | 184.69 | 17.41 |

That is about −28% decode and −10% prefill, because GLM's decode is bound by reading the
CPU-resident experts from RAM. A long prefill (we saw it after a 45-minute, 251k-token run)
heats the DIMMs enough to trigger it. From that day on, a speed sample counts only if the hottest
DIMM (read from the BMC) is ≤ 64 °C when the run starts and ≤ 70 °C when it ends.

**Correction to our 2026-10-10 numbers.** Those were taken before we watched DIMM temperature.
The 30k baseline we published then (169.48 / 12.42 tok/s) and the thread sweep (t8 12.0, t12
12.7, t16 16.9, t24 12.2) were very likely taken on throttled memory: the same baseline measured
cool reads 178.18 / 17.18 at 30k. As a result, the "+63.7% decode at 30k" we reported for DFlash2
was overstated; against the cool baseline it is about +19%. The superseded rows are kept in
`glm53-speed.csv` with status `superseded-thermal`.

## GLM-5.3-Flash Q4_K_M, 4x RTX 3090, warm and cool (2026-10-11)

Upstream llama.cpp, A16 placement, ub 2048, 16 threads, F16 KV, ctx 65,536 unless stated. Rates
are medians (n = number of fresh processes).

| Configuration | Prompt | Prefill tok/s | Decode tok/s | n | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| A16 baseline (no drafter) | 2,000 | 186.59 | 17.48 | 3 | — |
| A16 baseline (no drafter) | 30,000 | 178.18 | 17.18 | 3 | — |
| A16 baseline (no drafter) | 64,000 | 159.58 | 16.59 | 2 | — |
| A16 + built-in MTP, draft n = 2 | 2,000 | 168.94 | 18.58 | 3 | 72% |
| A16 + built-in MTP, draft n = 2 | 30,000 | 174.84 | 18.68 | 3 | 81–86% |
| A16 + built-in MTP, draft n = 2 | 64,000 | 149.65 | 15.72 | 2 | — |
| A16 + DFlash2 n_max 4 (ctx 32,768) | 2,000 | 157.18 | 18.62 | 1 | 59.7% |
| A16 + DFlash2 n_max 4 (ctx 32,768) | 30,000 | 173.76 | **20.43** | 3 | 66.7–71.4% |
| A16 + DFlash2 n_max 4 | 64,000 | — | — | — | does not fit (see below) |

What this shows:

- **Up to ~30k tokens of context, a drafter helps.** At 30k, MTP n = 2 decodes 8.7% faster than
  no drafter and DFlash2 19% faster, both at a 2–3% prefill cost. On short prompts both cost
  10–16% prefill.
- **At 64k, MTP n = 2 is slower than no drafter** (−5% decode, −6% prefill). We did not measure
  where between 30k and 64k it crosses over.
- **DFlash2 does not fit at 64k** with our 1,536 MiB per-card reserve, because its drafter needs
  VRAM of its own. It is limited to ctx 32,768 on this rig. Its three 30k runs decoded at 20.57,
  20.43 and 18.68 tok/s; the slowest run also produced different greedy tokens with lower draft
  acceptance, so its outputs are not repeatable run to run.

### MTP draft length (built-in MTP head, A16, ub 2048)

24 fresh processes on a quiet host (3 per cell), taken while the DIMMs were ≤ 69 °C, before the
temperature rule; they agree with the later cool samples.

| Draft n | 2k prefill | 2k decode | 30k prefill | 30k decode |
|---:|---:|---:|---:|---:|
| off | 186.59 | 17.48 | 178.87 | 17.24 |
| 1 | 169.18 | 18.68 | 175.40 | 17.90 |
| 2 | 168.94 | 18.58 | 174.90 | **18.71** |
| 3 | 167.88 | 20.79 | 174.61 | 17.80 |

n = 3 is fastest on short prompts but falls back at 30k, so we use n = 2.

### Threads (single cool samples at 2k, A16 baseline)

t8 14.3, t12 15.9, **t16 17.5**, t24 17.2, t32 15.1 tok/s decode. The CPU has 16 cores / 32
threads; one thread per physical core is best.

### Very long context: 292k-token configuration

The A16 baseline cannot allocate a 292,096-token context: the ubatch-2048 compute buffer alone
needs about 4.5 GB on the first card. The largest placement that fits with the 1,536 MiB reserve
is **A14 with ubatch 1024** (A11 was the largest at ubatch 2048). Cool single samples:

| Configuration (ctx 292,096) | Prompt | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| A14, ub 1024, no drafter | 2,000 | 100.50 | 16.75 |
| A14, ub 1024, no drafter | 30,000 | 99.06 | 16.54 |
| A14, ub 1024, MTP n = 2 | 2,000 | 94.29 | 20.10 |
| A14, ub 1024, MTP n = 2 | 30,000 | 97.90 | 18.19 |

Halving the ubatch roughly halves prefill; decode stays close to the baseline. In this
configuration the server processed a 251,136-token prompt and generated 400 tokens with the
reserve intact, and found a planted fact ("needle") at the 50% depth of a 129,030-token and of a
260,102-token prompt (prefill about 94 tok/s; these two runs were not temperature-controlled).
512k and 1M contexts do not fit.

### GLM on our legal task battery (aggregate only)

We ran the quick tier of our internal legal battery on the 292k configuration, once without a
drafter and once with MTP n = 2. Prompts, documents and outputs are private.

| Part | No drafter | MTP n = 2 |
|---|---|---|
| Tool calling (6 cases) | 6/6 | 6/6 |
| Long-document questions (7) | 7/7 | 7/7 |
| Orchestration plan | pass | pass |
| Drafting (statement of facts), fidelity check | fail | fail |

The drafting check requires every date, amount and quotation to appear in the source record. Both
drafts failed it: one stated figures and quotations the record does not contain; the other stated
one amount correctly computed from two figures in the record but not itself in the record. We
therefore do not use GLM-5.3-Flash as a drafting model. Because the quick tier failed, the full
battery was not run, and no role was admitted.

## GLM-5.3-Flash numerical gates

Method: two independent 8,192-token chunks of a fixed English text, 8,190 scored positions,
full-vocabulary logits. Each candidate is compared with all routed experts computed on the CPU.

| Candidate | vs reference | Mean KLD | Same top-1 | PPL | Result |
|---|---|---:|---:|---:|---|
| A16 ub 512 | CPU experts ub 512 | 0.000017 | 99.963% | 2.855923 (ref 2.8559) | PASS |
| A16 ub 2048 | CPU experts ub 2048 | 0.001431 | 99.158% | 2.861159 (ref 2.860488) | PASS |
| A16 ub 2048 | CPU experts ub 512 | 0.012230 | 96.410% | 2.860921 (+0.18%) | diagnostic only: ubatch change |
| A14 ub 1024 (long-context config) | CPU experts ub 1024 | 0.0 | 100.0% | 2.858379 (ref 2.857759) | PASS (see limit below) |
| ub 1024 scheduling change | published reference | — | — | 2.858379 (+0.09%) | PASS (PPL) |
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
- **Limit of the placement gate.** `llama-perplexity` processes large batches. At ubatch ≥ 32,
  llama.cpp copies CPU-resident expert weights to the GPU for the matrix multiply, so the
  "CPU experts" reference and the placed candidate run the same GPU kernels, which is why A14 at
  ubatch 1024 scored KLD 0.0 exactly. The gate therefore checks the batched (prefill) path only.
  Single-token decode, where CPU-resident experts really run on the CPU, is not covered by it;
  we rely on end-to-end task results and finite outputs for that path.
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
| ggml-org/llama.cpp PR #29963 (GLM) | at A16 leaves 969–1,272 MiB per card (< 1,536 reserve); at A12, same-ubatch KLD PASS, cool 2k 166.6 / 16.09 vs stock 165.6 / 16.24 tok/s | No measurable gain on this rig; does not fit A16 |
| GLM EXL3 2.05 bpw, tensor parallel over all four GPUs (ExLlamaV3), MTP 3 | 2k 950.6 / 69.0, 30k 964.9 / 69.9 tok/s prefill / decode (one sample each); PPL 3.219 (+12.7%); on our task battery the first drafting task used all 32,768 reasoning tokens without an answer | Quality loss. At 2 bpw the weights already take 85 of 96 GB, so no larger quant fits for tensor parallel |
| MTP draft n = 3 | fastest at 2k (20.79 decode) but 17.80 at 30k | n = 2 is better at 30k |
| MTP n = 2 with DFlash2 at the same time | not expressible: the engine has one draft context | — |
| Thread pinning / NUMA distribute or isolate | 2k 17.4–17.5, 30k 17.1–17.2 tok/s decode, same as default | No gain |
| `--load-mode none` (no mmap) for GLM | not measured: every attempt was stopped by our PCIe-link monitor during the 175 GiB load | — |
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
