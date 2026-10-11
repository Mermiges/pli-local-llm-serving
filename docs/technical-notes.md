# Technical notes

What we learned running two very large mixture-of-experts models (DeepSeek-V4.1-Flash and
GLM-5.3-Flash) with llama.cpp on four RTX 3090s and 503 GiB of DDR4. Everything here was
measured on our machine unless it says "reported". Numbers are in `results/`.

## 1. The OS page cache decides decode speed when two models share RAM

Both models are memory-mapped GGUFs. DeepSeek-V4.1-Flash MXFP4 is 473 GiB on disk and
GLM-5.3-Flash Q4_K_M is 188 GiB; together they exceed the 503 GiB of RAM. When jobs for the
two models alternated, each load evicted the other model's expert pages, and decode then read
from NVMe.

What it looked like:

- Three fresh GLM runs with byte-identical output tokens and identical draft acceptance decoded
  at 18.70, 13.30 and 13.30 tok/s.
- A DeepSeek control run was about 40% slower on every metric than its twin in the same hold.
- Once we read every model file to EOF inside the lock hold before starting the server, GLM's
  2k decode moved from about 12.7 to 17.5 tok/s for the same configuration (A16, ub 2048).

Rule we adopted: every speed sample prewarms all of its model files (plus any drafter) inside
the same lock hold as the server, and records `Cached` before and after
(`scripts/prewarm.sh`). Earlier samples are labelled "uncontrolled" and are not ranked against
warm ones. Prewarming is not a residency guarantee; `mlock` or a non-mmap load would be.

Caveat added 2026-10-11: we later found that this machine's DDR4 also throttles when the DIMMs
pass about 70 °C (section 10), and that effect is the same size (about −28% decode). Some of the
early swings we attributed to page-cache eviction may have been thermal. We now control both:
prewarm inside the hold, and DIMM temperature limits on every speed sample.

A related DeepSeek data point: loading into anonymous memory with transparent huge pages
(`--load-mode none --no-host --no-repack`) raised decode 34–41%, but startup went from 37 s to
330 s and prefill got slower, so we did not make it the default.

## 2. Numerical gates must compare at the same ubatch

We gate any placement, cache or engine change on two 8,192-token chunks against a reference
that computes all routed experts on the CPU: mean KL divergence ≤ 0.005, same top-1 token
≥ 98%, perplexity within ±0.5%.

In upstream llama.cpp, changing only the ubatch from 512 to 2048 gives mean KLD 0.0122 and
top-1 96.4% against the ub-512 reference, while perplexity moves +0.18%. Batched matrix
multiplication is not batch-invariant, so the logits drift slightly with batch shape. Our first
gate compared a ub-2048 candidate with the ub-512 reference and wrongly rejected a placement
that is fine. The same placement compared with a ub-2048 CPU reference gives KLD 0.0014 and
top-1 99.16%.

Rules we now use:

- Placement or cache changes: compare against a reference at the same ubatch, using the
  thresholds above (ub ≥ 64).
- Ubatch changes: judge by perplexity (±0.5%).
- Small ubatch (< 64, where expert caches typically engage): the fixed 0.005 threshold is below
  the noise floor (about 0.0115 at ub 8). Compare ON with the same build OFF at the same ubatch:
  KLD(ON, OFF) ≤ F and top-1 ≥ (top-1 of OFF) − 1 point, where F = KLD(OFF at that ub, CPU at
  ub 512). Perplexity within ±0.5%, and cache hits must actually be shown in the server log.

## 3. Warm prompt-cache reuse is not identical to a cold prompt (llama.cpp, DeepSeek V4)

On the DeepSeek V4.1 fork, a long drafting request (~31k tokens) gave a different answer when
29,159 of its tokens came from the prompt cache after 23 earlier requests than when it was
processed cold. The warm answer was byte-identical between two different builds, and the cold
answer was byte-identical between them too, so the difference comes from cache reuse, not from
either build's code. We have one such pair, which shows non-identity; it does not show that
warm answers are systematically worse.

For evaluation this means: compare builds cold-to-cold or warm-to-warm with the same request
history, never warm against cold.

Two server-side ideas of our own that tried to reuse more of the cache were dropped because
they changed outputs: an "append fast path" (cached vs cold diverged at generated token 13) and
turn-start checkpoints via the chat parser's message delimiters (restored output differed from
a fresh prefill).

## 4. A deterministic CUDA crash in MUL_MAT_ID (fixed)

A specific request sequence crashed the DeepSeek server every time with
`CUDA error: an illegal memory access` and kernel `Xid 31 ... FAULT_PDE ACCESS_TYPE_VIRT_READ`
at a 2 MiB-aligned address on the card holding three expert layers. Root cause: in the
`MUL_MAT_ID` (`ids`) path of `ggml_cuda_mul_mat_q`, the padding after the quantized activation
buffer was sized from `ne11`, which is 1 or `n_expert_used` there, so no padding was allocated.
The kernel's tile width is chosen from the token count (128 for 1,636 tokens), and the last
tile of the last expert reads up to 127 columns (18 KiB) past the buffer. When that buffer is
the newest allocation in a fresh 2 MiB virtual-memory-pool granule, the read lands on unmapped
memory. The regression came from ggml-org commit 6eddde06a (MMQ config refactor, #24127);
ggml-org later reworked this padding to be sized by the selected tile width in #29953
("CUDA: fix MMQ out-of-bounds reads"). Our minimal fix and a reproducer test are in
`patches/xid31-mul-mat-id-padding.patch`.

## 5. Layer split for DeepSeek V4.1: keep layer 20 with its readers

In DeepSeek V4.1, layer 20 owns the compressed-KV and index-key caches that layers 21–39 read.
With `-ts 1,1` (layers 0–20 / 21–39), every reader layer copied those caches across cards on
every ubatch and decoded token: 30k decode fell to 7.8 tok/s. `-ts 19.5,21.5` (0–19 / 20–39)
keeps them together and decoded 14.0 tok/s at 30k on the same (earlier, five-card) rig, with
5 GB less compute buffer on the second card.

## 6. A failing PCIe link looks like a numerical bug

Intermittent NaN perplexity and "unspecified launch failure" on DeepSeek turned out to be one
card whose Gen4 link logged tens of thousands of replays and nearly 2,000 replay-number
rollovers, followed by Xid 69, 32 and 13. Healthy cards gave 0 failures in 21 runs that agreed
to 2.2e-15. Plain 1 GiB copy tests passed on the bad link; command-stream DMA did not. Check
the PCIe replay counters in `nvidia-smi -q` before chasing a NaN in code. Derating that link to
Gen3 stopped the errors.

## 7. One shared GPU lock, one waiter per job

Everything that loads a model or touches the GPUs goes through one `flock` wrapper
(`scripts/gpu-lock.sh`). Two lessons:

- `flock` is not FIFO. One job that queued ten waiters starved the others for hours. We cap
  each job at one queued waiter.
- Downloads, CPU-only compiles and analysis run outside the lock; a large download once held it
  for most of an hour before we made that rule.

## 8. Speculative decoding on GLM-5.3-Flash

- **Built-in MTP head** (`--spec-type draft-mtp`). Draft lengths 1, 2 and 3, three fresh
  processes per cell at 2k and 30k. n = 2 was best at 30k: 18.68 tok/s decode against 17.18
  without a drafter (+9%), at a 2% prefill cost. n = 3 was fastest at 2k (20.79) but fell back
  to 17.80 at 30k. **At 64k, n = 2 was slower than no drafter** (15.72 vs 16.59 decode, 149.65
  vs 159.58 prefill). We did not find the crossover point between 30k and 64k.
- **DFlash2** (Anbeeld drafter, Q4_K_M) works on upstream llama.cpp once `glm5-next` fills
  `t_layer_inp` (ggml-org issue #30093; patch in `patches/`). With the drafter on three cards and
  the A16 placement, 30k decode was 20.43 tok/s (median of three cool runs), about +19% over the
  cool baseline. Our first report said +64%, against a baseline that hot memory had slowed. The
  drafter reuses the target's output head, so `-devd` must include the card that holds
  `output.weight`. With our 1,536 MiB per-card reserve it fits only at ctx 32,768.
- **DFlash output is not repeatable run to run.** In two separate sets of three greedy 30k runs,
  one run produced different output tokens, with lower draft acceptance and lower speed (18.7 vs
  about 20.5 tok/s). Batch-shape effects during verification are a plausible cause; not shown.
- MTP and DFlash cannot be combined in this engine: there is one draft context.
- Without GPU expert placement, DFlash did not help: n_max 2 decoded 10.2 tok/s at 2k against 13.0
  without a drafter, n_max 4 about 13.2 and n_max 7 about 7.0 (page cache not controlled).

## 9. What "reported by others" means here

Before testing we surveyed published GLM-5.3-Flash stacks for 3090s. Their reported numbers
(for example about 28 tok/s single-3090 EXL3 decode, or about 19 tok/s for a two-3090
llama.cpp fork with an expert cache) were measured on different CPUs, RAM and quantizations.
None of those numbers appear in our results tables, and our results neither confirm nor refute
them on their hardware.

## 10. Hot DIMMs cost a quarter of the decode speed

GLM's decode on this rig is bound by reading the 26 CPU-resident expert layers from RAM. When
the hottest DIMMs passed about 70 °C, the memory slowed down. Same configuration, same prompt,
same CPU clock (3,000 MHz), an hour apart:

| Hottest DIMM | 2k prefill | 2k decode |
|---|---:|---:|
| 73–75 °C | 167.24 | 12.66 |
| 61–64 °C | 184.69 | 17.41 |

How it showed up: every speed sample for hours was 17.1–17.5 tok/s, then, right after a
45-minute, 251k-token prefill, every sample read 12.0–12.7 regardless of configuration. One
sample taken when the DIMMs had cooled to 56–58 °C was fast again. Three "diagnosis" arms that
seemed to show a placement effect were all just hot. The DIMMs in question also logged
correctable ECC errors, so the fix is physical (airflow over the DIMMs). Our rule since then: a
speed sample counts only if the hottest DIMM, read from the BMC, is ≤ 64 °C at the start and
≤ 70 °C at the end, and samples are compared only within 5 °C of each other.

## 11. What the placement KLD gate does and does not check

We gate expert placements with `llama-perplexity --kl-divergence` against a reference with all
routed experts on the CPU. At ubatch ≥ 32, llama.cpp copies CPU-resident expert weights to the
GPU for large-batch multiplies, so both runs execute the same GPU kernels. A14 at ubatch 1024
scored KLD exactly 0.0 and top-1 100% for that reason. The gate is therefore a check of the
batched (prefill) path. Single-token decode, where CPU-resident experts really run on the CPU,
is not covered; end-to-end task results cover it.

## 12. Fitting a 292k context on 4 x 24 GB

At A16, ubatch 2048, the server failed to allocate at ctx 283,904 and 292,096: the compute
buffer on the first card alone was about 4.5 GB. Estimating with llama.cpp's no-allocation fit
pass, the largest placement keeping a 1,536 MiB reserve on every card was A11 at ubatch 2048,
or A14 at ubatch 1024. We chose A14 / ubatch 1024: prefill about 100 tok/s (half of ubatch 2048),
decode 16.5–16.8 (cool). It ran a 251,136-token prompt with the reserve intact, and found planted
facts in 129k- and 260k-token prompts. When you size a context, include the request's output
budget: our largest test prompt was 251,136 tokens plus a 32,768-token output budget.

## 13. Monitoring pitfalls that stopped good runs

Strict hardware gates caught real faults (section 6). They also stopped good runs, for reasons
worth knowing:

- **Busy but Gen1.** `nvidia-smi` reports GPU utilization averaged over the last sample period
  and the PCIe link generation as an instantaneous value. A card that just went idle between
  pipeline stages drops its link to Gen1 at once while its averaged utilization still reads
  above zero, so one reading can show "56% busy at Gen1". The same happens while a card receives
  weights at load time in a low power state. We now confirm a busy-at-Gen1 reading with an
  immediate second read, and apply the check only during the measured request. Narrowed width,
  replay and AER counters stay strict.
- **Kernel log out of order.** During a burst of correctable memory errors, several CPUs logged
  machine-check records at once and timestamps went backwards by 1 µs while the kernel's
  sequence numbers stayed in order. Sequence numbers are the authoritative order; µs inversions
  between CPUs are normal.
- **One marginal card.** One of our four cards logs a steady trickle of PCIe replays (89 in about
  four hours; 0 on the others) with no rollovers, AER errors or Xids. Replays are retransmissions
  checked by the link's CRC, so data is not corrupted, but the card or riser should be reseated.
