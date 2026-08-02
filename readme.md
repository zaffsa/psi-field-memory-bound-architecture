# Psi-field — an O(n) language model built for memory-bandwidth hardware

I'm an **independent researcher, working solo and self-funded**. This repository is the measured record
of an experiment I ran: an original, linear-time (**O(n)**) language model architecture of my own design,
trained head-to-head against a strong published baseline (**Mamba-2**) under identical conditions.

Everything below is backed by a raw log included in this repository. Nothing is cherry-picked, and the
architecture's internals are **not** disclosed here — this is about *what it does*, not *how it works*.

---

## What it achieves

Both models: **127M parameters · 2.0B tokens · same corpus (FineWeb-Edu) · same seeds.** The table below
is a **high-resolution held-out evaluation** of the two final checkpoints on the same validation set —
~19.6M tokens scanned, **544,029 qualifying retrieval events**, so even the long-range rows are on solid
statistical footing (n and 95% confidence intervals are in [`eval_highres_2B.log`](eval_highres_2B.log)):

| final (2.0B tokens) | **Psi-field** | Mamba-2 | verdict |
|---|---|---|---|
| perplexity | **32.68** | 32.79 | **matched** (within run-to-run noise) |
| in-context retrieval* | **5.85** | 15.17 | **2.6× better** |
| short-range recall (<512), n≈494k | **0.720** | 0.525 | **+0.195** |
| mid-range recall (512–2048), n≈42k | 0.466 | 0.479 | matched |
| long-range recall (2k–4k), n≈6.5k | 0.501 | 0.509 | matched |
| very-long-range recall (4k–8k), n≈1.5k | 0.574 | 0.579 | matched |

<sub>\* perplexity restricted to tokens that are recoverable from earlier context — i.e. how well the
model *retrieves* what it has already seen, rather than guessing from grammar. Lower is better.<br>
Beyond 8k tokens there were no qualifying samples, so that range isn't reported.</sub>

**In one line:** it reads as fluently as Mamba-2 (perplexity is a statistical tie), and its edge is
**concentrated where it counts most — short-range recall and in-context retrieval**, where it is **2.6×
better at retrieving what it just read** and lands nearly **20 points higher** on short-range recall. At
medium and long range the two models are statistically tied. This is not a tuning artifact — the gap held
from early training to the end, because it is structural.

---

## Built for memory-bandwidth hardware (and why that means AMD)

Large models are usually bottlenecked in one of two ways: by raw **compute** (matrix multiplies), or by a
specialized, **vendor-tuned kernel** that is hand-optimized for one company's GPUs — which quietly locks
the model to that vendor's hardware.

This architecture is neither. It runs in **plain PyTorch**, with **no custom kernels and nothing
vendor-specific**, and its speed is limited by **memory bandwidth**, not compute. I measured this
directly: throughput stays flat as batch size grows — the signature of a bandwidth-bound workload.

That has two consequences that matter:

**It is portable by construction.** There is no bespoke kernel to port, so the exact same code runs on any
modern accelerator. I ran it **unmodified on a single AMD MI300X** through ROCm — the same Python, just
pointed at the card, with zero code changes.

**It is a natural fit for AMD.** A bandwidth-bound, O(n) workload rewards exactly the two things the
MI300X leads on: memory **bandwidth** and memory **capacity** (192 GB HBM3). In my runs, a single MI300X
reached roughly **100,000 tokens/second**, and a scaled-up **7-billion-parameter** version fit comfortably
on **one** card. Because cost scales with memory rather than compute, this is a regime where AMD hardware
stops being a second-class "port" and becomes a first-class target.

Validating this at larger scale is what I intend to use **AMD Developer Cloud** credits for.

---

## What I've achieved so far

- **Perplexity parity** with a strong, published state-space baseline (Mamba-2) at equal parameters,
  equal corpus, and equal training tokens — within run-to-run noise.
- **2.6× better in-context retrieval** and **+0.195 short-range recall** vs that baseline, on a
  544k-event held-out evaluation.
- **O(n)** in sequence length — memory-bound, not compute-bound (measured, not assumed).
- **Pure PyTorch, no vendor-locked kernels** — runs anywhere without modification.
- **Ran unmodified on a single AMD MI300X** at ~100k tokens/second.
- A **7B-parameter** variant fits on **one** MI300X.
- Fully reproducible evaluation; every number here traces to a line in the training logs in this repository.

All of this was done by **one person**, on a self-funded budget.

---

## Proprietary components

The architecture is built on two original components of my own design — the **Psi-field** and the
**Prism**. Their internals are intentionally not published. This repository documents measured behavior
only, not mechanism.

---

## Honest limitations

- **The advantage is concentrated at short range.** Beyond ~512 tokens, my model and the baseline are
  statistically tied — the decisive wins are in short-range recall and in-context retrieval, not at long
  distances. I'm not claiming a long-range edge, because the data doesn't show one.
- This is a **127M-parameter** study. Larger-scale confirmation is the next step (and the reason for the
  AMD credits request).

---

## What's in here

- [`psi-field_127m.log`](psi-field_127m.log) — full raw training log of my model.
- [`mamba2_127m.log`](mamba2_127m.log) — full raw training log of the Mamba-2 baseline.
- [`eval_highres_2B.log`](eval_highres_2B.log) — the 544k-event held-out evaluation the results
  table is built from, with confidence intervals.

Every claim in this README is a line in one of those files.

---

## Author

Original architecture and experiments by **Samir Sauma** — an independent, self-funded researcher based
in Brazil. This work has no team behind it: one person, one budget.
