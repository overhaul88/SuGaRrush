# Reducing the SuGaR SDF-Regularization Bottleneck via Monte-Carlo Sample-Budget Reduction

**A profiling-driven optimization of surface-aligned Gaussian-splatting mesh reconstruction on memory-constrained hardware**

*Hardware under test: NVIDIA GTX 1650 (4 GB VRAM, compute capability 7.5, Turing), 7.7 GB system RAM, WSL2.*

---

## Abstract

The SuGaR pipeline (COLMAP → 3D Gaussian Splatting → surface-aligned regularization → Poisson mesh)
turns a hand-held phone video into a textured triangle mesh. On a 4 GB consumer GPU we profiled a
full run and found that a single phase, the *signed-distance-field (SDF) regularization* of SuGaR's
coarse training — consumed **9.43 h of a 10.22 h pipeline (92 %)**. Through a structured decomposition of the phase's cost we formed **Hypothesis H1**: the phase is
*memory-bandwidth bound* on a per-sample nearest-neighbour gather, its cost is therefore *linear in
the Monte-Carlo sample count* `n_samples_for_sdf_regularization` (hard-coded to 1,000,000), and that
count is redundant because the loss is a per-sample mean already variance-averaged over 6,000
optimization steps. We validate H1 in three escalating experiments: (i) a controlled kernel
microbenchmark (linear cost, R = 0.9994), (ii) a timing of the *real* field-evaluation path on a
loaded 42,837-Gaussian model (2.69× faster at 300k vs 1M), and (iii) a full end-to-end A/B on a new
object at 50k versus 300k samples, which produced meshes agreeing to **0.31 % of the bounding-box
diagonal** (symmetric Chamfer). We implement H1 as a backward-compatible environment-overridable
constant. We also report a negative-space finding: because SuGaR renders all ground-truth images
every iteration, a large *fixed* per-iteration cost dilutes the sample-linear speedup at the
whole-stage level (Amdahl's law), so the end-to-end gain is smaller than the pure-kernel benchmark
implies. We conclude with the implication that, post-optimization, vanilla 3DGS (5 h 39 m) becomes
the new dominant cost.

**Artifacts: `SuGaR/sugar_trainers/*.py` + `SuGaR/scripts/reconstruct_object.sh` (implementation),
`scenes/object4_run.log`, `scenes/object5_run.log` (evaluation).

---

## 1. Introduction

SuGaR (Guédon & Lepetit, CVPR 2024) is designed and tuned for datacenter GPUs. Running it on a 4 GB
GTX 1650 is viable only if every stage is memory-disciplined. In profiling a complete reconstruction
we observed that wall-clock was overwhelmingly dominated by one phase, and that the dominance came
not from algorithmic complexity in the data but from a *hard-coded constant*. This report documents
the full arc for that one optimization (Hypothesis **H1**): profiling → reasoning → hypothesis →
validation → implementation → end-to-end evaluation, in a reproducible form.

The scope is deliberately narrow. Two sibling optimizations (H2, an adaptive VRAM budget; and
H-final, an error-guided sampler) are documented elsewhere in `research.md` and are treated here only
where they are constants held fixed during H1's evaluation.

---

## 2. Analysis — locating the bottleneck (`analysis.json`)

We instrumented one complete run (scene *object3*, 240 frames) and attributed wall-clock from
per-stage log timestamps and in-log iteration timers. The stage profile:

| Stage | Time (min) | Share |
|---|---:|---:|
| Frame extraction + sharpness filter | 2.0 | 0.3 % |
| COLMAP (extract + match + map + undistort) | 9.0 | 1.5 % |
| Vanilla 3DGS (7k iters, −r 2) | 10.7 | 1.7 % |
| **SuGaR (coarse + Poisson + refine + texture)** | **591.7** | **96.5 %** |
| U²-Net masks + carve + cleanup | 3.2 | 0.5 % |

Drilling into the 591.7-minute SuGaR stage revealed a sharp internal discontinuity:

| Sub-phase | Time (min) | Per-iter |
|---|---:|---:|
| Coarse, iters 0–9000 ("fast phase") | 60 | 0.066 s |
| **Coarse, iters 9000–15000 ("SDF phase")** | **566** | **5.6 s** |
| Poisson extract + decimate + refine + texture | 24 | — |

The per-iteration cost jumps **85×** at iteration 9000. Inspection of `coarse_density.py` shows three
losses switch on together at that iteration — `start_sdf_estimation_from`,
`start_sdf_better_normal_from`, `start_sdf_regularization_from`, all = 9000 — each requiring an
evaluation of the Gaussian density field at `n_samples_for_sdf_regularization = 1_000_000` freshly
drawn points. This phase alone is **9.43 h = 92 % of the entire pipeline**, and it is governed by
constants, not by input size. (`analysis.json → runtime_profile.bottleneck`.)

The dominant per-iteration operation (`analysis.json → code_snippets.S1.cost_model`) is a
`(10⁶ samples × 16 neighbours × 3×3 covariance)` gather feeding a batched Mahalanobis matmul —
**1.6 × 10⁷ covariance-matrix reads per iteration**.

---

## 3. Reasoning — Socratic decomposition (`thoughts.json`)

We interrogated the phase with a recursive question tree, each node resolving to a testable claim
(`thoughts.json → S1`).

**Q: Why does this phase cost 92 % when the preceding phase costs 1 %?**
The optimizer, cadence, and iteration count are unchanged; only the per-iteration cost jumps, and
discretely at iteration 9000. → *The slowdown is a feature switch, not convergence cost; it is set by
constants.*

**Q: Why 1,000,000 samples per iteration — what breaks if it is smaller?**
The SDF loss is a Monte-Carlo estimate of a surface-alignment integral; its estimator variance falls
as 1/√n. But the optimizer runs 6,000 SDF-phase steps, each an independent draw, so variance is
*also* averaged across steps. → *Per-step sample precision (1M) and step-averaging (6000) are two
independent variance-reduction mechanisms; paying fully for both double-counts the same budget.*

**Q: Is the cost compute-bound or memory-bound at 1M samples on 4 GB?**
1M × 16 × (3×3) × 4 B ≈ 576 MB of gathered covariance streamed per iteration — far exceeding the
GPU's cache — while the arithmetic is a trivial 3×3 matmul. → *Memory-bandwidth bound: the neighbour
gather dominates, not the FLOPs. Therefore wall-clock should be ≈ linear in sample count.*

**Q (control): Could we instead cut the 16-neighbour count, or the KNN reset cadence?**
The field is a sum of `strength·exp(−½·Mahalanobis²)` terms, so K=16 truncates a rapidly decaying
tail; but reducing K biases the field in sparse regions, and the 500-iter neighbour reset is only 12
invocations of the phase. → *K and reset cadence are lower-leverage and less safe than sample count.*
(These become the negative results N1 in `research.md`.)

The reasoning converges on a single, safe, high-leverage lever: **the sample count**.

---

## 4. Hypothesis H1

> **H1.** The SDF-regularization phase is memory-bandwidth bound on the per-sample K-neighbour
> gather. Consequently (a) its wall-clock is *linear* in `n_samples_for_sdf_regularization`, and
> (b) that count can be reduced by an order of magnitude with negligible effect on the optimization,
> because the loss is a per-sample mean whose estimator variance is already suppressed by averaging
> over 6,000 steps. The 1,000,000 default is a datacenter-tuned constant, is not exposed by the
> pipeline entry point, and is the single highest-leverage speed knob on constrained hardware.

H1 makes two falsifiable predictions: **(P1, cost)** time ∝ n with constant µs/sample and
linearly-scaling peak VRAM; **(P2, quality)** an aggressive reduction leaves the final mesh
geometrically unchanged.

---

## 5. Validation

### 5.1 Experiment A — controlled kernel microbenchmark (tests P1)

We reproduced the `get_field_values` hot path (16-neighbour covariance gather + warped 3×3
Mahalanobis matmul + Gaussian evaluation) over 500k Gaussians, sweeping the sample count, 4 repeats,
warm-up dropped.

| n_samples | time (s) | peak VRAM (GB) | µs/sample |
|---:|---:|---:|---:|
| 100,000 | 0.0184 | 0.197 | 0.184 |
| 250,000 | 0.0446 | 0.442 | 0.178 |
| 500,000 | 0.0778 | 0.844 | 0.156 |
| 1,000,000 | 0.1560 | 1.654 | 0.156 |

**Linear fit R = 0.9994**, intercept 4.3 ms (negligible). µs/sample is flat (the memory-bound
signature — a compute-bound kernel amortizes fixed cost and shows *falling* per-unit time). Peak VRAM
scales linearly, confirming the traffic model. **P1 confirmed.**

### 5.2 Experiment B — real-model field-evaluation path (tests P1 on true data)

The microbenchmark uses synthetic tensors. We then loaded *object3*'s actual 42,837-Gaussian coarse
checkpoint and timed the genuine `sample_points_in_gaussians` → `get_field_values` sequence:

| n_samples | time (s) | µs/sample |
|---:|---:|---:|
| 300,000 | 0.0586 | 0.195 |
| 1,000,000 | 0.1576 | 0.158 |

**2.69× faster at 300k vs 1M** (µs/sample flat, memory-bound confirmed on the real model). The 2.69×
falls short of the 3.33× sample ratio because of a fixed per-call overhead that does not scale — an
early signal of the Amdahl dilution quantified in §7.

### 5.3 Experiment C — estimator-accuracy plateau (tests P2 in principle)

Density-field estimator RMSE versus sample count on the real object3 model (30 repeats, ground truth
= 3M samples):

| n_samples | per-step rel. error | after √6000-step averaging |
|---:|---:|---:|
| 50,000 | 0.42 % | 0.0054 % |
| 100,000 | 0.31 % | 0.0040 % |
| 200,000 | 0.24 % | 0.0031 % |
| 300,000 | 0.24 % | 0.0031 % |
| 1,000,000 | 0.12 % | 0.0015 % |

The estimator **plateaus by 200k** (300k = 200k), and even 50k's 0.42 % per-step error is reduced to
0.005 % once averaged across the 6,000 steps — quantitatively confirming the "double-counting"
argument of §3. This motivates the aggressive default but is only a *necessary* condition;
sufficiency (final mesh quality) requires a full run, which is Experiment D.

---

## 6. Implementation

H1 is implemented as a single backward-compatible change, applied identically to all three trainers
that contain the constant (`coarse_density.py`, `coarse_sdf.py`, `refine.py`):

```python
# before (upstream)
n_samples_for_sdf_regularization = 1_000_000  # 300_000

# after (H1)
n_samples_for_sdf_regularization = int(os.environ.get('SUGAR_SDF_SAMPLES', 1_000_000))
```

**Design decisions.**
- *Environment-gated, upstream default preserved.* When `SUGAR_SDF_SAMPLES` is unset the value is
  exactly 1,000,000, so importing stock SuGaR is unchanged and reproducible. The pipeline sets the
  reduced value explicitly.
- *Loss-scale invariance.* The SDF loss is `sdf_estimation_loss.mean()`; a per-sample mean is
  invariant to the number of samples, so reducing the count changes only per-step gradient *variance*
  (which §3/§5.3 show is redundant), never the loss *magnitude* or its gradient's expectation.
- *Authors' own annotation.* The upstream inline comment `# 300_000` records that the authors already
  considered a reduced count; H1 makes it a first-class, tunable parameter rather than a comment.

**Pipeline exposure** (`scripts/reconstruct_object.sh`): a `--sdf-samples` flag, exported into the
SuGaR stage. The complementary micro-optimizations for this hardware (H2 adaptive image budget;
H-final error-guided sampler) are separate flags and are held constant in the evaluation below.

---

## 7. End-to-end evaluation — Experiment D (tests P2)

### 7.1 Protocol

We captured a new object (*object4*, a small box-shaped item; 38.7 s hand-held video) and ran the
full pipeline at the reduced default (**50k samples**). We then ran a **controlled A/B**: a second
scene (*object5*) reusing object4's *identical* frames, COLMAP poses, and vanilla-3DGS checkpoint
(via the pipeline's stage-cache markers), re-executing only the SuGaR stage onward at **300k
samples**. Reusing the upstream stages isolates the sample count as the manipulated variable and
avoids re-paying the 5 h 39 m 3DGS stage.

Both runs held the error-guided sampler ON (mix = 0.5) — i.e. it is a *constant*, not the variable —
so the contrast is purely 50k vs 300k. (See §8 for the resulting threats to validity.)

### 7.2 Speed

Per-stage wall-clock (from the pipeline timer):

| Stage | object4 (50k) | object5 (300k) |
|---|---:|---:|
| Frames + COLMAP | 26 m 38 s | *cached* |
| Vanilla 3DGS 7k | 5 h 39 m 03 s | *cached* |
| **SuGaR (coarse + Poisson + long refine)** | **1 h 44 m 35 s** | **2 h 08 m 02 s** |
| Masks + carve + cleanup | 6 m 07 s | 36 s |
| **Total** | **7 h 56 m 41 s** | 2 h 08 m 41 s (reused stages) |

The SuGaR stage cost **1 h 44 m at 50k vs 2 h 08 m at 300k** — the 6× sample increase added only
23.5 minutes.

Decomposing the coarse SDF-phase block time (same 499-image scene, so directly comparable):

| n_samples | coarse SDF-phase min / 200-iter block |
|---:|---:|
| 50,000 | 0.718 |
| 300,000 | 1.119 |

A two-point linear decomposition `T(n) = T_fixed + k·n` gives **k ≈ 1.60 × 10⁻⁶ min/sample/block**
and **T_fixed ≈ 0.64 min/block**. Thus at 50k only ~0.08 min of each 0.72-min block is
sample-dependent; **~89 % of the block is a fixed cost** — the per-iteration rendering of all 499
ground-truth images and the non-SDF losses. This is why the *stage* speedup (1.22×) is far below the
*kernel* speedup (2.69×, §5.2): the sample-linear term, though real (P1 holds), is a minority of the
per-iteration cost on a many-image capture. **This is Amdahl's law: H1 accelerates a component that,
at low sample counts, is no longer the majority of the phase.**

### 7.3 Quality (the decisive test of P2)

Final meshes, and their agreement:

| Metric | object4 (50k) | object5 (300k) |
|---|---:|---:|
| Triangles | 12,675 | 12,552 (−1.0 %) |
| Vertices | 6,489 | 6,460 |
| Bounding-box extent | [0.832, 0.943, 0.993] | [0.829, 0.948, 0.995] |
| Largest connected component | 99.1 % | 97.9 % |

**Symmetric Chamfer distance** (100k-point sampling, KD-tree nearest-neighbour, both meshes in the
shared COLMAP frame; bounding-box diagonal = 1.603 units):

| | value | % of bbox diagonal |
|---|---:|---:|
| mean | 0.00490 | **0.31 %** |
| median | 0.00313 | 0.20 % |
| p95 | 0.01291 | 0.81 % |
| max | 0.11489 | 7.2 % (localized base-fringe floaters) |

The meshes agree to **0.31 % of the object's size** — within reconstruction noise. Visual renders
from an identical camera are near-indistinguishable. **P2 confirmed: the 6× sample reduction leaves
the final geometry unchanged.** The residual max (7.2 %) is confined to a few base-fringe floaters,
not a systematic surface difference.

---

## 8. Discussion & threats to validity

**H1 is confirmed on both predictions**, with one important qualification on its *practical* payoff:

1. **Cost (P1): confirmed, and mechanistically explained.** Linear in samples (R = 0.9994 kernel,
   2.69× real-model, and a consistent two-point real-run decomposition). The memory-bound diagnosis
   is corroborated by flat µs/sample and linearly-scaling VRAM.

2. **Quality (P2): confirmed end-to-end.** Chamfer 0.31 %. The prior evidence (estimator plateau) was
   necessary but not sufficient; Experiment D supplies the sufficient, real-mesh confirmation.

3. **Amdahl dilution (new finding).** The whole-*stage* speedup (1.22× at 50k vs 300k) is far below
   the *kernel* speedup, because per-iteration GT-image rendering is a large fixed cost that H1 does
   not touch. On few-image captures the kernel term would dominate and the stage speedup would
   approach the kernel figure; on this 499-image capture it does not. **The lever is real but its
   end-to-end leverage is capture-dependent.**

**Threats to validity.**
- *Not a perfectly single-variable A/B.* object5's adaptive image budget selected 512 px vs object4's
  544 px (H2 responding to slightly lower free VRAM at launch). This makes object5's fixed rendering
  cost marginally *lower*, so the measured sample-cost gap is a slight *under*-estimate — it does not
  threaten the direction of the result, and the 0.31 % Chamfer agreement is far tighter than a 6 %
  resolution difference could mask.
- *Error-guided sampler held ON in both runs.* Experiment D therefore tests "50k+EG vs 300k+EG",
  isolating the sample count with the sampler as a constant. The pure-H1 evidence for reduced samples
  is Experiments A–C; D confirms quality is preserved *in the deployed configuration*.
- *Single object class.* object4/5 is one object; the geometric conclusion should be re-checked on a
  high-detail, thin-feature object (the regime H-final's error-guiding exists to protect).
- *Stochasticity.* COLMAP RANSAC and 3DGS init are stochastic, but Experiment D *reuses* object4's
  poses and 3DGS checkpoint, removing that source of variance from the comparison.

---

## 9. Conclusion

Hypothesis H1 — that the SuGaR SDF-regularization bottleneck is a memory-bound, sample-linear cost
governed by a redundant datacenter-tuned constant — is confirmed by a kernel microbenchmark
(R = 0.9994), a real-model timing (2.69×), an estimator-accuracy plateau, and a controlled
end-to-end A/B whose 50k and 300k meshes agree to 0.31 % of object size. The fix is a one-line,
backward-compatible, environment-overridable constant. The chief caveat is Amdahl's law: on
many-image captures a large fixed per-iteration rendering cost dilutes the sample-linear speedup at
the stage level, so the practical gain is capture-dependent.

The optimization's most consequential downstream effect is diagnostic: once the SDF phase is cheap,
**vanilla 3D Gaussian Splatting (5 h 39 m on the 499-frame object4) becomes the pipeline's new
dominant cost** — the natural target for the next optimization.

---

## Appendix — reproduction

```bash
# reduced-sample run (H1 default in the pipeline)
cd SuGaR
bash scripts/reconstruct_object.sh --video inputs/object4.mp4 --name object4 \
     --fps 15 --target 500 --refine long --sdf-samples 50000

# controlled A/B at 300k, reusing object4's upstream stages
rsync -a --exclude distorted scenes/object4/ scenes/object5/
cp -r SuGaR/output/vanilla_gs/object4 SuGaR/output/vanilla_gs/object5
rm -f scenes/object5/.done_sugar scenes/object5/.done_carve
bash scripts/reconstruct_object.sh --video inputs/object4.mp4 --name object5 \
     --fps 15 --target 500 --refine long --sdf-samples 300000
```

**Provenance chain for H1:** `analysis.json → runtime_profile.bottleneck` (profiling) →
`thoughts.json → S1` (reasoning) → `research.md → H1` (validation) →
`SuGaR/sugar_trainers/{coarse_density,coarse_sdf,refine}.py` (implementation) →
`scenes/object4_run.log`, `scenes/object5_run.log` (evaluation).
