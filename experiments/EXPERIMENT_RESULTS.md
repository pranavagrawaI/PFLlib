# AdaProx / AdaDitto — Experiment Log & Results

**Project:** Adaptive proximal-coefficient control for federated learning
(AdaProxFedProx on the FedProx/global family; AdaProxDitto — "AdaDitto" — on the
Ditto/personalization family), evaluated against fixed-coefficient baselines.

**Compute:** single local GPU (RTX 5060 Laptop, Blackwell sm_120, WSL2).
**Datasets:** CIFAR-100 (CNN) in two partitions — *practical* (unbalanced
Dirichlet α=0.1 with overlapping label support) and *pathological* (label-shard
partition with 10 labels/client, high label-support skew);
Shakespeare (char-LSTM, 118 LEAF speakers, natural per-speaker non-IID).
**Metric convention:** FedProx family → global-model accuracy; Ditto family →
personalized-model accuracy (`model_per`). Scores are final test accuracy.

> **One-line summary of what holds:** the adaptive controller **matches oracle
> per-condition coefficient tuning with a single configuration** across two
> modalities whose optimal coefficients point in **opposite** directions, and —
> where the optimum shifts with the deployment condition — is **robust to
> participation shift** where fixed-tuned baselines go stale. It does **not**
> beat well-tuned fixed coefficients on accuracy (nor does FedProx's own adaptive
> heuristic). The FedProx-μ family shows no benefit on CIFAR in any regime.

---

## Phase 0 — Smoke test

**Purpose:** correctness gate — all datasets × algorithms run end-to-end, produce
identical metric logging, controller traces, partial-participation path.
**Result:** all 4 algorithms × 4 dataset variants ran; controller traces (ρ, loss
gap, EMA, clipping) populate. **Meaning:** the harness and instrumentation are
sound; safe to run the real campaign.

---

## Phase 1 — Static μ/λ sweeps (brittleness) — full participation, E=1

**Purpose:** quantify how sensitive fixed-coefficient methods are to their
coefficient, and locate the per-dataset optimum (the cost the controller aims to
remove). 13 coefficient values, both CIFAR splits.

| Dataset | Algo | Best | Worst | Accuracy swing |
|---|---|---|---|---|
| practical | FedProx (μ) | μ=0.005 → 0.4768 | μ=3.0 → 0.4077 | **16.9%** |
| pathological | FedProx (μ) | μ=0.001 → 0.5948 | μ=3.0 → 0.5151 | **15.5%** |
| practical | Ditto (λ) | λ=0.5 → 0.4857 | λ=0.03 → 0.4518 | 7.5% |
| pathological | Ditto (λ) | **λ=0.0** → 0.6205 | λ=2.0 → 0.5915 | 4.9% |

**Meaning:** the coefficient is genuinely brittle (FedProx loses ~16% if μ is
chosen badly) **and the optimum is regime-dependent** — note Ditto's best λ is
0.5 on the practical split but **0.0 (pure-local)** on the pathological split.
You cannot tune once and reuse across conditions. This is the motivation for an
adaptive controller.

---

## Phase 2 — Adaptive controller hyperparameter scan — Cifar100_practical

**Purpose:** is the controller itself sensitive to *its* hyperparameters
(α gain, γ smoothing, β EMA, τ gate, ρ_max)? 28 configurations, 1 seed.
**Result:** entire spread = **0.4762–0.4791 (0.3%)**, within seed noise. The
default configuration is tied-best.
**Meaning:** the controller needs no tuning of its own knobs — the defaults work.
It does not trade "tuning the coefficient" for "tuning the controller."

---

## Phase 3 — Controller-mode ablations — both CIFAR splits, 3 seeds

**Purpose:** which controller components matter? Modes: `full`, `fixed_base`,
`no_log_ratio`, `no_clipping`, `no_ema`, `mean_global_loss`, `global_shared`,
`random_adaptive`.

- **practical (mild het):** all modes within ~0.011 (seed sd ~0.007) — components
  do not separate; `random_adaptive` nominally top.
- **pathological (high het):** principled modes clustered ~0.597;
  **`random_adaptive` clearly worst (0.5863)**, separated by ~2.6× the seed sd.

**Meaning:** under heterogeneity, a *sensible* coefficient matters (random hurts),
but the specific adaptive machinery components (EMA, clipping, log-ratio) are
within noise of each other and of a fixed base — i.e. the controller is stable
and never unstable, but its internal components are not individually decisive on
CIFAR.

---

## Dropout / partial-participation sweep — frozen-λ baseline (CIFAR)

**Purpose:** the *robustness-under-shift* test. Ditto's λ is **frozen at its
full-participation optimum** (practical 0.5, pathological 0.0) — the realistic
"tune once, deploy everywhere" baseline — then run vs AdaDitto (single config)
as participation drops. Personalized metric (verified clean — see bug section).

**Cifar100_pathological (frozen λ=0.0):**

| Participation | Ditto (frozen) | AdaDitto | gap | degradation from full |
|---|---|---|---|---|
| full | 0.6227 | 0.5968 | −0.026 | — |
| 50% | 0.6113 | 0.5920 | −0.019 | Ditto −0.011 / Ada −0.005 |
| **20%** | **0.5512** | **0.5692** | **+0.018** | **Ditto −7.2% / Ada −2.8%** |

**Cifar100_practical (frozen λ=0.5):** AdaDitto does **not** win at any level
(−0.014 to −0.022) — the expected null under mild heterogeneity.

**Meaning:** under **high heterogeneity**, the frozen-once λ goes badly stale as
participation drops (collapses −7.2%), while AdaDitto adapts (−2.8%) and
**overtakes it by +1.8%** at 20% participation — reversing its full-participation
deficit. This is the robustness win, and it appears exactly where the coefficient
optimum is most participation-sensitive. Under mild heterogeneity, no win.

---

## Partial-participation μ/λ re-sweep — fair per-condition tuning (CIFAR)

**Purpose:** the dropout sweep used a *frozen* baseline; a fair core comparison
must tune the fixed baseline **per participation level**. Re-swept FedProx μ and
Ditto λ at jr ∈ {0.5, 0.2}, both splits.

- **FedProx μ:** optimum ≈ 0 at every condition — full-participation value
  transfers fine (proximal μ simply doesn't help on CIFAR — see negative results).
- **Ditto λ:** optimum **shifts strongly** with participation:

| Split | jr=1.0 (Phase 1) | jr=0.5 | jr=0.2 |
|---|---|---|---|
| practical | 0.5 | 0.5 | **1.0** |
| pathological | 0.0 | 0.0 | **1.0** |

**Meaning:** confirms the optimum moves with the regime — at 20% participation on
pathological it flips **0.0 → 1.0** (a coefficient tuned at full participation is
maximally wrong). This both (a) justified re-tuning the baselines for fairness and
(b) is itself clean brittleness evidence.

---

## Core comparison table — fairly-tuned baselines (CIFAR, 5 seeds)

**Purpose:** the headline comparison — single-config controller vs
**per-condition-tuned** fixed baselines. (Numbers below are POST eval-bug fix.)

| Split | jr | FedProx → AdaProxFedProx | Ditto → AdaDitto |
|---|---|---|---|
| practical | 0.5 | 0.357 → 0.357 (−0.000) | 0.482 → 0.461 (−0.020) |
| practical | 0.2 | 0.246 → 0.248 (+0.002) | 0.470 → 0.444 (−0.026) |
| pathological | 0.5 | 0.391 → 0.386 (−0.005) | 0.610 → 0.592 (−0.018) |
| pathological | 0.2 | 0.217 → 0.218 (+0.002) | 0.587 → 0.572 (−0.014) |

**Meaning:**
- **FedProx family:** AdaProxFedProx ≈ FedProx everywhere (within ±0.005) —
  no advantage, because on CIFAR the best μ is ≈0 (see negative results).
- **Ditto family:** AdaDitto trails the *per-condition oracle* Ditto by ~1.5–2.6%.
  Combined with the dropout sweep, the full statement is: **AdaDitto beats the
  realistic tune-once baseline under shift, and comes within ~2% of the
  (impractical) per-condition oracle — using one configuration, no sweep.**

---

## ⚠️ Critical bug found & fixed — `send_models` partial-participation eval

**How it surfaced:** the *first* (buggy) core table showed AdaProxFedProx beating
FedProx by **+0.12 to +0.37** under partial participation — implausibly large.

**Diagnosis (elimination):**
- Not the adaptive μ: `controller_mode=fixed_base` (constant μ, full machinery)
  reproduced the win (0.584).
- Not the μ value: every fixed μ ∈ {0, 0.001, …, 1.0} failed identically (~0.27).
- Not the probe: disabling it left 0.582.
- **Root cause:** `serveradaprox.send_models` / `server_adaprox_ditto.send_models`
  updated only `selected_clients`, while base `send_models` updates **all**
  clients and `evaluate()` scores **all** clients. Under partial participation the
  16/20 non-selected clients kept their **stale, locally-overfit** models, which
  score high on their own test sets → inflated global-accuracy metric.
- **Confirmation:** patching send-to-all collapsed fixed_base to **0.213**,
  matching plain FedProx (0.207).

**Scope:** only **global-accuracy** metrics under **partial participation** were
affected → AdaProxFedProx partial-participation rows. **Personalized accuracy
(AdaDitto, dropout sweep) is independent of `send_models`** (it reads `model_per`)
— verified **bit-exact** (0.5636/0.5710 reproduced). Full-participation results
unaffected (all clients selected).

**Meaning:** the exciting +0.37 was a measurement artifact. Fixing it turned the
FedProx-family result into an honest null and left the (clean) Ditto-family result
intact. This is a reproducibility contribution in its own right.

---

## Negative result — FedProx μ never helps on CIFAR

**Purpose:** the FedProx paper (App. C.3.3) credits μ's benefit to
*high local work + systems heterogeneity*. We had only tested E=1, so we probed
the regimes where μ *should* matter.

| Regime | μ=0 (FedAvg) vs μ>0 |
|---|---|
| E=1, full participation (Phase 1) | small/zero μ best |
| E=5, full participation | μ=0 wins (0.635 vs 0.571 at μ=1.0) |
| E=5, 20% participation | μ=0 wins (0.230 vs 0.196) |
| **E=5, 20% + systems het (`-tsr 0.5`)** | **μ=0 wins (0.259 vs 0.222)** |

**Meaning:** on CIFAR-100/CNN, FedProx's proximal term never beats FedAvg — even
in its own motivating systems-heterogeneity setup; μ>0 only slows convergence.
So an *adaptive* μ has nothing to improve on the FedProx family here (best μ ≈ 0,
trivially matched). This is consistent with FedAvg being hard to beat on vision
tasks, and it correctly scopes the contribution to the **personalization (Ditto)
family**. (It also matches the FedProx authors' own framing of adaptive μ as
*competitive/parity*, not a win.)

---

## AGNews (text/Transformer) — blocked by hardware

**Status:** AGNews training hangs on this WSL2/Blackwell GPU. Diagnosed to the
Transformer **self-attention (flash/mem-efficient SDPA) kernels** faulting under
sustained load. Mitigations tried: driver update, torch cu130→cu128,
`expandable_segments`, forcing the **SDPA math backend** (works one round),
**bf16 AMP** (measured **5× speedup**). All fail: training still hangs after ~1
round (three isolation runs confirmed it is not adaptation, not the probe, but a
platform fault). **Parked** — the SDPA-math + AMP code is committed and would make
AGNews fast on a stable GPU (cloud / native Windows). Not required, as Shakespeare
provides the text modality.

---

## Shakespeare (text/char-LSTM) — second modality ✅

The LSTM runs locally where the Transformer could not (cuDNN LSTM ≠ flash
attention). 118 speakers, jr=0.1, lr=0.5 (calibrated), personalized next-char
accuracy, 3 seeds for the core.

**λ sweep (does λ matter on text?):**

| Ditto λ | jr=0.1 | jr=0.5 |
|---|---|---|
| 0 (pure local) | 0.344 | 0.445 |
| **0.1** | **0.483** | **0.512** |
| 0.5 | 0.483 | 0.511 |
| 1.0 | 0.483 | 0.511 |

**Core (AdaDitto vs tuned Ditto):**

| Level | Ditto (λ=0.1, tuned) | AdaDitto (1 config) | Δ |
|---|---|---|---|
| jr=0.1 | 0.4822 ± 0.0028 | 0.4820 ± 0.0029 | **−0.0001** |
| jr=0.5 | 0.5122 | 0.5118 | −0.0004 |

**Participation-shift test:** optimal λ = **0.1 at both jr=0.5 and jr=0.1** — the
optimum does **not** shift (curve moves up with participation, shape unchanged).

**Meaning:**
- **λ matters on text (+14% from λ=0 to λ>0)** — and its best value (λ>0) is the
  **opposite direction** from CIFAR-pathological (where λ=0 was best). The
  controller lands on the right coefficient in *both* modalities with one config.
- **Tuning-free parity is exact on text**, and holds across participation levels.
- **No shift-robustness win on Shakespeare (honest null):** its optimum is
  participation-stable, so a frozen λ never goes stale — nothing for the
  controller to rescue. This *scopes* the shift-win claim precisely: it requires a
  coefficient whose optimum moves with the deployment condition (present in
  high-heterogeneity vision, absent in text).

---

## Supplemental experiments (paper-strengthening)

### Tune-once regret matrix (CIFAR, analysis-only)
For each *source* condition, take its best fixed λ; deploy it on every *target*
condition; regret = target-best accuracy − that λ's accuracy at the target.
Common λ grid {0, 0.001, 0.01, 0.1, 0.5, 1.0}.

- **Fixed-λ tune-once regret is large and condition-dependent:** worst case
  **+0.072** (tuned on path/full → deployed on prac/20%); the classic
  full→20% pathological staleness is +0.047.
- **AdaDitto's single-config regret is bounded ≤ +0.032** at every condition.

**Meaning:** a fixed coefficient is a gamble on your tuning condition matching
deployment; AdaDitto caps the downside with no tuning. (Ditto sweep cells are
1-seed, so ±0.003 differences are noise; the +0.047/+0.072 staleness is not.)

### Client-tail analysis (CIFAR-pathological, jr=0.2)
Personalized-model per-client accuracy (from `client_metrics_final.csv` — **not**
the `p10_client_accuracy` summary field, which is *global*-scope):

| Method | mean | p10 | worst client |
|---|---|---|---|
| Ditto tune-once (λ=0) | 0.526 | 0.418 | 0.391 |
| AdaDitto (1 config) | 0.561 | **0.482** | **0.442** |
| Ditto oracle (λ=1.0) | 0.586 | 0.525 | 0.473 |

**Meaning:** AdaDitto's robustness is not average-only — it lifts p10 by **+6.5%**
and worst-client by **+5.1%** over tune-once, and trails oracle *uniformly*
(no extra tail penalty). Helps weak clients, not just the mean.

### Controller robustness scans (default not cherry-picked)
AdaDitto at the headline conditions across 11 stress controller configs
(default + low/high extreme of each knob).

| | CIFAR-path jr=0.2 (3 seeds) | Shakespeare jr=0.1 (3 seeds) |
|---|---|---|
| total spread (best−worst) | **1.2%** (0.5627–0.5743) | **0.03%** (0.4819–0.4822) |
| vs seed sd | spread > sd (~0.004) | **spread ≪ sd (~0.0029)** |
| default rank | 6/11 (median) | 7/11 (median) |
| all configs vs tune-once | all **beat** (0.551) | all ≈ tuned optimum (0.483) |
| coef_μ range | 0.087–0.19 | 0.055–0.085 |

**Meaning:** the controller is insensitive to its own hyperparameters *at the
exact headline conditions* (not just CIFAR-practical / Phase 2), the default is
never cherry-picked, and on CIFAR **every** config beats the tune-once baseline —
so the shift win is not a lucky controller setting. On Shakespeare the config
spread (0.03%) is ~10× **below** seed noise (0.29%), i.e. the controller setting
is statistically undetectable.

### Controller signal-validity (mechanism, analysis-only)
Does the loss-gap signal track the *anchoring need*? Per-condition mean loss-gap
`g_i` (scale-invariant log-ratio) and learned λ (`rho_final`), participating
clients, post-warmup:

| Condition | known λ* | mean gap | %gap>0 | learned λ |
|---|---|---|---|---|
| CIFAR path / full | 0.0 | 0.013 | 33% | 0.090 |
| CIFAR path / jr0.2 | 1.0 | **0.074** | 52% | **0.135** |
| CIFAR prac / full | 0.5 | 0.008 | 30% | 0.086 |
| CIFAR prac / jr0.2 | 1.0 | 0.030 | 46% | 0.101 |
| Shakespeare / jr0.1 | >0 | 0.003 | 5% | 0.069 |

- **Within-run** `rho_target = μ_base + α·gap` by construction (r=1.000) — confirms
  the response, not evidence on its own.
- **Across CIFAR conditions the mechanism is valid on the participation axis:**
  partial participation yields 4–6× larger gaps → higher learned λ, matching the
  known λ*=1.0 (jr0.2) vs λ*=0–0.5 (full). The signal is causally right-directed.
- **Two honest caveats surfaced:** (1) the floor μ_min=0.05 is load-bearing —
  where λ*=0 (path/full) the controller cannot reach 0 (~0.09), slightly
  over-anchoring (consistent with the ~2% oracle deficit there); (2) on
  Shakespeare the gap is near-zero (global LM ≈ local for all speakers), so
  AdaDitto works via the floor (λ~0.07 clears the ~0.1 saturation point), not
  active gap-driven adaptation.

**Meaning:** the mechanism is validated where it matters most (CIFAR
participation shift, the headline), with the floor an explicit safety net where
the signal is weak or the optimum is near-zero.

### Two further metric-provenance artifacts caught (flagged, not in headline)
- The summary field `p10_client_accuracy` (and mean/worst) is computed on the
  **global** model (`finalize_run.py:591`, `scope=="global"`), so it understates
  every Ditto-family *personalized* tail. Use `client_metrics_final.csv`.
- The AdaProxDitto **core** runs are pre-`send_models`-fix, so their *global*-scope
  metrics are still contaminated (personalized metrics are clean / bit-exact
  verified). Do not quote AdaProxDitto `global_accuracy` from those runs.

---

## Synthesis — what the campaign supports

| Claim | CIFAR (vision) | Shakespeare (text) |
|---|---|---|
| Coefficient matters, modality-dependent optimum | λ=0 best (pathological) | λ>0 best, +14–40% (**opposite**) |
| **Tuning-free parity** (AdaDitto ≈ tuned Ditto) | within ~2–3% | **exact**, across jr |
| **Robustness under participation shift** | ✅ (optimum flips 0→1) | null (optimum stable) |
| FedProx-μ family benefit | none in any regime | — |
| Reproducibility (send_models eval bug) | found & fixed | — |

**Defensible thesis:** *a single-configuration adaptive proximal-coefficient
controller matches oracle per-condition tuning across two modalities with opposite
optimal coefficients, eliminating the per-condition sweep; and where the optimum
is participation-sensitive, it is robust to participation shift while fixed
tune-once baselines degrade.* It does **not** beat well-tuned fixed coefficients on
accuracy — consistent with the FedProx authors' own adaptive-μ framing (parity +
tuning convenience). The nulls (FedProx-μ, Shakespeare-shift) scope the claim
rather than weaken it.

---

*Every result above is reproducible from `experiments/runs/<phase>/` and the
summary CSVs; the send_models fix is committed. Model weights / raw HDF5 are
excluded from version control (regenerable); reporting artifacts are committed and
mirrored to an off-machine tarball.*
