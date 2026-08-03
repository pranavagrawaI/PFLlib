# Independent Verification Protocol

The verification target has four layers. Passing a later layer does not repair a
failure in an earlier one.

## Layer 1: Archive integrity and inventory

From the archive root:

```bash
sha256sum -c SHA256SUMS
python3 -m compileall -q analysis dataset experiments paper system
find experiments -maxdepth 1 -type f -name '*.sh' -print0 |
  xargs -0 -n1 bash -n
```

Inspect `IMPLEMENTATION_SPEC.md` before reading the production implementation.
This keeps the specification from becoming a paraphrase accepted only because
it resembles the code.

Acceptance:

- every checksum passes;
- Python compilation and shell parsing succeed;
- both adaptive client/server pairs and both baseline client/server pairs are
  present.

## Layer 2: Controller mathematics independent of production classes

Run:

```bash
python3 analysis/verify_controller_spec.py
```

This invokes only the dependency-free controller in
`analysis/reference_controller.py`. It checks:

- zero, negative, and clipped positive gaps;
- target, smoothing, bounds, and warmup order;
- missing-reference behavior;
- client-local state persistence; and
- median-plus-EMA server reference updates.

It then checks every row in the seven archived production controller traces
against the specification and independently reconstructs the server reference
sequence from client probe losses. The trace set must contain both AdaDitto and
AdaFedProx; otherwise the verifier fails.

This is a conformance check, not evidence of task accuracy.

## Layer 3: Behavioral equivalence and falsification

Implementors should run these controlled comparisons with matched initialization,
client selection, minibatch order, and seeds:

1. **Fixed-coefficient reduction.** Set `controller_mode=fixed_base` with
   `mu_base=x`. The proximal update must match the corresponding fixed Ditto or
   FedProx implementation at coefficient `x`, modulo controller probing and
   logging overhead.
2. **Zero-gain reduction.** Set `alpha_gain=0`, `mu_min=mu_base`, and complete
   warmup. The coefficient must remain constant at `mu_base`.
3. **No-reference first round.** All selected clients must receive zero gap
   boost; final coefficients must equal `mu_min` during warmup.
4. **Synthetic monotonicity.** Holding previous coefficient and reference fixed,
   increasing the probe loss must not decrease the next coefficient before a
   bound is reached.
5. **Bound stress.** Extreme positive gaps must never move the full controller
   outside `[mu_min,mu_max]`.
6. **Participation persistence.** A non-selected client's coefficient must
   remain unchanged.
7. **Reference lag.** Clients in round `t` must use the reference computed after
   round `t-1`, never the median from their own round.
8. **Selected-only controller state.** Non-selected clients may receive the
   global model for correct evaluation, but must not perform controller or local
   training transitions.

Falsification is simple: any violation is an implementation failure even if
final accuracy looks plausible.

## Layer 4: Empirical evidence

First recompute archived aggregates without training:

```bash
python3 analysis/verify_reported_results.py
```

Then regenerate the figures:

```bash
python3 paper/scripts/make_current_figures.py
```

Finally, rerun experiments using the commands in `README.md`. Judge each claim
against its evidence class:

- coefficient sweeps identify regimes but single-seed rows are not significance
  evidence;
- tune-once shift comparisons use matched seeds and test deployment staleness;
- per-condition sweeps are oracle comparisons, not the deployment baseline;
- controller scans test default sensitivity;
- AdaFedProx parity is a scoped negative result, not an improvement claim.

The claim-to-file map is in `CLAIMS_TO_ARTIFACTS.md`.

## Expected archived checks

The archived verifier should recover:

- CIFAR full-participation fixed-Ditto optima: practical `lambda=0.5`,
  pathological `lambda=0`;
- CIFAR-pathological 20% tune-once means: Ditto `0.5512`, AdaDitto `0.5692`,
  a `+1.80` point difference;
- CIFAR-pathological 20% per-condition means: tuned Ditto `0.5865`, AdaDitto
  `0.5721`;
- Shakespeare 10% means: tuned Ditto `0.4822`, AdaDitto `0.4820`; and
- controller-scan mean ranges: CIFAR `0.5627--0.5743`, Shakespeare
  `0.4819--0.4822`.

Small retraining differences across hardware and dependency builds are not by
themselves a failure. A change in comparison direction, coefficient regime,
state-transition invariants, or matched-seed interpretation is material and
must be explained.
