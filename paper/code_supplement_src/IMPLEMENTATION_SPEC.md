# AdaDitto and AdaFedProx Implementation Specification

## 1. Scope and conformance

This document specifies the controller shared by AdaDitto and AdaFedProx. An
implementation conforms when it preserves:

1. the baseline Ditto or FedProx optimization and aggregation semantics;
2. the timing and state transition defined below;
3. the default constants and edge-case behavior;
4. one client-local coefficient state per client; and
5. the scalar client-to-server and server-to-client controller messages.

The reference implementation in `analysis/reference_controller.py` is
dependency-free and intentionally separate from the training code.

## 2. Symbols and defaults

| Symbol | Meaning | Default |
|---|---|---:|
| \(c_i\) | client-local proximal coefficient; \(\lambda_i\) for AdaDitto and \(\mu_i\) for AdaFedProx | initialized to 0.05 |
| \(c_{\mathrm{base}}\) | coefficient target when the loss gap is zero | 0.08 |
| \(c_{\min},c_{\max}\) | coefficient bounds | 0.05, 3.0 |
| \(\alpha\) | loss-gap gain | 0.8 |
| \(\tau\) | upper clipping threshold for the nonnegative log gap | 0.7 |
| \(\gamma\) | client coefficient smoothing rate | 0.3 |
| \(\beta\) | server loss-reference EMA retention | 0.9 |
| \(\epsilon\) | numerical stabilizer | \(10^{-8}\) |
| \(r\) | probe fraction of the client's training examples | 0.1 |
| \(B_{\max}\) | maximum probe examples | 256 |
| \(T_w\) | warmup rounds | 2 |

Required parameter constraints are
\(0\leq\gamma\leq1\), \(0\leq\beta\leq1\),
\(0\leq c_{\min}\leq c_{\max}\), \(\alpha\geq0\), \(\tau\geq0\),
and \(r>0\). Probe and reference losses must be nonnegative finite values when
present.

The command-line names are `--mu_base`, `--mu_min`, `--mu_max`,
`--alpha_gain`, `--gap_tau`, `--mu_smooth_gamma`, `--ema_beta`,
`--loss_eval_sample_ratio`, and `--warmup_rounds`. The production code uses the
field name `mu` for both variants; in AdaDitto that field is the Ditto
coefficient \(\lambda\).

## 3. Persistent and transient state

Each client \(i\) stores:

- its baseline model state;
- for AdaDitto, a persistent personalized model \(v_i\);
- its last used coefficient \(c_i\), initialized to \(c_{\min}\);
- the most recently received server loss reference \(L_g\); and
- the global round index.

The server stores:

- the ordinary global model \(w\); and
- one scalar loss reference \(L_g\), initially absent.

A client that is not selected in a round does not update its coefficient. Thus
coefficient smoothing is over that client's participation events, even though
warmup is determined by the global round index.

## 4. Probe definition

At global round \(t\), after receiving \(w^{(t)}\) and before any local model
update, selected client \(i\):

1. puts the received global model in inference mode;
2. takes the first
   \[
   n_i^{\mathrm{probe}} =
   \max(1,\min(B_{\max},\lfloor r n_i\rfloor))
   \]
   examples yielded by its shuffled local training loader;
3. evaluates the ordinary task loss without gradients; and
4. reports the mean probe loss \(\hat L_i^{(t)}\).

This is not a held-out validation set and not a new communication payload of
examples. Only the scalar mean loss is reported. An independent implementation
may select the same number of examples by an equivalent deterministic sampling
rule, but exact trace reproduction requires the same seeded loader ordering.
The production loader uses `shuffle=True` and `drop_last=True`. If it yields no
batch, the production probe returns loss zero; conforming deployments should
instead ensure every client has at least one complete local batch.

## 5. Client controller transition

The full controller consumes client state \(c_i^{\mathrm{prev}}\), current probe
loss \(\hat L_i\), the previous server reference \(L_g\), and global round \(t\).

If \(L_g\) is absent or either input is non-finite, set \(g_i=0\). Otherwise:

\[
g_i^{\mathrm{raw}}
=
\log\frac{\hat L_i+\epsilon}{L_g+\epsilon},
\qquad
g_i
=
\min(\max(g_i^{\mathrm{raw}},0),\tau).
\]

Then compute:

\[
c_i^\star = c_{\mathrm{base}}+\alpha g_i,
\]
\[
\tilde c_i = (1-\gamma)c_i^{\mathrm{prev}}+\gamma c_i^\star,
\]
\[
\bar c_i = \min(\max(\tilde c_i,c_{\min}),c_{\max}).
\]

Warmup is the final override:

\[
c_i^{(t)} =
\begin{cases}
c_{\min}, & t<T_w,\\
\bar c_i, & t\geq T_w.
\end{cases}
\]

The client persists \(c_i^{(t)}\) as its next `previous` value and writes it into
the proximal optimizer before local optimization.

### Required edge behavior

- A negative log gap produces zero controller boost.
- A missing reference produces zero controller boost.
- Warmup overrides the computed coefficient but does not alter the logged
  target, smoothed, or bounded intermediate values.
- Bounds are applied after smoothing.
- The coefficient is updated only for selected clients.

## 6. Server reference transition

After all selected clients finish their local work in round \(t\), define:

\[
R^{(t)}=\operatorname{median}
\{\hat L_i^{(t)}:i\in\mathcal S_t\}.
\]

The reference for the next round is:

\[
L_g^{(t)} =
\begin{cases}
R^{(t)}, & L_g\text{ was absent},\\
\beta L_g^{(t-1)}+(1-\beta)R^{(t)}, & \text{otherwise}.
\end{cases}
\]

Clients in round \(t\) use \(L_g^{(t-1)}\); the newly computed
\(L_g^{(t)}\) is not used until the next round.

## 7. Round protocol

For each global round:

1. The server selects clients.
2. The server makes the current global model available to all clients so global
   evaluation cannot use stale local copies.
3. The server sends selected clients the previous scalar reference \(L_g\) and
   the global round index.
4. Evaluation runs when scheduled.
5. Each selected client probes the received global model and updates its
   coefficient.
6. AdaDitto clients train the persistent personalized model using the adaptive
   coefficient, then perform ordinary local global-model training. AdaFedProx
   clients perform ordinary FedProx local training using the adaptive
   coefficient.
7. The server receives the ordinary global-model updates.
8. The server computes the median probe loss and advances the EMA reference.
9. The server aggregates global-model updates exactly as in the baseline.

The controller adds one scalar probe loss per selected client and one scalar
reference per selected client broadcast. It does not communicate personalized
models.

## 8. Diagnostic ablation modes

These modes are verification probes, not the default algorithm:

- `fixed_base`: use \(c_{\mathrm{base}}\) directly, without warmup.
- `no_log_ratio`: replace the log gap with
  \(\max((\hat L_i-L_g)/(|L_g|+\epsilon),0)\).
- `no_clipping`: do not clip the gap and do not bound the final coefficient.
- `no_ema`: replace the server reference with the current round statistic.
- `mean_global_loss`: use the selected-client mean instead of median.
- `global_shared`: compute one server-side coefficient with the same
  gap/target/smoothing/bounds equations and send it to every selected client.
- `random_adaptive`: sample the client coefficient uniformly from
  \([c_{\min},c_{\max})\) using the framework RNG.

## 9. Baseline objectives

### AdaDitto

AdaDitto replaces fixed \(\lambda\) in the ordinary Ditto personalized
objective:

\[
\min_{v_i}\;F_i(v_i)+\frac{\lambda_i^{(t)}}{2}
\lVert v_i-w^{(t)}\rVert_2^2.
\]

The personalized model remains local. The client's ordinary global-model update
and server aggregation remain those of Ditto.

### AdaFedProx

AdaFedProx replaces fixed \(\mu\) in the ordinary FedProx client objective:

\[
\min_{u_i}\;F_i(u_i)+\frac{\mu_i^{(t)}}{2}
\lVert u_i-w^{(t)}\rVert_2^2.
\]

The resulting client model update is aggregated by the ordinary server
aggregation rule.

## 10. Observable conformance contract

For every selected client and round, an auditable implementation should emit:

- round and client identifier;
- probe loss and received reference;
- raw and clipped gap;
- previous, target, smoothed, bounded, and final coefficient;
- gap-clipped, bound-clipped, and warmup flags; and
- absolute coefficient change.

The production schema is `client_controller.csv`. Server logs should include the
round reference statistic, updated EMA, aggregate coefficient summaries, model
metrics, and round timing.

## 11. Production code map

| Responsibility | AdaDitto | AdaFedProx |
|---|---|---|
| Client probe and controller | `system/flcore/clients/client_adaprox_ditto.py` | `system/flcore/clients/clientadaprox.py` |
| Server reference and round protocol | `system/flcore/servers/server_adaprox_ditto.py` | `system/flcore/servers/serveradaprox.py` |
| Baseline client objective | `system/flcore/clients/clientditto.py` | `system/flcore/clients/clientprox.py` |
| Algorithm selection and CLI | `system/main.py` | `system/main.py` |

## 12. Non-goals and claims not implied

Conformance does not establish that the loss gap is universally optimal, that
the controller matches a per-condition oracle, or that AdaFedProx improves on
FedProx in every regime. Those are empirical questions. The implementation
claim is narrower: the controller changes the existing proximal coefficient
online while preserving the baseline objective form and aggregation pattern.
