# Claims to Artifacts

| Paper claim or check | Archived evidence | Full-run driver |
|---|---|---|
| The best fixed Ditto coefficient differs between the two full-participation CIFAR splits. | `experiments/phase1_selection.csv` | `experiments/phase1_static_sweeps.sh` |
| On CIFAR-pathological at 20% participation, AdaDitto improves over Ditto frozen at the full-participation coefficient by about 1.8 points. | `experiments/phase_dropout_summary.csv` | `experiments/phase_dropout.sh` |
| AdaDitto remains below Ditto re-tuned separately for each CIFAR condition. | `experiments/core_participation_summary.csv`, `experiments/core_partic_sweep_selection.csv` | `experiments/core_partic_sweep.sh`, `experiments/core_participation.sh` |
| The controller result is not tied to one lucky setting at the two headline conditions. | `experiments/cscan_cifar_path_jr0.2_summary.csv`, `experiments/cscan_shakespeare_jr0.1_summary.csv` | `experiments/controller_scan.sh` |
| Random adaptive coefficients hurt on the pathological split while principled modes cluster. | `experiments/phase3_ablation_summary.csv` | `experiments/phase3_controller_ablations.sh` |
| On Shakespeare, nonzero coupling is essential and AdaDitto matches tuned Ditto rather than producing a shift win. | `experiments/shakespeare_sweep_selection.csv`, `experiments/shakespeare_shift_selection.csv`, `experiments/shakespeare_core_summary.csv` | `experiments/shakespeare_sweep.sh`, `experiments/shakespeare_shift.sh`, `experiments/shakespeare_core.sh` |
| All current paper figures can be reconstructed. | Aggregate CSV files plus `experiments/traces/` | `paper/scripts/make_current_figures.py` |

Run `python3 analysis/verify_reported_results.py` to recompute the principal
numbers and check them against narrow tolerances.
