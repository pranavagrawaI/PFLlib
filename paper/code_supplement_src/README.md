# Adaptive Proximal Control for Federated Personalization

## Code and Data Supplement

This archive contains the implementation, experiment drivers, compact result
tables, and figure-generation code supporting the paper. It is a curated
snapshot: model checkpoints, generated CIFAR-100 arrays, raw upstream datasets,
terminal logs, and caches are excluded to remain within the submission limit and
to respect upstream dataset terms.

Start with the fast verification:

```bash
python3 analysis/verify_controller_spec.py
python3 analysis/verify_reported_results.py
```

The first command checks a dependency-free reference implementation against
known vectors and every archived controller trace. The second recomputes the
paper's principal aggregates from archived per-run summary rows. Neither trains
models.

## Archive layout

- `system/`: training entry point and federated-learning implementation.
- `dataset/`: CIFAR-100 and Shakespeare preprocessing code plus the exact
  generated partition metadata used in the paper.
- `experiments/`: campaign drivers, frozen settings, compact per-run result
  tables, six AdaDitto controller traces needed by the mechanism figures, and
  one AdaFedProx trace used for independent integration verification.
  Machine-local `run_dir` values are removed or replaced by archive-relative
  trace paths.
- `analysis/verify_reported_results.py`: recomputes key reported aggregates.
- `analysis/reference_controller.py`: dependency-free executable controller
  specification.
- `analysis/verify_controller_spec.py`: checks the mathematical transition,
  edge cases, client state, server EMA, and archived production traces.
- `paper/scripts/make_current_figures.py`: regenerates all current paper figures
  from the archived summary tables and selected controller traces.
- `IMPLEMENTATION_SPEC.md`: normative algorithm, state, timing, interface, and
  logging specification for independent reimplementation.
- `INDEPENDENT_VERIFICATION.md`: staged conformance and empirical verification
  protocol with explicit falsification conditions.
- `CLAIMS_TO_ARTIFACTS.md`: maps each principal empirical claim to its source
  table and reproduction driver.
- `environment.yml`: a reconstruction of the recorded software environment.
- `SHA256SUMS`: integrity hashes for every other file in the archive.

## Environment

The reported runs used Python 3.10.20, PyTorch 2.8.0 with CUDA 12.8,
torchvision 0.23.0, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, h5py 3.16.0,
and matplotlib 3.10.8. Create the environment with:

```bash
conda env create -f environment.yml
conda activate adaditto
```

A CPU-only installation can run the aggregate verifier and figure generator.
Full training is intended for a CUDA-capable machine and is computationally
expensive.

## Data preparation

Raw datasets are not redistributed. The CIFAR-100 generator obtains the
upstream dataset through torchvision and creates both paper partitions:

```bash
cd dataset
python generate_Cifar100.py --out_dir Cifar100_practical/ \
  --num_clients 20 --partition dir --balance false --class_per_client 10
python generate_Cifar100.py --out_dir Cifar100_pathological/ \
  --num_clients 20 --partition pat --balance false --class_per_client 10
cd ..
```

The resulting `config.json` files should match the archived metadata in
`dataset/partition_metadata/`.

Shakespeare preprocessing expects the standard LEAF train and test JSON files
at the paths named at the top of `dataset/generate_Shakespeare.py`. After those
files are placed there:

```bash
cd dataset
python generate_Shakespeare.py
cd ..
```

## Re-running the reported experiments

The drivers are serial and resumable. Set `SKIP_DATA_PREP=1` after preparing
only the datasets needed for the intended run. The following commands reproduce
the principal CIFAR evidence:

```bash
export PYTHON_BIN="$(command -v python3)"
export SKIP_DATA_PREP=1
export CAMPAIGN_DATASETS="Cifar100_practical Cifar100_pathological"

bash experiments/phase1_static_sweeps.sh
bash experiments/core_partic_sweep.sh
bash experiments/phase_dropout.sh
bash experiments/core_participation.sh experiments/frozen/freeze_manifest.json

SCAN_DATASET=Cifar100_pathological \
SCAN_JR=0.2 \
SCAN_SEEDS="0 1 2" \
SCAN_PHASE=cscan_cifar_path_jr0.2 \
bash experiments/controller_scan.sh
```

The controller ablation driver uses both CIFAR splits when invoked as follows:

```bash
CAMPAIGN_DATASETS="Cifar100_practical Cifar100_pathological" \
bash experiments/phase3_controller_ablations.sh
```

After preparing Shakespeare, reproduce the text-modality evidence with:

```bash
bash experiments/shakespeare_sweep.sh
bash experiments/shakespeare_shift.sh
bash experiments/shakespeare_core.sh

SCAN_DATASET=Shakespeare \
SCAN_JR=0.1 \
SCAN_SEEDS="0 1 2" \
SCAN_PHASE=cscan_shakespeare_jr0.1 \
bash experiments/controller_scan.sh
```

Each run writes its command, configuration, status, metrics, and diagnostics
under `experiments/runs/`. The summarizers then recreate the CSV files archived
in `experiments/`.

## Regenerating figures

The figure script reads the archived result tables directly:

```bash
python3 paper/scripts/make_current_figures.py
```

It regenerates the coefficient, participation, controller-sensitivity,
controller-trajectory, and shift-response figures.

## Scope

The archive supports the paper's empirical claims; it is not a frozen container
image and does not include third-party datasets or dependency packages. Small
floating-point differences across GPU and library builds are possible. The
multi-seed comparisons, fixed sweep grids, and controller settings are preserved
explicitly so such differences can be judged against the reported variation.
