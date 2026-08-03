#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ensure_splits
PHASE="phase0_smoke"
DATASETS=(Cifar100_practical Cifar100_pathological AGNews_practical AGNews_pathological)
ALGORITHMS=(FedProx AdaProxFedProx Ditto AdaProxDitto)
for dataset in "${DATASETS[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    extra=()
    [[ "$algorithm" == "Ditto" ]] && extra=(-lam 0.08)
    run_experiment "$PHASE" "$dataset" "$algorithm" 0 "seed0" "${extra[@]}"
  done
done
NUM_CLIENTS=100 JOIN_RATIO=0.1 run_experiment "$PHASE" Cifar100_practical_100 AdaProxDitto 0 "partial_100_10"
