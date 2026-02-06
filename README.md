# PFLlib: Personalized Federated Learning Library and Benchmark

This repository is a fork of [PFLlib](https://github.com/TsingZ0/PFLlib), a comprehensive personalized federated learning library and benchmark. It extends PFLlib with two novel adaptive algorithms: **AdaProxFedProx** and **AdaProxDitto**.

- **AdaProxFedProx** adapts the proximal regularization strength (mu) per round using an EMA-based loss gap signal, replacing FedProx's static mu with one that responds to training dynamics.
- **AdaProxDitto** applies the same adaptive proximal mechanism to the Ditto personalized FL framework, combining per-client personalization with server-governed adaptive regularization.

Both algorithms share a minimal configuration surface and require no manual mu tuning.

## Repository Structure

```
system/
├── main.py                                  # Entry point for all algorithms
├── flcore/
│   ├── servers/
│   │   ├── serveradaprox.py                 # AdaProxFedProx server
│   │   └── server_adaprox_ditto.py          # AdaProxDitto server
│   └── clients/
│       ├── clientadaprox.py                 # AdaProxFedProx client
│       └── client_adaprox_ditto.py          # AdaProxDitto client
├── profile_runtime.py                       # Runtime profiling script
dataset/
├── generate_Cifar100.py                     # Generate CIFAR-100 scenarios
├── generate_MNIST.py                        # Generate MNIST scenarios
├── generate_FashionMNIST.py                 # Generate Fashion-MNIST scenarios
tools/
└── analyze_results.py                       # Results analysis and visualization
results_cifar100/                            # CIFAR-100 experiment results
results_mnist/                               # MNIST experiment results
results_fmnist/                              # Fashion-MNIST experiment results
```

## Results

Experiment results are stored in `results_cifar100/`, `results_mnist/`, and `results_fmnist/`. Each contains:

- `results/` — Per-algorithm CSV logs (`adaprox_minimal.csv`, `adaprox_server_metrics.csv`, etc.)
- `analysis/` — Summary tables
- `hyperparameters.txt` — Exact commands and settings used
