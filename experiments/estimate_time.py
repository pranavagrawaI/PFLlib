#!/usr/bin/env python3
"""Project remaining-campaign wall-clock from measured per-run cost.

Scans experiments/runs for run_manifest.json (wall_clock_seconds) + the sibling
metadata.json (dataset, algorithm), computes mean cost per (dataset-family,
algorithm), and projects each phase's total from its known run composition.
Run after the timing probe (or any phase) to gate the budget decision.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "runs"


def family(dataset):
    d = dataset or ""
    if d.startswith("AGNews"):
        return "AGNews"
    if d.startswith("Cifar100"):
        return "Cifar100"
    return d or "other"


# phase -> list of (family, algorithm, count)
PHASES = {
    "phase1_static (104)": [
        ("Cifar100", "FedProx", 26), ("Cifar100", "Ditto", 26),
        ("AGNews", "FedProx", 26), ("AGNews", "Ditto", 26),
    ],
    "phase2_scan (56)": [
        ("Cifar100", "AdaProxDitto", 28), ("AGNews", "AdaProxDitto", 28),
    ],
    "phase2_confirm (~18)": [
        ("Cifar100", "AdaProxDitto", 9), ("AGNews", "AdaProxDitto", 9),
    ],
    "phase3_ablation (48)": [
        ("Cifar100", "AdaProxDitto", 24), ("AGNews", "AdaProxDitto", 24),
    ],
    "core_5seed (80)": [
        ("Cifar100", a, 10) for a in ("FedProx", "AdaProxFedProx", "Ditto", "AdaProxDitto")
    ] + [
        ("AGNews", a, 10) for a in ("FedProx", "AdaProxFedProx", "Ditto", "AdaProxDitto")
    ],
}

# cost fallbacks when a (family, algo) pair was never measured
ALGO_ALT = {"Ditto": "AdaProxDitto", "AdaProxDitto": "Ditto",
            "AdaProxFedProx": "FedProx", "FedProx": "AdaProxFedProx"}


def collect():
    samples = {}  # (family, algo) -> [seconds]
    for man in RUNS.rglob("run_manifest.json"):
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            meta = json.loads((man.parent / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        secs = m.get("wall_clock_seconds")
        if not secs or secs <= 0:
            continue
        key = (family(meta.get("dataset")), meta.get("algorithm"))
        samples.setdefault(key, []).append(float(secs))
    return samples


def fmt(seconds):
    if seconds is None:
        return "?"
    h = seconds / 3600.0
    return f"{seconds:.0f}s" if seconds < 90 else (f"{seconds/60:.1f}m" if h < 1 else f"{h:.1f}h")


def main():
    samples = collect()
    if not samples:
        print("No measured runs with wall_clock_seconds found. Run the timing probe first:")
        print("  bash experiments/run_campaign.sh timing")
        sys.exit(1)

    means = {k: sum(v) / len(v) for k, v in samples.items()}
    allv = [x for v in samples.values() for x in v]
    global_mean = sum(allv) / len(allv)

    def cost(fam, algo):
        if (fam, algo) in means:
            return means[(fam, algo)]
        alt = ALGO_ALT.get(algo)
        if alt and (fam, alt) in means:
            return means[(fam, alt)]
        fam_vals = [v for (f, _), v in means.items() if f == fam]
        if fam_vals:
            return sum(fam_vals) / len(fam_vals)
        return global_mean

    print("=== measured per-run cost (mean wall-clock) ===")
    for (fam, algo), v in sorted(means.items()):
        print(f"  {fam:10s} {algo:16s} {fmt(v):>8s}  (n={len(samples[(fam, algo)])})")

    print("\n=== projected wall-clock per phase (single-GPU serial) ===")
    grand = 0.0
    for phase, comp in PHASES.items():
        total = sum(count * cost(fam, algo) for (fam, algo, count) in comp)
        grand += total
        print(f"  {phase:24s} {fmt(total):>8s}")
    print(f"  {'-'*24} {'-'*8}")
    print(f"  {'TOTAL (phases 1-5)':24s} {fmt(grand):>8s}")
    print("\nNote: confirm counts assume ~3 controller configs x 3 seeds; "
          "Ditto~AdaProxDitto and AdaProxFedProx~FedProx where unmeasured.")


if __name__ == "__main__":
    main()
