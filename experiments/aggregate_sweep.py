#!/usr/bin/env python3
"""Aggregate a directory of finalized runs into sweep-level artifacts.

Reads every run under RUNS_DIR that has a final_metrics.json (produced by
finalize_run.py), joins it with that run's config.yaml and status.json, and emits:

    {OUT_PREFIX}_summary.csv   one row per run (config + final metrics + status)
    {OUT_PREFIX}_table.tex     mean +/- std accuracy, methods x datasets
    {OUT_PREFIX}_brittleness.png   final accuracy vs each varying config field
    {OUT_PREFIX}_stability.png     rho delta + clipping vs varying field (adaptive)

Stdlib-only except plots (matplotlib, best-effort). Designed to be called at the
end of a phase, e.g.:

    python aggregate_sweep.py experiments/runs/phase1_static experiments/phase1_static
"""
import csv
import json
import math
import sys
from pathlib import Path

ADAPTIVE = {"AdaProxFedProx", "AdaProxDitto"}
# config fields that a sweep might vary (numeric brittleness axes)
SWEEP_FIELDS = [
    "static_mu", "static_lambda", "mu_base", "alpha_gain", "gap_tau",
    "mu_smooth_gamma", "ema_beta", "warmup_rounds", "loss_eval_sample_ratio",
]


def fnum(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def parse_scalar(s):
    s = s.strip()
    if s in ("null", "", "~"):
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if (s[:1] == '"' and s[-1:] == '"') or (s[:1] == "'" and s[-1:] == "'"):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def read_yaml_flat(path):
    """Minimal reader for finalize_run.py's config.yaml. Nesting (the
    `controller:` block) is flattened; controller keys are already unique."""
    out = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, val = line.strip().partition(":")
        if not sep:
            continue
        val = val.strip()
        if val == "":  # section header (e.g. "controller:")
            continue
        out[key.strip()] = parse_scalar(val)
    return out


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def collect(runs_dir):
    rows = []
    for fm_path in sorted(Path(runs_dir).rglob("final_metrics.json")):
        run_dir = fm_path.parent
        fm = load_json(fm_path)
        cfg = read_yaml_flat(run_dir / "config.yaml")
        status = load_json(run_dir / "status.json")
        method = fm.get("method") or cfg.get("method")
        is_ditto = method in {"Ditto", "AdaProxDitto"}
        score = fm.get("personalized_accuracy_final") if is_ditto else fm.get("global_accuracy_final")
        rows.append({
            "run_dir": str(run_dir),
            "run_id": run_dir.name,
            "dataset": fm.get("dataset") or cfg.get("dataset"),
            "split": fm.get("split") or cfg.get("split"),
            "method": method,
            "seed": fm.get("seed", cfg.get("seed")),
            "controller_mode": cfg.get("controller_mode"),
            "static_mu": cfg.get("static_mu"),
            "static_lambda": cfg.get("static_lambda"),
            "mu_base": cfg.get("mu_base"),
            "alpha_gain": cfg.get("alpha_gain"),
            "gap_tau": cfg.get("gap_tau"),
            "mu_smooth_gamma": cfg.get("mu_smooth_gamma"),
            "ema_beta": cfg.get("ema_beta"),
            "warmup_rounds": cfg.get("warmup_rounds"),
            "loss_eval_sample_ratio": cfg.get("loss_eval_sample_ratio"),
            "score": fnum(score),
            "global_accuracy": fnum(fm.get("global_accuracy_final")),
            "personalized_accuracy": fnum(fm.get("personalized_accuracy_final")),
            "mean_client_accuracy": fnum(fm.get("mean_client_accuracy")),
            "p10_client_accuracy": fnum(fm.get("p10_client_accuracy")),
            "worst_client_accuracy": fnum(fm.get("worst_client_accuracy")),
            "std_client_accuracy": fnum(fm.get("std_client_accuracy")),
            "mean_final_rho": fnum(fm.get("mean_final_rho")),
            "std_final_rho": fnum(fm.get("std_final_rho")),
            "clipping_frequency": fnum(fm.get("clipping_frequency")),
            "mean_abs_rho_delta": fnum(fm.get("mean_abs_rho_delta")),
            "total_wall_clock_seconds": fnum(fm.get("total_wall_clock_seconds")),
            "status": status.get("status") or fm.get("status"),
        })
    return rows


def write_summary(rows, out_prefix):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(f"{out_prefix}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_tex(rows, out_prefix):
    """mean +/- std of score, rows = method, columns = dataset."""
    datasets = sorted({r["dataset"] for r in rows if r["dataset"]})
    methods = sorted({r["method"] for r in rows if r["method"]})
    if not datasets or not methods:
        return
    cell = {}
    for m in methods:
        for d in datasets:
            scores = [r["score"] for r in rows
                      if r["method"] == m and r["dataset"] == d and r["score"] is not None]
            cell[(m, d)] = (mean(scores), std(scores), len(scores))

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Final accuracy (mean $\\pm$ std over seeds).}",
        "\\begin{tabular}{l" + "c" * len(datasets) + "}",
        "\\toprule",
        "Method & " + " & ".join(d.replace("_", "\\_") for d in datasets) + " \\\\",
        "\\midrule",
    ]
    for m in methods:
        cells = []
        for d in datasets:
            mu, sd, n = cell[(m, d)]
            cells.append(f"{mu*100:.2f} $\\pm$ {(sd or 0)*100:.2f}" if mu is not None else "--")
        lines.append(m.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(f"{out_prefix}_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def varying_fields(rows):
    out = []
    for fld in SWEEP_FIELDS:
        vals = {r[fld] for r in rows if r.get(fld) is not None}
        if len({fnum(v) for v in vals if fnum(v) is not None}) >= 2:
            out.append(fld)
    return out


def make_plots(rows, out_prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[aggregate_sweep] plots skipped (matplotlib unavailable): {exc}", file=sys.stderr)
        return

    fields = varying_fields(rows)
    datasets = sorted({r["dataset"] for r in rows if r["dataset"]})
    adaptive_rows = [r for r in rows if r["method"] in ADAPTIVE and r["score"] is not None]

    # brittleness: score vs each varying field, per dataset; adaptive as a band
    if fields:
        fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4), squeeze=False)
        for ax, fld in zip(axes[0], fields):
            for d in datasets:
                pts = sorted(
                    (fnum(r[fld]), r["score"]) for r in rows
                    if r["dataset"] == d and r.get(fld) is not None and r["score"] is not None
                    and r["method"] not in ADAPTIVE
                )
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{d} (static)")
                ad = [r["score"] for r in adaptive_rows if r["dataset"] == d]
                if ad:
                    m, s = mean(ad), std(ad) or 0.0
                    ax.axhspan(m - s, m + s, alpha=0.15)
                    ax.axhline(m, ls="--", lw=1, label=f"{d} (adaptive)")
            ax.set_xlabel(fld)
            ax.set_ylabel("final accuracy")
            ax.set_title(f"Brittleness vs {fld}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_brittleness.png", dpi=120)
        plt.close(fig)

    # stability: rho delta + clipping vs varying field (adaptive runs)
    ad_fields = [f for f in fields if any(r.get(f) is not None for r in adaptive_rows)]
    if ad_fields:
        fld = ad_fields[0]
        pts = sorted(
            (fnum(r[fld]), r["mean_abs_rho_delta"], r["clipping_frequency"])
            for r in adaptive_rows if r.get(fld) is not None
        )
        if pts:
            fig, ax1 = plt.subplots(figsize=(6, 4))
            xs = [p[0] for p in pts]
            ax1.plot(xs, [p[1] for p in pts], "b-o", label="mean |delta rho|")
            ax1.set_xlabel(fld); ax1.set_ylabel("mean |delta rho|", color="b")
            ax2 = ax1.twinx()
            ax2.plot(xs, [p[2] for p in pts], "r-s", label="clipping frequency")
            ax2.set_ylabel("clipping frequency", color="r")
            ax1.set_title(f"Controller stability vs {fld}")
            fig.tight_layout()
            fig.savefig(f"{out_prefix}_stability.png", dpi=120)
            plt.close(fig)


def main():
    if len(sys.argv) < 3:
        print("usage: aggregate_sweep.py RUNS_DIR OUT_PREFIX", file=sys.stderr)
        sys.exit(2)
    runs_dir, out_prefix = sys.argv[1], sys.argv[2]
    rows = collect(runs_dir)
    if not rows:
        print(f"[aggregate_sweep] no finalized runs under {runs_dir}", file=sys.stderr)
        sys.exit(1)
    write_summary(rows, out_prefix)
    write_tex(rows, out_prefix)
    make_plots(rows, out_prefix)
    print(f"[aggregate_sweep] {len(rows)} runs -> {out_prefix}_summary.csv / _table.tex / plots")


if __name__ == "__main__":
    main()
