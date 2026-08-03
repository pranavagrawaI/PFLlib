#!/usr/bin/env python3
"""Derive the per-run artifact set from the CSVs already logged during training.

Everything here is reconstructed offline from files written by the training loop
(server_metrics.csv, client_eval.csv, client_controller.csv, events.jsonl,
terminal.log) plus run identity files (metadata.json, command.sh,
dataset_config.json). This keeps plotting/aggregation decoupled from training and
robust to crashes: the finalizer runs even when the run failed.

Core artifacts use only the standard library so the script works under the bare
`python3` used by validate_run.py. Plots and the torch/GPU manifest fields are
best-effort: they populate when run under an interpreter that has matplotlib /
torch (the pfllib env), and degrade gracefully otherwise.

Usage:
    python finalize_run.py RUN_DIR ALGORITHM [EXIT_STATUS]
"""
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- argparse flag -> canonical config key (matches system/main.py) ---------
ALIAS = {
    "-mu": "mu", "--mu": "mu",
    "-lam": "lamda", "--lamda": "lamda",
    "-pls": "plocal_epochs", "--plocal_epochs": "plocal_epochs",
    "-mubase": "mu_base", "--mu_base": "mu_base",
    "-mmin": "mu_min", "--mu_min": "mu_min",
    "-mmax": "mu_max", "--mu_max": "mu_max",
    "-ag": "alpha_gain", "--alpha_gain": "alpha_gain",
    "-gt": "gap_tau", "--gap_tau": "gap_tau",
    "-musmooth": "mu_smooth_gamma", "--mu_smooth_gamma": "mu_smooth_gamma",
    "-eb": "ema_beta", "--ema_beta": "ema_beta",
    "-lsr": "loss_eval_sample_ratio", "--loss_eval_sample_ratio": "loss_eval_sample_ratio",
    "-wr": "warmup_rounds", "--warmup_rounds": "warmup_rounds",
    "--controller_mode": "controller_mode",
    "--track_probe_accuracy": "track_probe_accuracy",
    "--track_overhead": "track_overhead",
    "-vs": "vocab_size", "--vocab_size": "vocab_size",
    "-ml": "max_len", "--max_len": "max_len",
    "-ncl": "num_classes", "--num_classes": "num_classes",
}
CONTROLLER_DEFAULTS = {
    "mu_base": 0.08, "mu_min": 0.05, "mu_max": 3.0, "alpha_gain": 0.8,
    "gap_tau": 0.7, "mu_smooth_gamma": 0.3, "ema_beta": 0.9,
    "warmup_rounds": 2, "loss_eval_sample_ratio": 0.1, "controller_mode": "full",
}
ADAPTIVE = {"AdaProxFedProx", "AdaProxDitto"}
DITTO_LIKE = {"Ditto", "AdaProxDitto"}


# --- small helpers ----------------------------------------------------------
def fnum(value, default=None):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def percentile(values, pct):
    values = [v for v in values if v is not None]
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (1.0 - (rank - lo)) + values[hi] * (rank - lo)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if values else None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    e = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            e -= p * math.log2(p)
    return e


def parse_arg_list(tokens):
    out = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if isinstance(t, str) and t.startswith("-") and not _looks_numeric(t):
            key = ALIAS.get(t, t.lstrip("-"))
            if i + 1 < len(tokens) and not (
                isinstance(tokens[i + 1], str)
                and tokens[i + 1].startswith("-")
                and not _looks_numeric(tokens[i + 1])
            ):
                out[key] = tokens[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            i += 1
    return out


def _looks_numeric(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# --- minimal YAML emitter (avoids a PyYAML dependency) ----------------------
def yaml_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if s == "" or s.strip() != s or any(c in s for c in ":#{}[],&*!|>'\"%@`"):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def to_yaml(obj, indent=0):
    pad = "  " * indent
    lines = []
    for k, v in obj.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(to_yaml(v, indent + 1))
        elif isinstance(v, (list, tuple)):
            lines.append(f"{pad}{k}: [{', '.join(yaml_scalar(x) for x in v)}]")
        else:
            lines.append(f"{pad}{k}: {yaml_scalar(v)}")
    return "\n".join(lines)


# --- run context ------------------------------------------------------------
class RunContext:
    def __init__(self, run_dir, algorithm):
        self.dir = Path(run_dir)
        self.results = self.dir / "results"
        self.algorithm = algorithm
        self.is_adaptive = algorithm in ADAPTIVE
        self.is_ditto = algorithm in DITTO_LIKE

        self.metadata = self._load_json(self.dir / "metadata.json")
        self.dataset_config = self._load_json(self.dir / "dataset_config.json")
        self.command = self._read_text(self.dir / "command.sh").strip()

        self.server_rows = read_csv(self.results / "server_metrics.csv")
        self.client_eval = read_csv(self.results / "client_eval.csv")
        self.controller = read_csv(self.results / "client_controller.csv")
        self.events = self._load_jsonl(self.results / "events.jsonl")

        self.args = self._build_args()
        # per-client label statistics keyed by client id (int)
        self.partition = self._build_partition_stats()

    @staticmethod
    def _load_json(path):
        path = Path(path)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}

    @staticmethod
    def _read_text(path):
        path = Path(path)
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _load_jsonl(path):
        path = Path(path)
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
        return out

    def _build_args(self):
        args = dict(CONTROLLER_DEFAULTS)
        extra = self.metadata.get("extra_args") or []
        args.update(parse_arg_list(extra))
        return args

    def _build_partition_stats(self):
        stat = self.dataset_config.get("Size of samples for labels in clients")
        out = {}
        if not isinstance(stat, list):
            return out
        for cid, pairs in enumerate(stat):
            counts = [int(c) for (_, c) in pairs]
            labels = sorted(int(l) for (l, _) in pairs)
            total = sum(counts)
            out[cid] = {
                "labels": labels,
                "counts": counts,
                "total": total,
                "num_unique_labels": len(labels),
                "entropy": entropy(counts),
                "majority_fraction": (max(counts) / total) if total else 0.0,
                "histogram": ";".join(f"{l}:{c}" for (l, c) in zip(labels, counts)),
            }
        return out

    # convenience accessors -------------------------------------------------
    def meta(self, key, default=None):
        return self.metadata.get(key, default)

    def final_eval_round(self, scope="global"):
        rounds = [int(fnum(r.get("round"), -1)) for r in self.client_eval
                  if r.get("scope") == scope]
        rounds = [r for r in rounds if r >= 0]
        return max(rounds) if rounds else None


# --- artifact writers -------------------------------------------------------
def write_config(ctx):
    a = ctx.args
    num_clients = int(fnum(ctx.meta("num_clients"), 0) or 0)
    join_ratio = fnum(ctx.meta("join_ratio"), 1.0)
    partition = ctx.dataset_config.get("partition")
    split = {"dir": "practical", "pat": "pathological"}.get(partition, "unknown")
    config = {
        "dataset": ctx.meta("dataset"),
        "split": split,
        "partition": partition,
        "partition_alpha": ctx.dataset_config.get("alpha"),
        "class_per_client": ctx.dataset_config.get("class_per_client"),
        "num_clients": num_clients,
        "clients_per_round": int(round(join_ratio * num_clients)) if num_clients else None,
        "participation_rate": join_ratio,
        "method": ctx.algorithm,
        "seed": int(fnum(ctx.meta("seed"), 0) or 0),
        "model": ctx.meta("model"),
        "optimizer": "SGD",
        "learning_rate": fnum(ctx.meta("lr")),
        "local_epochs": int(fnum(ctx.meta("local_epochs"), 0) or 0),
        "global_rounds": int(fnum(ctx.meta("rounds"), 0) or 0),
        "batch_size": int(fnum(ctx.meta("batch_size"), 0) or 0),
        "eval_gap": int(fnum(ctx.meta("eval_gap"), 1) or 1),
        "probe_cap": 256,
        "loss_eval_sample_ratio": fnum(a.get("loss_eval_sample_ratio")),
    }
    if "plocal_epochs" in a:
        config["plocal_epochs"] = int(fnum(a.get("plocal_epochs"), 0) or 0)
    if ctx.algorithm == "FedProx" and "mu" in a:
        config["static_mu"] = fnum(a.get("mu"))
    if ctx.algorithm == "Ditto" and "lamda" in a:
        config["static_lambda"] = fnum(a.get("lamda"))
    if ctx.is_adaptive:
        config["controller"] = {
            "mu_base": fnum(a.get("mu_base")),
            "mu_min": fnum(a.get("mu_min")),
            "mu_max": fnum(a.get("mu_max")),
            "alpha_gain": fnum(a.get("alpha_gain")),
            "gap_tau": fnum(a.get("gap_tau")),
            "mu_smooth_gamma": fnum(a.get("mu_smooth_gamma")),
            "ema_beta": fnum(a.get("ema_beta")),
            "warmup_rounds": int(fnum(a.get("warmup_rounds"), 0) or 0),
            "controller_mode": a.get("controller_mode", "full"),
        }
    (ctx.dir / "config.yaml").write_text(to_yaml(config) + "\n", encoding="utf-8")


def write_manifest(ctx, exit_status):
    def git(*cmd):
        try:
            return subprocess.check_output(
                ["git", "-C", str(ROOT), *cmd], stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return None

    torch_version = cuda_version = gpu_name = None
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    start = fnum(ctx.meta("created_at_unix"))
    log_path = ctx.dir / "terminal.log"
    end = log_path.stat().st_mtime if log_path.exists() else time.time()
    manifest = {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "start_time_unix": start,
        "end_time_unix": end,
        "wall_clock_seconds": (end - start) if start else None,
        "command": ctx.command,
        "exit_status": exit_status,
        "deterministic_flags": {
            "seed_set": True,
            "torch_deterministic": False,
            "cudnn_deterministic": False,
        },
    }
    (ctx.dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_partition_summary(ctx):
    # per-client train/test sample counts come from the final client_eval round
    final = ctx.final_eval_round("global")
    counts = {}
    for r in ctx.client_eval:
        if r.get("scope") != "global" or int(fnum(r.get("round"), -1)) != final:
            continue
        cid = int(fnum(r.get("client_id"), -1))
        counts[cid] = (int(fnum(r.get("train_samples"), 0) or 0),
                       int(fnum(r.get("test_samples"), 0) or 0))

    cids = sorted(set(ctx.partition) | set(counts))
    if not cids:
        return
    is_pat = ctx.dataset_config.get("partition") == "pat"
    rows = []
    for cid in cids:
        p = ctx.partition.get(cid, {})
        tr, te = counts.get(cid, (None, None))
        row = {
            "client_id": cid,
            "num_train_samples": tr,
            "num_test_samples": te,
            "num_unique_labels": p.get("num_unique_labels"),
            "label_entropy": _round(p.get("entropy")),
            "majority_label_fraction": _round(p.get("majority_fraction")),
            "label_histogram": p.get("histogram", ""),
            "classes_assigned": " ".join(map(str, p.get("labels", []))) if is_pat else "",
        }
        rows.append(row)
    write_csv(ctx.dir / "partition_summary.csv", rows, list(rows[0].keys()))


def write_round_metrics(ctx):
    if not ctx.server_rows:
        return
    # per-round per-client accuracy (global scope) for distribution stats
    by_round = {}
    for r in ctx.client_eval:
        if r.get("scope") != "global":
            continue
        rd = int(fnum(r.get("round"), -1))
        by_round.setdefault(rd, []).append(fnum(r.get("test_accuracy")))

    rows = []
    elapsed = 0.0
    for s in ctx.server_rows:
        rd = int(fnum(s.get("round"), -1))
        accs = by_round.get(rd, [])
        elapsed += fnum(s.get("time_cost"), 0.0) or 0.0
        sel_ids = (s.get("selected_client_ids") or "").split()
        rows.append({
            "round": rd,
            "global_train_loss": _round(fnum(s.get("global_train_loss"))),
            "global_test_accuracy": _round(fnum(s.get("global_accuracy"))),
            "global_test_loss": None,  # not measured by the framework
            "personalized_test_accuracy": _round(fnum(s.get("personalized_accuracy"))) if ctx.is_ditto else None,
            "personalized_test_loss": None,
            "mean_client_accuracy": _round(mean(accs)),
            "p10_client_accuracy": _round(percentile(accs, 0.10)),
            "worst_client_accuracy": _round(min([a for a in accs if a is not None], default=None)),
            "std_client_accuracy": _round(std(accs)),
            "selected_clients": s.get("selected_client_ids", ""),
            "num_selected_clients": int(fnum(s.get("selected_clients"), len(sel_ids)) or len(sel_ids)),
            "wall_clock_elapsed": _round(elapsed),
        })
    write_csv(ctx.dir / "round_metrics.csv", rows, list(rows[0].keys()))


def _client_rho_stats(ctx):
    """Per-client rho summary from the controller trace (adaptive only)."""
    by_client = {}
    for r in ctx.controller:
        cid = int(fnum(r.get("client_id"), -1))
        rd = int(fnum(r.get("round"), -1))
        rho = fnum(r.get("coefficient_final"))
        by_client.setdefault(cid, []).append((rd, rho))
    out = {}
    for cid, seq in by_client.items():
        seq.sort()
        vals = [v for (_, v) in seq if v is not None]
        out[cid] = {
            "final_rho": seq[-1][1] if seq else None,
            "mean_rho": mean(vals),
            "std_rho": std(vals),
            "min_rho": min(vals) if vals else None,
            "max_rho": max(vals) if vals else None,
        }
    return out


def _client_rows_at(ctx, scope, rnd):
    """{client_id: row} for the given scope at round rnd."""
    return {
        int(fnum(r.get("client_id"), -1)): r
        for r in ctx.client_eval
        if r.get("scope") == scope and int(fnum(r.get("round"), -1)) == rnd
    }


def write_client_final(ctx):
    final = ctx.final_eval_round("global")
    if final is None:
        return
    rho_stats = _client_rho_stats(ctx) if ctx.is_adaptive else {}
    per_final = ctx.final_eval_round("personalized")
    per_rows = _client_rows_at(ctx, "personalized", per_final) if per_final is not None else {}
    rows = []
    for r in ctx.client_eval:
        if r.get("scope") != "global" or int(fnum(r.get("round"), -1)) != final:
            continue
        cid = int(fnum(r.get("client_id"), -1))
        p = ctx.partition.get(cid, {})
        rs = rho_stats.get(cid, {})
        pr = per_rows.get(cid, {})
        rows.append({
            "client_id": cid,
            "num_train_samples": int(fnum(r.get("train_samples"), 0) or 0),
            "num_test_samples": int(fnum(r.get("test_samples"), 0) or 0),
            "label_entropy": _round(p.get("entropy")),
            "num_unique_labels": p.get("num_unique_labels"),
            "global_model_accuracy": _round(fnum(r.get("test_accuracy"))),
            "global_model_loss": _round(fnum(r.get("train_loss"))),
            "personalized_model_accuracy": _round(fnum(pr.get("test_accuracy"))) if pr else None,
            "personalized_model_loss": _round(fnum(pr.get("train_loss"))) if pr else None,
            "final_rho": _round(rs.get("final_rho")),
            "mean_rho": _round(rs.get("mean_rho")),
            "std_rho": _round(rs.get("std_rho")),
            "min_rho": _round(rs.get("min_rho")),
            "max_rho": _round(rs.get("max_rho")),
        })
    if rows:
        rows.sort(key=lambda x: x["client_id"])
        write_csv(ctx.dir / "client_metrics_final.csv", rows, list(rows[0].keys()))


def write_client_roundwise(ctx):
    """Per (round, client) global+personalized metrics with a participation flag.
    Participation = client was selected that round (from server_metrics)."""
    if not ctx.client_eval:
        return
    participated = {}  # round -> set(client_id)
    for s in ctx.server_rows:
        rd = int(fnum(s.get("round"), -1))
        ids = {int(x) for x in (s.get("selected_client_ids") or "").split() if x}
        participated[rd] = ids
    # controller rows are logged only for selected clients -> also mark them
    for r in ctx.controller:
        rd = int(fnum(r.get("round"), -1))
        participated.setdefault(rd, set()).add(int(fnum(r.get("client_id"), -1)))

    glob = {}
    per = {}
    for r in ctx.client_eval:
        key = (int(fnum(r.get("round"), -1)), int(fnum(r.get("client_id"), -1)))
        (glob if r.get("scope") == "global" else per)[key] = r

    rows = []
    for key in sorted(set(glob) | set(per)):
        rd, cid = key
        g = glob.get(key, {})
        p = per.get(key, {})
        rows.append({
            "round": rd,
            "client_id": cid,
            "participated": int(cid in participated.get(rd, set())),
            "global_model_loss_on_client": _round(fnum(g.get("train_loss"))),
            "global_model_accuracy_on_client": _round(fnum(g.get("test_accuracy"))),
            "personalized_model_loss_on_client": _round(fnum(p.get("train_loss"))) if p else None,
            "personalized_model_accuracy_on_client": _round(fnum(p.get("test_accuracy"))) if p else None,
            "num_local_samples": int(fnum(g.get("train_samples") or p.get("train_samples"), 0) or 0),
        })
    if rows:
        write_csv(ctx.dir / "client_metrics_roundwise.csv", rows, list(rows[0].keys()))


def write_traces(ctx):
    """coefficient_trace.csv + controller_trace.csv from client_controller.csv."""
    if not ctx.controller:
        return
    mu_min = fnum(ctx.args.get("mu_min"))
    mu_max = fnum(ctx.args.get("mu_max"))
    eps = 1e-9
    coef_rows, ctrl_rows = [], []
    for r in ctx.controller:
        rd = int(fnum(r.get("round"), -1))
        cid = int(fnum(r.get("client_id"), -1))
        rho = fnum(r.get("coefficient_final"))
        smoothed = fnum(r.get("coefficient_smoothed"))
        bound_clipped = str(r.get("bound_clipped")) in ("1", "True", "true")
        low = bool(bound_clipped and mu_min is not None and rho is not None and rho <= mu_min + eps)
        high = bool(bound_clipped and mu_max is not None and rho is not None and rho >= mu_max - eps)
        coef_rows.append({
            "round": rd,
            "client_id": cid,
            "participated": 1,
            "rho": _round(rho),
            "rho_target": _round(fnum(r.get("coefficient_target"))),
            "rho_before_clip": _round(smoothed),
            "rho_min": mu_min,
            "rho_max": mu_max,
            "was_clipped_low": int(low),
            "was_clipped_high": int(high),
        })
        ctrl_rows.append({
            "round": rd,
            "client_id": cid,
            "participated": 1,
            "probe_loss_L_hat": _round(fnum(r.get("Li"))),
            "reported_loss_L_i": _round(fnum(r.get("Li"))),
            "server_reference_loss_L_g": _round(fnum(r.get("Lg"))),
            "loss_gap_g_i": _round(fnum(r.get("client_loss_gap_clipped"))),
            "log_loss_ratio": _round(fnum(r.get("client_loss_gap_raw"))),
            "clipped_log_loss_ratio": _round(fnum(r.get("client_loss_gap_clipped"))),
            "rho_target": _round(fnum(r.get("coefficient_target"))),
            "rho_final": _round(rho),
        })
    write_csv(ctx.dir / "coefficient_trace.csv", coef_rows, list(coef_rows[0].keys()))
    write_csv(ctx.dir / "controller_trace.csv", ctrl_rows, list(ctrl_rows[0].keys()))


def write_final_metrics(ctx):
    last = ctx.server_rows[-1] if ctx.server_rows else {}
    final = ctx.final_eval_round("global")
    accs = [fnum(r.get("test_accuracy")) for r in ctx.client_eval
            if r.get("scope") == "global" and int(fnum(r.get("round"), -1)) == final]
    accs = [a for a in accs if a is not None]
    rho_stats = _client_rho_stats(ctx) if ctx.is_adaptive else {}
    final_rhos = [v["final_rho"] for v in rho_stats.values() if v["final_rho"] is not None]
    out = {
        "dataset": ctx.meta("dataset"),
        "split": {"dir": "practical", "pat": "pathological"}.get(
            ctx.dataset_config.get("partition"), "unknown"),
        "method": ctx.algorithm,
        "seed": int(fnum(ctx.meta("seed"), 0) or 0),
        "global_accuracy_final": fnum(last.get("global_accuracy")),
        "personalized_accuracy_final": fnum(last.get("personalized_accuracy")) if ctx.is_ditto else None,
        "mean_client_accuracy": mean(accs),
        "p10_client_accuracy": percentile(accs, 0.10),
        "worst_client_accuracy": min(accs) if accs else None,
        "std_client_accuracy": std(accs),
        "final_train_loss": fnum(last.get("global_train_loss")),
        "final_test_loss": None,
        "mean_final_rho": mean(final_rhos) if ctx.is_adaptive else None,
        "std_final_rho": std(final_rhos) if ctx.is_adaptive else None,
        "clipping_frequency": fnum(last.get("clipping_frequency")) if ctx.is_adaptive else None,
        "mean_abs_rho_delta": fnum(last.get("mean_abs_coefficient_delta")) if ctx.is_adaptive else None,
        "total_wall_clock_seconds": sum(fnum(s.get("time_cost"), 0.0) or 0.0 for s in ctx.server_rows),
        "rounds_completed": int(fnum(last.get("round"), 0) or 0),
        "status": "completed" if any(e.get("type") == "run_finished" for e in ctx.events) else "unknown",
    }
    (ctx.dir / "final_metrics.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_status(ctx, exit_status):
    losses = [fnum(s.get("global_train_loss")) for s in ctx.server_rows]
    losses = [l for l in losses if l is not None]
    raw_losses = [s.get("global_train_loss") for s in ctx.server_rows]
    nan_detected = any(
        (str(v).strip().lower() in ("nan", "inf", "-inf")) or (fnum(v) is None and v not in (None, ""))
        for v in raw_losses
    )
    max_loss = max(losses) if losses else None
    explosion = bool(max_loss is not None and max_loss > 1e3)
    last_round = int(fnum(ctx.server_rows[-1].get("round"), -1)) if ctx.server_rows else -1

    log_tail = ctx._read_text(ctx.dir / "terminal.log")
    exception_message = ""
    if "Traceback (most recent call last)" in log_tail:
        block = log_tail.rsplit("Traceback (most recent call last)", 1)[-1]
        exception_message = ("Traceback (most recent call last)" + block).strip()[-2000:]

    finished = any(e.get("type") == "run_finished" for e in ctx.events)
    if nan_detected:
        status = "nan_loss"
    elif explosion:
        status = "diverged"
    elif exception_message or (exit_status not in (None, 0)):
        status = "failed"
    elif finished:
        status = "completed"
    else:
        status = "incomplete"

    out = {
        "status": status,
        "last_completed_round": last_round,
        "exit_status": exit_status,
        "exception_message": exception_message,
        "max_loss_seen": max_loss,
        "nan_detected": nan_detected,
        "gradient_explosion_detected": explosion,
        "rho_clipping_frequency": fnum(ctx.server_rows[-1].get("clipping_frequency")) if (ctx.is_adaptive and ctx.server_rows) else None,
    }
    (ctx.dir / "status.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def _round(v, ndigits=6):
    return round(v, ndigits) if isinstance(v, float) else v


# --- plots (optional: needs matplotlib) -------------------------------------
def make_plots(ctx):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        (ctx.dir / "plots").mkdir(exist_ok=True)
        (ctx.dir / "plots" / "SKIPPED.txt").write_text(
            f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    plots = ctx.dir / "plots"
    plots.mkdir(exist_ok=True)
    rm = read_csv(ctx.dir / "round_metrics.csv")
    rounds = [fnum(r["round"]) for r in rm]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(plots / name, dpi=120)
        plt.close(fig)

    def col(rows, key):
        return [fnum(r.get(key)) for r in rows]

    # accuracy curve
    try:
        fig, ax = plt.subplots()
        ax.plot(rounds, col(rm, "global_test_accuracy"), label="global")
        if ctx.is_ditto and any(v is not None for v in col(rm, "personalized_test_accuracy")):
            ax.plot(rounds, col(rm, "personalized_test_accuracy"), label="personalized")
        ax.set_xlabel("round"); ax.set_ylabel("test accuracy"); ax.legend(); ax.set_title("Accuracy")
        save(fig, "accuracy_curve.png")
    except Exception:
        pass

    # loss curve
    try:
        fig, ax = plt.subplots()
        ax.plot(rounds, col(rm, "global_train_loss"), label="global train loss")
        ax.set_xlabel("round"); ax.set_ylabel("loss"); ax.legend(); ax.set_title("Loss")
        save(fig, "loss_curve.png")
    except Exception:
        pass

    # runtime curve
    try:
        times = [fnum(s.get("time_cost")) for s in ctx.server_rows]
        sr = [fnum(s.get("round")) for s in ctx.server_rows]
        fig, ax = plt.subplots()
        ax.plot(sr, times, label="round time")
        ax.set_xlabel("round"); ax.set_ylabel("seconds"); ax.legend(); ax.set_title("Runtime")
        save(fig, "runtime_curve.png")
    except Exception:
        pass

    # client accuracy distribution at final round
    try:
        final = ctx.final_eval_round("global")
        accs = [fnum(r.get("test_accuracy")) for r in ctx.client_eval
                if r.get("scope") == "global" and int(fnum(r.get("round"), -1)) == final]
        accs = [a for a in accs if a is not None]
        if accs:
            fig, ax = plt.subplots()
            ax.hist(accs, bins=min(20, max(5, len(accs))))
            ax.set_xlabel("client test accuracy"); ax.set_ylabel("count")
            ax.set_title(f"Client accuracy distribution (round {final})")
            save(fig, "client_accuracy_distribution.png")
    except Exception:
        pass

    # adaptive-only: coefficient + controller-signal curves
    if ctx.is_adaptive:
        coef = read_csv(ctx.dir / "coefficient_trace.csv")
        ctrl = read_csv(ctx.dir / "controller_trace.csv")
        try:
            by_round = {}
            for r in coef:
                by_round.setdefault(int(fnum(r["round"])), []).append(fnum(r.get("rho")))
            xs = sorted(by_round)
            means = [mean(by_round[x]) for x in xs]
            p10 = [percentile(by_round[x], 0.10) for x in xs]
            p90 = [percentile(by_round[x], 0.90) for x in xs]
            fig, ax = plt.subplots()
            ax.plot(xs, means, label="mean rho")
            ax.fill_between(xs, p10, p90, alpha=0.2, label="p10-p90")
            static = fnum(ctx.args.get("mu_base"))
            if static is not None:
                ax.axhline(static, ls="--", c="gray", label="mu_base")
            ax.set_xlabel("round"); ax.set_ylabel("rho"); ax.legend(); ax.set_title("Proximal coefficient")
            save(fig, "proximal_coefficient_curve.png")
        except Exception:
            pass
        try:
            by_round_gap, by_round_lg = {}, {}
            for r in ctrl:
                rd = int(fnum(r["round"]))
                by_round_gap.setdefault(rd, []).append(fnum(r.get("loss_gap_g_i")))
                by_round_lg.setdefault(rd, []).append(fnum(r.get("server_reference_loss_L_g")))
            xs = sorted(by_round_gap)
            fig, ax = plt.subplots()
            ax.plot(xs, [mean(by_round_gap[x]) for x in xs], label="mean loss gap")
            ax.plot(xs, [mean(by_round_lg[x]) for x in xs], label="ref loss Lg")
            ax.set_xlabel("round"); ax.set_ylabel("value"); ax.legend(); ax.set_title("Controller signal")
            save(fig, "controller_signal_curve.png")
        except Exception:
            pass


def main():
    if len(sys.argv) < 3:
        print("usage: finalize_run.py RUN_DIR ALGORITHM [EXIT_STATUS]", file=sys.stderr)
        sys.exit(2)
    run_dir, algorithm = sys.argv[1], sys.argv[2]
    exit_status = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "" else None

    ctx = RunContext(run_dir, algorithm)
    # each artifact is best-effort: one failure must not block the others
    steps = [
        ("config.yaml", lambda: write_config(ctx)),
        ("run_manifest.json", lambda: write_manifest(ctx, exit_status)),
        ("partition_summary.csv", lambda: write_partition_summary(ctx)),
        ("round_metrics.csv", lambda: write_round_metrics(ctx)),
        ("client_metrics_final.csv", lambda: write_client_final(ctx)),
        ("client_metrics_roundwise.csv", lambda: write_client_roundwise(ctx)),
        ("traces", lambda: write_traces(ctx)),
        ("final_metrics.json", lambda: write_final_metrics(ctx)),
        ("plots", lambda: make_plots(ctx)),
    ]
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[finalize_run] WARNING: {name} failed: {exc}", file=sys.stderr)
    try:
        status = write_status(ctx, exit_status)
    except Exception as exc:  # noqa: BLE001
        print(f"[finalize_run] WARNING: status.json failed: {exc}", file=sys.stderr)
        status = "unknown"
    print(f"[finalize_run] {run_dir}: status={status}")


if __name__ == "__main__":
    main()
