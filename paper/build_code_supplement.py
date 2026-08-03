#!/usr/bin/env python3
"""Build the final-submission code/data supplement as a deterministic ZIP."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SOURCE = PAPER / "code_supplement_src"
OUTPUT = PAPER / "adaditto_code_data_supplement.zip"
ARCHIVE_ROOT = "adaditto_code_data_supplement"
MAX_BYTES = 50 * 1024 * 1024
FIXED_TIMESTAMP = (2026, 7, 31, 0, 0, 0)

RESULT_TABLES = (
    "core_partic_sweep_selection.csv",
    "core_participation_summary.csv",
    "cscan_cifar_path_jr0.2_summary.csv",
    "cscan_shakespeare_jr0.1_summary.csv",
    "phase1_selection.csv",
    "phase2_selection.csv",
    "phase3_ablation_summary.csv",
    "phase_dropout_summary.csv",
    "shakespeare_core_summary.csv",
    "shakespeare_shift_selection.csv",
    "shakespeare_sweep_selection.csv",
)

EXPERIMENT_FILES = (
    "aggregate_sweep.py",
    "audit_phase.py",
    "common.sh",
    "controller_scan.sh",
    "core_partic_sweep.sh",
    "core_participation.sh",
    "finalize_run.py",
    "freeze_manifest.sh",
    "phase1_static_sweeps.sh",
    "phase2_adaptive_scan.sh",
    "phase2_confirm_selected.sh",
    "phase3_controller_ablations.sh",
    "phase_dropout.sh",
    "run_one.sh",
    "shakespeare_core.sh",
    "shakespeare_shift.sh",
    "shakespeare_sweep.sh",
    "summarize_phase.py",
    "tuned_baselines.json",
    "validate_run.py",
)


def normalized_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix in {".py", ".sh", ".md", ".yml", ".yaml", ".json", ".csv"}:
        try:
            return data.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
        except UnicodeDecodeError:
            pass
    return data


def trace_destination(summary_name: str, row: dict[str, str]) -> str | None:
    include = (
        summary_name == "phase_dropout_summary.csv"
        and row["dataset"] == "Cifar100_pathological"
        and row["algorithm"] == "AdaProxDitto"
        and "cdr0.0_jr0.2" in row["run_id"]
    ) or (
        summary_name == "shakespeare_core_summary.csv"
        and row["dataset"] == "Shakespeare"
        and row["algorithm"] == "AdaProxDitto"
    ) or (
        summary_name == "core_participation_summary.csv"
        and row["dataset"] == "Cifar100_pathological"
        and row["algorithm"] == "AdaProxFedProx"
        and "jr0.2_seed0" in row["run_id"]
    )
    if not include:
        return None
    return (
        f"experiments/traces/{Path(summary_name).stem}/"
        f"{row['dataset']}/{row['algorithm']}/{row['run_id']}"
    )


def sanitized_csv(path: Path) -> bytes:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        keep_trace_paths = path.name in {
            "core_participation_summary.csv",
            "phase_dropout_summary.csv",
            "shakespeare_core_summary.csv",
        }
        fieldnames = [
            name
            for name in (rows[0].keys() if rows else ())
            if name != "run_dir" or keep_trace_paths
        ]
    if keep_trace_paths:
        for row in rows:
            row["run_dir"] = trace_destination(path.name, row) or ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows({name: row[name] for name in fieldnames} for row in rows)
    return output.getvalue().encode("utf-8")


def sanitized_manifest() -> bytes:
    path = ROOT / "experiments" / "frozen" / "freeze_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("adaptive_confirmation_csv", "adaptive_selection_csv", "static_selection_csv"):
        value = data.get(key)
        if value:
            data[key] = f"experiments/{Path(value).name}"
    data.pop("created_at_unix", None)
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def collect() -> dict[str, bytes]:
    files: dict[str, bytes] = {
        "README.md": normalized_bytes(SOURCE / "README.md"),
        "CLAIMS_TO_ARTIFACTS.md": normalized_bytes(SOURCE / "CLAIMS_TO_ARTIFACTS.md"),
        "IMPLEMENTATION_SPEC.md": normalized_bytes(SOURCE / "IMPLEMENTATION_SPEC.md"),
        "INDEPENDENT_VERIFICATION.md": normalized_bytes(
            SOURCE / "INDEPENDENT_VERIFICATION.md"
        ),
        "environment.yml": normalized_bytes(SOURCE / "environment.yml"),
        "LICENSE": normalized_bytes(ROOT / "LICENSE"),
        "analysis/reference_controller.py": normalized_bytes(
            SOURCE / "reference_controller.py"
        ),
        "analysis/verify_controller_spec.py": normalized_bytes(
            SOURCE / "verify_controller_spec.py"
        ),
        "analysis/verify_reported_results.py": normalized_bytes(
            SOURCE / "verify_reported_results.py"
        ),
        "paper/scripts/make_current_figures.py": normalized_bytes(
            ROOT / "paper" / "scripts" / "make_current_figures.py"
        ),
        "experiments/frozen/freeze_manifest.json": sanitized_manifest(),
    }

    for path in sorted((ROOT / "system").rglob("*.py")):
        files[path.relative_to(ROOT).as_posix()] = normalized_bytes(path)

    dataset_files = [
        ROOT / "dataset" / "generate_Cifar100.py",
        ROOT / "dataset" / "generate_Shakespeare.py",
        *sorted((ROOT / "dataset" / "utils").glob("*.py")),
    ]
    for path in dataset_files:
        files[path.relative_to(ROOT).as_posix()] = normalized_bytes(path)

    metadata = {
        "dataset/partition_metadata/Cifar100_practical_config.json":
            ROOT / "dataset" / "Cifar100_practical" / "config.json",
        "dataset/partition_metadata/Cifar100_pathological_config.json":
            ROOT / "dataset" / "Cifar100_pathological" / "config.json",
    }
    for destination, path in metadata.items():
        files[destination] = normalized_bytes(path)

    for name in EXPERIMENT_FILES:
        path = ROOT / "experiments" / name
        files[f"experiments/{name}"] = normalized_bytes(path)
    for name in RESULT_TABLES:
        files[f"experiments/{name}"] = sanitized_csv(ROOT / "experiments" / name)

    for summary_name in (
        "core_participation_summary.csv",
        "phase_dropout_summary.csv",
        "shakespeare_core_summary.csv",
    ):
        with (ROOT / "experiments" / summary_name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            destination = trace_destination(summary_name, row)
            if destination is None:
                continue
            source = Path(row["run_dir"]) / "results" / "client_controller.csv"
            files[f"{destination}/results/client_controller.csv"] = normalized_bytes(source)

    return files


def add_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data, compresslevel=9)


def main() -> None:
    files = collect()
    checksums = [
        f"{hashlib.sha256(data).hexdigest()}  {name}"
        for name, data in sorted(files.items())
    ]
    files["SHA256SUMS"] = ("\n".join(checksums) + "\n").encode("utf-8")

    with zipfile.ZipFile(OUTPUT, "w") as archive:
        for name, data in sorted(files.items()):
            add_bytes(archive, name, data)

    size = OUTPUT.stat().st_size
    if size > MAX_BYTES:
        raise SystemExit(
            f"{OUTPUT} is {size / 1024 / 1024:.2f} MiB, above the 50 MiB limit"
        )
    print(f"Built {OUTPUT} ({size / 1024 / 1024:.2f} MiB, {len(files)} files)")


if __name__ == "__main__":
    main()
