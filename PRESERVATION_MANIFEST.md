# PFLlib Research Preservation Snapshot

This branch is a clean publication snapshot of the local
`paper-aaai-rewrite` research branch at `dce086e06a475599cc289a39330b704d0527e05b`.
It is based on the GitHub `origin/master` tip
`0bcc46561c8dc31e59e0c5c8cff9c36d9bd36cb6`.

The original local branch contains 34 unpublished commits and remains intact.
It could not be pushed directly because its history contains files above
GitHub's 100 MB per-file limit.

## Preserved

- adaptive proximal client and server implementations;
- dataset-generation and preprocessing source;
- experiment drivers, frozen settings, and compact result tables;
- paper and supplement sources, figures, and author materials;
- implementation and independent-verification specifications;
- controller traces and aggregate evidence selected by the supplement builder;
- deterministic code-data supplement
  `paper/adaditto_code_data_supplement.zip`.

The supplement SHA-256 is:

```text
fda8ea555f9f2be5550a132d0236a29136d6cc4f6fa9f1e4766c7a20d1e9312b
```

## Deliberately excluded

- Miniconda installers and local environments;
- raw or generated dataset bodies;
- `experiments/runs/` and terminal logs;
- checkpoints, caches, and compiled files;
- machine-local system result dumps.

These exclusions remove redistributable or regenerable bulk, not the compact
evidence used by the paper's claims. Raw third-party datasets remain obtainable
from their upstream sources. Full experiments can be rerun from the preserved
drivers and environment specification, subject to dependency and hardware
availability.

## Verification

Extract the supplement and run its staged protocol:

```bash
unzip paper/adaditto_code_data_supplement.zip
cd adaditto_code_data_supplement
sha256sum -c SHA256SUMS
python3 analysis/verify_controller_spec.py
python3 analysis/verify_reported_results.py
```

See `INDEPENDENT_VERIFICATION.md` inside the archive for compilation, shell
syntax, controlled-comparison, and figure-regeneration checks.
