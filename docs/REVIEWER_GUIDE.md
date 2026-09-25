# Reviewer guide

## Repository scope

This repository implements TCR for pi0.5 and includes demonstration and comparison videos.
The package name is `tcr_merging`; the CLI is `tcr`. The source includes four
baseline entries, component controls, small parameter grids, configurations, and tests.
Weights, datasets, calibration caches, historical evaluation logs, and a physical
deployment service are not included. See the [video index](VIDEOS.md) for media.

The implementation and numerical configuration examples are preserved from the
original source archive. The public repository adds media, presentation files,
and documentation; historical source-package checks are identified separately.

## Suggested review order

1. Read the method-to-code mapping in [reproduction.md](reproduction.md).
2. Install the pinned runtime using [SETUP.md](SETUP.md).
3. Run `pytest -q -rs` for equations, workflows, and reduced-dimension integration.
4. Run `tcr merge --config configs/pi05.json --dry-run` to inspect the two-pass plan.
5. With compatible models/caches, follow the README for collection, merging, and evaluation.

Steps 3 and 4 need no full checkpoints or robot access. They do not replace
full-budget closed-loop evaluation. Native-layer tests require the optional
LeRobot/Transformers runtime.

## Optional upstream comparison

One FeatCal test requires an independently supplied upstream solver, not bundled
here. Set `FEATCAL_REFERENCE_SOLVER=/absolute/path/to/featcal/solver.py` to enable
it. Without the variable the test explicitly skips; an invalid explicit path
fails. Independent equation-oracle tests remain active without this reference.

## Validation and limitations

Historical source-package checks and counts are in [SUBMISSION_CHECKS.md](SUBMISSION_CHECKS.md).
Prior full-sized reduced-budget smoke tests are separately described in
[VALIDATION.md](VALIDATION.md). No new full-budget GPU experiment is claimed.
Machine-specific validation logs are not included.
