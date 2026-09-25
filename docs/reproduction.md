# Reproduction scope

This repository packages the pi0.5 implementation of TCR and accompanying
demonstration videos. The numerical implementation is unchanged from the
original source archive.

| Component | Source |
| --- | --- |
| Prior-centered ridge regression and correction safeguard | `src/merging/regression.py` |
| Two-pass construction and ridge inheritance | `src/merging/engine.py`, `src/cli.py` |
| Forward-order native block replay | `src/pi05/replay.py` |
| Expert-execution and demonstration-input collection | `src/calibration/` |
| Request/view row allocation | `src/merging/sampling.py` |
| Standard PEFT adapter materialization | `src/pi05/adapters.py` |
| Model Soups, TIES, RegMean++, and FeatCal | `src/baselines/` |
| Matched controls and small parameter grids | `src/experiments.py` |

Scheduling, recovery, machine paths, private deployment implementations, and
result archives are excluded. Dense expert directories are the portable inputs.
This adapter checks an explicit adapted scope; it is not an arbitrary-architecture
or unrestricted full-finetuning merger. OpenVLA and Fast-WAM are not implemented
by this pi0.5 package, and continuation training is not bundled here.

## Matching the paper

Exact score reproduction also requires the original common base, four 10k-step
experts, normalization/tokenizer, calibration episode and generation-noise
identities, construction selections where applicable, and paired held-out
evaluation reset banks. These artifacts are not included in this repository. The example commands define a new run and do not reconstruct the
historical cache or reset selections. No published score is supplied as a
substitute for a missing measurement.

The supported LeRobot source revision is
`bf31dd794ffb4f87380aba3912f64421e8352d3c`. Native Gemma attention/cache interfaces
must match the replay adapter. A manual/native action-block comparison runs
during merging and raises on disagreement. Stored-reference feature differences
under changed weights are diagnostics, not replay-equivalence proofs.

The implementation retains local ridge solves, float32 corrections, fixed
within-pass priors, bias augmentation, correction-norm safeguards, block-local
expert computations, producer refresh, and forward-order replay. Unsupported
changed tensors are rejected rather than assigned an undocumented fallback.

## Baseline boundaries

Model Soups averages dense experts. TIES operates on global task vectors.
The default RegMean++ is the explicitly stabilized, one-pass uniform VLA
adaptation. The separate `original` solver uses an unregularized Gram system
with off-diagonal shrinkage and accepts only scales in (0,1).
FeatCal uses its own weight/bias equations, full-expert teacher features, and
student-feature refresh after each producer. Shared traversal does not mean
that these methods share the TCR objective.

These portable integrations do not establish that a particular new checkpoint
matches a historical baseline run. Physical-robot caches and the simulation
baselines' static-observation calibration protocols are separate.

## What can be checked without weights

Tests cover regression equations, weighting, ridge inheritance, nullspace prior
retention, sampling, prefix propagation, adapter rounding, configuration,
experiment generation, and reduced-dimension integration over the 418 supported
module names. Native-layer tests require the pinned optional runtime.
The only optional external-reference comparison uses `FEATCAL_REFERENCE_SOLVER`.

See [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) for package checks and
[VALIDATION.md](VALIDATION.md) for the clearly separated prior smoke-test scope.
None of these checks is a new closed-loop success-rate measurement.
