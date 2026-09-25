# Validation scope

For a guide to checking this repository, see [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md).
Historical source-archive checks are recorded in [SUBMISSION_CHECKS.md](SUBMISSION_CHECKS.md).
The records below describe prior implementation validation, not new experiments
performed by reviewers or bundled benchmark evidence.

For the later four-baseline extension, see [BASELINE_VALIDATION.md](BASELINE_VALIDATION.md).
The real-model smoke below applies to the original TCR release, not the added baselines.

These checks establish implementation and integration behavior, **not benchmark
success rates**. Models and private calibration inputs are not distributed.

## Automated and packaging checks

- 42 tests passed in the extraction runtime, including native LeRobot Gemma
  decoder/cache comparisons, regression equations, sampling, adapter rounding,
  collector lifecycle, configuration and experiment generation.
- The extracted regression kernel matched the original implementation bit for
  bit in 16 cases: four seeds, two expert-mass choices and two ridge settings.
- Ruff lint and formatting checks passed.
- Built and installed the wheel into a separate target directory. Imports,
  installed `tcr` dry-run and the test suite worked from outside the source tree.
- Dependency resolution succeeded for both `pi05,test` and
  `pi05,test,libero`, with the supplied constraints and pinned public LeRobot
  revision. This is not a full fresh-machine installation test.
- CUDA, LeRobot/Transformers interfaces and LIBERO/robosuite/MuJoCo imports
  passed the environment checker. An import check does not launch a simulator.

## Real pi0.5 two-pass smoke test

Used full-sized dense Spatial and Object experts, their common base, and real
execution-cache subsets. Each expert contributed one request with calls 0/5/9
to each pass. Both row caps were reduced to 1 and `strict_paper_budget` was
disabled. No network dimensions or merge-module scope were reduced.

| Check | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| Solved linear modules | 418 | 418 |
| Adapted tensor scope | 422 | 422 |
| Realized regression rows | 4,452 | 836 |
| Manual/native action-block maximum error | 0 | 0 |
| Finite exported weights | Passed | Passed |

All 418 numerical ridge values matched across passes. The second-pass prior
hash matched the first-pass checkpoint hash. Both output manifests were written.

The final checkpoint then loaded successfully through native `PI05Policy`.
On synthetic image/state observations with a valid cached prompt, each of two
collector modes ran seven native action-chunk requests. Every output had shape
`(1, 50, 7)` and finite values. Both reservoir and quantile collection selected
five requests, saved 15 generation-state records, and reloaded all 15 records
with three vision inputs per record. Synthetic observations test the interface;
they do not measure robot behavior.

Runtime: Python 3.12.3, PyTorch 2.7.1+cu128, Transformers 5.5.4, LeRobot commit
`bf31dd794ffb4f87380aba3912f64421e8352d3c`, A100 80GB. See
[environment setup](SETUP.md) for dependencies and limitations.

## Not established by this release check

- Four-expert, full-budget paper scores or held-out closed-loop LIBERO success.
- End-to-end demonstration-dataset capture or full-budget ablation/sweep runs.
- Minimum hardware requirements or compatibility with other runtime versions.

The corresponding entry points and configuration generators are included;
run those experiments with compatible artifacts and matched evaluation splits.
