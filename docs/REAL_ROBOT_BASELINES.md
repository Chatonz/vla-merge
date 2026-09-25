# Offline merging for physical-robot experts

Independent Model Soups, TIES, RegMean++, FeatCal, and TCR entries read local
artifacts and export models. They do not connect to hardware or start training.

## Inputs

Use [SETUP.md](SETUP.md) in a separate offline environment. LIBERO is unnecessary
for merging existing physical-execution caches. Configure `configs/real_robot.json`
with the common base, at least two compatible dense experts, and their A/B caches.
Task names `task1/task2` are placeholders that must match the caches. A third
compatible expert can be added. `strict_paper_budget=false` permits non-LIBERO
task counts, not incompatible checkpoints.

Models need complete tokenizer, processor, normalization, and configuration
files. The supported scope is 418 linear modules / 422 tensors. Do not overwrite
differing task statistics to circumvent checks. Materialize LoRA adapters first.

RegMean++, FeatCal, and TCR require `replay.safetensors` and `replay.json` from
the frozen experts, including native calls 0/5/9 and prefix inputs. Video, ROS
bags, or action logs alone are not replay caches. Baselines use one A-cache pass;
TCR uses A/B in two passes. Their total calibration budgets are not identical.

The [capture contract](ROBOT_CAPTURE.md) describes deployment integration not
implemented here. `examples/capture_policy.py` is a controlled rollout example
that sets sampling noise, not a passive production recorder. For a one-task
expert, set `CaptureConfig.expected_tasks=1` instead of the default 10.

## Commands

```bash
python scripts/check_env.py --cuda
tcr check --config configs/real_robot.json
tcr baseline --method soup --config configs/real_robot.json --output outputs/real_robot/soup
tcr baseline --method ties --config configs/real_robot.json --output outputs/real_robot/ties \
  --ties-keep-fraction 0.2 --ties-alpha 1.0
CUDA_VISIBLE_DEVICES=0 tcr baseline --method regmeanpp --config configs/real_robot.json \
  --regmeanpp-solver adapted --output outputs/real_robot/regmeanpp
CUDA_VISIBLE_DEVICES=0 tcr baseline --method featcal --config configs/real_robot.json \
  --featcal-lambda 0.05 --featcal-rho 2.0 --featcal-alpha 0.3 \
  --output outputs/real_robot/featcal
CUDA_VISIBLE_DEVICES=0 tcr merge --config configs/real_robot.json
```

Add `--dry-run` to preview baseline commands; it does not validate full weights.
Existing outputs are rejected. Soup/TIES do not read calibration caches or load
the inference model, but still require model support files. TIES uses substantial
CPU memory for global task vectors. No minimum hardware specification is claimed.

| Method | Implementation boundary |
| --- | --- |
| Model Soups | Uniform dense expert average; no calibration or test-set weight search |
| TIES | Global base-relative trimming, sign election, disjoint mean; not layerwise trimming |
| RegMean++ adapted | One pass, uniform expert masses, merged-prefix replay, Soup-centered ridge and correction cap |
| FeatCal | Full-expert teacher features, refreshed student features, independent weight/bias equations |
| TCR | Relative-error first pass; uniform second pass inherits ridge, with separate A/B caches |

The separate original RegMean++ option is:

```bash
CUDA_VISIBLE_DEVICES=0 tcr baseline --method regmeanpp --config configs/real_robot.json \
  --regmeanpp-solver original --offdiag-scale 0.95 \
  --output outputs/real_robot/regmeanpp_original
```

It uses off-diagonal Gram shrinkage without ridge, TCR caps, or pseudoinverse
fallback. It requires equal rows per expert and a scale strictly between 0 and 1.
Failure stops the run, not an automatic switch to `adapted`. Choose the solver
before evaluation; do not select between versions after seeing scores.

FeatCal uses separate teacher-feature and student-calibration processes and
retains `.featcal-features-*` audit intermediates, which can be large. Parameters
above are example recipes, not tuned hardware optima.

## Delivery and evaluation

- Baseline outputs contain a complete model and `baseline_manifest.json`.
- TCR exports `outputs/real_robot/tcr/merged/` and retains `pass1/` with manifests.
- Transfer the entire model directory, including tokenizer/processors/statistics.
- Preserve base/expert identities, configurations, cache hashes, and code versions.
- Do not submit tokens, keys, environment directories, or private images with code.

First verify action shape, finite values, normalization, and deployment parity
on fixed observations. Physical validation requires an operator and existing
safety procedures. Never bypass joint limits, tracking-error protections, or
emergency stops. Ending recording does not stop motion.

The two-task paper protocol uses 20 attempts per task per method, 40 per row,
for Experts, Soup, TIES, RegMean++, FeatCal, and TCR. Expert references use the
corresponding specialist. Freeze layouts and method order before formal testing,
separate calibration from evaluation, and log all attempts. The task limit is
60 seconds. Takeovers and policy-caused timeouts/safety stops are not autonomous
successes. Missing results remain missing; offline errors do not replace success.
