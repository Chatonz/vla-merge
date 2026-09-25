# Using TCR

[← Back to the project](../README.md)

## Installation

Detailed instructions: [environment setup](SETUP.md).

Use Python 3.12 (required by the pinned LeRobot revision) and a compatible pi0.5 environment. The extraction
environment used Python 3.12, PyTorch 2.7.1, Transformers 5.5.4, LeRobot 0.6.2,
PEFT 0.20.0 and safetensors 0.5.3. The block-replay adapter uses version-specific
LeRobot/Transformers internals; these are not interchangeable with arbitrary releases.

```bash
# From the root of your local TCR checkout:
python -m pip install -c requirements/constraints.txt -e '.[pi05,test]'
# Optional: LIBERO rollout collection and evaluation
python -m pip install -c requirements/constraints.txt -e '.[libero]'
pytest -q
```

With a preconfigured, matching LeRobot environment, install this package without
changing the runtime: `python -m pip install -e . --no-deps`.
LIBERO assets and simulator setup remain external prerequisites.

## 1. Prepare experts

Provide a shared dense base and at least two compatible dense expert directories.
The paper recipe uses four LIBERO suite specialists. Each directory needs a
`model.safetensors`, `config.json`, local `tokenizer/`, and matching LeRobot
pre/postprocessor files and normalization tensors. Sharded and OpenPI/JAX
checkpoints are not supported by this adapter.

To materialize an existing standard PEFT LoRA specialist:

```bash
tcr export-adapter --base-model checkpoints/base \
  --adapter checkpoints/spatial_adapter --output checkpoints/spatial
```

Export preserves the CPU safe-merge rounding order. Advanced PEFT variants are
explicitly rejected. The merger rejects differing expert tensors outside the
paper's supported adapted scope rather than silently discarding them.

## 2. Collect two execution caches

For each expert, collect complete frozen-expert episodes. Cache A uses five
reservoir-selected requests per task; cache B uses five requests spread across
a separate episode. Both retain native calls 0, 5 and 9 of ten generation calls.

Example for the Spatial expert (repeat for Object, Goal and Long):

```bash
tcr collect --name spatial --policy checkpoints/spatial --suite libero_spatial \
  --selection reservoir --init-state-offset 0 --seed 271001 \
  --noise-seed 272001 --output caches/a/spatial
tcr collect --name spatial --policy checkpoints/spatial --suite libero_spatial \
  --selection quantiles --init-state-offset 1 --seed 271101 \
  --noise-seed 272101 --output caches/b/spatial
```

These offsets/seeds are illustrative, not the historical paper evaluation bank.
Keep calibration and evaluation episodes separate. Each cache contains
`replay.safetensors` and `replay.json`; no demonstration actions are needed.
For a different rollout environment, use [the integration example](../examples/capture_policy.py).

## 3. Merge

Edit [configs/pi05.json](../configs/pi05.json). Paths resolve relative to the config
file, not the current working directory.

```bash
tcr check --config configs/pi05.json
tcr merge --config configs/pi05.json --dry-run
tcr merge --config configs/pi05.json
```

Pass 1 starts from the expert mean with relative-error masses. Pass 2 starts
from pass 1 with uniform masses and inherits its numerical ridge map. Each pass
runs in a separate process. The resulting policy is `outputs/main/merged/`;
`outputs/main/pass1/` is also retained. Both contain a `tcr_manifest.json` with
input identities, per-module row counts, numerical settings and output hashes.
Outputs are never overwritten automatically.

This implementation retains expert weights and replay states in memory and is
intended for machines with substantial host RAM and GPU memory. No automatic
GPU allocation, queue management or memory-capacity claim is included.

## Ablations and small parameter studies

The core controls are demonstration observations, final generation call only,
and expert-prefix features. A matched Full run supplies the ridge map. See
[the experiment guide](experiments.md) for demo capture and fair comparisons.

```bash
tcr sweep --config configs/pi05.json --study ablations \
  --demo-config configs/demo.json --output runs/ablations
tcr sweep --config configs/pi05.json --study ridge --output runs/ridge
tcr sweep --config configs/pi05.json --study budget --output runs/budget
tcr sweep --config configs/pi05.json --study passes --output runs/passes
tcr sweep --config configs/pi05.json --study weighting --output runs/weighting
```

`sweep` writes configs and `commands.json`; it does not start experiments. Execute
the listed commands in order. Review data paths and evaluate every arm, including
regressions. The parameter grids are exploration examples, not measured results.

## Evaluate

```bash
tcr evaluate --policy outputs/main/merged --suite libero_spatial \
  --episodes 10 --seed 274001 --init-state-offset 20 --output outputs/eval/spatial
```

Repeat across suites and prespecified repeat seeds/reset selections. This is a
minimal LIBERO evaluation entry point, not a replacement for the paper's exact
procedural reset-bank protocol.
