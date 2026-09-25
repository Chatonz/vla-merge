# Environment setup

Run commands from the repository root. Merging existing caches requires
the model runtime; collecting/evaluating LIBERO episodes additionally requires
the simulator and its assets. No private server or original workspace is needed.

## Python and dependencies

Use Linux and Python 3.12. The implementation was checked with PyTorch 2.7.1,
torchvision 0.22.1, Transformers 5.5.4, PEFT 0.20.0, safetensors 0.5.3, and
LeRobot revision `bf31dd794ffb4f87380aba3912f64421e8352d3c` (version 0.6.2).
The constraints file is not a complete cross-platform lock.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements/constraints.txt -e '.[pi05,test]'
python -m pip check
python scripts/check_env.py --cuda
pytest -q -rs
```

The CUDA wheel needs a compatible NVIDIA driver; the local `nvcc` version alone
does not establish compatibility. Installation requires Git and network access
for the pinned public dependency. With an already matching runtime, use
`python -m pip install -e . --no-deps` to avoid changing its dependencies.
Do not upgrade dependencies in a production robot-control environment.

CPU numerical tests do not require GPU execution. Full pi0.5 merging holds
multiple expert weights and replay states, requiring substantial host/GPU
memory. No minimum-memory or arbitrary-device compatibility claim is made.

## Inputs

```text
checkpoints/{base,spatial,object,goal,long}/
caches/a/{spatial,object,goal,long}/
caches/b/{spatial,object,goal,long}/
```

Every dense model needs `model.safetensors`, `config.json`, a local `tokenizer/`,
`policy_preprocessor.json`, `policy_postprocessor.json`, and all normalization
tensors referenced by those processors. Experts must share a compatible complete
initialization, architecture, tokenizer, and normalization contract. Materialize
standard PEFT adapters using the README command first. Do not replace differing
normalization files merely to bypass compatibility checks.

Each cache contains `replay.safetensors` and `replay.json`, with native generation
states, three visual inputs, and provenance. Caches must correspond to their
frozen expert; demonstration actions are not substitutes for generation states.
After moving caches, verify model content identities and relocate paths without
discarding the original provenance.

Edit `configs/pi05.json`; paths resolve relative to that file:

```bash
tcr check --config configs/pi05.json
tcr merge --config configs/pi05.json --dry-run
CUDA_VISIBLE_DEVICES=0 tcr merge --config configs/pi05.json
```

The final model is `outputs/main/merged/`. Commands do not download weights or
overwrite existing outputs. Dry runs do not prove model or cache compatibility.

## Optional LIBERO setup

```bash
python -m pip install -c requirements/constraints.txt -e '.[libero]'
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Prepare the BDDL, initial states, and assets from the
[public LIBERO project](https://github.com/Lifelong-Robot-Learning/LIBERO).
Installing the Python package alone does not provide all required assets.
Create `config.yaml` in a directory pointed to by `LIBERO_CONFIG_PATH`, replacing
these example paths with your local locations:

```yaml
benchmark_root: /absolute/path/to/LIBERO/libero/libero
bddl_files: /absolute/path/to/LIBERO/libero/libero/bddl_files
init_states: /absolute/path/to/LIBERO/libero/libero/init_files
datasets: /absolute/path/to/libero_datasets
assets: /absolute/path/to/LIBERO/libero/libero/assets
```

Run `python scripts/check_env.py --cuda --libero` before collection. The suite
names are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10` (Long).
Follow the README to collect separate A/B episodes. Demonstration-input controls
also require a local LeRobot dataset and predetermined episode lists; see
[experiments.md](experiments.md).

## Troubleshooting

- Import failure: install this package in the same environment as `python` and `tcr`.
- Attention/cache mismatch: use the pinned runtime; do not bypass replay checks.
- Tokenizer failure: verify local assets and relocate stale processor paths.
- Unsupported tensor differences: check common initialization and adapted scope.
- Interactive LIBERO setup: prepare the configuration and referenced assets first.
- Out of memory: a reduced-budget diagnostic is a different experiment, not an
  exact reproduction of the full calibration budget.

The environment checker only checks imports, interfaces, and device visibility;
it does not load a full policy, launch a simulator, or control hardware.
