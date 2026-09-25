# Prior baseline extension validation

This is an offline-code release check, not a real-robot or full-sized pi0.5 success claim.

## Completed

- 88 tests passed with the independently built/installed package, PyTorch 2.7.1,
  Transformers 5.5.4, and the pinned local LeRobot source revision
  `bf31dd794ffb4f87380aba3912f64421e8352d3c`. No tests were skipped in the final suite.
- Equation checks for FeatCal weights/biases include comparison with the locally
  available frozen upstream solver. That upstream source is not bundled.
  In this supplement the optional test requires `FEATCAL_REFERENCE_SOLVER`;
  when absent, the test is explicitly skipped rather than counted as passed.
- TIES global trimming/sign election matches an independent transcription and
  chunked/non-chunked reference kernels. No per-layer approximation is used.
- Original RegMean++ matches a three-expert explicit Gram-system oracle, including
  a row-space case and zero-support columns. Its numerical equation is not TCR ridge.
- Tiny CPU integration traverses all 418 production module names, using three
  experts and reduced tensor dimensions, and exports/reloads Soup, TIES, adapted
  RegMean++, original RegMean++, FeatCal, and TCR with 422 target tensors.
- FeatCal teacher feature files and receipt hashes are written/read independently
  of its student phase. Dependency ordering and separate weight/bias updates are tested.
- A two-pass TCR integration test checks that all 418 ridge values are inherited
  and the second-pass prior hash matches the first-pass output.
- Missing/out-of-scope adaptations, existing output directories, incompatible
  ablation configs, and nonfinite solver inputs have rejection tests.
- Native tiny Gemma decoder/cache parity tests pass with actual LeRobot/Transformers
  layers. The 418-module integration fixture uses a synthetic graph, not a real pi0.5.
- Ruff lint/format and independent wheel installation pass; CLI dry runs do not
  load weights, start GPU jobs, or communicate with a robot.

## Not yet established

- Full-sized pi0.5 baseline construction and native action parity of those outputs.
- End-to-end physical session recording, offline conversion, or compatibility with
  the local deployment runtime; the bundled recording document remains an implementation spec.
- Real-robot closed-loop success, full-budget baseline memory requirements, or a
  minimum GPU/RAM specification. There were no new robot rollouts in this release check.
- Exact reproduction of the simulation table: the hardware recipe uses physical
  execution caches; the simulation baselines' historical static-observation inputs
  and fixed checkpoint provenance are separate.

The original TCR release's real two-expert reduced-budget smoke test is documented
in [VALIDATION.md](VALIDATION.md); it is not reused as evidence that the newly added
four baseline integrations have already run on full-sized checkpoints.
