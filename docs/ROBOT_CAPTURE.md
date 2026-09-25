# Physical-execution recording contract

This is an integration specification, not a bundled web dashboard or converter.
No private deployment stack, driver, camera service, or recording endpoint is
provided by this supplement.

## Recording and motion are independent

A deployment may record continuously or offer separate recording start/stop
buttons. Neither button may start motion, reset/reseed the policy, cancel action
chunks, or replace the existing robot-stop path. Display recording and motion
states separately. A recording may span task attempts, waiting, and rearrangements.

Keep explicit attempt, scene-change, waiting, takeover, and actual reset events.
Unknown labels remain unknown. A segment boundary is not a policy reset or an
autonomous episode. No policy call means no calibration sample. Expert calls
during waiting can be valid inputs, but human/shadow control and interventions
need labels and predefined inclusion rules.

Start/stop acknowledgements identify exact request boundaries. In-flight replies
retain their original immutable session/request IDs. Finishing a recording must
not suppress otherwise valid actions. Motion stop stays independent and immediate.
Only trusted backend IDs resolve storage paths; do not expose arbitrary paths or
shell commands through a dashboard.

## Per-request data

| Item | Requirement |
| --- | --- |
| Camera inputs | All three actual views, lossless, original dtype/shape |
| State | State from the same inference request, not a later sensor read |
| Task | Actual prompt and semantic task ID |
| Initial noise | Actual tensor and dtype/shape; a seed alone is insufficient |
| Native output | Full action chunk before postprocessing |
| Returned output | Postprocessed actions before downstream clipping/smoothing |
| Processed inputs | Tokens/mask/state when available, for parity checks |
| Identity/time | Model, run, session, segment, request, timestamps and clock domains |

Log action publication, truncation, rejection, replacement, takeover, hold, and
reset events separately. Publication alone does not establish physical execution.
Keep actual camera mapping, normalization, chunk length, execution horizon,
generation steps, dtype, attention backend, and any memory/guidance configuration.

Observe sampling noise passively: call the original sampler exactly once, store
an independent snapshot, and return its original value. Do not draw extra noise,
reseed, or change dtype/device. Cover explicit-noise paths too. Unsupported
randomness or memory state must be recorded or marked unreplayable. Measure overhead.

`ReplayCollector.install()` sets controlled noise. Do not install it unchanged
in production. Offline integration must instead consume recorded noise and check
the number, order, shape, and identity of all consumed tensors.

## Storage and completeness

Use bounded asynchronous writing with explicit disk/queue error reporting.
Do not block motion stop on flushing. Missing data makes a recording incomplete;
never silently present it as a complete cache. Rolling shards must not reset the
model. Preserve failed recordings for inspection.

```text
captures/<campaign>/<expert>/<session>/
  manifest.json
  runtime.json
  requests.jsonl
  execution_events.jsonl
  annotations.jsonl
  shards/
    0000/manifest.json
    0000/requests/000000.npz
```

This is an illustrative layout, not an already supported input schema. Use safe
arrays (`allow_pickle=False`, no object arrays) or safetensors. Unsupported dtypes
need lossless bit representation and dtype metadata, not silent conversion.
Atomically publish complete payloads before index entries. Sealing checks unique
IDs, counts, hashes, events, and in-flight completion. Sealed does not mean successful.

Store model/config/tokenizer/processor/statistics hashes and runtime overrides.
A model change starts a new session. Paths are provenance, not cross-machine
identity. Never include credentials or full environment dumps.

## Offline conversion contract

1. Validate sealed recordings, identities, payload hashes, shapes, and finite values.
2. Freeze sessions/segments and control-mode inclusion rules before comparisons;
   do not filter only successful attempts or infer missing intervention labels.
3. Choose disjoint A/B requests from separate execution segments. Keep calibration
   separate from evaluation. The example recipe uses five requests per task and
   calls 0/5/9 out of ten; set `expected_tasks=1` for one-task experts.
4. Reconstruct preprocessing and generation inputs with the same frozen expert.
   Check recorded-noise consumption and native-output parity against a fixed
   tolerance; do not relax it after failures.
5. Export `replay.safetensors` and `replay.json`, retaining source IDs, hashes,
   selection rules, and model identity. This deployment converter must be
   implemented and validated separately; it is not an existing CLI command.

## Transfer to the merge machine

- Complete sealed sessions: manifests, runtime metadata, request/event/annotation
  indices, and every referenced shard and payload.
- Frozen selection and replay-validation plans.
- Complete expert models, processors, normalization, and tokenizer, unless already
  available with identical content hashes; also include the common base.
- Compatible offline preprocessing/conversion code and runtime specifications.

Videos or action logs alone are insufficient. Drivers, ROS environments, keys,
unrelated weights, and online internal features are unnecessary. Transfer only
sealed shards. Keep data/weights separate from the anonymous code archive.

Uncompressed RGB image bytes per request are `sum(height * width * 3)` for uint8
views, plus state, noise, actions, and metadata. Camera FPS is not inference rate.
Measure a short recording before allocating storage; do not assume compression
ratios or overwrite older sessions. Selected-request subsets need their own
validated provenance manifests, not manual deletion of unselected payloads.
