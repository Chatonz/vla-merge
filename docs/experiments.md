# Small experiment suite

Run the matching Full configuration first. All paired component controls use
its numerical per-module ridge map. Each variant's second pass starts from that
variant's own first pass, never the Full first-pass weights.

| Variant | Intervention | Budget consequence |
| --- | --- | --- |
| `full` | Frozen-expert execution, calls 0/5/9, merged-prefix replay | Paper caps 10 per call/view in pass 1; 16 shared per request/module in pass 2 |
| `demo` | Demonstration observations; frozen expert still runs its native generator | Same task/request/call and row allocations |
| `final_call` | Retain only native call 9 from the same raw caches | Pass-1 rows decrease; not an equal-row comparison |
| `expert_prefix` | Propagate each expert's own preceding blocks | Same requests/calls/rows; local inputs change |

The four-expert Full recipe realizes 4,441,200 rows in pass 1 and 1,331,600 in
pass 2. `strict_paper_budget` enforces the per-module counts, not merely totals.
Final-call pass 1 realizes 1,480,400 rows. Row caps are not comparable without
their allocation rules. Stored observations remain paired with their actual
generation states; removing calls is performed at load time, leaving raw caches intact.

## Demonstration observations

Prepare a JSON list of prespecified demonstration episode indices, one per task,
from your training/calibration split (not the evaluation split). Provide the
local LeRobot dataset and the matching execution cache as a noise template:

```bash
tcr collect-demo --name spatial --policy checkpoints/spatial \
  --dataset data/libero --episodes data/spatial_demo_episodes_a.json \
  --template caches/a/spatial --output caches/demo_a/spatial
tcr collect-demo --name spatial --policy checkpoints/spatial \
  --dataset data/libero --episodes data/spatial_demo_episodes_b.json \
  --template caches/b/spatial --output caches/demo_b/spatial
```

Repeat for every expert. The collector uses five observation positions per
demonstration, runs native expert generation, and pairs initial generation noise
with the template's five requests. Demonstration action labels are never passed
to the model. Frame/episode identities and template hashes are recorded.

Edit `configs/demo.json` to point to these caches, then generate the ablation
matrix as shown in the README. The generated matrix contains matched Full,
final-call, expert-prefix and demo configurations. Generation/prefix variants
verify that raw cache hashes match their Full reference.

The paper's main Full/demo comparison and its refreshed generation/prefix
comparison used different matching constructions. A newly generated matrix is
a new matched study; do not subtract its scores from the published main mean.

## Parameter studies

- `ridge`: ratios 0.01, 0.05, 0.1. Each first pass computes its own ridge map;
  its second pass inherits that map.
- `budget`: request row caps 8, 16, 24 in pass 2, keeping the same raw caches and
  reference numerical ridge. The linspace-based subsets are **not guaranteed
  nested**; this is an exploratory cap sweep, not the appendix's planned nested-row study.
- `passes`: one pass versus two passes. Two passes also change the prior and
  expert masses, so this is a recipe comparison rather than a pure iteration effect.
- `weighting`: relative-error versus uniform first-pass expert masses; the
  second pass remains uniform. Both use the relative reference's ridge map.

These generators write explicit configurations only. No metrics are invented,
no best configuration is selected automatically, and no benchmark jobs launch
until the user runs the generated commands. Keep evaluation resets, seeds and
native inference settings matched across arms. Report per-suite results and all
prespecified variants, not just the best setting.
