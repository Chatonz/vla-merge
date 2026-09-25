# Historical source artifact checks

> This record describes the original anonymous source archive. It predates the
> GitHub repository and its video collection; the counts and packaging checks
> below are historical, not a claim about the current checkout.
> For public repository setup, see [Publishing](PUBLISHING.md).

## Completed checks

- Built a wheel from a separate copy and installed it into a separate target.
- Verified imports resolved to the newly installed wheel rather than another checkout.
- Ran the installed package's suite: **87 passed, 1 skipped**, 88 collected.
- The one explicit skip is the optional upstream FeatCal solver comparison;
  its source is not distributed. Independent FeatCal equation tests passed.
- Native reduced-dimension Gemma/replay tests ran with the pinned runtime.
- Verified the installed CLI generates the two-pass merge plan without weights.
- Core `src/` files and numerical configuration examples are byte-identical to
  the source implementation; no numerical method changes were made for anonymity.
- Checked Python syntax, JSON parsing, and local Markdown links.
- Scanned the complete source artifact for CJK text, direct author/account
  identifiers, internal machine paths, credential patterns, and private remotes.
- Excluded Git metadata, caches, environments, build products, weights, and logs.
- ZIP entries use a fixed neutral timestamp, generic file permissions, no extra
  metadata fields, and no archive comment. File hashes and ZIP CRCs were checked.

## Test environment boundary

The checks used the documented dependency versions in an existing compatible
Python 3.12 runtime. The host also had an unrelated optional Transformer Engine
extension that attempted CUDA initialization even for CPU tests. The test process
used a temporary dependency view excluding that optional extension, with GPUs
hidden. No shared dependency was uninstalled or changed, and no solver or native
replay implementation was replaced. A clean environment created without inherited
system packages, as shown in SETUP.md, avoids this particular host extension.

This was not a full fresh-machine installation or a full-budget GPU merge.
No robot was controlled, no new benchmark episodes were evaluated, and no
historical scores were regenerated. Prior smoke checks remain clearly separated
in VALIDATION.md and BASELINE_VALIDATION.md.

## Original anonymous-submission instructions (historical)

Submit the source ZIP itself through the anonymous venue channel. Do not attach
the original Git checkout, private model/cache archives, server logs, or an
identifying account URL. The archive does not include weights or the historical
calibration/evaluation selections required to reproduce exact paper scores.
If those artifacts are requested, prepare a separate permitted anonymous release.
Direct-identifier checks do not prevent linkage through public method names or
matching source code. Venue-specific upload limits still need to be checked.
