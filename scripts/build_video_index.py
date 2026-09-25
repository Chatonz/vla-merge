#!/usr/bin/env python3
"""Generate the GitHub video index from assets/media.json (standard library only)."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "soup": "Soup", "ties": "TIES", "regmeanpp": "RegMean++", "featcal": "FeatCal",
    "regmean": "RegMean", "knots_ties": "KnOTS-TIES",
    "task_arithmetic": "Task Arithmetic", "wudi": "WUDI",
    "tcr": "**TCR (ours)**", "experts": "**Expert (reference)**",
}


def main():
    videos = json.loads((ROOT / "assets/media.json").read_text())["videos"]
    lines = [
        "# Video collection", "", "[← Back to TCR](../README.md)", "",
        "All **168 supplied recordings**: 48 real-robot clips and 120 LIBERO episodes. "
        "Click a thumbnail or a recording link to open its MP4.", "",
        "The main comparison order is **Soup → TIES → RegMean++ → FeatCal → TCR → Expert**. "
        "Expert is a reference policy and is listed last. Additional LIBERO baselines "
        "appear before TCR and the Expert reference in the complete tables.", "",
        "[Real robot](#real-robot) · [LIBERO](#libero) · [Provenance](#provenance)", "",
        "## Real robot", "",
        "Five merging methods plus the Expert reference, two tasks, and four recordings per entry/task. Task names "
        "follow the supplied folder labels. No success/failure annotations were "
        "provided for these recordings.", "",
    ]
    groups = [
        ("real_robot", "task1", "Task 1"),
        ("real_robot", "task2", "Task 2"),
        ("libero", "libero_spatial", "LIBERO-Spatial"),
        ("libero", "libero_object", "LIBERO-Object"),
        ("libero", "libero_goal", "LIBERO-Goal"),
        ("libero", "libero_10", "LIBERO-Long"),
    ]
    for domain, task, title in groups:
        if task == "libero_spatial":
            lines += [
                "## LIBERO", "",
                "Nine merging methods plus the Expert reference, four suites, and three episodes per entry/suite. "
                "All supplied episodes are task `00`, run `01`. Outcome labels "
                "come from source filenames and are not independently re-evaluated. "
                "These clips are qualitative examples, not aggregate benchmark results.", "",
                "LIBERO-Long uses the `libero_10` directory. The gallery and MP4s "
                "retain the supplied playback timing.", "",
            ]
        lines += [f"### {title}", "", "| Method / reference | Preview | Recordings |", "| :--- | :---: | :--- |"]
        for method, label in METHODS.items():
            items = sorted(
                (v for v in videos if v["domain"] == domain and v["task"] == task and v["method"] == method),
                key=lambda v: v["path"],
            )
            if not items:
                continue
            first = items[0]
            width = 200 if domain == "real_robot" else 100
            poster = (
                f'<a href="../{first["path"]}"><img src="../{first["poster"]}" '
                f'alt="{method} {title} preview" width="{width}"></a>'
            )
            links = []
            for video in items:
                if domain == "real_robot":
                    name = Path(video["original_filename"]).stem
                else:
                    episode = Path(video["path"]).stem.split("_")[2]
                    name = f'{episode} · {video["outcome"]}'
                links.append(f'[{name}](../{video["path"]}) ({video["duration"]:.1f}s)')
            lines.append(f"| {label} | {poster} | {' · '.join(links)} |")
        lines.append("")
    lines += [
        "## Provenance", "",
        "- Real-robot sources: `视频.zip`; method folders TCR, Expert, Featcal, "
        "RegMean++, Mean Soup, and TIES. All 48 source clips are represented.",
        "- Simulation sources: `videos(1).zip`; all 120 MP4s are retained byte-for-byte. "
        "Method/suite names and success/failure labels follow the supplied files.",
        "- Real-robot derivatives: full-duration H.264 MP4, at most 1280 × 720, "
        "30 fps, muted, and without source location/device metadata. Original filenames "
        "are preserved in the manifest; MP4 filenames are lowercase for portable links.",
        "- README previews: TCR real-robot GIFs at 2× source speed and LIBERO GIFs "
        "at 0.25× supplied file playback speed, including failure examples. "
        "Six comparison panels show the selected methods at those same speeds, "
        "with final frames held for ended clips. See the "
        "[comparison manifest](../assets/comparisons/manifest.json) for exact inputs. "
        "The 168 original recording MP4s keep their supplied durations.",
        "- Original ZIPs and MOVs are kept outside the repository. "
        "Videos are demonstrations, not model weights, datasets, or calibration caches.", "",
        "See [assets/media.json](../assets/media.json) for source filenames, paths, "
        "durations, dimensions, and outcome labels. For browser playback and "
        "GitHub Pages setup, see [Publishing](PUBLISHING.md).", "",
        "To rebuild this index after updating the manifest:", "",
        "```bash", "python scripts/build_video_index.py", "```", "",
    ]
    destination = ROOT / "docs/VIDEOS.md"
    destination.write_text("\n".join(lines))
    print(f"Wrote {destination.relative_to(ROOT)} for {len(videos)} videos")


if __name__ == "__main__":
    main()
