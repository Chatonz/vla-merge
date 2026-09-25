#!/usr/bin/env python3
"""Rebuild TCR demonstration and failure-case GIFs from the published MP4s."""

import argparse
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PREVIEWS = [
    ("real-task1", "real_robot/tcr/task1/img_5011.mp4", 0.5, 448),
    ("real-task2", "real_robot/tcr/task2/img_5021.mp4", 0.5, 448),
    ("libero-spatial", "libero/tcr/libero_spatial/task00_r01_ep00_success.mp4", 4, 280),
    ("libero-object", "libero/tcr/libero_object/task00_r01_ep00_success.mp4", 4, 280),
    ("libero-goal", "libero/tcr/libero_goal/task00_r01_ep01_success.mp4", 4, 280),
    ("libero-long", "libero/tcr/libero_10/task00_r01_ep00_success.mp4", 4, 280),
    ("real-task1-trial2", "real_robot/tcr/task1/img_5012.mp4", 0.5, 320),
    ("real-task1-trial3", "real_robot/tcr/task1/img_5013.mp4", 0.5, 320),
    ("real-task1-trial4", "real_robot/tcr/task1/img_5014.mp4", 0.5, 320),
    ("real-task2-trial2", "real_robot/tcr/task2/img_5022.mp4", 0.5, 320),
    ("real-task2-trial3", "real_robot/tcr/task2/img_5026.mp4", 0.5, 320),
    ("real-task2-trial4", "real_robot/tcr/task2/img_5028.mp4", 0.5, 320),
    ("libero-object-failure", "libero/tcr/libero_object/task00_r01_ep01_failure.mp4", 4, 280),
    ("libero-goal-failure", "libero/tcr/libero_goal/task00_r01_ep00_failure.mp4", 4, 280),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", help="Path to an FFmpeg binary")
    parser.add_argument("--only", nargs="+", help="Build only the named preview IDs")
    args = parser.parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            parser.error("Install FFmpeg or imageio-ffmpeg, or pass --ffmpeg.")
    output = ROOT / "assets/previews"
    output.mkdir(parents=True, exist_ok=True)
    for name, relative, multiplier, width in PREVIEWS:
        if args.only and name not in args.only:
            continue
        source = ROOT / "assets/videos" / relative
        filters = (
            f"[0:v:0]setpts={multiplier}*(PTS-STARTPTS),fps=10,"
            f"scale={width}:-2:flags=lanczos,split[a][b];"
            "[a]palettegen=max_colors=128:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3"
        )
        target = output / f"{name}.gif"
        subprocess.run(
            [
                ffmpeg, "-v", "error", "-y", "-i", str(source),
                "-filter_complex", filters, "-an", "-map_metadata", "-1",
                "-loop", "0", "-threads", "2", str(target),
            ],
            check=True,
        )
        print(f"{target.relative_to(ROOT)}: {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
