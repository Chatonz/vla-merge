#!/usr/bin/env python3
"""Prepare the supplied TCR videos and a portable gallery manifest.

Real-robot source layout: SOURCE/{method}/{task1,task2}/*.MOV
LIBERO input layout: REPO/assets/videos/libero/{method}/{suite}/*.mp4

Requires FFmpeg on PATH, or ``pip install imageio-ffmpeg``. Real-robot
exports retain the entire clip at its original speed and orientation,
remove audio and identifying file metadata, and use browser-compatible H.264.
LIBERO MP4 files are retained byte-for-byte, including their original FPS.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


METHOD_ORDER = {
    name: i for i, name in enumerate(
        ("tcr", "experts", "featcal", "regmeanpp", "regmean", "soup",
         "ties", "knots_ties", "task_arithmetic", "wudi")
    )
}
TITLES = {
    "task1": "Task 1", "task2": "Task 2",
    "libero_spatial": "LIBERO Spatial", "libero_object": "LIBERO Object",
    "libero_goal": "LIBERO Goal", "libero_10": "LIBERO Long",
}


def find_ffmpeg(explicit: str | None) -> str:
    if explicit:
        return explicit
    if executable := shutil.which("ffmpeg"):
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install FFmpeg or run: pip install imageio-ffmpeg") from exc


def run(ffmpeg: str, arguments: list[str]) -> None:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *arguments],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"FFmpeg exited {result.returncode}")


def probe(ffmpeg: str, path: Path) -> dict:
    # imageio-ffmpeg bundles FFmpeg, but not ffprobe. Use FFmpeg's standard
    # stream header so preparation also works with that small dependency.
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    video_line = next((line for line in result.stderr.splitlines() if "Video:" in line), "")
    resolution = re.search(r"\b(\d{2,})x(\d{2,})\b", video_line)
    fps = re.search(r"([\d.]+) fps", video_line)
    codec = re.search(r"Video: ([\w]+)", video_line)
    if not duration or not resolution or not codec:
        raise RuntimeError(f"Cannot read video metadata: {path}")
    h, m, s = duration.groups()
    return {
        "duration": round(int(h) * 3600 + int(m) * 60 + float(s), 3),
        "width": int(resolution[1]), "height": int(resolution[2]),
        "fps": float(fps[1]) if fps else None,
        "codec": codec[1], "has_audio": "Audio:" in result.stderr,
    }


def temporary_output(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial{path.suffix}")


def transcode(ffmpeg: str, source: Path, target: Path, crf: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_output(target)
    scale = "scale=w='trunc(iw*min(1,min(1280/iw,720/ih))/2)*2':h='trunc(ih*min(1,min(1280/iw,720/ih))/2)*2'"
    try:
        run(ffmpeg, [
            "-y", "-threads", "2", "-i", str(source), "-map", "0:v:0",
            "-an", "-sn", "-dn", "-vf", f"{scale},fps=30,setsar=1",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-threads", "2", "-filter_threads", "1",
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-movflags", "+faststart", str(temporary),
        ])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def make_poster(ffmpeg: str, source: Path, target: Path, duration: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_output(target)
    try:
        run(ffmpeg, [
            "-y", "-ss", str(round(duration / 3, 3)), "-threads", "2",
            "-i", str(source), "-map", "0:v:0", "-frames:v", "1",
            "-vf", "scale='min(480,iw)':-2", "-q:v", "3",
            "-threads", "2", "-filter_threads", "1", "-update", "1",
            "-map_metadata", "-1", str(temporary),
        ])
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_one(job: tuple, args: argparse.Namespace, ffmpeg: str) -> dict:
    domain, method, task, source, target = job
    if domain == "real_robot" and (args.force or not target.exists()):
        transcode(ffmpeg, source, target, args.crf)
    metadata = probe(ffmpeg, target)
    if metadata["codec"] != "h264":
        raise RuntimeError(f"Expected browser-compatible H.264, got {metadata['codec']}: {target}")
    if args.verify:
        run(ffmpeg, ["-xerror", "-threads", "2", "-i", str(target),
                     "-map", "0:v:0", "-an", "-f", "null", "-"])
    poster = args.repo / "assets" / "posters" / domain / method / task / f"{target.stem}.jpg"
    if args.force or not poster.exists() or poster.stat().st_mtime < target.stat().st_mtime:
        make_poster(ffmpeg, target, poster, metadata["duration"])
    outcome_match = re.search(r"_(success|failure)$", target.stem)
    return {
        "id": "-".join((domain, method, task, target.stem)),
        "domain": domain, "method": method, "task": task,
        "title": TITLES.get(task, task.replace("_", " ").title()),
        "path": target.relative_to(args.repo).as_posix(),
        "poster": poster.relative_to(args.repo).as_posix(),
        **metadata,
        "outcome": outcome_match[1] if outcome_match else None,
        "original_filename": source.name,
        "bytes": target.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path,
                        help="Extracted real-robot originals; omit to rebuild from existing MP4 exports")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ffmpeg", help="FFmpeg executable; otherwise use PATH or imageio-ffmpeg")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--crf", type=int, default=26, choices=range(18, 36))
    parser.add_argument("--force", action="store_true", help="Recreate all real exports and posters")
    parser.add_argument("--verify", action="store_true", help="Decode every video completely and fail on errors")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    ffmpeg = find_ffmpeg(args.ffmpeg)
    jobs = []
    if args.source_root:
        source_root = args.source_root.resolve()
        for source in sorted(source_root.glob("*/*/*")):
            if source.is_file() and source.suffix.lower() in {".mov", ".mp4"}:
                method, task, _ = source.relative_to(source_root).parts
                target = args.repo / "assets" / "videos" / "real_robot" / method / task / f"{source.stem.lower()}.mp4"
                jobs.append(("real_robot", method, task, source, target))
        if not jobs:
            parser.error(f"No real-robot clips found in {source_root}/<method>/<task>/")
    else:
        real_root = args.repo / "assets" / "videos" / "real_robot"
        for source in sorted(real_root.glob("*/*/*.mp4")):
            method, task, _ = source.relative_to(real_root).parts
            jobs.append(("real_robot", method, task, source, source))
        # Without originals, --force may refresh posters but must not transcode
        # an exported file onto itself. Mark these jobs as already prepared.
        if args.force and jobs:
            parser.error("--force requires --source-root when real-robot exports exist")
    libero_root = args.repo / "assets" / "videos" / "libero"
    for source in sorted(libero_root.glob("*/*/*.mp4")):
        method, task, _ = source.relative_to(libero_root).parts
        jobs.append(("libero", method, task, source, source))
    if not jobs:
        parser.error("No video files found")
    # Recover original filenames when regenerating from the committed exports.
    manifest_path = args.repo / "assets" / "media.json"
    previous = {}
    if manifest_path.exists():
        previous = {v["id"]: v.get("original_filename")
                    for v in json.loads(manifest_path.read_text())["videos"]}
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_one, job, args, ffmpeg): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                entry = future.result()
                if not args.source_root and previous.get(entry["id"]):
                    entry["original_filename"] = previous[entry["id"]]
                results.append(entry)
                print(f"[{len(results)}/{len(jobs)}] {entry['path']} "
                      f"({entry['duration']:.2f}s, {entry['width']}x{entry['height']})", flush=True)
            except Exception as exc:
                failures.append(f"{job[3]}: {exc}")
                print(f"ERROR: {failures[-1]}", file=sys.stderr, flush=True)
    if failures:
        print(f"Failed to prepare {len(failures)} videos; existing manifest was not replaced.", file=sys.stderr)
        return 1
    results.sort(key=lambda v: (v["domain"] != "real_robot",
                               METHOD_ORDER.get(v["method"], 99), v["task"], v["path"]))
    payload = {"version": 1, "videos": results}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    total_bytes = sum(v["bytes"] for v in results)
    by_domain = {d: sum(v["domain"] == d for v in results) for d in ("real_robot", "libero")}
    print(json.dumps({"videos": len(results), "by_domain": by_domain,
                      "total_video_mb": round(total_bytes / 1_000_000, 2),
                      "largest_video_mb": round(max(v["bytes"] for v in results) / 1_000_000, 2),
                      "full_decode_verified": args.verify,
                      "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
