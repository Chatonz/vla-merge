#!/usr/bin/env python3
"""Build the six reviewer comparison panels from the published source videos.

Requires Python 3, Pillow, and FFmpeg (or imageio-ffmpeg). Source selection,
speed, complete clip durations, and output details are recorded in
assets/comparisons/manifest.json. No tile loops independently: an ended clip
holds its actual final frame with a visible CLIP ENDED marker.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/comparisons"
METHODS = [
    ("soup", "Soup"),
    ("ties", "TIES"),
    ("regmeanpp", "RegMean++"),
    ("featcal", "FeatCal"),
    ("tcr", "TCR"),
    ("experts", "Expert"),
]
PANELS = [
    ("real-task1", "Real robot / Task 1", "real_robot", "task1", None),
    ("real-task2", "Real robot / Task 2", "real_robot", "task2", None),
    ("libero-long", "LIBERO-Long", "libero", "libero_10", "task00_r01_ep00"),
    ("libero-object", "LIBERO-Object", "libero", "libero_object", "task00_r01_ep00"),
    ("libero-goal", "LIBERO-Goal", "libero", "libero_goal", "task00_r01_ep01"),
    ("libero-spatial", "LIBERO-Spatial", "libero", "libero_spatial", "task00_r01_ep00"),
]
FPS = 24
GIF_MAX_BYTES = 8_000_000
BG = "#101923"
HEADER = "#213041"
WHITE = "#f4f7fb"
MUTED = "#b9c8d7"
TEAL = "#54dac4"
REFERENCE_HEADER = "#463d2d"
REFERENCE_ACCENT = "#f1d28d"


def resolve_ffmpeg(explicit):
    executable = explicit or shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit("Install FFmpeg or imageio-ffmpeg, or pass --ffmpeg.") from exc


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for candidate in [Path("/usr/share/fonts/truetype/dejavu") / filename, Path(filename)]:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    raise SystemExit("Install the DejaVu Sans fonts to reproduce readable panel labels.")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_check(ffmpeg, path):
    """Decode every video frame, returning the exact decoded frame count."""
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-nostdin", "-i", str(path),
         "-map", "0:v:0", "-an", "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    counts = re.findall(r"^frame=(\d+)\s*$", result.stdout, re.MULTILINE)
    if not counts or int(counts[-1]) <= 0:
        raise RuntimeError(f"No decoded frames: {path}")
    return int(counts[-1])


class Clip:
    """A sequential decoder; sample the same shared clock for every method."""

    def __init__(self, ffmpeg, metadata, size):
        self.metadata = metadata
        self.size = size
        self.index = -1
        self.frame = None
        width, height = size
        self.frame_bytes = width * height * 3
        filters = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
        self.process = subprocess.Popen(
            [ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-i",
             str(ROOT / metadata["source_path"]), "-map", "0:v:0", "-an",
             "-vf", filters, "-fps_mode", "passthrough", "-threads", "1",
             "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
            stdout=subprocess.PIPE,
        )

    def at(self, seconds, speed):
        count = self.metadata["source_frame_count"]
        desired = min(int(seconds * speed * self.metadata["source_fps"] + 1e-9), count - 1)
        while self.index < desired:
            data = self.process.stdout.read(self.frame_bytes)
            if len(data) != self.frame_bytes:
                raise RuntimeError(f"Unexpected decoder EOF: {self.metadata['source_path']}")
            self.frame = Image.frombytes("RGB", self.size, data)
            self.index += 1
        return self.frame

    def close(self):
        if self.process.stdout:
            self.process.stdout.close()
        # All source frames should have been consumed when the shared clock ends.
        result = self.process.wait()
        if result:
            raise RuntimeError(f"Decoder failed: {self.metadata['source_path']}")


def select_sources(ffmpeg, media, domain, task, episode, speed):
    sources = []
    for method, label in METHODS:
        matches = sorted(
            (item for item in media if item["domain"] == domain and item["task"] == task
             and item["method"] == method
             and (episode is None or Path(item["path"]).name.startswith(episode + "_"))),
            key=lambda item: item["path"],
        )
        if not matches or (episode is not None and len(matches) != 1):
            raise RuntimeError(f"Unexpected source selection: {domain}/{method}/{task}/{episode}")
        source = matches[0]
        path = ROOT / source["path"]
        count = decode_check(ffmpeg, path)
        duration = count / source["fps"]
        outcome = None
        if domain == "libero":
            outcome = path.stem.rsplit("_", 1)[-1]
            if outcome not in {"success", "failure"} or outcome != source["outcome"]:
                raise RuntimeError(f"Unrecognized or inconsistent original outcome: {path}")
        sources.append({
            "method": method,
            "label": label,
            "role": "reference_policy" if method == "experts" else "merged_policy",
            "role_label": ("REFERENCE POLICY" if method == "experts" else
                           "OURS / MERGED POLICY" if method == "tcr" else "MERGED POLICY"),
            "source_path": source["path"],
            "source_sha256": sha256(path),
            "source_fps": source["fps"],
            "source_frame_count": count,
            "source_duration_seconds": duration,
            "playback_duration_seconds": duration / speed,
            "source_width": source["width"],
            "source_height": source["height"],
            "original_outcome": outcome,
            "row": len(sources) // 3,
            "column": len(sources) % 3,
        })
    return sources


def layout(title, sources, domain, speed):
    tile_width, tile_height = (320, 180) if domain == "real_robot" else (240, 240)
    # Give the method, policy role, and original outcome their own lines.
    # This also keeps the narrow LIBERO tiles legible in the inline GIFs.
    gutter, top, footer = 12, 60, 38
    label_height = 56 if domain == "real_robot" else 74
    width = tile_width * 3 + gutter * 4
    height = top + (tile_height + label_height) * 2 + gutter * 3 + footer
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 12), title, font=font(23, True), fill=WHITE)
    speed_text = f"{speed:g}x MP4 playback"
    draw.text((width - gutter, 18), speed_text, font=font(15), fill=MUTED, anchor="ra")
    positions = []
    for item in sources:
        x = gutter + item["column"] * (tile_width + gutter)
        y = top + item["row"] * (tile_height + label_height + gutter)
        is_reference = item["role"] == "reference_policy"
        header_color = REFERENCE_HEADER if is_reference else HEADER
        draw.rectangle((x, y, x + tile_width - 1, y + label_height - 1), fill=header_color)
        if is_reference:
            # Draw in the gutter, preserving every source pixel in the scene.
            draw.rectangle((x - 3, y - 3, x + tile_width + 2,
                            y + label_height + tile_height + 2),
                           outline=REFERENCE_ACCENT, width=2)
        name_color = TEAL if item["method"] == "tcr" else WHITE
        draw.text((x + 8, y + 8), item["label"], font=font(18, True), fill=name_color)
        role_color = REFERENCE_ACCENT if is_reference else MUTED
        draw.text((x + 8, y + 33), item["role_label"], font=font(11, True), fill=role_color)
        if item["original_outcome"]:
            outcome = item["original_outcome"].upper()
            color = "#8ce5b1" if outcome == "SUCCESS" else "#ffb3ae"
            draw.text((x + 8, y + 52), outcome, font=font(11, True), fill=color)
        positions.append((x, y + label_height))
    clock_y = height - (38 if domain == "libero" else 31)
    draw.text((gutter, clock_y), "Shared clock | Shorter clips freeze at end; no individual replay",
              font=font(14), fill=MUTED)
    if domain == "libero":
        draw.text((width - gutter, height - 20), "SUCCESS / FAILURE: original clip labels",
                  font=font(11), fill=MUTED, anchor="ra")
    return canvas, positions, (tile_width, tile_height)


def build_gif(ffmpeg, mp4, target):
    # Keep the full sequence at the chosen playback speed. Only the display
    # frame rate / palette may be reduced to meet GitHub's inline image budget.
    for fps, colors in [(8, 128), (8, 96), (6, 96), (6, 64), (5, 64), (4, 64)]:
        graph = (
            f"fps={fps}:round=up,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
        )
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-nostdin", "-i", str(mp4),
             "-filter_complex", graph, "-an", "-map_metadata", "-1", "-map_chapters", "-1",
             "-loop", "0", "-threads", "2", str(target)], check=True,
        )
        if target.stat().st_size < GIF_MAX_BYTES:
            return fps, colors
    raise RuntimeError(f"GIF exceeds {GIF_MAX_BYTES} bytes: {target}")


def build_panel(ffmpeg, media, specification):
    name, title, domain, task, episode = specification
    speed = 2.0 if domain == "real_robot" else 0.25
    sources = select_sources(ffmpeg, media, domain, task, episode, speed)
    base, positions, tile_size = layout(title, sources, domain, speed)
    max_duration = max(item["playback_duration_seconds"] for item in sources)
    frame_count = math.ceil(max_duration * FPS - 1e-9)
    targets = {kind: OUTPUT / f"{name}.{kind}" for kind in ["mp4", "gif", "jpg"]}
    encoder = subprocess.Popen(
        [ffmpeg, "-v", "error", "-y", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{base.width}x{base.height}", "-r", str(FPS), "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-threads", "2", "-map_metadata", "-1",
         "-map_chapters", "-1", "-movflags", "+faststart", str(targets["mp4"])],
        stdin=subprocess.PIPE,
    )
    clips = [Clip(ffmpeg, source, tile_size) for source in sources]
    try:
        for number in range(frame_count):
            timestamp = number / FPS
            canvas = base.copy()
            for clip, source, position in zip(clips, sources, positions):
                canvas.paste(clip.at(timestamp, speed), position)
                if timestamp + 1e-9 >= source["playback_duration_seconds"]:
                    x, y = position
                    draw = ImageDraw.Draw(canvas)
                    draw.rounded_rectangle((x + 8, y + 8, x + 113, y + 31), radius=4, fill=BG)
                    draw.text((x + 14, y + 12), "CLIP ENDED", font=font(13, True), fill=WHITE)
            if number == 0:
                canvas.save(targets["jpg"], quality=93, optimize=True, subsampling=0)
            encoder.stdin.write(canvas.tobytes())
        # Consume the last source frame even if the final resampling instant
        # occurs just before it, then ensure each decoder exits successfully.
        for clip in clips:
            clip.at(max_duration, speed)
            clip.close()
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError(f"Encoder failed: {name}")
    finally:
        for clip in clips:
            if clip.process.poll() is None:
                clip.process.kill()
                clip.process.wait()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    gif_fps, gif_colors = build_gif(ffmpeg, targets["mp4"], targets["gif"])
    decoded_mp4_frames = decode_check(ffmpeg, targets["mp4"])
    decoded_gif_frames = decode_check(ffmpeg, targets["gif"])
    with Image.open(targets["gif"]) as animation:
        gif_duration_ms = 0
        for index in range(animation.n_frames):
            animation.seek(index)
            gif_duration_ms += animation.info.get("duration", 0)
    if gif_duration_ms / 1000 + 0.01 < max_duration:
        raise RuntimeError(f"GIF does not cover the full shared timeline: {name}")
    if decoded_mp4_frames != frame_count:
        raise RuntimeError(f"Wrong output frame count: {name}")
    with Image.open(targets["jpg"]) as poster:
        poster.verify()
    result = {
        "id": name, "title": title, "domain": domain, "task": task,
        "selection_rule": (
            "Lexicographically first published filename per method and task."
            if domain == "real_robot" else
            f"Same published task/repetition/episode identifier for every method: {episode}."
        ),
        "episode_id": episode,
        "playback_speed_relative_to_source_mp4": speed,
        "shared_timeline_duration_seconds": max_duration,
        "output_duration_seconds": frame_count / FPS,
        "output_duration_rounding": "Round up to the next 24 fps frame boundary.",
        "short_clip_behavior": "Hold the actual final source frame; overlay CLIP ENDED.",
        "outcome_labels": "Original filename and media-index labels only; real-robot clips have no outcome labels.",
        "layout": {"columns": 3, "rows": 2, "width": base.width, "height": base.height,
                   "scene_tile_width": tile_size[0], "scene_tile_height": tile_size[1],
                   "header_height": 56 if domain == "real_robot" else 74,
                   "reference_style": "Expert is last, labeled REFERENCE POLICY, with an amber header and an outline outside the scene.",
                   "scene_fit": "Preserve full scene and aspect ratio; letterbox if necessary."},
        "sources": sources,
        "outputs": {
            kind: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                   "sha256": sha256(path)}
            for kind, path in targets.items()
        },
        "verification": {"all_outputs_decode": True, "mp4_decoded_frames": decoded_mp4_frames,
                         "gif_decoded_frames": decoded_gif_frames, "jpg_verified": True},
    }
    result["outputs"]["mp4"].update({"codec": "h264", "fps": FPS, "crf": 18, "has_audio": False})
    result["outputs"]["gif"].update({"fps": gif_fps, "palette_colors": gif_colors,
                                     "duration_seconds": gif_duration_ms / 1000,
                                     "duration_rounding": "Round up to GIF frame boundary; GIF frame delays use 10 ms units.",
                                     "loop": "whole panel only", "target_max_bytes": GIF_MAX_BYTES})
    print(f"{name}: {frame_count / FPS:.3f}s; GIF {targets['gif'].stat().st_size:,} bytes", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", help="Path to FFmpeg; otherwise locate it automatically.")
    parser.add_argument("--panels", nargs="+", choices=[panel[0] for panel in PANELS],
                        help="Rebuild only selected panels, retaining other existing manifest entries.")
    args = parser.parse_args()
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    media = json.loads((ROOT / "assets/media.json").read_text())["videos"]
    manifest_path = OUTPUT / "manifest.json"
    old_panels = {}
    if args.panels and manifest_path.exists():
        old_panels = {item["id"]: item for item in json.loads(manifest_path.read_text())["panels"]}
    panels = []
    for specification in PANELS:
        if not args.panels or specification[0] in args.panels:
            panels.append(build_panel(ffmpeg, media, specification))
        elif specification[0] in old_panels:
            panels.append(old_panels[specification[0]])
        # Save after each completed panel so an interrupted rebuild remains reviewable.
        manifest = {
            "version": 2,
            "generator": "scripts/build_comparisons.py",
            "rebuild_command": "python scripts/build_comparisons.py",
            "dependencies": ["Python 3", "Pillow", "FFmpeg with libx264", "DejaVu Sans fonts"],
            "method_order": [method for method, _ in METHODS],
            "policy_roles": {"merged_policy": "Soup, TIES, RegMean++, FeatCal, and TCR are the merging methods.",
                             "reference_policy": "Expert is a separate expert-policy reference, displayed last."},
            "source_duration_basis": "Exact decoded frame count divided by the published source FPS (constant-frame-rate sources).",
            "timeline_sampling": "Every tile samples the same output time; source frame = floor(time * speed * source FPS), clamped to final frame.",
            "metadata_policy": "No source metadata, chapters, audio, or JPEG EXIF copied into outputs.",
            "panels": panels,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
