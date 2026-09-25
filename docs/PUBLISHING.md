# Repository and video gallery

[← Back to TCR](../README.md)

- **Repository:** [Chatonz/vla-merge](https://github.com/Chatonz/vla-merge)
- **Video gallery:** [chatonz.github.io/vla-merge](https://chatonz.github.io/vla-merge/)

## Update the repository

```bash
git clone https://github.com/Chatonz/vla-merge.git
cd vla-merge
# Make and review your changes.
git add .
git commit -m "Update TCR documentation and demonstrations"
git push origin main
```

Authentication uses your normal GitHub credentials or SSH configuration.
The source archives and original MOV recordings are kept outside the repository;
publish the prepared source and media files rather than those archives.

## README video display

GitHub renders the animated demonstrations and six comparison panels directly
in the README. Each preview links to its corresponding MP4; GitHub's file viewer can offer playback or download depending on
the client. The [Markdown index](VIDEOS.md) links every recording individually.
Relative `<video>` embeds are not relied on for README playback.

The complete browser gallery is `index.html`. To preview it locally:

```bash
python -m http.server 8000
```

Open **http://localhost:8000**. Loading the page through HTTP allows it to fetch
`assets/media.json`; opening the HTML as a local file may block that request.

## Publish the gallery

The gallery uses GitHub Pages with **GitHub Actions** as the source. After
pushing media or gallery changes, open
[Actions → Publish video gallery](https://github.com/Chatonz/vla-merge/actions/workflows/pages.yml)
and select **Run workflow** for `main`.

The [workflow](../.github/workflows/pages.yml) publishes the static gallery and
media, and links its documentation buttons back to the repository README.
It runs manually and does not deploy automatically on pushes. No API keys or
additional runtime dependencies are required for the gallery.

For a fork, enable **Settings → Pages → Source → GitHub Actions** and update the
repository and gallery URLs in the README and this guide. `.nojekyll` keeps the
published assets unchanged.

## Media maintenance

All 168 supplied clips are included as browser-compatible MP4s. Real-robot
exports are muted, metadata-stripped, and fit within 1280 × 720 while retaining
full visual duration; original MOVs and ZIPs remain outside the repository.
The LIBERO MP4s retain their supplied encoding and timing.

The media manifest and index preserve source filenames and supplied episode
outcomes. Real-robot outcomes were not supplied and are not inferred. The README previews and comparison panels use the playback speeds disclosed
beside each figure.

The media utilities require FFmpeg (or the optional `imageio-ffmpeg` package).
Comparison panels additionally require Pillow and DejaVu Sans fonts:

```bash
python scripts/prepare_media.py --verify
python scripts/build_previews.py
python scripts/build_comparisons.py
python scripts/build_video_index.py
```

The first command rebuilds the manifest/posters and checks decoding of the
published MP4s. Re-encoding MOV originals requires `--source-root` pointing to a
local `METHOD/task1/*.MOV` and `METHOD/task2/*.MOV` directory tree; use normalized
method names such as `tcr`, `experts`, and `regmeanpp`.
