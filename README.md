# manzai-archive / pipeline

Python pipeline that ingests video URLs (YouTube, Bilibili) and produces draft
markdown scripts with speaker labels for the [manzai-archive web site](https://github.com/manzai-archive/web).

Maintainer: [wheatfox](https://github.com/enkerewpo)

## Run via Docker (recommended for NVIDIA GPU hosts)

Reproducible across machines, includes CUDA + cuDNN. Requires Docker +
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git clone https://github.com/manzai-archive/pipeline
git clone https://github.com/manzai-archive/web      # sibling dir
cd pipeline
cp .env.example .env                                  # add HF_TOKEN
docker compose build                                  # ~5-10 min first time
docker compose run --rm pipeline ingest <url> --group-slug nakagawake
```

Output writes to `../web/src/content/manzai/`. HF model cache persists in
`./hf-cache/` (bind-mounted). First ingest downloads ~3GB of model weights
(faster-whisper large-v3-turbo + pyannote 3.1 + community-1).

GPU defaults: faster-whisper on CUDA float16, pyannote on CUDA.
Override via env in `.env` (see `.env.example`).

## Setup

```bash
./scripts/setup.sh
source .venv/bin/activate
# edit .env, set HF_TOKEN
```

System deps: `ffmpeg`, `yt-dlp` (already on PATH).

## Usage

Single URL:

```bash
python -m pipeline ingest https://www.youtube.com/watch?v=XXXX \
    --group-slug reiwaroman --tag M-1 --tag 2022
```

Batch (queue in `submissions.yaml`):

```bash
python -m pipeline batch
```

Output goes to `../web/src/content/manzai/<slug>.md` by default
(override with `MANZAI_CONTENT_DIR`). Pipeline does **not** touch git —
review the draft in the `web` repo, edit speaker names in frontmatter,
flip `status: draft` → `reviewed`, then commit/PR there.

## Architecture

```
URL → sources/   → audio.wav (in work/, gitignored)
    → asr/       → word-level transcript (faster-whisper large-v3)
    → diarize/   → speaker turns (pyannote 3.1, num_speakers=2)
    → postprocess/format → script.md with frontmatter
```

Audio files are deleted from `work/` after a successful run unless
`--keep-audio` is passed (TODO).

## Models

- **Whisper**: `large-v3` by default. On Apple Silicon, set
  `WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8` (default) — slower but
  works without GPU. For CUDA, `WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16`.
- **Diarization**: `pyannote/speaker-diarization-3.1`. Requires accepting the
  user agreement on HuggingFace and a `HF_TOKEN`.
