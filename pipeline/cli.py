from __future__ import annotations
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console

from pipeline import config
from pipeline.asr.transcribe import transcribe
from pipeline.diarize.speakers import diarize
from pipeline.postprocess.format import write_script
from pipeline.postprocess.lexicon import load_prompt
from pipeline.sources import fetch

console = Console()


@click.group()
def cli():
    """manzai-archive ingestion pipeline."""


def _load_members(group_slug: str, content_dir: Path) -> list[str]:
    """Read member names (in declared order) from the performer YAML.
    Returns [] if no YAML or no members."""
    try:
        py = content_dir.parent / "performers" / f"{group_slug}.yaml"
        if not py.exists():
            return []
        data = yaml.safe_load(py.read_text()) or {}
        return [
            m.get("name") for m in (data.get("members") or [])
            if m.get("name")
        ]
    except Exception:
        return []


@cli.command()
@click.argument("source")
@click.option("--group-slug", default=None, help="Slug of the performer group (e.g. nakagawake)")
@click.option("--title", default=None, help="Override title")
@click.option("--tag", "tags", multiple=True, help="Repeatable tag")
@click.option("--sensitivity", type=click.Choice(["normal", "high"]), default="normal")
@click.option(
    "--language",
    default=None,
    help="ISO code (ja, zh, en, ...). Default: auto-detect from audio.",
)
@click.option("--num-speakers", default=2, type=int)
def ingest(source, group_slug, title, tags, sensitivity, language, num_speakers):
    """Fetch URL or local file → transcribe → diarize → write draft script.md."""
    config.WORK_DIR.mkdir(exist_ok=True, parents=True)
    config.CONTENT_DIR.mkdir(exist_ok=True, parents=True)

    console.rule("[bold]1/4 fetch")
    console.print(f"  source: {source}")
    fetched = fetch(source, config.WORK_DIR)
    console.print(f"  audio:  {fetched.audio_path}")
    console.print(f"  title:  {fetched.title}")
    console.print(f"  duration: {fetched.duration_sec}s")

    # Sanity guards — see feedback memory: clip/compilation/multi-act content
    # breaks pyannote num_speakers=2 and pollutes the corpus.
    bad_keywords = ("合集", "高光", "精选", "纯享", "EP", "评委", "拜年", "采访")
    if fetched.duration_sec > 1500:
        raise click.ClickException(
            f"Refusing: {fetched.duration_sec}s > 1500s — likely a compilation."
        )
    for kw in bad_keywords:
        if kw in (fetched.title or ""):
            raise click.ClickException(
                f"Refusing: title contains '{kw}' — likely not a single act."
            )

    console.rule("[bold]2/4 transcribe")
    console.print(f"  backend: {config.ASR_BACKEND}")
    prompt = load_prompt(config.LEXICON_PATH)
    words, detected_lang, pre_turns = transcribe(
        fetched.audio_path,
        language=language,
        initial_prompt=prompt,
        group_slug=group_slug,
        content_dir=config.CONTENT_DIR,
        raw_title=fetched.title or "",
    )
    console.print(f"  {len(words)} segments, language={detected_lang}")

    console.rule("[bold]3/4 diarize")
    if pre_turns:
        turns = pre_turns
        speakers = sorted({t.speaker for t in turns})
        console.print(f"  using ASR-supplied turns: {len(turns)}, speakers={speakers}")
    else:
        # Try word-level voiceprint diarization when we know the group's
        # members. Refs are bootstrapped from self-intro lines in the
        # ASR output. Falls back to pyannote if not enough refs.
        members = _load_members(group_slug, config.CONTENT_DIR) if group_slug else []
        turns = []
        if len(members) >= 2:
            from pipeline.diarize.word_voiceprint import diarize_by_voiceprint
            turns, _word_labels = diarize_by_voiceprint(
                fetched.audio_path, words, members
            )
            speakers = sorted({t.speaker for t in turns})
            if turns:
                console.print(
                    f"  word-voiceprint: {len(turns)} turns, "
                    f"speakers={speakers}"
                )
        if not turns:
            turns = diarize(fetched.audio_path, num_speakers=num_speakers)
            speakers = sorted({t.speaker for t in turns})
            console.print(f"  pyannote: {len(turns)} turns, speakers={speakers}")

    console.rule("[bold]4/4 write")
    from pipeline.asr.transcribe import _resolve_backend
    actual_backend = _resolve_backend(detected_lang)

    # qwen-omni stashes title / form / roles in module-level vars.
    fm_form: Optional[str] = None
    fm_roles: Optional[dict] = None
    if actual_backend == "qwen-omni":
        from pipeline.asr import qwen_omni as _qo
        if title is None and _qo.LAST_TITLE:
            title = _qo.LAST_TITLE
            console.print(f"  title: {title}")
        if _qo.LAST_FORM:
            fm_form = _qo.LAST_FORM
            console.print(f"  form: {fm_form}")
        if _qo.LAST_ROLES:
            fm_roles = _qo.LAST_ROLES
            console.print(f"  roles: {fm_roles}")
    import os
    if actual_backend == "qwen-omni":
        asr_model = os.environ.get("QWEN_OMNI_MODEL") or "qwen3-omni-flash"
    elif actual_backend == "qwen":
        asr_model = os.environ.get("QWEN_MODEL") or os.environ.get("VLM_MODEL") or "qwen3-asr-flash"
    elif actual_backend == "sensevoice":
        asr_model = "FunAudioLLM/SenseVoiceSmall"
    elif actual_backend == "mlx":
        asr_model = config.MLX_WHISPER_REPO
    else:
        asr_model = config.WHISPER_MODEL
    out = write_script(
        out_dir=config.CONTENT_DIR,
        fetched=fetched,
        words=words,
        turns=turns,
        group_slug=group_slug,
        title_override=title,
        tags=list(tags),
        sensitivity=sensitivity,
        language=detected_lang,
        asr_backend=actual_backend,
        asr_model=asr_model,
        form=fm_form,
        roles=fm_roles,
    )
    console.print(f"  → {out}")
    console.rule("[green]done")


@cli.command("enroll-group")
@click.argument("group_slug")
def enroll_group_cmd(group_slug):
    """Compute voice embeddings for each member of a group with voice_samples."""
    from pathlib import Path as _P
    from pipeline.diarize.enroll import enroll_group

    web_root = config.WORKSPACE_ROOT / "web" / "src" / "content"
    performers_dir = web_root / "performers"
    embeddings_dir = web_root / "voice_embeddings"
    written = enroll_group(group_slug, performers_dir, embeddings_dir)
    for mslug, path in written.items():
        console.print(f"  → {path.name}  (member: {mslug})")
    console.print(f"[green]enrolled {len(written)} members for {group_slug}[/]")


@cli.command()
@click.option(
    "--batch-file",
    "batch_path",
    type=click.Path(exists=True),
    default="submissions.yaml",
)
@click.pass_context
def batch(ctx, batch_path):
    """Run ingest on every entry in submissions.yaml."""
    data = yaml.safe_load(Path(batch_path).read_text()) or {}
    entries = data.get("entries") or []
    if not entries:
        console.print("[yellow]No entries in submissions.yaml")
        return
    for e in entries:
        src = e.get("url") or e.get("path") or e.get("source")
        if not src:
            console.print(f"[red]skip: no url/path/source in {e}")
            continue
        group = e.get("group_slug")
        override = e.get("override", {}) or {}
        console.rule(f"[cyan]{src}")
        try:
            ctx.invoke(
                ingest,
                source=src,
                group_slug=group,
                title=override.get("title"),
                tags=tuple(override.get("tags", [])),
                sensitivity=override.get("sensitivity", "normal"),
                language=override.get("language"),
                num_speakers=override.get("num_speakers", 2),
            )
        except Exception as exc:
            console.print(f"[red]FAILED: {exc}")
