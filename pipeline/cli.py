from __future__ import annotations
from pathlib import Path

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


@cli.command()
@click.argument("url")
@click.option("--group-slug", default=None, help="Slug of the performer group (e.g. reiwaroman)")
@click.option("--title", default=None, help="Override title")
@click.option("--tag", "tags", multiple=True, help="Repeatable tag")
@click.option("--sensitivity", type=click.Choice(["normal", "high"]), default="normal")
@click.option("--language", default="ja")
@click.option("--num-speakers", default=2, type=int)
def ingest(url, group_slug, title, tags, sensitivity, language, num_speakers):
    """Fetch URL → transcribe → diarize → write draft script.md."""
    config.WORK_DIR.mkdir(exist_ok=True, parents=True)
    config.CONTENT_DIR.mkdir(exist_ok=True, parents=True)

    console.rule("[bold]1/4 fetch")
    fetched = fetch(url, config.WORK_DIR)
    console.print(f"  audio: {fetched.audio_path}")
    console.print(f"  title: {fetched.title}")

    console.rule("[bold]2/4 transcribe")
    prompt = load_prompt(config.LEXICON_PATH)
    words = transcribe(fetched.audio_path, language=language, initial_prompt=prompt)
    console.print(f"  {len(words)} words")

    console.rule("[bold]3/4 diarize")
    turns = diarize(fetched.audio_path, num_speakers=num_speakers)
    speakers = sorted({t.speaker for t in turns})
    console.print(f"  {len(turns)} turns, speakers={speakers}")

    console.rule("[bold]4/4 write")
    out = write_script(
        out_dir=config.CONTENT_DIR,
        fetched=fetched,
        words=words,
        turns=turns,
        group_slug=group_slug,
        title_override=title,
        tags=list(tags),
        sensitivity=sensitivity,
        language=language,
    )
    console.print(f"  → {out}")
    console.rule("[green]done")


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
        url = e["url"]
        group = e.get("group_slug")
        override = e.get("override", {}) or {}
        console.rule(f"[cyan]{url}")
        ctx.invoke(
            ingest,
            url=url,
            group_slug=group,
            title=override.get("title"),
            tags=tuple(override.get("tags", [])),
            sensitivity=override.get("sensitivity", "normal"),
            language=override.get("language", "ja"),
            num_speakers=override.get("num_speakers", 2),
        )
