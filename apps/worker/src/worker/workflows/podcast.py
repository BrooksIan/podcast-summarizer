"""Podcast summarizer workflow.

Checks a podcast RSS feed for new episodes, transcribes the episode audio
with Mistral's Voxtral speech-to-text API, and produces a structured summary
with a Mistral chat model. Each scheduled run only processes episodes
published within the last `since_hours` hours, so a daily schedule picks up
exactly the episodes that came out since the previous run.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from pydantic import BaseModel, Field

import mistralai.workflows as workflows

TRANSCRIPTION_MODEL = "voxtral-mini-transcribe-3"
SUMMARY_MODEL = "mistral-small-latest"

TRANSCRIPT_CHUNK_CHARS = 24_000

_FETCH_TIMEOUT = timedelta(minutes=2)
_TRANSCRIBE_TIMEOUT = timedelta(minutes=30)
_SUMMARIZE_TIMEOUT = timedelta(minutes=10)


class Episode(BaseModel):
    guid: str
    title: str
    audio_url: str | None = None
    published_at: datetime | None = None


class PodcastSummarizerInput(BaseModel):
    feed_url: str = Field(description="URL of the podcast RSS feed to watch.")
    since_hours: int = Field(
        default=25,
        description="Only episodes published within this many hours are processed.",
    )
    max_episodes: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of new episodes processed per run.",
    )
    episode_guid: str | None = Field(
        default=None,
        description="Process this specific episode GUID instead of the newest ones.",
    )


class EpisodeResult(BaseModel):
    guid: str
    title: str
    audio_url: str | None = None
    published_at: datetime | None = None
    transcript: str | None = None
    summary: str | None = None
    error: str | None = None


class PodcastSummarizerOutput(BaseModel):
    feed_url: str
    episodes_in_feed: int
    episodes_processed: int
    results: list[EpisodeResult] = Field(default_factory=list)
    message: str | None = None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if _local_name(child.tag) == name and child.text:
            return child.text.strip()
    return None


def _audio_url(item: ET.Element) -> str | None:
    for child in item.iter():
        local = _local_name(child.tag)
        if local == "enclosure":
            url = child.attrib.get("url")
            mime = child.attrib.get("type", "")
            if url and (mime.startswith("audio/") or "audio" in url):
                return url
        if local == "content" and child.attrib.get("url", "").startswith("audio"):
            return child.attrib["url"]
        if local == "link" and child.attrib.get("rel") == "enclosure":
            return child.attrib.get("href")
    return None


def _parse_published(item: ET.Element) -> datetime | None:
    for name in ("pubDate", "published", "updated"):
        raw = _child_text(item, name)
        if not raw:
            continue
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def parse_feed_episodes(xml_text: str) -> list[Episode]:
    root = ET.fromstring(xml_text)
    episodes: list[Episode] = []
    for item in root.iter():
        if _local_name(item.tag) not in {"item", "entry"}:
            continue
        guid = _child_text(item, "guid") or _child_text(item, "id")
        title = _child_text(item, "title")
        if guid is None:
            link = _child_text(item, "link")
            if link is None:
                for child in item.iter():
                    if _local_name(child.tag) == "link" and child.attrib.get("href"):
                        link = child.attrib["href"]
                        break
            guid = link
        if guid is None:
            continue
        episodes.append(
            Episode(
                guid=guid,
                title=title or guid,
                audio_url=_audio_url(item),
                published_at=_parse_published(item),
            )
        )
    return episodes


def select_episodes(
    episodes: list[Episode],
    now: datetime,
    since_hours: int,
    max_episodes: int,
    episode_guid: str | None,
) -> list[Episode]:
    if episode_guid is not None:
        return [episode for episode in episodes if episode.guid == episode_guid]
    cutoff = now - timedelta(hours=since_hours)
    recent = [
        episode
        for episode in episodes
        if episode.published_at is not None and episode.published_at > cutoff
    ]
    recent.sort(key=lambda episode: episode.published_at, reverse=True)
    return recent[:max_episodes]


def chunk_text(text: str, max_chars: int = TRANSCRIPT_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]


async def _fetch_feed(feed_url: str) -> list[Episode]:
    import httpx

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        response = await client.get(feed_url)
        response.raise_for_status()
    return parse_feed_episodes(response.text)


async def _transcribe_audio(audio_url: str) -> str:
    from utils.mistral import mistral_client

    client = mistral_client()
    transcription = await client.audio.transcriptions.complete_async(
        model=TRANSCRIPTION_MODEL,
        file_url=audio_url,
    )
    return transcription.text


_SUMMARY_PROMPT = """You are a podcast summarizer. You receive the full transcript of one \
podcast episode, including the episode title.

Summarize it using exactly this structure:

## Episode Summary
2-3 sentence overview of what this episode is about and why it matters.

## Key Points
- 5-8 bullet points covering the main arguments, stories, or findings, in the \
order they appear in the episode.

## Notable Quotes
- 1-3 short verbatim quotes (with speaker name if identifiable) that capture \
the episode's core ideas.

## Guests & Mentions
- People, companies, books, or tools mentioned, if any. Omit this section if none.

Rules:
- Use only information present in the transcript. Never invent names, numbers, or claims.
- If the transcript is partial or garbled, note that in one line at the end.
- Match the language of the transcript.
"""

_PARTIAL_PROMPT = """The following text is part {position} of {total} of a podcast episode \
transcript titled "{title}". Summarize the main points of this part in 3-5 bullet points. \
Use only information present in the text.

Transcript part:
{chunk}
"""

_FINAL_PROMPT = """You are a podcast summarizer. Below are section summaries of one podcast \
episode titled "{title}", in the order the sections appear in the episode. Combine them into \
one final summary using exactly this structure:

## Episode Summary
2-3 sentence overview of what this episode is about and why it matters.

## Key Points
- 5-8 bullet points covering the main arguments, stories, or findings, in the \
order they appear in the episode.

## Notable Quotes
- 1-3 short verbatim quotes from the episode, if any appear in the section summaries.

## Guests & Mentions
- People, companies, books, or tools mentioned, if any. Omit this section if none.

Rules:
- Use only information present in the section summaries. Never invent names, numbers, or claims.
- Match the language of the episode.

Section summaries:
{sections}
"""


async def _summarize_text(title: str, transcript: str) -> str:
    from utils.mistral import mistral_client

    client = mistral_client()
    chunks = chunk_text(transcript)
    if len(chunks) == 1:
        response = await client.chat.complete_async(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "user", "content": f"{_SUMMARY_PROMPT}\n\nEpisode title: {title}\n\nTranscript:\n{transcript}"}
            ],
        )
        return response.choices[0].message.content or ""

    sections: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        response = await client.chat.complete_async(
            model=SUMMARY_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _PARTIAL_PROMPT.format(position=position, total=len(chunks), title=title, chunk=chunk),
                }
            ],
        )
        sections.append(response.choices[0].message.content or "")
    final = await client.chat.complete_async(
        model=SUMMARY_MODEL,
        messages=[
            {
                "role": "user",
                "content": _FINAL_PROMPT.format(title=title, sections="\n\n".join(sections)),
            }
        ],
    )
    return final.choices[0].message.content or ""


@workflows.activity(start_to_close_timeout=_FETCH_TIMEOUT, retry_policy_max_attempts=3)
async def fetch_feed_episodes(feed_url: str) -> list[Episode]:
    return await _fetch_feed(feed_url)


@workflows.activity(start_to_close_timeout=_FETCH_TIMEOUT)
async def select_new_episodes(
    episodes: list[Episode],
    since_hours: int,
    max_episodes: int,
    episode_guid: str | None,
) -> list[Episode]:
    return select_episodes(
        episodes,
        now=datetime.now(timezone.utc),
        since_hours=since_hours,
        max_episodes=max_episodes,
        episode_guid=episode_guid,
    )


@workflows.activity(
    start_to_close_timeout=_TRANSCRIBE_TIMEOUT,
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
)
async def transcribe_episode(audio_url: str) -> str:
    return await _transcribe_audio(audio_url)


@workflows.activity(
    start_to_close_timeout=_SUMMARIZE_TIMEOUT,
    retry_policy_max_attempts=3,
    retry_policy_backoff_coefficient=2.0,
)
async def summarize_transcript(title: str, transcript: str) -> str:
    return await _summarize_text(title, transcript)


@workflows.workflow.define(
    name="podcast-summarizer",
    workflow_display_name="Podcast Summarizer",
    workflow_description=(
        "Checks a podcast RSS feed for new episodes, transcribes the episode audio, "
        "and returns the transcript plus a structured summary."
    ),
    execution_timeout=timedelta(hours=4),
)
class PodcastSummarizerWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: PodcastSummarizerInput) -> PodcastSummarizerOutput:
        episodes = await fetch_feed_episodes(feed_url=input.feed_url)

        if input.episode_guid is None:
            selected = await select_new_episodes(
                episodes=episodes,
                since_hours=input.since_hours,
                max_episodes=input.max_episodes,
                episode_guid=None,
            )
        else:
            selected = [episode for episode in episodes if episode.guid == input.episode_guid]

        if not selected:
            return PodcastSummarizerOutput(
                feed_url=input.feed_url,
                episodes_in_feed=len(episodes),
                episodes_processed=0,
                results=[],
                message="No new episodes found in the configured time window.",
            )

        results: list[EpisodeResult] = []
        for episode in selected:
            if episode.audio_url is None:
                results.append(
                    EpisodeResult(
                        guid=episode.guid,
                        title=episode.title,
                        published_at=episode.published_at,
                        error="Episode has no audio enclosure in the feed.",
                    )
                )
                continue
            transcript = await transcribe_episode(audio_url=episode.audio_url)
            summary = await summarize_transcript(title=episode.title, transcript=transcript)
            results.append(
                EpisodeResult(
                    guid=episode.guid,
                    title=episode.title,
                    audio_url=episode.audio_url,
                    published_at=episode.published_at,
                    transcript=transcript,
                    summary=summary,
                )
            )

        processed = sum(1 for result in results if result.error is None)
        return PodcastSummarizerOutput(
            feed_url=input.feed_url,
            episodes_in_feed=len(episodes),
            episodes_processed=processed,
            results=results,
        )
