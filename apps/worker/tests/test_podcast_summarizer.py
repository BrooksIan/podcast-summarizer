from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from worker.workflows.podcast import (
    Episode,
    PodcastSummarizerInput,
    PodcastSummarizerWorkflow,
    chunk_text,
    parse_feed_episodes,
    select_episodes,
)

EXECUTION_TIMEOUT = timedelta(seconds=10)
RESULT_TIMEOUT = 15

NEW_EPISODE_AT = datetime.now(timezone.utc) - timedelta(hours=1)
OLD_EPISODE_AT = datetime.now(timezone.utc) - timedelta(days=8)


def _rss() -> str:
    def rfc2822(moment: datetime) -> str:
        return moment.strftime("%a, %d %b %Y %H:%M:%S +0000")

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" version="2.0">
  <channel>
    <title>Example Show</title>
    <item>
      <title>Episode 2: The Follow-up</title>
      <guid>ep-2</guid>
      <pubDate>{rfc2822(NEW_EPISODE_AT)}</pubDate>
      <enclosure url="https://cdn.example.com/ep2.mp3" type="audio/mpeg" length="1000"/>
    </item>
    <item>
      <title>Episode 1: The Beginning</title>
      <guid>ep-1</guid>
      <pubDate>{rfc2822(OLD_EPISODE_AT)}</pubDate>
      <enclosure url="https://cdn.example.com/ep1.mp3" type="audio/mpeg" length="1000"/>
    </item>
  </channel>
</rss>
"""


def test_parse_feed_episodes() -> None:
    episodes = parse_feed_episodes(_rss())
    assert [episode.guid for episode in episodes] == ["ep-2", "ep-1"]
    assert episodes[0].title == "Episode 2: The Follow-up"
    assert episodes[0].audio_url == "https://cdn.example.com/ep2.mp3"
    assert episodes[0].published_at == NEW_EPISODE_AT.replace(microsecond=0)


def test_select_episodes_picks_only_recent() -> None:
    episodes = parse_feed_episodes(_rss())
    selected = select_episodes(episodes, now=datetime.now(timezone.utc), since_hours=25, max_episodes=1, episode_guid=None)
    assert [episode.guid for episode in selected] == ["ep-2"]

    none = select_episodes(
        parse_feed_episodes(_rss()),
        now=datetime.now(timezone.utc) + timedelta(days=30),
        since_hours=25,
        max_episodes=1,
        episode_guid=None,
    )
    assert none == []


def test_select_episodes_specific_guid() -> None:
    episodes = parse_feed_episodes(_rss())
    selected = select_episodes(episodes, now=datetime.now(timezone.utc), since_hours=25, max_episodes=1, episode_guid="ep-1")
    assert [episode.guid for episode in selected] == ["ep-1"]


def test_chunk_text_splits_on_sentence_boundary() -> None:
    text = ". ".join(f"Sentence {i}" for i in range(200)) + "."
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) > 1
    assert all(len(chunk) <= 301 for chunk in chunks)
    assert "".join(chunk for chunk in chunks).replace(" ", "") == text.replace(" ", "")


def test_chunk_text_short() -> None:
    assert chunk_text("short transcript") == ["short transcript"]


async def test_workflow_end_to_end(
    temporal_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mistralai.workflows.testing import create_test_worker

    from worker.workflows import podcast

    async def fake_fetch(feed_url: str) -> list[Episode]:
        return parse_feed_episodes(_rss())

    async def fake_transcribe(audio_url: str) -> str:
        return "Host: Welcome to the show. Guest: Thanks for having me."

    async def fake_summarize(title: str, transcript: str) -> str:
        return "## Episode Summary\nA conversation."

    monkeypatch.setattr(podcast, "_fetch_feed", fake_fetch)
    monkeypatch.setattr(podcast, "_transcribe_audio", fake_transcribe)
    monkeypatch.setattr(podcast, "_summarize_text", fake_summarize)

    async with temporal_env as env:
        async with create_test_worker(
            env,
            workflows=[PodcastSummarizerWorkflow],
            activities=[
                podcast.fetch_feed_episodes,
                podcast.select_new_episodes,
                podcast.transcribe_episode,
                podcast.summarize_transcript,
            ],
        ):
            handle = await env.client.start_workflow(
                "podcast-summarizer",
                PodcastSummarizerInput(feed_url="https://example.com/feed.xml").model_dump(),
                id="podcast-summarizer-test",
                task_queue="test-task-queue",
                execution_timeout=EXECUTION_TIMEOUT,
            )
            result = await asyncio.wait_for(handle.result(), timeout=RESULT_TIMEOUT)

    assert result["feed_urls"] == ["https://example.com/feed.xml"]
    assert result["episodes_in_feed"] == 2
    assert result["episodes_processed"] == 1
    assert result["feeds"][0]["episodes_processed"] == 1
    assert result["results"][0]["guid"] == "ep-2"
    assert result["results"][0]["transcript"].startswith("Host:")
    assert result["results"][0]["summary"].startswith("## Episode Summary")


async def test_workflow_no_new_episodes(
    temporal_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mistralai.workflows.testing import create_test_worker

    from worker.workflows import podcast

    async def fake_fetch(feed_url: str) -> list[Episode]:
        return parse_feed_episodes(_rss())

    monkeypatch.setattr(podcast, "_fetch_feed", fake_fetch)

    async with temporal_env as env:
        async with create_test_worker(
            env,
            workflows=[PodcastSummarizerWorkflow],
            activities=[
                podcast.fetch_feed_episodes,
                podcast.select_new_episodes,
                podcast.transcribe_episode,
                podcast.summarize_transcript,
            ],
        ):
            handle = await env.client.start_workflow(
                "podcast-summarizer",
                PodcastSummarizerInput(
                    feed_url="https://example.com/feed.xml",
                    since_hours=1,
                ).model_dump(),
                id="podcast-summarizer-test-empty",
                task_queue="test-task-queue",
                execution_timeout=EXECUTION_TIMEOUT,
            )
            result = await asyncio.wait_for(handle.result(), timeout=RESULT_TIMEOUT)

    assert result["episodes_processed"] == 0
    assert "No new episodes" in result["message"]


async def test_workflow_multiple_feeds(
    temporal_env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mistralai.workflows.testing import create_test_worker

    from worker.workflows import podcast

    RSS_ONLY_OLD = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Quiet Show</title>
        <item>
          <title>Episode 9: Long Ago</title>
          <guid>ep-9</guid>
          <pubDate>Mon, 02 Jan 2006 08:00:00 +0000</pubDate>
          <enclosure url="https://cdn.example.com/ep9.mp3" type="audio/mpeg" length="1000"/>
        </item>
      </channel>
    </rss>
    """
    feeds = {
        "https://example.com/feed-a.xml": _rss(),
        "https://example.com/feed-b.xml": RSS_ONLY_OLD,
    }

    async def fake_fetch(feed_url: str) -> list[Episode]:
        return parse_feed_episodes(feeds[feed_url])

    async def fake_transcribe(audio_url: str) -> str:
        return "Host: Welcome to the show."

    async def fake_summarize(title: str, transcript: str) -> str:
        return "## Episode Summary\nA conversation."

    monkeypatch.setattr(podcast, "_fetch_feed", fake_fetch)
    monkeypatch.setattr(podcast, "_transcribe_audio", fake_transcribe)
    monkeypatch.setattr(podcast, "_summarize_text", fake_summarize)

    async with temporal_env as env:
        async with create_test_worker(
            env,
            workflows=[PodcastSummarizerWorkflow],
            activities=[
                podcast.fetch_feed_episodes,
                podcast.select_new_episodes,
                podcast.transcribe_episode,
                podcast.summarize_transcript,
            ],
        ):
            handle = await env.client.start_workflow(
                "podcast-summarizer",
                PodcastSummarizerInput(feed_urls=list(feeds), since_hours=25, max_episodes=1).model_dump(),
                id="podcast-summarizer-test-multi",
                task_queue="test-task-queue",
                execution_timeout=EXECUTION_TIMEOUT,
            )
            result = await asyncio.wait_for(handle.result(), timeout=RESULT_TIMEOUT)

    by_feed = {stat["feed_url"]: stat for stat in result["feeds"]}
    assert by_feed["https://example.com/feed-a.xml"]["episodes_processed"] == 1
    assert by_feed["https://example.com/feed-b.xml"]["episodes_processed"] == 0
    assert "feed-b.xml" in result["message"]
    assert [res["feed_url"] for res in result["results"]] == ["https://example.com/feed-a.xml"]
