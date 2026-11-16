from __future__ import annotations

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

pytest_plugins = ["mistralai.workflows.testing.fixtures"]

NOW = datetime(2026, 11, 16, 12, 0, 0, tzinfo=timezone.utc)

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" version="2.0">
  <channel>
    <title>Example Show</title>
    <item>
      <title>Episode 2: The Follow-up</title>
      <guid>ep-2</guid>
      <pubDate>Mon, 16 Nov 2026 08:00:00 +0000</pubDate>
      <enclosure url="https://cdn.example.com/ep2.mp3" type="audio/mpeg" length="1000"/>
    </item>
    <item>
      <title>Episode 1: The Beginning</title>
      <guid>ep-1</guid>
      <pubDate>Mon, 09 Nov 2026 08:00:00 +0000</pubDate>
      <enclosure url="https://cdn.example.com/ep1.mp3" type="audio/mpeg" length="1000"/>
    </item>
  </channel>
</rss>
"""


def test_parse_feed_episodes() -> None:
    episodes = parse_feed_episodes(RSS)
    assert [episode.guid for episode in episodes] == ["ep-2", "ep-1"]
    assert episodes[0].title == "Episode 2: The Follow-up"
    assert episodes[0].audio_url == "https://cdn.example.com/ep2.mp3"
    assert episodes[0].published_at == datetime(2026, 11, 16, 8, 0, 0, tzinfo=timezone.utc)


def test_select_episodes_picks_only_recent() -> None:
    episodes = parse_feed_episodes(RSS)
    selected = select_episodes(episodes, now=NOW, since_hours=25, max_episodes=1, episode_guid=None)
    assert [episode.guid for episode in selected] == ["ep-2"]

    none = select_episodes(
        parse_feed_episodes(RSS),
        now=NOW + timedelta(days=30),
        since_hours=25,
        max_episodes=1,
        episode_guid=None,
    )
    assert none == []


def test_select_episodes_specific_guid() -> None:
    episodes = parse_feed_episodes(RSS)
    selected = select_episodes(episodes, now=NOW, since_hours=25, max_episodes=1, episode_guid="ep-1")
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
    temporal_env_with_converter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mistralai.workflows.testing import create_test_worker

    from worker.workflows import podcast

    async def fake_fetch(feed_url: str) -> list[Episode]:
        return parse_feed_episodes(RSS)

    async def fake_transcribe(audio_url: str) -> str:
        return "Host: Welcome to the show. Guest: Thanks for having me."

    async def fake_summarize(title: str, transcript: str) -> str:
        return "## Episode Summary\nA conversation."

    monkeypatch.setattr(podcast, "_fetch_feed", fake_fetch)
    monkeypatch.setattr(podcast, "_transcribe_audio", fake_transcribe)
    monkeypatch.setattr(podcast, "_summarize_text", fake_summarize)

    async with temporal_env_with_converter as env:
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
            )
            result = await handle.result()

    assert result["episodes_in_feed"] == 2
    assert result["episodes_processed"] == 1
    assert result["results"][0]["guid"] == "ep-2"
    assert result["results"][0]["transcript"].startswith("Host:")
    assert result["results"][0]["summary"].startswith("## Episode Summary")


async def test_workflow_no_new_episodes(
    temporal_env_with_converter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mistralai.workflows.testing import create_test_worker

    from worker.workflows import podcast

    async def fake_fetch(feed_url: str) -> list[Episode]:
        return parse_feed_episodes(RSS)

    monkeypatch.setattr(podcast, "_fetch_feed", fake_fetch)

    async with temporal_env_with_converter as env:
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
            )
            result = await handle.result()

    assert result["episodes_processed"] == 0
    assert "No new episodes" in result["message"]
