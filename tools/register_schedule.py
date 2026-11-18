#!/usr/bin/env python
"""Register the daily podcast-summarizer schedule on the Workflows platform.

Idempotent: a stable schedule_id plus the same definition update the existing
schedule instead of creating a second one. Run after code changes that alter the
schedule input to converge the platform state:

    bash tools/uv.sh run --all-packages python tools/register_schedule.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
import mistralai.workflows as workflows
from mistralai_capabilities.workflows import client as workflows_client
from worker.workflows.podcast import PodcastSummarizerWorkflow

REPO = Path(__file__).resolve().parent.parent

SCHEDULE_ID = "podcast-summarizer-daily"
# 13:00 UTC daily. All feeds, last 72h window, up to 2 episodes per feed
# (covers multi-episode days: Even Money and Fantasy Focus can publish 1-2/day).
SCHEDULE_CRON = "0 13 * * *"
SCHEDULE_INPUT = {
    "feed_urls": [
        "https://feeds.megaphone.fm/DFT4986441816",  # Even Money (Ross Tucker)
        "https://feeds.megaphone.fm/ESP7699980268",  # Fantasy Focus Football (ESPN)
        "https://feeds.simplecast.com/MqvHchJZ",  # Sports Gambling Podcast
        "https://rss.beehiiv.com/podcasts/019fd2ec-304e-7edb-81a3-3aadb83ce806.xml",  # The Simple Handicap (Adam Chernoff)
        "https://feeds.simplecast.com/sw7PGWfw",  # Fantasy Footballers
    ],
    "since_hours": 72,
    "max_episodes": 2,
}

logger = structlog.get_logger("register-schedule")


async def main() -> None:
    # The workflow must be registered before the platform accepts a schedule naming it; keep a
    # worker up for the duration of the upsert (converge retries cover the registration window).
    worker_task = asyncio.create_task(workflows_client.run_worker([PodcastSummarizerWorkflow]))
    try:
        await workflows_client.converge_schedule(
            schedule_id=SCHEDULE_ID,
            workflow_identifier="podcast-summarizer",
            enabled=True,
            input=SCHEDULE_INPUT,
            cron_expressions=[SCHEDULE_CRON],
            deployment_name=workflows_client.workflows_env.deployment_name,
        )
        schedule = await (
            await workflows_client.get_client()
        ).workflows.schedules.get_schedule_async(schedule_id=SCHEDULE_ID)
        logger.info(
            "schedule converged",
            schedule_id=SCHEDULE_ID,
            cron=SCHEDULE_CRON,
            paused=schedule.paused,
            next_run=getattr(schedule, "next_run_time", None),
        )
    finally:
        if worker_task and not worker_task.done():
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass


if __name__ == "__main__":
    asyncio.run(main())
