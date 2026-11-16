# podcast-summarizer

A Mistral Workflow that watches a podcast RSS feed, transcribes new episodes
with Mistral's Voxtral speech-to-text API, and returns a structured summary
(episode overview, key points, notable quotes, guests & mentions) produced by
a Mistral chat model.

## How it works

1. A daily schedule (or a manual run) starts the `podcast-summarizer` workflow
   with the feed URL as input.
2. The workflow fetches the RSS feed and selects episodes published within the
   last `since_hours` hours (default 25, so a daily run picks up exactly the
   new episodes since the previous run).
3. Each selected episode's audio enclosure is transcribed via
   `POST /v1/audio/transcriptions` (`voxtral-mini-transcribe-3`, audio passed
   by URL).
4. The transcript is summarized with `mistral-small-latest`. Long transcripts
   are chunked and reduced (map-reduce) so multi-hour episodes work.
5. The workflow result contains, per episode: title, full transcript, and the
   structured summary.

## Project layout

```
apps/worker/
  src/worker/workflows/podcast.py   # workflow + activities
  src/worker/entrypoints/worker.py  # worker entrypoint
  tests/                            # pytest suite (Temporal test environment)
deploy/docker/Dockerfile.worker     # image for the managed deployment
```

## Local development

```bash
cd apps/worker
uv sync --frozen
uv run pytest
```

To run a worker locally you need `MISTRAL_API_KEY` in `apps/worker/.env`, then
`mistral dev` from the repository root.

## Running it

- **Manual run**: execute the `podcast-summarizer` workflow with input:

  ```json
  {
    "feed_url": "https://example.com/feed.xml",
    "since_hours": 25,
    "max_episodes": 1
  }
  ```

  Optional: set `episode_guid` to force processing of one specific episode
  regardless of its publication date.

- **Automatic runs**: a workflow schedule fires the workflow on a cron
  schedule with a fixed input (feed URL, `since_hours`, `max_episodes`).

## Deployment

The worker deploys as a managed deployment built from this repository with
`dockerfile_path=deploy/docker/Dockerfile.worker` (build context: repository
root). No API key is needed in the image: the worker authenticates with the
platform-injected service-account token.
