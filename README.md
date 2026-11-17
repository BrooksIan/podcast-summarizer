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
  src/worker/entrypoints/worker.py  # worker entrypoint (SIGTERM-safe)
  src/worker/entrypoints/dev.py     # hot-reload dev entrypoint
  tests/                            # pytest suite (SDK test environment)
packages/py/env/                    # env settings shared across the app
deploy/docker/Dockerfile.worker     # image for the managed deployment
mistral.json                        # app manifest (`worker` module, workflows platform)
```

## Local development

```bash
bash tools/install.sh        # bun install + uv sync for the whole workspace
cd apps/worker
uv run --all-packages pytest
```

To run a worker locally, put `MISTRAL_API_KEY` in the repo-root `.env`, then
`mistral apps dev --filter worker` from the repository root.

## Running it

- **Manual run**: execute the `podcast-summarizer` workflow with input:

  ```json
  {
    "feed_urls": [
      "https://example.com/feed.xml",
      "https://feeds.megaphone.fm/ESP7699980268"
    ],
    "since_hours": 25,
    "max_episodes": 1
  }
  ```

  `feed_url` (single feed) still works; `feed_urls` watches several feeds in
  one run, up to `max_episodes` new episodes each. Optional: set
  `episode_guid` to force processing of one specific episode regardless of
  its publication date.

- **Automatic runs**: a workflow schedule fires the workflow on a cron
  schedule with a fixed input (feed URL, `since_hours`, `max_episodes`).

## Deployment

`mistral apps deploy` from the repository root deploys the `worker` module
(marked `"platform": "workflows"` in `mistral.json`) as a managed workflow
deployment built from `deploy/docker/Dockerfile.worker`. Push your commits
first: the image builds at the current commit. No API key is needed in the
image: the worker authenticates with the platform-injected service-account
token.
