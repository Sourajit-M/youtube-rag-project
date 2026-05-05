from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_settings


def create_scheduler(pipeline) -> BackgroundScheduler:
  """
  Creates a background scheduler that calls pipeline.run_scheduled_check()
  on a fixed interval.

  BackgroundScheduler runs in a daemon thread — it doesn't block
  the FastAPI event loop. The interval comes from config so it's
  adjustable via environment variable without code changes.

  Why APScheduler over Celery:
  - No Redis server required — runs inside the FastAPI process
  - Simple enough for this scale (hourly polling, not high-throughput)
  - Celery would be the upgrade path if we needed distributed workers
    or thousands of concurrent jobs
  """
  settings = get_settings()
  scheduler = BackgroundScheduler()

  scheduler.add_job(
    func=pipeline.run_scheduled_check,
    trigger="interval",
    minutes=settings.ingest_interval_minutes,
    id="scheduled_ingest",
    name="Check channels for new videos",
    replace_existing=True,
  )

  return scheduler