"""
Scheduled jobs for My Cases / Hearing Reminders (fleshed out in Phase 2 —
see app/tools/case_data_provider.py and app/services/notification_dispatch.py).
Registered from app.scheduler.register_jobs() at app startup.
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler


def register(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        poll_cause_lists_and_sync_cases,
        "interval",
        hours=6,
        id="poll_cause_lists_and_sync_cases",
        replace_existing=True,
    )
    scheduler.add_job(
        send_hearing_reminders,
        "interval",
        minutes=15,
        id="send_hearing_reminders",
        replace_existing=True,
    )


async def poll_cause_lists_and_sync_cases() -> None:
    """Re-syncs saved_cases whose last_synced_at is stale, diffs in new
    case_events, and fires new_order notifications + AI summaries."""
    from app.jobs._case_sync import run_poll_cause_lists_and_sync_cases

    await run_poll_cause_lists_and_sync_cases()


async def send_hearing_reminders() -> None:
    """Fires 7-day/1-day/2-hour reminders for upcoming hearings, deduped
    against existing `notifications` rows of that type."""
    from app.jobs._case_sync import run_send_hearing_reminders

    await run_send_hearing_reminders()
