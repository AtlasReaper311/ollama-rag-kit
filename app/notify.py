"""Optional atlas-notify integration.

This module is deliberately silent when NOTIFY_URL is not configured.
The kit runs standalone with no changes; the integration activates by
setting two environment variables. A failure to post an alert must never
take down the RAG service itself, so every path here swallows exceptions
and logs them instead.
"""

import logging

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


async def send_alert(settings: Settings, title: str, message: str, level: str = "info") -> None:
    """POST an alert envelope to atlas-notify, if configured.

    level maps to the alert source's colour: success, failure, warning, info.
    """
    if not settings.notify_url or not settings.notify_token:
        return

    payload = {
        "source": "alert",
        "title": title,
        "message": message,
        "level": level,
        "fields": {"service": "ollama-rag-kit"},
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                settings.notify_url,
                json=payload,
                headers={"Authorization": f"Bearer {settings.notify_token}"},
            )
            if not response.is_success:
                logger.warning(
                    "atlas-notify returned %s for alert '%s'", response.status_code, title
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not reach atlas-notify for alert '%s': %s", title, exc)
