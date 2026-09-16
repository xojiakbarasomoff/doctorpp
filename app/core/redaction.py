def preview(text: str, limit: int = 40) -> str:
    """A short, log-safe preview of user-provided text: length-capped so
    patient-adjacent content (an inbound message or a generated reply)
    doesn't sit in full at INFO level, where logs may ship to external
    monitoring (TZ section 7, personal data). Callers that need the full
    text for local debugging should log it separately at DEBUG.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "…"
