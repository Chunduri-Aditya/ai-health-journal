import re

EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE = re.compile(r"\b(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")


def redact(text: str) -> str:
    """
    Apply simple PII redaction to user text.

    - Emails -> [REDACTED_EMAIL]
    - Phone numbers -> [REDACTED_PHONE]

    (Can be extended later with additional rules, e.g. addresses.)
    """
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = PHONE.sub("[REDACTED_PHONE]", text)
    return text

