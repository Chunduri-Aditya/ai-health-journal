import re

EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PHONE = re.compile(r"\b(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
# Common international formats (country code + space/hyphen-separated groups
# of 2-5 digits, e.g. UK "+44 20 7946 0958", India "+91 98765 43210"). Not
# comprehensive -- formats with single-digit groupings (e.g. French mobile
# "+33 6 12 34 56 78") are not covered. Full i18n phone parsing needs a
# dedicated library (e.g. Google's libphonenumber); a hand-rolled regex
# chasing every country's grouping convention trades rising false-positive
# risk for diminishing coverage, so this stays intentionally partial.
INTL_PHONE = re.compile(r"\+\d{1,3}[\s-]\d{2,5}(?:[\s-]\d{2,5}){1,3}")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Shape-based, like PHONE above -- not Luhn-validated. A 13-19 digit run
# grouped in 4s (with optional space/hyphen separators, or none) is treated
# as a plausible card number; the cost of an occasional over-redacted
# reference number is far lower than under-redacting a real card number.
CREDIT_CARD = re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")


def redact(text: str) -> str:
    """
    Apply simple PII redaction to user text.

    - Emails -> [REDACTED_EMAIL]
    - Phone numbers (US-format and common international) -> [REDACTED_PHONE]
    - SSN -> [REDACTED_SSN]
    - Credit card numbers -> [REDACTED_CARD]
    - IPv4 addresses -> [REDACTED_IP]

    Deliberately NOT covered, and not attempted: street addresses and full
    names. Both need semantic understanding (a NER model, or at minimum an
    address/name dictionary) that a regex fundamentally cannot provide --
    any attempt would either miss most real instances or over-redact
    ordinary capitalized phrases ("New York", "Monday Morning") constantly.
    A regex that gives false confidence is worse than one that is honest
    about what it doesn't cover.
    """
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = PHONE.sub("[REDACTED_PHONE]", text)
    text = INTL_PHONE.sub("[REDACTED_PHONE]", text)
    text = SSN.sub("[REDACTED_SSN]", text)
    text = CREDIT_CARD.sub("[REDACTED_CARD]", text)
    text = IPV4.sub("[REDACTED_IP]", text)
    return text

