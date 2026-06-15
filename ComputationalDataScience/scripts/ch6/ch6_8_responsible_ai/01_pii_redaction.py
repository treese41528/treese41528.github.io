"""
STAT 418 · Chapter 6.8 — Responsible AI Practices
Script 01: PII detection and redaction — clean data before it reaches any model.

Before sending text to an LLM (even a local one), scan for personally identifiable
information and redact it. This is pure pattern-matching: no API, no data leaves the
machine. The PrivacyAuditor class packages scan + redact + a batch audit
(Exercise 6.8.1).

Pure Python - no API key required.

Prereqs / run: nothing beyond the standard library.
"""
from __future__ import annotations

import re

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
}


def detect_pii(text: str) -> dict:
    """Return {pii_type: [matches]} for any PII patterns found in the text."""
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[pii_type] = matches
    return found


def redact_pii(text: str) -> str:
    """Replace detected PII with type-specific placeholders.

    SSN (3-2-4 digits) is redacted before phone (3-3-4) so a 9-digit SSN is not
    partially mistaken for a phone number.
    """
    for pii_type in ("email", "ssn", "credit_card", "phone"):
        text = re.sub(PII_PATTERNS[pii_type], f"[{pii_type.upper()}]", text)
    return text


class PrivacyAuditor:
    """Scan + redact + batch-audit text for PII (Exercise 6.8.1)."""

    def scan(self, text: str) -> dict:
        found = detect_pii(text)
        return {"has_pii": bool(found), "findings": found}

    def redact(self, text: str) -> str:
        return redact_pii(text)

    def audit_batch(self, texts: list) -> list:
        results = [self.scan(t) for t in texts]
        n = sum(1 for r in results if r["has_pii"])
        print(f"Audited {len(texts)} texts: {n} contain PII ({n / len(texts):.0%})")
        return results


def main() -> None:
    sample = ("Contact John Smith at john.smith@example.com or call 765-555-1234. "
              "SSN: 123-45-6789.")
    print("Single-text scan:")
    print(f"  detected: {detect_pii(sample)}")
    print(f"  redacted: {redact_pii(sample)}")

    print("\nBatch audit:")
    auditor = PrivacyAuditor()
    texts = [
        "Contact me at user@example.com",
        "Call 555-123-4567 for details",
        "No personal information here.",
        "SSN: 123-45-6789 is sensitive",
    ]
    auditor.audit_batch(texts)
    for t in texts:
        print(f"  {t!r}\n    -> {auditor.redact(t)!r}")


if __name__ == "__main__":
    main()
