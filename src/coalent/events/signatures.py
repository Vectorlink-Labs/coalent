"""Webhook signature verification (HMAC-SHA256, GitHub-style)."""
from __future__ import annotations

import hashlib
import hmac


def compute_github_signature(secret: str, body: bytes) -> str:
    """Return the ``sha256=<hex>`` signature for a raw request body."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_github_signature(secret: str, body: bytes, signature: str | None) -> bool:
    """Constant-time check of a GitHub ``X-Hub-Signature-256`` header."""
    if not signature:
        return False
    expected = compute_github_signature(secret, body)
    return hmac.compare_digest(expected, signature)
