"""
A tenant: one team or service sharing this deployment with others.

Running the API as an internal platform means several teams use one instance,
and each must see only its own documents, be throttled on its own budget and
be billed for its own tokens. The tenant is the unit all three hang off. It is
resolved once, from the API key, before any route runs, and travels with the
request from there; nothing a caller sends in a body or a query string can
choose it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Tenant ids become metric labels, rate-limit bucket names and log fields, so
# they are kept short and plain: no spaces, no separators a log parser or a
# Redis key could misread.
_TENANT_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")

# ChromaDB's own rule for collection names: 3 to 63 characters, alphanumerics,
# dots, underscores and dashes, starting and ending with an alphanumeric.
_COLLECTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]")


class InvalidTenantError(ValueError):
    """A tenant definition that cannot be served safely."""


@dataclass(frozen=True, slots=True)
class Tenant:
    id: str
    # The vector-store collection holding this tenant's documents, and no one
    # else's: isolation is by collection, not by a filter every query must apply.
    collection: str
    # Requests admitted per rate-limit window; None uses the deployment default.
    rate_limit_requests: int | None = None
    # LLM tokens (input and output) the tenant may consume per UTC day; None is
    # unlimited.
    daily_token_budget: int | None = None

    def __post_init__(self) -> None:
        if not _TENANT_ID.fullmatch(self.id):
            raise InvalidTenantError(
                f"tenant id {self.id!r} must be 1-40 lowercase letters, digits or dashes"
            )
        if not _COLLECTION.fullmatch(self.collection):
            raise InvalidTenantError(
                f"tenant {self.id!r}: collection {self.collection!r} is not a valid "
                "collection name (3-63 letters, digits, '.', '_' or '-')"
            )
        for name, value in (
            ("rate_limit_requests", self.rate_limit_requests),
            ("daily_token_budget", self.daily_token_budget),
        ):
            if value is not None and value < 1:
                raise InvalidTenantError(f"tenant {self.id!r}: {name} must be at least 1")
