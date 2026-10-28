"""
Generates an API key for a tenant, and the digest to put in the tenants file.

Usage:
    uv run python scripts/tenant_key.py
    uv run python scripts/tenant_key.py --digest-of "an-existing-key"

The key is printed once and stored nowhere: hand it to the team that will use
it, through whatever channel carries secrets in your organisation. Only the
digest goes into TENANTS_FILE, which is why that file is not itself a secret.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from collections.abc import Sequence

from aegis.infrastructure.tenancy.registry import key_digest

# 32 random bytes, URL-safe: 43 characters, well above the length startup
# warns about.
_KEY_BYTES = 32


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--digest-of", metavar="KEY", help="print the digest of an existing key")
    args = parser.parse_args(argv)

    if args.digest_of:
        print(key_digest(args.digest_of).hex())
        return 0
    key = secrets.token_urlsafe(_KEY_BYTES)
    print(f"api key (give to the tenant, store nowhere else): {key}")
    print(f"api_key_sha256 (put in TENANTS_FILE):            {key_digest(key).hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
