# 0006 — A failed-authentication lockout also refuses valid keys

**Status:** Accepted

## Context

Requests with a wrong API key were rejected before the rate limiter ran, so nothing bounded
how quickly keys could be guessed. A per-address lockout was added: after a number of
rejected keys within a window, the address is throttled.

The question is what a locked-out address may still do. The user-friendly answer is "a
request with a correct key should still succeed" — it protects a legitimate client who
shares an address with someone mistyping a key.

## Decision

During a lockout, refuse **every** request from that address with a 429, including one
carrying a valid key. The lockout is checked before the key is examined, so nothing about
the key can influence the response.

Three properties keep this from being heavier than it sounds:

- Successful requests never touch the limiter. A client that authenticates correctly cannot
  contribute to a lockout and pays nothing for the feature.
- Throttled attempts are not recorded, so a client that keeps hammering does not extend its
  own lockout; it lifts when the oldest failure ages out.
- Public endpoints (`/health`, `/ready`, `/metrics`) are unaffected, so probes sharing the
  address keep working.

## Consequences

- The lockout is not an oracle. If valid keys were let through, a guesser would see 429 for
  a wrong key and 200 for the right one, and could keep guessing at full speed through the
  "lockout" — the control would be decorative.
- A legitimate client can be denied for up to one window because of failures from the same
  address. Behind a proxy with the address misconfigured (ADR 0005) that means everyone.
  The cost is real and is the price of the previous point.
- This is defence in depth, not the primary control. Against a high-entropy key online
  guessing is already infeasible; the lockout protects weak keys and limits log noise,
  which is why the service also warns at startup about keys shorter than 32 characters.

## Alternatives considered

- **Let valid keys through during a lockout.** Rejected for the oracle described above.
- **Lock the key instead of the address.** Not applicable: a wrong key identifies nothing to
  lock.
- **Exponential backoff per address.** A refinement worth having at scale; a fixed window
  is simpler to reason about and to test, and its worst case is bounded and documented.
