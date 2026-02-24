# Optimization Backlog

Last updated: 2026-02-24

## P1 (Higher Priority)

- [ ] Enforce minimum market BUY notional (`>= $1`) before order submission.
  - Why: repeated live failures like `invalid amount for a marketable BUY order ... min size: $1`.
  - Expected result: avoid avoidable failed orders and reduce decision/execution mismatch.

- [ ] Align model sizing output with risk cap (`MAX_NOTIONAL_USD`) to reduce `risk_reject`.
  - Why: models still occasionally emit actions above configured notional.
  - Expected result: fewer rejected actions and cleaner execution stats.

## P2 (Deferred / To Optimize)

- [ ] Harden Gemini empty-response handling on OpenAI-compatible gateway.
  - Why: intermittent `content=''` responses were observed and currently rely on fallback (`gemini-auto`).
  - Status: running stable now; keep as deferred optimization unless empty responses rise again.

