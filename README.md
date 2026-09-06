# Implied volatility surface

Builds a volatility surface from real option chains across strikes and expiries, with arbitrage checks that catch the artifacts a pretty 3D plot would hide.

**Status:** Not started · Next: Day 1 - option chain fetcher with snapshot-to-fixture capture

## What this is

Solves implied volatility per contract, computes Greeks analytically and cross-checks them against finite differences, then assembles smile, term structure and surface.

Illiquid strikes produce garbage IVs that look like real structure, so filtering and butterfly/calendar arbitrage checks run before anything is plotted.

## Correctness gate

IV round-trips (price to IV to price), analytic Greeks match finite differences, no static-arbitrage violations survive filtering.

This is the test that decides whether the repo is finished. A result that has not passed it is a draft.

## Data sources

Every source is free. Nothing in this project requires a paid tier, a subscription, or a funded account.

- NSE option chain public endpoint - polite headers, rate limited, snapshots committed as fixtures
- yfinance options - cross-checks

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m volsurface.build --underlying EXAMPLE --snapshot fixtures/chain_sample.json
```

Runs offline against committed fixtures by default. Live data needs a key in `.env` (see `.env.example`); the fixture path is the default so nothing blocks on network access.

## Findings

Nothing yet. This section fills in as the work lands, including the results that do not flatter the method.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Quotes go stale and spreads widen. A surface built from a thin snapshot describes the spread, not the volatility.
- Dividend and rate assumptions move implied vol; both are stated explicitly rather than buried.
- Strikes surviving the liquidity filter are a biased sample of the chain, skewed toward the money.

## Where this sits

Part of a nine-repo research pipeline. Stock Stalker screens the NSE universe; this repo publishes a versioned artifact it reads back:

```json
{
  "vol": {
    "atm_iv_30d": 0.28,
    "skew_25d": 0.04,
    "iv_rank_1y": 0.62
  }
}
```

Communication is by file contract, not imports, so either side can be refactored without breaking the other.

## Exam mapping

Series VIII (options, Greeks, implied volatility)

---

CLI only, by design. No dashboard, no server. Charts and documents are written to `outputs/`.
