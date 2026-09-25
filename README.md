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

Neither source needs a key. `.env.example` documents that explicitly - there is nothing to configure for the default path.

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest

# offline, against the committed fixture (default provider)
python -m volsurface.snapshot --underlying RELIANCE --provider fixture --out /tmp/RELIANCE.json

# live capture (network permitting - see Findings for why NSE currently refuses this)
python -m volsurface.snapshot --underlying RELIANCE --provider nse
python -m volsurface.snapshot --underlying RELIANCE --provider yfinance
```

`volsurface.chain.FixtureProvider` is the default read path for everything downstream, so tests and later days never block on network access. `volsurface.snapshot` is the capture CLI: it fetches live and writes the same normalised shape as a fixture under `fixtures/chains/<UNDERLYING>.json` - it never falls back to a stale fixture on failure, it just exits non-zero with the provider's error.

## Findings

Day 1 (chain fetcher): both live sources were tried end to end from this sandbox, not just written and assumed working.

- **NSE's endpoint reaches, but is empty.** A `requests.Session` GET to `nseindia.com` to warm a cookie returns a real 170KB homepage (HTTP 200) - it is not a network-level block. The follow-up GET to `/api/option-chain-equities?symbol=RELIANCE` also returns HTTP 200, but the body is `{}`. No error, no rate-limit message, just no data - NSE's edge is discriminating on something beyond headers and a warmed cookie (likely TLS/JA3 fingerprinting or a JS challenge this session can't run). `NSEChainProvider` surfaces that as `ProviderError: unexpected NSE payload ... no 'records' key`, which is at least legible, but it is not a fix.
- **yfinance genuinely lists no options for NSE single names.** `yfinance.Ticker("RELIANCE.NS").options` comes back empty, live, from this sandbox. This isn't a sandbox artifact - Yahoo's options data for NSE-listed equities is documented elsewhere as unreliable/absent; RELIANCE.NS reproduces that directly. The cross-check source in the README's original data-sources list works in code (see `TestYFinanceChainProviderOffline`) but has nothing real to check against for this project's target names.
- **Net effect:** the fixture path is not a fallback for this project, it is the only path that currently produces data. `fixtures/chains/RELIANCE.json` is a hand-assembled (Black-Scholes-priced, realistic equity skew) payload run through the real `parse_nse_payload()`, not a live capture - see Limitations.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- **`fixtures/chains/RELIANCE.json` is not a live capture.** Both live sources are blocked in practice from this sandbox (see Findings) - NSE returns HTTP 200 with an empty body, yfinance lists zero expiries for NSE single names. The fixture was built by assembling a plausible NSE-shaped payload (spot 1400.5, two expiries, 13 strikes, a downward equity skew, `r = 6.5%`, no dividend adjustment) and pricing every leg with Black-Scholes, then running that payload through the real `parse_nse_payload()` so the fixture is produced by the same code a live capture would use. It exercises the parser and the schema honestly; it does not exercise a real NSE response, and every IV computed against it in later days will be internally consistent by construction, not evidence the solver handles real market noise.
- Quotes go stale and spreads widen. A surface built from a thin snapshot describes the spread, not the volatility.
- Dividend and rate assumptions move implied vol; both are stated explicitly rather than buried.
- Strikes surviving the liquidity filter are a biased sample of the chain, skewed toward the money.
- `NSEChainProvider` and `YFinanceChainProvider` are implemented and unit-tested against injected payloads/mocks (`TestNSEChainProviderOffline`, `TestYFinanceChainProviderOffline`), but neither has been proven against a real successful response - only against the real *failure* modes this sandbox produced.

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
