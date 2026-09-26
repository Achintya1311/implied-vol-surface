# Implied volatility surface

Builds a volatility surface from real option chains across strikes and expiries, with arbitrage checks that catch the artifacts a pretty 3D plot would hide.

**Status:** Last checkpoint 2026-09-26 · Next: Day 3 - Greeks analytically (delta, gamma, vega, theta) plus a finite-difference cross-check

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

# solve implied vol for every contract in a chain (offline, fixture by default)
python -m volsurface.solve --underlying RELIANCE --out outputs/RELIANCE_iv.csv
```

`volsurface.chain.FixtureProvider` is the default read path for everything downstream, so tests and later days never block on network access. `volsurface.snapshot` is the capture CLI: it fetches live and writes the same normalised shape as a fixture under `fixtures/chains/<UNDERLYING>.json` - it never falls back to a stale fixture on failure, it just exits non-zero with the provider's error.

## Findings

Day 1 (chain fetcher): both live sources were tried end to end from this sandbox, not just written and assumed working.

- **NSE's endpoint reaches, but is empty.** A `requests.Session` GET to `nseindia.com` to warm a cookie returns a real 170KB homepage (HTTP 200) - it is not a network-level block. The follow-up GET to `/api/option-chain-equities?symbol=RELIANCE` also returns HTTP 200, but the body is `{}`. No error, no rate-limit message, just no data - NSE's edge is discriminating on something beyond headers and a warmed cookie (likely TLS/JA3 fingerprinting or a JS challenge this session can't run). `NSEChainProvider` surfaces that as `ProviderError: unexpected NSE payload ... no 'records' key`, which is at least legible, but it is not a fix.
- **yfinance genuinely lists no options for NSE single names.** `yfinance.Ticker("RELIANCE.NS").options` comes back empty, live, from this sandbox. This isn't a sandbox artifact - Yahoo's options data for NSE-listed equities is documented elsewhere as unreliable/absent; RELIANCE.NS reproduces that directly. The cross-check source in the README's original data-sources list works in code (see `TestYFinanceChainProviderOffline`) but has nothing real to check against for this project's target names.
- **Net effect:** the fixture path is not a fallback for this project, it is the only path that currently produces data. `fixtures/chains/RELIANCE.json` is a hand-assembled (Black-Scholes-priced, realistic equity skew) payload run through the real `parse_nse_payload()`, not a live capture - see Limitations.

Day 2 (IV solver, `volsurface/iv.py` + `volsurface/solve.py`): Newton-Raphson with analytic vega, falling back to Brent's method (`scipy.optimize.brentq`, bounded to sigma in `[1e-6, 5.0]`) when a Newton step's own size (not the price residual - see below) hasn't shrunk below tolerance. Solved against the bid/ask mid, not `last_price`, since a stale last trade is a worse target than a live two-sided quote.

- **A price-residual convergence check is the wrong test for deep ITM/OTM.** The first version of this solver declared Newton "converged" once the price residual fell under an absolute tolerance. That is wrong wherever vega is small: a deep ITM call's price is nearly flat in sigma, so many different sigma values reproduce the same price to 1e-8 - the loop stopped early at whichever sigma it happened to be visiting, not at the true one. Caught by `TestSolveIvRoundTrip::test_price_to_iv_to_price_round_trips[0.15-CE-1000.0]` failing with a recovered IV of 0.197 against a true 0.15 - a confidently wrong number, not a crash. Fixed by judging convergence on Newton's own step size in sigma, which is only small when the *sigma* estimate has actually stopped moving.
- **Some deep ITM/OTM prices have no recoverable IV in double precision at all**, not just a hard-to-find one. `S*N(d1)` and `K*e^{-rT}*N(d2)` are both O(1000) while their difference (the option's extrinsic value) can be O(1e-10) or smaller; below a certain moneyness/vol/tenor combination, every candidate sigma from near-zero to the search's upper bound prices to the *same* floating-point floor value - there's no sign change for Brent to bracket, because there's no distinguishable root to find. `solve_iv` checks for this explicitly (price within `1e-6` of its own no-arbitrage floor) and raises `NoSolution` naming it, rather than either crashing on Brent's bracket-not-found error or quietly returning whatever sigma Newton happened to stop at. See `TestDeepMoneynessNumericalFloor`.
- **Two contracts in the committed `RELIANCE.json` fixture have no solution for a more mundane reason: cent-rounding.** The nearest-expiry `1250 CE` and `1550 PE` are so deep ITM (5 days to expiry) that their bid/ask mid, rounded to two decimals when the fixture was hand-assembled, lands about half a paisa *below* the theoretical no-arbitrage floor implied by `r=6.5%`. This isn't a numerical-floor case (the two are 0.0025 and 0.0005 apart, well above the 1e-6 double-precision threshold above) - it's what real tick-rounded quotes do to a theoretical bound near the money's edge. `solve_iv` still refuses rather than returning a number for a price that, by its own arithmetic, implies riskless arbitrage. `python -m volsurface.solve --underlying RELIANCE` reports both by name; 50/52 contracts solve.
- **The recovered smile is internally consistent, and shows the skew the fixture was built with.** Put and call IVs at the same strike/expiry agree to 3-4 decimal places away from the tails (put-call parity holds, as it must for a solver built from the same closed-form pricer on both sides), and IV rises as strike falls - the "downward equity skew" the fixture's docstring claims. That agreement is a property of solving against a self-consistent synthetic chain, not evidence the solver would behave this well against noisy real quotes - see Limitations.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-26 | `d66a5f2` | Day 2: IV solver (volsurface.iv + volsurface.solve CLI). Newton-Raphson with analytic vega, judged on step size in sigma (not price residual - a price-residual check falsely converged deep ITM/OTM, caught by the round-trip test recovering IV 0.197 against a true 0.15), falling back to a bounded Brent search when a step leaves [1e-6, 5.0]. Two no-solution classes surfaced explicitly rather than as NaN: prices outside no-arbitrage bounds, and prices within 1e-6 of their own floor - genuinely unrecoverable in double precision (S*N(d1) and K*e^{-rT}*N(d2) both O(1000), their difference O(1e-10) or smaller, no sign change for Brent to bracket). 50/52 RELIANCE fixture contracts solve; the other two (1250 CE, 1550 PE, both 5-day) are real cent-rounding artifacts that violate the no-arbitrage floor by under a paisa - recorded, not hidden. Solved against bid/ask mid, not last_price. r=6.5%/q=0% match the fixture's own pricing assumptions (Day 1 README), documented as CLI defaults, not calibrated. 73/73 tests pass (44 new), CLI run by hand end to end. | Day 3 - Greeks analytically (delta, gamma, vega, theta) plus a finite-difference cross-check |
| 2026-09-25 | `8202588` | Day 1: option chain fetcher with snapshot-to-fixture capture (volsurface.chain + volsurface.snapshot CLI). Live NSE endpoint reaches but returns HTTP 200 with an empty body from this sandbox; yfinance genuinely lists zero option expiries for RELIANCE.NS. Both findings recorded in README rather than hidden. fixtures/chains/RELIANCE.json is a hand-assembled Black-Scholes-priced payload run through the real parser (parse_nse_payload), not a live capture. 29/29 tests pass, CLI run by hand against all three providers. | Day 2 - IV solver: Newton-Raphson with a Brent fallback; deep ITM/OTM and no-solution cases handled explicitly |
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- **`fixtures/chains/RELIANCE.json` is not a live capture.** Both live sources are blocked in practice from this sandbox (see Findings) - NSE returns HTTP 200 with an empty body, yfinance lists zero expiries for NSE single names. The fixture was built by assembling a plausible NSE-shaped payload (spot 1400.5, two expiries, 13 strikes, a downward equity skew, `r = 6.5%`, no dividend adjustment) and pricing every leg with Black-Scholes, then running that payload through the real `parse_nse_payload()` so the fixture is produced by the same code a live capture would use. It exercises the parser and the schema honestly; it does not exercise a real NSE response, and every IV computed against it in later days will be internally consistent by construction, not evidence the solver handles real market noise.
- Quotes go stale and spreads widen. A surface built from a thin snapshot describes the spread, not the volatility.
- Dividend and rate assumptions move implied vol; both are stated explicitly rather than buried.
- Strikes surviving the liquidity filter are a biased sample of the chain, skewed toward the money.
- `NSEChainProvider` and `YFinanceChainProvider` are implemented and unit-tested against injected payloads/mocks (`TestNSEChainProviderOffline`, `TestYFinanceChainProviderOffline`), but neither has been proven against a real successful response - only against the real *failure* modes this sandbox produced.
- **The IV solver has only ever seen a self-consistent synthetic chain.** Every price it has solved was itself produced by the same Black-Scholes formula it inverts, with no bid/ask noise beyond the fixture's own tick-rounding. A real NSE chain has crossed quotes, stale prints, and dividend/borrow effects this solver assumes away (`q=0` throughout - see `DEFAULT_DIVIDEND_YIELD` in `volsurface/iv.py`); none of that has been exercised yet.
- **The solver refuses two real, if narrow, classes of quote rather than guessing at them**: prices within `1e-6` of their own no-arbitrage floor (genuinely unrecoverable in double precision - see Findings) and prices that violate their no-arbitrage bound outright (as the two cent-rounded deepest-ITM legs in the RELIANCE fixture do). Both are correct refusals, not bugs, but they mean "IV solved for every contract" is not yet true for any chain, including the one committed here.
- `r=6.5%` and `q=0%` are CLI defaults (`--rate`/`--dividend-yield`), not calibrated to any real term structure; the recovered smile is only as trustworthy as that assumption.

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
