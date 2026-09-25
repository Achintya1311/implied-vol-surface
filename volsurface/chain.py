"""Option chain fetcher with snapshot-to-fixture capture (Day 1).

Two live sources, one committed fixture, all behind the same interface:

- :class:`FixtureProvider` reads a committed JSON snapshot. This is the
  default, so the project runs with no network and no key.
- :class:`NSEChainProvider` hits NSE's public option-chain endpoint. No key,
  but it wants a warmed-up session cookie and gets touchy about request
  rate, so this pauses between the cookie GET and the data GET.
- :class:`YFinanceChainProvider` cross-checks via yfinance. No key required,
  but NSE single-stock options are not reliably listed there in practice -
  see the README limitations.

All three normalise into the same shape: a :class:`ChainSnapshot` wrapping a
``CHAIN_COLUMNS`` DataFrame. A fixture is exactly that normalised shape
serialised to JSON, not a raw copy of whatever the source returned - the
day-2 IV solver and everything after it read one schema regardless of
where the snapshot came from.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CHAIN_COLUMNS = [
    "expiry",
    "strike",
    "option_type",
    "last_price",
    "bid",
    "ask",
    "open_interest",
    "volume",
]

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "fixtures" / "chains"

# Symbols NSE serves through the *indices* option-chain endpoint rather than
# the equities one. Everything else is assumed to be a single-stock name.
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}


class ProviderError(RuntimeError):
    """The chain could not be retrieved or did not parse into the shared shape."""


class RateLimited(ProviderError):
    """The source refused the request because it looked automated or too frequent."""


def validate_quotes(df: pd.DataFrame) -> None:
    """Raise :class:`ProviderError` with a specific reason, or return silently.

    Checked here rather than left to whatever breaks downstream: a chain with
    a negative open interest or a duplicate (expiry, strike, type) row is a
    parsing bug, and it should fail loudly at the fetch boundary, not surface
    as a confusing NaN three modules later.
    """
    missing = [c for c in CHAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ProviderError(f"chain frame is missing columns: {missing}")
    if df.empty:
        raise ProviderError("chain frame has no rows")

    bad_types = set(df["option_type"].unique()) - {"CE", "PE"}
    if bad_types:
        raise ProviderError(f"unexpected option_type values: {sorted(bad_types)}")

    if (df["strike"] <= 0).any():
        raise ProviderError("non-positive strike present")

    oi = df["open_interest"].dropna()
    if (oi < 0).any():
        raise ProviderError("negative open_interest present")

    dupes = df.duplicated(subset=["expiry", "strike", "option_type"])
    if dupes.any():
        raise ProviderError(
            f"{int(dupes.sum())} duplicate (expiry, strike, option_type) rows"
        )


@dataclass
class ChainSnapshot:
    """A normalised option chain at one point in time, for one underlying."""

    underlying: str
    spot: float
    timestamp: str  # ISO 8601
    source: str
    quotes: pd.DataFrame  # CHAIN_COLUMNS

    def to_dict(self) -> dict:
        return {
            "underlying": self.underlying,
            "spot": self.spot,
            "timestamp": self.timestamp,
            "source": self.source,
            "quotes": self.quotes[CHAIN_COLUMNS].to_dict(orient="records"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChainSnapshot":
        required = {"underlying", "spot", "timestamp", "source", "quotes"}
        missing = required - data.keys()
        if missing:
            raise ProviderError(f"snapshot json is missing keys: {sorted(missing)}")
        df = pd.DataFrame(data["quotes"])
        if df.empty:
            df = pd.DataFrame(columns=CHAIN_COLUMNS)
        validate_quotes(df)
        return cls(
            underlying=data["underlying"],
            spot=float(data["spot"]),
            timestamp=data["timestamp"],
            source=data["source"],
            quotes=df[CHAIN_COLUMNS].copy(),
        )

    def to_json_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ChainSnapshot":
        path = Path(path)
        if not path.exists():
            raise ProviderError(f"no snapshot file at {path}")
        return cls.from_dict(json.loads(path.read_text()))


def _parse_nse_date(raw: str) -> str:
    """``'25-Sep-2026'`` -> ``'2026-09-25'``."""
    return datetime.strptime(raw, "%d-%b-%Y").date().isoformat()


def _parse_nse_timestamp(raw: str) -> str:
    """``'24-Sep-2026 15:30:00'`` -> ISO 8601. Falls back to the raw string.

    NSE's own timestamp format is undocumented and has drifted before,
    so a snapshot that carries a format this doesn't expect still gets
    captured - with its original string rather than a parse crash.
    """
    try:
        return datetime.strptime(raw, "%d-%b-%Y %H:%M:%S").isoformat()
    except (ValueError, TypeError):
        return raw


def parse_nse_payload(payload: dict, underlying: str, source: str) -> ChainSnapshot:
    """Normalise NSE's option-chain JSON shape into a :class:`ChainSnapshot`.

    Shared by the live provider and the fixture-capture generator, so a
    fixture built from a hand-assembled payload of this shape exercises the
    exact same parsing path a live capture would.
    """
    records = payload.get("records")
    if not records:
        raise ProviderError(f"unexpected NSE payload for {underlying}: no 'records' key")

    spot = records.get("underlyingValue")
    if spot is None:
        raise ProviderError(f"NSE payload for {underlying} has no underlyingValue")

    rows: list[dict] = []
    for entry in records.get("data", []):
        strike = entry.get("strikePrice")
        expiry_raw = entry.get("expiryDate")
        if strike is None or not expiry_raw:
            continue
        expiry = _parse_nse_date(expiry_raw)
        for option_type in ("CE", "PE"):
            leg = entry.get(option_type)
            if not leg:
                continue
            rows.append(
                {
                    "expiry": expiry,
                    "strike": float(strike),
                    "option_type": option_type,
                    "last_price": leg.get("lastPrice"),
                    "bid": leg.get("bidprice"),
                    "ask": leg.get("askPrice"),
                    "open_interest": leg.get("openInterest"),
                    "volume": leg.get("totalTradedVolume"),
                }
            )

    if not rows:
        raise ProviderError(f"NSE payload for {underlying} parsed to zero option rows")

    df = pd.DataFrame(rows)[CHAIN_COLUMNS]
    validate_quotes(df)

    return ChainSnapshot(
        underlying=underlying.upper(),
        spot=float(spot),
        timestamp=_parse_nse_timestamp(records.get("timestamp", "")),
        source=source,
        quotes=df,
    )


class FixtureProvider:
    """Reads a committed snapshot JSON. Offline, deterministic, always available."""

    name = "fixture"

    def __init__(self, directory: Path | None = None):
        self.directory = directory or FIXTURE_DIR

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def fetch(self, underlying: str) -> ChainSnapshot:
        path = self.directory / f"{underlying.upper()}.json"
        if not path.exists():
            raise ProviderError(
                f"no fixture for {underlying}. available: {', '.join(self.available()) or 'none'}"
            )
        return ChainSnapshot.from_json_file(path)


class NSEChainProvider:
    """Live NSE option chain. No key, but wants a warmed cookie and a slow pace.

    NSE's API rejects requests that arrive without a session cookie set by a
    prior visit to the site, and rate-limits aggressively. A cookie warm-up
    GET followed by a paced data GET, with a browser-shaped User-Agent, is
    the polite version of what any browser does automatically.
    """

    name = "nse"
    BASE_URL = "https://www.nseindia.com"
    EQUITY_CHAIN_PATH = "/api/option-chain-equities"
    INDEX_CHAIN_PATH = "/api/option-chain-indices"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, pause: float = 2.0, timeout: float = 15.0):
        self.pause = pause
        self.timeout = timeout

    def fetch(self, underlying: str) -> ChainSnapshot:
        import requests

        session = requests.Session()
        session.headers.update(self.HEADERS)
        symbol = underlying.upper()
        path = self.INDEX_CHAIN_PATH if symbol in INDEX_SYMBOLS else self.EQUITY_CHAIN_PATH

        try:
            warmup = session.get(self.BASE_URL, timeout=self.timeout)
            if warmup.status_code == 429:
                raise RateLimited(f"NSE rate-limited the cookie warm-up for {symbol}")
            time.sleep(self.pause)
            response = session.get(
                f"{self.BASE_URL}{path}", params={"symbol": symbol}, timeout=self.timeout
            )
            if response.status_code == 429:
                raise RateLimited(f"NSE rate-limited the chain request for {symbol}")
            response.raise_for_status()
            payload = response.json()
        except RateLimited:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as our own error type
            raise ProviderError(f"NSE fetch failed for {symbol}: {exc}") from exc

        return parse_nse_payload(payload, symbol, source=self.name)


def _to_yahoo_symbol(underlying: str) -> str:
    symbol = underlying.upper()
    if symbol in INDEX_SYMBOLS:
        return {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}.get(symbol, symbol)
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


class YFinanceChainProvider:
    """Cross-check source via yfinance. No key, no documented quota.

    NSE single-stock options are not reliably exposed through yfinance in
    practice - ``Ticker.options`` commonly comes back empty for names that
    trade actively on NSE's own chain. That gap is recorded in the README
    rather than papered over; this provider exists so the gap is visible
    (an empty-expiries error) instead of silently unavailable.
    """

    name = "yfinance"

    def fetch(self, underlying: str, expiry: str | None = None) -> ChainSnapshot:
        import yfinance as yf

        symbol = _to_yahoo_symbol(underlying)
        ticker = yf.Ticker(symbol)
        try:
            expiries = ticker.options
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"yfinance failed listing expiries for {symbol}: {exc}") from exc
        if not expiries:
            raise ProviderError(
                f"yfinance lists no option expiries for {symbol} "
                "(expected for most NSE single-stock names - see README limitations)"
            )
        chosen_expiry = expiry or expiries[0]

        try:
            chain = ticker.option_chain(chosen_expiry)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"yfinance option_chain failed for {symbol}: {exc}") from exc

        rows: list[dict] = []
        for option_type, frame in (("CE", chain.calls), ("PE", chain.puts)):
            for _, leg in frame.iterrows():
                rows.append(
                    {
                        "expiry": chosen_expiry,
                        "strike": float(leg["strike"]),
                        "option_type": option_type,
                        "last_price": leg.get("lastPrice"),
                        "bid": leg.get("bid"),
                        "ask": leg.get("ask"),
                        "open_interest": leg.get("openInterest"),
                        "volume": leg.get("volume"),
                    }
                )
        if not rows:
            raise ProviderError(f"yfinance returned zero legs for {symbol} {chosen_expiry}")

        df = pd.DataFrame(rows)[CHAIN_COLUMNS]
        validate_quotes(df)

        spot = float("nan")
        try:
            spot = float(ticker.fast_info["lastPrice"])
        except Exception:  # noqa: BLE001 - spot is best-effort here
            pass

        return ChainSnapshot(
            underlying=underlying.upper(),
            spot=spot,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=self.name,
            quotes=df,
        )


PROVIDERS = {
    FixtureProvider.name: FixtureProvider,
    NSEChainProvider.name: NSEChainProvider,
    YFinanceChainProvider.name: YFinanceChainProvider,
}


def get_provider(name: str = "fixture"):
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider '{name}'. choose from {', '.join(PROVIDERS)}")
    return PROVIDERS[name]()


def capture_snapshot(
    underlying: str, provider: str = "nse", out_path: str | Path | None = None
) -> ChainSnapshot:
    """Fetch a chain and, if ``out_path`` is given, write it as a fixture.

    The write only happens after the fetched frame has passed
    :func:`validate_quotes` (inside the provider's own parsing), so a
    malformed response never overwrites a good committed fixture.
    """
    source = get_provider(provider)
    snapshot = source.fetch(underlying)
    if out_path is not None:
        snapshot.to_json_file(out_path)
    return snapshot
