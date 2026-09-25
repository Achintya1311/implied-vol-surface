"""Chain fetcher and fixture-capture tests. All offline: no network, no keys."""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from volsurface.chain import (
    CHAIN_COLUMNS,
    ChainSnapshot,
    FixtureProvider,
    NSEChainProvider,
    ProviderError,
    RateLimited,
    YFinanceChainProvider,
    capture_snapshot,
    get_provider,
    parse_nse_payload,
    validate_quotes,
)


def _quotes_df(**overrides) -> pd.DataFrame:
    row = {
        "expiry": "2026-10-29",
        "strike": 1400.0,
        "option_type": "CE",
        "last_price": 42.0,
        "bid": 41.0,
        "ask": 43.0,
        "open_interest": 1000,
        "volume": 500,
    }
    row.update(overrides)
    return pd.DataFrame([row])[CHAIN_COLUMNS]


def _nse_payload(**overrides) -> dict:
    payload = {
        "records": {
            "underlyingValue": 1400.5,
            "timestamp": "24-Sep-2026 15:30:00",
            "expiryDates": ["29-Sep-2026", "29-Oct-2026"],
            "data": [
                {
                    "strikePrice": 1400,
                    "expiryDate": "29-Sep-2026",
                    "CE": {
                        "lastPrice": 42.1,
                        "bidprice": 41.0,
                        "askPrice": 43.0,
                        "openInterest": 1200,
                        "totalTradedVolume": 600,
                    },
                    "PE": {
                        "lastPrice": 38.5,
                        "bidprice": 37.5,
                        "askPrice": 39.5,
                        "openInterest": 900,
                        "totalTradedVolume": 400,
                    },
                }
            ],
        }
    }
    payload.update(overrides)
    return payload


class TestValidateQuotes:
    def test_valid_frame_passes(self):
        validate_quotes(_quotes_df())

    def test_missing_columns_fail_loudly(self):
        df = _quotes_df().drop(columns=["ask"])
        with pytest.raises(ProviderError, match="missing columns"):
            validate_quotes(df)

    def test_empty_frame_is_an_error(self):
        with pytest.raises(ProviderError, match="no rows"):
            validate_quotes(pd.DataFrame(columns=CHAIN_COLUMNS))

    def test_unexpected_option_type_rejected(self):
        df = _quotes_df(option_type="XX")
        with pytest.raises(ProviderError, match="option_type"):
            validate_quotes(df)

    def test_non_positive_strike_rejected(self):
        df = _quotes_df(strike=0.0)
        with pytest.raises(ProviderError, match="strike"):
            validate_quotes(df)

    def test_negative_open_interest_rejected(self):
        df = _quotes_df(open_interest=-5)
        with pytest.raises(ProviderError, match="open_interest"):
            validate_quotes(df)

    def test_duplicate_contract_rejected(self):
        df = pd.concat([_quotes_df(), _quotes_df()], ignore_index=True)
        with pytest.raises(ProviderError, match="duplicate"):
            validate_quotes(df)


class TestChainSnapshotRoundTrip:
    def test_to_dict_from_dict(self):
        snap = ChainSnapshot(
            underlying="RELIANCE", spot=1400.5, timestamp="2026-09-24T15:30:00",
            source="fixture", quotes=_quotes_df(),
        )
        restored = ChainSnapshot.from_dict(snap.to_dict())
        assert restored.underlying == snap.underlying
        assert restored.spot == snap.spot
        pd.testing.assert_frame_equal(
            restored.quotes.reset_index(drop=True), snap.quotes.reset_index(drop=True)
        )

    def test_json_file_round_trip(self, tmp_path):
        snap = ChainSnapshot(
            underlying="RELIANCE", spot=1400.5, timestamp="2026-09-24T15:30:00",
            source="fixture", quotes=_quotes_df(),
        )
        path = tmp_path / "RELIANCE.json"
        snap.to_json_file(path)
        restored = ChainSnapshot.from_json_file(path)
        assert restored.spot == 1400.5
        assert restored.source == "fixture"

    def test_from_dict_missing_keys_fails_loudly(self):
        with pytest.raises(ProviderError, match="missing keys"):
            ChainSnapshot.from_dict({"underlying": "X"})

    def test_from_json_file_missing_path(self, tmp_path):
        with pytest.raises(ProviderError, match="no snapshot file"):
            ChainSnapshot.from_json_file(tmp_path / "nope.json")


class TestParseNsePayload:
    def test_parses_spot_timestamp_and_both_legs(self):
        snap = parse_nse_payload(_nse_payload(), "RELIANCE", source="nse")
        assert snap.underlying == "RELIANCE"
        assert snap.spot == pytest.approx(1400.5)
        assert snap.timestamp == "2026-09-24T15:30:00"
        assert len(snap.quotes) == 2
        assert set(snap.quotes["option_type"]) == {"CE", "PE"}
        assert snap.quotes["expiry"].iloc[0] == "2026-09-29"

    def test_missing_records_key_fails_loudly(self):
        with pytest.raises(ProviderError, match="no 'records' key"):
            parse_nse_payload({}, "RELIANCE", source="nse")

    def test_zero_rows_fails_loudly(self):
        payload = _nse_payload()
        payload["records"]["data"] = []
        with pytest.raises(ProviderError, match="zero option rows"):
            parse_nse_payload(payload, "RELIANCE", source="nse")

    def test_unparseable_timestamp_falls_back_to_raw_string(self):
        payload = _nse_payload()
        payload["records"]["timestamp"] = "not-a-timestamp"
        snap = parse_nse_payload(payload, "RELIANCE", source="nse")
        assert snap.timestamp == "not-a-timestamp"

    def test_entry_missing_a_leg_only_yields_the_present_one(self):
        payload = _nse_payload()
        del payload["records"]["data"][0]["PE"]
        snap = parse_nse_payload(payload, "RELIANCE", source="nse")
        assert set(snap.quotes["option_type"]) == {"CE"}


class TestFixtureProvider:
    def test_available_lists_committed_fixtures(self):
        assert "RELIANCE" in FixtureProvider().available()

    def test_fetch_returns_valid_snapshot(self):
        snap = FixtureProvider().fetch("RELIANCE")
        assert snap.source == "fixture" or snap.source  # source is whatever captured it
        assert snap.spot > 0
        assert list(snap.quotes.columns) == CHAIN_COLUMNS
        assert len(snap.quotes) > 0

    def test_unknown_underlying_names_the_alternatives(self):
        with pytest.raises(ProviderError, match="available:"):
            FixtureProvider().fetch("NOTATICKER")

    def test_fetch_is_case_insensitive(self):
        assert FixtureProvider().fetch("reliance").underlying == "RELIANCE"


class TestGetProvider:
    def test_known_providers_resolve(self):
        assert isinstance(get_provider("fixture"), FixtureProvider)
        assert isinstance(get_provider("nse"), NSEChainProvider)
        assert isinstance(get_provider("yfinance"), YFinanceChainProvider)

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="unknown provider"):
            get_provider("bloomberg")


class TestNSEChainProviderOffline:
    """Network calls are monkeypatched out - this only proves the plumbing."""

    def test_non_429_transport_failure_wraps_as_provider_error(self, monkeypatch):
        class FakeSession:
            headers = types.SimpleNamespace(update=lambda *_: None)

            def get(self, *args, **kwargs):
                raise OSError("connection refused")

        import requests

        monkeypatch.setattr(requests, "Session", lambda: FakeSession())
        with pytest.raises(ProviderError, match="NSE fetch failed"):
            NSEChainProvider(pause=0).fetch("RELIANCE")

    def test_rate_limit_on_warmup_raises_rate_limited(self, monkeypatch):
        class FakeResponse:
            status_code = 429

        class FakeSession:
            headers = types.SimpleNamespace(update=lambda *_: None)

            def get(self, *args, **kwargs):
                return FakeResponse()

        import requests

        monkeypatch.setattr(requests, "Session", lambda: FakeSession())
        with pytest.raises(RateLimited):
            NSEChainProvider(pause=0).fetch("RELIANCE")


class TestYFinanceChainProviderOffline:
    """Injects a fake yfinance module so this runs without the real package."""

    def test_empty_expiries_fails_loudly(self, monkeypatch):
        fake_ticker = types.SimpleNamespace(options=[])
        fake_module = types.SimpleNamespace(Ticker=lambda symbol: fake_ticker)
        monkeypatch.setitem(sys.modules, "yfinance", fake_module)
        with pytest.raises(ProviderError, match="no option expiries"):
            YFinanceChainProvider().fetch("RELIANCE")

    def test_parses_calls_and_puts(self, monkeypatch):
        calls = pd.DataFrame(
            [{"strike": 1400.0, "lastPrice": 42.0, "bid": 41.0, "ask": 43.0,
              "openInterest": 100, "volume": 50}]
        )
        puts = pd.DataFrame(
            [{"strike": 1400.0, "lastPrice": 38.0, "bid": 37.0, "ask": 39.0,
              "openInterest": 80, "volume": 30}]
        )
        fake_chain = types.SimpleNamespace(calls=calls, puts=puts)
        fake_ticker = types.SimpleNamespace(
            options=["2026-10-29"],
            option_chain=lambda expiry: fake_chain,
            fast_info={"lastPrice": 1401.0},
        )
        fake_module = types.SimpleNamespace(Ticker=lambda symbol: fake_ticker)
        monkeypatch.setitem(sys.modules, "yfinance", fake_module)

        snap = YFinanceChainProvider().fetch("RELIANCE")
        assert snap.spot == pytest.approx(1401.0)
        assert set(snap.quotes["option_type"]) == {"CE", "PE"}


class TestCaptureSnapshot:
    def test_writes_fixture_shaped_file(self, tmp_path):
        out = tmp_path / "RELIANCE.json"
        snap = capture_snapshot("RELIANCE", provider="fixture", out_path=out)
        assert out.exists()
        reloaded = ChainSnapshot.from_json_file(out)
        assert reloaded.underlying == snap.underlying
        assert reloaded.spot == snap.spot

    def test_no_out_path_does_not_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        snap = capture_snapshot("RELIANCE", provider="fixture", out_path=None)
        assert snap.underlying == "RELIANCE"
        assert list(tmp_path.iterdir()) == []
