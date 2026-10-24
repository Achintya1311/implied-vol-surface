"""Chain fetcher and fixture-capture tests. All offline: no network, no keys."""

from __future__ import annotations

import json
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
    RawCaptureProvider,
    YFinanceChainProvider,
    capture_snapshot,
    get_provider,
    merge_nse_payloads,
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
    """NSE's real option-chain-v3 wire shape (one expiry per call, confirmed
    against the home-connection captures in ``fixtures/chains/raw/``): the
    per-strike entry carries the expiry under the plural ``expiryDates`` key
    (``'DD-Mon-YYYY'``), and each leg's bid/ask are ``buyPrice1``/``sellPrice1``,
    not the retired v2 names ``bidprice``/``askPrice`` this helper assumed
    before Day 10 (D9) - a guess made with no real NSE response to check it
    against, and wrong on both counts.
    """
    payload = {
        "records": {
            "underlyingValue": 1400.5,
            "timestamp": "24-Sep-2026 15:30:00",
            "expiryDates": ["29-Sep-2026", "29-Oct-2026"],
            "data": [
                {
                    "strikePrice": 1400,
                    "expiryDates": "29-Sep-2026",
                    "CE": {
                        "lastPrice": 42.1,
                        "buyPrice1": 41.0,
                        "sellPrice1": 43.0,
                        "openInterest": 1200,
                        "totalTradedVolume": 600,
                    },
                    "PE": {
                        "lastPrice": 38.5,
                        "buyPrice1": 37.5,
                        "sellPrice1": 39.5,
                        "openInterest": 900,
                        "totalTradedVolume": 400,
                    },
                }
            ],
        }
    }
    payload.update(overrides)
    return payload


def _trimmed_real_v3_payload() -> dict:
    """Two rows copied verbatim from a real home-connection capture
    (``fixtures/chains/raw/RELIANCE/2026-10-16T2319_27-Oct-2026.json``,
    ``source: .../option-chain-v3?type=Equity&symbol=RELIANCE&expiry=27-Oct-2026``),
    trimmed to the fields the parser reads. Grounds the parser test in an
    actual NSE response shape, not a hand-guessed one.
    """
    return {
        "records": {
            "underlyingValue": 1211.2,
            "timestamp": "16-Oct-2026 15:40:00",
            "expiryDates": ["27-Oct-2026"],
            "data": [
                {
                    "expiryDates": "27-Oct-2026",
                    "strikePrice": 980,
                    "CE": {
                        "lastPrice": 0,
                        "buyPrice1": 221.45,
                        "sellPrice1": 236.65,
                        "openInterest": 0,
                        "totalTradedVolume": 0,
                    },
                    "PE": {
                        "lastPrice": 0.15,
                        "buyPrice1": 0.1,
                        "sellPrice1": 0.15,
                        "openInterest": 211,
                        "totalTradedVolume": 21,
                    },
                },
                {
                    "expiryDates": "27-Oct-2026",
                    "strikePrice": 1000,
                    "CE": {
                        "lastPrice": 211,
                        "buyPrice1": 204.65,
                        "sellPrice1": 210.7,
                        "openInterest": 161,
                        "totalTradedVolume": 48,
                    },
                    "PE": {
                        "lastPrice": 0.25,
                        "buyPrice1": 0.2,
                        "sellPrice1": 0.3,
                        "openInterest": 468,
                        "totalTradedVolume": 133,
                    },
                },
            ],
        }
    }


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

    def test_parses_a_trimmed_real_v3_capture(self):
        """D9/D1 (Day 10): against an actual home-connection v3 capture,
        not a guessed shape - this is the regression test for the bug
        (entry-level ``expiryDate`` singular and leg-level
        ``bidprice``/``askPrice`` are not the real NSE wire fields).
        """
        snap = parse_nse_payload(_trimmed_real_v3_payload(), "RELIANCE", source="nse")
        assert snap.spot == pytest.approx(1211.2)
        assert len(snap.quotes) == 4  # 2 strikes x 2 legs
        assert set(snap.quotes["expiry"]) == {"2026-10-27"}
        ce_980 = snap.quotes[
            (snap.quotes["strike"] == 980) & (snap.quotes["option_type"] == "CE")
        ].iloc[0]
        assert ce_980["bid"] == pytest.approx(221.45)
        assert ce_980["ask"] == pytest.approx(236.65)
        pe_1000 = snap.quotes[
            (snap.quotes["strike"] == 1000) & (snap.quotes["option_type"] == "PE")
        ].iloc[0]
        assert pe_1000["bid"] == pytest.approx(0.2)
        assert pe_1000["ask"] == pytest.approx(0.3)


class TestMergeNsePayloads:
    def test_merges_data_rows_across_expiries(self):
        p1 = _trimmed_real_v3_payload()
        p2 = _nse_payload()
        merged = merge_nse_payloads([p1, p2])
        assert len(merged["records"]["data"]) == 3  # 2 from p1 + 1 from p2
        assert merged["records"]["underlyingValue"] == pytest.approx(1211.2)

    def test_empty_list_rejected(self):
        with pytest.raises(ProviderError, match="no payloads"):
            merge_nse_payloads([])

    def test_no_usable_underlying_value_rejected(self):
        with pytest.raises(ProviderError, match="underlyingValue"):
            merge_nse_payloads([{"records": {"data": []}}])


class TestRawCaptureProvider:
    """D1/D9 (Day 10): the home-connection job's real NSE v3 captures,
    one file per expiry, merged into one chain through the same
    parse_nse_payload() a live fetch would use.
    """

    def _write_raw(
        self, tmp_path, symbol: str, expiry: str, payload: dict, stamp: str = "2026-10-16T2319"
    ):
        out = tmp_path / symbol / f"{stamp}_{expiry}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "captured_at": f"{stamp}:28.711585",
                    "source": f"https://www.nseindia.com/api/option-chain-v3?symbol={symbol}&expiry={expiry}",
                    "payload": payload,
                }
            )
        )

    def test_merges_every_expiry_file_for_the_symbol(self, tmp_path):
        self._write_raw(tmp_path, "RELIANCE", "27-Oct-2026", _trimmed_real_v3_payload())
        other = _nse_payload()
        other["records"]["data"][0]["expiryDates"] = "23-Nov-2026"
        self._write_raw(tmp_path, "RELIANCE", "23-Nov-2026", other)

        snap = RawCaptureProvider(directory=tmp_path).fetch("RELIANCE")
        assert snap.underlying == "RELIANCE"
        assert len(snap.quotes) == 6  # 4 rows from the trimmed capture + 2 from `other`
        assert set(snap.quotes["expiry"]) == {"2026-10-27", "2026-11-23"}

    def test_a_later_recapture_of_the_same_expiry_replaces_the_earlier_one(self, tmp_path):
        """D25: once a second batch lands for an expiry an earlier batch
        already captured (expected once the corpus grows past one
        snapshot), merging both would duplicate every (expiry, strike,
        option_type) row and fail validate_quotes -- reproduced live on
        GitHub's runner the day this path was added. Only the latest
        capture per expiry should be used.
        """
        old = _trimmed_real_v3_payload()
        self._write_raw(tmp_path, "RELIANCE", "27-Oct-2026", old, stamp="2026-10-16T2319")

        newer = _trimmed_real_v3_payload()
        newer["records"]["underlyingValue"] = 1190.0
        for entry in newer["records"]["data"]:
            entry["CE"]["lastPrice"] = 999.0
        self._write_raw(tmp_path, "RELIANCE", "27-Oct-2026", newer, stamp="2026-10-24T0225")

        snap = RawCaptureProvider(directory=tmp_path).fetch("RELIANCE")
        assert len(snap.quotes) == 4  # not 8 -- the older recapture is dropped, not merged
        ce_980 = snap.quotes[
            (snap.quotes["strike"] == 980) & (snap.quotes["option_type"] == "CE")
        ].iloc[0]
        assert ce_980["last_price"] == pytest.approx(999.0)

    def test_unknown_symbol_names_the_alternatives(self, tmp_path):
        self._write_raw(tmp_path, "NIFTY", "19-Oct-2026", _trimmed_real_v3_payload())
        with pytest.raises(ProviderError, match="available:"):
            RawCaptureProvider(directory=tmp_path).fetch("RELIANCE")

    def test_no_capture_files_in_the_symbol_dir_is_an_error(self, tmp_path):
        (tmp_path / "RELIANCE").mkdir(parents=True)
        with pytest.raises(ProviderError, match="no raw capture files"):
            RawCaptureProvider(directory=tmp_path).fetch("RELIANCE")

    def test_available_lists_captured_symbols(self, tmp_path):
        self._write_raw(tmp_path, "NIFTY", "19-Oct-2026", _trimmed_real_v3_payload())
        self._write_raw(tmp_path, "RELIANCE", "27-Oct-2026", _trimmed_real_v3_payload())
        assert RawCaptureProvider(directory=tmp_path).available() == ["NIFTY", "RELIANCE"]

    def test_fetch_is_case_insensitive(self, tmp_path):
        self._write_raw(tmp_path, "RELIANCE", "27-Oct-2026", _trimmed_real_v3_payload())
        assert RawCaptureProvider(directory=tmp_path).fetch("reliance").underlying == "RELIANCE"

    def test_against_the_actual_committed_raw_captures(self):
        """No tmp_path fixture here - this reads the real committed
        fixtures/chains/raw/ directory, proving the parser handles every
        real payload the home-connection job has landed, not just a
        synthetic stand-in."""
        provider = RawCaptureProvider()
        assert {"NIFTY", "RELIANCE"} <= set(provider.available())
        for symbol in ("NIFTY", "RELIANCE"):
            snap = provider.fetch(symbol)
            assert snap.spot > 0
            assert len(snap.quotes) > 0
            assert len(set(snap.quotes["expiry"])) == 3  # 3 captured expiries each


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
        assert isinstance(get_provider("nse_raw"), RawCaptureProvider)
        assert isinstance(get_provider("yfinance"), YFinanceChainProvider)

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="unknown provider"):
            get_provider("bloomberg")


class TestNSEChainProviderOffline:
    """Network calls are monkeypatched out - this only proves the plumbing.

    D9/Day 10: the live path now takes two calls to list/choose expiries
    (``option-chain-contract-info``, ``option-chain-v3`` once per expiry)
    rather than the retired v2 API's one-call-per-symbol shape, since the
    v3 endpoint returns one expiry per request.
    """

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

    def test_rate_limit_on_contract_info_raises_rate_limited(self, monkeypatch):
        class FakeResponse:
            def __init__(self, status_code):
                self.status_code = status_code

        class FakeSession:
            headers = types.SimpleNamespace(update=lambda *_: None)
            calls = 0

            def get(self, *args, **kwargs):
                self.calls += 1
                # call 1: warmup (200), call 2: contract-info (429)
                return FakeResponse(200 if self.calls == 1 else 429)

        import requests

        monkeypatch.setattr(requests, "Session", lambda: FakeSession())
        with pytest.raises(RateLimited, match="expiry list"):
            NSEChainProvider(pause=0).fetch("RELIANCE")

    def test_fetches_contract_info_then_one_v3_call_per_expiry_and_merges(self, monkeypatch):
        calls = []

        class FakeResponse:
            def __init__(self, payload):
                self.status_code = 200
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class FakeSession:
            headers = types.SimpleNamespace(update=lambda *_: None)

            def get(self, url, params=None, timeout=None):
                calls.append((url, params))
                if "contract-info" in url:
                    return FakeResponse({"expiryDates": ["27-Oct-2026", "23-Nov-2026"]})
                if "option-chain-v3" in url:
                    payload = _trimmed_real_v3_payload()
                    payload["records"]["data"][0]["expiryDates"] = params["expiry"]
                    payload["records"]["data"][1]["expiryDates"] = params["expiry"]
                    return FakeResponse(payload)
                return FakeResponse({})  # the /option-chain warm-up GET

        import requests

        monkeypatch.setattr(requests, "Session", lambda: FakeSession())
        snap = NSEChainProvider(pause=0).fetch("RELIANCE")

        v3_calls = [c for c in calls if "option-chain-v3" in c[0]]
        assert len(v3_calls) == 2
        assert all(c[1]["type"] == "Equity" for c in v3_calls)
        assert {c[1]["expiry"] for c in v3_calls} == {"27-Oct-2026", "23-Nov-2026"}
        assert len(snap.quotes) == 8  # 2 expiries x 2 strikes x 2 legs

    def test_raw_capture_dir_writes_one_file_per_expiry_in_the_home_job_shape(
        self, tmp_path, monkeypatch
    ):
        """D25: the per-expiry raw files this provider writes, when asked to,
        must be readable by :class:`RawCaptureProvider` exactly like
        ``automation/capture_chains.py``'s home-connection captures already
        committed under ``fixtures/chains/raw/`` -- same ``{captured_at,
        source, payload}`` shape, same ``<stamp>_<expiry>.json`` naming.
        """

        class FakeResponse:
            def __init__(self, payload, url):
                self.status_code = 200
                self._payload = payload
                self.url = url

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class FakeSession:
            headers = types.SimpleNamespace(update=lambda *_: None)

            def get(self, url, params=None, timeout=None):
                if "contract-info" in url:
                    return FakeResponse({"expiryDates": ["27-Oct-2026", "23-Nov-2026"]}, url)
                if "option-chain-v3" in url:
                    payload = _trimmed_real_v3_payload()
                    payload["records"]["data"][0]["expiryDates"] = params["expiry"]
                    payload["records"]["data"][1]["expiryDates"] = params["expiry"]
                    full_url = f"{url}?type={params['type']}&symbol={params['symbol']}&expiry={params['expiry']}"
                    return FakeResponse(payload, full_url)
                return FakeResponse({}, url)  # the /option-chain warm-up GET

        import requests

        monkeypatch.setattr(requests, "Session", lambda: FakeSession())
        raw_dir = tmp_path / "raw"
        NSEChainProvider(pause=0, raw_capture_dir=raw_dir).fetch("RELIANCE")

        written = sorted((raw_dir / "RELIANCE").glob("*.json"))
        assert len(written) == 2
        names = {p.name.split("_", 1)[1] for p in written}
        assert names == {"27-Oct-2026.json", "23-Nov-2026.json"}

        raw = json.loads(written[0].read_text())
        assert set(raw) == {"captured_at", "source", "payload"}
        assert "option-chain-v3" in raw["source"]
        assert raw["payload"]["records"]["data"]

        # RawCaptureProvider must be able to read these straight back.
        snap = RawCaptureProvider(directory=raw_dir).fetch("RELIANCE")
        assert len(snap.quotes) == 8  # 2 expiries x 2 strikes x 2 legs

    def test_no_raw_capture_dir_writes_nothing(self, tmp_path, monkeypatch):
        class FakeResponse:
            def __init__(self, payload):
                self.status_code = 200
                self._payload = payload
                self.url = "https://www.nseindia.com/api/option-chain-v3"

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class FakeSession:
            headers = types.SimpleNamespace(update=lambda *_: None)

            def get(self, url, params=None, timeout=None):
                if "contract-info" in url:
                    return FakeResponse({"expiryDates": ["27-Oct-2026"]})
                if "option-chain-v3" in url:
                    payload = _trimmed_real_v3_payload()
                    payload["records"]["data"][0]["expiryDates"] = params["expiry"]
                    payload["records"]["data"][1]["expiryDates"] = params["expiry"]
                    return FakeResponse(payload)
                return FakeResponse({})

        import requests

        monkeypatch.setattr(requests, "Session", lambda: FakeSession())
        NSEChainProvider(pause=0).fetch("RELIANCE")
        assert not (tmp_path / "raw").exists()

    def test_index_symbol_requests_the_indices_type(self, monkeypatch):
        seen_types = []

        class FakeResponse:
            def __init__(self, payload):
                self.status_code = 200
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class FakeSession:
            headers = types.SimpleNamespace(update=lambda *_: None)

            def get(self, url, params=None, timeout=None):
                if "contract-info" in url:
                    return FakeResponse({"expiryDates": ["19-Oct-2026"]})
                if "option-chain-v3" in url:
                    seen_types.append(params["type"])
                    return FakeResponse(_trimmed_real_v3_payload())
                return FakeResponse({})

        import requests

        monkeypatch.setattr(requests, "Session", lambda: FakeSession())
        NSEChainProvider(pause=0).fetch("NIFTY")
        assert seen_types == ["Indices"]

    def test_no_expiries_from_contract_info_fails_loudly(self, monkeypatch):
        class FakeResponse:
            def __init__(self, payload):
                self.status_code = 200
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class FakeSession:
            headers = types.SimpleNamespace(update=lambda *_: None)

            def get(self, url, params=None, timeout=None):
                if "contract-info" in url:
                    return FakeResponse({"expiryDates": []})
                return FakeResponse({})

        import requests

        monkeypatch.setattr(requests, "Session", lambda: FakeSession())
        with pytest.raises(ProviderError, match="no expiries"):
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
