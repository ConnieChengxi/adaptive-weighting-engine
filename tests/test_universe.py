from adaptive_weighting.data.universe import BENCHMARK, DEFAULT_ETFS, VIX_SYMBOL


def test_default_universe_is_not_empty() -> None:
    assert DEFAULT_ETFS
    assert BENCHMARK == "SPY"
    assert VIX_SYMBOL == "^VIX"
