from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import generate_liquidity_diagnostics as liquidity_diag
from scripts.generate_liquidity_cost_exploratory import (
    TABLE_I1_PATH,
    TABLE_I2_PATH,
    build_appendix_i_frames,
    build_table_i1,
    build_table_i2,
    main,
)


def test_appendix_i_lookup_is_complete() -> None:
    lookup = liquidity_diag.load_appendix_i_liquidity_lookup()
    assert not lookup.duplicated(subset=["Date", "symbol"]).any()
    assert lookup.groupby("Date")["symbol"].nunique().eq(9).all()
    assert not lookup[["liquidity_1m", "corwin_schultz_spread"]].isna().any().any()


def test_traded_leg_builder_does_not_depend_on_monthly_factor_panel(monkeypatch) -> None:
    monkeypatch.setattr(
        liquidity_diag,
        "MONTHLY_PANEL_PATH",
        Path("/tmp/appendix_i_unused_monthly_panel.csv"),
    )
    frame = liquidity_diag.build_traded_leg_linkage_frame_for_source("main_result")
    assert len(frame) > 0
    assert {"amihud", "log10_amihud", "effective_cost_bps", "execution_setting"} <= set(frame.columns)
    assert frame["execution_setting"].eq("F3").all()


def test_appendix_i_summary_matches_current_acceptance_values() -> None:
    f1, f3 = build_appendix_i_frames()
    assert len(f1) == 1402
    assert f1["Date"].nunique() == 274
    assert len(f3) == 366
    assert f3["Date"].nunique() == 130

    table_i1 = build_table_i1(f1, f3)
    f1_row = table_i1.loc[table_i1["Setting"] == "F1"].iloc[0]
    f3_row = table_i1.loc[table_i1["Setting"] == "F3"].iloc[0]

    assert float(f1_row["Spearman rho"]) == 0.250
    assert float(f1_row["Beta"]) == 2.322
    assert f1_row["Clustered p"] == "<0.001"
    assert float(f1_row["R-squared"]) == 0.063

    assert float(f3_row["Spearman rho"]) == 0.159
    assert float(f3_row["Beta"]) == 1.462
    assert f3_row["Clustered p"] == "0.117"
    assert float(f3_row["R-squared"]) == 0.031

    table_i2 = build_table_i2(f3)
    pooled = table_i2.loc[table_i2["Model"] == "Pooled"].iloc[0]
    t2 = table_i2.loc[table_i2["Model"] == "T2"].iloc[0]
    assert int(pooled["N"]) == 366
    assert float(pooled["Mean unit cost"]) == 15.885
    assert float(t2["Mean unit cost"]) == 16.167


def test_appendix_i_main_writes_current_outputs() -> None:
    main()
    assert TABLE_I1_PATH.exists()
    assert TABLE_I2_PATH.exists()

    table_i1 = pd.read_csv(TABLE_I1_PATH)
    table_i2 = pd.read_csv(TABLE_I2_PATH)
    assert list(table_i1["Setting"]) == ["F1", "F3"]
    assert list(table_i2["Model"]) == ["Pooled", "S1", "A1", "L1", "L2", "L3", "T1", "T2"]
