# app.py
from __future__ import annotations

import io
import random
from datetime import datetime
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.api import OLS, add_constant
from sklearn.linear_model import LinearRegression


# --------------------------------------------------------------------------------------
# Streamlit pagina-config
# --------------------------------------------------------------------------------------
try:
    st.set_page_config(page_title="Verwachte sterfte • RIVM-stijl", layout="wide")
except Exception:
    # Voorkom fout bij gedeelde omgevingen die set_page_config al hebben aangeroepen
    pass


# --------------------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------------------
DEFAULT_INPUT_URL = (
    "https://raw.githubusercontent.com/rcsmit/COVIDcases/refs/heads/main/input/"
    "overlijdens_per_week_2014_2025.csv"
)
RIVM_OFFICIAL_URL = (
    "https://raw.githubusercontent.com/rcsmit/COVIDcases/refs/heads/main/input/"
    "rivm_official_290825.csv"
)
RIVM_STERFTE_URL = (
    "https://raw.githubusercontent.com/rcsmit/COVIDcases/refs/heads/main/input/"
    "rivm_sterfte.csv"
)

TIMELINE_START = pd.Timestamp("2019-07-01")
TIMELINE_END = pd.Timestamp("2026-06-30")


# --------------------------------------------------------------------------------------
# Data & feature engineering
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data_from_url(url: str = DEFAULT_INPUT_URL) -> pd.DataFrame:
    """Lees overlijdensdata in en verrijk met datum, seizoensjaar, harmonischen en tijdindex.

    Verwacht CSV met kolommen: 'jaar';'week';'overleden' (scheidingsteken ';').

    Parameters
    ----------
    url : str
        HTTP(S)-pad naar CSV.

    Returns
    -------
    pd.DataFrame
        Gesorteerde tijdreeks met extra kolommen:
        - date (Timestamp)
        - season_year (int)
        - month (int)
        - t (int) oplopende index
        - sin1, cos1, sin2, cos2 (harmonischen)
        - season_week_idx (1..52; W27..W52 + W01..W26)
    """
    df = pd.read_csv(url, sep=";", dtype={"jaar": int, "week": int, "overleden": float})
    required = {"jaar", "week", "overleden"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        raise ValueError(f"Ontbrekende kolommen: {missing}")

    # Normaliseer kolomnamen
    cols = {c.lower(): c for c in df.columns}
    df = df.rename(
        columns={
            cols["jaar"]: "jaar",
            cols["week"]: "week",
            cols["overleden"]: "overleden",
        }
    )

    # ISO-week naar datum (maandag)
    df["date"] = pd.to_datetime(
        df["jaar"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )

    # Seizoensjaar: week 27..52 → jaar+1; week 1..26 → jaar
    df["season_year"] = np.where(df["week"] >= 27, df["jaar"] + 1, df["jaar"])
    df["month"] = df["date"].dt.month
    df = df.sort_values("date").reset_index(drop=True)

    # Tijdindex
    df["t"] = np.arange(len(df))

    # Harmonische termen
    add_harmonics_inplace(df, week_col="week")

    # Positie 1..52 in seizoensjaar: 27..52 → 1..26, 1..26 → 27..52
    df["season_week_idx"] = np.where(df["week"] >= 27, df["week"] - 26, df["week"] + 26)

    return df


def add_harmonics_inplace(frame: pd.DataFrame, week_col: str = "week") -> None:
    """Voeg 1e en 2e harmonische toe op basis van weeknummer.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame met een week-kolom (1..53).
    week_col : str
        Naam van de week-kolom.
    """
    week_frac = (frame[week_col] - 1) / 52.0
    frame["sin1"] = np.sin(2 * np.pi * week_frac)
    frame["cos1"] = np.cos(2 * np.pi * week_frac)
    frame["sin2"] = np.sin(4 * np.pi * week_frac)
    frame["cos2"] = np.cos(4 * np.pi * week_frac)


def build_season_frame(season_year: int, base_df: pd.DataFrame) -> pd.DataFrame:
    """Maak alle 52 weken van een seizoensjaar expliciet met juiste kalenderjaren en datum.

    Week 27..52 → kalenderjaar = season_year-1
    Week 1..26  → kalenderjaar = season_year

    Parameters
    ----------
    season_year : int
        Doel seizoensjaar (bijv. 2026 voor W27-2025..W26-2026).
    base_df : pd.DataFrame
        Bron met bestaande 't'-index voor eventuele koppeling.

    Returns
    -------
    pd.DataFrame
        Frame met kolommen: jaar, week, season_year, date, t, sin/cos-harmonics.
    """
    season_weeks = list(range(27, 53)) + list(range(1, 27))
    rows = []
    for w in season_weeks:
        jaar = season_year - 1 if w >= 27 else season_year
        date = pd.to_datetime(f"{jaar}-W{w:02d}-1", format="%G-W%V-%u", errors="coerce")
        rows.append({"jaar": jaar, "week": w, "season_year": season_year, "date": date})
    out = pd.DataFrame(rows)

    # t koppelen of doortrekken
    df_map = base_df.set_index(["jaar", "week"])
    out = out.join(df_map[["t"]], on=["jaar", "week"])
    if out["t"].isna().any():
        max_t = base_df["t"].max()
        miss_mask = out["t"].isna()
        out.loc[miss_mask, "t"] = np.arange(max_t + 1, max_t + 1 + miss_mask.sum())
    out["t"] = out["t"].astype(int)

    # Harmonischen
    add_harmonics_inplace(out, week_col="week")
    return out


def train_exclusion_mask(train: pd.DataFrame, q_all: float, q_summer: float) -> pd.Series:
    """Bepaal masker om piekweken uit te sluiten.

    Parameters
    ----------
    train : pd.DataFrame
        Trainingsdata met kolommen 'overleden' en 'month'.
    q_all : float
        Kwantiel voor alle weken, bv. 0.75 → hoogste 25% weg.
    q_summer : float
        Kwantiel voor juli/augustus, bv. 0.80 → hoogste 20% weg.

    Returns
    -------
    pd.Series
        Boolean masker met True = behouden.
    """
    q75_all_val = train["overleden"].quantile(q_all)
    mask_all = train["overleden"] <= q75_all_val

    is_jul_aug = train["month"].isin([7, 8])
    if is_jul_aug.any():
        q_summer_val = train.loc[is_jul_aug, "overleden"].quantile(q_summer)
        mask_summer = (~is_jul_aug) | (train["overleden"] <= q_summer_val)
    else:
        mask_summer = pd.Series(True, index=train.index)

    return mask_all & mask_summer


# --------------------------------------------------------------------------------------
# Modellering
# --------------------------------------------------------------------------------------
def fit_baseline(
    df: pd.DataFrame,
    season_year_target: int,
    q_all: float,
    q_summer: float,
    harmonics: int = 1,
) -> Dict[str, object]:
    """Fit OLS-baseline op 5 voorgaande seizoensjaren met trend + harmonischen.

    Parameters
    ----------
    df : pd.DataFrame
        Volledige tijdreeks met features (t, sin/cos) en label 'overleden'.
    season_year_target : int
        Doel seizoensjaar (bijv. 2026).
    q_all : float
        Kwantiel voor algemene filter (bv. 0.75).
    q_summer : float
        Kwantiel voor juli/augustus (bv. 0.80).
    harmonics : int
        Aantal harmonischen: 1 → sin1,cos1; 2 → ook sin2,cos2.

    Returns
    -------
    dict
        - target_df: DataFrame met baseline, lower, upper, overleden voor doeljaar
        - train_used: DataFrame met gebruikte trainingspunten
        - model_summary: str
        - resid_sd: float
    """
    prev_years = list(range(season_year_target - 5, season_year_target))
    train = df[df["season_year"].isin(prev_years)].copy()
    if train.empty or train["overleden"].isna().all():
        raise ValueError("Onvoldoende trainingsdata")

    mask = train_exclusion_mask(train, q_all=q_all, q_summer=q_summer)
    train_used = train.loc[mask].copy()

    X_cols = ["t", "sin1", "cos1"] if harmonics == 1 else ["t", "sin1", "cos1", "sin2", "cos2"]
    X_train = add_constant(train_used[X_cols])
    y_train = train_used["overleden"].values

    model = OLS(y_train, X_train).fit()

    # Doelframe & voorspellingen
    target = build_season_frame(season_year_target, df)
    X_target = add_constant(target[X_cols])
    target["baseline"] = model.predict(X_target)

    # Onzekerheidsband (±2*sd)
    resid_sd = float(np.sqrt(model.scale))
    target["lower"] = target["baseline"] - 2 * resid_sd
    target["upper"] = target["baseline"] + 2 * resid_sd

    # Werkelijk
    actual_map = df.set_index(["jaar", "week"])["overleden"]
    target["overleden"] = actual_map.reindex(list(zip(target["jaar"], target["week"]))).values

    # Visual: trainingsweken en historische baselines
    hist_preds = _historical_baselines(df, model, X_cols, season_year_target)
    _plot_training_and_hist(train, train_used, target, hist_preds, season_year_target)
    st.write(model.summary())
    return {
        "target_df": target,
        "train_used": train_used,
        "model_summary": model.summary().as_text(),
        "resid_sd": resid_sd,
    }


def _historical_baselines(
    df: pd.DataFrame, model, X_cols: List[str], season_year_target: int
) -> pd.DataFrame:
    """Genereer historische voorspellingen voor 5 voorgaande seizoensjaren."""
    df = df.copy()
    df["season_year"] = np.where(df["week"] <= 26, df["jaar"], df["jaar"] + 1)
    season_years_hist = list(range(season_year_target - 5, season_year_target))
    hist_list = []
    for y in season_years_hist:
        out = build_season_frame(y, df)
        Xp = add_constant(out[X_cols])
        out["baseline"] = model.predict(Xp)
        hist_list.append(out)

    return pd.concat(hist_list, ignore_index=True) if hist_list else pd.DataFrame(
        columns=["date", "baseline", "season_year"]
    )


def _plot_training_and_hist(
    train: pd.DataFrame, train_used: pd.DataFrame, target: pd.DataFrame, hist_pred: pd.DataFrame, season_year_target: int
) -> None:
    """Plot trainingspunten en baselines voor doeljaar en historie."""
    
    fig = go.Figure()

    color_line = "red" if season_year_target == 2026 else "green"
    for y in sorted(hist_pred["season_year"].unique()):
        sub = hist_pred[hist_pred["season_year"] == y]
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["baseline"],
                name=f"voorspeld",
                opacity=1,
                line=dict(width=1, color=color_line),
                #showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(x=train["date"], y=train["overleden"],  mode="markers",name="train")
    )
    fig.add_trace(
        go.Scatter(x=train_used["date"], y=train_used["overleden"],  mode="markers",name="train used")
    )
    fig.add_trace(
        go.Scatter(x=target["date"], y=target["baseline"], name="target", line=dict(color=color_line))
    )

    fig.update_layout(
        title=f"Waardes waarop getraind wordt, doeljaar: {season_year_target}",
        xaxis_title="Datum",
        yaxis_title="Overledenen",
        yaxis=dict(range=[2400, 3600]),
    )
    st.plotly_chart(fig, key=f"fig_train_{random.randint(1, 10_000)}")


# --------------------------------------------------------------------------------------
# Plothelpers
# --------------------------------------------------------------------------------------
def plot_single_season(target_df: pd.DataFrame, season_year_target: int) -> go.Figure:
    """Maak per-seizoensjaarplot met baseline, band en werkelijk."""
    def sort_key(row):
        w = int(row["week"])
        return (0 if w >= 27 else 1, w)

    dfp = target_df.copy()
    dfp["__order"] = dfp.apply(sort_key, axis=1)
    dfp = dfp.sort_values(["__order", "week"]).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dfp["date"], y=dfp["upper"], name="Base + 2×sd", mode="lines", line=dict(width=0), hovertemplate="%{y:.0f}"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dfp["date"],
            y=dfp["lower"],
            name="Base − 2×sd",
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            hovertemplate="%{y:.0f}",
        )
    )
    fig.add_trace(go.Scatter(x=dfp["date"], y=dfp["baseline"], name="Baselijn", mode="lines", hovertemplate="%{y:.0f}"))
    fig.add_trace(
        go.Scatter(x=dfp["date"], y=dfp["overleden"], name="Werkelijk", mode="lines+markers", hovertemplate="%{y:.0f}")
    )

    fig.update_layout(
        title=f"Verwachte sterfte • Seizoensjaar {season_year_target} (week 27 t/m 26)",
        xaxis_title="Week",
        yaxis_title="Overledenen",
        legend_title="Reeks",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_all_seasons(
    df: pd.DataFrame,
    seasons: Iterable[int],
    include_baselines: bool,
    q_all: float,
    q_summer: float,
    harmonics: int,
) -> go.Figure:
    """Toon alle seizoenen als lijnen, optioneel met baselines."""
    fig = go.Figure()
    for sy in seasons:
        sel = df[df["season_year"] == sy].copy()
        if sel.empty:
            continue
        sel = sel.sort_values("season_week_idx")
        fig.add_trace(
            go.Scatter(
                x=sel["season_week_idx"],
                y=sel["overleden"],
                name=f"{sy} werkelijk",
                mode="lines",
                hovertemplate=f"Seizoen {sy}",
            )
        )
        if include_baselines:
            try:
                res = fit_baseline(df, sy, q_all, q_summer, harmonics=harmonics)
                tgt = res["target_df"].sort_values("season_week_idx")
                fig.add_trace(
                    go.Scatter(
                        x=tgt["season_week_idx"],
                        y=tgt["baseline"],
                        name=f"{sy} baselijn",
                        mode="lines",
                        line=dict(dash="dot"),
                        hovertemplate=f"Baselijn {sy} • %{x} • %{y:.0f}",
                    )
                )
            except Exception:
                pass

    idx = list(range(1, 53))
    labels = [f"W{w}" if (w >= 27 and w <= 52) else f"W{w:02d}" for w in (list(range(27, 53)) + list(range(1, 27)))]
    fig.update_layout(
        title="Alle seizoenen 2021–2026 • Werkelijk" + (" + baselines" if include_baselines else ""),
        xaxis=dict(title="Week in seizoensjaar", tickmode="array", tickvals=idx, ticktext=labels),
        yaxis_title="Overledenen",
        legend_title="Reeks",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_timeline_all(
    df: pd.DataFrame, seasons: Iterable[int], q_all: float, q_summer: float, show_bands: bool, harmonics: int
) -> Tuple[go.Figure, pd.DataFrame]:
    """Toon alle seizoenen in één tijdlijn over 2019H2–2026H1. Geeft ook het samengevoegde DataFrame terug."""
    fig = go.Figure()
    tdf_complete = pd.DataFrame()

    for sy in seasons:
        try:
            res = fit_baseline(df, sy, q_all, q_summer, harmonics=harmonics)
        except Exception:
            continue

        tdf = res["target_df"].copy()
        tdf = tdf[(tdf["date"] >= TIMELINE_START) & (tdf["date"] <= TIMELINE_END)]
        if tdf.empty:
            continue

        if show_bands:
            tdf_complete = pd.concat([tdf_complete, tdf])
            fig.add_trace(
                go.Scatter(
                    x=tdf["date"], y=tdf["upper"], name=f"{sy} band boven", mode="lines", line=dict(width=0),
                    legendgroup=f"{sy}", showlegend=False, hovertemplate="%{y:.0f}"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=tdf["date"], y=tdf["lower"], name=f"{sy} band onder", mode="lines", line=dict(width=0),
                    fill="tonexty", legendgroup=f"{sy}", fillcolor=None, showlegend=False, hovertemplate="%{y:.0f}"
                )
            )

        fig.add_trace(
            go.Scatter(
                x=tdf["date"], y=tdf["baseline"], name=f"{sy} baselijn", mode="lines",
                legendgroup=f"{sy}", showlegend=False, hovertemplate="%{y:.0f}"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=tdf["date"], y=tdf["overleden"], name=f"{sy} werkelijk", mode="lines",
                legendgroup=f"{sy}", showlegend=False, hovertemplate="%{y:.0f}"
            )
        )

    fig.update_layout(
        title="Alle seizoenen op één tijdlijn • 2019-H2 t/m 2026-H1",
        xaxis_title="Datum",
        yaxis_title="Overledenen",
        hovermode="x unified",
        legend_title="Reeks",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(range=[TIMELINE_START, TIMELINE_END])

    return fig, tdf_complete


# --------------------------------------------------------------------------------------
# Vergelijking met RIVM
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_rivm_refs() -> pd.DataFrame:
    """Lees RIVM-reeksen (official + sterftebestand) en harmoniseer kolommen."""
    CUTOFF_DATE = "2023-07-03"

    df_off = pd.read_csv(RIVM_OFFICIAL_URL, sep=";", parse_dates=["date"])
    df_st = pd.read_csv(RIVM_STERFTE_URL, sep=";", parse_dates=["datum"])
    df_st = df_st[df_st["datum"] <= pd.to_datetime(CUTOFF_DATE)].copy()

    df_off = df_off.rename(
        columns={
            "aantal_overlijdens": "aantal",
            "ondergrens_verwachting": "ondergrens",
            "verwachting": "verwachting",
            "bovengrens_verwachting": "bovengrens",
        }
    )
    df_off["source"] = "official"
    df_off = df_off[["date", "aantal", "ondergrens", "verwachting", "bovengrens", "source"]]

    df_st = df_st.rename(
        columns={
            "datum": "date",
            "aantal_overlijdens": "aantal",
            "ondergrens_verwachting_rivm": "ondergrens",
            "verw_waarde_rivm": "verwachting",
            "bovengrens_verwachting_rivm": "bovengrens",
        }
    )
    df_st["source"] = "sterfte"
    df_st = df_st[["date", "aantal", "ondergrens", "verwachting", "bovengrens", "source"]]

    return pd.concat([df_off, df_st], ignore_index=True).sort_values("date").reset_index(drop=True)


def plot_model_vs_rivm(tdf_complete: pd.DataFrame) -> None:
    """Maak vergelijkingsplots tussen modelbanden en RIVM-banden."""
    df_rivm = load_rivm_refs()

    df_compleet = (
        tdf_complete.merge(df_rivm, on="date", how="outer").sort_values("date").reset_index(drop=True)
    )

    # Deltas (%)
    for col_model, col_rivm, out in [
        ("upper", "bovengrens", "upper_delta"),
        ("lower", "ondergrens", "lower_delta"),
        ("baseline", "verwachting", "baseline_delta"),
    ]:
        df_compleet[out] = (df_compleet[col_model] - df_compleet[col_rivm]) / df_compleet[col_rivm] * 100

    # Plot: reeksen
    fig_2 = go.Figure()
    # Model (rood/zwart)
    fig_2.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["overleden"], name="Model Overleden", line=dict(color="black")))
    fig_2.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["baseline"], name="Model Baseline", line=dict(color="red")))
    fig_2.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["lower"], name="Model ondergrens", line=dict(color="red", dash="dot")))
    fig_2.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["upper"], name="Model bovengrens", line=dict(color="red", dash="dot")))

    # RIVM (blauw)
    fig_2.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["verwachting"], name="RIVM verwachting", line=dict(color="blue")))
    fig_2.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["ondergrens"], name="RIVM ondergrens", line=dict(color="blue", dash="dash")))
    fig_2.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["bovengrens"], name="RIVM bovengrens", line=dict(color="blue", dash="dash")))

    for year in df_compleet["date"].dt.year.dropna().unique():
        fig_2.add_vline(x=pd.Timestamp(f"{int(year)}-07-01"), line=dict(color="gray", dash="dot"), opacity=0.7)

    fig_2.update_layout(
        title="Model vs RIVM official",
        xaxis_title="Datum",
        yaxis_title="Aantal overlijdens",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_2)

    # Plot: deltas
    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["baseline_delta"], name="baseline (model-rivm)/rivm", line=dict(dash="dot")))
    fig_3.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["lower_delta"], name="lower (model-rivm)/rivm", line=dict(dash="dot")))
    fig_3.add_trace(go.Scatter(x=df_compleet["date"], y=df_compleet["upper_delta"], name="upper (model-rivm)/rivm", line=dict(dash="dot")))
    fig_3.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

    fig_3.update_layout(title="Model - RIVM / rivm *100", xaxis_title="Datum", yaxis_title="Verschil (%)")
    st.plotly_chart(fig_3)


# --------------------------------------------------------------------------------------
# GROK (alternatieve benadering met sklearn LinearRegression)
# --------------------------------------------------------------------------------------
def main_grok(q_all: float, q_summer: float) -> None:
    """Alternatieve implementatie met sklearn, dezelfde filterlogica en visualisatie."""
    csv_path = DEFAULT_INPUT_URL
    df = pd.read_csv(csv_path, sep=";")
    df = df[["jaar", "week", "overleden"]]
    df.sort_values(["jaar", "week"], inplace=True)

    df["time"] = np.arange(len(df))
    df["week_rad"] = 2 * np.pi * df["week"] / 52
    df["sin_week"] = np.sin(df["week_rad"])
    df["cos_week"] = np.cos(df["week_rad"])
    df["date"] = pd.to_datetime(
        df["jaar"].astype(str) + "-" + df["week"].astype(str) + "-1",
        format="%Y-%W-%w",
        errors="coerce",
    )

    def calculate_expected_mortality(
        pred_year: int, q_all_local: float = 0.75, q_summer_local: float = 0.80
    ):
        past_years = range(pred_year - 5, pred_year)
        past_data = df[df["jaar"].isin(past_years)].copy()
        if past_data.empty:
            return None, None, None, None, None, None

        summer_weeks = range(27, 36)
        q_all_val = past_data["overleden"].quantile(q_all_local)
        summer_data = past_data[past_data["week"].isin(summer_weeks)]
        q_summer_val = summer_data["overleden"].quantile(q_summer_local) if not summer_data.empty else np.inf

        is_peak_all = past_data["overleden"] > q_all_val
        is_summer = past_data["week"].isin(summer_weeks)
        is_peak_summer = past_data["overleden"] > q_summer_val
        filtered_data = past_data[~(is_peak_all | (is_summer & is_peak_summer))]
        if len(filtered_data) < 10:
            return None, None, None, None, None, None

        features = ["time", "sin_week", "cos_week"]
        X = filtered_data[features]
        y = filtered_data["overleden"]
        model = LinearRegression().fit(X, y)

        pred_train = model.predict(X)
        residuals = y - pred_train
        rmse = float(np.sqrt(np.mean(residuals**2)))

        future_weeks = list(range(27, 53)) + list(range(1, 27))
        future_years = [pred_year] * 26 + [pred_year + 1] * 26
        future_df = pd.DataFrame({"jaar": future_years, "week": future_weeks})
        future_df["time"] = past_data["time"].max() + 1 + np.arange(len(future_df))
        future_df["week_rad"] = 2 * np.pi * future_df["week"] / 52
        future_df["sin_week"] = np.sin(future_df["week_rad"])
        future_df["cos_week"] = np.cos(future_df["week_rad"])
        future_df["date"] = pd.to_datetime(
            future_df["jaar"].astype(str) + "-" + future_df["week"].astype(str) + "-1",
            format="%Y-%W-%w",
            errors="coerce",
        )

        predicted = model.predict(future_df[features])
        upper = predicted + 2 * rmse
        lower = np.maximum(predicted - 2 * rmse, 0)

        observed_mask = ((df["jaar"] == pred_year) & (df["week"] >= 27)) | (
            (df["jaar"] == pred_year + 1) & (df["week"] <= 26)
        )
        observed_df = df[observed_mask].copy()
        observed = observed_df["overleden"].values if not observed_df.empty else None
        observed_dates = observed_df["date"].values if not observed_df.empty else None

        return future_df["date"].values, predicted, lower, upper, observed, observed_dates

    st.title("Verwachte Sterfte (Week 27, 2019 - Week 26, 2026)")
    fig = go.Figure()
    colors = ["blue", "green", "purple", "orange", "cyan", "magenta", "brown"]
    available_years = range(2019, 2026)
    last_year = max(available_years)

    for idx, pred_year in enumerate(available_years):
        if pred_year < min(df["jaar"]) + 5:
            continue
        dates, baseline, lower, upper, observed, observed_dates = calculate_expected_mortality(
            pred_year, q_all, q_summer
        )
        if baseline is None:
            continue

        mask = (dates >= TIMELINE_START) & (dates <= TIMELINE_END)
        dates = dates[mask]
        baseline = baseline[mask]
        lower = lower[mask]
        upper = upper[mask]

        if observed is not None:
            ob_mask = (observed_dates >= TIMELINE_START) & (observed_dates <= TIMELINE_END)
            observed_dates = observed_dates[ob_mask]
            observed = observed[ob_mask]

        color = colors[idx % len(colors)]
        if pred_year == last_year:
            fig.add_trace(go.Scatter(x=dates, y=upper, fill=None, mode="lines", line=dict(color="rgba(0,100,80,0.2)")))
            fig.add_trace(
                go.Scatter(
                    x=dates, y=lower, fill="tonexty", mode="lines", fillcolor="rgba(0,100,80,0.2)",
                    line=dict(color="rgba(0,100,80,0.2)")
                )
            )
            fig.add_trace(go.Scatter(x=dates, y=baseline, mode="lines", line=dict(color=color)))
            if observed is not None:
                fig.add_trace(go.Scatter(x=observed_dates, y=observed, mode="lines", line=dict(color=color, dash="dash")))
        else:
            if observed is not None and len(observed_dates) > 0:
                aligned_indices = [np.where(dates == od)[0][0] for od in observed_dates if od in dates]
                fig.add_trace(
                    go.Scatter(
                        x=observed_dates, y=upper[aligned_indices], fill=None, mode="lines",
                        line=dict(color="rgba(0,100,80,0.2)")
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=observed_dates, y=lower[aligned_indices], fill="tonexty", mode="lines",
                        fillcolor="rgba(0,100,80,0.2)", line=dict(color="rgba(0,100,80,0.2)")
                    )
                )
                fig.add_trace(go.Scatter(x=observed_dates, y=baseline[aligned_indices], mode="lines", line=dict(color=color)))
                fig.add_trace(go.Scatter(x=observed_dates, y=observed, mode="lines", line=dict(color=color, dash="dash")))

    fig.update_layout(
        title="Verwachte Sterfte (Week 27, 2019 - Week 26, 2026)",
        xaxis_title="Datum",
        yaxis_title="Aantal Overlijdens",
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(range=[TIMELINE_START, TIMELINE_END]),
    )
    st.plotly_chart(fig)


# --------------------------------------------------------------------------------------
# Uitleg
# --------------------------------------------------------------------------------------
def uitleg() -> None:
    """Toon korte uitleg en formules."""
    st.info(
        """### ℹ️ Uitleg: Hoe werkt deze methode?

1. **Tijdreeks**: wekelijks aantal overlijdens.
2. **Trend**: lineaire component voor langzame veranderingen.
3. **Seizoenseffect**: sinus en cosinus (1e en optioneel 2e harmonische).
4. **Uitschieters weg**: hoogste 25% alle weken, plus hoogste 20% in juli/augustus.
5. **OLS**: fit op gefilterde weken.
6. **Band**: baseline ± 2×sd (≈95%).
7. **Interpretatie**: binnen band = normaal, erboven = verhoogd.
"""
    )
    st.markdown("### Formules")
    st.markdown("**Trend**")
    st.latex(r"y = a + b\,t")
    st.markdown("**Seizoenseffect (1e harmonische)**")
    st.latex(r"y = a + b\,t + c\,\sin\!\left(\tfrac{2\pi t}{52}\right) + d\,\cos\!\left(\tfrac{2\pi t}{52}\right)")
    st.markdown("**Seizoenseffect (2 harmonischen)**")
    st.latex(
        r"y = a + b\,t + c\,\sin\!\left(\tfrac{2\pi t}{52}\right) + d\,\cos\!\left(\tfrac{2\pi t}{52}\right) + "
        r"e\,\sin\!\left(\tfrac{4\pi t}{52}\right) + f\,\cos\!\left(\tfrac{4\pi t}{52}\right)"
    )
    st.markdown("**Band**")
    st.latex(r"\text{band} = \text{baseline} \pm 2\,\text{sd}")

    st.info(
        """### Waarom kan 2025/2026 lager uitvallen?
**Trainingswindow** verschuift: 2019/2020 valt weg, 2024/2025 komt erbij.
**Exclusies** drukken de baseline als recente zomers relatief laag waren."""
    )
    st.info(
        """Serfling (1963): Methods for Current Statistical Analysis of Excess Pneumonia-Influenza Deaths.
https://sci-hub.se/https://doi.org/10.2307/4591848"""
    )


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def main() -> None:
    """Streamlit UI en aansturing."""
    st.title("Verwachte sterfte • RIVM-stijl baselijn")
    st.caption("5 voorgaande seizoensjaren • trend + sinus/cosinus • pieken uitgesloten")

    st.info("We reproduceren de methode van het RIVM n.a.v. https://x.com/infopinie/status/1960744770810073247")

   

    with st.expander("Opties"):
        keuze = st.selectbox("Welke kolom te gebruiken voor overledenen?", options=["Totaal leeftijd";"0 tot 65 jaar";"65 tot 80 jaar";"80 jaar of ouder"], index=0)
        use_harm2 = st.checkbox("Gebruik 2 harmonischen", value=True)
        show_model = st.checkbox("Toon modeldetails", value=False)
        show_train = st.checkbox("Toon trainingspunten (na uitsluiten)", value=False)
        include_baselines_all = st.checkbox("Toon baselines in 'Alle seizoenen'", value=False)
        q_all = 1 - (st.number_input("Percentage hoogste waarden dat wordt weggefilterd (alle waardes)", 0, 100, 25) / 100)
        q_summer = 1 - (st.number_input("Percentage hoogste waarden dat wordt weggefilterd (juli/augustus)", 0, 100, 20) / 100)

    df = load_data_from_url(DEFAULT_INPUT_URL)
    df["overleden"] = df[keuze]
    min_season_available = int(df["season_year"].min() + 5)
    max_season_available = int(df["season_year"].max())
    harmonics = 2 if use_harm2 else 1

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tijdlijn 2019H2–2026H1", "Per seizoen", "Alle seizoenen 2021–2026", "GROK", "Uitleg"])

    with tab1:
        st.subheader("Alle reeksen in één plot")
        show_bands_timeline = st.checkbox("Toon ±2×sd banden", value=True)
        seasons_timeline = [sy for sy in range(2020, 2027) if min_season_available <= sy <= max_season_available]
        if not seasons_timeline:
            st.info("Onvoldoende data om 2020 t/m 2026 te tonen.")
        else:
            fig, tdf_complete = plot_timeline_all(
                df, seasons=seasons_timeline, q_all=q_all, q_summer=q_summer, show_bands=show_bands_timeline, harmonics=harmonics
            )
            st.plotly_chart(fig, use_container_width=True)
            plot_model_vs_rivm(tdf_complete)
            st.info("https://chatgpt.com/share/68b0d9ad-fdb0-8004-b943-886a964a8baa")

    with tab2:
        season_choice = st.slider(
            "Kies seizoensjaar",
            min_value=min_season_available,
            max_value=max_season_available,
            value=max_season_available,
            step=1,
        )
        result = fit_baseline(df, season_choice, q_all=q_all, q_summer=q_summer, harmonics=harmonics)
        target_df = result["target_df"]
        fig_7 = plot_single_season(target_df, season_choice)
        st.plotly_chart(fig_7, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Residuele sd", f"{result['resid_sd']:.1f}")
        col2.metric("Aantal trainingsweken", f"{len(result['train_used'])}")
        col3.metric("Aantal doelweken", f"{len(target_df)}")

        if show_train:
            st.dataframe(result["train_used"][["date", "jaar", "week", "overleden"]].reset_index(drop=True))

        csv_buf = io.StringIO()
        target_df[["date", "jaar", "week", "overleden", "baseline", "lower", "upper"]].to_csv(csv_buf, index=False)
        st.download_button(
            "Download resultaten CSV",
            data=csv_buf.getvalue(),
            file_name=f"baseline_{season_choice}.csv",
            mime="text/csv",
        )

        if show_model:
            st.text(result["model_summary"])

    with tab3:
        seasons_range = [sy for sy in range(2021, 2027) if min_season_available <= sy <= max_season_available]
        if not seasons_range:
            st.info("Onvoldoende data voor 2021–2026.")
        else:
            fig2 = plot_all_seasons(
                df, seasons=seasons_range, include_baselines=include_baselines_all, q_all=q_all, q_summer=q_summer, harmonics=harmonics
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        main_grok(q_all=q_all, q_summer=q_summer)

    with tab5:
        uitleg()

        st.info("RIVM Grafiek: https://www.rivm.nl/monitoring-sterftecijfers-nederland")


if __name__ == "__main__":
    main()
