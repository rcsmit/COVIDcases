"""Eurostat Oversterfte Analyse — Europa.

Haalt Eurostat data op voor overlijdens en bevolkingsgrootte per leeftijdsgroep,
berekent sterfte per 100.000 inwoners, fit een exponentiële basislijn op
2000-2019 en extrapoleert naar 2025. Berekent oversterfte absoluut en
per 100k. Multi-country support.

Bron: Eurostat (CC-BY 4.0)

Tabellen:
    - demo_r_mwk_05: Deaths by week, sex and 5-year age group
    - demo_pjangroup: Population on 1 January by age group and sex
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Constanten
# ---------------------------------------------------------------------------

EUROSTAT_DEATHS_TABLE = "demo_r_mwk_05"
EUROSTAT_POP_TABLE = "demo_pjangroup"

DATA_DIR = Path("data")
DEATHS_CACHE = DATA_DIR / "eurostat_deaths.csv"
POP_CACHE = DATA_DIR / "eurostat_pop.csv"

# Mapping van Eurostat leeftijdscodes naar brede groepen
AGE_BROAD_MAP: dict[str, str] = {
    "Y_LT5":  "0-65", "Y5-9":   "0-65", "Y10-14": "0-65",
    "Y15-19": "0-65", "Y20-24": "0-65", "Y25-29": "0-65",
    "Y30-34": "0-65", "Y35-39": "0-65", "Y40-44": "0-65",
    "Y45-49": "0-65", "Y50-54": "0-65", "Y55-59": "0-65",
    "Y60-64": "0-65",
    "Y65-69": "65-80", "Y70-74": "65-80", "Y75-79": "65-80",
    "Y80-84": "80+", "Y85-89": "80+", "Y_GE90": "80+",
    "Y_GE85": "80+",
}

COUNTRY_NAMES: dict[str, str] = {
    "NL": "Netherlands", "DE": "Germany", "BE": "Belgium",
    "FR": "France", "IT": "Italy", "ES": "Spain",
    "AT": "Austria", "SE": "Sweden", "DK": "Denmark",
    "PT": "Portugal", "PL": "Poland", "CZ": "Czechia",
    "IE": "Ireland", "FI": "Finland", "NO": "Norway",
    "CH": "Switzerland", "HU": "Hungary", "EL": "Greece",
    "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia",
    "SK": "Slovakia", "SI": "Slovenia", "LT": "Lithuania",
    "LV": "Latvia", "EE": "Estonia", "LU": "Luxembourg",
    "CY": "Cyprus", "MT": "Malta", "IS": "Iceland",
    "LI": "Liechtenstein",
}

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
]


# ---------------------------------------------------------------------------
# Eurostat data ophalen met lokale csv cache
# ---------------------------------------------------------------------------


def _download_deaths_from_eurostat() -> pd.DataFrame:
    """Download wekelijkse overlijdensdata van Eurostat en verwerk tot long format.

    Gebruikt tabel demo_r_mwk_05: Deaths by week, sex and 5-year age group.

    Returns:
        DataFrame met kolommen: jaar, land, leeftijdsgroep, overlijdens.
    """
    import eurostat

    df = eurostat.get_data_df(
        EUROSTAT_DEATHS_TABLE,
        filter_pars={"sex": ["T"]},
    )

    if df is None or df.empty:
        raise ValueError("Geen data ontvangen van Eurostat voor overlijdens")

    geo_col = [c for c in df.columns if "geo" in c.lower()][0]
    meta_cols = ["freq", "sex", "age", geo_col]
    time_cols = [c for c in df.columns if c not in meta_cols and c not in ["ID"]]

    df = df[df[geo_col].str.len() == 2].copy()

    df_long = df.melt(
        id_vars=["age", geo_col],
        value_vars=time_cols,
        var_name="periode",
        value_name="overlijdens",
    )

    df_long["jaar"] = df_long["periode"].str.extract(r"(\d{4})").astype(float)
    df_long = df_long.dropna(subset=["jaar", "overlijdens"])
    df_long["jaar"] = df_long["jaar"].astype(int)

    df_long["leeftijdsgroep"] = df_long["age"].map(AGE_BROAD_MAP)
    df_long = df_long.dropna(subset=["leeftijdsgroep"])

    deaths = (
        df_long.groupby(["jaar", geo_col, "leeftijdsgroep"])["overlijdens"]
        .sum()
        .reset_index()
        .rename(columns={geo_col: "land"})
    )

    totaal = (
        deaths.groupby(["jaar", "land"])["overlijdens"]
        .sum()
        .reset_index()
    )
    totaal["leeftijdsgroep"] = "Totaal"
    deaths = pd.concat([deaths, totaal], ignore_index=True)

    return deaths


def _download_population_from_eurostat() -> pd.DataFrame:
    """Download bevolkingsdata van Eurostat en verwerk tot long format.

    Gebruikt tabel demo_pjangroup: Population on 1 January by age group and sex.

    Returns:
        DataFrame met kolommen: jaar, land, leeftijdsgroep, bevolking.
    """
    import eurostat

    df = eurostat.get_data_df(
        EUROSTAT_POP_TABLE,
        filter_pars={"sex": ["T"], "unit": ["NR"]},
    )

    if df is None or df.empty:
        raise ValueError("Geen data ontvangen van Eurostat voor bevolking")

    geo_col = [c for c in df.columns if "geo" in c.lower()][0]
    meta_cols = [c for c in ["freq", "sex", "age", "unit", geo_col] if c in df.columns]
    time_cols = [c for c in df.columns if c not in meta_cols and c not in ["ID"]]

    df = df[df[geo_col].str.len() == 2].copy()

    df_long = df.melt(
        id_vars=["age", geo_col],
        value_vars=time_cols,
        var_name="jaar_str",
        value_name="bevolking",
    )

    df_long["jaar"] = pd.to_numeric(df_long["jaar_str"], errors="coerce")
    df_long = df_long.dropna(subset=["jaar", "bevolking"])
    df_long["jaar"] = df_long["jaar"].astype(int)

    df_long["leeftijdsgroep"] = df_long["age"].map(AGE_BROAD_MAP)
    df_long = df_long.dropna(subset=["leeftijdsgroep"])

    pop = (
        df_long.groupby(["jaar", geo_col, "leeftijdsgroep"])["bevolking"]
        .sum()
        .reset_index()
        .rename(columns={geo_col: "land"})
    )

    totaal = (
        pop.groupby(["jaar", "land"])["bevolking"]
        .sum()
        .reset_index()
    )
    totaal["leeftijdsgroep"] = "Totaal"
    pop = pd.concat([pop, totaal], ignore_index=True)

    return pop


def load_deaths(force_refresh: bool = False) -> pd.DataFrame:
    """Laad overlijdensdata: uit csv-cache of van Eurostat.

    Args:
        force_refresh: Als True, download opnieuw ongeacht cache.

    Returns:
        DataFrame met kolommen: jaar, land, leeftijdsgroep, overlijdens.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if not force_refresh and DEATHS_CACHE.exists():
        st.toast(f"Overlijdensdata geladen uit {DEATHS_CACHE}")
        return pd.read_csv(DEATHS_CACHE)

    with st.spinner("Eurostat overlijdensdata downloaden (kan even duren)..."):
        deaths = _download_deaths_from_eurostat()
        deaths.to_csv(DEATHS_CACHE, index=False)
        st.toast(f"Overlijdensdata opgeslagen in {DEATHS_CACHE}")
    return deaths


def load_population(force_refresh: bool = False) -> pd.DataFrame:
    """Laad bevolkingsdata: uit csv-cache of van Eurostat.

    Args:
        force_refresh: Als True, download opnieuw ongeacht cache.

    Returns:
        DataFrame met kolommen: jaar, land, leeftijdsgroep, bevolking.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if not force_refresh and POP_CACHE.exists():
        st.toast(f"Bevolkingsdata geladen uit {POP_CACHE}")
        return pd.read_csv(POP_CACHE)

    with st.spinner("Eurostat bevolkingsdata downloaden (kan even duren)..."):
        pop = _download_population_from_eurostat()
        pop.to_csv(POP_CACHE, index=False)
        st.toast(f"Bevolkingsdata opgeslagen in {POP_CACHE}")
    return pop


# ---------------------------------------------------------------------------
# Berekeningen
# ---------------------------------------------------------------------------


def exponential_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponentieel model: y = a * exp(b * x).

    Args:
        x: Onafhankelijke variabele (jaar - offset).
        a: Amplitude parameter.
        b: Groeisnelheid parameter.

    Returns:
        Modelwaarden.
    """
    return a * np.exp(b * x)


def calculate_mortality_analysis(
    deaths: pd.DataFrame,
    pop: pd.DataFrame,
    country: str,
    group: str,
    baseline_start: int,
    baseline_end: int,
) -> pd.DataFrame | None:
    """Bereken sterfte per 100k, fit basislijn, en bereken oversterfte.

    Args:
        deaths: DataFrame met overlijdensdata.
        pop: DataFrame met bevolkingsdata.
        country: Landcode (2 letters).
        group: Leeftijdsgroep om te analyseren.
        baseline_start: Startjaar voor de basislijn.
        baseline_end: Eindjaar voor de basislijn (inclusief).

    Returns:
        DataFrame met alle berekende kolommen, of None als het mislukt.
    """
    d = deaths[
        (deaths["land"] == country) & (deaths["leeftijdsgroep"] == group)
    ].copy()
    p = pop[
        (pop["land"] == country) & (pop["leeftijdsgroep"] == group)
    ].copy()

    if d.empty or p.empty:
        return None

    merged = d.merge(p, on=["jaar", "land", "leeftijdsgroep"], how="inner")
    merged = merged.sort_values("jaar").reset_index(drop=True)

    if merged.empty or len(merged) < 5:
        return None

    # Sterfte per 100.000
    merged["sterfte_per_100k"] = (
        merged["overlijdens"] / merged["bevolking"] * 100_000
    )

    # Basislijn: exponentieel fit
    baseline = merged[
        (merged["jaar"] >= baseline_start) & (merged["jaar"] <= baseline_end)
    ].copy()

    if len(baseline) < 3:
        return None

    x_offset = baseline_start
    x_base = (baseline["jaar"] - x_offset).values.astype(float)
    y_base = baseline["sterfte_per_100k"].values.astype(float)

    fit_type = "exponentieel"
    try:
        popt, _ = curve_fit(
            exponential_model, x_base, y_base,
            p0=[y_base[0], -0.01], maxfev=10000,
        )
    except (RuntimeError, ValueError):
        coeffs = np.polyfit(x_base, y_base, 1)
        x_all = (merged["jaar"] - x_offset).values.astype(float)
        merged["basislijn_per_100k"] = np.polyval(coeffs, x_all)
        popt = [coeffs[1], coeffs[0]]
        fit_type = "lineair"

    if fit_type == "exponentieel":
        x_all = (merged["jaar"] - x_offset).values.astype(float)
        merged["basislijn_per_100k"] = exponential_model(x_all, *popt)

    merged["oversterfte_per_100k"] = (
        merged["sterfte_per_100k"] - merged["basislijn_per_100k"]
    )
    merged["oversterfte_absoluut"] = (
        merged["oversterfte_per_100k"] / 100_000 * merged["bevolking"]
    )
    merged["fit_type"] = fit_type
    merged["a"] = popt[0]
    merged["b"] = popt[1]

    return merged


# ---------------------------------------------------------------------------
# Grafieken
# ---------------------------------------------------------------------------


def plot_mortality_rate(
    df: pd.DataFrame,
    country: str,
    group: str,
    baseline_start: int,
    baseline_end: int,
) -> go.Figure:
    """Plot sterfte per 100k met basislijn.

    Args:
        df: DataFrame met analyse-resultaten.
        country: Landcode.
        group: Naam van de leeftijdsgroep.
        baseline_start: Startjaar basislijn.
        baseline_end: Eindjaar basislijn.

    Returns:
        Plotly Figure object.
    """
    country_name = COUNTRY_NAMES.get(country, country)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["jaar"], y=df["sterfte_per_100k"],
        mode="lines+markers",
        name="Actual mortality per 100k",
        line=dict(color=COLORS[0], width=2),
        marker=dict(size=5),
    ))

    fig.add_trace(go.Scatter(
        x=df["jaar"], y=df["basislijn_per_100k"],
        mode="lines",
        name=f"Baseline ({baseline_start}-{baseline_end})",
        line=dict(color=COLORS[1], width=2, dash="dash"),
    ))

    fig.add_vrect(
        x0=baseline_start, x1=baseline_end,
        fillcolor="lightgray", opacity=0.3,
        layer="below", line_width=0,
        annotation_text="Baseline", annotation_position="top left",
    )

    fit_type = df["fit_type"].iloc[0]
    a_val = df["a"].iloc[0]
    b_val = df["b"].iloc[0]

    fig.update_layout(
        title=f"Mortality per 100,000 — {country_name} — {group}",
        xaxis_title="Year",
        yaxis_title="Deaths per 100,000",
        template="plotly_white",
        height=500,
        annotations=[dict(
            text=f"Fit: {fit_type} | a={a_val:.2f}, b={b_val:.6f}",
            xref="paper", yref="paper",
            x=0.01, y=0.01, showarrow=False,
            font=dict(size=10, color="gray"),
        )],
    )

    return fig


def plot_excess_mortality(
    df: pd.DataFrame,
    country: str,
    group: str,
    per_100k: bool = True,
) -> go.Figure:
    """Plot oversterfte als staafdiagram.

    Args:
        df: DataFrame met analyse-resultaten.
        country: Landcode.
        group: Naam van de leeftijdsgroep.
        per_100k: Als True, toon per 100k; anders absoluut.

    Returns:
        Plotly Figure object.
    """
    country_name = COUNTRY_NAMES.get(country, country)
    col = "oversterfte_per_100k" if per_100k else "oversterfte_absoluut"
    label = "per 100k" if per_100k else "absolute"

    colors_bars = [COLORS[3] if v >= 0 else COLORS[0] for v in df[col]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["jaar"], y=df[col],
        marker_color=colors_bars,
        name=f"Excess mortality {label}",
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1))

    fig.update_layout(
        title=f"Excess mortality {label} — {country_name} — {group}",
        xaxis_title="Year",
        yaxis_title=f"Excess mortality ({label})",
        template="plotly_white",
        height=450,
    )

    return fig


def plot_multi_group(
    deaths: pd.DataFrame,
    pop: pd.DataFrame,
    country: str,
    groups: list[str],
    baseline_start: int,
    baseline_end: int,
) -> go.Figure:
    """Plot oversterfte per 100k voor meerdere groepen.

    Args:
        deaths: Overlijdensdata.
        pop: Bevolkingsdata.
        country: Landcode.
        groups: Lijst van leeftijdsgroepen.
        baseline_start: Startjaar basislijn.
        baseline_end: Eindjaar basislijn.

    Returns:
        Plotly Figure object.
    """
    country_name = COUNTRY_NAMES.get(country, country)
    fig = go.Figure()

    for i, group in enumerate(groups):
        df = calculate_mortality_analysis(
            deaths, pop, country, group, baseline_start, baseline_end
        )
        if df is None:
            continue
        fig.add_trace(go.Scatter(
            x=df["jaar"], y=df["oversterfte_per_100k"],
            mode="lines+markers",
            name=group,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=4),
        ))

    fig.add_hline(y=0, line=dict(color="black", width=1))

    fig.update_layout(
        title=f"Excess mortality per 100k by age group — {country_name}",
        xaxis_title="Year",
        yaxis_title="Excess mortality per 100,000",
        template="plotly_white",
        height=550,
    )

    return fig


def plot_multi_country(
    deaths: pd.DataFrame,
    pop: pd.DataFrame,
    countries: list[str],
    group: str,
    baseline_start: int,
    baseline_end: int,
) -> go.Figure:
    """Plot oversterfte per 100k voor meerdere landen.

    Args:
        deaths: Overlijdensdata.
        pop: Bevolkingsdata.
        countries: Lijst van landcodes.
        group: Leeftijdsgroep.
        baseline_start: Startjaar basislijn.
        baseline_end: Eindjaar basislijn.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for i, country in enumerate(countries):
        df = calculate_mortality_analysis(
            deaths, pop, country, group, baseline_start, baseline_end
        )
        if df is None:
            continue
        name = COUNTRY_NAMES.get(country, country)
        fig.add_trace(go.Scatter(
            x=df["jaar"], y=df["oversterfte_per_100k"],
            mode="lines+markers",
            name=name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=4),
        ))

    fig.add_hline(y=0, line=dict(color="black", width=1))

    fig.update_layout(
        title=f"Excess mortality per 100k — {group} — Country comparison",
        xaxis_title="Year",
        yaxis_title="Excess mortality per 100,000",
        template="plotly_white",
        height=550,
    )

    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------


def run_app() -> None:
    """Hoofdfunctie van de Streamlit app."""
    st.set_page_config(
        page_title="Eurostat Excess Mortality",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Eurostat Excess Mortality Analysis — Europe")
    st.markdown(
        "Mortality per 100,000 with exponential baseline and excess mortality "
        "calculation. Data: Eurostat (CC-BY 4.0)."
    )

    # --- Data laden ---
    with st.sidebar:
        st.header("💾 Data cache")
        deaths_exists = DEATHS_CACHE.exists()
        pop_exists = POP_CACHE.exists()

        if deaths_exists and pop_exists:
            from datetime import datetime
            d_mtime = datetime.fromtimestamp(DEATHS_CACHE.stat().st_mtime)
            st.success(f"Cache gevonden ({d_mtime:%Y-%m-%d %H:%M})")
        else:
            st.info("Geen cache — data wordt gedownload bij eerste run")

        force_refresh = st.button("🔄 Download opnieuw van Eurostat")

    try:
        deaths_raw = load_deaths(force_refresh=force_refresh)
        pop_raw = load_population(force_refresh=force_refresh)
    except Exception as e:
        st.error(f"Error fetching Eurostat data: {e}")
        st.info(
            "Install `eurostat` with: `pip install eurostat`\n\n"
            "Check your internet connection to ec.europa.eu."
        )
        st.stop()

    # Beschikbare landen (intersectie deaths & pop)
    death_countries = set(deaths_raw["land"].unique())
    pop_countries = set(pop_raw["land"].unique())
    available_countries = sorted(death_countries & pop_countries)

    available_groups = sorted(
        set(deaths_raw["leeftijdsgroep"].unique())
        & set(pop_raw["leeftijdsgroep"].unique())
    )

    if not available_countries or not available_groups:
        st.error("No matching countries/age groups found between datasets.")
        with st.expander("Debug info"):
            st.write("Deaths countries:", sorted(death_countries))
            st.write("Pop countries:", sorted(pop_countries))
            st.write("Deaths groups:", sorted(deaths_raw["leeftijdsgroep"].unique()))
            st.write("Pop groups:", sorted(pop_raw["leeftijdsgroep"].unique()))
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")

        # Land selectie
        country_options = {
            c: f"{COUNTRY_NAMES.get(c, c)} ({c})"
            for c in available_countries
        }
        default_country = "NL" if "NL" in available_countries else available_countries[0]

        selected_country = st.selectbox(
            "Country",
            available_countries,
            index=available_countries.index(default_country),
            format_func=lambda c: country_options.get(c, c),
        )

        baseline_start = st.number_input(
            "Baseline start year",
            min_value=1995, max_value=2019, value=2000, step=1,
        )
        baseline_end = st.number_input(
            "Baseline end year",
            min_value=2000, max_value=2019, value=2019, step=1,
        )

        if baseline_start >= baseline_end:
            st.error("Start year must be less than end year")
            st.stop()

        selected_group = st.selectbox(
            "Age group (detail)",
            available_groups,
            index=available_groups.index("Totaal") if "Totaal" in available_groups else 0,
        )

        compare_groups = st.multiselect(
            "Compare age groups",
            available_groups,
            default=[g for g in available_groups if g != "Totaal"][:3],
        )

        # Landenvergelijking
        compare_countries = st.multiselect(
            "Compare countries",
            available_countries,
            default=[
                c for c in ["NL", "DE", "BE", "SE"]
                if c in available_countries
            ][:4],
            format_func=lambda c: country_options.get(c, c),
        )

        min_year = st.number_input(
            "Data from year",
            min_value=1995, max_value=2020, value=2000, step=1,
        )

        max_year_deaths = deaths_raw["jaar"].max()
        max_year_pop = pop_raw["jaar"].max()
        st.markdown(
            f"**Deaths up to:** {max_year_deaths}  \n"
            f"**Population up to:** {max_year_pop}"
        )

    # Filter op jaar
    deaths = deaths_raw[deaths_raw["jaar"] >= min_year].copy()
    pop = pop_raw[pop_raw["jaar"] >= min_year].copy()

    # --- Analyse ---
    df_analysis = calculate_mortality_analysis(
        deaths, pop, selected_country, selected_group,
        baseline_start, baseline_end,
    )

    country_name = COUNTRY_NAMES.get(selected_country, selected_country)

    if df_analysis is None:
        st.error(
            f"Analysis not possible for {country_name} / {selected_group}. "
            "Check data availability for this country and period."
        )
        st.stop()

    # --- Grafieken ---

    st.plotly_chart(
        plot_mortality_rate(
            df_analysis, selected_country, selected_group,
            baseline_start, baseline_end,
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_excess_mortality(
                df_analysis, selected_country, selected_group, per_100k=True,
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            plot_excess_mortality(
                df_analysis, selected_country, selected_group, per_100k=False,
            ),
            use_container_width=True,
        )

    # Vergelijking groepen
    if compare_groups:
        st.plotly_chart(
            plot_multi_group(
                deaths, pop, selected_country, compare_groups,
                baseline_start, baseline_end,
            ),
            use_container_width=True,
        )

    # Vergelijking landen
    if compare_countries:
        st.plotly_chart(
            plot_multi_country(
                deaths, pop, compare_countries, selected_group,
                baseline_start, baseline_end,
            ),
            use_container_width=True,
        )

    # --- Samenvatting ---
    st.subheader(f"Excess mortality summary — {country_name}")
    recent = df_analysis[df_analysis["jaar"] >= 2020].copy()
    if not recent.empty:
        n_cols = min(len(recent), 6)
        cols = st.columns(n_cols)
        for i, (_, row) in enumerate(recent.iterrows()):
            with cols[i % n_cols]:
                st.metric(
                    label=str(int(row["jaar"])),
                    value=f"{row['oversterfte_absoluut']:,.0f}",
                    delta=f"{row['oversterfte_per_100k']:+.1f} /100k",
                    delta_color="inverse",
                )

        totaal = recent["oversterfte_absoluut"].sum()
        st.metric("Total excess mortality 2020+", f"{totaal:,.0f}")

    # --- Data tabel ---
    with st.expander("📋 Data table"):
        display_cols = [
            "jaar", "land", "leeftijdsgroep", "overlijdens", "bevolking",
            "sterfte_per_100k", "basislijn_per_100k",
            "oversterfte_per_100k", "oversterfte_absoluut",
        ]
        avail = [c for c in display_cols if c in df_analysis.columns]

        fmt = {
            "overlijdens": "{:,.0f}",
            "bevolking": "{:,.0f}",
            "sterfte_per_100k": "{:.1f}",
            "basislijn_per_100k": "{:.1f}",
            "oversterfte_per_100k": "{:.1f}",
            "oversterfte_absoluut": "{:,.0f}",
        }
        fmt_ok = {k: v for k, v in fmt.items() if k in avail}

        st.dataframe(
            df_analysis[avail].style.format(fmt_ok),
            width="stretch",
            hide_index=True,
        )

    # --- Methodologie ---
    with st.expander("ℹ️ Methodology"):
        st.markdown(f"""
**Data sources:**
- Deaths: Eurostat table `{EUROSTAT_DEATHS_TABLE}` (deaths by week, sex and 5-year age group)
- Population: Eurostat table `{EUROSTAT_POP_TABLE}` (population on 1 January by age group and sex)

**Method:**
1. Weekly deaths are aggregated to yearly totals
2. 5-year age groups are mapped to broad categories: 0-65, 65-80, 80+, Totaal
3. Mortality rate = (deaths / population) × 100,000
4. Exponential model **y = a·e^(b·x)** is fitted on baseline period ({baseline_start}–{baseline_end})
5. Baseline is extrapolated to all years
6. Excess mortality = actual mortality − baseline (both per 100k and absolute)
7. Absolute excess = excess per 100k × population / 100,000

**Age group mapping (Eurostat → Broad):**
- Y_LT5 through Y60-64 → 0-65
- Y65-69 through Y75-79 → 65-80
- Y80-84 through Y_GE90 → 80+

**Notes:**
- Baseline period choice strongly affects results
- Not all countries have complete data for all years
- Recent years may contain provisional figures
- The `eurostat` Python package downloads bulk data from the Eurostat SDMX API
        """)


run_app()
