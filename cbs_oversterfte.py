"""CBS Oversterfte Analyse — Nederland.

Laadt overlijdens- en bevolkingsdata uit lokale bestanden,
berekent sterfte per 100.000 inwoners, fit een lineaire / kwadratische /
exponentiële basislijn en extrapoleert naar 2025/2026.
Berekent oversterfte absoluut en per 100k.

Bron: CBS OpenData (CC-BY 4.0)

Inputbestanden:
    - 70895ned_overlijdens_cbs.xlsx : pivot-tabel overledenen per jaar/leeftijd
    - bevolking_leeftijd_nl_crosstable.csv : bevolking per leeftijd/jaar
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
import platform

# ---------------------------------------------------------------------------
# Constanten
# ---------------------------------------------------------------------------

if platform.processor() != "":
    INPUT_DIR = r"C:/Users/rcxsm/Documents/python_scripts/covid19_seir_models/COVIDcases/input/"
else:
    INPUT_DIR = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/"
DEATHS_FILE = INPUT_DIR + "70895ned_overlijdens_cbs.xlsx"
POP_FILE = INPUT_DIR + "bevolking_leeftijd_nl_crosstable.csv"

# Mapping xlsx-groepslabels -> leeftijdsgroepnamen
_DEATHS_GROUP_MAP: dict[str, str] = {
    "0-64":   "0 tot 65 jaar",
    "65-79":  "65 tot 80 jaar",
    "80-120": "80 jaar of ouder",
    "0-120":  "Totaal leeftijd",
}


FIT_TYPES = ["exponentieel", "lineair", "kwadratisch", "logaritmisch"]
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#5254a3", "#6b6ecf", "#9c9ede", "#637939",
    "#8ca252", "#b5cf6b", "#cedb9c", "#8c6d31", "#bd9e39",
]


# ---------------------------------------------------------------------------
# Data laden
# ---------------------------------------------------------------------------


def load_deaths(force_refresh: bool = False) -> pd.DataFrame:
    """Laad overlijdensdata uit lokale xlsx (CBS pivot-tabel formaat).

    Args:
        force_refresh: Niet gebruikt; aanwezig voor API-compatibiliteit.

    Returns:
        DataFrame met kolommen: jaar, leeftijdsgroep, overlijdens.
    """
    raw = pd.read_excel(DEATHS_FILE, header=None)

    # Rij 3 (0-indexed) bevat de jaarlabels; kolom 0 is leeg, kolom 1+ zijn jaren
    jaar_row = raw.iloc[3, 1:].tolist()

    current_geslacht: str | None = None
    records: list[dict] = []

    for _, row in raw.iterrows():
        label = str(row.iloc[0]).strip()

        if label in ("F", "M", "T"):
            current_geslacht = label
            continue

        if current_geslacht != "T":
            continue

        groep = _DEATHS_GROUP_MAP.get(label)
        if groep is None:
            continue

        for col_idx, jaar_val in enumerate(jaar_row):
            try:
                jaar = int(float(str(jaar_val)))
            except (ValueError, TypeError):
                continue
            val = row.iloc[col_idx + 1]
            try:
                overlijdens = int(float(val))
            except (ValueError, TypeError):
                continue
            records.append({
                "jaar": jaar,
                "leeftijdsgroep": groep,
                "overlijdens": overlijdens,
            })

    return pd.DataFrame(records)


def load_population(force_refresh: bool = False) -> pd.DataFrame:
    """Laad bevolkingsdata uit lokale CSV en aggregeer naar leeftijdsgroepen.

    Args:
        force_refresh: Niet gebruikt; aanwezig voor API-compatibiliteit.

    Returns:
        DataFrame met kolommen: jaar, leeftijdsgroep, bevolking.
    """
    df = pd.read_csv(POP_FILE, sep=";")
    df = df[df["Geslacht"] == "T"].copy()

    jaar_cols = [c for c in df.columns if str(c).isdigit()]

    long = df.melt(
        id_vars=["Geslacht", "Leeftijd"],
        value_vars=jaar_cols,
        var_name="jaar",
        value_name="bevolking",
    )
    long["jaar"] = long["jaar"].astype(int)
    long["bevolking"] = pd.to_numeric(long["bevolking"], errors="coerce")

    def _to_groups(age: int) -> list[str]:
        """Geeft alle leeftijdsgroepen terug waar deze leeftijd bij hoort."""
        groups = ["Totaal leeftijd"]
        if age < 65:
            groups.append("0 tot 65 jaar")
        elif age < 80:
            groups.append("65 tot 80 jaar")
        else:
            groups.append("80 jaar of ouder")
        return groups

    records: list[dict] = []
    for _, row in long.iterrows():
        for groep in _to_groups(int(row["Leeftijd"])):
            records.append({
                "jaar": row["jaar"],
                "leeftijdsgroep": groep,
                "bevolking": row["bevolking"],
            })

    pop = (
        pd.DataFrame(records)
        .groupby(["jaar", "leeftijdsgroep"])["bevolking"]
        .sum()
        .reset_index()
    )
    return pop


# ---------------------------------------------------------------------------
# Basislijn modellen
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


def fit_baseline(
    x_base: np.ndarray,
    y_base: np.ndarray,
    x_all: np.ndarray,
    fit_type: str,
) -> tuple[np.ndarray, str]:
    """Fit een basislijn op de basislijnperiode en extrapoleer naar x_all.

    Args:
        x_base: x-waarden van de basislijnperiode.
        y_base: y-waarden van de basislijnperiode.
        x_all: x-waarden voor alle jaren (extrapolatie).
        fit_type: Een van 'exponentieel', 'lineair', 'kwadratisch'.

    Returns:
        Tuple van (geëxtrapoleerde basislijnwaarden, daadwerkelijk gebruikte fit_type).
    """
    if fit_type == "exponentieel":
        try:
            popt, _ = curve_fit(
                exponential_model, x_base, y_base,
                p0=[y_base[0], -0.01], maxfev=10000,
            )
            return exponential_model(x_all, *popt), "exponentieel"
        except (RuntimeError, ValueError):
            fit_type = "lineair"  # fallback
    
    if fit_type == "kwadratisch":
        coeffs = np.polyfit(x_base, y_base, 2)
        return np.polyval(coeffs, x_all), "kwadratisch"

    if fit_type == "logaritmisch":
        # log(0) vermijden: x_base loopt vanaf 0, dus verschuif met 1
        try:
            coeffs = np.polyfit(np.log(x_base + 1), y_base, 1)
            return np.polyval(coeffs, np.log(x_all + 1)), "logaritmisch"
        except (ValueError, np.linalg.LinAlgError):
            fit_type = "lineair"  # fallback

    # lineair (default / fallback)
    coeffs = np.polyfit(x_base, y_base, 1)
    return np.polyval(coeffs, x_all), "lineair"


# ---------------------------------------------------------------------------
# Berekeningen
# ---------------------------------------------------------------------------


def calculate_mortality_analysis(
    deaths: pd.DataFrame,
    pop: pd.DataFrame,
    group: str,
    baseline_start: int,
    baseline_end: int,
    fit_type: str = "exponentieel",
) -> pd.DataFrame | None:
    """Bereken sterfte per 100k, fit basislijn, en bereken oversterfte.

    Args:
        deaths: DataFrame met overlijdensdata.
        pop: DataFrame met bevolkingsdata.
        group: Leeftijdsgroep om te analyseren.
        baseline_start: Startjaar voor de basislijn.
        baseline_end: Eindjaar voor de basislijn (inclusief).
        fit_type: Type basislijn: 'exponentieel', 'lineair' of 'kwadratisch'.

    Returns:
        DataFrame met alle berekende kolommen, of None als het mislukt.
    """
    d = deaths[deaths["leeftijdsgroep"] == group].copy()
    p = pop[pop["leeftijdsgroep"] == group].copy()

    if d.empty or p.empty:
        return None

    merged = d.merge(p, on=["jaar", "leeftijdsgroep"], how="inner")
    merged = merged.sort_values("jaar").reset_index(drop=True)

    if merged.empty:
        return None

    merged["sterfte_per_100k"] = (
        merged["overlijdens"] / merged["bevolking"] * 100_000
    )

    baseline = merged[
        (merged["jaar"] >= baseline_start) & (merged["jaar"] <= baseline_end)
    ].copy()

    if len(baseline) < 3:
        return None

    x_offset = baseline_start
    x_base = (baseline["jaar"] - x_offset).values.astype(float)
    y_base = baseline["sterfte_per_100k"].values.astype(float)
    x_all = (merged["jaar"] - x_offset).values.astype(float)

    basislijn_vals, used_fit_type = fit_baseline(x_base, y_base, x_all, fit_type)

    merged["basislijn_per_100k"] = basislijn_vals
    merged["fit_type"] = used_fit_type

    merged["oversterfte_per_100k"] = (
        merged["sterfte_per_100k"] - merged["basislijn_per_100k"]
    )
    merged["oversterfte_absoluut"] = (
        merged["oversterfte_per_100k"] / 100_000 * merged["bevolking"]
    )
    merged["basislijn_absoluut"] = (
        merged["basislijn_per_100k"] / 100_000 * merged["bevolking"]
    )

    return merged


# ---------------------------------------------------------------------------
# Grafieken
# ---------------------------------------------------------------------------


def plot_mortality_rate(
    df: pd.DataFrame,
    group: str,
    baseline_start: int,
    baseline_end: int,
) -> go.Figure:
    """Plot sterfte per 100k met basislijn.

    Args:
        df: DataFrame met analyse-resultaten.
        group: Naam van de leeftijdsgroep.
        baseline_start: Startjaar basislijn.
        baseline_end: Eindjaar basislijn.

    Returns:
        Plotly Figure object.
    """
    fit_type = df["fit_type"].iloc[0]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["jaar"], y=df["sterfte_per_100k"],
        mode="lines+markers",
        name="Werkelijke sterfte per 100k",
        line=dict(color=COLORS[0], width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=df["jaar"], y=df["basislijn_per_100k"],
        mode="lines",
        name=f"Basislijn ({fit_type}, {baseline_start}–{baseline_end})",
        line=dict(color=COLORS[1], width=2, dash="dash"),
    ))
    fig.add_vrect(
        x0=baseline_start, x1=baseline_end,
        fillcolor="lightgray", opacity=0.3,
        layer="below", line_width=0,
        annotation_text="Basislijn", annotation_position="top left",
    )
    fig.update_layout(
        title=f"Sterfte per 100.000 inwoners — {group}",
        xaxis_title="Jaar",
        yaxis_title="Sterfte per 100.000",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_absolute_mortality(
    df: pd.DataFrame,
    group: str,
    baseline_start: int,
    baseline_end: int,
) -> go.Figure:
    """Plot absolute sterfte (werkelijk vs verwacht).

    Args:
        df: DataFrame met analyse-resultaten.
        group: Naam van de leeftijdsgroep.
        baseline_start: Startjaar basislijn.
        baseline_end: Eindjaar basislijn.

    Returns:
        Plotly Figure object.
    """
    fit_type = df["fit_type"].iloc[0]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["jaar"], y=df["overlijdens"],
        mode="lines+markers",
        name="Werkelijke sterfte",
        line=dict(color=COLORS[0], width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=df["jaar"], y=df["basislijn_absoluut"],
        mode="lines",
        name=f"Verwachte sterfte ({fit_type})",
        line=dict(color=COLORS[1], width=2, dash="dash"),
    ))
    fig.add_vrect(
        x0=baseline_start, x1=baseline_end,
        fillcolor="lightgray", opacity=0.3,
        layer="below", line_width=0,
        annotation_text="Basislijn", annotation_position="top left",
    )
    fig.update_layout(
        title=f"Absolute sterfte — {group}",
        xaxis_title="Jaar",
        yaxis_title="Aantal overlijdens",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_excess_mortality(
    df: pd.DataFrame,
    group: str,
    per_100k: bool = True,
) -> go.Figure:
    """Plot oversterfte als staafdiagram.

    Args:
        df: DataFrame met analyse-resultaten.
        group: Naam van de leeftijdsgroep.
        per_100k: Als True, toon per 100k; anders absoluut.

    Returns:
        Plotly Figure object.
    """
    col = "oversterfte_per_100k" if per_100k else "oversterfte_absoluut"
    label = "per 100k" if per_100k else "absoluut"
    fit_type = df["fit_type"].iloc[0]

    colors_bars = [COLORS[3] if v >= 0 else COLORS[0] for v in df[col]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["jaar"], y=df[col],
        marker_color=colors_bars,
        name=f"Oversterfte {label}",
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(
        title=f"Oversterfte {label} ({fit_type}) — {group}",
        xaxis_title="Jaar",
        yaxis_title=f"Oversterfte ({label})",
        template="plotly_white",
        height=450,
    )
    return fig


def plot_fit_comparison(
    df_exp: pd.DataFrame,
    df_lin: pd.DataFrame,
    df_kwa: pd.DataFrame,
    df_log: pd.DataFrame,
    group: str,
    baseline_start: int,
    baseline_end: int,
) -> go.Figure:
    """Plot alle drie basislijnen over de werkelijke sterfte per 100k.

    Args:
        df_exp: Analyse-resultaten exponentieel.
        df_lin: Analyse-resultaten lineair.
        df_kwa: Analyse-resultaten kwadratisch.
        group: Naam van de leeftijdsgroep.
        baseline_start: Startjaar basislijn.
        baseline_end: Eindjaar basislijn.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_exp["jaar"], y=df_exp["sterfte_per_100k"],
        mode="lines+markers",
        name="Werkelijk",
        line=dict(color=COLORS[0], width=2),
        marker=dict(size=5),
    ))
    for df, color, name in [
        (df_exp, COLORS[1], "Exponentieel"),
        (df_lin, COLORS[2], "Lineair"),
        (df_kwa, COLORS[3], "Kwadratisch"),
        (df_log, COLORS[4], "Logaritmisch"),
    ]:
        fig.add_trace(go.Scatter(
            x=df["jaar"], y=df["basislijn_per_100k"],
            mode="lines",
            name=name,
            line=dict(color=color, width=2, dash="dash"),
        ))

    fig.add_vrect(
        x0=baseline_start, x1=baseline_end,
        fillcolor="lightgray", opacity=0.3,
        layer="below", line_width=0,
        annotation_text="Basislijn", annotation_position="top left",
    )
    fig.update_layout(
        title=f"Vergelijking basislijnen — {group}",
        xaxis_title="Jaar",
        yaxis_title="Sterfte per 100.000",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_multi_group(
    deaths: pd.DataFrame,
    pop: pd.DataFrame,
    groups: list[str],
    baseline_start: int,
    baseline_end: int,
    fit_type: str = "exponentieel",
) -> go.Figure:
    """Plot oversterfte per 100k voor meerdere groepen.

    Args:
        deaths: Overlijdensdata.
        pop: Bevolkingsdata.
        groups: Lijst van leeftijdsgroepen.
        baseline_start: Startjaar basislijn.
        baseline_end: Eindjaar basislijn.
        fit_type: Type basislijn.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for i, group in enumerate(groups):
        df = calculate_mortality_analysis(
            deaths, pop, group, baseline_start, baseline_end, fit_type
        )
        if df is None or "oversterfte_per_100k" not in df.columns:
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
        title=f"Oversterfte per 100k ({fit_type}) — Vergelijking leeftijdsgroepen",
        xaxis_title="Jaar",
        yaxis_title="Oversterfte per 100.000",
        template="plotly_white",
        height=550,
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

def main() -> None:
    """Hoofdfunctie van de Streamlit app."""
    st.set_page_config(
        page_title="CBS Oversterfte Analyse",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 CBS Oversterfte Analyse — Nederland")
    st.markdown(
        "Sterfte per 100.000 inwoners met lineaire / kwadratische / exponentiële "
        "basislijn en oversterfte-berekening. Data: CBS OpenData (CC-BY 4.0)."
    )

    # --- Data laden ---
    try:
        deaths_raw = load_deaths()
        pop_raw = load_population()
    except Exception as e:
        st.error(f"Fout bij laden van data: {e}")
        st.stop()

    # Beschikbare leeftijdsgroepen (intersectie deaths & pop)
    death_groups = set(deaths_raw["leeftijdsgroep"].unique())
    pop_groups = set(pop_raw["leeftijdsgroep"].unique())
    available_groups = sorted(death_groups & pop_groups)

    if not available_groups:
        st.error("Geen overeenkomende leeftijdsgroepen gevonden.")
        with st.expander("Debug info"):
            st.write("Groepen in overlijdensdata:", sorted(death_groups))
            st.write("Groepen in bevolkingsdata:", sorted(pop_groups))
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Instellingen")

        baseline_start = st.number_input(
            "Basislijn startjaar",
            min_value=1971, max_value=2019, value=2000, step=1,
        )
        baseline_end = st.number_input(
            "Basislijn eindjaar",
            min_value=1972, max_value=2019, value=2019, step=1,
        )

        if baseline_start >= baseline_end:
            st.error("Startjaar moet kleiner zijn dan eindjaar")
            st.stop()

        fit_type = st.selectbox(
            "Basislijn type",
            FIT_TYPES,
            index=0,
        )

        selected_group = st.selectbox(
            "Leeftijdsgroep (detail)",
            available_groups,
            index=0,
        )

        default_compare = [
            g for g in available_groups if "totaal" not in g.lower()
        ][:3]

        compare_groups = st.multiselect(
            "Vergelijk groepen",
            available_groups,
            default=default_compare,
        )

        min_year = st.number_input(
            "Data vanaf jaar",
            min_value=1971, max_value=2020, value=1995, step=1,
        )

        max_year_deaths = deaths_raw["jaar"].max()
        max_year_pop = pop_raw["jaar"].max()
        st.markdown(
            f"**Overlijdens t/m:** {max_year_deaths}  \n"
            f"**Bevolking t/m:** {max_year_pop}"
        )

    # Filter op jaar
    deaths = deaths_raw[deaths_raw["jaar"] >= min_year].copy()
    pop = pop_raw[pop_raw["jaar"] >= min_year].copy()

    # --- Analyse voor geselecteerde fit ---
    df_analysis = calculate_mortality_analysis(
        deaths, pop, selected_group, baseline_start, baseline_end, fit_type
    )

    if df_analysis is None:
        st.error(
            f"Analyse niet mogelijk voor '{selected_group}'. "
            "Controleer of er data beschikbaar is voor deze groep en periode."
        )
        st.stop()

    # --- Analyses voor alle drie fits (vergelijkingsgrafiek) ---
    df_exp = calculate_mortality_analysis(
        deaths, pop, selected_group, baseline_start, baseline_end, "exponentieel"
    )
    df_lin = calculate_mortality_analysis(
        deaths, pop, selected_group, baseline_start, baseline_end, "lineair"
    )
    df_kwa = calculate_mortality_analysis(
        deaths, pop, selected_group, baseline_start, baseline_end, "kwadratisch"
    )
    df_log = calculate_mortality_analysis(
        deaths, pop, selected_group, baseline_start, baseline_end, "logaritmisch"
    )

    # --- Grafieken ---

    # Rij 1: sterfte per 100k + absolute sterfte
   
    st.plotly_chart(
        plot_mortality_rate(df_analysis, selected_group, baseline_start, baseline_end),
        use_container_width=True,
    )

    st.plotly_chart(
        plot_absolute_mortality(df_analysis, selected_group, baseline_start, baseline_end),
        use_container_width=True,
    )

    # Rij 2: vergelijking alle drie basislijnen
    if df_exp is not None and df_lin is not None and df_kwa is not None:
        st.plotly_chart(
            plot_fit_comparison(
                df_exp, df_lin, df_kwa,df_log, selected_group, baseline_start, baseline_end
            ),
            use_container_width=True,
        )

    # Rij 3: oversterfte per 100k + absoluut
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            plot_excess_mortality(df_analysis, selected_group, per_100k=True),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            plot_excess_mortality(df_analysis, selected_group, per_100k=False),
            use_container_width=True,
        )

    # Rij 4: multi-groep vergelijking
    if compare_groups:
        st.plotly_chart(
            plot_multi_group(
                deaths, pop, compare_groups, baseline_start, baseline_end, fit_type
            ),
            use_container_width=True,
        )

    # --- Samenvatting recente jaren ---
    st.subheader("Samenvatting oversterfte recente jaren")
    recent = df_analysis[df_analysis["jaar"] >= 2020].copy()
    if not recent.empty:
        n_cols = min(len(recent), 7)
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
        st.metric("Totaal oversterfte 2020+", f"{totaal:,.0f}")

    # --- Data tabel ---
    with st.expander("📋 Data tabel"):
        display_cols = [
            "jaar", "leeftijdsgroep", "overlijdens", "bevolking",
            "sterfte_per_100k", "basislijn_per_100k", "basislijn_absoluut",
            "oversterfte_per_100k", "oversterfte_absoluut", "fit_type",
        ]
        avail = [c for c in display_cols if c in df_analysis.columns]
        fmt = {
            "overlijdens": "{:,.0f}",
            "bevolking": "{:,.0f}",
            "sterfte_per_100k": "{:.1f}",
            "basislijn_per_100k": "{:.1f}",
            "basislijn_absoluut": "{:,.0f}",
            "oversterfte_per_100k": "{:.1f}",
            "oversterfte_absoluut": "{:,.0f}",
        }
        fmt_ok = {k: v for k, v in fmt.items() if k in avail}
        st.dataframe(
            df_analysis[avail].style.format(fmt_ok),
            use_container_width=True,
            hide_index=True,
        )

    # --- Methodologie ---
    with st.expander("ℹ️ Methodologie"):
        st.markdown(f"""
**Databronnen:**
- Overlijdens: CBS tabel 70895NED (overledenen per jaar, geslacht en leeftijd)
- Bevolking: CBS bevolking per leeftijd, 1 januari

**Beschikbare basislijnmodellen:**
- **Lineair**: y = a + b·x
- **Kwadratisch**: y = a + b·x + c·x²
- **Exponentieel**: y = a·e^(b·x)
- **Logaritmisch**: y = a + b·ln(x+1)

**Methode:**
1. Overlijdens per leeftijdsgroep worden direct ingeladen uit de CBS pivot-tabel
2. Bevolkingsgrootte per leeftijdsgroep wordt geaggregeerd uit individuele leeftijden
3. Sterfte per 100.000 inwoners = (overlijdens / bevolking) × 100.000
4. Gekozen model wordt gefit op de basislijnperiode ({baseline_start}–{baseline_end})
5. Basislijn wordt geëxtrapoleerd naar alle jaren
6. Oversterfte = werkelijke sterfte − basislijn (zowel per 100k als absoluut)
7. Absolute oversterfte = oversterfte per 100k × bevolking / 100.000

**Kanttekeningen:**
- De keuze van basislijnperiode en modeltype beïnvloedt de resultaten sterk
- Vanaf 2010 telt CBS meer nagekomen berichten mee (mogelijke trendbreuk)
- Bevolkingsdata betreft stand op 1 januari
- Recente jaren kunnen voorlopige cijfers bevatten
        """)

if __name__ == "__main__":
    main()