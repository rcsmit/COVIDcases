import os
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pygam import LinearGAM, s


# -------------------- Data inlezen -------------------- #
@st.cache_data()
def get_dataframe(file: str, delimiter: str = ";") -> pd.DataFrame:
    """Get data from a file or url and return as a pandas DataFrame."""
    return pd.read_csv(file, delimiter=delimiter, low_memory=False)


# -------------------- GAM hulpmethoden -------------------- #

def make_results_table(
    geslacht: str,
    leeftijd: int,
    gam: LinearGAM,
    X_future: np.ndarray,
    d_future: pd.DataFrame,
    y_act: np.ndarray,
    y_pred: np.ndarray,
    df_subset: pd.DataFrame,
) -> pd.DataFrame:
    """Maak tabel met actual, voorspeld, CI en oversterfte in aantallen."""

    intervals = gam.prediction_intervals(X_future, width=0.95)

    res = pd.DataFrame(
        {
            "Geslacht": geslacht,
            "Leeftijd": leeftijd,
            "Jaar": d_future["Jaar"].to_numpy(),
            "actual_per100k": y_act,
            "pred_per100k": y_pred,
            "conf_low": intervals[:, 0],
            "conf_high": intervals[:, 1],
        }
    )

    res["afwijking_per100k"] = res["actual_per100k"] - res["pred_per100k"]
    res["rel_afwijking_pct"] = 100 * res["afwijking_per100k"] / res["pred_per100k"]

    # df_subset bevat al alleen deze leeftijd/geslacht
    # we hebben hier vooral Aantal per jaar nodig
    df_small = df_subset[["Jaar", "Aantal"]].drop_duplicates()

    res2 = df_small.merge(res, on="Jaar", how="right")
    res2["Oversterfte"] = res2["afwijking_per100k"] * res2["Aantal"] / 100_000

    return res2


def make_plot(
    d_train: pd.DataFrame,
    d_future: pd.DataFrame,
    y_train: np.ndarray,
    XX: np.ndarray,
    preds: np.ndarray,
    conf_low: np.ndarray,
    conf_high: np.ndarray,
    leeftijd: int,
    geslacht: str,
    train_end: int = 2019,
    pred_end: int = 2024,
) -> go.Figure:
    """Plot basisdata, toekomstdata, GAM-lijn en 95%-interval."""

    fig = go.Figure()

    # 1. zwarte punten: data t/m train_end
    fig.add_trace(
        go.Scatter(
            x=d_train["Jaar"],
            y=y_train,
            mode="markers",
            name=f"Data t/m {train_end}",
            marker=dict(color="black", size=7),
        )
    )

    # 2. rode punten: werkelijke data train_end+1 – pred_end
    if not d_future.empty:
        y_future = d_future["werkelijke_sterftekans"].to_numpy() * 100_000
        fig.add_trace(
            go.Scatter(
                x=d_future["Jaar"],
                y=y_future,
                mode="markers",
                name=f"Data {train_end+1}–{pred_end}",
                marker=dict(color="red", size=7),
                line=dict(color="red", width=2),
            )
        )

    # 3. lichtblauwe band: 95% interval
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([XX, XX[::-1]]),
            y=np.concatenate([conf_low, conf_high[::-1]]),
            fill="toself",
            fillcolor="rgba(135, 206, 250, 0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% CI",
        )
    )

    # 4. blauwe lijn: GAM trend
    fig.add_trace(
        go.Scatter(
            x=XX,
            y=preds,
            mode="lines",
            name=f"GAM trend (fit t/m {train_end})",
            line=dict(color="blue", width=3),
        )
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Deaths per 100k",
        title=f"GAM sterfte • Leeftijd {leeftijd} • {geslacht} (basis t/m {train_end})",
        template="simple_white",
    )
    return fig


def predict_oversterfte_gam(
    df: pd.DataFrame,
    leeftijd: int,
    geslacht: str,
    train_end: int = 2019,
    pred_end: int = 2024,
):
    """Fit GAM t/m train_end en voorspel tot pred_end voor één leeftijd/geslacht."""

    d_all = df[(df["Leeftijd"] == leeftijd) & (df["Geslacht"] == geslacht)].copy()
    d_all = d_all.sort_values("Jaar")

    # trainingdata t/m train_end
    d_train = d_all[d_all["Jaar"] <= train_end].copy()
    if d_train.empty:
        print (d_all)
        raise ValueError("Geen trainingsdata t/m train_end voor deze combinatie")

    y_train = d_train["werkelijke_sterftekans"].to_numpy() * 100_000
    X_train = d_train[["Jaar"]].to_numpy()

    # echte waarden voor 2020–pred_end
    jaren_eval = np.arange(train_end + 1, pred_end + 1)
    d_future = d_all[d_all["Jaar"].isin(jaren_eval)].copy().sort_values("Jaar")

    if d_future.empty:
        raise ValueError("Geen data voor jaren > train_end in deze combinatie")

    # GAM fit
    gam = LinearGAM(s(0, n_splines=5)).fit(X_train, y_train)

    # voorspellingsgrid voor gladde lijn
    jaar_start = int(d_train["Jaar"].min())
    XX = np.linspace(jaar_start, pred_end, 200)

    preds = gam.predict(XX)
    intervals = gam.prediction_intervals(XX, width=0.95)
    conf_low = intervals[:, 0]
    conf_high = intervals[:, 1]

    # voorspelling alleen op de jaren met echte data
    y_act = d_future["werkelijke_sterftekans"].to_numpy() * 100_000
    X_future = d_future[["Jaar"]].to_numpy()
    y_pred = gam.predict(X_future)

    return gam, d_train, d_future, y_train, XX, preds, conf_low, conf_high, y_act, y_pred, X_future


def calculate_and_plot__gam_sterfte(
    df: pd.DataFrame,
    leeftijd: int,
    geslacht: str,
    train_end: int = 2019,
    pred_end: int = 2024,
):
    """Bereken GAM-basis, afwijkingen en oversterfte en maak de Plotly figuur."""

    (
        gam,
        d_train,
        d_future,
        y_train,
        XX,
        preds,
        conf_low,
        conf_high,
        y_act,
        y_pred,
        X_future,
    ) = predict_oversterfte_gam(df, leeftijd, geslacht, train_end, pred_end)

    res2 = make_results_table(geslacht, leeftijd,gam, X_future, d_future, y_act, y_pred, df)
    oversterfte_totaal = float(res2["Oversterfte"].sum())

    fig = make_plot(
        d_train,
        d_future,
        y_train,
        XX,
        preds,
        conf_low,
        conf_high,
        leeftijd,
        geslacht,
        train_end,
        pred_end,
    )

    return fig, res2, oversterfte_totaal


# -------------------- Data ophalen en interface -------------------- #
@st.cache_data()
def get_data(geslacht: str, startjaar: int, leeftijd: int) -> pd.DataFrame:
    """Haal bevolking en overlijdens op en maak gecombineerde tabel voor één leeftijd/geslacht."""

    # Bevolking
    # https://opendata.cbs.nl/#/CBS/nl/dataset/7461bev/table?https:%2F%2Fopendata.cbs.nl%2F#%2FCBS%2Fnl%2Fdataset%2F03747%2Ftable%3Fts=1763998647352
    bevolking_ = get_dataframe(
        r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/Bevolking__geslacht__leeftijd_en_burgerlijke_staat__2024.csv",
        ",",
    )

    bevolking = bevolking_.melt(
        id_vars=["Geslacht", "Leeftijd"],
        var_name="Jaar",
        value_name="Aantal",
    )

    bevolking = bevolking[["Leeftijd", "Geslacht", "Jaar", "Aantal"]]
    bevolking["Jaar"] = bevolking["Jaar"].astype(int)

    # Overlijdens
    # https://opendata.cbs.nl/#/CBS/nl/dataset/37168/table?ts=1764041242474
    overlijdens_ = get_dataframe(
        r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_geslacht_leeftijd_burgelijkstaat2024.csv",
        ",",
    )

    overlijdens = overlijdens_.melt(
        id_vars=["Leeftijd", "Jaar"],
        value_vars=["Mannen", "Vrouwen"],
        var_name="Geslacht",
        value_name="OverledenenLeeftijdBijOverlijden_1",
    )

    totaal_tabel = bevolking.merge(
        overlijdens,
        on=["Jaar", "Leeftijd", "Geslacht"],
        how="right",
    )

    subset = totaal_tabel[
        (totaal_tabel["Leeftijd"] == leeftijd)
        & (totaal_tabel["Jaar"] >= startjaar)
        & (totaal_tabel["Geslacht"] == geslacht)
    ].copy()

    subset["werkelijke_sterftekans"] = (
        subset["OverledenenLeeftijdBijOverlijden_1"] / subset["Aantal"]
    )

    return subset


def interface():
    col1, col2, col3, _ = st.columns(4)

    with col1:
        geslacht = st.selectbox("Geslacht", ["Mannen", "Vrouwen"])
    with col2:
        startjaar = st.number_input("Start year", min_value=2000, max_value=2019, value=2000)
    with col3:
        leeftijd = st.number_input("Leeftijd", min_value=0, max_value=105, value=40)

    return geslacht, startjaar, leeftijd


# -------------------- Streamlit app -------------------- #

def main():
    st.header("Oversterfte berekening met GAM.")

    geslacht, startjaar, leeftijd = interface()
    totaal_tabel_leeftijd = get_data(geslacht, startjaar, leeftijd)

    # fig, res, oversterfte = calculate_and_plot__gam_sterfte(
    #     totaal_tabel_leeftijd,
    #     leeftijd=leeftijd,
    #     geslacht=geslacht,
    # )

    (
        gam,
        d_train,
        d_future,
        y_train,
        XX,
        preds,
        conf_low,
        conf_high,
        y_act,
        y_pred,
        X_future,
    ) = predict_oversterfte_gam(totaal_tabel_leeftijd, leeftijd, geslacht)

    res2 = make_results_table(geslacht, leeftijd,gam, X_future, d_future, y_act, y_pred, totaal_tabel_leeftijd)
    oversterfte = float(res2["Oversterfte"].sum())

    fig = make_plot(
        d_train,
        d_future,
        y_train,
        XX,
        preds,
        conf_low,
        conf_high,
        leeftijd,
        geslacht,
        
    )

    st.plotly_chart(fig)
 
    st.metric("Totale oversterfte 2020–2024", int(oversterfte))
    st.write(res2)

def make_scatter(eindtabel):
    fig_scatter = go.Figure()

    for g, kleur in [("Mannen", "blue"), ("Vrouwen", "red")]:
        df_g = eindtabel[eindtabel["Geslacht"] == g]

        fig_scatter.add_trace(
            go.Scatter(
                x=df_g["Leeftijd"],
                y=df_g["Oversterfte"],
                mode="markers",
                text=df_g["Jaar"],  # hier stop je Jaar in
                hovertemplate=(
                    "Leeftijd: %{x}<br>"
                    "Oversterfte: %{y:.0f}<br>"
                    "Jaar: %{text}<extra></extra>"
                ),name=g,
                line=dict(color=kleur),
                marker=dict(size=7),
            )
        )

    fig_scatter.update_layout(
        xaxis_title="Leeftijd",
        yaxis_title="Oversterfte 2020–2024",
        title="Oversterfte 2020–2024 per leeftijd en geslacht",
        template="simple_white",
    )

    st.plotly_chart(fig_scatter)

def show_metrics(eindtabel):
    tot_mannen = eindtabel.loc[
        eindtabel["Geslacht"] == "Mannen", "Oversterfte"
    ].sum()

    tot_vrouwen = eindtabel.loc[
        eindtabel["Geslacht"] == "Vrouwen", "Oversterfte"
    ].sum()

    tot_totaal = tot_mannen + tot_vrouwen

    col1, col2, col3 = st.columns(3)
   
    def fmt(x):
        return f"{x:,.0f}".replace(",", ".")

    col1.metric("Oversterfte mannen 2020–2024", fmt(tot_mannen))
    col2.metric("Oversterfte vrouwen 2020–2024", fmt(tot_vrouwen))
    col3.metric("Totale oversterfte 2020–2024", fmt(tot_totaal))


def plot_afwijking_leeftijd(eindtabel_afwijking_geslacht):
    fig = go.Figure()

    for g, kleur in [("Mannen", "blue"), ("Vrouwen", "red")]:
        df_g = eindtabel_afwijking_geslacht[eindtabel_afwijking_geslacht["Geslacht"] == g].sort_values("Leeftijd")

        fig.add_trace(
            go.Scatter(
                x=df_g["Leeftijd"],
                y=df_g["afwijking_per100k"],
                mode="lines+markers",
                name=g,
                line=dict(color=kleur),
                marker=dict(size=7),
                hovertemplate=(
                    "Leeftijd: %{x}<br>"
                    "Gem. afwijking: %{y:.2f} per 100k<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Leeftijd",
        yaxis_title="Gemiddelde afwijking per 100k (2020–2024)",
        title="Gemiddelde afwijking per leeftijd en geslacht",
        template="simple_white",
    )

    st.plotly_chart(fig)

def main_2():
    st.header("Oversterfte berekening met GAM ")
    startjaar = st.number_input("Start year", min_value=2000, max_value=2019, value=2000, key="main2_startjaar")
    eindtabel = pd.DataFrame()
    placeholder = st.empty()

    for geslacht in ["Vrouwen", "Mannen"]:
        for leeftijd in range(0,105):
            placeholder.text(f"Bezig met berekening • Geslacht: {geslacht} • Leeftijd: {leeftijd}/105")
            
            totaal_tabel_leeftijd = get_data(geslacht, startjaar, leeftijd) 
       
            (
                gam,
                d_train,
                d_future,
                y_train,
                XX,
                preds,
                conf_low,
                conf_high,
                y_act,
                y_pred,
                X_future,
            ) = predict_oversterfte_gam(totaal_tabel_leeftijd, leeftijd, geslacht)

            res2 = make_results_table(geslacht, leeftijd, gam, X_future, d_future, y_act, y_pred, totaal_tabel_leeftijd)
            oversterfte = float(res2["Oversterfte"].sum())
            # newrow = pd.DataFrame({
            #     "Leeftijd": [leeftijd],
            #     "Geslacht": [geslacht],
            #     "Oversterfte_2020_2024": [oversterfte]
            # })
            #print (f"Leeftijd: {leeftijd} • Geslacht: {geslacht} • Oversterfte 2020–2024: {oversterfte:.0f}")
            eindtabel = pd.concat([eindtabel, res2], ignore_index=True)
     # Zorg dat leeftijden netjes oplopen
    placeholder.empty()
    oversterfte = float(eindtabel["Oversterfte"].sum())
    eindtabel = eindtabel.sort_values(["Geslacht", "Leeftijd"]) 
   
    eindtabel_total_geslacht=eindtabel.groupby(["Jaar","Geslacht"])["Oversterfte"].sum().reset_index()
    eindtabel_total_leeftijd_geslacht=eindtabel.groupby(["Leeftijd","Geslacht"])["Oversterfte"].sum().reset_index()
    eindtabel_total_leeftijd_geslacht["Jaar"] = "2020-2024"
    eindtabel_total_geslacht_pivot=eindtabel_total_geslacht.pivot(index="Jaar", columns="Geslacht", values="Oversterfte").round().astype(int).reset_index()
    eindtabel_total_geslacht_pivot["Totaal"] = eindtabel_total_geslacht_pivot["Mannen"] + eindtabel_total_geslacht_pivot["Vrouwen"]
    
    # make_scatter(eindtabel)
    make_scatter(eindtabel_total_leeftijd_geslacht)
    show_metrics(eindtabel)
    st.write(eindtabel_total_geslacht_pivot)

    # st.write(eindtabel)
    eindtabel_afwijking_geslacht=eindtabel.groupby(["Leeftijd","Geslacht"])["afwijking_per100k"].mean().reset_index()
    # st.write(eindtabel_afwijking_geslacht)
    plot_afwijking_leeftijd(eindtabel_afwijking_geslacht)


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")
    print(f"--------------{datetime.datetime.now()}-------------------------")
    # main()
    # main_2()
    tab1,tab2= st.tabs(["Enkele leeftijd/geslacht", "Alle leeftijden/geslacht"])
    with tab1:
        main()
       
    with tab2:
        main_2()
        