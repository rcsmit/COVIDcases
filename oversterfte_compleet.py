import pandas as pd
import cbsodata
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import get_rioolwater

# from streamlit import caching
import scipy.stats as stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import datetime
import statsmodels.api as sm


import statsmodels.api as sm

# Impact of different mortality forecasting methods and explicit assumptions on projected future
# life expectancy
# Stoeldraijer, L.; van Duin, C.; van Wissen, L.J.G.; Janssen, F
# https://pure.rug.nl/ws/portalfiles/portal/13869387/stoeldraijer_et_al_2013_DR.pdf


# 70895ned = https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1659307527578
# Overledenen; geslacht en leeftijd, per week

# Downloaden van tabeloverzicht
# toc = pd.DataFrame(cbsodata.get_table_list())
try:
    st.set_page_config(layout="wide")
except:
    pass

# TO SUPRESS
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
pd.options.mode.chained_assignment = None


def interface():
    how = st.sidebar.selectbox(
        "How",
        [
            "quantiles",
            "Lines",
            "over_onder_sterfte",
            "meer_minder_sterfte",
            "year_minus_avg",
            "p_score",
        ],
        index=0,
    )
    yaxis_to_zero = st.sidebar.selectbox("Y as beginnen bij 0", [False, True], index=0)
    if (
        (how == "year_minus_avg")
        or (how == "p_score")
        or (how == "over_onder_sterfte")
        or (how == "meer_minder_sterfte")
    ):
        rightax = st.sidebar.selectbox(
            "Right-ax",
            ["boosters", "herhaalprik", "herfstprik", "rioolwater", "kobak", None],
            index=1,
            key="aa",
        )
        mergetype = st.sidebar.selectbox(
            "How to merge", ["inner", "outer"], index=0, key="bb"
        )
        sec_y = st.sidebar.selectbox(
            "Secondary Y axis", [True, False], index=0, key="cc"
        )
    else:
        rightax = None
        mergetype = None
        sec_y = None
    return how, yaxis_to_zero, rightax, mergetype, sec_y


def calculate_year_data(df_merged, year, show_official, series_name, smooth):
    st.subheader(year)
    if year != "All":
        df_merged_jaar = df_merged[df_merged["jaar_x_x"] == year].copy()
    else:
        df_merged_jaar = df_merged.copy()

    df_merged_jaar["verw_cbs"] = df_merged_jaar["avg"]
    for n in ["cbs", "rivm"]:
        df_merged_jaar[f"oversterfte_{n}_simpel"] = (
            df_merged_jaar["aantal_overlijdens"] - df_merged_jaar[f"verw_{n}"]
        )
        df_merged_jaar[f"oversterfte_{n}_simpel_cumm"] = df_merged_jaar[
            f"oversterfte_{n}_simpel"
        ].cumsum()

        df_merged_jaar[f"oversterfte_{n}_complex"] = np.where(
            df_merged_jaar["aantal_overlijdens"] > df_merged_jaar[f"high_{n}"],
            df_merged_jaar["aantal_overlijdens"] - df_merged_jaar[f"high_{n}"],
            np.where(
                df_merged_jaar["aantal_overlijdens"] < df_merged_jaar[f"low_{n}"],
                df_merged_jaar["aantal_overlijdens"] - df_merged_jaar[f"low_{n}"],
                0,
            ),
        )

        df_merged_jaar[f"oversterfte_{n}_middel"] = np.where(
            df_merged_jaar["aantal_overlijdens"] > df_merged_jaar[f"high_{n}"],
            df_merged_jaar["aantal_overlijdens"] - df_merged_jaar[f"verw_{n}"],
            np.where(
                df_merged_jaar["aantal_overlijdens"] < df_merged_jaar[f"low_{n}"],
                df_merged_jaar["aantal_overlijdens"] - df_merged_jaar[f"verw_{n}"],
                0,
            ),
        )

        df_merged_jaar[f"oversterfte_{n}_complex_cumm"] = df_merged_jaar[
            f"oversterfte_{n}_complex"
        ].cumsum()
        df_merged_jaar[f"oversterfte_{n}_middel_cumm"] = df_merged_jaar[
            f"oversterfte_{n}_middel"
        ].cumsum()

    columnlist = [
        "low_cbs",
        "verw_cbs",
        "high_cbs",
        "q05",
        "q25",
        "verw_cbs",
        "q75",
        "q95",
    ]
    for what_to_sma in columnlist:
        if smooth :
            df_merged_jaar = rolling(df_merged_jaar, f"{what_to_sma}")
        else:
            df_merged_jaar[f'{what_to_sma}_sma'] = df_merged_jaar[what_to_sma]
    return df_merged_jaar


def display_cumulative_oversterfte(df_merged_jaar, year):

    cbs_middel = df_merged_jaar["oversterfte_cbs_middel_cumm"].iloc[-1]
    cbs_simpel = df_merged_jaar["oversterfte_cbs_simpel_cumm"].iloc[-1]
    cbs_complex = df_merged_jaar["oversterfte_cbs_complex_cumm"].iloc[-1]

    rivm_middel = df_merged_jaar["oversterfte_rivm_middel_cumm"].iloc[-1]
    rivm_simpel = df_merged_jaar["oversterfte_rivm_simpel_cumm"].iloc[-1]
    rivm_complex = df_merged_jaar["oversterfte_rivm_complex_cumm"].iloc[-1]
    try:
        simpel_str = f"Simpel: rivm: {int(rivm_simpel)} | cbs: {int(cbs_simpel)} | verschil {int(rivm_simpel-cbs_simpel)}"
        middel_str = f"Middel: rivm: {int(rivm_middel)} | cbs: {int(cbs_middel)} | verschil {int(rivm_middel-cbs_middel)}"
        complex_str = f"Complex: rivm: {int(rivm_complex)} | cbs: {int(cbs_complex)} | verschil {int(rivm_complex-cbs_complex)}"
        texts = [simpel_str, middel_str, complex_str]
    except:
        texts = [None, None, None]
    temp1 = [None, None, None]
    col1, col2, col3 = st.columns(3)
    temp1[0], temp1[1], temp1[2] = col1, col2, col3

    for i, p in enumerate(["simpel", "middel", "complex"]):
        with temp1[i]:
            fig = go.Figure()
            for n in ["rivm", "cbs"]:
                fig.add_trace(
                    go.Scatter(
                        x=df_merged_jaar["weeknr"],
                        y=df_merged_jaar[f"oversterfte_{n}_{p}_cumm"],
                        mode="lines",
                        name=f"cummulatieve oversterfte {n}",
                    )
                )

            fig.update_layout(
                title=f"Cumm oversterfte ({p}) - {year}",
                xaxis_title="Tijd",
                yaxis_title="Aantal",
            )

            st.plotly_chart(fig)
            st.write(texts[i])


def display_results(df_merged_jaar, year):
    st.subheader(f"Results - {year}")

    df_grouped = df_merged_jaar.groupby(by="jaar_x_x").sum().reset_index()
    df_grouped = df_grouped[
        [
            "jaar_x_x",
            "oversterfte_rivm_simpel",
            "oversterfte_rivm_middel",
            "oversterfte_rivm_complex",
            "oversterfte_cbs_simpel",
            "oversterfte_cbs_middel",
            "oversterfte_cbs_complex",
        ]
    ]

    for x in ["simpel", "middel", "complex"]:
        df_grouped[f"verschil_{x}"] = (
            df_grouped[f"oversterfte_rivm_{x}"] - df_grouped[f"oversterfte_cbs_{x}"]
        )

    _df_grouped_transposed = df_grouped.transpose().astype(int)

    if year == "All":
        st.write(_df_grouped_transposed)
    else:
        new_data = {
            "rivm": {
                "simpel": df_grouped["oversterfte_rivm_simpel"].iloc[0],
                "middel": df_grouped["oversterfte_rivm_middel"].iloc[0],
                "complex": df_grouped["oversterfte_rivm_complex"].iloc[0],
            },
            "cbs": {
                "simpel": df_grouped["oversterfte_cbs_simpel"].iloc[0],
                "middel": df_grouped["oversterfte_cbs_middel"].iloc[0],
                "complex": df_grouped["oversterfte_cbs_complex"].iloc[0],
            },
            "verschil": {
                "simpel": df_grouped["verschil_simpel"].iloc[0],
                "middel": df_grouped["verschil_middel"].iloc[0],
                "complex": df_grouped["verschil_complex"].iloc[0],
            },
        }

        new_df_grouped = pd.DataFrame(new_data).transpose().astype(int)
        st.write(new_df_grouped)


def make_df_merged(df_data, df_rivm, series_name):
    _, df_corona, df_quantile = make_df_quantile(series_name, df_data)
    df_official = get_df_offical()

    df_merged = df_corona.merge(df_quantile, left_on="weeknr", right_on="weeknr").merge(
        df_rivm, on="weeknr", how="outer"
    )
    df_merged = df_merged.merge(
        df_official, left_on="weeknr", right_on="weeknr_z", how="outer"
    )
    df_merged = df_merged.drop(columns=["week_"])

    df_merged["shifted_jaar"] = df_merged["jaar_x_x"]  # .shift(28)
    df_merged["shifted_week"] = df_merged["weeknr"]  # .shift(28)

    columns = [
        [series_name, "aantal_overlijdens"],
        ["q50", "verw_cbs_q50"],
        ["low05", "low_cbs"],
        ["avg_", "avg"],
        ["high95", "high_cbs"],
        ["voorspeld", "verw_rivm"],
        ["lower_ci", "low_rivm"],
        ["upper_ci", "high_rivm"],
    ]
    # ["avg_", "verw_cbs"],

    for c in columns:
        df_merged = df_merged.rename(columns={c[0]: c[1]})
    return df_merged


def plot_steigstra(df_transformed, series_name):
    # replicatie van https://twitter.com/SteigstraHerman/status/1801641074336706839

    # Pivot table
    df_pivot = df_transformed.set_index("week")

    # Function to transform the DataFrame
    def create_spaghetti_data(df, year1, year2=None):
        part1 = df.loc[32:52, year1]
        if year2 is not None:
            part2 = df.loc[1:31, year2]
            combined = pd.concat([part1, part2]).reset_index(drop=True)
        else:
            combined = part1.reset_index(drop=True)
        return combined

    # Create the spaghetti data
    years = df_pivot.columns[:-3] 

    years = list(map(int, years))

    spaghetti_data = {
        year: create_spaghetti_data(df_pivot, year, year + 1) if (year + 1) in years else create_spaghetti_data(df_pivot, year)
        for year in years
    }
    df_spaghetti = pd.DataFrame(spaghetti_data)

    df_spaghetti = df_spaghetti.cumsum(axis=0)

    # Generate the sequence from 27 to 52 followed by 1 to 26
    sequence = list(range(32, 53)) + list(range(1, 32))
    # Add the sequence as a new column
    df_spaghetti["weeknr_real"] = sequence

    df_spaghetti["average"] = df_spaghetti.iloc[:, :4].mean(axis=1)

    # Calculate low and high (mean - 1.96*std and mean + 1.96*std) for the first 5 columns
    df_spaghetti["low"] = df_spaghetti["average"] - 1.96 * df_spaghetti.iloc[:, :4].std(
        axis=1
    )
    df_spaghetti["high"] = df_spaghetti["average"] + 1.96 * df_spaghetti.iloc[
        :, :4
    ].std(axis=1)
    #st.write(df_spaghetti)
    # Plotting with Plotly
    fig = go.Figure()
    fig.add_vline(x=25, name="week 1", line=dict(color="gray", width=1, dash="dash"))
    fig.add_hline(y=0, line=dict(color="black", width=2))

    for year in df_spaghetti.columns[:-4]:
        fig.add_trace(
            go.Scatter(
                x=df_spaghetti.index,
                y=df_spaghetti[year],
                mode="lines",
                name=f"{year} - {year+1}",
            )
        )

    fig.add_trace(
        go.Scatter(
            name="low",
            x=df_spaghetti.index,
            y=df_spaghetti["low"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
            fillcolor="rgba(68, 68, 68, 0.2)",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="high",
            x=df_spaghetti.index,
            y=df_spaghetti["high"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255, 188, 0, 0.1)"),
            fill="tonexty",
            fillcolor="rgba(68, 68, 68, 0.2)",
        )
    )



    # Update layout to customize x-axis labels
    fig.update_layout(
        title=f"Steigstra Plot of {series_name} over Different Years week 32 to week 31",
        xaxis_title="Weeks (32 to 31)",
        yaxis_title="Values",
        xaxis=dict(
            tickvals=df_spaghetti.index,  # Set the tick positions to the DataFrame index
            ticktext=df_spaghetti['weeknr_real'].astype(str)  # Set the tick labels to 'weeknr_real'
        )
    )

    st.plotly_chart(fig)


def calculate_steigstra(df_merged, series_naam, cumm=False, m="cbs"):

    # Set 'week' to 52 if 'weeknr' is '2022_52'

    # Get list of current columns excluding "shifted_jaar"
    other_columns = [col for col in df_merged.columns if col != "shifted_jaar"]
    # Reorder columns to put "shifted_jaar" at the beginning
    new_columns = ["shifted_jaar"] + other_columns

    # Reindex DataFrame with new column order
    df_merged = df_merged.reindex(columns=new_columns)

    df_merged["verw_cbs"] = df_merged["avg"]

    df_compleet = pd.DataFrame()
    for year in range(2015, 2025):
        df_merged_jaar = df_merged[df_merged["jaar_x_x"] == year].copy()
        for n in ["cbs"]:
            df_merged_jaar[f"oversterfte_{n}_simpel"] = (
                df_merged_jaar["aantal_overlijdens"] - df_merged_jaar[f"verw_{n}"]
            )
            df_merged_jaar[f"oversterfte_{n}_simpel_cumm"] = df_merged_jaar[
                f"oversterfte_{n}_simpel"
            ].cumsum()
            df_compleet = pd.concat([df_compleet, df_merged_jaar])

    df_compleet["shifted_jaar"] = df_compleet["jaar_x_x"]  # .shift(24)

    if cumm:
        df = df_compleet.pivot(
            index=["week"],
            columns="shifted_jaar",
            values=f"oversterfte_{m}_simpel_cumm",
        ).reset_index()
    else:
        df = df_compleet.pivot(
            index=["week"], columns="shifted_jaar", values=f"oversterfte_{m}_simpel"
        ).reset_index()

    # Calculate average and standard deviation
    df["average"] = df.mean(axis=1)

    # Calculate low and high (mean - 1.96*std and mean + 1.96*std)
    df["low"] = df["average"] - 1.96 * df.std(axis=1)
    df["high"] = df["average"] + 1.96 * df.std(axis=1)

    return df


def comparison(df_merged, series_name, smooth):
    
    show_official = st.sidebar.selectbox("Show official values", [True, False], 0)
    st.subheader("Vergelijking")

    for year in ["All", 2020, 2021, 2022, 2023, 2024]:
        expanded = True if year == "All" else False
        
        with st.expander(f"{year}", expanded=expanded):
            df_merged_jaar = calculate_year_data(
                df_merged, year, show_official, series_name, smooth
            )
            show_difference(df_merged_jaar, "weeknr", show_official, year)

            display_cumulative_oversterfte(df_merged_jaar, year)
            display_results(df_merged_jaar, year)


def filter_rivm(df, series_name, y):
    """

        # replicating https://www.rivm.nl/monitoring-sterftecijfers-nederland

        # Doorgetrokken lijn: gemelde sterfte tot en met 2 juni 2024.

        # Band: het aantal overlijdens dat het RIVM verwacht.
        # Dit is gebaseerd op cijfers uit voorgaande jaren.
        # Als de lijn hoger ligt dan de band, overleden er meer mensen dan verwacht.
        # De band geeft de verwachte sterfte weer tussen een bovengrens en een ondergrens.
        # De bovengrens is de verwachte sterfte plus twee standaarddeviaties ten opzichte
        # van de verwachte sterfte. De ondergrens is de verwachte sterfte min twee standaarddeviaties
        # ten opzichte van de verwachte sterfte. Dit betekent dat 95% van de cijfers van de afgelopen
        # vijf jaar (met uitzondering van de pieken)2 in de band zat.

        # De gestippelde lijn geeft schattingen van de sterftecijfers voor de 6 meest recente weken.
        # Deze cijfers kunnen nog veranderen. Gemeenten geven hun sterfgevallen door aan het CBS.
        # Daar zit meestal enkele dagen vertraging in. Dat zorgt voor een vertekend beeld.
        # Om dat tegen te gaan, zijn de al gemelde sterftecijfers voor de laatste 6 weken opgehoogd.
        # Voor deze ophoging kijkt het RIVM naar het patroon van de vertragingen in de meldingen
        # van de sterfgevallen in de afgelopen weken.
        # Het RIVM berekent elk jaar in de eerste week van juli de verwachte sterfte
        # voor het komende jaar. Hiervoor gebruiken we de sterftecijfers van de afgelopen
        # vijf jaar. Om vertekening van de verwachte sterfte te voorkomen, tellen we
        # eerdere pieken niet mee. Deze pieken vallen vaak samen met koude- en hittegolven of
        # uitbraken van infectieziekten. Het gaat hierbij om de 25% hoogste sterftecijfers
        # van de afgelopen vijf jaar en de 20% hoogste sterftecijfers in juli en augustus.
        # De berekening maakt gebruik van een lineair regressiemodel met een lineaire tijdstrend
        # en sinus/cosinus-termen om mogelijke seizoensschommelingen te beschrijven.
        # Als de sterfte hoger is dan 2 standaarddeviaties boven de verwachte sterfte,
        # noemen we de sterfte licht verhoogd. Bij 3 standaarddeviaties noemen we de sterfte
        # verhoogd. Bij 4 of meer standaarddeviaties noemen we de sterfte sterk verhoogd.
        #
        # Hiervoor gebruiken we de sterftecijfers van de afgelopen
    vijf jaar. Om vertekening van de verwachte sterfte te voorkomen, tellen we
    eerdere pieken niet mee. Deze pieken vallen vaak samen met koude- en hittegolven of
    uitbraken van infectieziekten. Het gaat hierbij om de 25% hoogste sterftecijfers
    van de afgelopen vijf jaar en de 20% hoogste sterftecijfers in juli en augustus.

    Resultaat vd functie is een df van de 5 jaar voór jaar y (y=2020: 2015-2019) met
    de gefilterde waardes

    """
    # Selecteer de gegevens van de afgelopen vijf jaar
    recent_years = df["boekjaar"].max() - 6

    df = df[(df["boekjaar"] >= recent_years) & (df["boekjaar"] < y)]

    # Bereken de drempelwaarde voor de 25% hoogste sterftecijfers van de afgelopen vijf jaar
    threshold_25 = df[series_name].quantile(0.75)

    # Filter de data voor juli en augustus (weken 27-35)
    summer_data = df[df["week"].between(27, 35)]
    threshold_20 = summer_data[series_name].quantile(0.80)
    # st.write(f"drempelwaarde voor de 25% hoogste sterftecijfers : {threshold_25=} /  drempelwaarde voor 20% hoogste sterftecijfers in juli en augustus {threshold_20=}")
    set_to_none = False
    if set_to_none:
        # de 'ongeldige waardes' worden vervangen door None
        df.loc[df["jaar"] >= recent_years, series_name] = df.loc[
            df["jaar"] >= recent_years, series_name
        ].apply(lambda x: np.nan if x > threshold_25 else x)
        df.loc[df["week"].between(27, 35), series_name] = df.loc[
            df["week"].between(27, 35), series_name
        ].apply(lambda x: np.nan if x > threshold_20 else x)
    else:
        # verwijder de rijen met de 'ongeldige waardes'
        df = df[~((df["jaar"] >= recent_years) & (df[series_name] > threshold_25))]
        df = df[~((df["week"].between(27, 35)) & (df[series_name] > threshold_20))]

    return df


def add_columns_lin_regression(df):
    """voeg columns tijd, sin en cos toe. de sinus/cosinus-termen zijn om mogelijke
    seizoensschommelingen te beschrijven
    """
    # Maak een tijdsvariabele
    df["tijd"] = df["boekjaar"] + (df["boekweek"] - 1) / 52

    # Voeg sinus- en cosinustermen toe voor seizoensgebondenheid (met een periode van 1 jaar)

    df.loc[:, "sin"] = np.sin(2 * np.pi * df["boekweek"] / 52)
    df.loc[:, "cos"] = np.cos(2 * np.pi * df["boekweek"] / 52)
    return df


def filter_period(df_new, start_year, start_week, end_year, end_week, add):
    """Filter de dataframe en dus de grafiek in tijd (x-as)"""

    condition1 = (df_new[f"jaar{add}"] == start_year) & (
        df_new[f"week{add}"] >= start_week
    )
    condition2 = (df_new[f"jaar{add}"] >= start_year) & (
        df_new[f"jaar{add}"] <= end_year
    )
    condition3 = (df_new[f"jaar{add}"] == end_year) & (df_new[f"week{add}"] <= end_week)

    # Rijen selecteren die aan een van deze voorwaarden voldoen
    df_new = df_new[condition1 | condition2 | condition3]
    return df_new


def get_data_rivm():
    """laad de waardes zoals RIVM die heeft vastgesteld (11 juni 2024)

    Returns:
        _df: df
    """
    url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rivm_sterfte.csv"
    df_ = pd.read_csv(
        url,
        delimiter=";",
        low_memory=False,
    )
    return df_


def do_lin_regression(df_filtered, df_volledig, series_naam, y):
    """lineair regressiemodel met een lineaire tijdstrend
        en sinus/cosinus-termen om mogelijke seizoensschommelingen te beschrijven.
    Args:
        df_filtered : df zonder de uitschieters
        df_volledig : volledige df
        series_naam (_type_): welke serie
        y : jaar

    Returns:
        df : (volledige) df met de CI's
    """
    df_volledig = add_columns_lin_regression(df_volledig)
    df_filtered = add_columns_lin_regression(df_filtered)

    # Over een tijdvak [j-6-tot j-1] wordt per week wordt de standard deviatie berekend.
    # Hier wordt dan het gemiddelde van genomen

    weekly_std = df_filtered.groupby("boekweek")[series_naam].std().reset_index()
    weekly_std.columns = ["week", "std_dev"]
    sd = weekly_std["std_dev"].mean()
    # st.write(f"Standard deviatie = {sd}")

    X = df_filtered[["tijd", "sin", "cos"]]
    X = sm.add_constant(X)  # Voegt een constante term toe aan het model
    y = df_filtered[f"{series_naam}"]

    model = sm.OLS(y, X).fit()

    X2 = df_volledig[["tijd", "sin", "cos"]]
    X2 = sm.add_constant(X2)

    df_volledig.loc[:, "voorspeld"] = model.predict(X2)
    ci_model = False
    if ci_model:
        # Geeft CI van de voorspelde waarde weer. Niet de CI van de meetwaardes
        voorspellings_interval = model.get_prediction(X2).conf_int(alpha=0.05)
        df_volledig.loc[:, "lower_ci"] = voorspellings_interval[:, 0]
        df_volledig.loc[:, "upper_ci"] = voorspellings_interval[:, 1]
    else:
        df_volledig.loc[:, "lower_ci"] = df_volledig["voorspeld"] - 2 * sd
        df_volledig.loc[:, "upper_ci"] = df_volledig["voorspeld"] + 2 * sd
    df_new = pd.merge(df_filtered, df_volledig, on="weeknr", how="outer")

    df_new = df_new.sort_values(by=["jaar_y", "week_y"]).reset_index(drop=True)

    return df_new


def plot_graph_rivm(df_, series_naam, rivm):
    """plot the graph

    Args:
        df_ (str): _description_
        series_naam (str): _description_
        rivm (bool): show the official values from the RIVM graph
                        https://www.rivm.nl/monitoring-sterftecijfers-nederland
    """
    st.subheader("RIVM methode")
    df_rivm = get_data_rivm()

    df = pd.merge(df_, df_rivm, on="weeknr", how="outer")
    df = df.sort_values(by=["weeknr"])  # .reset_index()

    # Maak een interactieve plot met Plotly
    fig = go.Figure()

    # Voeg de werkelijke data toe
    fig.add_trace(
        go.Scatter(
            x=df["weeknr"],
            y=df[f"{series_naam}_y"],
            mode="lines",
            name="Werkelijke data cbs",
        )
    )

    # Voeg de voorspelde lijn toe
    fig.add_trace(
        go.Scatter(
            x=df["weeknr"], y=df["voorspeld"], mode="lines", name="Voorspeld model"
        )
    )
    if rivm == True:
        # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(
            go.Scatter(
                x=df["weeknr"],
                y=df["verw_waarde_rivm"],
                mode="lines",
                name="Voorspeld RIVM",
            )
        )
        # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(
            go.Scatter(
                x=df["weeknr"],
                y=df["ondergrens_verwachting_rivm"],
                mode="lines",
                name="onder RIVM",
            )
        )  # Voeg de voorspelde lijn RIVM toe
        fig.add_trace(
            go.Scatter(
                x=df["weeknr"],
                y=df["bovengrens_verwachting_rivm"],
                mode="lines",
                name="boven RIVM",
            )
        )

    # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(
        go.Scatter(
            x=df["weeknr"],
            y=df["upper_ci"],
            mode="lines",
            fill=None,
            line_color="lightgrey",
            name="Bovenste CI",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["weeknr"],
            y=df["lower_ci"],
            mode="lines",
            fill="tonexty",  # Vul het gebied tussen de lijnen
            line_color="lightgrey",
            name="Onderste CI",
        )
    )

    # Titel en labels toevoegen
    fig.update_layout(
        title="Voorspelling van Overledenen met 95% Betrouwbaarheidsinterval RIVM",
        xaxis_title="Tijd",
        yaxis_title="Aantal Overledenen",
    )

    st.plotly_chart(fig)


def verwachte_sterfte_rivm(df, series_naam):
    """Verwachte sterfte/baseline  uitrekenen volgens RIVM methode

    _
    """

    # adding week 52, because its not in the data
    # based on the rivm-data, we assume that the numbers are quit the same

    df["boekjaar"] = df["jaar"].shift(26)
    df["boekweek"] = df["week"].shift(26)

    df_compleet = pd.DataFrame()
    for y in [2019, 2020, 2021, 2022, 2023,2024]:
        # we filteren 5 jaar voor jaar y (y=2020: 2015 t/m 2020 )
        recent_years = y - 5
        df_ = df[(df["boekjaar"] >= recent_years) & (df["boekjaar"] <= y)]

        df_volledig = df_[
            ["weeknr", "jaar", "week", "boekjaar", "boekweek", series_naam]
        ]
        df_filtered = filter_rivm(df_, series_naam, y)

        df_do_lin_regression = do_lin_regression(
            df_filtered, df_volledig, series_naam, y
        )
        df_do_lin_regression = df_do_lin_regression[
            (df_do_lin_regression["boekjaar_y"] == y)
        ]
        df_compleet = pd.concat([df_compleet, df_do_lin_regression])
    return df_compleet

def get_boosters():
    """_summary_

    Returns:
        _type_: _description_
    """
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\boosters_per_week_per_leeftijdscat.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/boosters_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )
    df_["weeknr"] = (
        df_["jaar"].astype(str) + "_" + df_["weeknr"].astype(str).str.zfill(2)
    )
    df_ = df_.drop("jaar", axis=1)

    df_["boosters_m_v_0_64"] = df_["boosters_m_v_0_49"] + df_["boosters_m_v_50_64"]
    df_["boosters_m_v_80_999"] = df_["boosters_m_v_80_89"] + df_["boosters_m_v_90_999"]

    return df_


def get_herhaalprik():
    """_summary_

    Returns:
        _type_: _description_
    """
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\herhaalprik_per_week_per_leeftijdscat.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herhaalprik_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )
    df_["herhaalprik_m_v_0_64"] = (
        df_["herhaalprik_m_v_0_49"] + df_["herhaalprik_m_v_50_64"]
    )
    df_["herhaalprik_m_v_80_999"] = (
        df_["herhaalprik_m_v_80_89"] + df_["herhaalprik_m_v_90_999"]
    )

    df_["weeknr"] = (
        df_["jaar"].astype(str) + "_" + df_["weeknr"].astype(str).str.zfill(2)
    )
    df_ = df_.drop("jaar", axis=1)

    return df_



def get_herfstprik():
    """_summary_

    Returns:
        _type_: _description_
    """
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\herfstprik_per_week_per_leeftijdscat.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herfstprik_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )
    df_["herfstprik_m_v_0_64"] = (
        df_["herfstprik_m_v_0_49"] + df_["herfstprik_m_v_50_64"]
    )
    df_["herfstprik_m_v_80_999"] = (
        df_["herfstprik_m_v_80_89"] + df_["herfstprik_m_v_90_999"]
    )

    df_["weeknr"] = (
        df_["jaar"].astype(str) + "_" + df_["weeknr"].astype(str).str.zfill(2)
    )
    df_ = df_.drop("jaar", axis=1)

    return df_


def get_all_data(seriename):
    """_summary_

    Returns:
        _type_: df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater,df_
    """
    df_boosters = get_boosters()
    df_herhaalprik = get_herhaalprik()
    df_herfstprik = get_herfstprik()
    # df_rioolwater_dag, df_rioolwater = None, None # get_rioolwater.scrape_rioolwater()
    df_kobak = get_kobak()
    df_rioolwater = get_rioolwater_simpel()
    df_ = get_sterftedata(seriename)

    return df_, df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_kobak


def get_rioolwater_simpel():
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rioolwaarde2024.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwaarde2024.csv"
    df_rioolwater = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )
    df_rioolwater["weeknr"] = (
        df_rioolwater["jaar"].astype(int).astype(str)
        + "_"
        + df_rioolwater["week"].astype(int).astype(str)
    )
    df_rioolwater["rioolwater_sma"] = (
        df_rioolwater["rioolwaarde"].rolling(window=5, center=False).mean().round(1)
    )

    return df_rioolwater


def get_df_offical():
    """Laad de waardes zoals door RIVM en CBS is bepaald. Gedownload dd 11 juni 2024
    Returns:
        _df
    """
    file = "C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\overl_cbs_vs_rivm.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overl_cbs_vs_rivm.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )
    df_["weeknr_z"] = (
        df_["jaar_z"].astype(str) + "_" + df_["week_z"].astype(str).str.zfill(2)
    )
    df_["verw_rivm_official"] = (
        df_["low_rivm_official"] + df_["high_rivm_official"]
    ) / 2

    return df_


def get_data_for_series(df_, seriename):

    df = df_[["jaar", "week", "weeknr", seriename]].copy(deep=True)

    df = df[(df["jaar"] > 2014)]
    # df = df[df["jaar"] > 2014 | (df["weeknr"] != 0) | (df["weeknr"] != 53)]
    df = df.sort_values(by=["jaar", "weeknr"]).reset_index()

    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]
    # som_2015_2019 = 0
    noemer = 149832 # average deaths per year 2015-2019
    for y in range(2015, 2020):
        df_year = df[(df["jaar"] == y)]
        # som = df_year["m_v_0_999"].sum()
        # som_2015_2019 +=som

        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        #  https://www.cbs.nl/nl-nl/nieuws/2024/06/sterfte-in-2023-afgenomen/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2023%20is%20deze%20156%20666.
        # https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85753NED/table?dl=A787C
        # Define the factors for each year
        

        # 2.5.1 Verwachte sterfte en oversterfte
        # De oversterfte is het verschil tussen het waargenomen aantal overledenen en een verwacht 
        # aantal overledenen in dezelfde periode. Het verwachte aantal overledenen wanneer er geen 
        # COVID-19-epidemie zou zijn geweest, wordt geschat op basis van de waargenomen sterfte in 
        # 2015–2019 in twee stappen. Eerst wordt voor elk jaar de sterfte per week bepaald.
        # Vervolgens wordt per week de gemiddelde sterfte in die week en de zes omliggende weken bepaald. 
        # Deze gemiddelde sterfte per week levert een benadering van de verwachte wekelijkse sterfte. 
        # Er is dan nog geen rekening gehouden met de ontwikkeling van de bevolkingssamenstelling. 

        # Daarom is de sterfte per week nog herschaald naar de verwachte totale sterfte voor het jaar. 
        # Het verwachte aantal overledenen in het hele jaar wordt bepaald op basis van de prognoses 
        # die het CBS jaarlijks maakt. Deze prognoses geven de meest waarschijnlijke toekomstige 
        # ontwikkelingen van de bevolking en de sterfte. De prognoses houden rekening met het feit 
        # dat de bevolking continu verandert door immigratie en vergrijzing. Het CBS gebruikt voor 
        # de prognose van de leeftijds- en geslachtsspecifieke sterftekansen een extrapolatiemodel 
        # (L. Stoeldraijer, van Duin et al., 2013
        # https://pure.rug.nl/ws/portalfiles/portal/13869387/stoeldraijer_et_al_2013_DR.pdf
        # ): er wordt van uitgegaan dat de toekomstige trends 
        # een voortzetting zijn van de trends uit het verleden. In het model wordt niet alleen 
        # uitgegaan van de trends in Nederland, maar ook van de meer stabiele trends in andere 
        # West-Europese landen. Tijdelijke versnellingen en vertragingen die voorkomen in de 
        # Nederlandse trends hebben zo een minder groot effect op de toekomstverwachtingen. 
        # Het model houdt ook rekening met het effect van rookgedrag op de sterfte, wat voor 
        # Nederland met name belangrijk is om de verschillen tussen mannen en vrouwen in sterftetrends 
        # goed te beschrijven.
        # https://www.cbs.nl/nl-nl/longread/rapportages/2023/oversterfte-en-doodsoorzaken-in-2020-tot-en-met-2022?onepage=true
        # Op basis van de geprognosticeerde leeftijds- en geslachtsspecifieke sterftekansen en de verwachte bevolkingsopbouw in dat jaar, wordt het verwachte aantal overledenen naar leeftijd en geslacht berekend voor een bepaald jaar. Voor 2020 is de verwachte sterfte 153 402, voor 2021 is deze 154 887 en voor 2022 is dit 155 493. 
        # Op basis van de geprognosticeerde leeftijds- en geslachtsspecifieke sterftekansen en de verwachte
        #  bevolkingsopbouw in dat jaar, wordt het verwachte aantal overledenen naar leeftijd en geslacht 
        # berekend voor een bepaald jaar. Voor 2020 is de verwachte sterfte 153 402, 
        # voor 2021 is deze 154 887 en voor 2022 is dit 155 493. 
        # Het aantal voor 2020 is ontleend aan de Kernprognose 2019-2060 
        # (L. Stoeldraijer, van Duin, C., Huisman, C., 2019), het aantal voor 2021 aan de
        #  Bevolkingsprognose 2020-2070 exclusief de aanname van extra sterfgevallen door de 
        # COVID-19-epidemie; (L. Stoeldraijer, de Regt et al., 2020) en het aantal voor 2022 
        # aan de Kernprognose 2021-2070 (exclusief de aanname van extra sterfgevallen door de coronapandemie) (L. Stoeldraijer, van Duin et al., 2021). 
        # https://www.cbs.nl/nl-nl/longread/rapportages/2023/oversterfte-en-doodsoorzaken-in-2020-tot-en-met-2022/2-data-en-methode
        
        # geen waarde voor 2024, zie https://twitter.com/Cbscommunicatie/status/1800505651833270551
        # huidige waarde 2024 is geexptrapoleerd 2022-2023
        factors = {
            2014: 1,
            2015: 1,
            2016: 1,
            2017: 1,
            2018: 1,
            2019: 1,
            2020: 153402 / noemer,
            2021: 154887 / noemer,
            2022: 155494 / noemer,
            2023: 156666 / noemer,  # or 169333 / som if you decide to use the updated factor
            2024: 157846 / noemer,
        }

#           # 2015	16,9	0,5
            # 2016	17	0,6
            # 2017	17,1	0,6
            # 2018	17,2	0,6
            # 2019	17,3	0,6
            # 2020	17,4	0,7
            # 2021	17,5	0,4
            # 2022	17,6	0,7
            # 2023	17,8	1,3
            # 2024	17,9	0,7
    # avg_overledenen_2015_2019 = (som_2015_2019/5)
    # st.write(avg_overledenen_2015_2019)
    # Loop through the years 2014 to 2024 and apply the factors
    for year in range(2014, 2025):
        new_column_name = f"{seriename}_factor_{year}"
        factor = factors[year]
        # factor=1
        df[new_column_name] = df[seriename] * factor

    return df


def rolling(df, what):

    df[f"{what}_sma"] = df[what].rolling(window=7, center=True).mean()

    # df[what] = df[what].rolling(window=6, center=False).mean()
    # df[f'{what}_sma'] = savgol_filter(df[what], 7,2)
    return df


def plot_wrapper(
    df_boosters,
    df_herhaalprik,
    df_herfstprik,
    df_rioolwater,
    df_,
    df_corona,
    df_quantile,
    df_kobak,
    series_name,
    how,
    yaxis_to_zero,
    rightax,
    mergetype,
    sec_y,
):
    """wrapper for the plots

    Args:
        df_ : df_sterfte
        series_names (_type_): _description_
        how (_type_): _description_
        yaxis_to_zero (_type_): _description_
        rightax (_type_): _description_
        mergetype (_type_): _description_
    """

    def plot_graph_oversterfte(
        how,
        df,
        df_corona,
        df_boosters,
        df_herhaalprik,
        df_herfstprik,
        df_rioolwater,
        df_kobak,
        series_name,
        rightax,
        mergetype,
        sec_y,
    ):

        """_summary_

        Args:
            how (_type_): _description_
            df (_type_): _description_
            df_corona (_type_): _description_
            df_boosters (_type_): _description_
            df_herhaalprik (_type_): _description_
            series_name (_type_): _description_
            rightax (_type_): _description_
            mergetype (_type_): _description_
        """
        booster_cat = ["m_v_0_999", "m_v_0_64", "m_v_65_79", "m_v_80_999"]

        df_oversterfte = pd.merge(
            df, df_corona, left_on="week_", right_on="weeknr", how="outer"
        )

        if rightax == "boosters":
            df_oversterfte = pd.merge(
                df_oversterfte, df_boosters, on="weeknr", how=mergetype
            )
        if rightax == "herhaalprik":
            df_oversterfte = pd.merge(
                df_oversterfte, df_herhaalprik, on="weeknr", how=mergetype
            )
        if rightax == "herfstprik":
            df_oversterfte = pd.merge(
                df_oversterfte, df_herfstprik, on="weeknr", how=mergetype
            )
        if rightax == "rioolwater":
            df_oversterfte = pd.merge(
                df_oversterfte, df_rioolwater, on="weeknr", how=mergetype
            )
        if rightax == "kobak":
            df_oversterfte = pd.merge(
                df_oversterfte, df_kobak, on="weeknr", how=mergetype
            )

        df_oversterfte["over_onder_sterfte"] = 0
        df_oversterfte["meer_minder_sterfte"] = 0

        df_oversterfte["year_minus_high95"] = (
            df_oversterfte[series_name] - df_oversterfte["high95"]
        )
        df_oversterfte["year_minus_avg"] = (
            df_oversterfte[series_name] - df_oversterfte["avg"]
        )
        df_oversterfte["p_score"] = (
            df_oversterfte[series_name] - df_oversterfte["avg"]
        ) / df_oversterfte["avg"]
        df_oversterfte = rolling(df_oversterfte, "p_score")

        for i in range(len(df_oversterfte)):
            if df_oversterfte.loc[i, series_name] > df_oversterfte.loc[i, "high95"]:
                df_oversterfte.loc[i, "over_onder_sterfte"] = (
                    df_oversterfte.loc[i, series_name] - df_oversterfte.loc[i, "avg"]
                )  # ["high95"]
                df_oversterfte.loc[i, "meer_minder_sterfte"] = (
                    df_oversterfte.loc[i, series_name] - df_oversterfte.loc[i, "high95"]
                )
            elif df_oversterfte.loc[i, series_name] < df_oversterfte.loc[i, "low05"]:
                df_oversterfte.loc[i, "over_onder_sterfte"] = (
                    df_oversterfte.loc[i, series_name] - df_oversterfte.loc[i, "avg"]
                )  # ["low05"]
                df_oversterfte.loc[i, "meer_minder_sterfte"] = (
                    df_oversterfte.loc[i, series_name] - df_oversterfte.loc[i, "low05"]
                )
        
        
        
           

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=df_oversterfte["week_"],
                y=df_oversterfte[how],
                # line=dict(width=2), opacity = 1, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                line=dict(width=2, color="rgba(205, 61,62, 1)"),
                mode="lines",
                name=how,
            )
        )

        if how == "p_score":
            # the p-score is already plotted
            pass
        elif how == "year_minus_avg":
            show_avg = False
            if show_avg:
                grens = "avg"
                fig.add_trace(
                    go.Scatter(
                        name=grens,
                        x=df_oversterfte["weeknr"],
                        y=df_oversterfte[grens],
                        mode="lines",
                        line=dict(width=1, color="rgba(205, 61,62, 1)"),
                    )
                )
        else:
            grens = "95%_interval"

            fig.add_trace(
                go.Scatter(
                    name="low",
                    x=df_oversterfte["week_"],
                    y=df_oversterfte["low05"],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
                    fillcolor="rgba(68, 68, 68, 0.2)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    name="high",
                    x=df_oversterfte["week_"],
                    y=df_oversterfte["high95"],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
                    fill="tonexty",
                )
            )

            # data = [high, low, fig_, sterfte ]
            fig.add_trace(
                go.Scatter(
                    name="Verwachte Sterfte",
                    x=df_oversterfte["weeknr"],
                    y=df_oversterfte["avg"],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(204, 63, 61, .8)"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    name="Sterfte",
                    x=df_oversterfte["weeknr"],
                    y=df_oversterfte[series_name],
                    mode="lines",
                    line=dict(width=1, color="rgba(204, 63, 61, 1)"),
                )
            )
        # rightax = "boosters" # "herhaalprik"
        if series_name in booster_cat or rightax == "rioolwater":
            if rightax == "boosters":

                b = "boosters_" + series_name
                fig.add_trace(
                    go.Scatter(
                        name="boosters",
                        x=df_oversterfte["week_"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )
                corr = df_oversterfte[b].corr(df_oversterfte[how])
                st.write(f"Correlation = {round(corr,3)}")
            elif rightax == "herhaalprik":

                b = "herhaalprik_" + series_name
                fig.add_trace(
                    go.Scatter(
                        name="herhaalprik",
                        x=df_oversterfte["week_"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )

                corr = df_oversterfte[b].corr(df_oversterfte[how])

                st.write(f"Correlation = {round(corr,3)}")
            elif rightax == "herfstprik":

                b = "herfstprik_" + series_name
                fig.add_trace(
                    go.Scatter(
                        name="herfstprik",
                        x=df_oversterfte["week_"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )

                corr = df_oversterfte[b].corr(df_oversterfte[how])

                st.write(f"Correlation = {round(corr,3)}")
            elif rightax == "rioolwater":
                b = "rioolwater_sma"
                fig.add_trace(
                    go.Scatter(
                        name="rioolwater",
                        x=df_oversterfte["week_"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )

                corr = df_oversterfte[b].corr(df_oversterfte[how])

                st.write(f"Correlation = {round(corr,3)}")
            elif rightax == "kobak":

                b = "excess deaths"
                fig.add_trace(
                    go.Scatter(
                        name="excess deaths(kobak)",
                        x=df_oversterfte["week_"],
                        y=df_oversterfte[b],
                        mode="lines",
                        line=dict(width=2, color="rgba(94, 172, 219, 1)"),
                    ),
                    secondary_y=sec_y,
                )

                corr = df_oversterfte[b].corr(df_oversterfte[how])

                st.write(f"Correlation = {round(corr,3)}")

        # data.append(booster)

        title = how
        layout = go.Layout(
            xaxis=dict(title="Weeknumber"),
            yaxis=dict(title="Number of persons"),
            title=title,
        )

        fig.add_hline(y=0)

        fig.update_yaxes(rangemode="tozero")

        st.plotly_chart(fig, use_container_width=True)
        # /plot_graph_oversterfte

    def plot_lines(series_name, df_data):
        # fig = plt.figure()

        year_list = df_data["jaar"].unique().tolist()

        data = []

        for idx, year in enumerate(year_list):
            df = df_data[df_data["jaar"] == year].copy(
                deep=True
            )  # [['weeknr', series_name]].reset_index()

            # df = df.sort_values(by=['weeknr'])
            if (
                year == 2020
                or year == 2021
                or year == 2022
                or year == 2023
                or year == 2024
            ):
                width = 3
                opacity = 1
            else:
                width = 0.7
                opacity = 0.3

            fig_ = go.Scatter(
                x=df["week"],
                y=df[series_name],
                line=dict(width=width),
                opacity=opacity,  # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                mode="lines",
                name=year,
                legendgroup=str(year),
            )

            data.append(fig_)

        title = f"Stefte - {series_name}"
        layout = go.Layout(
            xaxis=dict(title="Weeknumber"),
            yaxis=dict(title="Number of persons"),
            title=title,
        )

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        # end of plot_lines

    def plot_quantiles(yaxis_to_zero, series_name, df_corona, df_quantile):
        columnlist = ['avg_', 'low05', 'high95']
        for what_to_sma in columnlist:
            df_quantile[what_to_sma] = df_quantile[what_to_sma].rolling(window=7, center=True).mean().round(1)


        # df_quantile = df_quantile.sort_values(by=['jaar','week_'])
        
        fig = go.Figure()
        low05 = go.Scatter(
            name="low",
            x=df_quantile["weeknr"],
            y=df_quantile["low05"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
            fillcolor="rgba(68, 68, 68, 0.1)",
            fill="tonexty",
        )

        avg = go.Scatter(
            name="gemiddeld",
            x=df_quantile["weeknr"],
            y=df_quantile["avg_"],
            mode="lines",
            line=dict(width=0.75, color="rgba(68, 68, 68, 0.8)"),
        )

        sterfte = go.Scatter(
            name="Sterfte",
            x=df_quantile["weeknr"],
            y=df_corona[series_name],
            mode="lines",
            line=dict(width=2, color="rgba(255, 0, 0, 0.8)"),
        )

        high95 = go.Scatter(
            name="high",
            x=df_quantile["weeknr"],
            y=df_quantile["high95"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255, 188, 0, 0.5)"),
            fillcolor="rgba(68, 68, 68, 0.2)",
        )

        # data = [ q95, high95, q05,low05,avg, sterfte] #, value_in_year_2021 ]
        data = [high95, low05, avg, sterfte]  # , value_in_year_2021 ]
        title = f"Overleden {series_name}"
        layout = go.Layout(
            xaxis=dict(title="Weeknumber"),
            yaxis=dict(title="Number of persons"),
            title=title,
        )

        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(xaxis=dict(tickformat="%d-%m"))

        #             — eerste oversterftegolf: week 13 tot en met 18 van 2020 (eind maart–eind april 2020);
        # — tweede oversterftegolf: week 39 van 2020 tot en met week 3 van 2021 (eind
        # september 2020–januari 2021);
        # — derde oversterftegolf: week 33 tot en met week 52 van 2021 (half augustus 2021–eind
        # december 2021).
        # De hittegolf in 2020 betreft week 33 en week 34 (half augustus 2020).

        fig.add_vrect(
            x0="2020_13",
            x1="2020_18",
            annotation_text="Eerste golf",
            annotation_position="top left",
            fillcolor="pink",
            opacity=0.25,
            line_width=0,
        )
        fig.add_vrect(
            x0="2020_39",
            x1="2021_03",
            annotation_text="Tweede golf",
            annotation_position="top left",
            fillcolor="pink",
            opacity=0.25,
            line_width=0,
        )
        fig.add_vrect(
            x0="2021_33",
            x1="2021_52",
            annotation_text="Derde golf",
            annotation_position="top left",
            fillcolor="pink",
            opacity=0.25,
            line_width=0,
        )

        # hittegolven
        fig.add_vrect(
            x0="2020_33",
            x1="2020_34",
            annotation_text=" ",
            annotation_position="top left",
            fillcolor="yellow",
            opacity=0.35,
            line_width=0,
        )

        fig.add_vrect(
            x0="2022_32",
            x1="2022_33",
            annotation_text=" ",
            annotation_position="top left",
            fillcolor="yellow",
            opacity=0.35,
            line_width=0,
        )

        fig.add_vrect(
            x0="2023_23",
            x1="2023_24",
            annotation_text=" ",
            annotation_position="top left",
            fillcolor="yellow",
            opacity=0.35,
            line_width=0,
        )
        fig.add_vrect(
            x0="2023_36",
            x1="2023_37",
            annotation_text="Geel = Hitte golf",
            annotation_position="top left",
            fillcolor="yellow",
            opacity=0.35,
            line_width=0,
        )

        if yaxis_to_zero:
            fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)
        return df_quantile
        # end of plot quantiles

    st.subheader(series_name)
    if how == "quantiles":

        plot_quantiles(yaxis_to_zero, series_name, df_corona, df_quantile)

    elif (
        (how == "year_minus_avg")
        or (how == "over_onder_sterfte")
        or (how == "meer_minder_sterfte")
        or (how == "p_score")
    ):
        plot_graph_oversterfte(
            how,
            df_quantile,
            df_corona,
            df_boosters,
            df_herhaalprik,
            df_herfstprik,
            df_rioolwater,
            df_kobak,
            series_name,
            rightax,
            mergetype,
            sec_y,
        )
    else:
        plot_lines(series_name, df_corona)


def get_baseline_kobak():
    """Load the csv with the baseline as calculated by Ariel Karlinsky and Dmitry Kobak
        https://elifesciences.org/articles/69336#s4
        https://github.com/dkobak/excess-mortality/

    Returns:
        _type_: _description_
    """
    url = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/kobak_baselines.csv"
    # url ="C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\kobak_baselines.csv"     # Maak een interactieve plot met Plotly
    df_ = pd.read_csv(
        url,
        delimiter=",",
        low_memory=False,
    )

    df_["weeknr"] = df_["jaar"].astype(str) + "_" + df_["week"].astype(str).str.zfill(2)
    df_ = df_[["weeknr", "baseline_kobak"]]
    return df_


def show_difference(df, date_field, show_official, year):
    """Function to show the difference between the two methods quickly"""

    df_baseline_kobak = get_baseline_kobak()
    df = pd.merge(df, df_baseline_kobak, on="weeknr", how="outer")
    
    if year!= "All":
        df= df[df["jaar_x_x"] == year]
  
    # rolling(df, 'baseline_kobak')

    # Maak een interactieve plot met Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["baseline_kobak"],
            mode="lines",
            name="Baseline Kobak",
        )
    )

    # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["high_rivm"],
            mode="lines",
            fill=None,
            line_color="yellow",
            name="high rivm",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["low_rivm"],
            mode="lines",
            fill="tonexty",  # Vul het gebied tussen de lijnen
            line_color="yellow",
            name="low rivm",
        )
    )

    # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["low_cbs_sma"],
            mode="lines",
            fill=None,
            line_color="lightgrey",
            name="low cbs",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["high_cbs_sma"],
            mode="lines",
            fill="tonexty",  # Vul het gebied tussen de lijnen
            line_color="lightgrey",
            name="high cbs",
        )
    )
    if show_official:
        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["high_rivm_official"],
                mode="lines",
                fill=None,
                line_color="orange",
                name="high rivm official",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["low_rivm_official"],
                mode="lines",
                fill="tonexty",  # Vul het gebied tussen de lijnen
                line_color="orange",
                name="low rivm  official",
            )
        )

        # Voeg de betrouwbaarheidsinterval toe
        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["low_cbs_official"],
                mode="lines",
                fill=None,
                line_color="lightblue",
                name="low cbs  official",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["high_cbs_official"],
                mode="lines",
                fill="tonexty",  # Vul het gebied tussen de lijnen
                line_color="lightblue",
                name="high cbs  official",
            )
        )
        # Voeg de voorspelde lijn toe
        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["verw_rivm_official"],
                mode="lines",
                name="Baseline model rivm  official",
            )
        )
        # Voeg de voorspelde lijn toe
        fig.add_trace(
            go.Scatter(
                x=df[date_field],
                y=df["verw_cbs_official"],
                mode="lines",
                name="Baseline model cbs  official",
            )
        )

        # Voeg de voorspelde lijn toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["verw_rivm"],
            mode="lines",
            name="Baseline model rivm",
        )
    )

    # Voeg de voorspelde lijn toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["verw_cbs_sma"],
            mode="lines",
            name="Baseline model cbs",
        )
    )

    # Voeg de voorspelde lijn RIVM toe
    fig.add_trace(
        go.Scatter(
            x=df[date_field],
            y=df["aantal_overlijdens"],
            mode="lines",
            name="Werkelijk overleden",
        )
    )
    # Titel en labels toevoegen
    fig.update_layout(
        title="Vergelijking CBS vs RIVM",
        xaxis_title="Tijd",
        yaxis_title="Aantal Overledenen",
    )

    st.plotly_chart(fig)


def duplicate_row(df, from_, to):
    """Duplicates a row

    Args:
        df (df): df
        from_ (str): oorspronkelijke rij eg. '2022_51'
        to (str): bestemmingsrij eg. '2022_52'
    """
    # Find the row where weeknr is '2022_51' and duplicate it
    row_to_duplicate = df[df["weeknr"] == from_].copy()

    # Update the weeknr value to '2022_52' in the duplicated row
    row_to_duplicate["weeknr"] = to
    row_to_duplicate["week"] = int(to.split("_")[1])
    # row_to_duplicate['m_v_0_999'] = 0
    # df_merged['week'] = np.where(df_merged['weeknr'] == '2021_52', 52, df_merged['week'])
    # df_merged['week'] = np.where(df_merged['weeknr'] == '2022_52', 52, df_merged['week'])
    # df_merged['week'] = np.where(df_merged['weeknr'] == '2019_01', 1, df_merged['week'])
    # df_merged['week'] = np.where(df_merged['weeknr'] == '2015_01', 1, df_merged['week'])

    # Append the duplicated row to the DataFrame
    df = pd.concat([df, row_to_duplicate], ignore_index=True)

    df = df.sort_values(by=["weeknr"]).reset_index(drop=True)

    return df


def footer():
    st.write(
        "De correctiefactor voor 2020, 2021 en 2022 is berekend over de gehele populatie."
    )
    st.write(
        "Het 95%-interval is berekend aan de hand van het gemiddelde en standaarddeviatie (z=2)  over de waardes per week van 2015 t/m 2019"
    )
    # st.write("Week 53 van 2020 heeft een verwachte waarde en 95% interval van week 52")
    st.write(
        "Enkele andere gedeeltelijke weken zijn samengevoegd conform het CBS bestand"
    )
    st.write("Code: https://github.com/rcsmit/COVIDcases/blob/main/oversterfte.py")
    st.write("P score = (verschil - gemiddelde) / gemiddelde, gesmooth over 6 weken")
    st.info("Karlinsky & Kobak: https://elifesciences.org/articles/69336#s4")
    st.info("Steigstra: https://twitter.com/SteigstraHerman/status/1801641074336706839")



def get_kobak():
    """Load the csv with the baselines as calculated by Ariel Karlinsky and Dmitry Kobak
    https://elifesciences.org/articles/69336#s4
    https://github.com/dkobak/excess-mortality/


    One line is deleted: Netherlands, 2020, 53, 3087.2
    since all other years have 52 weeks

    Returns:
        _type_: _description_
    """

    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/kobak_baselines.csv"
    # file = r"C:\\Users\\rcxsm\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\kobak_baselines.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )

    return df_


@st.cache_data(ttl=60 * 60 * 24)
def get_sterftedata(seriename="m_v_0_999"):
    """Get and manipulate data of the deaths

    Args:
        seriename (str, optional): _description_. Defaults to "m_v_0_999".
    """

    def manipulate_data_df(data):
        """Filters out week 0 and 53 and makes a category column (eg. "M_V_0_999")"""

        # data = data[~data['week'].isin([0, 53])] #filter out week 2020-53
        data["weeknr"] = (
            data["jaar"].astype(str) + "_" + data["week"].astype(str).str.zfill(2)
        )

        data["Geslacht"] = data["Geslacht"].replace(
            ["Totaal mannen en vrouwen"], "m_v_"
        )
        data["Geslacht"] = data["Geslacht"].replace(["Mannen"], "m_")
        data["Geslacht"] = data["Geslacht"].replace(["Vrouwen"], "v_")
        data["LeeftijdOp31December"] = data["LeeftijdOp31December"].replace(
            ["Totaal leeftijd"], "0_999"
        )
        data["LeeftijdOp31December"] = data["LeeftijdOp31December"].replace(
            ["0 tot 65 jaar"], "0_64"
        )
        data["LeeftijdOp31December"] = data["LeeftijdOp31December"].replace(
            ["65 tot 80 jaar"], "65_79"
        )
        data["LeeftijdOp31December"] = data["LeeftijdOp31December"].replace(
            ["80 jaar of ouder"], "80_999"
        )
        data["categorie"] = data["Geslacht"] + data["LeeftijdOp31December"]

        return data

    def extract_period_info(period):
        """Function to extract the year, week, and days

        Args:
            period (string): e.g. 2024 week 12 (3 dagen)

        Returns:
            year, week, days: number of yy,ww and dd
        """
        #
        import re

        pattern = r"(\d{4}) week (\d{1,2}) \((\d+) dag(?:en)?\)"
        match = re.match(pattern, period)
        if match:
            year, week, days = match.groups()
            return int(year), int(week), int(days)
        return None, None, None

    def adjust_overledenen(df):
        """# Adjust "Overledenen_1" based on the week number
        # if week = 0, overledenen_l : add to week 52 of the year before
        # if week = 53: overleden_l : add to week 1 to the year after

        TODO: integrate chagnes from calculate_baselines.py
        """

        for index, row in df.iterrows():
            if row["week"] == 0:
                previous_year = row["year"] - 1
                df.loc[
                    (df["year"] == previous_year) & (df["week"] == 52), "Overledenen_1"
                ] += row["Overledenen_1"]
            elif row["week"] == 53:
                next_year = row["year"] + 1
                df.loc[
                    (df["year"] == next_year) & (df["week"] == 1), "Overledenen_1"
                ] += row["Overledenen_1"]
        # Filter out the rows where week is 0 or 53 after adjusting
        df = df[~df["week"].isin([0, 53])]
        return df

    data_ruw = pd.DataFrame(cbsodata.get_data("70895ned"))

    # Filter rows where Geslacht is 'Totaal mannen en vrouwen' and LeeftijdOp31December is 'Totaal leeftijd'
    # data_ruw = data_ruw[(data_ruw['Geslacht'] == 'Totaal mannen en vrouwen') & (data_ruw['LeeftijdOp31December'] == 'Totaal leeftijd')]

    data_ruw[["jaar", "week"]] = data_ruw.Perioden.str.split(
        " week ",
        expand=True,
    )

    data_ruw["jaar"] = data_ruw["jaar"].astype(int)
    data_ruw = data_ruw[(data_ruw["jaar"] > 2013)]
    data_ruw = manipulate_data_df(data_ruw)
    data_ruw = data_ruw[data_ruw["categorie"] == seriename]
    data_compleet = data_ruw[~data_ruw["Perioden"].str.contains("dag")]
    data_incompleet = data_ruw[data_ruw["Perioden"].str.contains("dag")]

    # Apply the function to the "perioden" column and create new columns
    data_incompleet[["year", "week", "days"]] = data_incompleet["Perioden"].apply(
        lambda x: pd.Series(extract_period_info(x))
    )

    data_incompleet = adjust_overledenen(data_incompleet)
    data = pd.concat([data_compleet, data_incompleet])
    data = data[data["week"].notna()]
    data["week"] = data["week"].astype(int)

    data = data.sort_values(by=["jaar", "week"]).reset_index()

    # Combine the adjusted rows with the remaining rows

    df_ = data.pivot(
        index=["weeknr", "jaar", "week"], columns="categorie", values="Overledenen_1"
    ).reset_index()
    df_["week"] = df_["week"].astype(int)
    df_["jaar"] = df_["jaar"].astype(int)

    # dit moet nog ergens anders
    df_[["weeknr", "delete"]] = df_.weeknr.str.split(
        r" \(",
        expand=True,
    )
    df_ = df_.replace("2015_1", "2015_01")
    df_ = df_.replace("2020_1", "2020_01")

    df_ = df_[(df_["jaar"] > 2014)]

    df = df_[["jaar", "weeknr", "week", seriename]].copy(deep=True)

    df = df.sort_values(by=["jaar", "weeknr"]).reset_index()

    df = get_data_for_series(df, seriename)

    return df


def make_df_quantile(series_name, df_data):
    """_Makes df quantile
    make_df_quantile -> make_df_quantile -> make_row_quantile

    Args:
        series_name (_type_): _description_
        df_data (_type_): _description_

    Returns:
        df : merged df
        df_corona: df with baseline
        df_quantiles : df with quantiles
    """

    def make_df_quantile_year(series_name, df_data, year):

        """Calculate the quantiles for a certain year
            make_df_quantile -> make_df_quantile -> make_row_quantile


        Returns:
            _type_: _description_
        """

        def make_row_df_quantile(series_name, year, df_to_use, w_):
            """Calculate the percentiles of a certain week
                make_df_quantile -> make_df_quantile -> make_row_quantile

            Args:
                series_name (_type_): _description_
                year (_type_): _description_
                df_to_use (_type_): _description_
                w_ (_type_): _description_

            Returns:
                _type_: _description_
            """
            if w_ == 53:
                w = 52
            else:
                w = w_
            column_to_use = series_name + "_factor_" + str(year)
            df_to_use_ = df_to_use[(df_to_use["week"] == w)].copy(deep=True)

            data = df_to_use_[column_to_use]  # .tolist()
            avg = round(data.mean(), 0)
            
          
            df_to_use_ = df_to_use[(df_to_use["week"] == w)].copy(deep=True)

            data = df_to_use_[column_to_use]  # .tolist()
            avg = round(data.mean(), 0)
                
            try:
                q05 = np.percentile(data, 5)
                q25 = np.percentile(data, 25)
                q50 = np.percentile(data, 50)
                q75 = np.percentile(data, 75)
                q95 = np.percentile(data, 95)
            except:
                q05, q25, q50, q75, q95 = 0, 0, 0, 0, 0

           

            sd = round(data.std(), 0)
            low05 = round(avg - (2 * sd), 0)
            high95 = round(avg + (2 * sd), 0)

            df_quantile_ = pd.DataFrame(
                [
                    {
                        "week_": w_,
                        "jaar": year,
                        "q05": q05,
                        "q25": q25,
                        "q50": q50,
                        "avg_": avg,
                        "q75": q75,
                        "q95": q95,
                        "low05": low05,
                        "high95": high95,
                    }
                ]
            )

            return df_quantile_

        df_to_use = df_data[(df_data["jaar"] >= 2015) & (df_data["jaar"] < 2020)].copy(
            deep=True
        )

        df_quantile = None

        week_list = df_to_use["weeknr"].unique().tolist()
        for w in range(1, 53):
            df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, w)
            df_quantile = pd.concat([df_quantile, df_quantile_], axis=0)
        return df_quantile

    df_corona = df_data[df_data["jaar"].between(2015, 2025)]

    # List to store individual quantile DataFrames
    df_quantiles = []

    # Loop through the years 2014 to 2024
    for year in range(2015, 2025):
        df_quantile_year = make_df_quantile_year(series_name, df_data, year)
        df_quantiles.append(df_quantile_year)

    # Concatenate all quantile DataFrames into a single DataFrame
    df_quantile = pd.concat(df_quantiles, axis=0)

    df_quantile["weeknr"] = (
        df_quantile["jaar"].astype(str)
        + "_"
        + df_quantile["week_"].astype(str).str.zfill(2)
    )

    df = pd.merge(df_corona, df_quantile, on="weeknr")
    return df, df_corona, df_quantile


def predict(X, verbose=False, excess_begin=None):
    """Function to predict the baseline with linear regression - Karlinksy & Kobak
       Source: https://github.com/dkobak/excess-mortality/blob/main/all-countries.ipynb

    Args:
        X (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.
        excess_begin (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    def get_excess_begin(datapoints_per_year=53):
        if datapoints_per_year > 12:
            beg = 9  # week 10

        elif datapoints_per_year > 4 and datapoints_per_year <= 12:
            beg = 2  # March

        elif datapoints_per_year == 4:
            beg = 0

        return beg

    # Fit regression model on pre-2020 data
    ind = (X[:, 0] < 2020) & (X[:, 1] < 53)
    m = np.max(X[ind, 1])
    onehot = np.zeros((np.sum(ind), m))
    for i, k in enumerate(X[ind, 1]):
        onehot[i, k - 1] = 1
    predictors = np.concatenate((X[ind, :1], onehot), axis=1)
    reg = LinearRegression(fit_intercept=False).fit(predictors, X[ind, 2])

    if verbose:
        est = sm.OLS(X[ind, 2], predictors).fit()
        print(est.summary())

    # Compute 2020 baseline
    ind2 = X[:, 0] == 2020
    predictors2020 = np.concatenate((np.ones((m, 1)) * 2020, np.eye(m)), axis=1)
    baseline = reg.predict(predictors2020)

    # Week 53 usually does not have enough data, so we'll use
    # the same baseline value as for week 52
    if np.max(X[:, 1]) == 53:
        baseline = np.concatenate((baseline, [baseline[-1]]))

    # Compute 2021 baseline
    predictors2021 = np.concatenate((np.ones((m, 1)) * 2021, np.eye(m)), axis=1)
    baseline2021 = reg.predict(predictors2021)

    # Compute 2022 baseline
    predictors2022 = np.concatenate((np.ones((m, 1)) * 2022, np.eye(m)), axis=1)
    baseline2022 = reg.predict(predictors2022)

    # Compute 2023 baseline
    predictors2023 = np.concatenate((np.ones((m, 1)) * 2023, np.eye(m)), axis=1)
    baseline2023 = reg.predict(predictors2023)

    # Excess mortality
    ind2 = X[:, 0] == 2020
    diff2020 = X[ind2, 2] - baseline[X[ind2, 1] - 1]
    ind3 = X[:, 0] == 2021
    diff2021 = X[ind3, 2] - baseline2021[X[ind3, 1] - 1]
    ind4 = X[:, 0] == 2022
    diff2022 = X[ind4, 2] - baseline2022[X[ind4, 1] - 1]
    ind5 = X[:, 0] == 2023
    diff2023 = X[ind5, 2] - baseline2023[X[ind5, 1] - 1]
    if excess_begin is None:
        excess_begin = get_excess_begin(baseline.size)
    total_excess = (
        np.sum(diff2020[excess_begin:])
        + np.sum(diff2021)
        + np.sum(diff2022)
        + np.sum(diff2023)
    )
    # Manual fit for uncertainty computation
    if np.unique(X[ind, 0]).size > 1:
        y = X[ind, 2][:, np.newaxis]
        beta = np.linalg.pinv(predictors.T @ predictors) @ predictors.T @ y
        yhat = predictors @ beta
        sigma2 = np.sum((y - yhat) ** 2) / (y.size - predictors.shape[1])

        S = np.linalg.pinv(predictors.T @ predictors)
        w = np.zeros((m, 1))
        w[X[(X[:, 0] == 2020) & (X[:, 1] < 53), 1] - 1] = 1
        if np.sum((X[:, 0] == 2020) & (X[:, 1] == 53)) > 0:
            w[52 - 1] += 1
        w[:excess_begin] = 0

        p = 0
        for i, ww in enumerate(w):
            p += predictors2020[i] * ww

        w2021 = np.zeros((m, 1))
        w2021[X[ind3, 1] - 1] = 1
        for i, ww in enumerate(w2021):
            p += predictors2021[i] * ww

        w2022 = np.zeros((m, 1))
        w2022[X[ind4, 1] - 1] = 1
        for i, ww in enumerate(w2022):
            p += predictors2022[i] * ww

        w2023 = np.zeros((m, 1))
        w2023[X[ind5, 1] - 1] = 1
        for i, ww in enumerate(w2023):
            p += predictors2023[i] * ww

        p = p[:, np.newaxis]

        predictive_var = (
            sigma2 * (np.sum(w) + np.sum(w2021) + np.sum(w2022) + np.sum(w2023))
            + sigma2 * p.T @ S @ p
        )
        total_excess_std = np.sqrt(predictive_var)[0][0]
    else:
        total_excess_std = np.nan

    return (
        (baseline, baseline2021, baseline2022, baseline2023),
        total_excess,
        excess_begin,
        total_excess_std,
    )


def show_plot(df, df_covid, df_kobak_github):
    """_summary_

    Args:
        df (df): df with the calculated values of the Kobak baseline
        df_covid (df): df with the calcualted values with the CBS method
        df_kobak_github (df): df with the values of the Kobak baseline from their Github repo
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["weeknr"], y=df["baseline_kobak"], mode="lines", name=f"kobak_baseline"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["weeknr"],
            y=df_kobak_github["baseline_kobak"],
            mode="lines",
            name=f"kobak_baseline GITHUB",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["weeknr"],
            y=df_covid["avg_"],
            mode="lines",
            name=f"CBS_method_baseline",
        )
    )

    fig.update_layout(title=f"Koak vs CBS", xaxis_title="Tijd", yaxis_title="Aantal")

    st.plotly_chart(fig)


def do_kobak_vs_cbs(df_deaths):

    """Main function
        Calculate the baseline with the function of Karlinksy & Kobak
        (and compare with CBS and their own results)
    Results :
        df_kobak_calculated (df): df with the calculated values of the Kobak baseline
        df_covid (df): df with the calcualted values with the CBS method
        df_kobak_github (df): df with the values of the Kobak baseline from their Github repo
    """
    st.subheader("Kobak vs CBS")
    # df_deaths = get_sterftedata()
    df, _, _ = make_df_quantile("m_v_0_999", df_deaths)

    df_covid = df[(df["jaar_x"] >= 2020) & (df["jaar_x"] <= 2023)]

    X = df[["jaar_x", "week", "m_v_0_999"]].values
    X = X[~np.isnan(X[:, 2]), :]
    X = X.astype(int)

    baselines, total_excess, excess_begin, total_excess_std = predict(X)
    list1 = baselines[0].tolist()
    list2 = baselines[1].tolist()
    list3 = baselines[2].tolist()
    list4 = baselines[3].tolist()

    # Combine the lists
    combined_list = list1 + list2 + list3 + list4

    # Generate a date range from the start of 2020 to the end of 2023
    date_range = pd.date_range(start="2020-01-01", end="2023-12-31", freq="W-MON")

    # Extract year and week number
    df_kobak_calculated = pd.DataFrame(
        {"year": date_range.year, "week": date_range.isocalendar().week}
    )
    df_kobak_calculated["baseline_kobak"] = combined_list
    df_kobak_calculated["weeknr"] = (
        df_kobak_calculated["year"].astype(str)
        + "_"
        + df_kobak_calculated["week"].astype(str).str.zfill(2)
    )

    df_kobak_github = get_kobak()

    show_plot(df_kobak_calculated, df_covid, df_kobak_github)


def main():

    serienames_ = [
        "m_v_0_999",
        "m_v_0_64",
        "m_v_65_79",
        "m_v_80_999",
        "m_0_999",
        "m_0_64",
        "m_65_79",
        "m_80_999",
        "v_0_999",
        "v_0_64",
        "v_65_79",
        "v_80_999",
    ]

    series_name = st.sidebar.selectbox("Leeftijden", serienames_, 0)
    smooth = st.sidebar.selectbox("Smooth div lijnen in de vergelijking", [True,False],0)
    st.header("Oversterfte - minder leeftijdscategorieen - v240829a")
    st.subheader("CBS & RIVM Methode")
    st.write(
        """Berekening van de oversterfte met de CBS en RIVM methode. Tevens wordt een vergelijking gemaakt met de methode van Kobak en de methode Steigstra wordt gerepliceerd. Dit script heeft minder leeftijdscategorieen in vergelijking met Eurostats, maar de sterftedata wordt real-time opgehaald van het CBS. 
        Daarnaast wordt het 95% betrouwbaarheids interval berekend vanuit de jaren 2015-2019"""
    )
    st.info("https://rene-smit.com/de-grote-oversterftekloof-rivm-vs-cbs/")
    how, yaxis_to_zero, rightax, mergetype, sec_y = interface()

    (
        df_sterfte,
        df_boosters,
        df_herhaalprik,
        df_herfstprik,
        df_rioolwater,
        df_kobak,
    ) = get_all_data(series_name)

    print(f"---{series_name}----")
    df_data = get_data_for_series(df_sterfte, series_name).copy(deep=True)

    _, df_corona, df_quantile = make_df_quantile(series_name, df_data)

    plot_wrapper(
        df_boosters,
        df_herhaalprik,
        df_herfstprik,
        df_rioolwater,
        df_sterfte,
        df_corona,
        df_quantile,
        df_kobak,
        series_name,
        how,
        yaxis_to_zero,
        rightax,
        mergetype,
        sec_y,
    )
    if how == "quantiles":

        df_rivm = verwachte_sterfte_rivm(df_sterfte, series_name)
        df_merged = make_df_merged(df_data, df_rivm, series_name)
        # df_merged.to_csv(f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\df_merged_{series_name}.csv")
        if series_name == "m_v_0_999":
            plot_graph_rivm(df_rivm, series_name, False)
            comparison(df_merged, series_name, smooth)
            do_kobak_vs_cbs(df_sterfte)
        df_steigstra = calculate_steigstra(df_merged, series_name)
        plot_steigstra(df_steigstra, series_name)
    else:
        st.info(
            "De vergrlijking met vaccinateies, rioolwater etc is vooralsnog alleen mogelijk met CBS methode "
        )
    footer()


if __name__ == "__main__":
    import datetime

    print(
        f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------"
    )

    main()
