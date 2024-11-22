import pandas as pd
import cbsodata
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import statsmodels.api as sm

from oversterfte_plot_functions import plot_wrapper, plot_graph_rivm, show_difference_plot,  plot_steigstra_wrapper
from oversterfte_get_data import get_all_data, get_df_offical, get_baseline_kobak
from oversterfte_rivm_functions import verwachte_sterfte_rivm
from oversterfte_cbs_functions import get_sterftedata, get_data_for_series_wrapper, make_df_quantile,make_df_quantile_year,make_row_df_quantile

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
                        x=df_merged_jaar["periodenr"],
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

def make_df_merged(df_data, df_rivm, series_name, period):
    _, df_corona, df_quantile = make_df_quantile(series_name, df_data, period)
    df_official = get_df_offical()

    df_merged = df_corona.merge(df_quantile, left_on="periodenr", right_on="periodenr").merge(
        df_rivm, on="periodenr", how="outer"
    )
    df_merged = df_merged.merge(
        df_official, left_on="periodenr", right_on="periodenr_z", how="outer"
    )

    #df_merged = df_merged.drop(columns=["week"])

    df_merged["shifted_jaar"] = df_merged["jaar_x_x"]  # .shift(28)
    df_merged["shifted_week"] = df_merged["week_x_x"]  # .shift(28)

    columns = [
        [series_name, "aantal_overlijdens"],
        ["q50", "verw_cbs_q50"],
        ["low05", "low_cbs"],
        ["avg_", "avg_x"],
        ["high95", "high_cbs"],
        ["voorspeld", "verw_rivm"],
        ["lower_ci", "low_rivm"],
        ["upper_ci", "high_rivm"],
    ]
    # ["avg_", "verw_cbs"],

    for c in columns:
        df_merged = df_merged.rename(columns={c[0]: c[1]})
    return df_merged

def calculate_steigstra(df_merged, series_naam, cumm=False, m="cbs"):

    # Set 'week' to 52 if 'periodenr' is '2022_52'

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
            index=["week_x_x"],
            columns="shifted_jaar",
            values=f"oversterfte_{m}_simpel_cumm",
        ).reset_index()
    else:
        df = df_compleet.pivot(
            index=["week_x_x"], columns="shifted_jaar", values=f"oversterfte_{m}_simpel"
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
            
            show_difference_wrapper(df_merged_jaar, "periodenr", show_official, year)

            display_cumulative_oversterfte(df_merged_jaar, year)
            display_results(df_merged_jaar, year)


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

def rolling(df, what):

    df[f"{what}_sma"] = df[what].rolling(window=7, center=True).mean()

    # df[what] = df[what].rolling(window=6, center=False).mean()
    # df[f'{what}_sma'] = savgol_filter(df[what], 7,2)
    return df


def show_difference_wrapper(df, date_field, show_official, year):
    """Function to show the difference between the two methods quickly"""

    df_baseline_kobak = get_baseline_kobak()

    df = pd.merge(df, df_baseline_kobak, on="periodenr", how="outer")
    
    if year!= "All":
        df= df[df["jaar_x_x"] == year]
  
    # rolling(df, 'baseline_kobak')
   
    show_difference_plot(df, date_field, show_official, year)


def duplicate_row(df, from_, to):
    """Duplicates a row

    Args:
        df (df): df
        from_ (str): oorspronkelijke rij eg. '2022_51'
        to (str): bestemmingsrij eg. '2022_52'
    """
    # Find the row where periodenr is '2022_51' and duplicate it
    row_to_duplicate = df[df["periodenr"] == from_].copy()

    # Update the periodenr value to '2022_52' in the duplicated row
    row_to_duplicate["periodenr"] = to
    row_to_duplicate["week"] = int(to.split("_")[1])
   
    # Append the duplicated row to the DataFrame
    df = pd.concat([df, row_to_duplicate], ignore_index=True)

    df = df.sort_values(by=["periodenr"]).reset_index(drop=True)

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
            x=df["periodenr"], y=df["baseline_kobak"], mode="lines", name=f"kobak_baseline"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["periodenr"],
            y=df_kobak_github["baseline_kobak"],
            mode="lines",
            name=f"kobak_baseline GITHUB",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["periodenr"],
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
    df, _, _ = make_df_quantile("m_v_0_999", df_deaths, "week")

    df_covid = df[(df["jaar_x"] >= 2020) & (df["jaar_x"] <= 2023)]
   
    X = df[["jaar_x", "week_x", "m_v_0_999"]].values
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
    df_kobak_calculated["periodenr"] = (
        df_kobak_calculated["year"].astype(str)
        + "_"
        + df_kobak_calculated["week"].astype(str).str.zfill(2)
    )

    df_kobak_github = get_baseline_kobak()

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
    vanaf_jaar = st.sidebar.number_input ("Beginjaar voor CI-interv. (incl.)", 2000, 2022,2015)
    period = "week" # st.sidebar.selectbox("Period", ["week", "maand"], index = 0)
    st.header("Oversterfte - minder leeftijdscategorieen - v240829a")
    st.subheader("CBS & RIVM Methode")
    st.write(
        """Berekening van de oversterfte met de CBS en RIVM methode. Tevens wordt een vergelijking gemaakt met de methode van Kobak en de methode Steigstra wordt gerepliceerd. Dit script heeft minder leeftijdscategorieen in vergelijking met Eurostats, maar de sterftedata wordt real-time opgehaald van het CBS. 
        Daarnaast wordt het 95% betrouwbaarheids interval berekend vanuit de jaren 2015-2019"""
    )
    st.info("https://rene-smit.com/de-grote-oversterftekloof-rivm-vs-cbs/")
    how, yaxis_to_zero, rightax, mergetype, sec_y = interface()
    

    (
        
        df_boosters,
        df_herhaalprik,
        df_herfstprik,
        df_rioolwater,
        df_kobak, cbs_data_ruw
    ) = get_all_data(series_name, vanaf_jaar)
    df_sterfte = get_sterftedata(cbs_data_ruw, vanaf_jaar, series_name)
    
    df_corona, df_quantile, df_rivm, df_merged = calculate_dataframes(series_name, vanaf_jaar, period, how, df_sterfte)
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
        if series_name == "m_v_0_999":
           
            plot_graph_rivm(df_rivm, series_name, False)
            comparison(df_merged, series_name, smooth)
            do_kobak_vs_cbs(df_sterfte)

        df_steigstra = calculate_steigstra(df_merged, series_name)
        plot_steigstra_wrapper(df_steigstra, series_name)
    else:
        st.info(
            "De vergrlijking met vaccinateies, rioolwater etc is vooralsnog alleen mogelijk met CBS methode "
        )
    footer()

def calculate_dataframes(series_name, vanaf_jaar, period, how, df_sterfte):
    """
    Calculate various dataframes related to mortality data.

    This function processes mortality data to generate several dataframes, including
    quantile data, expected mortality, and merged dataframes based on the specified method.

    Parameters:
    series_name (str): The name of the series to process.
    vanaf_jaar (int): The starting year for the data.
    period (str): The period for which the data is processed.
    how (str): The method to use for processing ('quantiles' or other).
    df_sterfte (pd.DataFrame): The dataframe containing mortality data.

    Returns:
    tuple: A tuple containing the following dataframes:
        - df_corona (pd.DataFrame): Dataframe with corona-related data.
        - df_quantile (pd.DataFrame): Dataframe with quantile data.
        - df_rivm (pd.DataFrame): Dataframe with expected mortality data from RIVM.
        - df_merged (pd.DataFrame): Merged dataframe based on the specified method.
    """
     
    print(f"---{series_name}----")
    df_data = get_data_for_series_wrapper(df_sterfte, series_name, vanaf_jaar).copy(deep=True)
    
    _, df_corona, df_quantile = make_df_quantile(series_name, df_data, period)
    df_rivm = verwachte_sterfte_rivm(df_sterfte, series_name)

    if how == "quantiles":
        
        df_merged = make_df_merged(df_data, df_rivm, series_name, period)
        
        if 1==2: #if you want an export
            df_merged = df_merged.assign(
                jaar_week=df_merged["periodenr"],
                basevalue=df_merged["avg"],
                OBS_VALUE_=df_merged["aantal_overlijdens"]
            )

           
            df_to_export = df_merged[["jaar_week","basevalue","OBS_VALUE_"]]
            df_to_export["age_sex"]= "Y0-120_T"
            
            try:
                df_to_export.to_csv(f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\basevalues_sterfte_Y0-120_T.csv")
            except:
                pass
    return df_corona,df_quantile,df_rivm,df_merged

if __name__ == "__main__":
    import datetime

    print(
        f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------"
    )

    main()
