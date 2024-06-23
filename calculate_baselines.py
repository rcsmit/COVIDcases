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



def verwachte_sterfte_rivm(df, series_naam):
    """Verwachte sterfte/baseline  uitrekenen volgens RIVM methode

    _
    """

    # adding week 52, because its not in the data
    # based on the rivm-data, we assume that the numbers are quit the same

    df["boekjaar"] = df["jaar"].shift(26)
    df["boekweek"] = df["week"].shift(26)

    df_compleet = pd.DataFrame()
    for y in [2019, 2020, 2021, 2022, 2023]:
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




def get_kobak():
    """Load the csv with the excess mortality as calculated by Ariel Karlinsky and Dmitry Kobak
    https://elifesciences.org/articles/69336#s4
    https://github.com/dkobak/excess-mortality/
    Returns:
        _type_: _description_
    """

    # if platform.processor() != "":
    #     # C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\excess-mortality-timeseries_NL_kobak.csv

    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\excess-mortality-timeseries_NL_kobak.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/excess-mortality-timeseries_NL_kobak.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
    )

    return df_



def get_all_data(seriename):
    """_summary_

    Returns:
        _type_: df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater,df_
    """
   
    df_kobak = get_kobak()
    df_ = get_sterftedata(seriename)

    return df_,  df_kobak



def get_data_for_series(df_, seriename):

    df = df_[["jaar", "week", "weeknr", f"totaal_{seriename}"]].copy(deep=True)

    df = df[(df["jaar"] > 2014)]
    # df = df[df["jaar"] > 2014 | (df["weeknr"] != 0) | (df["weeknr"] != 53)]
    df = df.sort_values(by=["jaar", "weeknr"]).reset_index()

    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]
    # som_2015_2019 = 0
    noemer = 149832
    for y in range(2015, 2020):
        df_year = df[(df["jaar"] == y)]
        # som = df_year["m_v_0_999"].sum()
        # som_2015_2019 +=som

        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        #  https://www.cbs.nl/nl-nl/nieuws/2024/06/sterfte-in-2023-afgenomen/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2023%20is%20deze%20156%20666.
        # https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85753NED/table?dl=A787C
        # Define the factors for each year
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
            2023: 156666
            / noemer,  # or 169333 / som if you decide to use the updated factor
            2024: 157846 / noemer,
        }
    # avg_overledenen_2015_2019 = (som_2015_2019/5)
    # st.write(avg_overledenen_2015_2019)
    # Loop through the years 2014 to 2024 and apply the factors
    for year in range(2014, 2025):
        new_column_name = f"{seriename}_factor_{year}"
        factor = factors[year]
        # factor=1
        df[new_column_name] = df[f"totaal_{seriename}"] * factor

    return df


def rolling(df, what):

    df[f"{what}_sma"] = df[what].rolling(window=6, center=True).mean()

    # df[what] = df[what].rolling(window=6, center=False).mean()
    # df[f'{what}_sma'] = savgol_filter(df[what], 7,2)
    return df



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

def adjust_overledenen(df):
        """# Adjust "Overledenen_1" based on the week number
        # if week = 0, overledenen_l : add to week 52 of the year before
        # if week = 53: overleden_l : add to week 1 to the year after
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

# @st.cache_data(ttl=60 * 60 * 24)
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

  
    data_ruw = pd.DataFrame(cbsodata.get_data("70895ned"))
    data_ruw.to_csv("cbsodata_ruw.cvs")
    print ("opgeslagen")
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

def get_sterfte_data_fixed():
    #C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_per_week.csv
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_per_week.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )   
    df_ = df_.rename(columns={"weeknr": "week"})
    df_ = df_.rename(columns={"jaar": "year"})
    # jaar;weeknr;aantal_dgn;voorlopig;totaal_m_v_0_999;totaal_m_0_999;totaal_v_0_999;totaal_m_v_0_65;totaal_m_0_65;totaal_v_0_65;totaal_m_v_65_80;totaal_m_65_80;totaal_v_65_80;totaal_m_v_80_999;totaal_m_80_999;totaal_v_80_999

    return df_
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

            df_to_use_ = df_to_use[(df_to_use["week"] == w)].copy(deep=True)

            column_to_use = series_name + "_factor_" + str(year)
            data = df_to_use_[column_to_use]  # .tolist()

            try:
                q05 = np.percentile(data, 5)
                q25 = np.percentile(data, 25)
                q50 = np.percentile(data, 50)
                q75 = np.percentile(data, 75)
                q95 = np.percentile(data, 95)
            except:
                q05, q25, q50, q75, q95 = 0, 0, 0, 0, 0

            avg = round(data.mean(), 0)

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
    """Function to predict the baseline with linear regression
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



def do_kobak_vs_cbs(df_deaths):

    """Main function

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
def do_poisson(df):

    # Voeg een constante toe aan het model (intercept)
    df['intercept'] = 1

    # Selecteer de jaren 2015-2019 voor het trainen van het model
    train_data = df[df['jaar'] < 2020]

    # Poisson model
    poisson_model = sm.GLM(train_data['observed_deaths'], train_data[['intercept', 'week']], family=sm.families.Poisson())
    poisson_results = poisson_model.fit()

    # Voor een quasi-Poisson model gebruik je de Tweedie familie met power=1
    quasi_poisson_model = sm.GLM(train_data['observed_deaths'], train_data[['intercept', 'week']], family=sm.families.Tweedie(var_power=1))
    quasi_poisson_results = quasi_poisson_model.fit()

    # Voorspellingen voor 2020 maken
    predict_data = df[df['year'] == 2020]
    predict_data['expected_deaths'] = poisson_results.predict(predict_data[['intercept', 'week']])
    # Voor quasi-Poisson model:
    predict_data['expected_deaths_quasi_poisson'] = quasi_poisson_results.predict(predict_data[['intercept', 'week']])

    predict_data['excess_deaths'] = predict_data['observed_deaths'] - predict_data['expected_deaths']
    predict_data['excess_deaths_quasi_poisson'] = predict_data['observed_deaths'] - predict_data['expected_deaths_quasi_poisson']

    # Bekijk de resultaten
    st.write(predict_data[['week', 'year', 'observed_deaths', 'expected_deaths', 'excess_deaths', "expected_deaths_quasi_poisson", "excess_deaths_quasi_poisson"]])

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
    st.header("Oversterfte - minder leeftijdscategorieen")
    st.subheader("CBS Methode")
    
    # (
    #     df_sterfte,
    #     df_kobak,
    # ) = get_all_data(series_name)

    # print(f"---{series_name}----")
    # df_data = get_data_for_series(df_sterfte, series_name).copy(deep=True)
    df_data = get_sterfte_data_fixed()
    
    df_data = df_data.sort_values(by=["year", "week"])
    df_data["observed_deaths"]= df_data[f"totaal_{series_name}"]
    df_data["Overledenen_1"] = df_data[f"totaal_{series_name}"]
    df_data["jaar"] = df_data["year"]
    df_data["weeknr"] = (
        df_data["jaar"].astype(str) + "_" + df_data["week"].astype(str).str.zfill(2)
    )
    get_data_for_series(df_data, series_name)
    df_data = adjust_overledenen(df_data)
    #df_compleet, df_corona, df_quantile = make_df_quantile(series_name, df_data)

    do_poisson(df_data)
    
if __name__ == "__main__":
    import datetime

    print(
        f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------"
    )

    main()
