# Bereken het percentuele verschil tov een x-aantal dagen ervoor
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sn
import platform
import datetime
import datetime as dt
import streamlit as st
from streamlit import caching
from helpers import cell_background, select_period, save_df, drop_columns
from datetime import datetime



def day_to_day(df, column_, numberofdays):
    column_ = column_ if type(column_) == list else [column_]
    newcolumns = []   # percentuele verandering
    newcolumns2 = []  # index
    df_new =  pd.DataFrame()

    df_new["date"] = None
    for c in column_:
        newname = str(c) + "_daydiff"
        #newname2 = str(c) + "_daydiff_index"
        newcolumns.append(newname)
        #newcolumns2.append(newname2)
        df[newname] = np.nan
        #df[newname2] = np.nan
        df_new[c] = np.nan
        #df_new[newname2] = np.nan
        for n in range(numberofdays, len(df)):
            #df_new.at[n, "date"] = df.iloc[n]["pos_test_Date_statistics"]
            df_new.at[n, "date"] = df.iloc[n]["date"]

            vorige_day = df.iloc[n - numberofdays][c]
            nu = df.iloc[n][c]
            waarde = round((((nu - vorige_day) / vorige_day) * 100), 2)
            #waarde2 = round((((nu) / vorige_day) * 100), 2)

            df.at[n, newname] = waarde
            #df.at[n, newname2] = waarde2

            df_new.at[n, c] = waarde
            #df_new.at[n, newname2] = waarde2
            # df_new.at[n, "date"] = datetime.datetime.strftime(df_new.at[n,"date"], '%Y-%m-%d')
            #df_new.at[n, "date"] = datetime.datetime.strptime(df_new.at[n,"date"], '%Y-%m-%d').date()


    return df, df_new, newcolumns

def week_to_week(df, column_):
    column_ = column_ if type(column_) == list else [column_]
    newcolumns = []
    newcolumns2 = []

    for c in column_:
        newname = str(c) + "_weekdiff"
        newname2 = str(c) + "_weekdiff_index"
        newcolumns.append(newname)
        newcolumns2.append(newname2)
        df[newname] = np.nan
        df[newname2] = np.nan
        for n in range(7, len(df)):
            vorige_week = df.iloc[n - 7][c]
            nu = df.iloc[n][c]
            waarde = round((((nu - vorige_week) / vorige_week) * 100), 2)
            waarde2 = round((((nu) / vorige_week) * 100), 2)
            df.at[n, newname] = waarde
            df.at[n, newname2] = waarde2
    return df, newcolumns, newcolumns2

@st.cache(ttl=60 * 60 * 24)
def get_data():
    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
    else:
        url1= "https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv"

    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")
    df = df.groupby(["Date_statistics", "Agegroup"], sort=True).count().reset_index()
    return df

def do_the_rudi(df):
    """Calculate the fractions per age group. Calculate the difference related to day 0 as a % of day 0.
    Made for Rudi Lousberg
    Inspired by Ian Denton https://twitter.com/IanDenton12/status/1407379429526052866

    Args:
        df (df): table with numbers

    Returns:
        df : table with the percentual change of the fracctions
    """
    # calculate the sum
    #df = df.drop(columns="index", axis=1)

    df["sum"] = df. sum(axis=1)

    # make a new df with the fraction, row-wize  df_fractions A
    nr_of_columns = len (df.columns)
    nr_of_rows = len(df)
    column_list = df.columns.tolist()

    # calculate the fraction of each age group
    data  = []
    for r in range(nr_of_rows):
        row_data = []
        for c in range(nr_of_columns):
            try:
                row_data.append(round((df.iat[r,c]/df.at[r,"sum"]*100),2))
            except:
                row_data.append( df.iat[r,c])
        data.append(row_data)
    df_fractions = pd.DataFrame(data, columns=column_list)

    # calculate the percentual change of the fractions
    data  = []
    for r in range(nr_of_rows):
        row_data = []
        for c in range(nr_of_columns):
            try:
                row_data.append( round(((df_fractions.iat[r,c]  -  df_fractions.iat[0,c]  )/df_fractions.iat[0,c]*100),2))
            except:
                row_data.append(  df_fractions.iat[r,c])
        data.append(row_data)
    return pd.DataFrame(data, columns=column_list)


def main():
    # online version : https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv
    # DAILY STATISTICS ################

    start_ = "2021-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    global from_
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is valid and/or in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    if until_ == "2023-08-23":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
            caching.clear_cache()
            st.success("Cache is cleared, please reload to scrape new values")

    df_getdata = get_data()
    df = df_getdata.copy(deep=False)

    df = select_period(df, FROM, UNTIL)
    df.rename(
        columns={
            "Date_file": "count",
        },
        inplace=True,
    )
    # df_hospital = df[df["Hospital_admission"] == True].copy(deep=False)
    # df_deceased = df[df["Deceased"] == True].copy(deep=False)

    df_pivot = (
        pd.pivot_table(
            df,
            values="count",
            index=["Date_statistics"],
            columns=["Agegroup"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    # df_pivot_hospital = (
    #     pd.pivot_table(
    #         df_hospital,
    #         values="count",
    #         index=["Date_statistics"],
    #         columns=["Agegroup"],
    #         aggfunc=np.sum,
    #     )
    #     .reset_index()
    #     .copy(deep=False)
    # )

    # df_pivot_deceased = (
    #     pd.pivot_table(
    #         df_deceased,
    #         values="count",
    #         index=["Date_statistics"],
    #         columns=["Agegroup"],
    #         aggfunc=np.sum,
    #     )
    #     .reset_index()
    #     .copy(deep=False)
    # )

    df_pivot = df_pivot.add_prefix("pos_test_")
    # df_pivot_hospital = df_pivot_hospital.add_prefix("hosp_")
    # df_pivot_deceased = df_pivot_deceased.add_prefix("deceased_")

    todrop = [
        "Date_statistics_type",
        "Sex",
        "Province",
        "Hospital_admission",
        "Deceased",
        "Week_of_death",
        "Municipal_health_service",
    ]
    df = drop_columns(df, todrop)
    # save_df(df, "landelijk_leeftijd_2")

    # save_df(df_pivot, "landelijk_leeftijd_pivot")
    # save_df(df_pivot_hospital, "landelijk_leeftijd_pivot_hospital")
    # save_df(df_pivot_deceased, "landelijk_leeftijd_pivot_deceased")

    # df_temp = pd.merge(
    #     df_pivot,
    #     df_pivot_hospital,
    #     how="outer",
    #     left_on="pos_test_Date_statistics",
    #     right_on="hosp_Date_statistics",
    # )
    # df_temp = pd.merge(
    #     df_temp,
    #     df_pivot_deceased,
    #     how="outer",
    #     left_on="pos_test_Date_statistics",
    #     right_on="deceased_Date_statistics",
    # )
    # save_df(df_temp, "final_result")

    column_list = df_pivot.columns.tolist()

    column_list = column_list[1:]
    numberofdays = st.sidebar.slider("Vergelijken met x dagen ervoor", 0, 21, 7)
    st.subheader("Number of cases per age")
    df_pivot['pos_test_Date_statistics'] = df_pivot['pos_test_Date_statistics'].dt.date
    df_pivot.rename(columns={"pos_test_Date_statistics": "date"},  inplace=True)
    st.write (df_pivot)
    df_pivot_2,df_new, newcolumns,= day_to_day(df_pivot, column_list, numberofdays)
    st.sidebar.write("Attention : slow script!!!")

    # save_df(df_new, "daily_changes_casus_landelijk_age")
    #st.dataframe(df_new.style.applymap(cell_background))

    #df_new['date'] = df_new['date'].date
    #df_new.rename(columns={"Date_statistics": "date"},  inplace=True)
    df_new.reset_index(drop=True)
    # st.write(df_new)
    # st.write(df_new.dtypes)
    st.subheader(f"Percentual change with {numberofdays} days before")
    st.write(df_new.style.format(None, na_rep="-").applymap(cell_background).set_precision(2))

    #df_pivot_2['date'] = df_pivot_2['date'].dt.date

    df_new_rudi = do_the_rudi(df_pivot)
    #df_new_rudi['pos_test_Date_statistics'] = df_new_rudi['pos_test_Date_statistics'].dt.date
    #df_new_rudi.rename(columns={"pos_test_Date_statistics": "date"},  inplace=True)
    st.subheader("Percentual change of the fractions per agegroup with day 0 - under construction")
    st.write(df_new_rudi.style.format(None, na_rep="-").applymap(cell_background).set_precision(2))

if __name__ == "__main__":
    main()