# PREPARE A CSV-FILE TO ENABLE AN STACKED PLOT FOR POSITIVE TESTS, HOSPITALIZATIONS AND DECEASED
# Hospitalizations and deceased are not lagged in time, the date of the result of the "desease onset", positieve test or notification is leading
# https://data.rivm.nl/geonetwork/srv/dut/catalog.search#/metadata/2c4357c8-76e4-4662-9574-1deb8a73f724

# MARCH 2021, Rene Smit (@rcsmit) - MIT license

# Fields in
# Date_file;Date_statistics;Date_statistics_type;Agegroup;Sex;
# Province;Hospital_admission;Deceased;Week_of_death;Municipal_health_service

# Fields out
# pos_test_Date_statistics,pos_test_0-9,pos_test_10-19,pos_test_20-29,pos_test_30-39,
# pos_test_40-49,pos_test_50-59,pos_test_60-69,pos_test_70-79,pos_test_80-89,pos_test_90+,
# pos_test_<50,pos_test_Unknown,hosp_Date_statistics,hosp_0-9,hosp_10-19,hosp_20-29,hosp_30-39,
# hosp_40-49,hosp_50-59,hosp_60-69,hosp_70-79,hosp_80-89,hosp_90+,hosp_<50,hosp_Unknown,
# deceased_Date_statistics,deceased_0-9,deceased_10-19,deceased_20-29,deceased_30-39,
# deceased_40-49,deceased_50-59,deceased_60-69,deceased_70-79,deceased_80-89,deceased_90+,
# deceased_<50,deceased_Unknown

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sn
import platform
import datetime
import datetime as dt
import streamlit as st
from streamlit import caching


def save_df(df, name):
    """  save dataframe on harddisk """
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\"
    )

    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")


def drop_columns(df, what_to_drop):
    """  drop columns. what_to_drop : list """
    if what_to_drop != None:
        print("dropping " + str(what_to_drop))
        for d in what_to_drop:
            df = df.drop(columns=[d], axis=1)
    return df

def day_to_day(df, column_, numberofdays):
    if type(column_) == list:
        column_ = column_
    else:
        column_ = [column_]
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
            df_new.at[n, "date"] = df.iloc[n]["pos_test_Date_statistics"]
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
    if type(column_) == list:
        column_ = column_
    else:
        column_ = [column_]
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

def color_value(val):


    try:
        v = abs(val)
        opacity = 1 if v >100 else v/100
        # color = 'green' if val >0 else 'red'
        if val > 0 :
             color = '255, 0, 0'
        elif val < 0:
            color = '76, 175, 80'
        else:
            color = '255,255,255'

        # if val == NaN:
        #     color = 'white'
        # elif val > 0.0:
        #     color = 'green'
        # elif val<0.0 :
        #     color = '#ff0000'
        # else:
        #     color = '#ffffff'
    except:
        color = '255,255,255'
        opacity = 1

    #return f'background-color: {color}; '
    return f'background: rgba({color}, {opacity})'

def main():
    # online version : https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv
    df_getdata = get_data()
    df = df_getdata.copy(deep=False)
    df.rename(
        columns={
            "Date_file": "count",
        },
        inplace=True,
    )
    df_hospital = df[df["Hospital_admission"] == True].copy(deep=False)
    df_deceased = df[df["Deceased"] == True].copy(deep=False)

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
    save_df(df, "landelijk_leeftijd_2")

    save_df(df_pivot, "landelijk_leeftijd_pivot")
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
    df_pivot,df_new, newcolumns,= day_to_day(df_pivot, column_list, 1)
    #df_new["date"] = pd.to_datetime(df_new["date"], format="%Y-%m-%d")
    df_new.set_index('date')
    save_df(df_new, "daily_changes_casus_landelijk_age")
    # fig, ax = plt.subplots(figsize=(11, 9))

    #ax.set_xticklabels(df_new['date'].dt.strftime('%Y-%m-%d'))
    #df_new['date'] = df_new.date.date().astype('string')
    # df_new = df_new.drop(columns="date", axis=1)
    # sn.heatmap(df_new, annot=True, annot_kws={"fontsize": 7}, vmax=100, vmin = -100)
    # plt.show()




    # cm = sn.light_palette("green", as_cmap=True)

    # df_new.style.background_gradient(cmap=cm)

    st.dataframe(df_new.style.applymap(color_value))
if __name__ == "__main__":
    main()