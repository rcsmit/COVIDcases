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


import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime

def save_df(df, name):
    """  save dataframe on harddisk """
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\output\\"
    )
    OUTPUT_DIR = (
      "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\COVIDcases\\input\\")
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


def main_x():
    # online version : https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv
    url1 = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")
    df.rename(
        columns={
            "Date_file": "count",
        },
        inplace=True,
    )

    #until = dt.datetime.strptime("2021-1-1", "%Y-%m-%d").date()
    #mask = (df["Date_statistics"].dt.date >= dt.datetime.strptime("2020-1-1", "%Y-%m-%d").date()) & (df["Date_statistics"].dt.date <= until)
    #df = df.loc[mask]

    df_hospital = df[df["Hospital_admission"] == "Yes"].copy(deep=False)
    df_deceased = df[df["Deceased"] == "Yes"].copy(deep=False)

    df_all = df.groupby([ "Agegroup"], sort=True).count().reset_index()
    df_hospital = df_hospital.groupby([ "Agegroup"], sort=True).count().reset_index()
    df_deceased = df_deceased.groupby(["Date_statistics", "Agegroup"], sort=True).count().reset_index()
    #df_deceased = df_deceased.groupby([ "Agegroup"], sort=True).count().reset_index()

    df = df.groupby(["Date_statistics", "Agegroup"], sort=True).count().reset_index()
    print ("CASES")
    #df_all = df_all[["Agegroup", "count"]]
    #df_hospital = df_hospital[["Agegroup", "count"]]
    print (df_all)
    print ("ZIEKENHUISOPNAMES")
    print (df_hospital)

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

    df_pivot_hospital = (
        pd.pivot_table(
            df_hospital,
            values="count",
            index=["Date_statistics"],
            columns=["Agegroup"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    df_pivot_deceased = (
        pd.pivot_table(
            df_deceased,
            values="count",
            index=["Date_statistics"],
            columns=["Agegroup"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    df_pivot = df_pivot.add_prefix("pos_test_")
    df_pivot_hospital = df_pivot_hospital.add_prefix("hosp_")
    save_df(df_pivot_hospital, "df_hospital_per_dag_vanuit_casus_landelijk")
    df_pivot_deceased = df_pivot_deceased.add_prefix("deceased_")
    print(df_pivot_deceased.dtypes)
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
    save_df(df, "landelijk_leeftijd_2_vanuit_casus_landelijk")

    save_df(df_pivot, "landelijk_leeftijd_pivot_vanuit_casus_landelijk")
    save_df(df_pivot_hospital, "landelijk_leeftijd_pivot_hospital_vanuit_casus_landelijk")
    save_df(df_pivot_deceased, "landelijk_leeftijd_pivot_deceased_vanuit_casus_landelijk")


    df_pivot_cases_per_week = df_pivot.groupby(pd.Grouper(key='pos_test_Date_statistics', freq='W')).sum()
    df_pivot_cases_per_week.index -= pd.Timedelta(days=6)
    df_pivot_cases_per_week["weekstart"]= df_pivot_cases_per_week.index
    save_df(df_pivot_cases_per_week, "landelijk_leeftijd_pivot_per_week_vanuit_casus_landelijk")

    df_temp = pd.merge(
        df_pivot,
        df_pivot_hospital,
        how="outer",
        left_on="pos_test_Date_statistics",
        right_on="hosp_Date_statistics",
    )
    df_temp = pd.merge(
        df_temp,
        df_pivot_deceased,
        how="outer",
        left_on="pos_test_Date_statistics",
        right_on="deceased_Date_statistics",
    )

    df_temp_per_week = df_temp.groupby(pd.Grouper(key='pos_test_Date_statistics', freq='W')).sum()
    df_temp_per_week.index -= pd.Timedelta(days=6)
    print(df_temp_per_week)
    df_temp_per_week["weekstart"]= df_temp_per_week.index
    save_df(df_temp, "final_result_vanuit_casus_landelijk")
    save_df(df_temp_per_week, "final_result_per_week_vanuit_casus_landelijk")


def main_week_data():
    """Het maken van weekcijfers en gemiddelden tbv cases_hospital_decased_NL.py
    """
    # online version : https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv
    url1 = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    todrop = [
        "Date_statistics_type",
        "Sex",
        "Province",
        "Week_of_death",
        "Municipal_health_service",
    ]
    df = drop_columns(df, todrop)

    df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")
    df = df.replace("Yes", 1)
    df = df.replace("No", 0)
    df = df.replace("Unknown", 0)
    df["cases"] = 1
    print(df)
    #df = df.groupby([ "Date_statistics", "Agegroup"], sort=True).sum().reset_index()
    df_week = df.groupby([  pd.Grouper(key='Date_statistics', freq='W'), "Agegroup",] ).sum().reset_index()
    print (df)
    df_week["Hosp_per_reported"] = df_week["Hospital_admission"]/df_week["cases"]
    df_week["Deceased_per_reported"] = df_week["Deceased"]/df_week["cases"]
    save_df(df_week, "landelijk_leeftijd_week_vanuit_casus_landelijk_20211006")

main_week_data()
