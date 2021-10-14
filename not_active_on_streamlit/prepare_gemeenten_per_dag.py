# PREPARE A CSV-FILE TO WITH THE NUMBER OF CASES PER MUNICIPALITY IN A CERTAIN PERIOD

# MARCH 2021, Rene Smit (@rcsmit) - MIT license



import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime

def save_df(df, name):
    """  save dataframe on harddisk """
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\"
    )
    OUTPUT_DIR = (
      "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\COVIDcases\\input\\")
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")




def main_week_data():
    """Het maken van weekcijfers en gemiddelden tbv cases_hospital_decased_NL.py
    """
    # online version : https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv
    url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_aantallen_gemeente_per_dag.csv"
    #C:\Users\rcxsm\Documents\phyton_scripts\covid19_seir_models\COVIDcases\input
    datefield="Date_of_report"
    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    df[datefield] = pd.to_datetime(df[datefield], format="%Y-%m-%d")
    df = df[df["Municipality_code"] != None]
    print (df)
    from_  = dt.datetime.strptime("2021-10-7", "%Y-%m-%d").date()
    until = dt.datetime.strptime("2021-10-14", "%Y-%m-%d").date()
    mask = (df[datefield].dt.date >= from_) & (df[datefield].dt.date <= until)
    df = df.loc[mask]
    print (df)
    df = df.groupby(["Municipality_code"] ).sum().reset_index()
    #df_week = df.groupby([  pd.Grouper(key='Date_statistics', freq='W'), "Agegroup",] ).sum().reset_index()
    print (df)
    save_df(df,"gemeente_reported_hospital_deceased")

main_week_data()
