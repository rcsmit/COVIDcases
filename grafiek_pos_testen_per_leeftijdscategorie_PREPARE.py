# Prepare Grafiek positief getest naar leeftijd door de tijd heen, per leeftijdscategorie
# René Smit, (@rcsmit) - MIT Licence

# IN: tabel met positief aantal testen en totaal aantal testen per week, gecategoriseerd naar leeftijd
#     handmatig overgenomen uit Tabel 14 vh wekelijkse rapport van RIVM
#     Wekelijkse update epidemiologische situatie COVID-19 in Nederland
#     https://www.rivm.nl/coronavirus-covid-19/actueel/wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland

# Uitdagingen : Kopieren en bewerken Tabel 14. 3 verschillende leeftijdsindelingen. Tot dec. 2020 alles
# cummulatief. X-as in de grafiek

# TODO : - Nog enkele weken toevoegen voorafgaand het huidige begin in de datafile (waren weken met weinig besmettingen).
#        - integreren in het dashboard
#        - 'Total reported' toevoegen


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from labellines import *  #https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
import streamlit as st
#from streamlit import caching
# from matplotlib.backends.backend_agg import RendererAgg
from datetime import datetime
# _lock = RendererAgg.lock

def save_df(df,name):
    """  _ _ _ """
    OUTPUT_DIR = 'C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\output\\'

    name_ =  OUTPUT_DIR + name+'.csv'
    compression_opts = dict(method=None,
                            archive_name=name_)
    try:
        df.to_csv(name_, index=False,
                compression=compression_opts)

        print ("--- Saving "+ name_ + " ---" )
    except:
        print("NOT saved")
def read_df():
    # Local file
    #url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\covid19_seir_models\input\pos_test_leeftijdscat_wekelijks.csv"
    # File before regroping the agecategories. Contains ; as delimiter and %d-%m-%Y as dateformat
    url= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/pos_test_leeftijdscat_wekelijks.csv"



    df_new   = pd.read_csv(url,
                        delimiter=";",
                        low_memory=False)
    df_new["datum"]=pd.to_datetime(df_new["datum"], format='%d-%m-%Y')
    return df_new


def regroup_df(kids_split_up):
    """ regroup the age-categories """

    df_= read_df()
    df = df_.copy(deep=False)
    df["datum"]=pd.to_datetime(df["datum"], format='%d-%m-%Y')

    list_dates = df["datum"].unique()

    cat_oud =            [ "0-4",  "05-09",  "10-14",  "15-19",  "20-24",  "25-29",  "30-34",  "35-39",
                "40-44",  "45-49",  "50-54",  "55-59",  "60-64",  "65-69",  "70-74",  "75-79",  "80-84",  "85-89",  "90-94",  "95+", "5-sep", "okt-14", "4-dec" ]

    cat_nieuw =           [ "0-12", "13-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld", "5-sep", "okt-14", "4-dec"]

    cat_nieuwst=            ["0-3", "04-12", "13-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld", "5-sep", "okt-14", "4-dec"]

    #originele groepsindeling
    if kids_split_up:
        cat_oud_vervanging = [ "0-4",  "05-09",  "10-14",  "15-19",  "20-29",  "20-29",  "30-39",  "30-39",
                "40-49",  "40-49",  "50-59",  "50-59",  "60-69",  "60-69",   "70+",   "70+",      "70+",   "70+",    "70+",   "70+", "05-09", "10-14", "04-12" ]

        cat_nieuw_vervanging = [ "0-12", "13-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld","05-09", "10-14", "04-12"]
        cat_nieuwst_vervanging= ["0-3", "04-12", "13-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld","05-09", "10-14", "04-12"]
    else:
        cat_oud_vervanging = [ "0-19",  "0-19",  "0-19",  "0-19",  "20-29",  "20-29",  "30-39",  "30-39",
                "40-49",  "40-49",  "50-59",  "50-59",  "60-69",  "60-69",   "70+",   "70+",      "70+",   "70+",    "70+",   "70+" , "0-19",  "0-19",  "0-19"]
        cat_nieuw_vervanging = [ "0-17", "0-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld", "0-17", "0-17", "0-17"]
        cat_nieuwst_vervanging= ["0-17", "0-17", "0-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld","0-17", "0-17", "0-17"]

    #####################################################
    df_new= pd.DataFrame({'date': [],'cat_oud': [],
                    'cat_nieuw': [], "positief_testen": [],"totaal_testen": [], "methode":[]})

    for i in range(len(df)):
        print(i, end = '')
        d =  df.loc[i, "datum"]
        for x in range(len(cat_oud)-1):
            c_o,c,p,t,m = None,None,None,None,None
            if df.loc[i, "methode"] == "oud":
                if df.loc[i, "leeftijdscat"] == cat_oud[x]:
                    c = cat_oud_vervanging[x]
            elif df.loc[i, "methode"] == "nieuw":
                if (
                    x <= len(cat_nieuw) - 1
                    and df.loc[i, "leeftijdscat"] == cat_nieuw[x]
                ):
                    c =  cat_nieuw_vervanging[x]
            elif (
                x <= len(cat_nieuw) - 1
                and df.loc[i, "leeftijdscat"] == cat_nieuwst[x]
            ):
                c =  cat_nieuwst_vervanging[x]
            c_o =  df.loc[i, "leeftijdscat"]
            p =df.loc[i, "totaal_pos"]
            t = df.loc[i, "totaal_getest"]
            m = df.loc[i, "methode"]
            
                        # Initialize an empty list to collect rows
            rows = []

            # Assuming this code is inside a loop where d, c_o, c, p, t, and m are defined
            rows.append({
                'date': d,
                'cat_oud': c_o,
                'cat_nieuw': c,
                'positief_testen': p,
                'totaal_testen': t,
                'methode': m
            })

            # After the loop, create a DataFrame from the collected rows
            df_new = pd.DataFrame(rows)

            c_o,c,p,t,m = None,None,None,None,None

    df_new = df_new.groupby(['date','cat_nieuw'], sort=True).sum().reset_index()
    if kids_split_up:
        save_df(df_new, "input_latest_age_pos_test_kids_seperated.csv")
    else:
        save_df(df_new, "input_latest_age_pos_tests.csv")

regroup_df( True)
regroup_df( False)