import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

# import seaborn as sn
from scipy import stats
import datetime as dt
from datetime import datetime, timedelta

import json
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as ticker
import math

_lock = RendererAgg.lock
from scipy.signal import savgol_filter
import streamlit as st
import urllib
import urllib.request
from pathlib import Path
from streamlit import caching
from inspect import currentframe, getframeinfo

import covid_dashboard_rcsmit

def save_df(df, name):
    """  _ _ _ """


    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\output\\"
    )
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")

@st.cache(ttl=60 * 60 * 24)
def download_data_file(url, filename, delimiter_, fileformat):
    """Download the external datafiles
    IN :  url : the url
          filename : the filename (without extension) to export the file
          delimiter : delimiter
          fileformat : fileformat
    OUT : df_temp : the dataframe
    """
    INPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\"
    )
    # df_temp = None
    download = True
    with st.spinner(f"Downloading...{url}"):
        if download:  # download from the internet
            url = url
        elif fileformat == "json":
            url = INPUT_DIR + filename + ".json"
        else:
            url = INPUT_DIR + filename + ".csv"

        if fileformat == "csv":
            df_temp = pd.read_csv(url, delimiter=delimiter_, low_memory=False)
        elif fileformat == "json":
            df_temp = pd.read_json(url)

        # elif fileformat =='json_x':   # workaround for NICE IC data
        #     pass
        #     # with urllib.request.urlopen(url) as url_x:
        #     #     datajson = json.loads(url_x.read().decode())
        #     #     for a in datajson:
        #     #         df_temp = pd.json_normalize(a)
        else:
            print ("error")
            # st.error("Error in fileformat")
            # st.stop()
        df_temp = df_temp.drop_duplicates()
        # df_temp = df_temp.replace({pd.np.nan: None})  Let it to test
        #save_df(df_temp, filename)
        return df_temp

def drop_columns(df,what_to_drop):
    """  _ _ _ """
    if what_to_drop != None:
        #print("dropping " + what_to_drop)
        for d in what_to_drop:
            df = df.drop(columns=[d], axis=1)
    return df

def show_graph(df_new):

    color_list = [  "#ff6666",  # reddish 0
                    "#ac80a0",  # purple 1
                    "#3fa34d",  # green 2
                    "#EAD94C",  # yellow 3
                    "#EFA00B",  # orange 4
                    "#7b2d26",  # red 5
                    "#3e5c76",  # blue 6
                    "#e49273" , # dark salmon 7
                    "#1D2D44",  # 8
                    "#02A6A8",
                    "#4E9148",
                    "#F05225",
                    "#024754",
                    "#FBAA27",
                    "#302823",
                    "#F07826",
                     ]

    df_new["rolling"] = (
                    df_new.iloc[:, df_new.columns.get_loc("gevallen_per_100k_inw")]
                    .rolling(window=7, center=True)
                    .mean()
                )
    with _lock:
        fig1y = plt.figure()
        ax = fig1y.add_subplot(111)
        #for l in ages_to_show_in_graph:

        ax.xaxis_date()
        ax.set_xticks(df_new["date"].index)

        xticks = ax.xaxis.get_major_ticks()

        for i, tick in enumerate(xticks):
            if i % 10 != 0:
                tick.label1.set_visible(False)

        plt.xticks()

        ax =  df_new["rolling"].plot( label = "label")
        ax.yaxis.grid(True, which="major", alpha=0.4)

        ax.set_xticklabels(df_new["date"], fontsize=6, rotation=90)
        ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',
                    transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
        plt.title(f"gevallen per province per agegroup\nper 100k inwoners" , fontsize=10)
        plt.tight_layout()
        plt.show()
        st.pyplot(fig1y)


def groupeer_data():
    url = "COVID-19_casus_landelijk.csv"
    filename = "COVID-19_casus_landelijk"
    delimiter_ = ";"
    fileformat = "csv"
    print ("downloading")
    df_1 = download_data_file(url, filename, delimiter_, fileformat)
    df_1["Date_statistics"] = pd.to_datetime(
             df_1["Date_statistics"] , format="%Y-%m-%d")
    to_drop= ["Sex","Hospital_admission","Deceased","Week_of_death","Municipal_health_service"]
    df_1 = drop_columns(df_1, to_drop)
    df_1['Count']=1
    df_2 =  df_1.groupby(["Date_statistics", "Agegroup", "Province"], sort=True).sum().reset_index()
    save_df(df_2, "cases_per_age_per_provincex.csv")

    df_prov = download_data_file("provincies.csv", "provincies", ",", "csv")
    df_merge = pd.merge(
                df_2, df_prov, how="outer", left_on="Province", right_on="provincie"
            )
    df_merge["gevallen_per_100k_inw"] = round(
        (df_merge["Count"] *100_000 / df_merge["inwoners"] ), 4)
    df_merge["date"] = df_merge["Date_statistics"]
    df_merge["date"] = pd.to_datetime(
             df_merge["date"] , format="%Y-%m-%d")
    save_df(df_merge, "cases_per_age_per_province.csv")

    return df_merge

def main():
    local = True
    if local:
        df = download_data_file("https://raw.githubusercontent.com/rcsmit/COVIDcases/main/cases_per_age_per_province.csv", "cases_per_age_per_province", ",",  "csv")
    else:
        df = groupeer_data()

    print (df)
    provincielijst = ["Groningen","Fryslân","Drenthe","Overijssel","Flevoland","Gelderland","Utrecht","Noord-Holland","Zuid-Holland","Zeeland","Noord-Brabant","Limburg"]
    leeftijdslijst = ["0-9","10-19","20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]

    prov_1 = st.sidebar.selectbox(
            "provincie links",
            provincielijst,
            index=0,
        )

    leeftijd1 =  st.sidebar.selectbox(
            "provincie links",
            leeftijdslijst,
            index=0,
        )

    df1 = df[(df['Agegroup'] == leeftijd1) & (df['provincie'] ==prov_1 )]
    df1.rename(columns={"Date_statistics": "date"},
        inplace=True,
    )
    #st.write(df1.dtypes)
    show_graph(df1)
    st.write ("Er is gedeeld door de gehele bevolking van de provincie. Bij het vergelijken tussen provincies zorgt dit voor onnauwkeurigheden aangezien er verschillen van leeftijdsopbouw tussen de provincies is.")
    # if st.sidebar.button("Clear cache"):
    #         caching.clear_cache()

if __name__ == "__main__":
    main()