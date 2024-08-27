import pandas as pd
# import numpy as np
import platform

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib as mpl
# import matplotlib.dates as mdates
# from textwrap import wrap
# import matplotlib.cm as cm
# import seaborn as sn
# from scipy import stats
# import datetime as dt
# from datetime import datetime, timedelta

# from streamlit.errors import NoSessionContext

# import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from matplotlib.font_manager import FontProperties
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
# import matplotlib.ticker as ticker
# import math
# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock
# from scipy.signal import savgol_filter
# from sklearn.metrics import r2_score
import streamlit as st
# import urllib
# import urllib.request
# from pathlib import Path
# #from streamlit import caching
# from inspect import currentframe, getframeinfo
# from helpers import *
# import covid_dashboard_show_toelichting_footer


def get_dataframe():
    if platform.processor() != "":
        url_mzelst = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"
    
    else:
        url_mzelst = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"
    df_mzelst = pd.read_csv(url_mzelst, delimiter=',', low_memory=False)
    df_mzelst= df_mzelst[["date", "Hospital_Intake_Proven"]]
    df_mzelst,column_sma = make_sma(df_mzelst,"Hospital_Intake_Proven", 7, False)
    df_mzelst['date'] = pd.to_datetime(df_mzelst['date'])  

    # add columns we are going to use/calculate
    df_mzelst["R_werkelijk_ziekenhuisopn"]  =None
    df_mzelst["R0_virus"] = 0
    df_mzelst["avg"]  = None
    df_mzelst["Reff"]  = None
    df_mzelst["Immuun"]  = None
    df_mzelst["Besmettelijk"]  = None
    df_mzelst["Besmet"]  = None
    df_mzelst["Opnames"]  = None
    return df_mzelst,column_sma

  
def make_scatter(df_totaal, x,y, title):
    fig1 = px.scatter(df_totaal, x=x, y=y, title=title)               
    st.plotly_chart(fig1, use_container_width=True)

def make_sma(df_totaal, column, window, center):
    column_sma = column+"_sma"
    df_totaal[column_sma] =  df_totaal[column].rolling(window = window, center = center).mean()
    return df_totaal, column_sma




def main():
    df,column_sma  = get_dataframe()
   
    d = 7
    tg = 3.8
    
            
    cut_off1 = datetime.strptime("2022-04-20", "%Y-%m-%d")
    cut_off2 = datetime.strptime("2022-08-17", "%Y-%m-%d")
    cut_off3 = datetime.strptime("2022-11-03", "%Y-%m-%d")
    print (cut_off1)
    for i in range(len(df)-7):
        df.at[i,"R_werkelijk_ziekenhuisopn"] = round(((df.at[i+7,column_sma] / df.at[i,column_sma]) ** (tg / d)), 2)

        if df.at[i,"date"] >= cut_off1 and df.at[i,"date"] < cut_off2:
            df.at[i,"R0_virus"] = 3.5
        elif df.at[i,"date"] >= cut_off2 and  df.at[i,"date"] < cut_off3:
            if df.at[i-1,"R0_virus"] < 5.49:
                df.at[i,"R0_virus"] = df.at[i-1,"R0_virus"]*1.015
            else:
                df.at[i,"R0_virus"] = 5.5
        elif df.at[i,"date"] >= cut_off3:
            if df.at[i-1,"R0_virus"] < 7.99:
                df.at[i,"R0_virus"] = df.at[i-1,"R0_virus"]*1.02
            else:
                df.at[i,"R0_virus"] = 8.0

    make_scatter(df, "date","R0_virus", "RO_virus door de tijd")
    df,column_sma =make_sma(df, "R0_virus", 15, True)

        
    #print (df.to_string())



main()

