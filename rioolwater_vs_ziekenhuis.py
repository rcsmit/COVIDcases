import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
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


# https://www.cbs.nl/nl-nl/maatwerk/2022/42/inwoners-per-rioolwaterzuiveringsinstallatie-1-1-2022
# https://www.rivm.nl/documenten/berekening-cijfers-rioolwatermetingen-covid-19



def move_column(df, column_, days):
    """Move/shift a column

    Args:
        df (df): df
        column_ (string): which column to move
        days (int): how many days

    Returns:
        df: df
        new_column : name of the new column
    """    """  _ _ _ """
    column_ = column_ if type(column_) == list else [column_]
    for column in column_:
        new_column = column + "_moved_" + str(days)
        df[new_column] = df[column].shift(days)
    return df, new_column



#url_inwoners = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\inwoners_rzwi.csv"
url_inwoners = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/inwoners_rzwi.csv"
#url_rioolwaterdata= "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
url_rioolwaterdata = r"C:\Users\rcxsm\Downloads\COVID-19_rioolwaterdata.csv"

#url_lcps = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/lcps_by_day.csv"
url_lcps = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"
df_inwoners =  pd.read_csv(url_inwoners, delimiter=';', low_memory=False)

df_inwoners = df_inwoners[df_inwoners["regio_type"] == "VR"]
df_inwoners["aandeel_maal_inwonders"]= df_inwoners["aandeel"] * df_inwoners["inwoners"]/100
df_rioolwaterdata = pd.read_csv(url_rioolwaterdata, delimiter=';', low_memory=False)
df_rioolwaterdata["Date_measurement"] = pd.to_datetime(
            df_rioolwaterdata["Date_measurement"], format="%Y-%m-%d")



df_rioolwaterdata = pd.merge(df_rioolwaterdata, df_inwoners, how="outer", right_on="rwzi_code", left_on="RWZI_AWZI_code")
df_rioolwaterdata["product"] = df_rioolwaterdata["RNA_flow_per_100000"] * df_rioolwaterdata["inwoners"] / 100_000


df_rioolwaterdata = df_rioolwaterdata.groupby(df_rioolwaterdata["Date_measurement"]).sum()

df_rioolwaterdata["result"] = ((df_rioolwaterdata["product"]/df_rioolwaterdata["aandeel_maal_inwonders"] )/ 100_000_000_000 )*100_000


df_lcps = pd.read_csv(url_lcps, delimiter=',', low_memory=False)
df_lcps["date"] = pd.to_datetime(
            df_lcps["date"], format="%Y-%m-%d")

df_totaal = pd.merge(df_rioolwaterdata, df_lcps, how="outer", left_on="Date_measurement", right_on = "date")
df_totaal = df_totaal.fillna(0)
df_totaal = df_totaal.sort_values(by='date') 
df_totaal["result_sma"] =  df_totaal["result"].rolling(window = 7, center = False).mean()
df_totaal["Kliniek_Nieuwe_Opnames_COVID_Nederland_sma"] = df_totaal["Kliniek_Nieuwe_Opnames_COVID_Nederland"].rolling(window = 7, center = False).mean()
df_totaal["rioolwaarde_gedeeld_door_opname"] = df_totaal["Kliniek_Nieuwe_Opnames_COVID_Nederland_sma"] / df_totaal["result_sma"]


df_totaal, new_column = move_column(df_totaal, "Kliniek_Nieuwe_Opnames_COVID_Nederland_sma" , -5)

st.header (f"rioolwaardes vs {new_column}")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace( go.Scatter(x=df_totaal['date'],

                            y=df_totaal["result_sma"],
                            name ="rioolwaterdeeltjes",
                            #line=dict(width=2), opacity = 1, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                            line=dict(width=2,color='rgba(205, 61,62, 1)'),
                            mode='lines',
                           
                           ))
   
fig.add_trace(go.Scatter(
                    name="opnames",
                    
                    x=df_totaal["date"],
                    y=df_totaal[new_column],
                    #mode='lines',
                    line=dict(width=1,color='rgba(2, 61,62, 1)'),
                    
                    ) ,secondary_y=True)

st.plotly_chart(fig, use_container_width=True)
st.write(df_totaal)

fig2 = px.scatter(df_totaal, x="date", y="rioolwaarde_gedeeld_door_opname")
st.plotly_chart(fig2, use_container_width=True)
