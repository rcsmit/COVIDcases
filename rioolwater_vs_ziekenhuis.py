import pandas as pd
# import numpy as np
import platform
import datetime as dt
from datetime import datetime, timedelta
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
# https://coronadashboard.rijksoverheid.nl/landelijk/rioolwater
# 853 x 100 miljard = 853 * 10E11 = 8.53 *10E13 per 100.000 inwoners



def select_period_oud(df, field, show_from, show_until):
    """Shows two inputfields (from/until and Select a period in a df (helpers.py).
    Args:
        df (df): dataframe
        field (string): Field containing the date
    Returns:
        df: filtered dataframe
    """

    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df




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

def transform_data(df_inwoners, df_rioolwaterdata, df_lcps, window, centersmooth):
    

    df_inwoners = df_inwoners[df_inwoners["regio_type"] == "VR"]
    df_inwoners = df_inwoners.groupby(df_inwoners["rwzi_code"]).mean().reset_index()
    
    df_inwoners["aandeel_maal_inwoners"]= df_inwoners["aandeel"] * df_inwoners["inwoners"]/100
    print ("TOTAAL INWONERS")
    print (df_inwoners["aandeel_maal_inwoners"].sum())
    df_rioolwaterdata["Date_measurement"] = pd.to_datetime(
            df_rioolwaterdata["Date_measurement"], format="%Y-%m-%d")
    df_rioolwaterdata["aantal"] = 1
    # df_rioolwaterdata_simpel is eenvoudigweg groeperen en totaal berekenen op datum. Houdt geen rekening met 
    # het aantal inwoners per locatie

    df_rioolwaterdata_simpel = df_rioolwaterdata.groupby(df_rioolwaterdata["Date_measurement"]).sum().reset_index()
    df_rioolwaterdata_simpel.rename(columns = {'RNA_flow_per_100000':'RNA_flow_per_100000_simpel'}, inplace = True)

    df_rioolwaterdata = pd.merge(df_rioolwaterdata, df_inwoners, how="inner", right_on="rwzi_code", left_on="RWZI_AWZI_code")
    # https://www.rivm.nl/documenten/berekening-cijfers-rioolwatermetingen-covid-19
    #df_rioolwaterdata["product"] = df_rioolwaterdata["RNA_flow_per_100000"] * df_rioolwaterdata["inwoners"] / 100_000
    df_rioolwaterdata["product"] = (df_rioolwaterdata["RNA_flow_per_100000"] * df_rioolwaterdata["aandeel_maal_inwoners"] / 100_000 ) / 100_000_000_000


    df_rioolwaterdata = df_rioolwaterdata.groupby(df_rioolwaterdata["Date_measurement"]).sum()

    df_rioolwaterdata["result"] = ((df_rioolwaterdata["product"]/df_rioolwaterdata["inwoners"]  ) ) *100_000
    print (df_rioolwaterdata)

    
    df_lcps["date"] = pd.to_datetime(df_lcps["date"], format="%Y-%m-%d")

    df_totaal = pd.merge(df_rioolwaterdata, df_lcps, how="inner", left_on="Date_measurement", right_on = "date")
    df_totaal = pd.merge(df_totaal, df_rioolwaterdata_simpel, how="inner", left_on="date", right_on="Date_measurement")
    df_totaal["RNA_flow_per_100000_simpel"] = df_totaal["RNA_flow_per_100000_simpel"] / 100_000_000_000
    print (df_totaal.dtypes)
    df_totaal["RNA_flow_per_100000_simpel_gedeeld_door_aantal"] = df_totaal["RNA_flow_per_100000_simpel"] / df_totaal["aantal_x"]
    df_totaal = df_totaal.fillna(0)
    df_totaal = df_totaal.sort_values(by='date') 
    for t in ["result","Kliniek_Nieuwe_Opnames_COVID_Nederland","RNA_flow_per_100000_simpel", "RNA_flow_per_100000_simpel_gedeeld_door_aantal"]:
        make_sma(df_totaal, t, window, centersmooth)

    
    df_totaal["rioolwaarde_gedeeld_door_opname"] = df_totaal["Kliniek_Nieuwe_Opnames_COVID_Nederland_sma"] / df_totaal["result_sma"]
    return df_totaal

def get_data():
    if platform.processor() != "":
        url_inwoners = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\inwoners_rzwi.csv"
        url_rioolwaterdata = r"C:\Users\rcxsm\Downloads\COVID-19_rioolwaterdata.csv"
        url_lcps = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"
    else:
        url_inwoners = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/inwoners_rzwi.csv"
        url_rioolwaterdata= "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
    #url_lcps = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/lcps_by_day.csv"
        url_lcps = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"

    df_inwoners =  pd.read_csv(url_inwoners, delimiter=';', low_memory=False)
    df_rioolwaterdata = pd.read_csv(url_rioolwaterdata, delimiter=';', low_memory=False)
    df_lcps = pd.read_csv(url_lcps, delimiter=',', low_memory=False)
    return df_inwoners,df_rioolwaterdata,df_lcps

def make_sma(df_totaal, column, window, center):
    column_sma = column+"_sma"
    df_totaal[column_sma] =  df_totaal[column].rolling(window = window, center = center).mean()


def make_graphs(df_totaal, new_column):
    title_1 = ("Opnames en Gemiddeld aantal virusdeeltjes [(per 100.000 inwoners)  x 100 miljard] door de tijd heen")
    title_1b= (f"rioolwaardes vs {new_column}")
    st.write(title_1)
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

    fig1b = px.scatter(df_totaal, x="result_sma", y=new_column, title=title_1b )
    st.plotly_chart(fig1b, use_container_width=True)

    fig1c = px.scatter(df_totaal, x="date", y="result_sma", title= "aantal deeltjes door de tijd heen")

    st.plotly_chart(fig1c, use_container_width=True)


    fig2 = px.scatter(df_totaal, x="date", y="rioolwaarde_gedeeld_door_opname", title= "opnames per rioolwaarde")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df_totaal, x="result_sma", y="RNA_flow_per_100000_simpel_sma", title="gewogen waarde vs opgetelde waardes")
    st.plotly_chart(fig3, use_container_width=True)

    fig3b = px.scatter(df_totaal, x="result_sma", y="RNA_flow_per_100000_simpel_gedeeld_door_aantal_sma", title = "gewogen waarde vs gemiddelde waarde per meetstation")

    st.plotly_chart(fig3b, use_container_width=True)


    fig4 = px.scatter(df_totaal, x="date", y="aantal_x", title= "aantal meetstations door de tijd heen")

    st.plotly_chart(fig4, use_container_width=True)

def interface():
    start_ = "2021-01-01"
    end = "2029-01-01"
    
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", end)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()
    days_move_columns = st.sidebar.slider("Move curves at right axis (days)", -31, 31, -5)

    window = st.sidebar.slider("window for SMA (days)", 1, 31, 7)
    centersmooth =  st.sidebar.selectbox(
        "Smooth in center", [True, False], index=0
    )
    
    return FROM,UNTIL,days_move_columns,window,centersmooth

def main():
    
    FROM, UNTIL, days_move_columns, window, centersmooth = interface()
    df_inwoners, df_rioolwaterdata, df_lcps = get_data()

    df_totaal = transform_data(df_inwoners, df_rioolwaterdata, df_lcps, window, centersmooth)
    df_totaal = select_period_oud(df_totaal, "date", FROM, UNTIL)

    df_totaal, new_column = move_column(df_totaal, "Kliniek_Nieuwe_Opnames_COVID_Nederland_sma" , days_move_columns)
    
    make_graphs(df_totaal, new_column)

if __name__ == "__main__":
    main()