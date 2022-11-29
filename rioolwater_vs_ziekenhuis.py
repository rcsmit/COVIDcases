import pandas as pd
# import numpy as np
import platform
import datetime as dt
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
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
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

def transform_data(df_inwoners, df_rioolwaterdata, df_riool_rivm, df_lcps, window, centersmooth,what_to_show):

    df_inwoners = df_inwoners[df_inwoners["regio_type"] == "VR"]
    df_inwoners = df_inwoners.groupby(df_inwoners["rwzi_code"]).mean().reset_index()
    
    df_inwoners["aandeel_maal_inwoners"]= df_inwoners["aandeel"] * df_inwoners["inwoners"]/100
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
 
    # df_riool_rivm["date_rivm"] = pd.to_datetime(df_riool_rivm["date_unix"], unit='s' )
    # df_riool_rivm["date_rivm"] =  df_riool_rivm["date_rivm"].dt.strftime('%Y-%m-%d')
    
    df_riool_rivm["date_rivm"] =  pd.to_datetime( df_riool_rivm["date_rivm"] , format="%Y-%m-%d")

    # name_="C:\\Users\\rcxsm\\Documents\\riool_rivm.csv"
    # compression_opts = dict(method=None, archive_name=name_)
    # df_riool_rivm.to_csv(name_, index=False, compression=compression_opts)
  
    df_lcps["date"] = pd.to_datetime(df_lcps["date"], format="%Y-%m-%d")
    
    df_totaal = pd.merge(df_rioolwaterdata, df_lcps, how="inner", left_on="Date_measurement", right_on = "date")
    df_totaal = pd.merge(df_totaal, df_rioolwaterdata_simpel, how="inner", left_on="date", right_on="Date_measurement")
    df_totaal = pd.merge(df_totaal, df_riool_rivm, how="inner", left_on = "date", right_on="date_rivm")
    df_totaal["RNA_flow_per_100000_simpel"] = df_totaal["RNA_flow_per_100000_simpel"] / 100_000_000_000
   
    df_totaal["RNA_flow_per_100000_simpel_gedeeld_door_aantal"] = df_totaal["RNA_flow_per_100000_simpel"] / df_totaal["aantal_x"]
    df_totaal = df_totaal.fillna(0)
    df_totaal = df_totaal.sort_values(by='date') 
    
    for t in ["result", what_to_show,"RNA_flow_per_100000_simpel", "RNA_flow_per_100000_simpel_gedeeld_door_aantal","value_rivm_official"]:
        make_sma(df_totaal, t, window, centersmooth)

    what_to_show_sma = what_to_show +"_sma"
    df_totaal["rioolwaarde_gedeeld_door_what_to_show"] = df_totaal[what_to_show_sma] / df_totaal["result_sma"]
    return df_totaal

def get_data():
    if platform.processor() != "":
        url_inwoners = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\inwoners_rzwi.csv"
        url_rioolwaterdata = r"C:\Users\rcxsm\Downloads\COVID-19_rioolwaterdata.csv"
        url_lcps = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"
        url_riool_rivm = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rioolwaardes_official_rivm.csv"
    
    else:
        url_inwoners = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/inwoners_rzwi.csv"
        url_rioolwaterdata= "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
    #url_lcps = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/lcps_by_day.csv"
        url_lcps = "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv"
        url_riool_rivm = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwaardes_official_rivm.csv"
    df_inwoners =  pd.read_csv(url_inwoners, delimiter=';', low_memory=False)
    df_rioolwaterdata = pd.read_csv(url_rioolwaterdata, delimiter=';', low_memory=False)
    df_lcps = pd.read_csv(url_lcps, delimiter=',', low_memory=False)
    df_riool_rivm = pd.read_csv(url_riool_rivm, delimiter=',', low_memory=False)
    
    return df_inwoners,df_rioolwaterdata,df_lcps,df_riool_rivm

def make_sma(df_totaal, column, window, center):
    column_sma = column+"_sma"
    df_totaal[column_sma] =  df_totaal[column].rolling(window = window, center = center).mean()
    return df_totaal, column_sma

def find_lag_time(df_, what_happens_first, what_happens_second, r1, r2):
    df = df_.copy(deep=True)
    b = what_happens_first
    a = what_happens_second  # shape (266,1)
    x = []
    y = []
    y_sma =[]

    max = 0
    max_sma = 0
    n_max, n_max_sma = 0,0
    df, b_sma = make_sma(df, b, 7, True )
    df, a_sma = make_sma(df, a, 7, True )

  
    df, nx = move_column(df, a, 0) #strange way to prevent error
    df, nx_sma = move_column(df, a_sma, 0) #strange way to prevent error

    max_column = None
    for n in range(r1, (r2 + 1)):

        df, m = move_column(df, b, n) #(shape (266,)
        c = round(df[m].corr(df[nx]), 3)
        if c<0 : c=c*-1
        if c > max:
            max = c
            n_max = n

        x.append(n)
        y.append(c)

        df, m_sma = move_column(df, b_sma, n) #(shape (266,)
        c_sma = round(df[m_sma].corr(df[nx_sma]), 3)
        if c_sma <0:c_sma = c_sma*-1
        if c_sma > max_sma:
            max_sma = c_sma
            n_max_sma = n
        y_sma.append(c_sma)


    title = f"Correlation between : {a} - {b} with moved days\n"#({FROM} - {UNTIL})"

    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)
        plt.xlabel("shift in days")
        plt.plot(x, y, label = "Values")
        plt.plot(x, y_sma, label = "Smoothed")
        #plt.axvline(x=0, color="yellow", alpha=0.6, linestyle="--")
        # Add a grid
        plt.legend()
        plt.grid(alpha=0.2, linestyle="--")
        plt.title(title, fontsize=10)
        st.pyplot(fig1x)
    # plt.show()
    st.write (f"Values: heightest correlation at  {n_max} days - correlation = {max}")
    st.write (f"Smoothed: heightest correlation at {n_max_sma} days - correlation = {max_sma}")

    # graph_daily(df, [a], [b], "SMA", "line", showday)
    # graph_daily(df, [a], [max_column], "SMA", "line", showday)
    # if the optimum is negative, the second one is that x days later


def make_graphs(df_totaal, new_column, which_riooldeeltjes):
    title_1 = (f"{new_column} en Gemiddeld aantal virusdeeltjes [(per 100.000 inwoners)  x 100 miljard] door de tijd heen")
    title_1b= (f"rioolwaardes vs {new_column}")
    st.write(title_1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace( go.Scatter(x=df_totaal['date'],
                                y=df_totaal[which_riooldeeltjes],
                                name ="rioolwaterdeeltjes",
                                #line=dict(width=2), opacity = 1, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                                line=dict(width=2,color='rgba(205, 61,62, 1)'),
                                mode='lines',
                            ))
    # fig.add_trace( go.Scatter(x=df_totaal['date'],
    #                             y=df_totaal["value_rivm_official_sma"],
    #                             name ="rioolwaterdeeltjes official",
    #                             #line=dict(width=2), opacity = 1, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
    #                             line=dict(width=2,color='rgba(205, 261,62, 1)'),
    #                             mode='lines',
    #                         ))
    
    fig.add_trace(go.Scatter(
                        name="result",
                        x=df_totaal["date"],
                        y=df_totaal[new_column],
                        #mode='lines',
                        line=dict(width=1,color='rgba(2, 61,62, 1)'),
                        ) ,secondary_y=True)
    make_annotations(fig)
    st.write(f"Result = {new_column}")
    st.plotly_chart(fig, use_container_width=True)
    #find_lag_time(df_totaal,which_riooldeeltjes, new_column,-31,31)
    make_scatter(df_totaal, x=which_riooldeeltjes, y=new_column, title=title_1b )
    make_scatter(df_totaal, x="date", y=which_riooldeeltjes, title= "aantal deeltjes door de tijd heen")
    make_scatter(df_totaal, x="date", y="rioolwaarde_gedeeld_door_what_to_show", title=(f"{new_column} per rioolwaarde"))
    make_scatter(df_totaal, x=which_riooldeeltjes, y="RNA_flow_per_100000_simpel_sma", title="gewogen waarde vs opgetelde waardes")
    make_scatter(df_totaal, x=which_riooldeeltjes, y="RNA_flow_per_100000_simpel_gedeeld_door_aantal_sma", title = "gewogen waarde vs gemiddelde waarde per meetstation")
    make_scatter(df_totaal, x="date", y="aantal_x", title= "aantal meetstations door de tijd heen")
   
    make_scatter(df_totaal, x=which_riooldeeltjes, y="value_rivm_official_sma", title="gewogen waarde vs officiele waarde")
    
def make_scatter(df_totaal, x,y, title):
    fig1 = px.scatter(df_totaal, x=x, y=y, title=title)
    # fig1b.add_trace(go.Scatter(x=[500], 
                          
    #                      mode='lines', 
    #                      line=dict(color='green', width=2, dash='dash'),
    #                      ))
    if x == "date":
        make_annotations(fig1)
        
   
                
    st.plotly_chart(fig1, use_container_width=True)

def make_annotations(fig1):
    fig1.add_vrect(x0="2021-1-01", x1="2021-12-31", 
                annotation_text="2021", annotation_position="bottom left",
                fillcolor="pink", opacity=0.25, line_width=0)

    fig1.add_vrect(x0='2021-01-6', x1="2021-01-7",
                annotation_text="Start Vaccinatie", annotation_position="top left",
                fillcolor="green", opacity=0.25)
     
    fig1.add_vrect(x0='2021-11-18', x1="2021-11-19",
                annotation_text="Start Booster", annotation_position="top left",
                fillcolor="green", opacity=0.25)
    fig1.add_vrect(x0='2022-01-01', x1="2022-01-02",
                annotation_text="Opmars omnicron", annotation_position="bottom left",
                fillcolor="green", opacity=0.25)
    fig1.add_vrect(x0='2022-09-19', x1="2022-09-20",
                annotation_text="Start Herhaalprik", annotation_position="top left",
                fillcolor="green", opacity=0.25)
        
    

def interface():
    mzelst =  ["IC_Nieuwe_Opnames_COVID_Nederland","Kliniek_Nieuwe_Opnames_COVID_Nederland","cases","hospitalization","deaths","positivetests","hospital_intake_rivm","Hospital_Intake_Proven","Hospital_Intake_Suspected",
        "IC_Intake_Proven","IC_Intake_Suspected","IC_Current","ICs_Used","IC_Cumulative","Hospital_Currently","IC_Deaths_Cumulative",
        "IC_Discharge_Cumulative","IC_Discharge_InHospital","Hospital_Cumulative","Hospital_Intake","IC_Intake","Hosp_Intake_Suspec_Cumul",
        "IC_Intake_Suspected_Cumul","IC_Intake_Proven_Cumsum","new.infection","corrections.cases","net.infection","new.hospitals",
        "corrections.hospitals","net.hospitals","new.deaths","corrections.deaths","net.deaths","positive_7daverage","positive_14d",
        "growth_infections","infections.today.nursery","infections.total.nursery","deaths.today.nursery","deaths.total.nursery",
        "mutations.locations.nursery","total.current.locations.nursery","values.tested_total","values.infected","values.infected_percentage",
        "pos.rate.3d.avg","pos.rate.7d.avg","IC_Bedden_COVID_Nederland","IC_Bedden_COVID_Internationaal","IC_Bedden_Non_COVID_Nederland",
        "Kliniek_Bedden_Nederland","Totaal_Bezetting",
        "IC_Opnames_7d","Kliniek_Opnames_7d","Totaal_opnames","Totaal_opnames_7d","Totaal_IC","IC_opnames_14d","Kliniek_opnames_14d",
        "OMT_Check_IC","OMT_Check_Kliniek","Kliniek_Bedden_7d","IC_Bedden_7d","Totaal_Bedden_7d","IC_Bedden_14d","Kliniek_Bedden_14d",
        "Totaal_Bedden_14d","OMT_Check_IC_Bezetting","OMT_Check_Kliniek_Bezetting","OMT_Check_Totaal_Bezetting"]
    what_to_show = st.sidebar.selectbox(
            "What to show", mzelst, index=1
        )
    which_riooldeeltjes = st.sidebar.selectbox(
            "Welke riooldeeltjes", ["result_sma", "value_rivm_official_sma"], index=1
        )

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
    
    return FROM,UNTIL,days_move_columns,window,centersmooth,what_to_show, which_riooldeeltjes

def main():
    FROM, UNTIL, days_move_columns, window, centersmooth,what_to_show,which_riooldeeltjes = interface()
    df_inwoners, df_rioolwaterdata, df_lcps,df_riool_rivm = get_data()

    df_totaal = transform_data(df_inwoners, df_rioolwaterdata, df_riool_rivm, df_lcps, window, centersmooth,what_to_show)
   
    df_totaal = select_period_oud(df_totaal, "date", FROM, UNTIL)
    what_to_show_sma = what_to_show +"_sma"
    df_totaal, new_column = move_column(df_totaal, what_to_show_sma , days_move_columns)
    
    make_graphs(df_totaal, new_column, which_riooldeeltjes)

if __name__ == "__main__":
    main()