#import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats
#from scipy.stats import weibull_min
import pandas as pd
from statistics import mean
# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock
import streamlit as st
import random
from itertools import cycle
#from streamlit import caching
import time
# partly derived from https://stackoverflow.com/a/37036082/4173718
import pandas as pd
import numpy as np
#import openpyxl
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta
from sklearn.metrics import r2_score

import plotly.express as px
import plotly.graph_objects as go



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



#@st.cache(ttl=60 * 60 * 24)
def read():
    url="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/landelijk_leeftijd_week_vanuit_casus_landelijk_20211006.csv"
    #url="C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\COVIDcases\\input\\landelijk_leeftijd_week_vanuit_casus_landelijk_20211006.csv"

    #url_pop="C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\COVIDcases\\input\\pop_size_age_NL.csv"
    url_pop = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/pop_size_age_NL.csv"
    df= pd.read_csv(url, delimiter=',')
    df_pop = pd.read_csv(url_pop, delimiter=',')


    df = df[df["Agegroup"] != "<50"]
    df = df[df["Agegroup"] != "0"]
    df  = pd.merge(
                df, df_pop, how="outer", on="Agegroup"
            )
    df["Hosp_per_reported"]= df["Hosp_per_reported"]*100
    df["Deceased_per_reported"]= df["Deceased_per_reported"]*100

    df["Cases_per_100k"]= df["cases"]/df["pop_size"]*100000
    df["Hosp_per_100k"]= df["Hospital_admission"]/df["pop_size"]*100000
    df["Deceased_per_100k"]= df["Deceased"]/df["pop_size"]*100000


    df_week = df.groupby( "Date_statistics" ).sum().reset_index().copy(deep = True)
    df_week["Hosp_per_reported_week"] = df_week["Hospital_admission"]/df_week["cases"]*100
    df_week["Deceased_per_reported_week"] = df_week["Deceased"]/df_week["cases"]*100

    return df, df_week

def prepare_data():
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
    save_df(df_week, "landelijk_leeftijd_week_vanuit_casus_landelijk_20211006")

def line_chart_enkel (df, what_to_show, aggregate,y_ax_label_supp, log_y):
    """Make a linechart from an unpivoted table, with different lines (agegroups)

    Args:
        df ([type]): [description]
        what_to_show ([type]): [description]
    """
    # fig = go.Figure()
    if aggregate == True:
        fig = px.line(df, x="Date_statistics", y=what_to_show, color='Agegroup', log_y=log_y, hover_data=["Hospital_admission", "Deceased", "cases"])
    else:
        fig = px.line(df, x="Date_statistics", y=what_to_show,log_y=log_y, hover_data=["Hospital_admission", "Deceased", "cases"])

    fig.update_layout(
        title=what_to_show,
        xaxis_title="Date_statistics (week)",
        yaxis_title=what_to_show + " ("+ y_ax_label_supp+ ")",
    )
    st.plotly_chart(fig, use_container_width=True)

def make_pivot(df, valuefield):
    df_pivot = (
    pd.pivot_table(
        df,
        values=valuefield,
        index=["Date_statistics"],
        columns=["Agegroup"],
        aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    return df_pivot

def line_chart  (df, what_to_show, aggregate,y_ax_label_supp, sma_wdw,log_y):
    """Makes a linechart from a pivoted table, each column in a differnt line. Smooths the lines too.

    Args:
        df ([type]): [description]
        title ([type]): [description]
    """
    df = make_pivot(df,what_to_show)
    fig = go.Figure()

    columns = df.columns.tolist()
    columnlist = columns[1:]
    # st.write(columnlist)
    for col in columnlist:
        col_sma = col +"_sma"
        df[col_sma] =  df[col].rolling(window = sma_wdw, center = False).mean()
        fig.add_trace(go.Scatter(x=df["Date_statistics"], y= df[col_sma], mode='lines', name=col ))
    if log_y == True:
        fig.update_yaxes(type="log")

    fig.update_layout(
        title=dict(
                text=what_to_show+ " (SMA "+ str(sma_wdw)+")",
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )),


        xaxis_title="Einddag vd week",
        yaxis_title=what_to_show + " ("+ y_ax_label_supp+ ")"    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander (f"dataframe pivottable {what_to_show}"):
        df_temp = df.astype(str).copy(deep = True)
        st.write (df_temp)

def make_scatterplot(df_temp, what_to_show_l, what_to_show_r,  show_cat, categoryfield, hover_name, hover_data):
    """Makes a scatterplot with trendline and statistics

    Args:
        df_temp ([type]): [description]
        what_to_show_l ([type]): [description]
        what_to_show_r ([type]): [description]
        show_cat ([type]): [description]
        categoryfield ([type]): [description]
    """
    # with _lock:
    fig1xy,ax = plt.subplots()
    try:

        x_ = np.array(df_temp[what_to_show_l])
        y_ = np.array(df_temp[what_to_show_r])
        #obtain m (slope) and b(intercept) of linear regression line
        idx = np.isfinite(x_) & np.isfinite(y_)
        m, b = np.polyfit(x_[idx], y_[idx], 1)
        model = np.polyfit(x_[idx], y_[idx], 1)

        predict = np.poly1d(model)
        r2 = r2_score  (y_[idx], predict(x_[idx]))
    except:
        m,b,model,predict,r2 =None,None,None,None,None

    try:

        fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data, trendline="ols", trendline_scope = 'overall', trendline_color_override = 'black')
    except:
        # avoid exog contains inf or nans
        fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name, hover_data=hover_data)

    #add linear regression line to scatterplot


    correlation_sp = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
    correlation_p = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='pearson'), 3)


    title_scatter = (f"{what_to_show_l} -  {what_to_show_r}<br>Correlation spearman = {correlation_sp} - Correlation pearson = {correlation_p}<br>y = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

    fig1xy.update_layout(
        title=dict(
            text=title_scatter,
            x=0.5,
            y=0.95,
            font=dict(
                family="Arial",
                size=14,
                color='#000000'
            )
        ),
        xaxis_title=what_to_show_l,
        yaxis_title=what_to_show_r,
        font=dict(
            family="Courier New, Monospace",
            size=12,
            color='#000000'
        )
    )

    ax.text(
        1,
        1.3,
        "Created by Rene Smit — @rcsmit",
        transform=ax.transAxes,
        fontsize="xx-small",
        va="top",
        ha="right",
    )


    st.plotly_chart(fig1xy, use_container_width=True)

def main():
    # prepare_data()

    df, df_agg = read()


    st.header("Chance of hospitalization / death after having covid - per week")
    st.subheader("These graphs says nothing about vaccination effect, since the numbers aren't splitted up")
    st.write ("You can zoom in the graph by using your mouse, click in one corner and drag to the opposite corner ")
    sma_wdw= st.sidebar.slider("Smoothing window", 1, 9, 3)
    log_y  = st.sidebar.selectbox("Y axis on log-scale", [True, False], index=1)

    st.subheader("Per case")
    line_chart (df, "Hosp_per_reported", True, "%", sma_wdw,log_y)
    line_chart (df, "Deceased_per_reported", True, "%", sma_wdw,log_y)
    st.subheader("Per 100k (agegroup)")
    line_chart (df, "Cases_per_100k", True, "#", sma_wdw,log_y)

    line_chart (df, "Hosp_per_100k", True, "#", sma_wdw,log_y)
    line_chart (df, "Deceased_per_100k", True, "#", sma_wdw,log_y)
    st.subheader("Total per week ")
    line_chart_enkel (df_agg, "Hosp_per_reported_week", False, "%", log_y)
    line_chart_enkel (df_agg, "Deceased_per_reported_week", False, "%", log_y)
    st.subheader("Absolute numbers")
    line_chart (df, "cases", True, "#", sma_wdw,log_y)
    line_chart (df, "Hospital_admission", True, "#", sma_wdw,log_y)
    line_chart (df, "Deceased", True, "#, sma_wdw,log_y")

if __name__ == "__main__":
    caching.clear_cache()
    #st.set_page_config(layout="wide")

    main()