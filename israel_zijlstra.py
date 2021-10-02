#import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats
#from scipy.stats import weibull_min
import pandas as pd
from statistics import mean
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import streamlit as st
import random
from itertools import cycle
from streamlit import caching
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

from plotly.subplots import make_subplots


# na 21/8 zelfde waardes voor de vaccinaties voor 90+ aangehoude
#@st.cache(ttl=60 * 60 * 24)
def read():
    sheet_id = "104fiQWDNmLP73CBEo5Lu-fXeTOYiTx_7PZb0N-b3joE"
    sheet_name = "mastersheet"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    url_data = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/israel_zijlstra/israel_zijlstra_mastersheet.csv"
    url_populatie = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/israel_zijlstra/populatie_grootte.csv"
    #url = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\in\\schoonmaaktijden.csv",
    df_data = pd.read_csv(url_data, delimiter=',', error_bad_lines=False)
    df_populatie = pd.read_csv(url_populatie, delimiter=',')
    df = pd.merge(df_data, df_populatie, how="outer", on="Age_group" )
    #df = df[:-1]  #remove last row which appears to be a Nan

    df["einddag_week"] = pd.to_datetime(df["einddag_week"], format="%d-%m-%Y")
    # for col in df.select_dtypes(include=['object']).columns:
    #     try:
    #         df[col] = pd.to_numeric(df[col])
    #         df[col] = df[col].fillna(0)
    #     except:
    #         pass
    return df


def line_chart (df, what_to_show):
    """Make a linechart from an unpivoted table, with different lines (agegroups)

    Args:
        df ([type]): [description]
        what_to_show ([type]): [description]
    """
    # fig = go.Figure()
    fig = px.line(df, x="einddag_week", y=what_to_show, color='Age_group')


    fig.update_layout(
        title=what_to_show,
        xaxis_title="Einddag vd week",
        yaxis_title="VE",
    )
    st.plotly_chart(fig)

def line_chart_pivot (df, title):
    """Makes a linechart from a pivoted table, each column in a differnt line. Smooths the lines too.

    Args:
        df ([type]): [description]
        title ([type]): [description]
    """
    with st.expander (f"dataframe pivottable {title}"):
        df_temp = df.astype(str).copy(deep = True)
        st.write (df_temp)

    fig = go.Figure()

    columns = df.columns.tolist()
    columnlist = columns[1:]
    # st.write(columnlist)
    for col in columnlist:
        col_sma = col +"_sma"

        df[col_sma] =  df[col].rolling(window = 3, center = False).mean()
        fig.add_trace(go.Scatter(x=df["einddag_week"], y= df[col_sma], mode='lines', name=col ))

    fig.update_layout(
        title=dict(
                text=title+ " (SMA 3)",
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )),


        xaxis_title="Einddag vd week",
        yaxis_title="VE"    )
    st.plotly_chart(fig)


def make_scatterplot(df_temp, what_to_show_l, what_to_show_r,  show_cat, categoryfield, hover_name, hover_data):
    """Makes a scatterplot with trendline and statistics

    Args:
        df_temp ([type]): [description]
        what_to_show_l ([type]): [description]
        what_to_show_r ([type]): [description]
        show_cat ([type]): [description]
        categoryfield ([type]): [description]
    """
    with _lock:
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


        st.plotly_chart(fig1xy)

def main():
    df = read()
    df = df.fillna(0)

    df["unvaxxed_new"] = df["populatie_grootte"] -  df["sec_cumm"]
    df["unboostered_new"] = df["populatie_grootte"] -  df["third_cumm"]
    df["perc_sec_dose"] = round((df["sec_cumm"] / df["populatie_grootte"])*100,1)
    df["perc_boostered"] = round((df["third_cumm"] / df["populatie_grootte"])*100,1)
    df["ziek_V_2_per_100k"] = df["positive_above_20_days_after_2nd_dose"] /  df["sec_cumm"] *100_000
    df["ziek_V_3_per_100k"] = df["positive_above_20_days_after_3rd_dose"] /  df["third_cumm"] *100_000
    df["ziek_N_per_100k"] = df["Sum_positive_without_vaccination"] / df["unvaxxed_new"] *100_000
    df["VE_2_N"] = (1 - ( df["ziek_V_2_per_100k"]/ df["ziek_N_per_100k"]))*100
    df["VE_3_N"] = (1 - ( df["ziek_V_3_per_100k"]/ df["ziek_N_per_100k"]))*100
    df["VE_3_N_2_N"]= (1 - ( df["ziek_V_3_per_100k"]/ df["ziek_V_2_per_100k"]))*100

    # after second dose
    # https://timeseriesreasoning.com/contents/estimation-of-vaccine-efficacy-using-logistic-regression/

    df["p_inf_vacc"] = df["positive_above_20_days_after_2nd_dose"] /  df["sec_cumm"]
    df["p_inf_non_vacc"] = df["Sum_positive_without_vaccination"] / df["unvaxxed_new"]

    df["odds_ratio"] = (  df["p_inf_vacc"]/(1-  df["p_inf_vacc"])) /  (   df["p_inf_non_vacc"] / (1-   df["p_inf_non_vacc"]))

    df["IRR"] =  df["odds_ratio"] / ((1-df["p_inf_non_vacc"]) + (df["p_inf_non_vacc"] *  df["odds_ratio"] ))
    st.write(df)


    df_pivot_VE_2_N = (
        pd.pivot_table(
            df,
            values="VE_2_N",
            index=["einddag_week"],
            columns=["Age_group"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )
    df_pivot_VE_2_N_SMA = pd.DataFrame()
    df_pivot_VE_2_N_SMA["einddag_week"] = df_pivot_VE_2_N["einddag_week"]

    columns =  df_pivot_VE_2_N .columns.tolist()
    columnlist = columns[1:]
    # st.write(columnlist)


    df_pivot_VE_3_N = (
        pd.pivot_table(
            df,
            values="VE_3_N",
            index=["einddag_week"],
            columns=["Age_group"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    df_pivot_VE_3_N_2_N = (
        pd.pivot_table(
            df,
            values="VE_3_N_2_N",
            index=["einddag_week"],
            columns=["Age_group"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )


    line_chart (df, "VE_2_N")
    line_chart (df, "perc_sec_dose")
    line_chart (df, "perc_boostered")
    line_chart_pivot ( df_pivot_VE_2_N, "VE (2 vaccins / N)")
    df_boostered = df[df["perc_boostered"] >0 ]
    make_scatterplot(df_boostered, "perc_boostered","VE_2_N",True, "Age_group", None, ["Age_group", "einddag_week"])
    make_scatterplot(df, "odds_ratio","VE_2_N",True, "Age_group", None, ["Age_group", "einddag_week"])
    make_scatterplot(df, "odds_ratio","IRR",True, "Age_group", None, ["Age_group", "einddag_week"])


    line_chart_pivot ( df_pivot_VE_3_N,  "VE (3 vaccins / N)")
    line_chart_pivot ( df_pivot_VE_3_N_2_N, "VE (3 vaccins / 2vaccins")

main()
