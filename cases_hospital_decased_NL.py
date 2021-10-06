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

from scipy.stats import fisher_exact

#@st.cache(ttl=60 * 60 * 24)
def read():
    url="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/landelijk_leeftijd_week_vanuit_casus_landelijk_20211006.csv"
    # url="C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\COVIDcases\\input\\landelijk_leeftijd_week_vanuit_casus_landelijk_20211006.csv"
    df= pd.read_csv(url, delimiter=',', error_bad_lines=False)
    df = df[df["Agegroup"] != "<50"]
    df = df[df["Agegroup"] != "0"]
    df["Hosp_per_reported"]= df["Hosp_per_reported"]*100
    df["Deceased_per_reported"]= df["Deceased_per_reported"]*100
    df_agg = df.groupby( "Date_statistics" ).sum().reset_index().copy(deep = True)
    return df, df_agg


def line_chart (df, what_to_show, aggregate):
    """Make a linechart from an unpivoted table, with different lines (agegroups)

    Args:
        df ([type]): [description]
        what_to_show ([type]): [description]
    """
    # fig = go.Figure()
    if aggregate == True:
        try:
            fig = px.line(df, x="Date_statistics", y=what_to_show, color='Agegroup')
        except:
            fig = px.line(df, x="Date_statistics", y=what_to_show)
    else:
        fig = px.line(df, x="Date_statistics", y=what_to_show)
    fig.update_layout(
        title=what_to_show,
        xaxis_title=" Date_statistics",
        yaxis_title=what_to_show + " (%)",
    )
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
    df, df_agg = read()


    st.header("Chance of hospitalization / death after having covid.")
    st.subheader("These graphs says nothing about vaccination effect, since the numbers aren't splitted up")

    line_chart (df, "Hosp_per_reported", True)
    line_chart (df, "Deceased_per_reported", True)
    line_chart (df_agg, "Hosp_per_reported", False)
    line_chart (df_agg, "Deceased_per_reported", False)

if __name__ == "__main__":
    caching.clear_cache()
    #st.set_page_config(layout="wide")
    main()