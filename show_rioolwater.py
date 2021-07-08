import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from textwrap import wrap
import matplotlib.cm as cm
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
from sklearn.metrics import r2_score
import streamlit as st
import urllib
import urllib.request
from pathlib import Path
from streamlit import caching
from inspect import currentframe, getframeinfo
from helpers import *

###################################################################

@st.cache(ttl=60 * 60 * 24)
def get_data():
    """Get the data
    In : -
    Out : df        : dataframe
         UPDATETIME : Date and time from the last update"""
    with st.spinner(f"GETTING ALL DATA ..."):
        url1 = "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
        df = pd.read_csv(url1, delimiter=";", low_memory=False)
        df["Date_measurement"] = pd.to_datetime(df["Date_measurement"], format="%Y-%m-%d")
        df.rename(columns={"Date_measurement": 'date'}, inplace=True)
        UPDATETIME = datetime.now()
        return df, UPDATETIME

def graph_day(df, what_to_show_l, title):
    """  _ _ _ """
    #st.write(f"t = {t}")
    df_temp = pd.DataFrame(columns=["date"])
    if what_to_show_l is None:
        st.warning("Choose something")
        st.stop()

    if type(what_to_show_l) == list:
        what_to_show_l_ = what_to_show_l
    else:
        what_to_show_l_ = [what_to_show_l]
    aantal = len(what_to_show_l_)
    # SHOW A GRAPH IN TIME / DAY

    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)

        color_list = [
            "#02A6A8",
            "#4E9148",
            "#F05225",
        ]

        n = 0  # counter to walk through the colors-list

        for b in what_to_show_l_:
            df_temp = df

            df_temp[b].plot(
                label="_nolegend_",
                color=color_list[n],
                linestyle="--",
                alpha=0.9,
                linewidth=0.8,
                )
            n += 1

        plt.title(title, fontsize=10)

        a__ = (max(df_temp["date"].tolist())).date() - (
            min(df_temp["date"].tolist())
        ).date()
        freq = int(a__.days / 10)
        ax.xaxis.set_major_locator(MultipleLocator(freq))
        if what_to_show_l == ["reported_div_tested"]:
            ax.set_ylim(0,0.3)
        ax.set_xticks(df_temp["date"].index)
        ax.set_xticklabels(df_temp["date"].dt.date, fontsize=6, rotation=90)
        xticks = ax.xaxis.get_major_ticks()

        for i, tick in enumerate(xticks):
            if i % 10 != 0:
                tick.label1.set_visible(False)
        plt.xticks()

        # layout of the x-axis
        ax.xaxis.grid(True, which="major", alpha=0.4, linestyle="--")
        ax.yaxis.grid(True, which="major", alpha=0.4, linestyle="--")

        left, right = ax.get_xlim()
        ax.set_xlim(left, right)
        fontP = FontProperties()
        fontP.set_size("xx-small")

        plt.xlabel("date")
        # everything in legend
        # https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
        handles, labels = [], []
        for ax in fig1x.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        # plt.legend(handles,labels)
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
        plt.legend(handles, labels, bbox_to_anchor=(0, -0.5), loc="lower left", ncol=2)
        ax.text(
            1,
            1.1,
            "Created by Rene Smit — @rcsmit",
            transform=ax.transAxes,
            fontsize="xx-small",
            va="top",
            ha="right",
        )


        st.pyplot(fig1x)



def main():
    """  _ _ _ """

    df_getdata, UPDATETIME = get_data()
    df___ = df_getdata.copy(deep=False)

    countrylist =  df___['RWZI_AWZI_name'].drop_duplicates().sort_values().tolist()

    country_ = st.sidebar.selectbox("Welke plaats",countrylist, 0)
    df = df___.loc[df___['RWZI_AWZI_name'] ==country_].copy(deep=False)

    st.title("Interactive Corona Dashboard")
    # st.header("")
    st.subheader("Under construction - Please send feedback to @rcsmit")

    # DAILY STATISTICS ################

    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2020-09-07"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    if until_ == "2023-08-23":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
            caching.clear_cache()
            st.success("Cache is cleared, please reload to scrape new values")

    df = select_period(df, "date", FROM, UNTIL)
    title = f"Rioolwaardes in {country_}"

    graph_day(
        df,
        ["RNA_per_ml", "RNA_flow_per_100000"], title
    )


    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/blob/main/covid_dashboard_rcsmit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        'Restrictions by <a href="https://twitter.com/hk_nien" target="_blank">Han-Kwang Nienhuys</a> (MIT-license).</div>'
    )


    st.sidebar.markdown(tekst, unsafe_allow_html=True)
    now = UPDATETIME
    UPDATETIME_ = now.strftime("%d/%m/%Y %H:%M:%S")
    st.write(f"\n\n\nData last updated : {str(UPDATETIME_)}")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.image(
        "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png"
    )

    st.markdown(
        '<a href="https://www.buymeacoffee.com/rcsmit" target="_blank">If you are happy with this dashboard, you can buy me a coffee</a>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<br><br><a href="https://www.linkedin.com/in/rcsmit" target="_blank">Contact me for custom dashboards and infographics</a>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":

    main()