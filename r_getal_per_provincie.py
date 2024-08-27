import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from textwrap import wrap

# import seaborn as sn
from scipy import stats
import datetime as dt
from datetime import datetime, timedelta

import json
# from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as ticker
import math

# _lock = RendererAgg.lock

from sklearn.metrics import r2_score
import streamlit as st
import urllib
import urllib.request
from pathlib import Path
#from streamlit import caching
from inspect import currentframe, getframeinfo
from helpers import *
import platform

###################################################################


@st.cache(ttl=60 * 60 * 24)
def get_data():
    """Get the data from various sources
    In : -
    Out : df        : dataframe
         UPDATETIME : Date and time from the last update"""
    with st.spinner(f"GETTING ALL DATA ..."):
        init()
        # #CONFIG


        with st.spinner(f"Downloading...(it will take some time!)"):
            if platform.processor() != "":
                url1 = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
                url1= "https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv"

            else:
                url1= "https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv"

            df = pd.read_csv(url1, delimiter=";", low_memory=False)

        df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")

        # the tool is build around "date"
        df = df.rename(columns={"Date_statistics": "date"})

        UPDATETIME = datetime.now()



        df.rename(
            columns={
                "Date_file": "count",
            },
            inplace=True,


        )
        df["count"] = 1
        df_pivot = (
            pd.pivot_table(
                df,
                values="count",
                index=["date"],
                columns=["Province"],
                aggfunc=np.sum,
            )
            .reset_index()
            .copy(deep=False)
        )
        lijst = [ "Groningen","Fryslân",
           "Drenthe",   "Overijssel",
           "Gelderland",  "Flevoland",
           "Utrecht",  "Noord-Holland",
           "Zuid-Holland",   "Noord-Brabant",
           "Limburg",  "Zeeland", "Total"]

        df_pivot = df_pivot.replace({np.nan: 0})
        df_pivot.loc[:,'Total'] = df_pivot.sum(numeric_only=True, axis=1)

        provincielijst = lijst.copy()



        columnlist, t, WDW2, centersmooth,tg, d = lijst, "SMA", 7, True,4, 4


        df, smoothed_columns = smooth_columnlist(df_pivot, columnlist, t, WDW2, centersmooth)
        df, column_list_r_smoothened=  add_walking_r(df, smoothed_columns, "date", t,WDW2, tg, d)
        df, column_list_r_smoothened_moved =  move_column(df, column_list_r_smoothened, -8)
        lijst.extend(column_list_r_smoothened_moved)

        provincies = [["Groningen",586_813],
            ["Fryslân",651_459],
            ["Drenthe",494_760],
            ["Overijssel",1_166_478],
            ["Flevoland",428_264],
            ["Gelderland",2_096_620],
            ["Utrecht",1_361_093],
            ["Noord-Holland",2_887_906],
            ["Zuid-Holland",3_726_173],
            ["Zeeland", 385_379],
            ["Noord-Brabant",2_573_853],
            ["Limburg",1_115_895],
            ["Total", 17_474_693]]
        provincies_per_inw = []
        for p in provincies:
            column_name= p[0] + "_per_100kinw"
            df[column_name] = df[p[0]]/p[1]*100000
            lijst.append(column_name)
            provincies_per_inw.append(column_name)

        return df, UPDATETIME, lijst, provincielijst,column_list_r_smoothened_moved,provincies_per_inw



def move_column(df, column_, days):
    """  _ _ _ """
    column_ = column_ if type(column_) == list else [column_]
    moved = []
    for column in column_:
        new_column = column + "_moved_" + str(days)
        df[new_column] = df[column].shift(days)
        moved.append(new_column)
    return df, moved





def save_df(df, name):
    """  _ _ _ """
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")




def normeren(df, what_to_norm):
    """In : columlijst
    Bewerking : max = 1
    Out : columlijst met genormeerde kolommen"""
    # print(df.dtypes)

    normed_columns = []

    for column in what_to_norm:
        maxvalue = (df[column].max()) / 100
        firstvalue = df[column].iloc[int(WDW2 / 2)] / 100
        name = f"{column}_normed"
        for i in range(len(df)):
            if how_to_norm == "max":
                df.loc[i, name] = df.loc[i, column] / maxvalue
            else:
                df.loc[i, name] = df.loc[i, column] / firstvalue
        normed_columns.append(name)
        print(f"{name} generated")
    return df, normed_columns


def graph_daily_normed(
    df, what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display):
    """IN : df, de kolommen die genormeerd moeten worden
    ACTION : de grafieken met de genormeerde kolommen tonen"""

    if what_to_show_day_l is None:
        st.warning("Choose something")
        st.stop()

    df, smoothed_columns_l = smooth_columnlist(df, what_to_show_day_l, how_to_smoothen,WDW2, centersmooth)
    df, normed_columns_l = normeren(df, smoothed_columns_l)

    df, smoothed_columns_r = smooth_columnlist(df, what_to_show_day_r, how_to_smoothen, WDW2, centersmooth)
    df, normed_columns_r = normeren(df, smoothed_columns_r)

    graph_daily(df, normed_columns_l, normed_columns_r, None, how_to_display)


def graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t):
    """  _ _ _ """

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

    if 1==1: #with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)
        # Some nice colors chosen with coolors.com

        # #CONFIG
        bittersweet = "#ff6666"  # reddish 0
        operamauve = "#ac80a0"  # purple 1
        green_pigment = "#3fa34d"  # green 2
        minion_yellow = "#EAD94C"  # yellow 3
        mariagold = "#EFA00B"  # orange 4
        falu_red = "#7b2d26"  # red 5
        COLOR_weekday = "#3e5c76"  # blue 6
        COLOR_weekend = "#e49273"  # dark salmon 7
        prusian_blue = "#1D2D44"  # 8
        white = "#eeeeee"
        color_list = [
            "#02A6A8",
            "#4E9148",
            "#F05225",

            "#024754",
            "#FBAA27",
            "#302823",

            "#F07826",
             "#ff6666",  # reddish 0
         "#ac80a0",  # purple 1

         "#3fa34d",  # green 2
         "#EAD94C",  # yellow 3
         "#EFA00B",  # orange 4

         "#7b2d26",  # red 5
         "#3e5c76",  # blue 6
         "#e49273",  # dark salmon 7

         "#1D2D44",  # 8

        "#3e5c76" ,
        "#3e5c76" ,
        "#3e5c76"
        ]

        columnlist, t, WDW2, centersmooth,tg = lijst, "SMA", 7, True,4

        df, columnlist_sm_l = smooth_columnlist(df, what_to_show_l_, how_to_smooth, WDW2, centersmooth)



        for n, b in enumerate(what_to_show_l_):
            # if type(a) == list:
            #     a_=a
            # else:
            #     a_=[a]

            # PROBEERSEL OM WEEK GEMIDDELDES MEE TE KUNNEN PLOTTEN IN DE DAGELIJKSE GRAFIEK

            # dfweek_ = df.groupby('weekalt', sort=False).mean().reset_index()
            # save_df(dfweek_,"whatisdftemp1")
            # w = b + "_week"
            # print ("============="+ w)
            # df_temp = dfweek_[["weekalt",b ]]
            # df_temp = df_temp(columns={b: w})

            # print (df_temp.dtypes)
            # #df_temp is suddenly a table with all the rows
            # print (df_temp)
            # save_df(df_temp,"whatisdftemp2")

            if t != "bar":  # t = line
                df_temp = df

                if how_to_smooth is None:
                    how_to_smooth_ = "unchanged_"
                else:
                    how_to_smooth_ = how_to_smooth + "_" + str(WDW2)
                b_ = str(b) + "_" + how_to_smooth_
                # df_temp[b_].plot(
                #     label=b, color=color_list[n], linewidth=1.1
                # )  # label = b_ for uitgebreid label
                df_temp[b].plot(
                    label=b,
                    color=color_list[n],
                    #linestyle="dotted",
                    #alpha=0.9,
                    #linewidth=0.8,
                )
        if what_to_show_r != None:
            if type(what_to_show_r) == list:
                what_to_show_r = what_to_show_r
            else:
                what_to_show_r = [what_to_show_r]

            n = len(color_list)
            x = n
            for a in what_to_show_r:
                x -= 1
                lbl = a + " (right ax)"
                # df, columnlist = smooth_columnlist(df, [a], how_to_smooth, WDW2, centersmooth)
                # for c_ in columnlist:
                #     # smoothed
                #     lbl2 = a + " (right ax)"
                #     ax3 = df_temp[c_].plot(
                #         secondary_y=True,
                #         label=lbl2,
                #         color=color_list[x],
                #         linestyle="--",
                #         linewidth=1.1,
                #     )  # abel = lbl2 voor uitgebreid label
                ax3 = df_temp[a].plot(
                    secondary_y=True,
                    #linestyle="dotted",
                    color=color_list[x],
                    linewidth=1,
                    #alpha=0.9,
                    label=lbl,
                )
                ax3.set_ylabel("_")


            if len(what_to_show_r) == 1:
                mean = df[what_to_show_r].mean()
                std =df[what_to_show_r].std()
                # print (f"mean {mean}")
                # print (f"st {std}")
                low = mean -2*std
                up = mean +2*std
                #ax3.set_ylim = (-100, 100)
        plt.title(title, fontsize=10)

        a__ = (max(df_temp["date"].tolist())).date() - (
            min(df_temp["date"].tolist())
        ).date()
        freq = int(a__.days / 10)
        ax.xaxis.set_major_locator(MultipleLocator(freq))

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
        plt.legend(handles, labels, bbox_to_anchor=(0, -0.5), loc="lower left", ncol=1)
        ax.text(
            1,
            1.1,
            "Created by Rene Smit — @rcsmit",
            transform=ax.transAxes,
            fontsize="xx-small",
            va="top",
            ha="right",
        )
        if show_R_value_graph or show_R_value_RIVM:
            plt.axhline(y=1, color="yellow", alpha=0.6, linestyle="--")


        #plt.axhline(y=1, color="yellow", alpha=1, linestyle="--")
        if t == "line":
            set_xmargin(ax, left=-0.04, right=-0.04)
        st.pyplot(fig1x)



def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta * left
    right = lim[1] + delta * right
    ax.set_xlim(left, right)





def graph_daily(df, what_to_show_l, what_to_show_r, how_to_smooth, t):
    """  _ _ _ """
    title = ""
    if t == "bar":
        if type(what_to_show_l) == list:
            what_to_show_l = what_to_show_l
        else:
            what_to_show_l = [what_to_show_l]
        for c in what_to_show_l:

            #    what_to_show_r = what_to_show_r


            title += str(c) + " "

        t1 =wrap(title, 40)
        title = "".join(tx + "\n" for tx in t1)
        print (f"titel 1277{title}")

    else:
        tl = ""
        tr = ""
        if what_to_show_l is not None:
            i = 0
            for l in what_to_show_l:
                if i != len(what_to_show_l) - 1:
                    i += 1
                else:
                    tl += l

        if what_to_show_r is not None:
            if type(what_to_show_r) == list:
                what_to_show_r = what_to_show_r
            else:
                what_to_show_r = [what_to_show_r]
            tl += " - \n"
            j = 0
            for r in what_to_show_r:
                if j != len(what_to_show_r) - 1:

                    tl += r + " / "
                    j += 1
                else:

                    tl +=r
        tl = tl.replace("_", " ")

        #title = f"{tl}"
        t1 =wrap(tl, 80)
        for t in t1:
            title += t + "\n"

    graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t)






def init():
    """  _ _ _ """

    global download

    global INPUT_DIR
    global OUTPUT_DIR

    INPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\"
    )
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\output\\"
    )

    # GLOBAL SETTINGS
    download = True  # True : download from internet False: download from INPUT_DIR
    # De open data worden om 15.15 uur gepubliceerd


def get_locations(df_ungrouped, field):
    """ Get a list of the Municipalities """
    return df_ungrouped[field].unique()
    # Municipality_name;
    # Province;
    # Security_region_code;
    # Security_region_name;
    # Municipal_health_service;
    # ROAZ_region


def get_duplicate_cols(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.columns).value_counts()[lambda x: x>1]

def main():
    """  _ _ _ """
    global FROM
    global UNTIL
    global WDW2
    global WDW3, WDW4
    global number_days_contagious
    global showoneday
    global showday
    global MOVE_WR
    global showR
    global lijst  # Lijst in de pull down menu's voor de assen
    global show_scenario
    global how_to_norm
    global Rnew1_, Rnew2_
    global ry1, ry2, total_cases_0, sec_variant, extra_days
    global show_R_value_graph, show_R_value_RIVM, centersmooth
    global OUTPUT_DIR
    global INPUT_DIR
    init()


    df_getdata, UPDATETIME, lijst, provincielijst,column_list_r_smoothened_moved,provincies_per_inw  = get_data()
    df = df_getdata.copy(deep=False)






    st.sidebar.markdown("<hr>", unsafe_allow_html=True)



    st.title("COVID cases en Re-getal per provincie")
    # st.header("")
    st.subheader("Under construction - Please send feedback to @rcsmit")

    # DAILY STATISTICS ################
    df_temp = None
    what_to_show_day_l = None

    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2021-01-01"
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
    options = ["total cases", "R numbers", "cases per 100k inwoners"]
    menu_choice = st.sidebar.radio("What to show",options, index=1)

    if menu_choice == "total cases":
        to_show = provincielijst
    elif menu_choice == "R numbers":
        to_show = column_list_r_smoothened_moved
    elif menu_choice == "cases per 100k inwoners":
        to_show = provincies_per_inw


    what_to_show_day_l = st.sidebar.multiselect(
        "What to show left-axis (multiple possible)", lijst, to_show
    )




    what_to_show_day_r = st.sidebar.multiselect(
        "What to show right-axis (multiple possible)", lijst,
    )
    if what_to_show_day_l is None:
        st.warning("Choose something")
        st.stop()

    move_right = 0

    show_R_value_graph = False
    show_R_value_RIVM = False

    MOVE_WR = st.sidebar.slider("Move the R-curve", -20, 10, -8)

    if what_to_show_day_l == []:
        st.error("Choose something for the left-axis")
        st.stop()

    if what_to_show_day_l is not None:



        how_to_smoothen, how_to_display = "SMA", "line"


        graph_daily(
            df,
            what_to_show_day_l,
            what_to_show_day_r,
            how_to_smoothen,
            how_to_display,
        )
        if len(what_to_show_day_l) > 1:
            for xx in what_to_show_day_l:
                graph_daily(df, [xx], None, how_to_smoothen, how_to_display)




    else:
        st.error("Choose what to show")

    # EXTRA POSSIBLE CALCULATIONS - INTERFACE HAS TO BE WRITTEN







    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/edit/main/covid_dashboard_rcsmit.py" target="_blank">github.com/rcsmit</a><br>'
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
