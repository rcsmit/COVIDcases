from numpy.core.numeric import NaN
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
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as ticker
import math
import platform
_lock = RendererAgg.lock
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import streamlit as st
import urllib
import urllib.request
from pathlib import Path
from streamlit import caching
from inspect import currentframe, getframeinfo

###################################################################

def download_data_file(url, filename, delimiter_, fileformat):
    """Download the external datafiles
    IN :  url : the url
          filename : the filename (without extension) to export the file
          delimiter : delimiter
          fileformat : fileformat
    OUT : df_temp : the dataframe
    """

    # df_temp = None
    download = True
    with st.spinner(f"Downloading...{url}"):
        if download:  # download from the internet
            url = url
        else:  # download from the local drive
            if fileformat == "json":
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
            st.error("Error in fileformat")
            st.stop()
        df_temp = df_temp.drop_duplicates()
        # df_temp = df_temp.replace({pd.np.nan: None})  Let it to test
        save_df(df_temp, filename)
        return df_temp


@st.cache(ttl=60 * 60 * 24, suppress_st_warning=True)
def get_data():
    """Get the data from various sources
    In : -
    Out : df        : dataframe
         UPDATETIME : Date and time from the last update"""
    with st.spinner(f"GETTING ALL DATA ..."):
        init()
        # #CONFIG

        if platform.processor() != "":
                 data = [

                {
                    "url": "C:\\Users\\rcxsm\\Documents\phyton_scripts\\covid19_seir_models\\input\\owid-covid-data.csv",
                    "name": "owid",
                    "delimiter": ",",
                    "key": "date",
                    "key2": "location",
                    "dateformat": "%Y-%m-%d",
                    "groupby": None,
                    "fileformat": "csv",
                    "where_field": None,
                    "where_criterium": None
                },


                {
                    "url": "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\waze_mobility.csv",
                    "name": "waze",
                    "delimiter": ",",
                    "key": "date",
                    "key2": "country",
                    "dateformat":  "%Y-%m-%d",
                    "groupby": None,
                    "fileformat": "csv",
                    "where_field": "geo_type",
                    "where_criterium": "country"
                },
                {
                    "url": "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\google_mob_world.csv",
                    "name": "googlemobility",
                    "delimiter": ",",
                    "key": "date",
                    "key2": "country_region",
                    "dateformat":  "%Y-%m-%d",
                    "groupby": None,
                    "fileformat": "csv",
                    "where_field": None,
                    "where_criterium": None

                }
                

            ]

        else:


            data = [

                {
                    "url": "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
                    "name": "owid",
                    "delimiter": ",",
                    "key": "date",
                    "key2": "location",
                    "dateformat": "%Y-%m-%d",
                    "groupby": None,
                    "fileformat": "csv",
                    "where_field": None,
                    "where_criterium": None
                },


                {
                    "url": "https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/waze_reports/waze_mobility.csv",
                    "name": "waze",
                    "delimiter": ",",
                    "key": "date",
                    "key2": "country",
                    "dateformat":  "%Y-%m-%d",
                    "groupby": None,
                    "fileformat": "csv",
                    "where_field": "geo_type",
                    "where_criterium": "country"
                },
                {
                    "url": "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/google_mob_world.csv",
                   
                    #  https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
                    "name": "googlemobility",
                    "delimiter": ",",
                    "key": "date",
                    "key2": "country_region",
                    "dateformat":  "%Y-%m-%d",
                    "groupby": None,
                    "fileformat": "csv",
                    "where_field": None,
                    "where_criterium": None

                },



            ]

        type_of_join = "outer"
        d = 0

        # Read first datafile
        df_temp_x = download_data_file(
            data[d]["url"], data[d]["name"], data[d]["delimiter"], data[d]["fileformat"]
        )
        # df_temp_x = df_temp_x.replace({pd.np.nan: None})
        df_temp_x[data[d]["key"]] = pd.to_datetime(
            df_temp_x[data[d]["key"]], format=data[d]["dateformat"]
        )
        firstkey = data[d]["key"]
        firstkey2 = data[d]["key2"]


        if data[d]["where_field"] != None:
            where_field = data[d]["where_field"]
            df_temp_x = df_temp_x.loc[df_temp_x[where_field] == data[d]["where_criterium"]]

        if data[d]["groupby"] != None:
            df_temp_x = (
                df_temp_x.groupby([data[d]["key"]], sort=True).sum().reset_index()
            )
            df_ungrouped = df_temp_x.reset_index()
            firstkey_ungrouped = data[d]["key"]
        else:
            df_temp_x = df_temp_x.sort_values(by=firstkey)
            df_ungrouped = None

        df_temp = (
            df_temp_x  # df_temp is the base to which the other databases are merged to
        )
        # Read the other files

        for d in range(1, len(data)):

            df_temp_x = download_data_file(
                data[d]["url"],
                data[d]["name"],
                data[d]["delimiter"],
                data[d]["fileformat"],
            )
            # df_temp_x = df_temp_x.replace({pd.np.nan: None})
            oldkey = data[d]["key"]
            newkey = "key" + str(d)
            oldkey2 = data[d]["key2"]
            newkey2 = "key2_" + str(d)
            df_temp_x = df_temp_x.rename(columns={oldkey: newkey})
            df_temp_x = df_temp_x.rename(columns={oldkey2: newkey2})
            #st.write (df_temp_x.dtypes)
            try:
                df_temp_x[newkey] = pd.to_datetime(df_temp_x[newkey], format=data[d]["dateformat"]           )
            except:
                st.error(f"error in {oldkey} {newkey}")
                st.stop()

            if data[d]["where_field"] != None:
                where_field = data[d]["where_field"]
                df_temp_x = df_temp_x.loc[df_temp_x[where_field] == data[d]["where_criterium"]]

            if data[d]["groupby"] != None:
                if df_ungrouped is not None:
                    df_ungrouped = df_ungrouped.append(df_temp_x, ignore_index=True)
                    print(df_ungrouped.dtypes)
                    print(firstkey_ungrouped)
                    print(newkey)
                    df_ungrouped.loc[
                        df_ungrouped[firstkey_ungrouped].isnull(), firstkey_ungrouped
                    ] = df_ungrouped[newkey]

                else:
                    df_ungrouped = df_temp_x.reset_index()
                    firstkey_ungrouped = newkey
                df_temp_x = df_temp_x.groupby([newkey], sort=True).sum().reset_index()

            df_temp = pd.merge(
                df_temp, df_temp_x, how=type_of_join, left_on=[firstkey, firstkey2], right_on=[newkey, newkey2]
            )
            df_temp.loc[df_temp[firstkey].isnull(), firstkey] = df_temp[newkey]
            df_temp = df_temp.sort_values(by=firstkey)
        # the tool is build around "date"
        df = df_temp.rename(columns={firstkey: "date"})

        UPDATETIME = datetime.now()


        return df, df_ungrouped, UPDATETIME

def prepare_google_mob_worlddata():
    """ Bringing back a file of 549 MB to 9 MB. Works only locally"""
    # original location  https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
    url = "C:\\Users\\rcxsm\\Documents\phyton_scripts\\covid19_seir_models\\input\\Global_Mobility_Report.csv"

    df = pd.read_csv(url, delimiter=",", low_memory=False)
    print (df)
    #df = df.loc[df['sub_region_1'] == None]
    df = df[df.sub_region_1.isnull()]
    print (df)
    name_ = "C:\\Users\\rcxsm\\Documents\phyton_scripts\\covid19_seir_models\\input\\google_mob_world.csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)
    print("--- Saving " + name_ + " ---")


def week_to_week(df, column_):
    if type(column_) == list:
        column_ = column_
    else:
        column_ = [column_]
    newcolumns = []
    newcolumns2 = []

    for c in column_:
        newname = str(c) + "_weekdiff"
        newname2 = str(c) + "_weekdiff_index"
        newcolumns.append(newname)
        newcolumns2.append(newname2)
        df[newname] = np.nan
        df[newname2] = np.nan
        for n in range(7, len(df)):
            vorige_week = df.iloc[n - 7][c]
            nu = df.iloc[n][c]
            waarde = round((((nu - vorige_week) / vorige_week) * 100), 2)
            waarde2 = round((((nu) / vorige_week) * 100), 2)
            df.at[n, newname] = waarde
            df.at[n, newname2] = waarde2
    return df, newcolumns, newcolumns2

def rh2q (rh, t, p ):
    # https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html

    #Td = math.log(e/6.112)*243.5/(17.67-math.log(e/6.112))
    es = 6.112 * math.exp((17.67 * t)/(t + 243.5))
    e = es * (rh / 100)
    q_ = (0.622 * e)/(p - (0.378 * e)) * 1000
    q= round(q_,2)
    return q




def move_column(df, column_, days):
    """  _ _ _ """
    if type(column_) == list:
        column_ = column_
    else:
        column_ = [column_]
   
    for column in column_:
        new_column = column + "_moved_" + str(days)
        df[new_column] = df[column].shift(days)
      
    return df, new_column

def move_columnlist(df, column_, days):
    """  _ _ _ """
    if type(column_) == list:
        column_ = column_
    else:
        column_ = [column_]
    moved_columns = []
    for column in column_:
        new_column = column + "_moved_" + str(days)
        df[new_column] = df[column].shift(days)
        moved_columns.append(new_column)
    return df, moved_columns



def drop_columns(df, what_to_drop):
    """  _ _ _ """
    if what_to_drop != None:
        for d in what_to_drop:
            print("dropping " + d)

            df = df.drop(columns=[d], axis=1)
    return df


def select_period(df, show_from, show_until):
    """ _ _ _ """
    if show_from == None:
        show_from = "2020-1-1"

    if show_until == None:
        show_until = "2030-1-1"

    mask = (df["date"].dt.date >= show_from) & (df["date"].dt.date <= show_until)
    df = df.loc[mask]

    df = df.reset_index()

    return df


def agg_week(df, how):
    """  _ _ _ """
    # #TODO
    # HERE ARE SOME PROBLEMS DUE TO MISSING .isotype()
    # FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.
    #  Please use Series.dt.isocalendar().week instead.
    df["weeknr"] = df["date"].dt.week
    df["yearnr"] = df["date"].dt.year

    df["weekalt"] = (
        df["date"].dt.year.astype(str) + "-" + df["date"].dt.week.astype(str)
    )

    for i in range(0, len(df)):
        if df.iloc[i]["weekalt"] == "2021-53":
            df.iloc[i]["weekalt"] = "2020-53"

    # how = "mean"
    if how == "mean":
        dfweek = (
            df.groupby(["weeknr", "yearnr", "weekalt"], sort=False).mean().reset_index()
        )
    elif how == "sum":
        dfweek = (
            df.groupby(["weeknr", "yearnr", "weekalt"], sort=False).sum().reset_index()
        )
    else:
        print("error agg_week()")
        st.stop()
    return df, dfweek





def save_df(df, name):
    """  _ _ _ """
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")


##########################################################
def correlation_matrix(df, werkdagen, weekend_):
    """  _ _ _ """

    # CALCULATE CORRELATION

    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True, annot_kws={"fontsize": 7})
    plt.title("ALL DAYS", fontsize=20)
    plt.show()

    # corrMatrix = werkdagen.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # plt.title("WORKING DAYS", fontsize =20)
    # plt.show()

    # corrMatrix = weekend_.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # plt.title("WEEKEND", fontsize =20)
    # plt.show()

    # MAKE A SCATTERPLOT

    # sn.regplot(y="Rt_avg", x="Kliniek_Nieuwe_Opnames_COVID", data=df)
    # plt.show()


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
        for i in range(0, len(df)):
            if how_to_norm == "max":
                df.loc[i, name] = df.loc[i, column] / maxvalue
            else:
                df.loc[i, name] = df.loc[i, column] / firstvalue
        normed_columns.append(name)
        print(f"{name} generated")
    return df, normed_columns


def graph_daily_normed(
    df, what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display
):
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
    #st.write(f"t = {t}")
    df_temp = pd.DataFrame(columns=["date"])
    if what_to_show_l is None:
        st.warning("Choose something")
        st.stop()

    if type(what_to_show_l) == list:
        what_to_show_l_ = what_to_show_l
    else:
        what_to_show_l_ = [what_to_show_l]

    if type(what_to_show_r) == list:
        what_to_show_r_ = what_to_show_r
    else:
        what_to_show_r_ = [what_to_show_r]
    aantal = len(what_to_show_l_)
    # SHOW A GRAPH IN TIME / DAY

    with _lock:
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

        ]

        n = 0  # counter to walk through the colors-list

        df, columnlist_sm_l = smooth_columnlist(df, what_to_show_l_, how_to_smooth, WDW2, centersmooth)
        df, columnlist_sm_r = smooth_columnlist(df, what_to_show_r_, how_to_smooth, WDW2, centersmooth)

        # CODE TO MAKE STACKED BARS - DOESNT WORK
        # stackon=""
        # if len(what_to_show_l_)>1:
        #     w = ["Datum"]
        #     for s in what_to_show_l_:
        #         w.append(s)
        #     #st.write(w)
        #     df_stacked = df[w].copy()
        #     #print (df_stacked.dtypes)
        #     #df_stacked.set_index('Datum')

        # st.write(df_stacked)
        # if t == "bar":
        # ax = df_stacked.plot.bar(stacked=True)
        # ax = df_stacked.plot(rot=0)
        # st.bar_chart(df_stacked)
        # ax = df[c_smooth].plot(label=c_smooth, color = color_list[2],linewidth=1.5)         # SMA

        for b in what_to_show_l_:
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

            if t == "bar":
                # weekends have a different color
                firstday = df.iloc[0]["WEEKDAY"]  # monday = 0
                if firstday == 0:
                    color_x = [
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekend,
                        COLOR_weekend,
                    ]
                elif firstday == 1:
                    color_x = [
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekend,
                        COLOR_weekend,
                        COLOR_weekday,
                    ]
                elif firstday == 2:
                    color_x = [
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekend,
                        COLOR_weekend,
                        COLOR_weekday,
                        COLOR_weekday,
                    ]
                elif firstday == 3:
                    color_x = [
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekend,
                        COLOR_weekend,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                    ]
                elif firstday == 4:
                    color_x = [
                        COLOR_weekday,
                        COLOR_weekend,
                        COLOR_weekend,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                    ]
                elif firstday == 5:
                    color_x = [
                        COLOR_weekend,
                        COLOR_weekend,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                    ]
                elif firstday == 6:
                    color_x = [
                        COLOR_weekend,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekday,
                        COLOR_weekend,
                    ]

                if showoneday:
                    if showday == 6:
                        color_x = [
                            white,
                            white,
                            white,
                            white,
                            white,
                            white,
                            bittersweet,
                        ]
                    elif showday == 5:
                        color_x = [
                            white,
                            white,
                            white,
                            white,
                            white,
                            bittersweet,
                            white,
                        ]
                    elif showday == 4:
                        color_x = [
                            white,
                            white,
                            white,
                            white,
                            bittersweet,
                            white,
                            white,
                        ]
                    elif showday == 3:
                        color_x = [
                            white,
                            white,
                            white,
                            bittersweet,
                            white,
                            white,
                            white,
                        ]
                    elif showday == 2:
                        color_x = [
                            white,
                            white,
                            bittersweet,
                            white,
                            white,
                            white,
                            white,
                        ]
                    elif showday == 1:
                        color_x = [
                            white,
                            bittersweet,
                            white,
                            white,
                            white,
                            white,
                            white,
                        ]
                    elif showday == 0:
                        color_x = [
                            bittersweet,
                            white,
                            white,
                            white,
                            white,
                            white,
                            white,
                        ]
                # MAYBE WE CAN LEAVE THIS OUT HERE
                df, columnlist = smooth_columnlist(df, [b], how_to_smooth, WDW2, centersmooth)

                df.set_index("date")

                df_temp = df
                if len(what_to_show_l_) == 1:
                    ax = df_temp[b].plot.bar(
                        label=b, color=color_x, alpha=0.6
                    )  # number of cases

                    for c_smooth in columnlist:
                        ax = df[c_smooth].plot(
                            label=c_smooth, color=color_list[2], linewidth=1.5
                        )  # SMA

                    if showR:
                        if show_R_value_RIVM:
                            ax3 = df["Rt_avg"].plot(
                                secondary_y=True,
                                linestyle="--",
                                label="Rt RIVM",
                                color=green_pigment,
                                alpha=0.8,
                                linewidth=1,
                            )
                            ax3.fill_between(
                                df["date"].index,
                                df["Rt_low"],
                                df["Rt_up"],
                                color=green_pigment,
                                alpha=0.2,
                                label="_nolegend_",
                            )
                        tgs = [3.5, 4, 5]

                        teller = 0
                        dfmin = ""
                        dfmax = ""
                        for TG in tgs:
                            df, R_smooth, R_smooth_sec = add_walking_r(
                                df, columnlist, how_to_smooth, TG
                            )

                            for R in R_smooth:
                                # correctie R waarde, moet naar links ivm 2x smoothen
                                df, Rn = move_column(df, R, MOVE_WR)

                                if teller == 1:
                                    if show_R_value_graph:
                                        ax3 = df[Rn].plot(
                                            secondary_y=True,
                                            label=Rn,
                                            linestyle="--",
                                            color=falu_red,
                                            linewidth=1.2,
                                        )
                                else:
                                    if teller == 0:
                                        dfmin = Rn
                                    if teller == 2:
                                        dfmax = Rn
                                        # print (dfmax)
                                teller += 1
                            for R in R_smooth_sec:  # SECOND METHOD TO CALCULATE R
                                # correctie R waarde, moet naar links ivm 2x smoothen
                                df, Rn = move_column(df, R, MOVE_WR)
                                # ax3=df[Rn].plot(secondary_y=True, label=Rn,linestyle='--',color=operamauve, linewidth=1)
                        if show_R_value_graph:
                            ax3.fill_between(
                                df["date"].index,
                                df[dfmin],
                                df[dfmax],
                                color=falu_red,
                                alpha=0.3,
                                label="_nolegend_",
                            )

            else:  # t = line
                df_temp = df

                if how_to_smooth == None:
                    how_to_smooth_ = "unchanged_"
                else:
                    how_to_smooth_ = how_to_smooth + "_" + str(WDW2)
                b_ = str(b) + "_" + how_to_smooth_
                df_temp[b_].plot(
                    label=b, color=color_list[n], linewidth=1.1
                )  # label = b_ for uitgebreid label
                df_temp[b].plot(
                    label="_nolegend_",
                    color=color_list[n],
                    linestyle="dotted",
                    alpha=0.9,
                    linewidth=0.8,
                )
            n += 1
        if show_scenario == True:
            df = calculate_cases(df, ry1, ry2, total_cases_0, sec_variant, extra_days)
            # print (df.dtypes)
            l1 = f"R = {ry1}"
            l2 = f"R = {ry2}"
            ax = df["variant_1"].plot(
                label=l1, color=color_list[4], linestyle="dotted", linewidth=1, alpha=1
            )
            ax = df["variant_2"].plot(
                label=l2, color=color_list[5], linestyle="dotted", linewidth=1, alpha=1
            )
            ax = df["variant_12"].plot(
                label="TOTAL", color=color_list[6], linestyle="--", linewidth=1, alpha=1
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
                df, columnlist = smooth_columnlist(df, [a], how_to_smooth, WDW2, centersmooth)
                for c_ in columnlist:
                    # smoothed
                    lbl2 = a + " (right ax)"
                    ax3 = df_temp[c_].plot(
                        secondary_y=True,
                        label=lbl2,
                        color=color_list[x],
                        linestyle="--",
                        linewidth=1.1,
                    )  # abel = lbl2 voor uitgebreid label
                ax3 = df_temp[a].plot(
                    secondary_y=True,
                    linestyle="dotted",
                    color=color_list[x],
                    linewidth=1,
                    alpha=0.9,
                    label="_nolegend_",
                )
                ax3.set_ylabel("_")


            if len(what_to_show_l) == 1 and len(what_to_show_r) == 1:  # add correlation
                correlation = find_correlation_pair(df, what_to_show_l, what_to_show_r)
                correlation_sm = find_correlation_pair(df, b_, c_)
                title_scatter =  f"{title}({str(FROM)} - {str(UNTIL)})\nCorrelation = {correlation}"
                title = f"{title} \nCorrelation = {correlation}\nCorrelation smoothed = {correlation_sm}"
            

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
        if what_to_show_l == ["reported_div_tested"]:
            ax.set_ylim(0,0.3)
        ax.set_xticks(df_temp["date"].index)
        ax.set_xticklabels(df_temp["date"].dt.date, fontsize=6, rotation=90)
        xticks = ax.xaxis.get_major_ticks()
        if groupby_timeperiod == "none":
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
        if show_R_value_graph or show_R_value_RIVM:
            plt.axhline(y=1, color="yellow", alpha=0.6, linestyle="--")
        if groupby_timeperiod == "none":
            add_restrictions(df, ax)
        plt.axhline(y=0, color="black", alpha=0.6, linestyle="--")
        if t == "line":
            set_xmargin(ax, left=-0.04, right=-0.04)
        st.pyplot(fig1x)

    for left in what_to_show_l:
        for right in what_to_show_r:
            correlation = find_correlation_pair(df, left, right)
            st.write(f"Correlation: {left} - {right} : {correlation}")

    for left_sm in columnlist_sm_l:
        for right_sm in columnlist_sm_r:
            correlation = find_correlation_pair(df, left_sm, right_sm)
            st.write(f"Correlation: {left_sm} - {right_sm} : {correlation}")
    if len(what_to_show_l) == 1 and len(what_to_show_r) == 1:  # add scatter plot
        left_sm = str(what_to_show_l[0]) + "_" + how_to_smooth_
        right_sm = str(what_to_show_r[0]) + "_" + how_to_smooth_
        make_scatterplot(df_temp, what_to_show_l, what_to_show_r, False)
        make_scatterplot(df_temp,left_sm, right_sm, True)
def make_scatterplot(df_temp, what_to_show_l, what_to_show_r, smoothed):
    if type(what_to_show_l) == list:
        what_to_show_l = what_to_show_l
    else:
        what_to_show_l = [what_to_show_l]
    if type(what_to_show_r) == list:
        what_to_show_r = what_to_show_r
    else:
        what_to_show_r = [what_to_show_r]
    with _lock:
            fig1xy = plt.figure()
            ax = fig1xy.add_subplot(111)
            # st.write (x_)
            # print (type(x_))
            
            x_ = df_temp[what_to_show_l].values.tolist()
            y_ = df_temp[what_to_show_r].values.tolist()
            
            plt.scatter(x_, y_)
            x_ = np.array(df_temp[what_to_show_l])
            
           
            y_ = np.array(df_temp[what_to_show_r]) 
           


            #obtain m (slope) and b(intercept) of linear regression line
            idx = np.isfinite(x_) & np.isfinite(y_)
            m, b = np.polyfit(x_[idx], y_[idx], 1)
            model = np.polyfit(x_[idx], y_[idx], 1)

            predict = np.poly1d(model)
            r2 = r2_score  (y_[idx], predict(x_[idx]))
            #print (r2)
            #m, b = np.polyfit(x_, y_, 1)
            # print (m,b)

            #add linear regression line to scatterplot
            plt.plot(x_, m*x_+b, 'r')
            plt.xlabel (what_to_show_l[0])
            plt.ylabel (what_to_show_r[0])
            if smoothed:
                title_scatter = (f"Smoothed: {what_to_show_l[0]} -  {what_to_show_r[0]}\n({FROM} - {UNTIL})\nCorrelation = {find_correlation_pair(df_temp, what_to_show_l, what_to_show_r)}\ny = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")
            else:
                title_scatter = (f"{what_to_show_l[0]} -  {what_to_show_r[0]}\n({FROM} - {UNTIL})\nCorrelation = {find_correlation_pair(df_temp, what_to_show_l, what_to_show_r)}\ny = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

            plt.title(title_scatter)


            ax.text(
                1,
                1.1,
                "Created by Rene Smit — @rcsmit",
                transform=ax.transAxes,
                fontsize="xx-small",
                va="top",
                ha="right",
            )
            st.pyplot(fig1xy)


def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta * left
    right = lim[1] + delta * right
    ax.set_xlim(left, right)


def add_restrictions(df, ax):
    pass

def add_restrictions_original(df, ax):


    """  _ _ _ """
    # Add restrictions
    # From Han-Kwang Nienhuys - MIT-licence
    df_restrictions = pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/restrictions.csv",
        comment="#",
        delimiter=",",
        low_memory=False,
    )

    a = (min(df["date"].tolist())).date()
    b = (max(df["date"].tolist())).date()

    ymin, ymax = ax.get_ylim()
    y_lab = ymin
    for i in range(0, len(df_restrictions)):
        d_ = df_restrictions.iloc[i]["Date"]  # string
        d__ = dt.datetime.strptime(d_, "%Y-%m-%d").date()  # to dateday

        diff = d__ - a
        diff2 = b - d__

        if diff.days > 0 and diff2.days > 0:

            ax.text(
                (diff.days),
                0,
                f'  {df_restrictions.iloc[i]["Description"] }',
                rotation=90,
                fontsize=4,
                horizontalalignment="center",
            )
            # plt.axvline(x=(diff.days), color='yellow', alpha=.3,linestyle='--')


def graph_week(df, what_to_show_l, how_l, what_to_show_r, how_r):
    """  _ _ _ """

    # SHOW A GRAPH IN TIME / WEEK
    df_l, dfweek_l = agg_week(df, how_l)

    if str(FROM) != "2021-01-01":
        st.info(
            "To match the weeknumbers on the ax with the real weeknumbers, please set the startdate at 2021-1-1"
        )
    if what_to_show_r != None:
        df_r, dfweek_r = agg_week(df, how_r)

    if type(what_to_show_l) == list:
        what_to_show_l = what_to_show_l
    else:
        what_to_show_l = [what_to_show_l]

    for show_l in what_to_show_l:
        fig1y = plt.figure()
        ax = fig1y.add_subplot(111)
        ax.set_xticks(dfweek_l["weeknr"])
        ax.set_xticklabels(dfweek_l["weekalt"], fontsize=6, rotation=45)
        label_l = show_l + " (" + how_l + ")"
        dfweek_l[show_l].plot.bar(label=label_l, color="#F05225")

        if what_to_show_r != None:
            for what_to_show_r_ in what_to_show_r:
                label_r = what_to_show_r_ + " (" + how_r + ")"
                ax3 = dfweek_r[what_to_show_r_].plot(
                    secondary_y=True, color="r", label=label_r
                )

        # Add a grid
        plt.grid(alpha=0.2, linestyle="--")

        # Add a Legend
        fontP = FontProperties()
        fontP.set_size("xx-small")
        plt.legend(loc="best", prop=fontP)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.xaxis.set_major_formatter()
        # everything in legend
        # https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
        handles, labels = [], []
        for ax in fig1y.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)

        plt.legend(handles, labels)
        plt.xlabel("Week counted from " + str(FROM))
        # configgraph(titlex)
        if show_R_value_graph or show_R_value_RIVM:
            pass
            #ax3.axhline(y=1, color="yellow", alpha=0.6, linestyle="--")
        st.pyplot(fig1y)
        # plt.show()


def graph_daily(df, what_to_show_l, what_to_show_r, how_to_smooth, t):
    """  _ _ _ """
    if t == "bar":
        if type(what_to_show_l) == list:
            what_to_show_l = what_to_show_l
        else:
            what_to_show_l = [what_to_show_l]
        title = (f"{country_} | ")
        for c in what_to_show_l:

            #    what_to_show_r = what_to_show_r


            title += str(c) + " "

        t1 =wrap(title, 40)
        title = ""
        #st.write (t1)
        for tx in t1:
            title += tx + "\n"
        print (f"titel 1277{title}")

        graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t)
    else:
        title = (f"{country_} | ")
        tl = ""
        tr = ""
        i = 0
        j = 0
        if what_to_show_l is not None:
            for l in what_to_show_l:
                if i != len(what_to_show_l) - 1:
                    if groupby_how == "sum":
                        tl += l+" (sum) /"
                    elif groupby_how == "mean":
                        tl += l+" (mean) /"
                    elif groupby_how == "max":
                        tl += l+" (max) /"
                    else:
                        tl += l + " / "
                    i += 1
                else:

                    if groupby_how == "sum":
                        tl += " (sum) "
                    elif groupby_how == "mean":
                        tl += " (mean) "
                    elif groupby_how == "max":
                        tl += l+" (max) "
                    else:
                        tl += l

        if what_to_show_r is not None:
            if type(what_to_show_r) == list:
                what_to_show_r = what_to_show_r
            else:
                what_to_show_r = [what_to_show_r]
            tl += " - \n"
            for r in what_to_show_r:
                if j != len(what_to_show_r) - 1:
                    if groupby_how == "sum":
                        tl += r+" (sum) /"
                    elif groupby_how == "mean":
                        tl += r+" (mean) /"
                    elif groupby_how == "max":
                        tl += r+" (max) /"
                    else:
                        tl += r + " / "
                    j += 1
                else:

                    if groupby_how == "sum":
                        tl += r+" (sum) "
                    elif groupby_how == "mean":
                        tl += r+" (mean) "
                    elif groupby_how == "max":
                        tl += r+" (max) "
                    else:
                        tl +=r
        tl = tl.replace("_", " ")

        #title = f"{tl}"
        title = (f"{country_} | ")
        t1 =wrap(tl, 80)


        for t in t1:
            title += t + "\n"
        graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t)


def smooth_columnlist(df, columnlist, t, WDW2, centersmooth):
    """  _ _ _ """
    c_smoothen = []
    wdw_savgol = 7
    #if __name__ = "covid_dashboard_rcsmit":
    # global WDW2, centersmooth, show_scenario
    # WDW2=7
    # st.write(__name__)
    # centersmooth = False
    # show_scenario = False
    if columnlist is not None:
        if type(columnlist) == list:
            columnlist_ = columnlist
        else:
            columnlist_ = [columnlist]
            # print (columnlist)
        for c in columnlist_:
            print(f"Smoothening {c}")
            if t == "SMA":
                new_column = c + "_SMA_" + str(WDW2)
                print("Generating " + new_column + "...")
                df[new_column] = (
                    df.iloc[:, df.columns.get_loc(c)]
                    .rolling(window=WDW2, center=centersmooth)
                    .mean()
                )

            elif t == "savgol":
                new_column = c + "_savgol_" + str(WDW2)
                print("Generating " + new_column + "...")
                df[new_column] = df[c].transform(lambda x: savgol_filter(x, WDW2, 2))

            elif t == None:
                new_column = c + "_unchanged_"
                df[new_column] = df[c]
                print("Added " + new_column + "...~")
            else:
                print("ERROR in smooth_columnlist")
                st.stop()
            c_smoothen.append(new_column)
    return df, c_smoothen


###################################################################
def find_correlations(df, treshold, fields):
    al_gehad = []
    paar = []
    # column_list = list(df.columns)
    column_list = fields
    # print (column_list)
    st.header("Found correlations in the data :")
    for i in column_list:
        for j in column_list:
            # paar = [j, i]
            paar = str(i) + str(j)
            if paar not in al_gehad:
                if i == j:
                    pass
                else:
                    try:
                        c = round(df[i].corr(df[j]), 3)
                        if c >= treshold or c <= (treshold * -1):
                            st.write(f"{i} - {j} - {str(c)}")

                    except:
                        pass
            else:
                pass  # ("algehad")
            al_gehad.append(str(j) + str(i))


def find_correlation_pair(df, first, second):
    al_gehad = []
    paar = []
    if type(first) == list:
        first = first
    else:
        first = [first]
    if type(second) == list:
        second = second
    else:
        second = [second]
    for i in first:
        for j in second:
            c = round(df[i].corr(df[j]), 3)
    return c


def find_lag_time(df, what_happens_first, what_happens_second, r1, r2):
    b = what_happens_first
    a = what_happens_second
    x = []
    y = []
    max = 0
    max_column = None
    for n in range(r1, (r2 + 1)):
        df, m = move_column(df, b, n)
        c = round(df[m].corr(df[a]), 3)
        if c > max:
            max = c
            max_column = m
            m_days = n
        x.append(n)
        y.append(c)
    title = f"Correlation between : {a} - {b} "
    title2 = f" {a} - b - moved {m_days} days "

    fig1x = plt.figure()
    ax = fig1x.add_subplot(111)
    plt.xlabel("shift in days")
    plt.plot(x, y)
    plt.axvline(x=0, color="yellow", alpha=0.6, linestyle="--")
    # Add a grid
    plt.grid(alpha=0.2, linestyle="--")
    plt.title(title, fontsize=10)
    plt.show()
    graph_daily(df, [a], [b], "SMA", "line")
    graph_daily(df, [a], [max_column], "SMA", "line")
    # if the optimum is negative, the second one is that x days later


def init():
    """  _ _ _ """

    global download

    global INPUT_DIR
    global OUTPUT_DIR

    INPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\"
    )
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\"
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
    print(df_ungrouped)

def get_duplicate_cols(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.columns).value_counts()[lambda x: x>1]

def isNaN(num):
    if float('-inf') < float(num) < float('inf'):
        return False
    else:
        return True

def google_or_waze(df___):
    move_r_number = st.sidebar.slider("Move the R-rate (days", -21, 21, -7)

    
  
    #df_output = pd.DataFrame(columns=header)
    output=[]
    to_compare_ = ["transit_stations", "driving_waze"]
    
    countrylist =  df___['location'].drop_duplicates().sort_values().tolist()
   
    header = ["_", "transit_stations", "driving_waze", "transit_stations_SMA", "driving_waze_SMA", "GoogleWazeIndex", "Who_wins"]
    
    text = "Welcome to the first day... of the rest... of your life"
    #to_compare_corr = ["transit_stations", "driving_waze", "transit_stations_SMA", "driving_waze_SMA"]

    t = st.empty()
    l = len(countrylist)
    google_wins,waze_wins = 0, 0
    
    for i, country in enumerate(countrylist):
        
        # progress = ("#" * i) + ("_" * (l-i))
        # if i % 30 == 0:
        #     progress += "\n"
        # t.markdown(progress)
        NumberofNotaNumber = 0
        
        df = df___.loc[df___['location'] ==country].copy(deep=False)
        df, to_compare_sma = smooth_columnlist(df, to_compare_, "SMA",7 , True)
        df, moved_column_repr_rate = move_column(df, "reproduction_rate", move_r_number )
        
        to_compare_corr = to_compare_ + to_compare_sma
        
        output_ = [country]
    	
        for f in to_compare_corr: 
            
            correlation = find_correlation_pair(df,  moved_column_repr_rate, f)
            if isNaN(correlation):
                NumberofNotaNumber += 1

            output_.append(correlation)

        if NumberofNotaNumber <2:
            try:
                output_.append(output_[1]/output_[2])
            except:
                output_.append(None)

            if abs(output_[1])>abs(output_[2]):
                output_.append("Google")
                google_wins +=1
            elif abs(output_[1])<abs(output_[2]):
                output_.append("Waze")
                waze_wins +=1
            else:
                output_.append("Equal")
            
            output.append(output_)
        
        df_output=pd.DataFrame(output,columns=header)
    save_df(df_output, "Google_or_waze.csv")

        #df_output = df_output.append(output, ignore_index=True)
    st.write (df_output)
    st.write(f"Google wins {google_wins} - Waze wins {waze_wins}")
   
    #url ="C:\\Users\\rcxsm\\Documents\phyton_scripts\\covid19_seir_models\\COVIDcases\\motorvehicles.csv"
    url ="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/motorvehicles.csv"
    # https://ourworldindata.org/grapher/road-vehicles-per-1000-inhabitants-vs-gdp-per-capita?yScale=log
    df_motorveh = pd.read_csv(url, delimiter=";", low_memory=False)
    
    df_temp1 = pd.merge(
                df_output, df_motorveh, how="left", left_on="_", right_on="country"
            )
    
    url ="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/GDPpercapita.csv"
    # https://ourworldindata.org/grapher/road-vehicles-per-1000-inhabitants-vs-gdp-per-capita?yScale=log
    df_gdp_per_capita = pd.read_csv(url, delimiter=",", low_memory=False)
    for column in df_gdp_per_capita:
        if column !="Country Name":
            df_gdp_per_capita.rename(columns={column:'GDP_'+column}, inplace=True)
    
    #df_gdp_per_capita = df_gdp_per_capita[["Country Name", "2019"]]
    df_temp = pd.merge(
                df_temp1, df_gdp_per_capita, how="left", left_on="_", right_on="Country Name"
            )
    

    make_scatterplot(df_temp, "driving_waze", "transit_stations", False)
    make_scatterplot(df_temp, "motorvehicles", "GoogleWazeIndex", False)
    make_scatterplot(df_temp,  "motorvehicles","driving_waze", False),
    make_scatterplot(df_temp, "motorvehicles", "transit_stations", False)
    make_scatterplot(df_temp, "GDP_2019", "GoogleWazeIndex", False)
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
    global UPDATETIME
    global country_
    WDW2 = 7
    centersmooth = True
    init()
    show_scenario = False
    df_getdata, df_ungrouped_, UPDATETIME = get_data()
    df = df_getdata.copy(deep=False)
    if df_ungrouped_ is not None:
        df_ungrouped = df_ungrouped_.copy(deep=False)

    # rioolwaterplaatsen = (get_locations(df_ungrouped, "RWZI_AWZI_name"))


    # #CONFIG



    df.rename(
        columns={

            "retail_and_recreation_percent_change_from_baseline":  "retail_and_recreation",
            "grocery_and_pharmacy_percent_change_from_baseline": "grocery_and_pharmacy",
            "parks_percent_change_from_baseline" :  "parks",
            "transit_stations_percent_change_from_baseline" : "transit_stations",
            "workplaces_percent_change_from_baseline":   "workplaces",
            "residential_percent_change_from_baseline":  "residential",

        },
        inplace=True,
    )
    lijst = df.columns.tolist()


    del lijst[0:4]


    st.title("Interactive Corona Dashboard OWID/waze")
    # st.header("")
    st.subheader("Under construction - Please send feedback to @rcsmit")

    # DAILY STATISTICS ################
    df_temp = None
    what_to_show_day_l = None

    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2020-01-01"
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

    df = select_period(df, FROM, UNTIL)


    df = df.drop_duplicates()
    google_or_waze(df)
    dashboard(df)


def dashboard(df___):
    global country_
    countrylist =  df___['location'].drop_duplicates().sort_values().tolist()

    country_ = st.sidebar.selectbox("Which country",countrylist, 216)
    df = df___.loc[df___['location'] ==country_].copy(deep=False)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
   

    # df,newcolumns = week_to_week(df,["Total_reported"])
    global show_R_value_graph, show_R_value_RIVM, show_scenario
    show_R_value_graph, show_R_value_RIVM, show_scenario = False, False, False

    # st.write(df.dtypes)

    w2w = [

    ]

    how_to_smoothen = "SMA"

    centersmooth = True

    WDW2 = 7
    #st.write(get_duplicate_cols(df))
    df, smoothed_columns_w2w0 = smooth_columnlist(df, w2w, how_to_smoothen, WDW2, centersmooth)
    df, newcolumns_w2w, newcolumns2_w2w = week_to_week(df, smoothed_columns_w2w0)

    lijst.extend(newcolumns_w2w) # percentage
    lijst.extend(newcolumns2_w2w) # index

    df, smoothed_columns_w2w1 = smooth_columnlist(df, newcolumns_w2w, how_to_smoothen, WDW2, centersmooth)
    df, newcolumns_w2w2, newcolumns2_w2w2 = week_to_week(df, smoothed_columns_w2w1)

    lijst.extend(newcolumns_w2w2) # percentage


    # for n in newcolumns:
    #     .write(df[n])
    # graph_daily       (df,newcolumns,None, "SMA", "line")
    # st.stop()

    week_or_day = st.sidebar.selectbox("Day or Week", ["day", "week"], index=0)
    if week_or_day != "week":
        how_to_display = st.sidebar.selectbox(
            "What to plot (line/bar)",
            ["line", "line_scaled_to_peak", "line_first_is_100", "bar"],
            index=0,
        )
    else:
        how_to_display = "bar"

    if how_to_display != "bar":
        what_to_show_day_l = st.sidebar.multiselect(
            "What to show left-axis (multiple possible)", lijst, ["reproduction_rate"]
        )
        what_to_show_day_r = st.sidebar.multiselect(
            "What to show right-axis (multiple possible)", lijst, ["driving_waze", "transit_stations"]
        )
        if what_to_show_day_l == None:
            st.warning("Choose something")
            st.stop()
        move_right = st.sidebar.slider("Move curves at right axis (days)", -14, 14, 7)
    else:
        move_right = 0
    showR = False
    if how_to_display == "bar":
        what_to_show_day_l = st.sidebar.selectbox(
            "What to show left-axis (bar -one possible)", lijst, index=7
        )
        # what_to_show_day_l = st.sidebar.multiselect('What to show left-axis (multiple possible)', lijst, ["Total_reported"]  )

        showR = st.sidebar.selectbox("Show R number", [True, False], index=0)
        if what_to_show_day_l == []:
            st.error("Choose something for the left-axis")
        if showR == False:
            what_to_show_day_r = st.sidebar.multiselect(
                "What to show right-axis (multiple possible)", lijst, ["Total_reported"]
            )
            show_R_value_graph = False
            show_R_value_RIVM = False
        else:

            show_R_value_graph = st.sidebar.checkbox(
                f"Show R from {what_to_show_day_l}", value=True
            )
            show_R_value_RIVM = st.sidebar.checkbox("Show R-value RIVM", value=True)
            what_to_show_day_r = None
            pass  # what_to_show_day_r = st.sidebar.selectbox('What to show right-axis (line - one possible)',lijst, index=6)
        lijst_x = [0, 1, 2, 3, 4, 5, 6]
    else:
        show_R_value_graph = False
        show_R_value_RIVM = False
    if week_or_day == "day" and how_to_display == "bar":
        firstday = int(df.iloc[0]["WEEKDAY"])  # monday = 0
        dagenvdweek = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        showonedaylabel = "Show which day (0 = " + dagenvdweek[firstday] + ")"
        showoneday = st.sidebar.selectbox("Show one day", [True, False], index=1)
        if showoneday:
            showday = st.sidebar.selectbox(showonedaylabel, lijst_x, index=0)
    else:
        showoneday = False
        showday = 0
    global groupby_timeperiod
    global groupby_how

    groupby_timeperiod =  st.sidebar.selectbox("GROUPBY : none, week or month", ["none", "1W", "1M"], index=0)
    if groupby_timeperiod != "none":
        groupby_how = st.sidebar.selectbox("GROUPBY : Sum / mean / max", ["sum", "mean"], index=0)
    else:
        groupby_how = "None"

        if groupby_how == "sum":
            df = df.groupby(pd.Grouper(key="date", freq=groupby_timeperiod)).sum().reset_index()
        elif groupby_how == "mean":
            df = df.groupby(pd.Grouper(key="date", freq=groupby_timeperiod)).mean().reset_index()
        elif groupby_how == "max" :
            # TOFIX : gives error
            df = df.groupby(pd.Grouper(key="date", freq=groupby_timeperiod)).max() # .reset_index()

    how_to_smoothen = st.sidebar.selectbox(
        "How to smooth (SMA/savgol)", ["SMA", "savgol"], index=0
    )
    centersmooth =  st.sidebar.selectbox(
        "Smooth in center", [True, False], index=0
    )
    if groupby_timeperiod == "none":
        WDW2 = st.sidebar.slider("Window smoothing curves (days)", 1, 45, 7)
    else:
        WDW2 = st.sidebar.slider("Window smoothing curves (days)", 1, 45, 1)
    if how_to_smoothen == "savgol" and int(WDW2 / 2) == (WDW2 / 2):
        st.warning("When using Savgol, the window has to be uneven")
        st.stop()
    if showR == True:
        WDW3 = st.sidebar.slider("Window smoothing R-number", 1, 14, 7)
        WDW4 = st.sidebar.slider("Calculate R-number over .. days", 1, 14, 4)

        MOVE_WR = st.sidebar.slider("Move the R-curve", -20, 10, -8)
    else:
        showR = False

    if week_or_day == "week":
        how_to_agg_l = st.sidebar.selectbox(
            "How to agg left (sum/mean)", ["sum", "mean"], index=0
        )
        how_to_agg_r = st.sidebar.selectbox(
            "How to agg right (sum/mean)", ["sum", "mean"], index=0
        )
    number_days_contagious = st.sidebar.slider("Aantal dagen besmettelijk", 1, 21, 8)

    show_scenario = st.sidebar.selectbox("Show Scenario", [True, False], index=1)
    if show_scenario:

        total_cases_0 = st.sidebar.number_input(
            "Total number of positive tests", None, None, 8000
        )

        Rnew_1_ = st.sidebar.slider("R-number first variant", 0.1, 10.0, 0.84)
        Rnew_2_ = st.sidebar.slider("R-number second variant", 0.1, 6.0, 1.16)
        f = st.sidebar.slider("Correction factor", 0.0, 2.0, 1.00)
        ry1 = round(Rnew_1_ * f, 2)
        ry2 = round(Rnew_2_ * f, 2)
        sec_variant = st.sidebar.slider(
            "Percentage second variant at start", 0.0, 100.0, 10.0
        )
        extra_days = st.sidebar.slider("Extra days", 0, 60, 0)

    if what_to_show_day_l == []:
        st.error("Choose something for the left-axis")
        st.stop()

    if what_to_show_day_l is not None:

        if week_or_day == "day":
            if move_right != 0 and len(what_to_show_day_r) != 0:
                df, what_to_show_day_r = move_columnlist(df, what_to_show_day_r, move_right)
            if how_to_display == "line":
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

            elif how_to_display == "line_scaled_to_peak":
                how_to_norm = "max"
                graph_daily_normed(
                    df,
                    what_to_show_day_l,
                    what_to_show_day_r,
                    how_to_smoothen,
                    how_to_display,
                )
                if len(what_to_show_day_l) > 1:
                    for xx in what_to_show_day_l:
                        graph_daily_normed(
                            df, [xx], None, how_to_smoothen, how_to_display
                        )
            elif how_to_display == "line_first_is_100":
                how_to_norm = "first"
                graph_daily_normed(
                    df,
                    what_to_show_day_l,
                    what_to_show_day_r,
                    how_to_smoothen,
                    how_to_display,
                )
                if len(what_to_show_day_l) > 1:
                    for xx in what_to_show_day_l:
                        graph_daily_normed(
                            df, [xx], None, how_to_smoothen, how_to_display
                        )

            elif how_to_display == "bar":
                # st.write(what_to_show_day_l)
                graph_daily(
                    df,
                    what_to_show_day_l,
                    what_to_show_day_r,
                    how_to_smoothen,
                    how_to_display,
                )

        else:
            if showR == True:
                if what_to_show_day_r != None:
                    st.warning("On the right axis the R number will shown")
                graph_week(df, what_to_show_day_l, how_to_agg_l, None, how_to_agg_r)
            else:
                graph_week(
                    df,
                    what_to_show_day_l,
                    how_to_agg_l,
                    what_to_show_day_r,
                    how_to_agg_r,
                )
                if len(what_to_show_day_r) > 0:
                    for xx in what_to_show_day_r:
                        graph_daily_normed(
                            df, [xx], None, how_to_smoothen, how_to_display
                        )

    else:
        st.error("Choose what to show")

    # EXTRA POSSIBLE CALCULATIONS - INTERFACE HAS TO BE WRITTEN

    if st.sidebar.button("Google or Waze"):
        google_or_waze(df)

    if st.sidebar.button("Find Correlations"):
        treshhold = st.sidebar.slider("R-number first variant", 0.0, 1.0, 0.8)
        find_correlations(df, treshhold, lijst)
 
    # find_lag_time(df,"transit_stations","Rt_avg", 0,10)
    # correlation_matrix(df,werkdagen, weekend_)

    toelichting = (
      ""
    )

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/edit/main/covid_dashboard_rcsmit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        'Restrictions by <a href="https://twitter.com/hk_nien" target="_blank">Han-Kwang Nienhuys</a> (MIT-license).</div>'
    )

    st.markdown(toelichting, unsafe_allow_html=True)
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
