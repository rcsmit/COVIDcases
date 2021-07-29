import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import datetime as dt

from dashboard_helpers import *
from textwrap import wrap

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker
import math
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import streamlit as st



from helpers import *


def normeren(df, what_to_norm):
    """In : columlijst
    Bewerking : max = 1
    Out : columlijst met genormeerde kolommen"""
    # print(df.dtypes)

    normed_columns = []

    for column in what_to_norm:
        maxvalue = (df[column].max()) / scale_to_x
        firstvalue = df[column].iloc[int(WDW2 / 2)] / scale_to_x
        name = f"{column}_normed"
        for i in range(len(df)):
            if how_to_norm == "max":
                df.loc[i, name] = df.loc[i, column] / maxvalue
            else:
                df.loc[i, name] = df.loc[i, column] / firstvalue
        normed_columns.append(name)
        print(f"{name} generated")
    return df, normed_columns




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


def graph_daily_normed(
    df, what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display, WDW2, centersmooth
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


def graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t, WDW2, centersmooth, FROM,UNTIL):
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
        # Some nice colors chosen with coolors.com

        # #CONFIG

        operamauve = "#ac80a0"  # purple 1
        green_pigment = "#3fa34d"  # green 2
        minion_yellow = "#EAD94C"  # yellow 3
        mariagold = "#EFA00B"  # orange 4
        falu_red = "#7b2d26"  # red 5

        prusian_blue = "#1D2D44"  # 8

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
                color_x = find_color_x(firstday, showoneday, showday)
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

                                if teller == 0:
                                    dfmin = Rn
                                elif teller == 1:
                                    if show_R_value_graph:
                                        ax3 = df[Rn].plot(
                                            secondary_y=True,
                                            label=Rn,
                                            linestyle="--",
                                            color=falu_red,
                                            linewidth=1.2,
                                        )
                                elif teller == 2:
                                    dfmax = Rn
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
        if showlogyaxis == "10":
            ax.semilogy()
        if showlogyaxis == "2":
            ax.semilogy(2)
        if showlogyaxis == "logit":
            ax.set_yscale("logit")

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

        plt.axhline(y=horiz_line, color="black", alpha=0.6, linestyle="--")
        if t == "line":
            set_xmargin(ax, left=-0.04, right=-0.04)
        st.pyplot(fig1x)

    #if len(what_to_show_l) >= 1 and len(what_to_show_r) >= 1:  # add scatter plot
    if what_to_show_l is not None and what_to_show_r is not None:
        for l in what_to_show_l:
            for r in what_to_show_r:

                left_sm = str(l) + "_" + how_to_smooth_
                right_sm = str(r) + "_" + how_to_smooth_
                make_scatterplot(df_temp, l,r, FROM, UNTIL,  True, False)
                make_scatterplot(df_temp,left_sm, right_sm, FROM, UNTIL, True, True)


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


def graph_daily(df, what_to_show_l, what_to_show_r, how_to_smooth, t,  WDW2, centersmooth, FROM,UNTIL, groupby_how):
    """  _ _ _ """
    if t == "bar":
        if type(what_to_show_l) == list:
            what_to_show_l = what_to_show_l
        else:
            what_to_show_l = [what_to_show_l]
        title = ""
        for c in what_to_show_l:

            #    what_to_show_r = what_to_show_r


            title += str(c) + " "

        t1 =wrap(title, 40)
        title = ""
        #st.write (t1)
        for tx in t1:
            title += tx + "\n"
        print (f"titel 1277{title}")

        graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t,  WDW2, centersmooth, FROM,UNTIL)
    else:
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
        t1 =wrap(tl, 80)
        title = ""

        for t in t1:
            title += t + "\n"
        graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t)
