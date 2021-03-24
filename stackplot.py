# Make a stackplot and a stackplot where total = 100% of agegroups of positive tested people
# René Smit (@rcsmit) - MIT Licence

# IN: File generated with https://github.com/rcsmit/COVIDcases/blob/main/prepare_casuslandelijk.py
# OUT : Stackplots
#
# TODO : Legend DONE
#        Nice colors DONE
#        Restrictions ??
#        Set a date-period DONE
#        Make everything a function call
#        Integration in the dashboard
#
# Inspired by a graph by @chivotweets
# https://twitter.com/rubenivangaalen/status/1374443261704605697

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import datetime as dt
from datetime import datetime, timedelta


def save_df(df, name):
    """  _ _ _ """
    OUTPUT_DIR = (
          "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\"
    )
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")

def smooth(df, columnlist):
    columnlist_sma_df = []
    columnlist_df= []
    columnlist_names= []
    columnlist_ages = []
    for c in columnlist:
        new_column = c + "_SMA"

        print("Generating " + new_column + "...")
        df[new_column] = (
            df.iloc[:, df.columns.get_loc(c)].rolling(window=7, center=True).mean()
        )
        columnlist_sma_df.append(df[new_column])
        columnlist_df.append(df[c])
        columnlist_names.append(new_column)
        columnlist_ages.append(c[-5])

    return df, columnlist_sma_df,columnlist_names,columnlist_ages

def hundred_stack_area(df, column_list):
    l = len(df)
    df["rowtotal"] = np.nan
    columnlist_names = []
    dfcolumnlist = []
    columnlist_ages = []
    for c in column_list:
            new_column = str(c) + "_hstack"
            if c[-10]=="_":
                columnlist_ages.append(c[-9:-4])
            else:
                columnlist_ages.append(c[-7:-4])

            df[new_column] = np.nan
            columnlist_names.append(new_column)
    for r in range(df.first_valid_index(),(df.first_valid_index()+l)):
        row_total = 0
        for c in column_list:
            # print (r)
            # print (df)
            # print (df.loc[r ,c]
            row_total += df.loc[r ,c]
            df.loc[r, "rowtotal"] = row_total
    for c in column_list:
        new_column = str(c) + "_hstack"
        for r in range(df.first_valid_index(),(df.first_valid_index()+l)):
            df.loc[r, new_column] = round((100 * df.loc[r, c] / df.loc[r, "rowtotal"]),2)
        dfcolumnlist.append(df[new_column])

    df = df.drop(columns=["rowtotal"], axis=1)

    return df, columnlist_names, dfcolumnlist,columnlist_ages
def drop_columns(df, what_to_drop):
    """  drop columns. what_to_drop : list """
    if what_to_drop != None:
        what_to_drop = [what_to_drop]
        print("dropping " + str(what_to_drop))
        for d in what_to_drop:
            df = df.drop(columns=[d], axis=1)
    return df
def convert(list):
    return tuple(list)

def make_graph(df, columns_df,columnlist_names, columnlist_ages, datumveld, titel):
    #datumlijst = df[datumveld].tolist()
    datumlijst = df[datumveld].tolist()
    color_list = [  "#ff6666",  # reddish 0
                    "#ac80a0",  # purple 1
                    "#3fa34d",  # green 2
                    "#EAD94C",  # yellow 3
                    "#EFA00B",  # orange 4
                    "#7b2d26",  # red 5
                    "#3e5c76",  # blue 6
                    "#e49273" , # dark salmon 7
                    "#1D2D44",  # 8
                    "#02A6A8",
                    "#4E9148",
                    "#F05225",
                    "#024754",
                    "#FBAA27",
                    "#302823",
                    "#F07826",
                     ]

    fig, ax = plt.subplots()

    sp = ax.stackplot(datumlijst, columns_df, colors=color_list)
    #ax.legend(loc="upper left")
    plt.title(titel)

    proxy = [mpl.patches.Rectangle((0,0), 0,0, facecolor=pol.get_facecolor()[0]) for pol in sp]
    ax.legend(proxy, tuple (columnlist_ages),  bbox_to_anchor=(1.3, 1),loc="best")

    plt.tight_layout()
    plt.show()

def show(c1,titel,datumveld):
    url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\final_result2.csv"
    df = pd.read_csv(url1, delimiter=",", low_memory=False)

    show_from = "2020-3-1"
    show_until = "2030-1-1"

    df = df.reset_index()
    df.fillna(value=0, inplace=True)
    save_df(df,"waaromx")
    df[datumveld] = pd.to_datetime(
        df[datumveld], format="%Y-%m-%d"
    )

    startdate = pd.to_datetime(show_from).date()
    enddate = pd.to_datetime(show_until).date()

    mask = (df[datumveld].dt.date >= startdate) & (df[datumveld].dt.date <= enddate)
    df = df.loc[mask]

    df, columnlist_sma_df, columnlist_names, columnlist_ages = smooth(df, c1)
    # make_graph (df, columnlist_sma_df, columnlist_names, datumveld, titel)

    df, columnlist_hdred_names, columnlist_hdred_df, columnlist_ages = hundred_stack_area(df, columnlist_names)
    make_graph (df, columnlist_hdred_df,columnlist_names,  columnlist_ages , datumveld, titel)


def main():
    cx1  = [
        "pos_test_0-9",
        "pos_test_10-19",
        "pos_test_20-29",
        "pos_test_30-39",
        "pos_test_40-49",
        "pos_test_50-59",
        "pos_test_60-69",
        "pos_test_70-79",
        "pos_test_80-89",
        "pos_test_90+"]

    cx2 = [
        "deceased_50-59",
        "deceased_60-69",
        "deceased_70-79",
        "deceased_80-89",
        "deceased_90+"]

    cx3 = ["hosp_0-9",
        "hosp_10-19",
        "hosp_20-29",
        "hosp_30-39",
        "hosp_40-49",
        "hosp_50-59",
        "hosp_60-69",
        "hosp_70-79",
        "hosp_80-89",
        "hosp_90+"]
    datumveld = "pos_test_Date_statistics"
    cxx = [[cx1, "pos test",datumveld],
            [cx2, "deceased",datumveld],
           [cx3, "hospital (onderrapportage)",datumveld]]
    for d in cxx:
        show (d[0],d[1],d[2])

main()
