# Make a stackplot and a stackplot where total = 100% of agegroups
# René Smit (@rcsmit) - MIT Licence

# IN: https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv
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
import prepare_casuslandelijk


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
    #       0-9     10-19   20-29  30-39   40-49   50-59   60-69   70-79  80+
    #pop_ =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    #fraction = [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]

    for c in columnlist:
        #new_column = c + "_SMA"
        new_column = c

        # print("Generating " + new_column + "...")
        df[new_column] = (
            df.iloc[:, df.columns.get_loc(c)].rolling(window=1, center=True).mean()
        )
        columnlist_sma_df.append(df[new_column])
        columnlist_df.append(df[c])
        columnlist_names.append(new_column)
        columnlist_ages.append(c)           # alleen de leeftijden, voor de legenda

    return df,columnlist_df, columnlist_sma_df,columnlist_names,columnlist_ages

def hundred_stack_area(df, column_list):
    l = len(df)
    df["rowtotal"] = np.nan
    columnlist_names = []
    dfcolumnlist = []
    columnlist_ages = []
    for c in column_list:
            new_column = str(c) + "_hstack"

            columnlist_ages.append(c)


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

def make_age_graph(df, d, titel):

    # df = agg_ages(df)
    fig, ax = plt.subplots()
    for d_ in d:
        plt.plot(df["Date_of_statistics_week_start"], df[d_], label = d_)
    plt.legend()
    titel_ = titel + " (weekcijfers)"
    plt.title(titel_)


    plt.tight_layout()
    plt.show()

def make_graph(df, columns_df,columnlist_names, columnlist_ages, datumveld, titel):
    #datumlijst = df[datumveld].tolist()
    #df = df[:-1] # drop last row since this one is incomplete

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

def show(df, c1,titel):


    datumveld = "Date_of_statistics_week_start"
    df, columnlist_df, columnlist_sma_df, columnlist_names, columnlist_ages = smooth(df, c1)

    titel = titel + " (weekcijfers)"


    make_graph (df,       columnlist_df,     columnlist_sma_df, columnlist_names, datumveld, titel)
    df, columnlist_hdred_names, columnlist_hdred_df, columnlist_ages = hundred_stack_area(df, columnlist_names)
    make_graph (df, columnlist_hdred_df,columnlist_names,  columnlist_ages , datumveld, titel)

def agg_ages(df):
    # make age groups
    df["0-49"] = df["0-14"] + df["15-19"] + df["20-24"] + df["25-29"] + df["30-34"] + df["35-39"] + df["40-44"] + df["45-49"]
    df["50-79"] = df["50-54"] + df["55-59"] + df["60-64"] + df["65-69"] + df["70-74"] + df["75-79"]
    df["80+"] = df["80-84"] + df["85-89"] + df["90+"]
    return df

def prepare_data():

    show_from = "2020-1-1"
    show_until = "2030-1-1"

    url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv"
    # url1 = "https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv"
    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    datumveld = "Date_of_statistics_week_start"
    df[datumveld] = pd.to_datetime(df[datumveld], format="%Y-%m-%d")

    df = df.reset_index()
    df.fillna(value=0, inplace=True)

    startdate = pd.to_datetime(show_from).date()
    enddate = pd.to_datetime(show_until).date()

    mask = (df[datumveld].dt.date >= startdate) & (df[datumveld].dt.date <= enddate)
    df = df.loc[mask]

    df_pivot_hospital = (
        pd.pivot_table(
            df,
            values="Hospital_admission",
            index=["Date_of_statistics_week_start"],
            columns=["Age_group"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )


    df_pivot_ic = (
        pd.pivot_table(
            df,
            values="IC_admission",
            index=["Date_of_statistics_week_start"],
            columns=["Age_group"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    return df_pivot_hospital, df_pivot_ic


def main():
    df_pivot_hospital, df_pivot_ic  = prepare_data()
    df_pivot_hospital = agg_ages(df_pivot_hospital)
    df_pivot_ic = agg_ages(df_pivot_ic)

    #cx = ["0-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+", "Unknown"]
    cx =  [ "0-49", "50-79","80+"]
    # SHOW STACKGRAPHS
    cxx = [[df_pivot_hospital, cx, "ziekenhuisopname naar leeftijd"],[df_pivot_ic, cx, "IC opname naar leeftijd"]]
    for d in cxx:
        show (d[0],d[1],d[2])

    # SHOW LINEGRAPHS
    #d = [ "0-49", "50-79","80+"]
    d = [ "0-49"]
    make_age_graph(df_pivot_hospital, d, "ziekenhuisopnames")
    make_age_graph(df_pivot_ic, d, "IC opnames")

if __name__ == "__main__":
    main()
