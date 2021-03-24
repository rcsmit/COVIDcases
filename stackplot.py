# Make a stackplot of agegroups of positive tested people
# René Smit (@rcsmit) - MIT Licence

# TODO : Legend
#        Nice colors
#        Restrictions
#        Set a date-period
#        Make everything a function call
#        Integration in the dashboard
#
# Inspired by a graph by @chivotweets
# https://twitter.com/rubenivangaalen/status/1374443261704605697

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def smooth(df, columnlist):
    for c in columnlist:
        print(f"Smoothening {c}")
        new_column = c + "_SMA"
        print("Generating " + new_column + "...")
        df[new_column] = (
            df.iloc[:, df.columns.get_loc(c)].rolling(window=7, center=True).mean()
        )
    return df


def main():
    url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\landelijk_leeftijd_pivot.csv"
    df = pd.read_csv(url1, delimiter=",", low_memory=False)

    show_from = "2020-7-1"
    show_until = "2030-1-1"

    # mask = (df["pos_test_Date_statistics"].dt.date >= show_from) & (df["pos_test_Date_statistics"].dt.date <= show_until)
    # df = df.loc[mask]

    df = df.reset_index()
    df["pos_test_Date_statistics"] = pd.to_datetime(
        df["pos_test_Date_statistics"], format="%Y-%m-%d"
    )

    cl = cl = [
        "pos_test_0-9",
        "pos_test_10-19",
        "pos_test_20-29",
        "pos_test_30-39",
        "pos_test_40-49",
        "pos_test_50-59",
        "pos_test_60-69",
        "pos_test_70-79",
        "pos_test_80-89",
        "pos_test_90+",
        "pos_test_<50",
        "pos_test_Unknown",
    ]
    df = smooth(df, cl)
    datumlijst = df["pos_test_Date_statistics"].tolist()

    # cols2 =[ 'deceased_0-9', 'deceased_10-19', 'deceased_20-29', 'deceased_30-39', 'deceased_40-49', 'deceased_50-59', 'deceased_60-69', 'deceased_70-79', 'deceased_80-89', 'deceased_90+', 'deceased_<50']
    cols2 = [
        df["pos_test_0-9_SMA"],
        df["pos_test_10-19_SMA"],
        df["pos_test_20-29_SMA"],
        df["pos_test_30-39_SMA"],
        df["pos_test_40-49_SMA"],
        df["pos_test_50-59_SMA"],
        df["pos_test_60-69_SMA"],
        df["pos_test_70-79_SMA"],
        df["pos_test_80-89_SMA"],
        df["pos_test_90+_SMA"],
        df["pos_test_<50_SMA"],
        df["pos_test_Unknown_SMA"],
    ]
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    fig, ax = plt.subplots()
    ax.stackplot(datumlijst, cols2)
    ax.legend(loc="upper left")
    plt.legend()
    plt.show()

main()
