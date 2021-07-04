# PREPARE A CSV-FILE TO ENABLE AN STACKED PLOT FOR POSITIVE TESTS, HOSPITALIZATIONS AND DECEASED
# Hospitalizations and deceased are not lagged in time, the date of the result of the "desease onset", positieve test or notification is leading
# https://data.rivm.nl/geonetwork/srv/dut/catalog.search#/metadata/2c4357c8-76e4-4662-9574-1deb8a73f724

# MARCH 2021, Rene Smit (@rcsmit) - MIT license

# Fields in
# Date_file;Date_statistics;Date_statistics_type;Agegroup;Sex;
# Province;Hospital_admission;Deceased;Week_of_death;Municipal_health_service

# Fields out
# pos_test_Date_statistics,pos_test_0-9,pos_test_10-19,pos_test_20-29,pos_test_30-39,
# pos_test_40-49,pos_test_50-59,pos_test_60-69,pos_test_70-79,pos_test_80-89,pos_test_90+,
# pos_test_<50,pos_test_Unknown,hosp_Date_statistics,hosp_0-9,hosp_10-19,hosp_20-29,hosp_30-39,
# hosp_40-49,hosp_50-59,hosp_60-69,hosp_70-79,hosp_80-89,hosp_90+,hosp_<50,hosp_Unknown,
# deceased_Date_statistics,deceased_0-9,deceased_10-19,deceased_20-29,deceased_30-39,
# deceased_40-49,deceased_50-59,deceased_60-69,deceased_70-79,deceased_80-89,deceased_90+,
# deceased_<50,deceased_Unknown


import pandas as pd
import numpy as np
import datetime
import datetime as dt
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
_lock = RendererAgg.lock
from streamlit import caching


def save_df(df, name):
    """  save dataframe on harddisk """
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\"
    )

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



def make_age_graph(df, d, columns_original,  titel):
    if d is None:
        st.warning("Choose ages to show")
        st.stop()
    with _lock:
        color_list = [    "#3e5c76",  # blue 6,
                        "#ff6666",  # reddish 0
                        "#ac80a0",  # purple 1
                        "#3fa34d",  # green 2
                        "#EAD94C",  # yellow 3
                        "#EFA00B",  # orange 4
                        "#7b2d26",  # red 5
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


        # df = agg_ages(df)
        fig1y, ax = plt.subplots()
        for i, d_ in enumerate(d):
            ax.plot(df["Date_statistics"], df[d_], color = color_list[i+1], label = columns_original[i])
            # #if d_ == "TOTAAL_index":
            # if d_[:6] == "TOTAAL":
            #     ax.plot(df["Date_of_statistics_week_start"], df[d_], color = color_list[0], label = columns_original[i], linestyle="--", linewidth=2)
            #     #ax.plot(df["Date_of_statistics_week_start"], df[columns_original[i]], color = color_list[0], alpha =0.5, linestyle="dotted", label = '_nolegend_',  linewidth=2)
            # else:
            #     ax.plot(df["Date_of_statistics_week_start"], df[d_], color = color_list[i+1], label = columns_original[i])
            #     #ax.plot(df["Date_of_statistics_week_start"], df[columns_original[i]], color = color_list[i+1], alpha =0.5, linestyle="dotted", label = '_nolegend_' )
        plt.legend()

        titel_ = titel + " "
        plt.title(titel_)
        plt.xticks(rotation=270)
        plt.axhline(y=1, color="yellow", alpha=0.6, linestyle="--")

        ax.text(
        1,
        1.1,
        "Created by Rene Smit — @rcsmit",
        transform=ax.transAxes,
        fontsize="xx-small",
        va="top",
        ha="right",
    )
        # plt.tight_layout()
        # plt.show()
        st.pyplot(fig1y)


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



def add_walking_r(df, smoothed_columns, how_to_smooth, tg,d):
    """  _ _ _ """
    # Calculate walking R from a certain base. Included a second methode to calculate R
    # de rekenstappen: (1) n=lopend gemiddelde over 7 dagen; (2) Rt=exp(Tc*d(ln(n))/dt)
    # met Tc=4 dagen, (3) data opschuiven met rapportagevertraging (10 d) + vertraging
    # lopend gemiddelde (3 d).
    # https://twitter.com/hk_nien/status/1320671955796844546
    # https://twitter.com/hk_nien/status/1364943234934390792/photo/1
    column_list_r_smoothened = []
    for base in smoothed_columns:
        column_name_R = "R_value_from_" + base + "_tg" + str(tg)

        column_name_r_smoothened = (
            "R_value_from_"
            + base
            + "_tg"
            + str(tg)
            + "_"
            + "over_" + str(d) + "_days_"
            + how_to_smooth
            + "_"
            + str(WDW2)
        )

        sliding_r_df = pd.DataFrame(
            {"date_sR": [], column_name_R: []}
        )


        for i in range(len(df)):
            if df.iloc[i][base] != None:
                date_ = pd.to_datetime(df.iloc[i]["Date_statistics"], format="%Y-%m-%d")
                date_ = df.iloc[i]["Date_statistics"]
                if df.iloc[i - d][base] != 0 or df.iloc[i - d][base] is not None:
                    slidingR_ = round(
                        ((df.iloc[i][base] / df.iloc[i - d][base]) ** (tg / d)), 2
                    )

                else:
                    slidingR_ = None

                sliding_r_df = sliding_r_df.append(
                    {
                        "date_sR": date_,
                        column_name_R: slidingR_

                    },
                    ignore_index=True,
                )

        sliding_r_df[column_name_r_smoothened] = round(
            sliding_r_df.iloc[:, 1].rolling(window=WDW2, center=True).mean(), 2
        )


        sliding_r_df = sliding_r_df.reset_index()
        df = pd.merge(
            df,
            sliding_r_df,
            how="outer",
            left_on="Date_statistics",
            right_on="date_sR",
            #left_index=True,
        )
        column_list_r_smoothened.append(column_name_r_smoothened)


        sliding_r_df = sliding_r_df.reset_index()
    return df, column_list_r_smoothened

def select_period(df, field, show_from, show_until):
    """ _ _ _ """
    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"

    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df

def main():
    # online version : https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv
    url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"

    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
    else:
        url1= "https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv"

    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")
    df = df.groupby(["Date_statistics", "Agegroup"], sort=True).count().reset_index()


    start_ = "2021-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    global from_, FROM, UNTIL
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is valid and/or in format yyyy-mm-dd")
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

    df.rename(
        columns={
            "Date_file": "count",
        },
        inplace=True,
    )
    df_hospital = df[df["Hospital_admission"] == "Yes"].copy(deep=False)
    df_deceased = df[df["Deceased"] == "Yes"].copy(deep=False)
    df = select_period(df,"Date_statistics", FROM, UNTIL)
    df_pivot = (
        pd.pivot_table(
            df,
            values="count",
            index=["Date_statistics"],
            columns=["Agegroup"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )

    # df_pivot_hospital = (
    #     pd.pivot_table(
    #         df_hospital,
    #         values="count",
    #         index=["Date_statistics"],
    #         columns=["Agegroup"],
    #         aggfunc=np.sum,
    #     )
    #     .reset_index()
    #     .copy(deep=False)
    # )

    # df_pivot_deceased = (
    #     pd.pivot_table(
    #         df_deceased,
    #         values="count",
    #         index=["Date_statistics"],
    #         columns=["Agegroup"],
    #         aggfunc=np.sum,
    #     )
    #     .reset_index()
    #     .copy(deep=False)
    # )

    #df_pivot = df_pivot.add_prefix("pos_test_")
    # df_pivot_hospital = df_pivot_hospital.add_prefix("hosp_")
    # save_df(df_pivot_hospital, "df_hospital_per_dag")
    # df_pivot_deceased = df_pivot_deceased.add_prefix("deceased_")
    # print(df_pivot_deceased.dtypes)
    todrop = [
        "Date_statistics_type",
        "Sex",
        "Province",
        "Hospital_admission",
        "Deceased",
        "Week_of_death",
        "Municipal_health_service",
    ]
    df = drop_columns(df, todrop)

    # save_df(df, "landelijk_leeftijd_2")

    # save_df(df_pivot, "landelijk_leeftijd_pivot")
    #save_df(df_pivot_hospital, "landelijk_leeftijd_pivot_hospital")
    #save_df(df_pivot_deceased, "landelijk_leeftijd_pivot_deceased")

    # df_temp = pd.merge(
    #     df_pivot,
    #     df_pivot_hospital,
    #     how="outer",
    #     left_on="pos_test_Date_statistics",
    #     right_on="hosp_Date_statistics",
    # )
    # df_temp = pd.merge(
    #     df_temp,
    #     df_pivot_deceased,
    #     how="outer",
    #     left_on="pos_test_Date_statistics",
    #     right_on="deceased_Date_statistics",
    # )

    #df_temp_per_week = df_temp.groupby(pd.Grouper(key='pos_test_Date_statistics', freq='W')).sum()
    #df_temp_per_week.index -= pd.Timedelta(days=6)
    #print(df_temp_per_week)
    #df_temp_per_week["weekstart"]= df_temp_per_week.index
    #save_df(df_temp, "final_result")
    #save_df(df_temp_per_week, "final_result_per_week")

    lijst = ["0-9", "10-19" , "20-29" , "30-39" , "40-49" , "50-59" , "60-69" , "70-79" , "80-89" , "90+"]
    ages_to_show = st.sidebar.multiselect(
                "Ages to show (multiple possible)", lijst, lijst)
    global WDW2
    df = df_pivot.copy(deep=False)
    t = "SMA"
    tg = st.sidebar.slider("Generation time", 1, 7, 4)
    d =  st.sidebar.slider("Look back how many days", 1, 14, 7)
    WDW2 = st.sidebar.slider("Window smoothing curves (days)", 1, 45, 7)
    centersmooth =  st.sidebar.selectbox(
        "Smooth in center", [True, False], index=1)
    df, smoothed_columns = smooth_columnlist(df, ages_to_show, t, WDW2, centersmooth)
    df, column_list_r_smoothened = add_walking_r(df, smoothed_columns, t, tg,d)
    make_age_graph(df,  column_list_r_smoothened, lijst,  "R getal naar leeftijd")
    st.write("Attentie: DIt is het R getal op basis van moment van rapportage. RIVM berekent het R getal over het moment van besmetting of eerste symptomen")


main()
