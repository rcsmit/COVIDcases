# Bereken het percentuele verschil tov een x-aantal dagen ervoor
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sn
import platform
import datetime
import datetime as dt
import streamlit as st
from streamlit import caching
from helpers import *  # cell_background, select_period, save_df, drop_columns
from datetime import datetime

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


def week_to_week(df, column_):
    column_ = column_ if type(column_) == list else [column_]
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

@st.cache(ttl=60 * 60 * 24)
def get_data_casus_landelijk():
    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
        url1 = "C:/Users/rcxsm/Documents/phyton_scripts/covid19_seir_models/input/COVID-19_casus_landelijk.csv"
    else:
        url1= "https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv"

    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")
    df = df.groupby(["Date_statistics", "Agegroup"], sort=True).count().reset_index()
    return df

def select_period(df, field, show_from, show_until):
    """Shows two inputfields (from/until and Select a period in a df (helpers.py).

    Args:
        df (df): dataframe
        field (string): Field containing the date

    Returns:
        df: filtered dataframe
    """

    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df

def calculate_fraction(df):
    nr_of_columns = len (df.columns)
    nr_of_rows = len(df)
    column_list = df.columns.tolist()
    max_waarde = 0
    data = []
    waardes = []
     #              0-9     10-19     20-29    30-39   40-49   50-59   60-69      70-79    80-89    90+
    pop_ =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 709000, 130000]  # tot 17 464 000
    fraction =  [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.0405978, 0.0074438846]

    for r in range(nr_of_rows):
            row_data = []
            for c in range(0,nr_of_columns):
                if c==0 :
                    row_data.append( df.iat[r,c])
                else:
                #try
                    waarde = df.iat[r,c]/pop_[c-1] * 100_000
                    row_data.append(waarde)
                    waardes.append(waarde)
                    if waarde > max_waarde:
                        max_waarde = waarde

                #except:

                    # date
                    # row_data.append( df.iat[r,c])
                 #   pass
            data.append(row_data)


    df_fractie = pd.DataFrame(data, columns=column_list)
    top_waarde = 0.975*max_waarde
    return df_fractie, top_waarde









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
        new_column = c + "_SMA"
        #new_column = c

        # print("Generating " + new_column + "...")
        df[new_column] = (
            df.iloc[:, df.columns.get_loc(c)].rolling(window=WDW2, center=True).mean()
        )
        columnlist_sma_df.append(df[new_column])
        columnlist_df.append(df[c])
        columnlist_names.append(new_column)
        columnlist_ages.append(c)           # alleen de leeftijden, voor de legenda

    return df,columnlist_df, columnlist_sma_df,columnlist_names,columnlist_ages, columnlist

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

def make_age_graph(df, d, columns_original, legendanames, titel):
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

            #if d_ == "TOTAAL_index":
            if d_[:6] == "TOTAAL":
                ax.plot(df["Date_of_statistics_week_start"], df[d_], color = color_list[0], label = columns_original[i], linestyle="--", linewidth=2)
                ax.plot(df["Date_of_statistics_week_start"], df[columns_original[i]], color = color_list[0], alpha =0.5, linestyle="dotted", label = '_nolegend_',  linewidth=2)
            else:
                ax.plot(df["Date_of_statistics_week_start"], df[d_], color = color_list[i+1], label = columns_original[i])
                ax.plot(df["Date_of_statistics_week_start"], df[columns_original[i]], color = color_list[i+1], alpha =0.5, linestyle="dotted", label = '_nolegend_' )
        plt.legend()
        if y_zero == True:
            ax.set_ylim(bottom = 0)
        titel_ = titel + " (weekcijfers)"
        plt.title(titel_)
        plt.xticks(rotation=270)

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
def show_age_graph (df,d, titel):
    df, columnlist_df, columnlist_sma_df, columnlist_sma, columnlist_ages_legenda, columnlist_original = smooth(df, d)
    make_age_graph(df,  columnlist_sma, columnlist_original, columnlist_ages_legenda, titel)


def agg_ages(df):
    # make age groups
    df["0-29"] = df["0-14"] + df["15-19"] + df["20-24"] + df["25-29"]
    df["30-49"] = df["30-34"] + df["35-39"] + df["40-44"] + df["45-49"]
    df["50-69"] = df["50-54"] + df["55-59"] + df["60-64"] + df["65-69"]
    df["70-89"] = df["70-74"] + df["75-79"] +  df["80-84"] + df["85-89"]

    # # extra groep
    df["30-69"] = df["30-34"] + df["35-39"] + df["40-44"] + df["45-49"] + df["50-54"] + df["55-59"] + df["60-64"] + df["65-69"]

    # # indeling RIVM
    df["0-39"] = df["0-14"] + df["15-19"] + df["20-24"] + df["25-29"] + df["30-34"] + df["35-39"]
    df["40-59"] =  df["40-44"] + df["45-49"] + df["50-54"] + df["55-59"]
    df["60-79"] =  df["60-64"] + df["65-69"] +  df["70-74"] + df["75-79"]
    df["80+"] =  df["80-84"] + df["85-89"] + df["90+"]

    # CORRESPONDEREND MET CASUS LANDELIJK
     # indeling RIVM
    df["0-19"] = df["0-14"] + df["15-19"]
    df["20-29"] = df["20-24"] + df["25-29"]
    df["30-39"] = df["30-34"] + df["35-39"]
    df["40-49"] = df["40-44"] + df["45-49"]
    df["50-59"] = df["50-54"] + df["55-59"]
    df["60-69"] = df["60-64"] + df["65-69"]
    df["70-79"] = df["70-74"] + df["75-79"]
    df["80-89"] = df["80-84"] + df["85-89"]



    df["TOTAAL"] = df["0-14"] + df["15-19"] + df["20-24"] + df["25-29"] + df["30-34"] + df["35-39"] + df["40-44"] + df["45-49"] + df["50-54"] + df["55-59"] + df["60-64"] + df["65-69"] + df["70-74"] + df["75-79"] +  df["80-84"] + df["85-89"] +  df["90+"]+ df["Unknown"]


    return df

@st.cache(ttl=60 * 60 * 24)
def load_data():
    url1 = "https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv"
    return pd.read_csv(url1, delimiter=";", low_memory=False)

def prepare_data():
    #url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv"
    df_getdata = load_data()
    df = df_getdata.copy(deep=False)  # prevent an error [Return value of `prepare_data()` was mutated between runs.]


    datumveld = "Date_of_statistics_week_start"
    df[datumveld] = pd.to_datetime(df[datumveld], format="%Y-%m-%d")

    df = df.reset_index()
    df.fillna(value=0, inplace=True)

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

    save_df(df_pivot_hospital,"df_pivot_hospital")
    save_df(df_pivot_ic,"df_pivot_ic")
    if delete_last_row == True:
        df_pivot_hospital = df_pivot_hospital[:-1]
        df_pivot_ic = df_pivot_ic[:-1]
    return df_pivot_hospital, df_pivot_ic




def make_pivot_casus_landelijk_per_week():


    df_getdata_casus_landelijk_ = get_data_casus_landelijk()
    df_casus_landelijk = df_getdata_casus_landelijk_.copy(deep=False)

    #df_casus_landelijk = select_period(df_casus_landelijk, "Date_of_statistics_week_start", FROM, UNTIL)
    df_casus_landelijk.rename(
        columns={
            "Date_file": "count",
        },
        inplace=True,
    )

    df_pivot_casus_landelijk = (
        pd.pivot_table(
            df_casus_landelijk,
            values="count",
            index=["Date_statistics"],
            columns=["Agegroup"],
            aggfunc=np.sum,
        )
        .reset_index()
        .copy(deep=False)
    )
    # option to drop agegroup 0-9 due to changes in testbeleid en -bereidheid
    #st.write(df_pivot_casus_landelijk.dtypes)
    df_pivot_casus_landelijk = df_pivot_casus_landelijk.drop(columns=["<50"], axis=1)
    try:
        df_pivot_casus_landelijk = df_pivot_casus_landelijk.drop(columns=["Unknown"], axis=1)
    except:
        pass
    df_pivot_casus_landelijk=df_pivot_casus_landelijk.fillna(0)
    drop_0_9  = st.sidebar.selectbox("Delete agegroup 0-9", [True, False], index=1)
    if drop_0_9 == True:
        df_pivot_casus_landelijk = df_pivot_casus_landelijk.drop(columns="0-9", axis=1)
    df_pivot_casus_landelijk_original = df_pivot_casus_landelijk.copy(deep=False)

    #df_pivot_casus_landelijk = df_pivot_casus_landelijk.add_prefix("pos_test_")
    todrop = [
        "Date_statistics_type",
        "Sex",
        "Province",
        "Hospital_admission",
        "Deceased",
        "Week_of_death",
        "Municipal_health_service",
    ]

    #df_casus_landelijk = drop_columns(df_casus_landelijk, todrop)
    column_list = df_pivot_casus_landelijk.columns.tolist()
    #df_pivot_casus_landelijk['Date_statistics'] = df_pivot_casus_landelijk['Date_statistics'].dt.date
    #df_pivot_casus_landelijk['date'] = df_pivot_casus_landelijk['Date_statistics']
    df_pivot_casus_landelijk.rename(columns={"Date_statistics": "date"},  inplace=True)
    column_list = column_list[1:]
    st.write (df_pivot_casus_landelijk)
    df_pivot_casus_landelijk_per_week = df_pivot_casus_landelijk.groupby(pd.Grouper(key='date', freq='W')).sum()
    df_pivot_casus_landelijk_per_week.index -= pd.Timedelta(days=6)
    df_pivot_casus_landelijk_per_week["weekstart"]= df_pivot_casus_landelijk_per_week.index
    #df_pivot_casus_landelijk_per_week["weekstart"] = df_pivot_casus_landelijk['weekstart'].dt.date
    st.write(df_pivot_casus_landelijk_per_week)

    #df_pivot_casus_landelijk_per_week["0-19"]= df_pivot_casus_landelijk_per_week["0-9"] + df_pivot_casus_landelijk_per_week["10-19"]

    return df_pivot_casus_landelijk_per_week






def main():


    lijst  = ["0-14", "15-19", "20-24", "25-29", "30-34",
             "35-39", "40-44", "45-49", "50-54", "55-59",
             "60-64", "65-69", "70-74", "75-79", "80-84",
             "85-89", "90+", "Unknown",
             "0-29","30-49","50-69","70-89","90+",
             "30-69", "0-39", "40-59", "60-79", "80+",
             "0-19","20-29","30-39","40-49",
             "50-59", "60-69",  "70-79","80-89","90+",

             "TOTAAL"]
    population = [2707000,1029000,1111000,1134000,1124000,
                  1052000,1033000,1131000,1285000,1263000,

                  1138000,1003000,971000,644000,450000,
                  259000,130000,10,
                  5981000,4340000,4689000,2_324_000,130000,
                  9029000,8157000,4712000,3756000,839000,

    #              0-9     10-19     20-29    30-39   40-49
                  1756000, 1980000, 2245000, 2176000, 2164000,
                   #50-59   60-69      70-79    80-89    90+
                   2548000, 2141000, 1615000, 709000, 130000, 17464000]  # tot 17 464 000


    st.header("Hospital / ICU admissions in the Netherlands")
    st.subheader("Please send feedback to @rcsmit")

    # DAILY STATISTICS ################

    start_ = "2020-01-01"
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
    global WDW2
    WDW2 = st.sidebar.slider("Window smoothing curves (weeks)", 1, 8, 1)
    global delete_last_row
    delete_last_row =  st.sidebar.selectbox("Delete last week/row of complete dataset", [True, False], index=0)

    df_pivot_hospital, df_pivot_ic  = prepare_data()


    df_pivot_hospital = select_period(df_pivot_hospital, "Date_of_statistics_week_start", FROM, UNTIL)
    df_pivot_ic = select_period(df_pivot_ic, "Date_of_statistics_week_start", FROM, UNTIL)

    df_pivot_hospital_basic = df_pivot_hospital.copy(deep=False)
    df_pivot_ic_basic =  df_pivot_ic.copy(deep=False)


    df_pivot_hospital = agg_ages(df_pivot_hospital)
    df_pivot_ic = agg_ages(df_pivot_ic)

    save_df(df_pivot_hospital, "hospital_voor_maarten")
    save_df(df_pivot_ic, "ic_voor_maarten")

    df_pivot_casus_landelijk_per_week = make_pivot_casus_landelijk_per_week()
    save_df(df_pivot_casus_landelijk_per_week, "casus_per_age_per_week_voor_maarten")

    hospital_or_ic = st.sidebar.selectbox("Hospital or IC", ["hospital", "icu"], index=0)
    what_to_do = st.sidebar.selectbox("What type of graph", ["stack", "line"], index=1)

    default_age_groups = ["0-29","30-49","50-69","70-89","90+"]
    default_age_groups_perc = ["0-29_perc","30-49_perc","50-69_perc","70-89_perc","90+_perc"]
    default_age_groups_cumm_all = ["0-29_cumm_all","30-49_cumm_all","50-69_cumm_all","70-89_cumm_all","90+_cumm_all"]
    default_age_groups_cumm_period = ["0-29_cumm_period","30-49_cumm_period","50-69_cumm_period","70-89_cumm_period","90+_cumm_period"]
    default_age_groups_per_capita = ["0-29_per_capita","30-49_per_capita","50-69_per_capita","70-89_per_capita","90+_per_capita"]
    if what_to_do == "line":

        age_groups = ["0-29","30-49","50-69","70-89","90+", "TOTAAL"]
        absolute_or_index = st.sidebar.selectbox(f"Absolute | percentages of TOTAAL |\n index (start = 100) | per capita | cummulatief from 2020-1-1 | cummulatief from {FROM}", ["absolute",  "percentages", "index",  "per_capita", "cummulatief_all", "cummulatief_period"], index=0)

        normed = absolute_or_index == "index"
        if absolute_or_index  == "percentages":
            ages_to_show = st.sidebar.multiselect(
                "Ages to show (multiple possible)", lijst_perc, default_age_groups_perc)
        elif  absolute_or_index  == "cummulatief_all":
            ages_to_show = st.sidebar.multiselect(
                "Ages to show (multiple possible)", lijst_cumm_all, default_age_groups_cumm_all)
        elif  absolute_or_index  == "cummulatief_period":
            ages_to_show = st.sidebar.multiselect(
                "Ages to show (multiple possible)", lijst_cumm_period, default_age_groups_cumm_period)
        elif  absolute_or_index  == "per_capita":
            ages_to_show = st.sidebar.multiselect(
                "Ages to show (multiple possible)", lijst_per_capita, default_age_groups_per_capita)
        else:
            # absolute
            ages_to_show = st.sidebar.multiselect(
                "Ages to show (multiple possible)", lijst, default_age_groups)
    else:
        #stackplot
        absolute_or_relative = st.sidebar.selectbox("Absolute or relative (total = 100%)", ["absolute", "relative"], index=0)
        ages_to_show = st.sidebar.multiselect(
                "Ages to show (multiple possible)", lijst, default_age_groups)




    if len(ages_to_show) == 0:
        st.warning("Choose ages to show")
        st.stop()
    global y_zero
    y_zero =  st.sidebar.selectbox("Y-ax starts at 0", [True, False], index=1)


    if what_to_do == "stack":

        #  SHOW STACKGRAPHS
        if hospital_or_ic == "hospital":

            to_do_stack = [[df_pivot_hospital, ages_to_show, "ziekenhuisopname naar leeftijd"]]
        else:
            to_do_stack = [[df_pivot_ic, ages_to_show, "IC opname naar leeftijd"]]

        for d in to_do_stack:
            show_stack (d[0],d[1],d[2], absolute_or_relative)

    elif what_to_do == "line":
        # SHOW LINEGRAPHS
        if normed == True:
            df_pivot_hospital, d = normeren(df_pivot_hospital, ages_to_show)
            df_pivot_ic, d = normeren(df_pivot_ic, ages_to_show)
        else:
            d = ages_to_show
        if hospital_or_ic == "hospital":
            show_age_graph(df_pivot_hospital, d, "ziekenhuisopnames")
        else:
            show_age_graph(df_pivot_ic, d, "IC opnames")
    else:
        st.error ("ERROR")
        st.stop

    if hospital_or_ic == "hospital":
            st.subheader("Ziekenhuisopnames (aantallen)")

            st.write (df_pivot_hospital_basic)

            df_new = do_the_rudi(df_pivot_hospital_basic)


            st.write(df_new.style.format(None, na_rep="-").applymap(color_value).set_precision(2))

            #st.dataframe(df_new.style.applymap(color_value))
    else:
            st.subheader("Ziekenhuisopnames (aantallen)")
            st.write(df_pivot_ic_basic)
            df_new = do_the_rudi(df_pivot_ic_basic)
            st.dataframe(df_new.style.applymap(color_value))

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Data source :  <a href="https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv" target="_blank">RIVM</a> (daily retrieved)<br>'
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/edit/main/plot_hosp_ic_streamlit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
    )


    st.sidebar.markdown(tekst, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.image(
        "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png"
    )

    st.markdown(
        '<a href="https://www.buymeacoffee.com/rcsmit" target="_blank">If you are happy with this dashboard, you can buy me a coffee</a>',
        unsafe_allow_html=True,
    )
if __name__ == "__main__":
    main()

