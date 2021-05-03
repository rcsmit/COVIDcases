# Make a stackplot and a stackplot where total = 100% of agegroups OR
# Make a lineplot or a lineplot where the start = 100

# René Smit (@rcsmit) - MIT Licence

# IN: https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv
# OUT : Line-/Stackplots
#
# TODO : Make everything a function call
#        Integration in the dashboard
#        Make an index compared to [total reported], [total hospital admissions] or [total ICU admissions]
#
# Inspired by a graph by @chivotweets
# https://twitter.com/rubenivangaalen/status/1374443261704605697

import datetime
import datetime as dt
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
_lock = RendererAgg.lock
from streamlit import caching

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

def make_age_graph(df, d, columns_original, legendanames, titel):
    if d is None:
        st.warning("Choose ages to show")
        st.stop()
    with _lock:
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


        # df = agg_ages(df)
        fig1y, ax = plt.subplots()
        for i, d_ in enumerate(d):

            if d_ == "TOTAAL_index":
                ax.plot(df["Date_of_statistics_week_start"], df[d_], color = color_list[i], label = columns_original[i],  linewidth=2)
                ax.plot(df["Date_of_statistics_week_start"], df[columns_original[i]], color = color_list[i], alpha =0.5, linestyle="dotted", label = '_nolegend_',  linewidth=2)
            else:
                ax.plot(df["Date_of_statistics_week_start"], df[d_], color = color_list[i], label = columns_original[i])
                ax.plot(df["Date_of_statistics_week_start"], df[columns_original[i]], color = color_list[i], alpha =0.5, linestyle="dotted", label = '_nolegend_' )
        plt.legend()
        titel_ = titel + " (weekcijfers)"
        plt.title(titel_)
        plt.xticks(rotation=270)

        # plt.tight_layout()
        # plt.show()
        st.pyplot(fig1y)
def show_age_graph (df,d, titel):
    df, columnlist_df, columnlist_sma_df, columnlist_sma, columnlist_ages_legenda, columnlist_original = smooth(df, d)
    make_age_graph(df,  columnlist_sma, columnlist_original, columnlist_ages_legenda, titel)

def make_stack_graph(df, columns_df,columnlist_names, columnlist_ages, datumveld, titel):
    if columnlist_ages is None:
        st.warning("Choose ages to show")
        st.stop()
    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)

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



        sp = ax.stackplot(datumlijst, columns_df, colors=color_list)
        #ax.legend(loc="upper left")
        plt.title(titel)

        proxy = [mpl.patches.Rectangle((0,0), 0,0, facecolor=pol.get_facecolor()[0]) for pol in sp]
        ax.legend(proxy, tuple (columnlist_ages),  bbox_to_anchor=(1.3, 1),loc="best")
        plt.xticks(rotation=270)
        #plt.tight_layout()
        #plt.show()
        st.pyplot(fig1x)

def show_stack(df, c1,titel,absolute_or_relative):


    datumveld = "Date_of_statistics_week_start"
    df, columnlist_df, columnlist_sma_df, columnlist_names, columnlist_ages, columnlist_original = smooth(df, c1)

    titel = titel + " (weekcijfers)"

    if absolute_or_relative == "absolute":
        make_stack_graph (df,       columnlist_df,     columnlist_sma_df, columnlist_names, datumveld, titel)
    else:
        df, columnlist_hdred_names, columnlist_hdred_df, columnlist_ages = hundred_stack_area(df, columnlist_names)
        make_stack_graph (df, columnlist_hdred_df,columnlist_names,  columnlist_ages , datumveld, titel)

def agg_ages(df):
    # make age groups
    df["0-29"] = df["0-14"] + df["15-19"] + df["20-24"] + df["25-29"]
    df["30-49"] = df["30-34"] + df["35-39"] + df["40-44"] + df["45-49"]
    df["50-69"] = df["50-54"] + df["55-59"] + df["60-64"] + df["65-69"]
    df["70-89"] = df["70-74"] + df["75-79"] +  df["80-84"] + df["85-89"]
    df["TOTAAL"] = df["0-14"] + df["15-19"] + df["20-24"] + df["25-29"] + df["30-34"] + df["35-39"] + df["40-44"] + df["45-49"] + df["50-54"] + df["55-59"] + df["60-64"] + df["65-69"] + df["70-74"] + df["75-79"] +  df["80-84"] + df["85-89"]
    return df

@st.cache(ttl=60 * 60 * 24)
def load_data():
    url1 = "https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv"
    df = pd.read_csv(url1, delimiter=";", low_memory=False)

    return df

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
    if delete_last_row == True:
        df_pivot_hospital = df_pivot_hospital[:-1]
        df_pivot_ic = df_pivot_ic[:-1]
    return df_pivot_hospital, df_pivot_ic

def normeren(df, what_to_norm):
    """In : columlijst
    Bewerking : max = 1
    Out : columlijst met genormeerde kolommen"""
    # print(df.dtypes)

    normed_columns = []
    how_to_norm = "first"
    for column in what_to_norm:
        #maxvalue = (df[column].max()) / 100
        firstvalue = df[column].iloc[0] / 100
        name = f"{column}_index"

        for i in range(0, len(df)):
            if how_to_norm == "max":
                df.loc[i, name] = df.loc[i, column] / maxvalue
            else:
                df.loc[i, name] = df.loc[i, column] / firstvalue
        normed_columns.append(name)
        #print(f"{name} generated")
        #print (df)
    return df, normed_columns

def select_period(df, show_from, show_until):
    """ _ _ _ """
    if show_from == None:
        show_from = "2021-1-1"

    if show_until == None:
        show_until = "2030-1-1"

    mask = (df["Date_of_statistics_week_start"].dt.date >= show_from) & (df["Date_of_statistics_week_start"].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df

def main():


    lijst  = ["0-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+", "Unknown", "0-29","30-49","50-69","70-89","90+", "TOTAAL"]





    st.header("Hospital / ICU admissions in the Netherlands")
    st.subheader("Please send feedback to @rcsmit")

    # DAILY STATISTICS ################

    start_ = "2021-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
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
    delete_last_row =  st.sidebar.selectbox("Delete last row", [True, False], index=0)

    df_pivot_hospital, df_pivot_ic  = prepare_data()
    df_pivot_hospital = agg_ages(df_pivot_hospital)
    df_pivot_ic = agg_ages(df_pivot_ic)

    df_pivot_hospital = select_period(df_pivot_hospital, FROM, UNTIL)
    df_pivot_ic = select_period(df_pivot_ic, FROM, UNTIL)
    hospital_or_ic = st.sidebar.selectbox("Hospital or IC", ["hospital", "icu"], index=0)
    what_to_do = st.sidebar.selectbox("What type of graph", ["stack", "line"], index=0)

    if what_to_do == "line":
        age_groups = ["0-29","30-49","50-69","70-89","90+", "TOTAAL"]
        absolute_or_index = st.sidebar.selectbox("Absolute or index (start = 100)", ["absolute", "index"], index=0)

        if absolute_or_index == "index":
            normed = True
        else:
            normed = False
    else:
        absolute_or_relative = st.sidebar.selectbox("Absolute or relative (total = 100%)", ["absolute", "relative"], index=0)

        age_groups = ["0-29","30-49","50-69","70-89","90+"]

    ages_to_show = st.sidebar.multiselect(
            "Ages to show (multiple possible)", lijst, age_groups)

    if len(ages_to_show) == 0:
        st.warning("Choose ages to show")
        st.stop()


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

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Data source :  <a href="https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv" target="_blank">RIVM</a><br>'
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
