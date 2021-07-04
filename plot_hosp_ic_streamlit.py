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
    """[sla df op]

    Args:
        df ([dataframe]): [df-naam]
        name ([filename]): [bestandsnaam]
    """

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


def make_age_graph_per_total_reported(df,  d, titel):
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
                ax.plot(df["weekstart"], df[d_], color = color_list[0], label = d[i], linestyle="--", linewidth=2)
                ax.plot(df["weekstart"], df[d[i]], color = color_list[0], alpha =0.5, linestyle="dotted", label = '_nolegend_',  linewidth=2)
            else:
                ax.plot(df["weekstart"], df[d_], color = color_list[i+1], label = d[i])
                ax.plot(df["weekstart"], df[d[i]], color = color_list[i+1], alpha =0.5, linestyle="dotted", label = '_nolegend_' )
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

    # extra groep
    df["30-69"] = df["30-34"] + df["35-39"] + df["40-44"] + df["45-49"] + df["50-54"] + df["55-59"] + df["60-64"] + df["65-69"]

    # indeling RIVM
    df["0-39"] = df["0-14"] + df["15-19"] + df["20-24"] + df["25-29"] + df["30-34"] + df["35-39"]
    df["40-59"] =  df["40-44"] + df["45-49"] + df["50-54"] + df["55-59"]
    df["60-79"] =  df["60-64"] + df["65-69"] +  df["70-74"] + df["75-79"]
    df["80+"] =  df["80-84"] + df["85-89"] + df["90+"]

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

def get_data_per_total_reported():

    import pandas as pd
    sheet_id = "1trUoOPbDjBo8Q8XKg7BuJnawVDPvapnhuJjBfD6ehG0"
    sheet_name_hosp = "hospital/casus*100"
    sheet_name_IC = "IC/casus*100"
    url_hosp = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_hosp}"
    url_IC = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_IC}"
    df_hosp_per_cases = pd.read_csv(url_hosp)

    df_IC_per_cases = pd.read_csv(url_IC)
    df_hosp_per_cases["weekstart"] = pd.to_datetime(df_hosp_per_cases["weekstart"], format="%d-%m-%Y")

    df_IC_per_cases["weekstart"] = pd.to_datetime(df_IC_per_cases["weekstart"], format="%d-%m-%Y")

    df_hosp_per_cases = select_period(df_hosp_per_cases, "weekstart", FROM, UNTIL)
    df_IC_per_cases = select_period(df_IC_per_cases, "weekstart", FROM, UNTIL)

    columns = [ "0-19" , "20-29" , "30-39" , "40-49" , "50-59" , "60-69" , "70-79" , "80-89" , "90+"]
    for c in columns:
        df_hosp_per_cases[c] = [str(val).replace(',', '.') for val in df_hosp_per_cases[c]]
        df_hosp_per_cases[c] = df_hosp_per_cases[c].astype(float)
        df_IC_per_cases[c] = [str(val).replace(',', '.') for val in df_IC_per_cases[c]]
        df_IC_per_cases[c] = df_IC_per_cases[c].astype(float)

    return df_hosp_per_cases, df_IC_per_cases
def normeren(df, what_to_norm):
    """In : columlijst
    Bewerking : max = 1
    Out : columlijst met genormeerde kolommen"""
    # print(df.dtypes)

    normed_columns = []
    how_to_norm = "first"
    for column in what_to_norm:
        #maxvalue = (df[column].max()) / 100
        if df[column].iloc[0] == 0:
            st.error(f"The agegroup [{column}] has a value of zero at {from_}, so I can't calculate an index.\n\nPlease change the startdate and/or agegroup(s)")
            st.stop()
        else:
            firstvalue = df[column].iloc[0] / 100
            name = f"{column}_index"

            for i in range(len(df)):
                if how_to_norm == "max":
                    df.loc[i, name] = df.loc[i, column] / maxvalue
                else:

                    try:
                        df.loc[i, name] = df.loc[i, column] / firstvalue
                    except:
                        df.loc[i, name] = 0
                        st.alert("dividebyzero")
            normed_columns.append(name)

            #print(f"{name} generated")
            #print (df)
    return df, normed_columns

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
def calculate_percentages(df, lijst):
    lijst_perc = []

    for d in lijst:
        new_name = d + "_perc"
        lijst_perc.append(new_name)
        df[new_name] = round((df[d] / df["TOTAAL"] * 100),2)
    return df, lijst_perc

def calculate_cumm(df, lijst, all_or_period):
    """ Calculate walking cummulative """
    cumlist = []
    for l in lijst:
        name = l + "_cumm_all" if all_or_period == "all" else l + "_cumm_period"
        df[name] = df[l].cumsum()
        cumlist.append(name)
    return df, cumlist

def calculate_per_capita(df, lijst, population):
    capitalist = []
    for i,l in enumerate(lijst):
        name = l + "_per_capita"
        df[name] = df[l] / population[i]
        capitalist.append(name)
    return df, capitalist

def color_value(val):
    try:
        v = abs(val)
        opacity = 1 if v >100 else v/100
        # color = 'green' if val >0 else 'red'
        if val > 0 :
             color = '255, 0, 0'
        elif val < 0:
            color = '76, 175, 80'
        else:
            color = '255,255,255'
    except:
        # give cells with eg. text or dates a white background
        color = '255,255,255'
        opacity = 1
    return f'background: rgba({color}, {opacity})'



def do_the_rudi(df):
    """Calculate the fractions per age group. Calculate the difference related to day 0 as a % of day 0.
    Made for Rudi Lousberg
    Inspired by Ian Denton https://twitter.com/IanDenton12/status/1407379429526052866

    Args:
        df (df): table with numbers

    Returns:
        df : table with the percentual change of the fracctions
    """
    # calculate the sum
    df = df.drop(columns="index", axis=1)

    df["sum"] = df. sum(axis=1)

    # make a new df with the fraction, row-wize  df_fractions A
    nr_of_columns = len (df.columns)
    nr_of_rows = len(df)
    column_list = df.columns.tolist()

    # calculate the fraction of each age group
    data  = []
    for r in range(nr_of_rows):
        row_data = []
        for c in range(nr_of_columns):
            try:
                row_data.append(round((df.iat[r,c]/df.at[r,"sum"]*100),2))
            except:
                row_data.append( df.iat[r,c])
        data.append(row_data)
    df_fractions = pd.DataFrame(data, columns=column_list)

    # calculate the percentual change of the fractions
    data  = []
    for r in range(nr_of_rows):
        row_data = []
        for c in range(nr_of_columns):
            try:
                row_data.append( round(((df_fractions.iat[r,c]  -  df_fractions.iat[0,c]  )/df_fractions.iat[0,c]*100),2))
            except:
                row_data.append(  df_fractions.iat[r,c])
        data.append(row_data)
    df_rudi = pd.DataFrame(data, columns=column_list)
    df_rudi['Date_of_statistics_week_start'] = df['Date_of_statistics_week_start'].dt.date
    df_rudi.rename(columns={"Date_of_statistics_week_start": "date"},  inplace=True)
    return df_rudi


def main():


    lijst  = ["0-14", "15-19", "20-24", "25-29", "30-34",
             "35-39", "40-44", "45-49", "50-54", "55-59",
             "60-64", "65-69", "70-74", "75-79", "80-84",
             "85-89", "90+", "Unknown",
             "0-29","30-49","50-69","70-89","90+",
             "30-69", "0-39", "40-59", "60-79", "80+",

             "0-19" , "20-29" , "30-39" , "40-49" , "50-59" , "60-69" , "70-79" , "80-89" , "90+",
              "TOTAAL"]

    population = [2707000,1029000,1111000,1134000,1124000,
                  1052000,1033000,1131000,1285000,1263000,
                  1138000,1003000,971000,644000,450000,
                  259000,130000,10,
                  5981000,4340000,4689000,2324000,130000,
                  9029000,8157000,4712000,3756000,839000,
                  1756000, 1980000, 2245000, 2176000, 2164000,

                   2548000, 2141000, 1615000, 709000, 130000, 17464000]  # tot 17 464 000

    st.header("Hospital / ICU admissions in the Netherlands")
    st.subheader("Please send feedback to @rcsmit")

    # DAILY STATISTICS ################

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
    global WDW2
    WDW2 = st.sidebar.slider("Window smoothing curves (weeks)", 1, 8, 1)
    global delete_last_row
    delete_last_row =  st.sidebar.selectbox("Delete last week/row of complete dataset", [True, False], index=0)

    df_pivot_hospital, df_pivot_ic  = prepare_data()


    df_pivot_hospital = select_period(df_pivot_hospital,"Date_of_statistics_week_start", FROM, UNTIL)
    df_pivot_ic = select_period(df_pivot_ic, "Date_of_statistics_week_start",FROM, UNTIL)

    df_pivot_hospital_basic = df_pivot_hospital.copy(deep=False)
    df_pivot_ic_basic =  df_pivot_ic.copy(deep=False)


    df_pivot_hospital = agg_ages(df_pivot_hospital)
    df_pivot_ic = agg_ages(df_pivot_ic)

    df_pivot_hospital, lijst_perc = calculate_percentages(df_pivot_hospital,lijst)
    df_pivot_ic, lijst_perc = calculate_percentages(df_pivot_ic,lijst)

    df_pivot_hospital, lijst_cumm_all =  calculate_cumm(df_pivot_hospital, lijst, "all")
    df_pivot_ic, lijst_cumm_all =  calculate_cumm(df_pivot_ic, lijst, "all")



    df_pivot_hospital, lijst_per_capita = calculate_per_capita(df_pivot_hospital, lijst, population)
    df_pivot_ic, lijst_per_capita = calculate_per_capita(df_pivot_ic, lijst, population)

    df_pivot_hospital, lijst_cumm_period =  calculate_cumm(df_pivot_hospital, lijst, "period")
    df_pivot_ic, lijst_cumm_period =  calculate_cumm(df_pivot_ic, lijst, "period")
    df_hosp_per_cases, df_IC_per_cases = get_data_per_total_reported()
    hospital_or_ic = st.sidebar.selectbox("Hospital or IC", ["hospital", "icu"], index=0)
    what_to_do = st.sidebar.selectbox("What type of graph", ["stack", "line", "per_total_reported"], index=1)

    #default_age_groups = ["0-29","30-49","50-69","70-89","90+"]
    default_age_groups = [ "0-19" , "20-29" , "30-39" , "40-49" , "50-59" , "60-69" , "70-79" , "80-89" , "90+"]
    default_age_groups_perc = ["0-29_perc","30-49_perc","50-69_perc","70-89_perc","90+_perc"]
    default_age_groups_cumm_all = ["0-29_cumm_all","30-49_cumm_all","50-69_cumm_all","70-89_cumm_all","90+_cumm_all"]
    default_age_groups_cumm_period = ["0-29_cumm_period","30-49_cumm_period","50-69_cumm_period","70-89_cumm_period","90+_cumm_period"]
    default_age_groups_per_capita = ["0-29_per_capita","30-49_per_capita","50-69_per_capita","70-89_per_capita","90+_per_capita"]
    if what_to_do == "line":

        age_groups = ["0-29","30-49","50-69","70-89","90+", "TOTAAL"]
        absolute_or_index = st.sidebar.selectbox(f"Absolute | percentages of TOTAAL |\n index (start = 100) | per capita | cummulatief from 2020-1-1 | cummulatief from {FROM} | per total reported", ["absolute",  "percentages", "index",  "per_capita", "cummulatief_all", "cummulatief_period", "per_total_reported"], index=0)

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
        elif  absolute_or_index == "per_total_reported":
            ages_to_show = st.sidebar.multiselect(
                    "Ages to show (multiple possible)", lijst, default_age_groups)
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

    if  absolute_or_index  == "per_total_reported":
        if hospital_or_ic == "hospital":
            make_age_graph_per_total_reported(df_hosp_per_cases, ages_to_show, "ziekenhuisopnames per total reported (%)")
        else:
            make_age_graph_per_total_reported(df_IC_per_cases, ages_to_show, "IC opnames per total reported (%)")
        st.write("Let op: Ziekenhuisopnames worden vergeleken met de total reported van dezelfde week, wat eigenlijk onjuist is.")
        st.write("Plot is gemaakt met data verkregen via een omweg en wordt handmatig geupdate. Laatste update 2 juli 2021")

    if what_to_do == "stack":
        #  SHOW STACKGRAPHS
        if hospital_or_ic == "hospital":

            to_do_stack = [[df_pivot_hospital, ages_to_show, "ziekenhuisopname naar leeftijd"]]
        else:
            to_do_stack = [[df_pivot_ic, ages_to_show, "IC opname naar leeftijd"]]

        for d in to_do_stack:
            show_stack (d[0],d[1],d[2], absolute_or_relative)

    elif what_to_do == "line" and  absolute_or_index !=   "per_total_reported":
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
