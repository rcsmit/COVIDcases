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

def cell_background_number_of_cases(val,max):
    """Creates the CSS code for a cell with a certain value to create a heatmap effect
    Args:
        val ([int]): the value of the cell

    Returns:
        [string]: the css code for the cell
    """
    opacity = 0
    try:
        v = abs(val)
        color = '193, 57, 43'
        value_table = [ [0,0],
                        [0.00390625,0.0625],
                        [0.0078125, 0.125],
                        [0.015625,0.25],
                        [0.03125,0.375],
                        [0.0625,0.50],
                        [0.125,0.625],
                        [0.25,0.75],
                        [0.50,0.875],
                        [0.75,0.9375],
                        [1,1]]
        for vt in value_table:
            #print (f"{v} - {vt[0]}")
            if v >= round(vt[0]*max) :
                opacity = vt[1]
                #print (f"{v} - {vt[0]} YES")
    except:
        # give cells with eg. text or dates a white background
        color = '255,255,255'
        opacity = 1
    return f'background: rgba({color}, {opacity})'


def day_to_day(df, column_, numberofdays):
    column_ = column_ if type(column_) == list else [column_]
    newcolumns = []   # percentuele verandering
    newcolumns2 = []  # index
    df_new =  pd.DataFrame()

    df_new["date"] = None
    for c in column_:
        newname = str(c) + "_daydiff"
        #newname2 = str(c) + "_daydiff_index"
        newcolumns.append(newname)
        #newcolumns2.append(newname2)
        df[newname] = np.nan
        #df[newname2] = np.nan
        df_new[c] = np.nan
        #df_new[newname2] = np.nan
        for n in range(numberofdays, len(df)):
            #df_new.at[n, "date"] = df.iloc[n]["pos_test_Date_statistics"]
            df_new.at[n, "date"] = df.iloc[n]["date"]

            vorige_day = df.iloc[n - numberofdays][c]
            nu = df.iloc[n][c]
            waarde = round((((nu - vorige_day) / vorige_day) * 100), 2)
            #waarde2 = round((((nu) / vorige_day) * 100), 2)

            df.at[n, newname] = waarde
            #df.at[n, newname2] = waarde2

            df_new.at[n, c] = waarde
            #df_new.at[n, newname2] = waarde2
            # df_new.at[n, "date"] = datetime.datetime.strftime(df_new.at[n,"date"], '%Y-%m-%d')
            #df_new.at[n, "date"] = datetime.datetime.strptime(df_new.at[n,"date"], '%Y-%m-%d').date()


    return df, df_new, newcolumns

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
def get_data():
    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
    else:
        url1= "https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv"

    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")
    df = df.groupby(["Date_statistics", "Agegroup"], sort=True).count().reset_index()
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

def st_dev(test_list):
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5
    return mean, res

def accumulate_first_rows(df, x):
    """Accumulate the first X rows

    Args:
        df (df): table with numbers

    Returns:
        df : table with the first x rows accumulated

    """
    # calculate the sum
    #df = df.drop(columns="index", axis=1)

    # df['Date_statistics'] = df['Date_statistics'].dt.date # from 2021-01-01T00:00:00+01:00 to yyyy-mm-dd
    # make a new df with the fraction, row-wize  df_fractions A
    nr_of_columns = len (df.columns)
    nr_of_rows = len(df)
    column_list = df.columns.tolist()

    # calculate the fraction of each age group
    data  = []
    first_row_values = []
    first_row_sums = []
    number_of_first_rows = st.sidebar.slider("Eerste x aantal dagen samenvoegen", 0, 21, 7)
    first_row_data = []
    for c in range(nr_of_columns):
        first_row_sums.append(0.0)

    first_row_data.append( df.iat[number_of_first_rows-1,0]) # date of row number_of_first_rows
    for r in range(nr_of_rows):
        if r < number_of_first_rows:

            for c in range(1,nr_of_columns):
                first_row_sums[c] += df.iat[r,c]


            if  r == number_of_first_rows-1:
                for t in range(1,len(first_row_sums)):
                    first_row_data.append(first_row_sums[t])

                data.append(first_row_data)
        else:
            row_data = []
            for c in range(nr_of_columns):
                try:
                    row_data.append(df.iat[r,c])
                except:
                    # date
                    row_data.append( df.iat[r,c])
            data.append(row_data)

    df_accumulated = pd.DataFrame(data, columns=column_list)
    with st.beta_expander('First rows accumulated',  expanded=False):
        st.subheader (f"The first {number_of_first_rows} rows accumated")
        st.write (df_accumulated)
    return df_accumulated

def select_period_oud(df, field, show_from, show_until):
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

def do_the_rudi(df_):
    """Calculate the fractions per age group. Calculate the difference related to day 0 as a % of day 0.
    Made for Rudi Lousberg
    Inspired by Ian Denton https://twitter.com/IanDenton12/status/1407379429526052866

    Args:
        df (df): table with numbers

    Returns:
        df : table with the percentual change of the fracctions
    """
    # calculate the sum
    #df = df.drop(columns="index", axis=1)
    df__ = accumulate_first_rows(df_,7)
    df = df__.copy(deep=False)

    df["sum"] = df. sum(axis=1)
    df['Date_statistics'] = df['Date_statistics'].dt.date # from 2021-01-01T00:00:00+01:00 to yyyy-mm-dd
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
                    # date
                    row_data.append( df.iat[r,c])
            data.append(row_data)
    df_fractions = pd.DataFrame(data, columns=column_list)
    with st.beta_expander('The fractions',  expanded=False):
        st.subheader ("The fractions")
        st.write (df_fractions)



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
    return pd.DataFrame(data, columns=column_list)

def  make_legenda(max_value):
        stapfracties =   [0, 0.00390625, 0.0078125, 0.015625,  0.03125,  0.0625 , 0.125,  0.25,  0.50, 0.75,  1]
        stapjes =[]
        for i in range(len(stapfracties)):
            stapjes.append((stapfracties[i]*max_value))
        d = {'legenda': stapjes}

        df_legenda = pd.DataFrame(data=d)
        if platform.processor() != "":
            st.write (df_legenda.style.format(None, na_rep="-").applymap(lambda x:  cell_background_number_of_cases(x,max_value)).set_precision(2))
        else:
            st.write (df_legenda)
def main():
    start_ = "2021-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    global from_
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

    df_getdata = get_data()
    df = df_getdata.copy(deep=False)

    df = select_period_oud(df,"Date_statistics", FROM, UNTIL)
    df.rename(
        columns={
            "Date_file": "count",
        },
        inplace=True,
    )


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

    # option to drop agegroup 0-9 due to changes in testbeleid en -bereidheid
    #st.write(df_pivot.dtypes)
    try:
        df_pivot = df_pivot.drop(columns=["<50"], axis=1)
    except:
        pass
    try:
        df_pivot = df_pivot.drop(columns=["Unknown"], axis=1)
    except:
        pass
    df_pivot=df_pivot.fillna(0)
    drop_0_9  = st.sidebar.selectbox("Delete agegroup 0-9", [True, False], index=1)
    if drop_0_9 == True:
        df_pivot = df_pivot.drop(columns="0-9", axis=1)
    df_pivot_original = df_pivot.copy(deep=False)

    #df_pivot = df_pivot.add_prefix("pos_test_")
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
    column_list = df_pivot.columns.tolist()
    df_pivot['Date_statistics'] = df_pivot['Date_statistics'].dt.date
    df_pivot.rename(columns={"Date_statistics": "date"},  inplace=True)
    column_list = column_list[1:]
    numberofdays = st.sidebar.slider("Vergelijken met x dagen ervoor", 0, 21, 7)
    with st.beta_expander('Number of cases',  expanded=True):
        st.subheader("Number of cases per age")
        st.write ("Er wordt teruggerekend naar eeste ziektedag")
        #df_pivot['pos_test_Date_statistics'] = df_pivot['pos_test_Date_statistics'].dt.date
        max_value = 1600
        if platform.processor() != "":
            st.write (df_pivot.style.format(None, na_rep="-").applymap(lambda x:  cell_background_number_of_cases(x,max_value)).set_precision(0))
            make_legenda(max_value)
        else:
            st.write (df_pivot)


        st.subheader("Number of cases per age / number of people per age * 100.000")

        df_naar_fractie, top_waarde = calculate_fraction(df_pivot)
        if platform.processor() != "":
            st.write (df_naar_fractie.style.format(None, na_rep="-").applymap(lambda x:  cell_background_number_of_cases(x,top_waarde)).set_precision(2))
            make_legenda(top_waarde)
        else:
            st.write(df_naar_fractie)
    df_pivot_2,df_new, newcolumns,= day_to_day(df_pivot, column_list, numberofdays)
    st.sidebar.write("Attention : slow script!!!")
    df_new.reset_index(drop=True)
    with st.beta_expander('Percentual changes of cases',  expanded=False):
        st.subheader(f"Percentual change with {numberofdays} days before")
        if platform.processor() != "":
            st.write(df_new.style.format(None, na_rep="-").applymap(cell_background).set_precision(2))
        else:
            st.write(df_new)

    df_new_rudi = do_the_rudi(df_pivot_original)

    with st.beta_expander('Percentual changes of fractions',  expanded=False):
        st.subheader("Percentual change of the fractions per agegroup with the first day(s)")
        st.write ("fraction = cases in an agegroup / total cases")
        if platform.processor() != "":
            st.write(df_new_rudi.style.format(None, na_rep="-").applymap(cell_background).set_precision(2))
        else:
            st.write(df_new_rudi)


    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/edit/main/dag_verschillen_casus_landelijk.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        'On the request of  <a href="https://twitter.com/LousbergRudi" target="_blank">Rudi Lousberg</a><br>'
        'Inspired by <a href="https://twitter.com/IanDenton12/status/1407734030926336008" target="_blank">Ian Denton</a></div>'
    )
    st.sidebar.markdown(tekst, unsafe_allow_html=True)
if __name__ == "__main__":
    main()