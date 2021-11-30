# FIT DATA TO A CURVE
# René Smit - MIT Licence

# inspired by @dimgrr. Based on
# https://towardsdatascience.com/basic-curve-fitting-of-scientific-data-with-python-9592244a2509?gi=9c7c4ade0880
# https://github.com/venkatesannaveen/python-science-tutorial/blob/master/curve-fitting/curve-fitting-tutorial.ipynb


# https://www.reddit.com/r/CoronavirusUS/comments/fqx8fn/ive_been_working_on_this_extrapolation_for_the/
# to explore : https://github.com/fcpenha/Gompertz-Makehan-Fit/blob/master/script.py


# Import required packages

import numpy as np

import matplotlib.dates as mdates

import pandas as pd
import streamlit as st
import datetime as dt
from datetime import datetime

import platform

from pandas import read_csv, Timestamp, Timedelta, date_range
from matplotlib.pyplot import subplots
from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import ConciseDateFormatter, AutoDateLocator
from matplotlib.backends.backend_agg import RendererAgg

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
from scipy import stats


from sklearn.linear_model import LinearRegression

def select_period(df, show_from, show_until):
    """ Select a period in the dataframe """
    if show_from is None:
        show_from = "2020-2-27"

    if show_until is None:
        show_until = "2020-4-1"

    mask = (df[DATEFIELD].dt.date >= show_from) & (df[DATEFIELD].dt.date <= show_until)
    df = df.loc[mask]

    df = df.reset_index()

    return df



def find_slope_sklearn(x_,y_):
    """Find slope of regression line - DOESNT WORK

    Args:
        df_temp ([type]): [description]
        what_to_show_l (string): The column to show on x axis
        what_to_show_r (string): The column to show on y axis
        intercept_100(boolean)) : intercept on (0,100) ie. VE starts at 100% ?
    Returns:
        [type]: [description]
    """
    x = np.reshape(x_, (-1, 1))
    y = np.reshape(y_, (-1, 1))
    #x = x.reshape((-1, 1))
    #y = y.reshape((-1, 1))
    # st.write(x)

    # st.write(y)

    #obtain m (slope) and b(intercept) of linear regression line

    model = LinearRegression()
    model.fit(x, y)
    m = model.coef_[0]
    b = model.intercept_
    r_sq = model.score(x, y)
    return m,b,r_sq

def find_slope_scipy(x_,y_):

    m, b, r_value, p_value, std_err = stats.linregress(x_, y_)
    r_sq = r_value**2
    return m,b,r_sq
def straight_line(x,m,b):
    return x*m+b

def do_levitt(df, what_to_display):
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325180/#FD4
    # https://docs.google.com/spreadsheets/d/1MNXQTFOLN-bMDAyUjeQ4UJR2bp05XYc8qpLsyAqdrZM/edit#gid=329426677
    # G(T)=N/e=0.37N.

    # G(t) = N exp(− exp(− (t − T)/ U)),
    # U is time constant, measured in days

    # . Parameter N is the asymptotic number, the maximum plateau value that G(t) reaches after a long time, t.
    # Parameter T, is the point of inflection, which is the time in days at which the second-derivative of G(t)
    #  is zero and its first derivative is a maximum. It is a natural mid-point of the function where the value
    # of G(T)=N/e=0.37N. The Parameter U, is the most important as it changes the shape of the curve; it is a
    #  time-constant measured in days.

    df, last_value, background_new_cases, background_total_cases =  add_column_levit(df, what_to_display)
    #print (df)
    df = df.set_index(DATEFIELD)
    firstday = df.index[0] + Timedelta('1d')
    nextday = df.index[-1] + Timedelta('1d')
    lastday = df.index[-1] + Timedelta(TOTAL_DAYS_IN_GRAPH - len(df), 'd') # extrapolate

    #yd = df['Deceased_cumm'].values # dependent
    exrange = range((Timestamp(nextday)
        - Timestamp(firstday)) // Timedelta('1d'),
        (Timestamp(lastday) + Timedelta('1d')
        - Timestamp(firstday)) // Timedelta('1d')) # day-of-year ints
    indates = date_range(df.index[0], df.index[-1])
    exdates = date_range(nextday, lastday)
    alldates = date_range(df.index[0], df.index[-1])
    len_original = len(indates)
    len_total = int(TOTAL_DAYS_IN_GRAPH)

    t = np.linspace(0.0, TOTAL_DAYS_IN_GRAPH, 10000)
    r_sq_opt = 0

    optimim  = st.sidebar.selectbox("Find optimal period for trendline", [True, False], index=0)

    #x_values = list(range(2, len(df)+1))
    x_values = list(range(0,len(df)+1))
    y_values = df["log_exp_gr_factor"].to_list()
    if optimim == True:
        # we search for the optimal value for how many days at the end we take to draw the trendline
        for i in range (2,10): # we start a bit later, because the first values have an R_sq = 1

            x = x_values[-i:]
            y = y_values[-i:]

            df_ = df.tail(i)
            m_,b_,r_sq_ = find_slope_scipy(x,y)
            if r_sq_ > r_sq_opt:
                m,b,r_sq,i_opt = m_,b_,r_sq_, i
                r_sq_opt = r_sq
        st.write(f"Optimal R SQ if I = {i_opt}")
    else:
        x_range = np.arange(len(df))
        m,b,r_sq = find_slope_sklearn(df, x_range,"log_exp_gr_factor")

    # Number of months to extend
    extend = len(exdates)

    # Extrapolate the index first based on original index
    df = pd.DataFrame(
        data=df,
        index=pd.date_range(
            start=df.index[0],
            periods=len(df.index) + extend,
            freq=df.index.freq
        )
    )
    df['rownumber'] = np.arange(len(df))
    alldates = date_range(df.index[0], df.index[-1])
    # Display
    print (df)

    with _lock:
        #fig1y = plt.figure()
        fig1yz, ax = subplots()
        ax3 = ax.twinx()
        ax.set_title('Prediction of COVID-19 à la Levitt')

        x = ((df.index - Timestamp('2020-01-01')) # independent
            // Timedelta('1d')).values # small day-of-year integers
        yi = df[what_to_display].values # dependent
        yi2 = df["log_exp_gr_factor"].values
        ax.scatter(alldates, yi, color="#00b3b3", s=1, label=what_to_display)
        ax3.scatter(alldates, yi2, color="#b300b3", s=1, label=what_to_display)
        st.write(f"m = {round(m,2)} | b = {round(b,2)} | r_sq = {round(r_sq,2)}")

        U  =(-1/m)/np.log(10)
        st.write(f"U = {U} days [ (-1/m)/log(10) ] ")

        jtdm = np.log10(np.exp(1/U)-1)

        st.write(f"J(t) delta_max = {round(jtdm,2)} [ log(exp(1/U)-1)] ")

        day = ( jtdm-b) / m
        st.write(f"Top reached on day  {round(day)}")

        df["predicted_growth"] = np.nan
        #df["predicted_value"] = np.nan
        df["predicted_new_cases"] = np.nan
        df["cumm_cases_predicted"] = np.nan
        df['trendline'] = (df['rownumber'] *m +b)
        df = df.reset_index()


        df["cumm_cases_minus_background"] = df["total_cases"] - background_total_cases + df.iloc[0]["new_cases_smoothed"]
        df["cumm_cases_minus_background"] = df["cumm_cases_minus_background"].rolling(7).mean()

        # we make the trendline
        for i in range(len_total):
            df.loc[i, "predicted_growth"] =    np.exp(10**df.iloc[i]["trendline"] )
            df.loc[i, "real_growth"] =    df.iloc[i]["cumm_cases_minus_background"] / df.iloc[i-1]["cumm_cases_minus_background"]


        # we transfer the last known total cases to the column predicted cases
        df.loc[len_original-1, "cumm_cases_predicted"] =  df.iloc[len_original-1]["cumm_cases_minus_background"]

        # we make the predictions
        df.loc[len_original, "cumm_cases_predicted"] =  df.iloc[len_original]["predicted_growth"] * df.iloc[len_original-1]["cumm_cases_predicted"]

        for i in range(len_original, len_total):
            df.loc[i, "cumm_cases_predicted"] = df.iloc[i-1]["cumm_cases_predicted"] * df.iloc[i]["predicted_growth"]
            df.loc[i, "predicted_new_cases"] = df.iloc[i]["cumm_cases_predicted"] - df.iloc[i-1]["cumm_cases_predicted"]

        df["date"] = pd.to_datetime(df["index"], format="%Y-%m-%d")
        df = df.set_index("index")
        ax3.scatter(alldates, df["trendline"], color="#b30000", s=1, label=what_to_display)
        ax.scatter(alldates, df["predicted_new_cases"].values, color="#0000b3", s=1, label="predicted new cases")

        df_= df[["date", what_to_display, "cumm_cases_minus_background", "log_exp_gr_factor", 'trendline',"real_growth", "predicted_growth" ,"predicted_new_cases" ,"cumm_cases_predicted"]]
        df_as_str = df_.astype(str)
        st.write(df_as_str)

        ax.set_xlim(df.index[0], lastday)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # comma separators
        ax.grid()
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator(), show_offset=False))

        st.pyplot(fig1yz)



    with _lock:

        fig1yza, ax = subplots()
        ax3 = ax.twinx()
        ax.set_title('Cummulative cases and growth COVID-19 à la Levitt')

        x = ((df.index - Timestamp('2020-01-01')) # independent
            // Timedelta('1d')).values # small day-of-year integers
        yi = df["cumm_cases_predicted"].values # dependent
        y2 = df["minus_background_cumm"].values # dependent

        yi2 = df["log_exp_gr_factor"].values
        ax.scatter(alldates, yi, color="#00b3b3", s=1, label="cumm_cases_predicted")
        ax.scatter(alldates, y2, color="#00c3c3", s=1, label="minus_background_cum")

        ax3.scatter(alldates,  df["real_growth"].values, color="#b300b3", s=1, label="real growth")
        ax3.scatter(alldates,  df["predicted_growth"].values, color="#b3bbb3", s=1, label="predicted growth")

        ax.set_xlim(df.index[0], lastday)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # comma separators
        ax.grid()
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator(), show_offset=False))

        st.pyplot(fig1yza)



def add_column_levit(df, what_to_display):
    """Add column with G(t)
    Args:
        df ([type]): [description]
        what_to_display ([type]): [description]

    Returns:
        [type]: [description]
    """
    print (what_to_display)

    background_new_cases = df.iloc[0][what_to_display]
    background_total_cases = df.iloc[0]["total_cases"]

    st.write (f"Background new cases {background_new_cases}")
    df["minus_background"] = df[what_to_display] - background_new_cases
    df["what_to_display_cumm"] = df[what_to_display].cumsum()

    df["minus_background_cumm"] = df["new_cases_smoothed"] .cumsum()
    what_to_display_x = "minus_background_cumm"
    last_value = df[what_to_display].iloc[-1]

    log_factor_df = pd.DataFrame(
        {"date_log_factor": [], "waarde": [], "log_exp_gr_factor": []}
    )
    d = 1 # we compare with 1 day before
    for i in range(2,len(df)):

        if df.iloc[i][what_to_display] != None:
            date_ = pd.to_datetime(df.iloc[i]["date"], format="%Y-%m-%d")
            date_ = df.iloc[i]["date"]
            if (df.iloc[i - d][what_to_display_x] != 0) or (df.iloc[i - d][what_to_display_x] is not None) or (df.iloc[i][what_to_display_x] != df.iloc[i - d][what_to_display_x]) :
                log_factor_ = round(np.log10(np.log (((df.iloc[i][what_to_display_x] / df.iloc[i - d][what_to_display_x]))                  )), 2)
                # st.write (f"{df.iloc[i][what_to_display]} | {(df.iloc[i][what_to_display_x] / df.iloc[i - d][what_to_display_x])} |  {log_factor_}" )

            else:
                log_factor_ = 0

            log_factor_df = log_factor_df.append(
                {
                    "date_log_factor": date_,
                    "waarde": df.iloc[i][what_to_display_x],
                    "log_exp_gr_factor": log_factor_,
                },
                ignore_index=True,
            )
    log_factor_df = log_factor_df.fillna(0)

    log_factor_df = log_factor_df.reset_index()
    #print (log_factor_df)
    df = pd.merge(
        df,
        log_factor_df,
        how="outer",
        left_on="date",
        right_on="date_log_factor",
        #left_index=True,
    )
    df = df.fillna(0)

    return df, last_value, background_new_cases, background_total_cases
###################################################################
@st.cache(ttl=60 * 60 * 24, allow_output_mutation=True)
def getdata():
    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\owid-covid-data_NL.csv"
    else:
        url1= "https://covid.ourworldindata.org/data/owid-covid-data.csv"

    return pd.read_csv(url1, delimiter=",", low_memory=False)

def main():
    global placeholder0, placeholder, placeholder1
    global DATEFIELD
    global COUNTRY
    COUNTRY = ""
    DATEFIELD = "date"
    DATE_FORMAT = "%Y-%m-%d"
    df_getdata = getdata()
    df = df_getdata.copy(deep=False)

    df[DATEFIELD] = pd.to_datetime(df[DATEFIELD], format=DATE_FORMAT)
    df.fillna(value=0, inplace=True)

    # df = df[(df.location == COUNTRY)].deepcopy()


    global start__
    global OUTPUT_DIR


    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\output\\"
    )

    start__ = "2021-10-1"
    until__ = "2022-1-31"
    what_default = 2
    days_to_show = 150
    what_method_default = 1


    today = datetime.today().strftime("%Y-%m-%d")
    global from_
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start__)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is valid and/or in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", until__)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is valid and/or in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()
    lijst = df.columns.tolist()

    del lijst[0:4]
    what_to_display = st.sidebar.selectbox(
            "What to display", lijst,
            index=what_default,
        )
    which_method = st.sidebar.selectbox("Which method (lmfit)", [ "sigmoidal", "derivate", "exponential","lineair", "gaussian"], index=what_method_default)

    #what_to_display = st.sidebar.selectbox("What", ["Confirmed","Deceased", "dConfirmed"], index=2)

    countrylist =  df['location'].drop_duplicates().sort_values().tolist()

    #st.write (statelist)
    # statelist = ["Goa", "Delhi", "India"]
    global country_
    if platform.processor() == "":
        country_ = st.sidebar.selectbox("Which country",countrylist, 149)
        df = df.loc[df['location'] == country_]
    else:
        country_ = st.sidebar.selectbox("Which country",countrylist, 0)
        df = df.loc[df['location'] == country_]


    total_days = st.sidebar.number_input('Total days to show',None,None,days_to_show)
    global TOTAL_DAYS_IN_GRAPH
    TOTAL_DAYS_IN_GRAPH = total_days  # number of total days

    # df['dConfirmed'] =  df['Confirmed'].shift(-1) - df['Confirmed']

    d1 = datetime.strptime(from_, "%Y-%m-%d")
    d2 = datetime.strptime(until_, "%Y-%m-%d")
    datediff = abs((d2 - d1).days)
    if  datediff > total_days:
        st.warning("Make sure that the number of total days is bigger than the date difference")
        st.stop()
    if  datediff < 4:
        st.warning("Make sure that the date difference is at least 4 days")
        st.stop()
    global BASEVALUE

    # df["cases_double_smooth"] =( df.iloc[:, "new_cases_smoothed"]
    #                 .rolling(window=7, center=True)
    #                 .mean()
    #             )

    df_to_use = select_period(df, FROM, UNTIL)
    df_to_use.fillna(value=0, inplace=True)

    values_to_fit = df_to_use[what_to_display].tolist()
    base_value__ = values_to_fit[0]
    BASEVALUE = st.sidebar.number_input('Base value',None,None,base_value__)

    to_do_list = [[what_to_display, values_to_fit]]

    then = d1 + dt.timedelta(days=total_days)
    daterange = mdates.drange(d1,then,dt.timedelta(days=1))








    global prepare_for_animation
    if platform.processor() != "":

        prepare_for_animation = st.sidebar.selectbox("Make animation (SLOW!)", [True, False], index=1)
    else:
        st.sidebar.write ("Animation disabled")
        prepare_for_animation = False

    st.write("Trying to replicate https://docs.google.com/spreadsheets/d/1MNXQTFOLN-bMDAyUjeQ4UJR2bp05XYc8qpLsyAqdrZM/edit#gid=329426677 as described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325180/#FD4")

    do_levitt(df_to_use, what_to_display)

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/blob/main/fit_to_data_streamlit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        'Thanks to <a href="https://twitter.com/dimgrr" target="_blank">@dimgrr</a> for the inspiration and help.</div>'
    )

    st.sidebar.markdown(tekst, unsafe_allow_html=True)


if __name__ == "__main__":
    main()