# FIT DATA TO A CURVE
# René Smit - MIT Licence

# Import required packages

import numpy as np

import matplotlib.dates as mdates
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime

import platform
import os
from pandas import read_csv, Timestamp, Timedelta, date_range
from matplotlib.pyplot import subplots
from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import ConciseDateFormatter, AutoDateLocator
from matplotlib.backends.backend_agg import RendererAgg

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
from scipy import stats

import imageio
import webbrowser

from sklearn.linear_model import LinearRegression

global  placeholder

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

def find_trendline(df, optimim):
    #x_values = list(range(2, len(df)+1))
    r_sq_opt = 0
    x_values = list(range(0,len(df)))
    y_values = df["log_exp_gr_factor"].to_list()
    #print (x_values, y_values)
    if optimim == True:
        # we search for the optimal value for how many days at the end we take to draw the trendline
        for ix in range (5,10):
            x = x_values[-ix:]
            y = y_values[-ix:]
            m_,b_,r_sq_ = find_slope_scipy(x,y)
            if r_sq_ > r_sq_opt:
                m,b,r_sq,i_opt = m_,b_,r_sq_, ix
                r_sq_opt = r_sq
    else:
        x_range = np.arange(len(df))
        m,b,r_sq = find_slope_sklearn(x_values, y_values)
        i_opt = None
    return m,b,r_sq, i_opt

def extrapolate(df, df_complete, show_from, extend):
    """Extrapolate df to number of dates.
    """
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

    #st.write (f"LENGTE DF {len(df)}")
    # OK 150 items
    df_complete["new_cases_smoothed_original"] = df_complete["new_cases_smoothed"]
    df_complete["date_original"] = df_complete["date"]
    df_complete = df_complete[["date_original", "new_cases_smoothed_original"]]

    mask = (df_complete["date_original"].dt.date >= show_from)
    df_complete = df_complete.loc[mask]
    df_complete = df_complete[:int(TOTAL_DAYS_IN_GRAPH)]

    #df__complete_as_str = df_complete.astype(str)
    # st.write(df__complete_as_str)
    # st.write(len(df_complete))
    df = df.reset_index()

    df["date"] = pd.to_datetime(df["index"], format="%Y-%m-%d")

    df = pd.merge(
        df,
        df_complete,
        how="outer",
        left_on="date",
        right_on="date_original",
        #left_index=True,
    )

    df = df.set_index("date")
    #df_as_str = df.astype(str)
    #st.write(df_as_str)

    return df

def give_info(df, m, b, r_sq, i_opt):
    x = ((df.index - Timestamp('2020-01-01')) # independent
        // Timedelta('1d')).values # small day-of-year integers
    st.write(f"m = {round(m,2)} | b = {round(b,2)} | r_sq = {round(r_sq,2)}")
    U  =(-1/m)/np.log(10)
    st.write(f"U = {round(U,1)} days [ (-1/m)/log(10) ] ")
    jtdm = np.log10(np.exp(1/U)-1)
    st.write(f"J(t) delta_max = {round(jtdm,2)} [ log(exp(1/U)-1)] ")
    day = ( jtdm-b) / m
    topday = df.index[0] + Timedelta(day, 'd') # extrapolate
    st.write(f"Top reached on day  {round(day)} ({topday.date()})")
    st.write(f"Optimal R SQ if I = {i_opt}")

def do_levitt(df, what_to_display, df_complete, show_from, optimim, make_animation,i, total, showlogyaxis, title):
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

    df =  add_column_levit(df, what_to_display)

    df = df.set_index(DATEFIELD)
    nextday = df.index[-1] + Timedelta('1d')
    lastday = df.index[-1] + Timedelta(TOTAL_DAYS_IN_GRAPH - len(df), 'd') # extrapolate

    exdates = date_range(nextday, lastday)
    len_original = len(df)
    extend = len(exdates)
    len_total = int(TOTAL_DAYS_IN_GRAPH)


    m,b,r_sq, i_opt = find_trendline(df, optimim)

    # Number of months to extend

    df = extrapolate(df, df_complete, show_from, extend)
    df = make_calculations(df,m,b, len_original, len_total)
    placeholder  = st.empty()

    filename = make_graph_delta(df, make_animation,i, total, showlogyaxis, title)
    if make_animation == False:
        give_info(df, m, b, r_sq, i_opt)
        make_graph_cumm(df)
    return filename

def make_calculations(df, m, b, len_original, len_total):
    df["predicted_growth"] = np.nan
    #df["predicted_value"] = np.nan
    df["new_cases_smoothed_predicted"] = np.nan
    df["new_cases_smoothed_predicted_cumm"] = np.nan
    df['trendline'] = (df['rownumber'] *m +b)
    df = df.reset_index()

    # we make the trendline
    for i in range(len_total):
        df.loc[i, "predicted_growth"] =    np.exp(10**df.iloc[i]["trendline"] )
        df.loc[i, "real_growth"] =    df.iloc[i]["new_cases_smoothed_cumm"] / df.iloc[i-1]["new_cases_smoothed_cumm"]

    # we transfer the last known total cases to the column predicted cases
    df.loc[len_original-1, "new_cases_smoothed_predicted_cumm"] =  df.iloc[len_original-1]["new_cases_smoothed_cumm"]

    # we make the predictions
    df.loc[len_original, "new_cases_smoothed_predicted_cumm"] =  df.iloc[len_original]["predicted_growth"] * df.iloc[len_original-1]["new_cases_smoothed_predicted_cumm"]

    for i in range(len_original, len_total):
        df.loc[i, "new_cases_smoothed_predicted_cumm"] = df.iloc[i-1]["new_cases_smoothed_predicted_cumm"] * df.iloc[i]["predicted_growth"]
        df.loc[i, "new_cases_smoothed_predicted"] = df.iloc[i]["new_cases_smoothed_predicted_cumm"] - df.iloc[i-1]["new_cases_smoothed_predicted_cumm"]
    df["date"] = pd.to_datetime(df["index"], format="%Y-%m-%d")
    df = df.set_index("index")
    return df

def make_graph_delta(df, animated,i, total, showlogyaxis, title):
    with _lock:
        #fig1y = plt.figure()
        fig1yz, ax = subplots()
        ax3 = ax.twinx()
        ax.set_title(f'Prediction of COVID-19 à la Levitt - {title} - ({i}/{total})')
        # ax.scatter(alldates, df[what_to_display].values, color="#00b3b3", s=1, label=what_to_display)
        ax3.scatter(df["date"] , df["log_exp_gr_factor"].values, color="#b300b3", s=1, label="J(t) reality")
        ax3.scatter(df["date"] , df["trendline"], color="#b30000", s=1, label="J(t) predicted")  
        ax.scatter(df["date"] , df["new_cases_smoothed_original"].values, color="orange", s=1, label="reality new cases")
        ax.scatter(df["date"] , df["new_cases_smoothed"].values, color="green", s=1, label="reality new cases")
        ax.scatter(df["date"] , df["new_cases_smoothed_predicted"].values, color="#0000b3", s=1, label="predicted new cases")
        ax.set_xlim(df.index[0], df.index[-1])

        if showlogyaxis == "10":
            ax.semilogy()
            ax.set_ylim(1, 100_000)
        elif showlogyaxis == "2":
            ax.semilogy(2)
            ax.set_ylim(1, 100_000)
        elif showlogyaxis == "logit":
            ax.set_yscale("logit")
            ax.set_ylim(1, 100_000)
        else:
            ax.set_ylim(0, 1_000)
            #pass

        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # comma separators
        ax.grid()
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator(), show_offset=False))

        filename= (f"{OUTPUT_DIR}\LEVITT_{i}")

        plt.savefig(filename, dpi=100, bbox_inches="tight")

        placeholder1.pyplot(fig1yz)
        plt.close()
    return filename

def make_graph_cumm(df):
    with _lock:
        fig1yza, ax = subplots()
        ax3 = ax.twinx()
        ax.set_title('Cummulative cases and growth COVID-19 à la Levitt')
        ax.scatter(df["date"], df["new_cases_smoothed_cumm"].values, color="green", s=1, label="reality cumm cases")
        ax.scatter(df["date"],  df["new_cases_smoothed_predicted_cumm"].values , color="#00b3b3", s=1, label="predicted cumm cases")

        ax3.scatter(df["date"],  df["real_growth"].values, color="#b300b3", s=1, label="real growth")
        ax3.scatter(df["date"],  df["predicted_growth"].values, color="#b3bbb3", s=1, label="predicted growth")
        ax.set_xlim(df.index[0],  df.index[-1])
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # comma separators
        ax.grid()
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator(), show_offset=False))

        st.pyplot(fig1yza)

def add_column_levit(df, what_to_display):
    """Add column with G(t)

    """

    df["new_cases_smoothed_cumm"] = df["new_cases_smoothed"] .cumsum()
    # df["new_cases_smoothed_cumm"] = df["new_cases_smoothed_cumm"].rolling(window=3, center=False).mean()# geeft rare overgangen
    what_to_display_x = "new_cases_smoothed_cumm"


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

            else:
                log_factor_ = 0

            log_factor_df = log_factor_df.append(
                {
                    "date_log_factor": date_,
                    "waarde": df.iloc[i][what_to_display_x],
                    "log_exp_gr_factor_": log_factor_,
                },
                ignore_index=True,
            )
    log_factor_df["log_exp_gr_factor"] = log_factor_df["log_exp_gr_factor_"].rolling(window=3, center=False).mean()

    log_factor_df = log_factor_df.fillna(0)

    log_factor_df = log_factor_df.reset_index()

    df = pd.merge(
        df,
        log_factor_df,
        how="outer",
        left_on="date",
        right_on="date_log_factor",
        #left_index=True,
    )
    df = df.fillna(0)

    return df
###################################################################
@st.cache(ttl=60 * 60 * 24, allow_output_mutation=True)
def getdata():
    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\COVIDcases\\input\\owid-covid-data_NL.csv"

    else:
        url1= "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        #url1="https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/owid-covid-data_NL.csv"
    return pd.read_csv(url1, delimiter=",", low_memory=False)

def main():

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
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\output\\levitt2021"
    )

    df_complete, FROM, UNTIL, what_to_display, datediff, showlogyaxis, optimim, make_animation, title = sidebar_input(df)

    global placeholder1
    placeholder1  = st.empty()


    if make_animation == True:
            filenames = []
            for i in range(10, datediff+1):
                print (f"DOING {i} of {datediff}")
                until_loop =  df_complete.index[-1] +  i #Timedelta(i , 'd')
                df_to_use = select_period(df_complete, FROM, UNTIL)
                df_to_use = df_to_use[:i]
                df_to_use.fillna(value=0, inplace=True)
                filename = do_levitt(df_to_use, what_to_display,df_complete, FROM, optimim, make_animation,i, datediff+1,showlogyaxis, title)
                filenames.append(filename)

            # build gif
            with imageio.get_writer('mygif.gif', mode='I') as writer:
                for filename_ in filenames:
                    image = imageio.imread(f"{filename_}.png")
                    writer.append_data(image)
            webbrowser.open('mygif.gif')
            tekst = (
                    "<img src='mygif.gif'></img>"
                    )

            #placeholder1.markdown(tekst, unsafe_allow_html=True)
            placeholder1.image("mygif.gif",caption=f"Image",use_column_width= True)

            # Remove files
            # for filename__ in set(filenames):
            #     os.remove(f"{filename__}.png")
    else:
        df_to_use = select_period(df_complete, FROM, UNTIL)
        df_to_use.fillna(value=0, inplace=True)
        filename = do_levitt(df_to_use, what_to_display,df, FROM, optimim, make_animation,i, datediff+1,showlogyaxis, title)
    st.write("Trying to replicate https://docs.google.com/spreadsheets/d/1MNXQTFOLN-bMDAyUjeQ4UJR2bp05XYc8qpLsyAqdrZM/edit#gid=329426677 as described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7325180/#FD4")

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/blob/main/fit_to_data_owid_levitt_streamlit_animated.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
       '</div>'
    )

    st.sidebar.markdown(tekst, unsafe_allow_html=True)
    df_as_str = df_complete.astype(str)
    st.write(df_as_str)

def select_default_options():
    options = [["NL maart 2020", "2020-3-1", "2020-5-1", 149],
                ["NL okt 2020", "2020-9-1", "2020-11-22", 149],
                ["NL dec 2020", "2020-11-22", "2021-2-10", 149],
                ["NL march 2021", "2021-2-10", "2021-6-28", 149],
                ["NL july 2021", "2021-6-28", "2021-9-2", 149],
                ["NL okt 2021", "2021-10-1", "2021-12-31", 149],
                ["NL test", "2021-10-1", "2021-10-20", 149],
    ]

    menuchoicelist = [options[n][0] for n, l in enumerate(options)]
    menu_choice = st.sidebar.radio("",menuchoicelist, index=6)

    for n, l in enumerate(options):
        if menu_choice == options[n][0]:
            title = options[n][0]
            start__ = options[n][1]
            until__ = options[n][2]
            country__ = options[n][3]

    return title, start__, until__, country__

def sidebar_input(df):
    title_, start__, until__, country_default  = select_default_options()
    what_default = 2
    title = st.sidebar.text_input("Title", title_)
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
    d1 = datetime.strptime(from_, "%Y-%m-%d")
    d2 = datetime.strptime(until_, "%Y-%m-%d")
    datediff = abs((d2 - d1).days)
    lijst = df.columns.tolist()

    del lijst[0:4]
    # what_to_display = st.sidebar.selectbox(
    #         "What to display", lijst,
    #         index=what_default,
    #     )
    what_to_display=what_default
    countrylist =  df['location'].drop_duplicates().sort_values().tolist()

    global country_
    # if platform.processor() == "":
    try:
        country_ = st.sidebar.selectbox("Which country",countrylist, index = country_default)
    except:
        country_ = st.sidebar.selectbox("Which country",countrylist, 0)
    df = df.loc[df['location'] == country_]
    print ("486")
    print (df)

    #df = df.loc[df['location'] == "Netherlands"]
    
    # else:
    #     country_ = st.sidebar.selectbox("Which country",countrylist, 0)
    #     df = df.loc[df['location'] == country_]

    #df = df[["date", "new_cases_smoothed"]]
    global TOTAL_DAYS_IN_GRAPH

    extra_days_in_graph = st.sidebar.number_input('Extra days to show',None,None,30)
    TOTAL_DAYS_IN_GRAPH = datediff + extra_days_in_graph
    #TOTAL_DAYS_IN_GRAPH  = st.sidebar.number_input('Extra days to show',None,None,150)

    if  datediff > TOTAL_DAYS_IN_GRAPH:
        st.warning("Make sure that the number of total days is bigger than the date difference")
        st.stop()
    if  datediff < 10:
        st.warning("Make sure that the date difference is at least 4 days")
        st.stop()
    global BASEVALUE

    showlogyaxis =  st.sidebar.selectbox("Y axis as log", ["No", "2", "10", "logit"], index=0)
    optimim  = st.sidebar.selectbox("Find optimal period for trendline", [True, False], index=0)

    if platform.processor() != "":
        make_animation = st.sidebar.selectbox("Make animation (SLOW!)", [True, False], index=0)
    else:
        make_animation = st.sidebar.selectbox("Make animation (SLOW!)", [True, False], index=0)
        # st.sidebar.write ("Animation disabled")
        # make_animation =
    df_= df.copy()
    return df_,FROM,UNTIL,what_to_display,datediff,showlogyaxis,optimim,make_animation, title


if __name__ == "__main__":
    main()
