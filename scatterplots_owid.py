#from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.dates as mdates
from textwrap import wrap

# import seaborn as sn
from scipy import stats
import datetime as dt
from datetime import datetime, timedelta
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import json
# from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as ticker
import math
import platform
# _lock = RendererAgg.lock
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import streamlit as st
import urllib
import urllib.request
from pathlib import Path
#from streamlit import caching
from inspect import currentframe, getframeinfo
import plotly.express as px
import plotly.graph_objects as go

###################################################################

def download_data_file(url, filename, delimiter_, fileformat):
    """Download the external datafiles
    IN :  url : the url
          filename : the filename (without extension) to export the file
          delimiter : delimiter
          fileformat : fileformat
    OUT : df_temp : the dataframe
    """

    # df_temp = None
    download = True
    with st.spinner(f"Downloading...{url}"):
        if download:  # download from the internet
            url = url
        elif fileformat == "json":
            url = INPUT_DIR + filename + ".json"
        else:
            url = INPUT_DIR + filename + ".csv"

        if fileformat == "csv":
            df_temp = pd.read_csv(url, delimiter=delimiter_, low_memory=False)
        elif fileformat == "json":
            df_temp = pd.read_json(url)

        # elif fileformat =='json_x':   # workaround for NICE IC data
        #     pass
        #     # with urllib.request.urlopen(url) as url_x:
        #     #     datajson = json.loads(url_x.read().decode())
        #     #     for a in datajson:
        #     #         df_temp = pd.json_normalize(a)
        else:
            st.error("Error in fileformat")
            st.stop()
        df_temp = df_temp.drop_duplicates()
        # df_temp = df_temp.replace({pd.np.nan: None})  Let it to test
        save_df(df_temp, filename)
        return df_temp


def find_correlation_pair(df, first, second):
    al_gehad = []
    paar = []
    if type(first) == list:
        first = first
    else:
        first = [first]
    if type(second) == list:
        second = second
    else:
        second = [second]
    for i in first:
        for j in second:
            c = round(df[i].corr(df[j]), 3)
    return c


@st.cache_data(ttl=60 * 60 * 24)
def get_data():
    """Get the data from various sources
    In : -
    Out : df        : dataframe
         UPDATETIME : Date and time from the last update"""
    with st.spinner(f"GETTING ALL DATA ..."):
        init()
        # #CONFIG

        if platform.processor() != "":
                 data = [

                {
                    "url": "C:\\Users\\rcxsm\\Documents\python_scripts\\covid19_seir_models\\COVIDcases\input\\owid-covid-data_20211202.csv",
                    "name": "owid",
                    "delimiter": ",",
                    "key": "date",
                    "key2": "location",
                    "dateformat": "%Y-%m-%d",
                    "groupby": None,
                    "fileformat": "csv",
                    "where_field": None,
                    "where_criterium": None
                },



            ]

        else:


            data = [

                {
                    "url": "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
                    "name": "owid",
                    "delimiter": ",",
                    "key": "date",
                    "key2": "location",
                    "dateformat": "%Y-%m-%d",
                    "groupby": None,
                    "fileformat": "csv",
                    "where_field": None,
                    "where_criterium": None
                },




            ]

        type_of_join = "outer"
        d = 0

        # Read first datafile
        df_temp_x = download_data_file(
            data[d]["url"], data[d]["name"], data[d]["delimiter"], data[d]["fileformat"]
        )
        # df_temp_x = df_temp_x.replace({pd.np.nan: None})
        df_temp_x[data[d]["key"]] = pd.to_datetime(
            df_temp_x[data[d]["key"]], format=data[d]["dateformat"]
        )
        firstkey = data[d]["key"]
        firstkey2 = data[d]["key2"]


        if data[d]["where_field"] != None:
            where_field = data[d]["where_field"]
            df_temp_x = df_temp_x.loc[df_temp_x[where_field] == data[d]["where_criterium"]]

        if data[d]["groupby"] is None:
            df_temp_x = df_temp_x.sort_values(by=firstkey)
            df_ungrouped = None

        else:
            df_temp_x = (
                df_temp_x.groupby([data[d]["key"]], sort=True).sum().reset_index()
            )
            df_ungrouped = df_temp_x.reset_index()
            firstkey_ungrouped = data[d]["key"]
        df = (
            df_temp_x  # df_temp is the base to which the other databases are merged to
        )
        # Read the other files

        # for d in range(1, len(data)):

        #     df_temp_x = download_data_file(
        #         data[d]["url"],
        #         data[d]["name"],
        #         data[d]["delimiter"],
        #         data[d]["fileformat"],
        #     )
        #     # df_temp_x = df_temp_x.replace({pd.np.nan: None})
        #     oldkey = data[d]["key"]
        #     newkey = "key" + str(d)
        #     oldkey2 = data[d]["key2"]
        #     newkey2 = "key2_" + str(d)
        #     df_temp_x = df_temp_x.rename(columns={oldkey: newkey})
        #     df_temp_x = df_temp_x.rename(columns={oldkey2: newkey2})
        #     #st.write (df_temp_x.dtypes)
        #     try:
        #         df_temp_x[newkey] = pd.to_datetime(df_temp_x[newkey], format=data[d]["dateformat"]           )
        #     except:
        #         st.error(f"error in {oldkey} {newkey}")
        #         st.stop()

        #     if data[d]["where_field"] != None:
        #         where_field = data[d]["where_field"]
        #         df_temp_x = df_temp_x.loc[df_temp_x[where_field] == data[d]["where_criterium"]]

        #     if data[d]["groupby"] != None:
        #         if df_ungrouped is not None:
        #             df_ungrouped = df_ungrouped.append(df_temp_x, ignore_index=True)
        #             print(df_ungrouped.dtypes)
        #             print(firstkey_ungrouped)
        #             print(newkey)
        #             df_ungrouped.loc[
        #                 df_ungrouped[firstkey_ungrouped].isnull(), firstkey_ungrouped
        #             ] = df_ungrouped[newkey]

        #         else:
        #             df_ungrouped = df_temp_x.reset_index()
        #             firstkey_ungrouped = newkey
        #         df_temp_x = df_temp_x.groupby([newkey], sort=True).sum().reset_index()

        #     df_temp = pd.merge(
        #         df_temp, df_temp_x, how=type_of_join, left_on=[firstkey, firstkey2], right_on=[newkey, newkey2]
        #     )
        #     df_temp.loc[df_temp[firstkey].isnull(), firstkey] = df_temp[newkey]
        #     df_temp = df_temp.sort_values(by=firstkey)
        # # the tool is build around "date"
        # df = df_temp.rename(columns={firstkey: "date"})

        UPDATETIME = datetime.now()


        return df,  UPDATETIME


def drop_columns(df, what_to_drop):
    """  _ _ _ """
    if what_to_drop != None:
        for d in what_to_drop:
            print("dropping " + d)

            df = df.drop(columns=[d], axis=1)
    return df


def select_period(df, show_from, show_until):
    """ _ _ _ """
    if show_from is None:
        show_from = "2020-1-1"

    if show_until is None:
        show_until = "2030-1-1"

    mask = (df["date"].dt.date >= show_from) & (df["date"].dt.date <= show_until)
    #mask = (df["date"].dt.date == date) # & (df["date"].dt.date <= show_until)
    df = df.loc[mask]

    df = df.reset_index()

    return df


def save_df(df, name):
    """  _ _ _ """
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")


def find_slope_sklearn(df_temp, what_to_show_l, what_to_show_r, intercept_100,  log_x, log_y):
    """Find slope of regression line - DOESNT WORK

    Args:
        df_temp ([type]): [description]
        what_to_show_l (string): The column to show on x axis
        what_to_show_r (string): The column to show on y axis
        intercept_100(boolean)) : intercept on (0,100) ie. VE starts at 100% ?
    Returns:
        [type]: [description]
    """



    x = np.array(df_temp[what_to_show_l]).reshape((-1, 1))
    y = np.array(df_temp[what_to_show_r])
    #obtain m (slope) and b(intercept) of linear regression line
    if intercept_100 :
        fit_intercept_=False
        i = 100
    else:
        fit_intercept_=True
        i = 0
    model = LinearRegression(fit_intercept=fit_intercept_)
    model.fit(x, y - i)
    m = model.coef_[0]
    b = model.intercept_+ i
    r_sq = model.score(x, y- i)
    return m,b,r_sq

def create_trendline(l,m,b):
    """creates a dataframe with the values for a trendline. Apperentlu plotlyexpress needs a dataframe to plot sthg

    Args:
        l (int) : length
        m (float): slope
        b (float): intercept
        complete (boolean) : Show trendline until VE =0 or only the given dataset

    Returns:
        df: dataframe
    """
    t = []

    x_ = int(l)

    for i in range (x_):
        j = m*i +b
        t.append([i,j])
    df_trendline = pd.DataFrame(t, columns = ['x', 'y'])
    return df_trendline

def make_scatterplot(df_temp, what_to_show_l, what_to_show_r,  categoryfield, hover_name, log_x, log_y, FROM, UNTIL):

    """Makes a scatterplot with trendline and statistics

    Args:
        df_temp ([type]): [description]
        what_to_show_l (string): The column to show on x axis
        what_to_show_r (string): The column to show on y axis
        show_cat ([type]): [description]
        categoryfield ([type]): [description]
    """

    if FROM==UNTIL:
        date = FROM
    else:
        date = f"{FROM} - {UNTIL}"
    df_temp =df_temp[[what_to_show_l, what_to_show_r, categoryfield, hover_name]]
    if log_x == True:
        new_column_x = "log10_" + what_to_show_l
        df_temp[new_column_x] = np.log(df_temp[what_to_show_l])
        what_to_show_l_calc = new_column_x
    else:
        what_to_show_l_calc = what_to_show_l

    if log_y == True:
        new_column_y = "log10_" + what_to_show_r
        df_temp[new_column_y] = np.log(df_temp[what_to_show_r])
        what_to_show_r_calc = new_column_y
    else:
        what_to_show_r_calc = what_to_show_r

    df_temp= df_temp.dropna()
    #print (df_temp)
    if len(df_temp) == 0:
        st.error("No data")
        return
    #with _lock:
    if 1==1:
        fig1xy,ax = plt.subplots()
        m,b,r2 = find_slope_sklearn(df_temp, what_to_show_l_calc, what_to_show_r_calc, False, log_x, log_y)


        fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, color=categoryfield, hover_name=hover_name,  trendline="ols",  trendline_options=dict(log_x=log_x,log_y=log_y ),  trendline_scope="overall", log_x=log_x, log_y = log_y)
        # l = df_temp[what_to_show_l].max()
        # df_trendline = create_trendline(l,m,b)

        # fig2 = px.line(df_trendline, x="x", y="y")
        # fig2.update_traces(line=dict(color = 'rgba(50,50,50,0.8)'))
        # #add linear regression line to scatterplot

        # fig3 = go.Figure(data=fig1xy.data + fig2.data)
        # correlation_sp = round(df_temp[what_to_show_l_calc].corr(df_temp[what_to_show_r_calc], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
        correlation_p = round(df_temp[what_to_show_l_calc].corr(df_temp[what_to_show_r_calc], method='pearson'), 3)

        title_scatter = (f"{what_to_show_l} -  {what_to_show_r}<br>({date})<br>Correlation pearson = {correlation_p}<br>y = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")  #Rankcorrelation spearman = {correlation_sp} -

        fig1xy.update_layout(
            title=dict(
                text=title_scatter,
                x=0.5,
                y=0.95,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )
            ),
            xaxis_title=what_to_show_l,
            yaxis_title=what_to_show_r,
            font=dict(
                family="Courier New, Monospace",
                size=12,
                color='#000000'
            )
        )

        ax.text(
            1,
            1.3,
            "Created by Rene Smit — @rcsmit",
            transform=ax.transAxes,
            fontsize="xx-small",
            va="top",
            ha="right",
        )

        st.plotly_chart(fig1xy, use_container_width=True)

def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta * left
    right = lim[1] + delta * right
    ax.set_xlim(left, right)


def init():
    """  _ _ _ """

    global download

    global INPUT_DIR
    global OUTPUT_DIR

    INPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\"
    )
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\python_scripts\\output\\"
    )

    # GLOBAL SETTINGS
    download = True  # True : download from internet False: download from INPUT_DIR
    # De open data worden om 15.15 uur gepubliceerd


def show_footer():
    st.write ("Original Standard values were replicating: Palash Basak, Global Perspective of COVID-19 Vaccine Nationalism,")
    st.write("https://www.medrxiv.org/content/10.1101/2021.12.31.21268580v1.full.pdf")
    st.write ("R-code: https://rstudio.cloud/project/2771953")

    toelichting = (
      ""
    )

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/edit/main/covid_dashboard_rcsmit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
      )

    st.markdown(toelichting, unsafe_allow_html=True)
    st.sidebar.markdown(tekst, unsafe_allow_html=True)

    st.image(
        "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png"
    )

    st.markdown(
        '<a href="https://www.buymeacoffee.com/rcsmit" target="_blank">If you are happy with this dashboard, you can buy me a coffee</a>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<br><br><a href="https://www.linkedin.com/in/rcsmit" target="_blank">Contact me for custom dashboards and infographics</a>',
        unsafe_allow_html=True,
    )


def main():
    """  _ _ _ """
    global FROM
    global UNTIL

    global OUTPUT_DIR
    global INPUT_DIR
    global UPDATETIME

    init()

    df_getdata, UPDATETIME = get_data()
    df = df_getdata.copy(deep=False)

    st.title("Scatterplots OWID")
    # st.header("")

    FROM, UNTIL, lijst, what_to_show_l, what_to_show_r, log_x, log_y, method_x, method_y = interface(df)
    df = select_period(df, FROM, UNTIL)
    #df = select_period(df, date)

    df = df.drop_duplicates()

    df = df[df["population"] > 1000000]

    if FROM == UNTIL:

        show_scatterplots(df, what_to_show_l, what_to_show_r,  "continent",  "location", log_x, log_y, FROM, UNTIL)
    else:
        df_period_left,column_name_l = calculate_df_period(df, what_to_show_l, method_x, True)
        df_period_right,column_name_r = calculate_df_period(df, what_to_show_r, method_y, False)
        df_merged = df_period_left.merge(df_period_right, on="location", how ="inner")

        st.write(df_merged)
        show_scatterplots(df_merged, column_name_l, column_name_r,  "continent",  "location", log_x, log_y, FROM, UNTIL)
    show_footer()
def show_scatterplots(df, what_to_show_l, what_to_show_r,  continent,  location, log_x, log_y, FROM, UNTIL):
    """First show a scatterplot for all the continents, and then for each continent seperately

    Args:
        df ([type]): [description]
        what_to_show_l ([type]): [description]
        what_to_show_r ([type]): [description]
        continent ([type]): [description]
        location ([type]): [description]
        log_x ([type]): [description]
        log_y ([type]): [description]
        date_to_show ([type]): [description]
    """

    make_scatterplot(df, what_to_show_l, what_to_show_r,  "continent",  "location", log_x, log_y,FROM, UNTIL)
    continent_list = df['continent'].unique()
    for continent in continent_list:
        df_continent = df[df["continent"] == continent]
        if len(df_continent) != 0:
            st.subheader(continent)
            make_scatterplot(df_continent, what_to_show_l, what_to_show_r,  "continent", "location", log_x, log_y,FROM, UNTIL)

def calculate_df_period(df, what_to_show, method, add_contintent):

    location_list = df['location'].unique()
    table = []

    for location in location_list:
        df_location = df[df["location"] == location].copy(deep=True)
        if len(df_location) != 0:
            if method=="perc_delta_min_max":

                min = df_location[what_to_show].min()
                max = df_location[what_to_show].max()
                value = ((max - min) / min)*100
            elif method=="perc_delta_first_last":

                min = df_location[what_to_show].iloc[0]
                max = df_location[what_to_show].iloc[-1]
                value = ((max - min) / min)*100

            elif method == "mean":
                value =df_location[what_to_show].mean()
            elif method == "last":
                value =df_location[what_to_show].iloc[-1]
            elif method == "first":
                value =df_location[what_to_show].iloc[0]
            elif method == "lowest":
                value =df_location[what_to_show].min()
            elif method == "highest":
                value =df_location[what_to_show].max()


            postfix = method

            if add_contintent:
                continent = df_location["continent"].iloc[0]
                            #st.write(to_add)
                to_add = [location,continent, value]
            else:
                to_add = [location,value]
            table.append(to_add)
    column_name = f"{what_to_show}_{postfix}"
    if add_contintent:
        df = pd.DataFrame(table, columns = ['location','continent',column_name])
    else:
        df = pd.DataFrame(table, columns = ['location',column_name])
    return df, column_name




def interface(df):
    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2022-1-31"
    #today = datetime.today().strftime("%Y-%m-%d")
    today = "2022-1-31"
    from_ = st.sidebar.text_input("date (yyyy-mm-dd)", start_)
    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)
    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM > UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    if from_ == "2023-08-23":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
            caching.clear_cache()
            st.success("Cache is cleared, please reload to scrape new values")



    lijst = df.columns.tolist()
    del lijst[0:5]

    for i,x in enumerate(lijst):
        print (f"{i} - {x}")

    what_to_show_l = st.sidebar.selectbox(
        "What to show X-axis", lijst, index=37 #37 (pple fully vacc per hundred)
    )
    what_to_show_r = st.sidebar.selectbox(
        "What to show Y-axis", lijst, index=8 #10 (new_deaths_smoothed_per_million)
    )

    log_x = st.sidebar.selectbox(
        "X-ax as log", [True, False], index=1)
    log_y = st.sidebar.selectbox(
        "Y-ax as log", [True, False], index=1)

    if  FROM != UNTIL:
        method_x =  st.sidebar.selectbox( "Method X-ax", ["mean", "perc_delta_min_max","perc_delta_first_last", "first", "last", "lowest", "highest" ], index=0)
        method_y =  st.sidebar.selectbox( "Method Y-ax", ["mean", "perc_delta_min_max","perc_delta_first_last","first", "last", "lowest", "highest"], index=0)
    else:
        method_x,method_y = None, None
    return FROM, UNTIL, lijst, what_to_show_l, what_to_show_r, log_x, log_y, method_x, method_y

if __name__ == "__main__":
    main()