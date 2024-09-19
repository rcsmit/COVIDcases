# FIT DATA TO A CURVE
# René Smit - MIT Licence

# inspired by @dimgrr. Based on
# https://towardsdatascience.com/basic-curve-fitting-of-scientific-data-with-python-9592244a2509?gi=9c7c4ade0880
# https://github.com/venkatesannaveen/python-science-tutorial/blob/master/curve-fitting/curve-fitting-tutorial.ipynb


# https://www.reddit.com/r/CoronavirusUS/comments/fqx8fn/ive_been_working_on_this_extrapolation_for_the/
# to explore : https://github.com/fcpenha/Gompertz-Makehan-Fit/blob/master/script.py


# Import required packages

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
import math
from lmfit import Model
import pandas as pd
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta
import matplotlib.animation as animation
import imageio
import streamlit.components.v1 as components
import os
import platform
import webbrowser
from pandas import read_csv, Timestamp, Timedelta, date_range
from io import StringIO
from numpy import log, exp, sqrt, clip, argmax, put
from scipy.special import erfc, erf
from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import ConciseDateFormatter, AutoDateLocator
# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock
#from streamlit import caching

from PIL import Image
import glob


# Functions to calculate values a,b  and c ##########################
def sigmoidal(x, a, b, c):
    ''' Standard sigmoidal function
        a = height, b= halfway point, c = growth rate
        https://en.wikipedia.org/wiki/sigmoidal_function '''
    return a * np.exp(-b * np.exp(-c * x))

def derivate(x, a, b, c, d):
    ''' First derivate of the sigmoidal function. Might contain an error'''
    #return  (np.exp(b * (-1 * np.exp(-c * x)) - c * x) * a * b * c ) + d

    return ( a * x**3) +( b*x**2) +( c*x) +d



def cases_from_r (x, r, d):
    return d * (r**(x/4))

def fit_the_values(country_, y_values , total_days, graph, output):
    """
    We are going to fit the values

    """

    try:
        base_value = y_values[0]
        # some preperations
        number_of_y_values = len(y_values)
        global TOTAL_DAYS_IN_GRAPH
        TOTAL_DAYS_IN_GRAPH = total_days  # number of total days
        x_values = np.linspace(start=0, stop=number_of_y_values - 1, num=number_of_y_values)

        x_values_extra = np.linspace(
            start=0, stop=TOTAL_DAYS_IN_GRAPH - 1, num=TOTAL_DAYS_IN_GRAPH
        )


        popt_d, pcov_d = curve_fit(
            f=derivate,
            xdata=x_values,
            ydata=y_values,
            #p0=[0, 0, 0],
            p0 = [26660, 9, 0.03, base_value],  # IC BEDDEN MAART APRIL
            bounds=(-np.inf, np.inf),
            maxfev=10000,
        )
        residuals = y_values - derivate(x_values, *popt_d)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_values - np.mean(y_values))**2)
        r_squared = round(  1 - (ss_res / ss_tot),4)
        l = (f"polynome fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f / r2 = {r_squared}" % tuple(popt_d))

        a,b,c,d = popt_d
        deriv_0 = derivate(0, a,b,c,d)
        deriv_last = derivate(number_of_y_values,a,b,c,d)
        r_0_last_formula = (deriv_last/deriv_0)**(4/number_of_y_values)
        #r_0_last_cases = (y_values[number_of_y_values-1]/ y_values[0])**(4/number_of_y_values)
        r_total = sum(
            (derivate(i, a, b, c, d) / derivate(i - 1, a, b, c, d)) ** (4 / 1)
            for i in range(number_of_y_values - 5, number_of_y_values)
        )

        #r_avg_formula = r_total/(number_of_y_values-1)
        r_avg_formula = r_total/6
        r_cases_list = []


        if output == True:
            #st.write (f"{country_} : {x_values} / {y_values}/ base {base_value} /  a {a} / b {b} / c {c} / d {d} / r2 {r_squared}")
            st.write (f"{country_} : / base {base_value} /  a {a} / b {b} / c {c} / d {d} / r2 {r_squared}")

            #st.write (f"Number of Y values {number_of_y_values}")


            #st.write (f"Number of R-values {len(r_cases_list)}")
            st.write (f"R from average values from formula day by day {r_avg_formula} (purple)")

            #st.write (f"R from average values from cases day by day {r_cases_avg} (yellow)")

            st.write (f"R getal from formula from day 0 to day {number_of_y_values}: {r_0_last_formula} (red)")
            #st.write (f"R getal from cases from day 0 to day {number_of_y_values}: {r_0_last_cases} (orange)")


        if graph == True:
            #with _lock:
            if 1==1:
                fig1x = plt.figure()
                plt.plot(
                        x_values_extra,
                        derivate(x_values_extra, *popt_d),
                        "g-",
                        label=l
                    )

                plt.plot(
                            x_values,
                            cases_from_r(x_values, r_0_last_formula,deriv_0),
                            "r-",
                            label=(f"cases_from_r_0_last_formula ({round(r_0_last_formula,2)})")
                        )
                # plt.plot(
                #             x_values,
                #             cases_from_r(x_values, r_0_last_cases,deriv_0),
                #             "orange",
                #             label=(f"cases_from_r_0_last_cases ({round(r_0_last_cases,2)})")
                #         )

                plt.plot(
                            x_values,
                            cases_from_r(x_values, r_avg_formula,deriv_0),
                            "purple",
                            label=(f"cases_from_r_avg_formula  ({round(r_avg_formula,2)})")
                        )
                # plt.plot(
                #             x_values,
                #             cases_from_r(x_values, r_cases_avg,deriv_0),
                #             "yellow",
                #             label=(f"cases_from_r_avg_cases ({round(r_cases_avg,2)})")
                #         )
                plt.scatter(x_values, y_values, s=20, color="#00b3b3", label="Data")

                plt.legend()
                plt.title(f"{country_} / curve_fit (scipy)")
                #plt.ylim(bottom=0)
                plt.xlabel(f"Days from {from_}")


                st.pyplot(fig1x)


    except RuntimeError as e:
        #str_e = str(e)
        #st.info(f"Could not find derivate fit :\n{str_e}")
        pass
    try:
        a = 1* r_avg_formula
    except NameError:
        r_avg_formula = None

    return r_avg_formula

def select_period(df, show_from, show_until):
    """ _ _ _ """
    if show_from is None:
        show_from = "2020-2-27"

    if show_until is None:
        show_until = "2020-4-1"

    mask = (df[DATEFIELD].dt.date >= show_from) & (df[DATEFIELD].dt.date <= show_until)
    df = df.loc[mask]

    df = df.reset_index()

    return df


###################################################################
@st.cache_data(ttl=60 * 60 * 24)
def getdata():
    if platform.processor() != "":
        url1 = "C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\owid-covid-data.csv"
       

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
        "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\output\\"
    )

    start__ = "2021-05-22"
    until__ = "2021-06-12"
    what_default = 1
    days_to_show = 21
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
    what_to_fit = st.sidebar.selectbox(
            "What to display", lijst,
            index=what_default,
        )
    #which_method = st.sidebar.selectbox("Which method (lmfit)", [ "sigmoidal", "derivate", "exponential","lineair", "gaussian"], index=what_method_default)


    countrylist =  df['location'].drop_duplicates().sort_values().tolist()
    one_country = st.sidebar.selectbox("One country", [True, False], index=0)
    if one_country == True:
        country_ = st.sidebar.selectbox("Which country",countrylist, 216)

    total_days = st.sidebar.number_input('Total days to show',None,None,days_to_show)



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



    then = d1 + dt.timedelta(days=total_days)
    daterange = mdates.drange(d1,then,dt.timedelta(days=1))



    global a_start, b_start, c_start, default_values

    default_values  = st.sidebar.selectbox("Default values", [True, False], index=1)
    if default_values:

        st.sidebar.write("Default values:") #  UK  New cases 15/5/21-10/6/21 2339.47 110.156 9.2619
        a_start = st.sidebar.number_input('a',None,None,2339)
        b_start = st.sidebar.number_input('b',None,None,110)
        c_start = st.sidebar.number_input('c',None,None,9)
    else:
        a_start, b_start, c_start = 0,0,0



    if one_country == True:
        df_to_fit = df.loc[df['location'] ==country_]
        df_to_use = select_period(df_to_fit, FROM, UNTIL)
        df_to_use.fillna(value=0, inplace=True)
        values_to_fit = df_to_use[what_to_fit].tolist()
        R_value_country = fit_the_values(country_, values_to_fit, total_days,  True, True)

    else:
        df_country_r = pd.DataFrame(columns=['country', "R_value"])
        for country_ in countrylist:
            df_to_fit = df.loc[df['location'] == country_]
            df_to_use = select_period(df_to_fit, FROM, UNTIL)
            df_to_use.fillna(value=0, inplace=True)

            values_to_fit = df_to_use[what_to_fit].tolist()
            if len(values_to_fit) != 0:
                R_value_country = fit_the_values(country_, values_to_fit, total_days,  False, False)
                if R_value_country != None and R_value_country < 5:
                    st.write (f"{country_}  - {R_value_country}")


                    df_country_r = df_country_r.append({'country': country_, "R_value": R_value_country}, ignore_index=True)
        df_country_r.sort_values(by='R_value', ascending=False)
        st.dataframe(df_country_r, 500,2000)

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
