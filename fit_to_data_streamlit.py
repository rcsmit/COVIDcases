# FIT DATA TO A CURVE
# René Smit - MIT Licence

# inspired by @dimgrr. Based on
# https://towardsdatascience.com/basic-curve-fitting-of-scientific-data-with-python-9592244a2509?gi=9c7c4ade0880
# https://github.com/venkatesannaveen/python-science-tutorial/blob/master/curve-fitting/curve-fitting-tutorial.ipynb


# https://www.reddit.com/r/CoronavirusUS/comments/fqx8fn/ive_been_working_on_this_extrapolation_for_the/
# to explore : https://github.com/fcpenha/Gompertz-Makehan-Fit/blob/master/script.py


# Import required packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
import copy, math
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
from lmfit import Model
from matplotlib.pyplot import subplots
from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import ConciseDateFormatter, AutoDateLocator

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

from PIL import Image
import glob

# Functions to calculate values a,b  and c ##########################
def exponential(x, a, b, c):
    ''' Standard gompertz function
        a = height, b= halfway point, c = growth rate
        https://en.wikipedia.org/wiki/Gompertz_function '''
    return a * np.exp(-b * np.exp(-c * x))

def derivate(x, a, b, c):
    ''' First derivate of the Gompertz function. Might contain an error'''
    return a * b * c * np.exp(b * (-1 * np.exp(-c * x)) - c * x)
  # return a * b * c * np.exp(-b*np.exp(-c*x))*np.exp(-c*x)

def gaussian(x, a, b, c):
    ''' Standard Guassian function. Doesnt give results, Not in use'''
    return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))

def gaussian_2(x, a, b, c):
    ''' Another gaussian fuctnion. in use
        a = height, b = cen (?), c= width '''
    return a * np.exp(-((x - b) ** 2) / c)

def growth(x, a, b):
    """ Growth model. a is the value at t=0. b is the so-called R number.
        Doesnt work. FIX IT """
    return np.power(a * 0.5, (x / (4 * (math.log(0.5) / math.log(b)))))

# https://replit.com/@jsalsman/COVID19USlognormals
def lognormal_c(x, s, mu, h): # x, sigma, mean, height
  return h * 0.5 * erfc(- (log(x) - mu) / (s * sqrt(2)))
# https://en.wikipedia.org/wiki/Log-normal_distribution#Cumulative_distribution_function


def normal_c(x, s, mu, h): # x, sigma, mean, height
  return h * 0.5 * (1 + erf((x - mu) / (s * sqrt(2))))

# #####################################################################

def find_gaussian_curvefit(x_values, y_values):
    try:
        popt_g2, pcov_g2 = curve_fit(
            f=gaussian_2,
            xdata=x_values,
            ydata=y_values,
            p0=[0, 0, 0],
            bounds=(-np.inf, np.inf),
            maxfev=10000,
        )
    except RuntimeError as e:
        str_e = str(e)
        st.error(f"gaussian fit :\n{str_e}")

    return tuple(popt_g2)

def use_curvefit(x_values, x_values_extra, y_values,  title, daterange,i):
    """
    Use the curve-fit from scipy.
    IN : x- and y-values. The ___-extra are for "predicting" the curve
    """
    with _lock:
        st.subheader(f"Curvefit (scipy) - {title}")

        fig1x = plt.figure()
        try:
            popt, pcov = curve_fit(
            f=exponential,
            xdata=x_values,
            ydata=y_values,
            p0=[4600, 11, 0.5],
            bounds=(-np.inf, np.inf),
            maxfev=10000,
            )
            plt.plot(
            x_values_extra,
            exponential(x_values_extra, *popt),
            "r-",
            label="exponential fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt),
        )
        except RuntimeError as e:
            str_e = str(e)
            st.error(f"Exponential fit :\n{str_e}")


        try:
            popt_d, pcov_d = curve_fit(
                f=derivate,
                xdata=x_values,
                ydata=y_values,
                p0=[0, 0, 0],
                bounds=(-np.inf, np.inf),
                maxfev=10000,
            )
            plt.plot(
                x_values_extra,
                derivate(x_values_extra, *popt_d),
                "g-",
                label="derivate fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt_d),
            )
        except RuntimeError as e:
            str_e = str(e)
            st.error(f"Derivate fit :\n{str_e}")


        # FIXIT
        # try:
        #     popt_growth, pcov_growth = curve_fit(
        #         f=growth,
        #         xdata=x_values,
        #         ydata=y_values,
        #         p0=[500, 0.0001],
        #         bounds=(-np.inf, np.inf),
        #         maxfev=10000,
        #     )
        #     plt.plot(
        #         x_values_extra,
        #         growth(x_values_extra, *popt_growth),
        #         "y-",
        #         label="growth: a=%5.3f, b=%5.3f" % tuple(popt_growth),
        #     )
        # except:
        #     st.write("Error with growth model fit")


        try:
            popt_g, pcov_g = curve_fit(
                f=gaussian_2,
                xdata=x_values,
                ydata=y_values,
                p0=[0.1, 0.1, 0.1],
                bounds=(-np.inf, np.inf),
                maxfev=10000,
            )


            plt.plot(
                x_values_extra,
                gaussian_2(x_values_extra, *popt_g),
                "b-",
                label="gaussian fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt_g),
            )

        except RuntimeError as e:
            str_e = str(e)
            st.error(f"Gaussian fit :\n{str_e}")


        plt.scatter(x_values, y_values, s=20, color="#00b3b3", label="Data")
        plt.legend()
        plt.title(f"{title} / curve_fit (scipy)")
        plt.ylim(bottom=0)
        plt.xlabel(f"Days from {from_}")

         # POGING OM DATUMS OP DE X-AS TE KRIJGEN (TOFIX)
        # plt.xlim(daterange[0], daterange[-1])
        # lay-out of the x axis
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # interval_ = 5
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval_))
        # plt.gcf().autofmt_xdate()

        #plt.show()
        filename= (f"{OUTPUT_DIR}scipi_{title}_{i}")
        plt.savefig(filename, dpi=100, bbox_inches="tight")

        st.pyplot(fig1x)


# def make_gif(filelist):
#     # Create the frames
#     frames = []
#     imgs = glob.glob("*.png")
#     for i in imgs:
#         new_frame = Image.open(i)
#         frames.append(new_frame)
#      
#     # Save into a GIF file that loops forever
#     frames[0].save('png_to_gif.gif', format='GIF',
#                    append_images=frames[1:],
#                    save_all=True,
#                    duration=300, loop=0)

def use_lmfit(x_values, y_values,  functionlist, title,i, max_y_values):
    """
    Use lmfit.
    IN : x- and y-values.
         functionlist (which functions to use)

          adapted from https://stackoverflow.com/a/49843706/4173718

    TODO: Make all graphs in one graph
    """
    for function in functionlist:
        st.subheader(f"LMFIT - {title} - {function}")

        # create a Model from the model function
        if function == "exponential":
            bmodel = Model(exponential)
            formula = "a * np.exp(-b * np.exp(-c * x))"
        elif function == "derivate":
            bmodel = Model(derivate)
            formula = "a * b * c * np.exp(b * (-1 * np.exp(-c * x)) - c * x)"
        elif function == "gaussian":
            bmodel = Model(gaussian_2)
            formula =  "a * np.exp(-((x - b) ** 2) / c)"
        else:
            st.write("Please choose a function")
            st.stop()

        # create Parameters, giving initial values
        params = bmodel.make_params(a=4711, b=12, c=0.06)
        # params = bmodel.make_params()
        params["a"].min = 0
        params["b"].min = 0
        params["c"].min = 0

        # do fit, st.write result
        result = bmodel.fit(y_values, params, x=x_values)
        a = round(result.params['a'].value,5)
        b= round(result.params['b'].value,5)
        c =round(result.params['c'].value,5)

        st.text(result.fit_report())
        with _lock:
            fig1y = plt.figure()
            # plot results -- note that `best_fit` is already available
            plt.scatter(x_values, y_values, color="#00b3b3")
            plt.plot(x_values, result.best_fit, "r--")
            res = (f"a: {a} / b: {b} / c: {c}")
            plt.title(f"{title} / lmfit - {function}\n{formula}\n{res}")
            t = np.linspace(0.0, TOTAL_DAYS_IN_GRAPH, 10000)

            # use `result.eval()` to evaluate model given params and x
            plt.plot(t, bmodel.eval(result.params, x=t), "r-")
            #plt.ylim(bottom=0)
            plt.ylim(0, max_y_values*1.1)
            plt.xlabel(f"Days from {from_}")

            plt.ylabel(title)
            #plt.show()
            filename= (f"{OUTPUT_DIR}lmfit_{title}_{function}_{i}")

            plt.savefig(filename, dpi=100, bbox_inches="tight")
            st.pyplot(fig1y)

        if prepare_for_animation == False:
            with _lock:
                fig1z = plt.figure()
                # plot results -- note that `best_fit` is already available


                if function == "exponential":
                    plt.plot(t, derivate(t,a,b,c))
                    function_x = "derivate"
                    formula_x = "a * b * c * np.exp(b * (-1 * np.exp(-c * x)) - c * x)"

                elif function == "derivate":
                    plt.plot(t, exponential(t, a,b,c))
                    function_x = "exponential"
                    formula_x = "a * np.exp(-b * np.exp(-c * x))"

                else:
                    st.error("ERROR")
                    st.stop()
                plt.title(f"{title} / {function_x}\n{formula_x}\n{res}")
                t = np.linspace(0.0, TOTAL_DAYS_IN_GRAPH, 10000)

                # use `result.eval()` to evaluate model given params and x
                #plt.plot(t, bmodel.eval(result.params, x=t), "r-")
                plt.ylim(bottom=0)
                plt.xlabel(f"Days from {from_}")

                plt.ylabel(title)
                #plt.show()
                #filename= (f"{OUTPUT_DIR}lmfit_{title}_{function}_{i}")
                #plt.savefig(filename, dpi=100, bbox_inches="tight")
                st.pyplot(fig1z)

    return filename

def fit_the_values_really(x_values,  y_values, which_method, title, daterange,i, max_y_values):
        x_values_extra = np.linspace(
            start=0, stop=TOTAL_DAYS_IN_GRAPH - 1, num=TOTAL_DAYS_IN_GRAPH
        )
        x_values = x_values[:i]
        y_values = y_values[:i]
        if prepare_for_animation == False:
            use_curvefit(x_values, x_values_extra, y_values,  title, daterange,i)
        filename = use_lmfit(x_values,y_values, [which_method], title,i, max_y_values)
        return filename

def fit_the_values(to_do_list , total_days, daterange, which_method, prepare_for_animation):
    """
    We are going to fit the values

    """
    # Here we go !
    st.header("Fitting data to formulas")
    infox = (
    '<br>Exponential / Standard gompertz function : <i>a * exp(-b * np.exp(-c * x))</i></li>'
    '<br>First derivate of the Gompertz function :  <i>a * b * c * exp(b * (-1 * exp(-c * x)) - c * x)</i></li>'
    '<br>Gaussian : <i>a * exp(-((x - b) ** 2) / c)</i></li>'
    '<br>Working on growth model: <i>(a * 0.5 ^ (x / (4 * (math.log(0.5) / math.log(b)))))</i> (b will be the Rt-number)</li>'
        )
    st.markdown(infox, unsafe_allow_html=True)
    el = st.empty()
    for v in to_do_list:
        title = v[0]
        y_values = v[1]
        max_y_values = max(y_values)

        # some preperations
        number_of_y_values = len(y_values)
        global TOTAL_DAYS_IN_GRAPH
        TOTAL_DAYS_IN_GRAPH = total_days  # number of total days
        x_values = np.linspace(start=0, stop=number_of_y_values - 1, num=number_of_y_values)

        filenames = []
        if prepare_for_animation == True:
            for i in range(5, len(x_values)):
                filename = fit_the_values_really(x_values,  y_values, which_method,  title, daterange, i, max_y_values)
                filenames.append(filename)
            # build gif
            with imageio.get_writer('mygif.gif', mode='I') as writer:
                for filename_ in filenames:
                    image = imageio.imread(f"{filename_}.png")
                    writer.append_data(image)
            webbrowser.open('mygif.gif')

            # Remove files
            for filename__ in set(filenames):
                os.remove(f"{filename__}.png")
        else:
            for i in range(len(x_values)-1, len(x_values)):
                filename = fit_the_values_really(x_values,  y_values, which_method, title, daterange, i, max_y_values)

        # FIXIT
        # aq, bq, cq = find_gaussian_curvefit(x_values, y_values)
        # st.write(f"Find Gaussian curvefit - a:{aq}  b:{bq}  c: {cq}")

def select_period(df, show_from, show_until):
    """ _ _ _ """
    if show_from == None:
        show_from = "2020-2-27"

    if show_until == None:
        show_until = "2020-4-1"

    mask = (df["date"].dt.date >= show_from) & (df["date"].dt.date <= show_until)
    df = df.loc[mask]

    df = df.reset_index()

    return df


def normal_c(df):
    #https://replit.com/@jsalsman/COVID19USlognormals
    st.subheader("Normal_c")
    df = df.set_index('date')
    firstday = df.index[0] + Timedelta('1d')
    nextday = df.index[-1] + Timedelta('1d')
    lastday = df.index[-1] + Timedelta(TOTAL_DAYS_IN_GRAPH - len(df), 'd') # extrapolate
    with _lock:
        #fig1y = plt.figure()
        fig1yz, ax = subplots()
        ax.set_title('NL COVID-19 cumulative log-lognormal extrapolations\n'
            + 'Source: repl.it/@jsalsman/COVID19USlognormals')

        x = ((df.index - Timestamp('2020-01-01')) # independent
            // Timedelta('1d')).values # small day-of-year integers
        yi = df['Total_reported_cumm'].values # dependent
        yd = df['Deceased_cumm'].values # dependent
        exrange = range((Timestamp(nextday)
            - Timestamp(firstday)) // Timedelta('1d'),
            (Timestamp(lastday) + Timedelta('1d')
            - Timestamp(firstday)) // Timedelta('1d')) # day-of-year ints
        indates = date_range(df.index[0], df.index[-1])
        exdates = date_range(nextday, lastday)

        ax.scatter(indates, yi, color="#00b3b3", label='Infected')
        ax.scatter(indates, yd, color="#00b3b3", label='Dead')

        sqrt2 = sqrt(2)

        im = Model(normal_c)
        st.write (x)
        iparams = im.make_params(s=0.3, mu=4.3, h=16.5)
        st.write (iparams)
        #iparams['s'].min = 0; iparams['h'].min = 0
        iresult = im.fit(log(yi+1), iparams, x=x)
        st.text('---- Infections:\n' + iresult.fit_report())
        ax.plot(indates, exp(iresult.best_fit)-1, 'b', label='Infections fit')
        ipred = iresult.eval(x=exrange)
        ax.plot(exdates, exp(ipred)-1, 'b--',
            label='Forecast: {:,.0f}'.format(exp(ipred[-1])-1))
        iupred = iresult.eval_uncertainty(x=exrange, sigma=0.95) # 95% interval
        iintlow = clip(ipred-iupred, ipred[0], None)
        put(iintlow, range(argmax(iintlow), len(iintlow)), iintlow[argmax(iintlow)])
        ax.fill_between(exdates, exp(iintlow), exp(ipred+iupred), alpha=0.35, color='b')

        dm = Model(normal_c)

        dparams = dm.make_params(s=19.8, mu=79.1, h=11.4) # initial guesses
        dparams['s'].min = 0; iparams['h'].min = 0
        dresult = dm.fit(log(yd+1), dparams, x=x)
        st.text('---- Deaths:\n' + dresult.fit_report())
        ax.plot(indates, exp(dresult.best_fit)-1, 'r', label='Deaths fit')
        dpred = dresult.eval(x=exrange)
        ax.plot(exdates, exp(dpred)-1, 'r--',
            label='Forecast: {:,.0f}'.format(exp(dpred[-1])-1))
        dupred = dresult.eval_uncertainty(x=exrange, sigma=0.95) # 95% interval
        dintlow = clip(dpred-dupred, log(max(yd)+1), None)
        put(dintlow, range(argmax(dintlow), len(dintlow)), dintlow[argmax(dintlow)])
        ax.fill_between(exdates, exp(dintlow), exp(dpred+dupred), alpha=0.35, color='r')
        ax.fill_between(exdates, 0.012 * (exp(iintlow)), 0.012 * (exp(ipred+iupred)),
            alpha=0.85, color='g', label='Deaths from observed fatality rate')

        ax.set_xlim(df.index[0], lastday)
        #ax.set_yscale('log') # semilog
        #ax.set_ylim(0, 1500000)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # comma separators
        ax.grid()
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator(), show_offset=False))
        ax.set_xlabel('95% prediction confidence intervals shaded')

        #fig.savefig('plot.png', bbox_inches='tight')
        #print('\nTO VIEW GRAPH: click on plot.png in the file pane to the left.')
        #fig.show()
        st.pyplot(fig1yz)

    st.text('Infections at end of period shown: {:,.0f}. Deaths: {:,.0f}.'.format(
        exp(ipred[-1])-1, exp(dpred[-1])-1))


def loglognormal(df, what_to_display):
    #https://replit.com/@jsalsman/COVID19USlognormals
    st.subheader("Log Normal")
    df = df.set_index('date')
    firstday = df.index[0] + Timedelta('1d')
    nextday = df.index[-1] + Timedelta('1d')
    lastday = df.index[-1] + Timedelta(TOTAL_DAYS_IN_GRAPH - len(df), 'd') # extrapolate
    with _lock:
        #fig1y = plt.figure()
        fig1yz, ax = subplots()
        ax.set_title('NL COVID-19 cumulative log-lognormal extrapolations\n'
            + 'Source: repl.it/@jsalsman/COVID19USlognormals')

        x = ((df.index - Timestamp('2020-01-01')) # independent
            // Timedelta('1d')).values # small day-of-year integers
        yi = df[what_to_display].values # dependent
        #yd = df['Deceased_cumm'].values # dependent
        exrange = range((Timestamp(nextday)
            - Timestamp(firstday)) // Timedelta('1d'),
            (Timestamp(lastday) + Timedelta('1d')
            - Timestamp(firstday)) // Timedelta('1d')) # day-of-year ints
        indates = date_range(df.index[0], df.index[-1])
        exdates = date_range(nextday, lastday)

        ax.scatter(indates, yi, color="#00b3b3", label='Infected')
        #ax.scatter(indates, yd, color="#00b3b3", label='Dead')

        sqrt2 = sqrt(2)

        #im = Model(normal_c)
        im = Model(lognormal_c)

        iparams = im.make_params(s=0.3, mu=4.3, h=16.5)
        iparams['s'].min = 0; iparams['h'].min = 0
        iresult = im.fit(log(yi+1), iparams, x=x)
        st.text(f'---- {what_to_display}:\n' + iresult.fit_report())
        label_ = (f"{what_to_display} fit")
        ax.plot(indates, exp(iresult.best_fit)-1, 'b', label=label_)
        ipred = iresult.eval(x=exrange)
        ax.plot(exdates, exp(ipred)-1, 'b--',
            label='Forecast: {:,.0f}'.format(exp(ipred[-1])-1))
        iupred = iresult.eval_uncertainty(x=exrange, sigma=0.95) # 95% interval
        iintlow = clip(ipred-iupred, ipred[0], None)
        put(iintlow, range(argmax(iintlow), len(iintlow)), iintlow[argmax(iintlow)])
        ax.fill_between(exdates, exp(iintlow), exp(ipred+iupred), alpha=0.35, color='b')

        #dm = Model(normal_c)
        # dm = Model(lognormal_c)

        # dparams = dm.make_params(s=19.8, mu=79.1, h=11.4) # initial guesses
        # dparams['s'].min = 0; iparams['h'].min = 0
        # dresult = dm.fit(log(yd+1), dparams, x=x)
        # st.text('---- Deaths:\n' + dresult.fit_report())
        # ax.plot(indates, exp(dresult.best_fit)-1, 'r', label='Deaths fit')
        # dpred = dresult.eval(x=exrange)
        # ax.plot(exdates, exp(dpred)-1, 'r--',
        #     label='Forecast: {:,.0f}'.format(exp(dpred[-1])-1))
        # dupred = dresult.eval_uncertainty(x=exrange, sigma=0.95) # 95% interval
        # dintlow = clip(dpred-dupred, log(max(yd)+1), None)
        # put(dintlow, range(argmax(dintlow), len(dintlow)), dintlow[argmax(dintlow)])
        # ax.fill_between(exdates, exp(dintlow), exp(dpred+dupred), alpha=0.35, color='r')
        # ax.fill_between(exdates, 0.012 * (exp(iintlow)), 0.012 * (exp(ipred+iupred)),
        #     alpha=0.85, color='g', label='Deaths from observed fatality rate')

        ax.set_xlim(df.index[0], lastday)
        #ax.set_yscale('log') # semilog
        #ax.set_ylim(0, 1500000)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # comma separators
        ax.grid()
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(ConciseDateFormatter(AutoDateLocator(), show_offset=False))
        ax.set_xlabel('95% prediction confidence intervals shaded')

        #fig.savefig('plot.png', bbox_inches='tight')
        #print('\nTO VIEW GRAPH: click on plot.png in the file pane to the left.')
        #fig.show()
        st.pyplot(fig1yz)

    st.text(f"{what_to_display} at end of period shown: {int( exp(ipred[-1])-1)}.")
def main():
    #url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\EINDTABEL.csv"
    url1= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/EINDTABEL.csv"
    df = pd.read_csv(url1, delimiter=",", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df.fillna(value=0, inplace=True)
    df["Total_reported_cumm"] = df["Total_reported"].cumsum()
    df["Deceased_cumm"] = df["Deceased"].cumsum()
    #df["total_reported_k_value"] = 1- df["Total_reported_cumm"].pct_change(periods=7, fill_method='ffill')
    df["IC_Nieuwe_Opnames_LCPS_cumm"] = df["IC_Nieuwe_Opnames_LCPS"].cumsum()
    DATE_FORMAT = "%m/%d/%Y"
    global start__
    global OUTPUT_DIR
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\output\\"
    )
    scenario = st.sidebar.radio(
    "Select a datarange",
    ("Total_reported_march_2020","Total_reported_cumm_march_2020", "IC_Bedden_march_2021", "IC_opnames_march_2021", "Hosp_adm_march_2021")
    )
    if scenario =='Total_reported_march_2020':
        start__ = "2020-02-27"
        until__ = "2020-05-31"
        what_default = 0
        days_to_show = 180
        what_method_default = 1
    elif scenario =='Total_reported_cumm_march_2020':
        start__ = "2020-02-27"
        until__ = "2020-05-31"
        what_default = 1
        days_to_show = 180
        what_method_default = 0
    elif scenario =='IC_opnames_march_2021':
        start__ = "2021-03-1"
        until__ = "2021-03-31"
        what_default = 5
        days_to_show = 60
        what_method_default = 1
    elif scenario =='IC_Bedden_march_2021':
        start__ = "2021-03-1"
        until__ = "2021-03-31"
        what_default = 2
        days_to_show = 60
        what_method_default = 1
    elif scenario =='Hosp_adm_march_2021':
        start__ = "2021-03-1"
        until__ = "2021-03-31"
        what_default = 7
        days_to_show = 60
        what_method_default = 1

    else:
        st.error ("ERROR in scenario")
        st.stop()


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

    lijst = [
        "Total_reported",
        "Total_reported_cumm",
        "IC_Bedden_COVID",
        "IC_Bedden_Non_COVID",
        "Kliniek_Bedden",
        "IC_Nieuwe_Opnames_LCPS",
        "IC_Nieuwe_Opnames_LCPS_cumm",
        "Hospital_admission_RIVM",
        "Hospital_admission_LCPS",
        "Hospital_admission_GGD",
        "Deceased",
        "Deceased_cumm",
        "Tested_with_result",
        "Tested_positive",
        "Percentage_positive"]
    # for l in lijst:
    #     l_ = l + "_cumm"
    #     df[l_] = df[l].cumsum()
    #     lijst.append(l_)

    what_to_display = st.sidebar.selectbox(
            "What to display", lijst,
            index=what_default,
        )
    which_method = st.sidebar.selectbox("Which method", ["exponential", "derivate"], index=what_method_default)

    total_days = st.sidebar.number_input('Total days to show',None,None,days_to_show)

    d1 = datetime.strptime(from_, "%Y-%m-%d")
    d2 = datetime.strptime(until_, "%Y-%m-%d")
    datediff = abs((d2 - d1).days)
    if  datediff > total_days:
        st.warning("Make sure that the number of total days is bigger than the date difference")
        st.stop()
    if  datediff < 4:
        st.warning("Make sure that the date difference is at least 4 days")
        st.stop()


    df_to_use = select_period(df, FROM, UNTIL)
    df_to_use.fillna(value=0, inplace=True)

    values_to_fit = df_to_use[what_to_display].tolist()
    to_do_list = [[what_to_display, values_to_fit]]

    then = d1 + dt.timedelta(days=total_days)
    daterange = mdates.drange(d1,then,dt.timedelta(days=1))
    global prepare_for_animation
    if platform.processor() is not "":

        prepare_for_animation = st.sidebar.selectbox("Make animation (SLOW!)", [True, False], index=1)
    else:
        st.sidebar.write ("Animation disabled")
        prepare_for_animation = False

    fit_the_values(to_do_list, total_days, daterange, which_method,prepare_for_animation)
    #normal_c(df_to_use)  #FIXIT doesnt work :()
    loglognormal(df_to_use, what_to_display)
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