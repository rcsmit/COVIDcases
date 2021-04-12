# FIT DATA TO A CURVE
# René Smit - MIT Licence

# inspired by @dimgrr. Based on
# https://towardsdatascience.com/basic-curve-fitting-of-scientific-data-with-python-9592244a2509?gi=9c7c4ade0880
# https://github.com/venkatesannaveen/python-science-tutorial/blob/master/curve-fitting/curve-fitting-tutorial.ipynb

# Import required packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import copy, math
from lmfit import Model
import pandas as pd
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

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

# #####################################################################
def use_curvefit(x_values, x_values_extra, y_values, y_values_extra, title):
    """
    Use the curve-fit from scipy.
    IN : x- and y-values. The ___-extra are for "predicting" the curve
    """
    st.subheader(f"Curvefit (scipy) - {title}")
    with _lock:
        fig1x = plt.figure()
        try:
            popt, pcov = curve_fit(
            f=exponential,
            xdata=x_values,
            ydata=y_values,
            p0=[0, 0, 0],
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
                "r-",
                label="gaussian fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt_g),
            )
        except RuntimeError as e:
            str_e = str(e)
            st.error(f"Gaussian fit :\n{str_e}")


        plt.scatter(x_values_extra, y_values_extra, s=20, color="#00b3b3", label="Data")
        plt.legend()
        plt.title(f"{title} / curve_fit (scipy)")
        plt.ylim(bottom=0)
        #plt.show()
        st.pyplot(fig1x)


def use_lmfit(x_values, y_values, functionlist, title):
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
            exit()

        # create Parameters, giving initial values
        params = bmodel.make_params(a=500, b=25, c=0.5)
        # params = bmodel.make_params()
        params["a"].min = 0
        params["b"].min = 0
        params["c"].min = 0

        # do fit, st.write result
        result = bmodel.fit(y_values, params, x=x_values)
        st.text(result.fit_report())
        with _lock:
            fig1y = plt.figure()
            # plot results -- note that `best_fit` is already available
            plt.scatter(x_values, y_values, color="#00b3b3")
            plt.plot(x_values, result.best_fit, "r--")
            res = (f"a: {result.params['a'].value} / b: {result.params['b'].value} / c: {result.params['c'].value}")
            plt.title(f"{title} / lmfit - {function}\n{formula}\n{res}")
            t = np.linspace(0.0, TOTAL_DAYS_IN_GRAPH, 10000)

            # use `result.eval()` to evaluate model given params and x
            plt.plot(t, bmodel.eval(result.params, x=t), "r-")
            plt.ylim(bottom=0)
            plt.xlabel("Days")
            plt.ylabel(title)
            #plt.show()
            st.pyplot(fig1y)

def fit_the_values(to_do_list , total_days):
    """
    We are going to fit the values

    """
    for v in to_do_list:
        title = v[0]
        y_values = v[1]
        # some preperations
        number_of_y_values = len(y_values)
        global TOTAL_DAYS_IN_GRAPH
        TOTAL_DAYS_IN_GRAPH = total_days  # number of total days
        x_values = np.linspace(start=0, stop=number_of_y_values - 1, num=number_of_y_values)
        x_values_extra = np.linspace(
            start=0, stop=TOTAL_DAYS_IN_GRAPH - 1, num=TOTAL_DAYS_IN_GRAPH
        )

        # make a longer list for the prediction
        y_values_extra = copy.deepcopy(y_values)
        d = len(x_values_extra) - len(y_values_extra)
        for i in range(0, d):
            y_values_extra.append(None)

        # Here we go !
        use_curvefit(x_values, x_values_extra, y_values, y_values_extra, title)
        use_lmfit(x_values,y_values, ["exponential", "derivate", "gaussian"], title)


def main_oud():
    """
    Main function.

    Manually input:  a list of the values to fit as a list as 'y_values'
    """
    # Total reported vanaf 27-2-2020
    total_reported = [ 0, 1, 1, 3, 5, 4, 10, 17, 40, 50, 32, 61, 130, 123, 127,
                 113, 219, 173, 270, 288, 347, 396, 533, 637, 605, 553, 749,
                 843, 1007, 1167, 1146, 1092, 874, 836, 1008, 1083, 1020, 1002, 1099]

    # IC Bedden vanaf 1 maart 2021
    ic_bedden = [536.0, 540.0, 549.0, 539.0, 541.0, 542.0, 545.0, 558.0,
                  556.0, 560.0, 572.0, 576.0, 554.0, 554.0, 564.0, 586.0,
                  568.0, 564.0, 579.0, 602.0, 611.0, 638.0, 623.0, 625.0,
                  619.0, 625.0, 643.0, 655.0, 675.0, 682.0, 681.0, 688.0,
                  705.0, 734.0, 730.0, 746.0, 750.0, 776.0, 798.0]

    # IC nieuwe opnames vanaf 1 maart 2021
    ic_opnames = [30, 52, 49, 31, 39, 36, 39, 33, 35, 28, 50, 39, 44, 37, 46,
                  38, 36, 29, 46, 54, 31, 50, 32, 37, 34, 44, 51, 41, 36, 50, 42,
                  61, 48, 54, 53, 58, 44, 57, 57]
    #to_do_list = [["Total reported", total_reported],
    #            ["IC bedden", ic_bedden],
    #            ["IC opnames", ic_opnames]]
    to_do_list = [["Total reported", total_reported]]
    fit_the_values(to_do_list)

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

def main():
    url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\EINDTABEL.csv"
    url1= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/EINDTABEL.csv"
    df = pd.read_csv(url1, delimiter=",", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    DATE_FORMAT = "%m/%d/%Y"

    scenario = st.sidebar.radio(
    "Select a datarange",
    ("Total_reported_march_2020","IC_Bedden_march_2021", "IC_opnames_march_2021")
    )
    if scenario =='Total_reported_march_2020':
        start__ = "2020-02-27"
        until__ = "2020-03-31"
        what_default = 0
    elif scenario =='IC_opnames_march_2021':
        start__ = "2021-03-1"
        until__ = "2021-03-31"
        what_default = 4

    elif scenario =='IC_Bedden_march_2021':
        start__ = "2021-03-1"
        until__ = "2021-03-31"
        what_default = 2

    else:
        st.error ("ERROR in scenario")
        st.stop()


    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start__)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", until__)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    lijst = [
        "Total_reported",
        "IC_Bedden_COVID",
        "IC_Bedden_Non_COVID",
        "Kliniek_Bedden",
        "IC_Nieuwe_Opnames_LCPS",
        "Hospital_admission_RIVM",
        "Hospital_admission_LCPS",
        "Hospital_admission_GGD",

        "Deceased",
        "Tested_with_result",
        "Tested_positive",
        "Percentage_positive"
    ]

    what_to_display = st.sidebar.selectbox(
            "What to display", lijst,
            index=what_default,
        )
    total_days = st.sidebar.number_input('Total days to show',None,None,60)

    d1 = datetime.strptime(from_, "%Y-%m-%d")
    d2 = datetime.strptime(until_, "%Y-%m-%d")
    datediff = abs((d2 - d1).days)
    if  datediff > total_days:
        st.warning("Make sure that the number of total days is bigger than the date difference")
        st.stop()


    df_to_use = select_period(df, FROM, UNTIL)
    values_to_fit = df_to_use[what_to_display].tolist()
    to_do_list = [[what_to_display, values_to_fit]]
    fit_the_values(to_do_list, total_days)

if __name__ == "__main__":
    main()
