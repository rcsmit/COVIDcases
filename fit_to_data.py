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
    except:
        print("Error with exponential fit")

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
    #     print("Error with growth model fit")
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
    except:
        print("Error with derivate")
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
    except:
        print("Error with Guassian")

    plt.scatter(x_values_extra, y_values_extra, s=20, color="#00b3b3", label="Data")
    plt.legend()
    plt.title(f"{title} / curve_fit (scipy)")
    plt.ylim(bottom=0)
    plt.savefig("dummy_dataset_exponential_fit.png", dpi=100, bbox_inches="tight")
    plt.show()


def use_lmfit(x_values, y_values, functionlist, title):
    """
    Use lmfit.
    IN : x- and y-values.
         functionlist (which functions to use)

          adapted from https://stackoverflow.com/a/49843706/4173718

    TODO: Make all graphs in one graph
    """

    for function in functionlist:

        # create a Model from the model function
        if function == "exponential":
            bmodel = Model(exponential)
        elif function == "derivate":
            bmodel = Model(derivate)
        elif function == "gaussian":
            bmodel = Model(gaussian_2)
        else:
            print("Please choose a function")
            exit()

        # create Parameters, giving initial values
        params = bmodel.make_params(a=500, b=25, c=0.5)
        # params = bmodel.make_params()
        params["a"].min = 0
        params["b"].min = 0
        params["c"].min = 0

        # do fit, print result
        result = bmodel.fit(y_values, params, x=x_values)
        print(result.fit_report())

        # plot results -- note that `best_fit` is already available
        plt.plot(x_values, y_values, "bo")
        plt.plot(x_values, result.best_fit, "r--")
        plt.title(f"{title} / lmfit - {function}")
        t = np.linspace(0.0, TOTAL_DAYS_IN_GRAPH, 10000)

        # use `result.eval()` to evaluate model given params and x
        plt.plot(t, bmodel.eval(result.params, x=t), "k-")
        plt.ylim(bottom=0)
        plt.xlabel("Days")
        plt.ylabel("Y-value")
        plt.show()

def fit_the_values(to_do_list):
    """
    We are going to fit the values

    """
    for v in to_do_list:
        title = v[0]
        y_values = v[1]
        # some preperations
        number_of_y_values = len(y_values)
        global TOTAL_DAYS_IN_GRAPH
        TOTAL_DAYS_IN_GRAPH = 60  # number of total days
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
        #use_lmfit(x_values,y_values, ["exponential", "derivate", "gaussian"], title)
        use_lmfit(x_values,y_values, ["exponential"], title)


def main():
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
    to_do_list = [["Total reported", total_reported],
                ["IC bedden", ic_bedden],
                ["IC opnames", ic_opnames]]
    fit_the_values(to_do_list)

if __name__ == "__main__":
    main()
