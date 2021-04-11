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
import copy
from lmfit import Model

# Function to calculate the exponential with constants a and b
def exponential(x, a, b,c):
    return   a*np.exp(-b * np.exp(-c*x))
def derivate(x,a,b,c):
    return np.exp(b*(-1*np.exp(-c*x))-c*x) * a * b * c
    #return a*b*c*np.exp(-b*np.exp(-c*x))*np.exp(-c*x)
def gaussian(x,a,b,c):
    return  a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))

def gaussian_2(x, a, b, c):   # (x, amp, cen, wid)
    return a * np.exp(-(x-b)**2 / c)

def use_curvefit(x_values, x_values_extra, y_values, y_values_extra):
    try:
        popt, pcov = curve_fit(f=exponential, xdata=x_values, ydata=y_values,  p0=[0,0, 0], bounds=(-np.inf, np.inf), maxfev=10000)
        plt.plot(x_values_extra, exponential(x_values_extra, *popt), 'r-',
                label='exponential fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    except:
        print ("error with exponential fit")
    try:
        popt_d, pcov_d = curve_fit(f=derivate, xdata=x_values, ydata=y_values,  p0=[0,0, 0], bounds=(-np.inf, np.inf), maxfev=10000)
        plt.plot(x_values_extra, derivate(x_values_extra, *popt_d), 'g-',
                label='derivate fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_d))
    except:
        print ("Error with derivate")
    try:
        popt_g,pcov_g = curve_fit(f=gaussian_2, xdata=x_values, ydata=y_values, p0=[0.1, 0.1, 0.1], bounds=(-np.inf, np.inf), maxfev=10000)
        plt.plot(x_values_extra, gaussian_2(x_values_extra, *popt_g), 'r-',
             label='gaussian fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_g))
    except:
        print ("Error with Guassian")

    plt.scatter(x_values_extra, y_values_extra, s=20, color='#00b3b3', label='Data')
    plt.legend()
    plt.title("curve_fit (scipy)")
    plt.savefig('dummy_dataset_exponential_fit.png', dpi=100, bbox_inches='tight')
    plt.show()

def use_lmfit(x_values, y_values, functionlist):
    # adapted from https://stackoverflow.com/a/49843706/4173718
    for function in functionlist:

        # create a Model from the model function
        if function == "exponential":
            bmodel = Model(exponential)
        elif function == "derivate":
            bmodel = Model(derivate)
        elif function == "gaussian":
            bmodel = Model(gaussian_2)
        else:
            print ("Please choose a function")
            exit()

        # create Parameters, giving initial values
        params = bmodel.make_params(a=500, b=25, c=0.5)
        #params = bmodel.make_params()
        params['a'].min = 0
        params['b'].min = 0
        params['c'].min = 0

        # do fit, print result
        result = bmodel.fit(y_values, params, x=x_values)
        print(result.fit_report())

        # plot results -- note that `best_fit` is already available
        plt.plot(x_values, y_values, 'bo')
        plt.plot(x_values, result.best_fit, 'r--')
        plt.title(f"lmfit - {function}")
        t=np.linspace(0.,TOTAL_DAYS_IN_GRAPH,10000)

        # use `result.eval()` to evaluate model given params and x
        plt.plot(t, bmodel.eval(result.params, x=t), 'k-')

        plt.xlabel('Days')
        plt.ylabel('Y-value')
        plt.show()



def main():
    # Total reported vanaf 27-2-2020
    #y_values = [0, 1, 1, 3, 5, 4, 10, 17, 40, 50, 32, 61, 130, 123, 127, 113, 219, 173, 270, 288, 347, 396, 533, 637, 605, 553, 749, 843, 1007, 1167, 1146, 1092, 874, 836, 1008, 1083, 1020, 1002, 1099]

    # IC Bedden vanaf 1 maart 2021
    y_values = [536.0, 540.0, 549.0, 539.0, 541.0, 542.0, 545.0, 558.0, 556.0, 560.0, 572.0, 576.0, 554.0, 554.0, 564.0, 586.0, 568.0, 564.0, 579.0, 602.0, 611.0, 638.0, 623.0, 625.0, 619.0, 625.0, 643.0, 655.0, 675.0, 682.0, 681.0, 688.0, 705.0, 734.0, 730.0, 746.0, 750.0, 776.0, 798.0]

    # IC nieuwe opnames vanaf 1 maart 2021
    #y_values = [30, 52, 49, 31, 39, 36, 39, 33, 35, 28, 50, 39, 44, 37, 46, 38, 36, 29, 46, 54, 31, 50, 32, 37, 34, 44, 51, 41, 36, 50, 42, 61, 48, 54, 53, 58, 44, 57, 57]


    # Generate dummy dataset
    number_of_y_values = len(y_values)
    global TOTAL_DAYS_IN_GRAPH
    TOTAL_DAYS_IN_GRAPH = 60  # number of total days
    x_values = np.linspace(start=0, stop=number_of_y_values-1, num=number_of_y_values)
    x_values_extra = np.linspace(start=0, stop=TOTAL_DAYS_IN_GRAPH-1, num=TOTAL_DAYS_IN_GRAPH)

    # make an lists longer for the prediction
    y_values_extra = copy.deepcopy(y_values)
    d = len(x_values_extra) - len (y_values_extra)
    for i in range(0,d):
        y_values_extra.append(None)

    use_curvefit(x_values, x_values_extra, y_values, y_values_extra)
    use_lmfit(x_values,y_values, ["exponential", "derivate", "gaussian"])


if __name__ == "__main__":
    main()
