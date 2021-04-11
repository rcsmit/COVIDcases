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

# Function to calculate the exponential with constants a and b
def exponential(x, a, b,c):
    return   a*np.exp(-b*np.exp(-c*x))
   
# Generate dummy dataset
prognose = 60  # number of extra days 
x_dummy = np.linspace(start=0, stop=38, num=39)
x_dummy_extra = np.linspace(start=0, stop=prognose-1, num=prognose)

# IC Bedden vanaf 1 maart 2021
y_dummy = [536.0, 540.0, 549.0, 539.0, 541.0, 542.0, 545.0, 558.0, 556.0, 560.0, 572.0, 576.0, 554.0, 554.0, 564.0, 586.0, 568.0, 564.0, 579.0, 602.0, 611.0, 638.0, 623.0, 625.0, 619.0, 625.0, 643.0, 655.0, 675.0, 682.0, 681.0, 688.0, 705.0, 734.0, 730.0, 746.0, 750.0, 776.0, 798.0]

# make an lists longer for the prediction
y_dummy_extra = copy.deepcopy(y_dummy)
d = len(x_dummy_extra) - len (y_dummy_extra)
for i in range(0,d):
    y_dummy_extra.append(None)

# Fit the dummy exponential data
popt, pcov = curve_fit(f=exponential, xdata=x_dummy, ydata=y_dummy,  p0=[0,0, 0], bounds=(-np.inf, np.inf))

# Draw the fitted curve
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0, 0, 1, 1])
ax.scatter(x_dummy_extra, y_dummy_extra, s=20, color='#00b3b3', label='Data')
plt.plot(x_dummy_extra, exponential(x_dummy_extra, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.legend()
plt.savefig('dummy_dataset_exponential_fit.png', dpi=100, bbox_inches='tight')
plt.show()
