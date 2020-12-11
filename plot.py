# Import our modules that we are using
import matplotlib.pyplot as plt
import numpy as np
import math

import matplotlib.dates as mdates
import datetime as dt

# VARIABLES
# number of 'besmettelijken' on 26th of November 2020 in the Netherlands
# startdate in m/d/yyyy

START = 87875
# STARTDATE = "11/26/2020"


STARTDATE = "10/26/2020"

NUMBEROFDAYS = 100

# R-numbers. Decrease and increase in two seperate figures
Rz = [[0.75,0.8,0.85, 0.9,0.95,0.98,0.99], [1.01,1.05,1.1,1.2]]

# Some manipulation of the x-values
start = dt.datetime.strptime(STARTDATE,'%m/%d/%Y').date() 
then = start + dt.timedelta(days=NUMBEROFDAYS)
x = mdates.drange(start,then,dt.timedelta(days=1))
z  = np.array(range(NUMBEROFDAYS))

# Here we go
for Rx in Rz:
    for R in Rx:
        thalf = 4 * math.log(0.5) / math.log(R)  
        y = START * (0.5**(z/thalf))
        labelx = 'R = ' + str(R)

        # Create the plot
        plt.plot(x,y,label=labelx)

    # Add a title
    titlex = 'COVID cases in time depending on R - (' + str(START) + ' cases on ' + str(STARTDATE) + ')'  
    plt.title(titlex)

    # Add X and y Label
    plt.xlabel('date')
    plt.ylabel('number of cases')

    # add horizontal lines
    plt.axhline(y=START/2, color='r', alpha=.4,linestyle='--')
    plt.axhline(y=START/4, color='g', alpha=.4,linestyle='--')
    plt.axhline(y=START/8, color='b', alpha=.4,linestyle='--')
    plt.axhline(y=START/16, color='r', alpha=.4,linestyle='--')


    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    # Add a Legend
    plt.legend()
    
    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()

    # Show the plot
    plt.show()
  
