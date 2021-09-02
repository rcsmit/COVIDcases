# Calculate the number of cases with a decreasing R-number
# For information only. Provided "as-is" etc.

# Import our modules that we are using
import matplotlib.pyplot as plt
import numpy as np
import math

import matplotlib.dates as mdates
import datetime as dt
from matplotlib.font_manager import FontProperties
from datetime import datetime

# VARIABLES
# number of 'besmettelijken' on 26th of November 2020 in the Netherlands
# startdate in m/d/yyyy

numberofcasesdayzero = [331]
STARTDATE = "12/15/2020"
NUMBEROFDAYS = 90
TURNINGPOINTDAY = 5
# R-numbers. Decrease and increase in two seperate figures
Rold = 1.2
Rvalues = [[0.95, 0.9,0.85, 0.8,0.75, 0.7]]
# Some manipulation of the x-values
startx = dt.datetime.strptime(STARTDATE,'%m/%d/%Y').date() 
then = startx + dt.timedelta(days=NUMBEROFDAYS)
x = mdates.drange(startx,then,dt.timedelta(days=1)) 
# x = dagnummer gerekend vanaf 1 januari 1970 (?)
# y = aantal gevallen
# z = dagnummer van 1 tot NUMBEROFDAYS
z  = np.array(range(NUMBEROFDAYS))
k = []

date_format = "%m/%d/%Y"
a = datetime.strptime(STARTDATE, date_format)

# Here we go
for s in numberofcasesdayzero:
    for Rx in Rvalues:
        for R in Rx:
        # nested list because first I had two graphs (one for r>1 and another one for r<1)
            k.append (s) 
            Rnew = R
            for t in range(1, NUMBEROFDAYS):
                if t<TURNINGPOINTDAY :        
                    Ry = Rold - (t/TURNINGPOINTDAY * (Rold - Rnew))
                else:
                    Ry = Rnew
                if Ry == 1:
                    # prevent an [divide by zero]-error
                    Ry = 1.000001
                
                thalf = 4 * math.log(0.5) / math.log(Ry)  
                k.append( k[t-1] * (0.5**(1/thalf)))
            labelx = 'Rnew = ' + str(R)
            plt.plot(x,k,label =labelx)
            k = []

# Add X and y Label and limits
plt.xlabel('date')
plt.xlim(x[0], x[-1]) 
plt.ylabel('positive tests per 100k inhabitants in 7 days')
plt.ylim(0,450)

# add horizontal lines and surfaces
plt.fill_between(x, 0, 49, color='yellow', alpha=0.3, label='waakzaam')
plt.fill_between(x, 50, 149, color='orange', alpha=0.3, label='zorgelijk')
plt.fill_between(x, 150, 249, color='red', alpha=0.3, label='ernstig')
plt.fill_between(x, 250, 499, color='purple', alpha=0.3, label='zeer ernstig')
plt.fill_between(x, 500, 600, color='grey', alpha=0.3, label='zeer zeer ernstig')

plt.axhline(y=0, color='green', alpha=.6,linestyle='--' )
plt.axhline(y=49, color='yellow', alpha=.6,linestyle='--')
plt.axhline(y=149, color='orange', alpha=.6,linestyle='--')
plt.axhline(y=249, color='red', alpha=.6,linestyle='--')
plt.axhline(y=499, color='purple', alpha=.6,linestyle='--')
plt.axvline(x=x[0]+35, color='purple', alpha=.6,linestyle='--',label = "19/01/2021")

# Add a grid
plt.grid(alpha=.4,linestyle='--')

#Add a Legend
fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend(  loc='upper right', prop=fontP)

# Add a title
titlex = (
    'Pos. tests per 100k inhabitants in 7 days.\n'
    'Number of cases on '+ str(STARTDATE) + ' = ' + str(numberofcasesdayzero) + '\n'
    'Rold = ' + str(Rold) + 
    ' // Rnew reached in ' + str(TURNINGPOINTDAY) + ' days (linear decrease)'  )
plt.title(titlex , fontsize=10)

# lay-out of the x axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()

# Show the plot
plt.show()

  
