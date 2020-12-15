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

start = [331]
# STARTDATE = "11/26/2020"


STARTDATE = "12/15/2020"

NUMBEROFDAYS = 60

# R-numbers. Decrease and increase in two seperate figures
Rz = [[0.95,0.9,0.85,0.8,1.05,1.1,1.2]]

# Some manipulation of the x-values
startx = dt.datetime.strptime(STARTDATE,'%m/%d/%Y').date() 
then = startx + dt.timedelta(days=NUMBEROFDAYS)
x = mdates.drange(startx,then,dt.timedelta(days=1))
z  = np.array(range(NUMBEROFDAYS))
print (startx)
print (x)
date_format = "%m/%d/%Y"
a = datetime.strptime(STARTDATE, date_format)
print (a)
# Here we go
for s in start:
    for Rx in Rz:
        for R in Rx:
        # nested list because first I had two graphs (one for r>1 and another one for r<1)
            thalf = 4 * math.log(0.5) / math.log(R)  
            y = s * (0.5**(z/thalf))
            labelx = 'R = ' + str(R)
            #labelx = 'cases op 15 dec = ' + str(s)
            # Create the plot
            plt.plot(x,y,label=labelx)

# Add a title
#titlex = 'COVID cases in time depending on R - (' + str(START) + ' cases on ' + str(STARTDATE) + ')'  
titlex = 'Pos. tests per 100k inhabitants in 7 days '  
plt.title(titlex)

# Add X and y Label
plt.xlabel('date')
plt.ylabel('positive tests per 100k inhabitants in 7 days')
plt.ylim(top=600)

plt.xlim( x[0], x[-1]) 
# add horizontal lines
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
#plt.axhline(y=1000, color='orange', alpha=.6,linestyle='--')

# Add a grid
plt.grid(alpha=.4,linestyle='--')

#Add a Legend
plt.legend()


fontP = FontProperties()
fontP.set_size('xx-small')

plt.legend(  loc='upper right', prop=fontP)
# lay-out of the x axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()

# Show the plot
plt.show()
  
