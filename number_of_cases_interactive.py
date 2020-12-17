# Calculate the number of cases with a decreasing R-number
# For information only. Provided "as-is" etc.

# The subplots are generated as 3 normal plots, thus repeating code :(
# 

# Import our modules that we are using
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.font_manager import FontProperties
from datetime import datetime
from matplotlib import figure


# VARIABLES
# startdate in m/d/yyyy
# https://www.bddataplan.nl/corona/

numberofcasesdayzero = 331  
numberofhospitaldayzero = 175
numberofICdayzero = 34
STARTDATE = "12/15/2020"
NUMBEROFDAYS = st.slider('Number of days in graph', 15, 150, 60)
TURNINGPOINTDAY = st.slider('Number of days needed to go to new R', 1, 30,10)

Rold = 1.24
Rnew = st.slider('R-number', 0.1, 2.0, 0.75)
# Some manipulation of the x-values

startx = dt.datetime.strptime(STARTDATE,'%m/%d/%Y').date() 
then = startx + dt.timedelta(days=NUMBEROFDAYS)
x = mdates.drange(startx,then,dt.timedelta(days=1)) 
# x = dagnummer gerekend vanaf 1 januari 1970 (?)
# y = aantal gevallen
# z = dagnummer van 1 tot NUMBEROFDAYS
z  = np.array(range(NUMBEROFDAYS))
positivetests = []
inhospital = []
inIC=[]
date_format = "%m/%d/%Y"
a = datetime.strptime(STARTDATE, date_format)

# START CALCULATING --------------------------------------------------------------------

positivetests.append (numberofcasesdayzero) 
inhospital.append(numberofhospitaldayzero)
inIC.append(numberofICdayzero)

for t in range(1, NUMBEROFDAYS):
    if t<TURNINGPOINTDAY :        
        Ry = Rold - (t/TURNINGPOINTDAY * (Rold - Rnew))
    else:
        Ry = Rnew
    
    if Ry == 1:
        # prevent an [divide by zero]-error
        Ry = 1.000001

    thalf = 4 * math.log(0.5) / math.log(Ry)  
    positivetests.append(positivetests[t-1] * (0.5**(1/thalf)))
    inhospital.append(inhospital[t-1] * (0.5**(1/thalf)))
    inIC.append(inIC[t-1] * (0.5**(1/thalf)))


st.title('Positive COVID-tests in NL')

# POS TESTS /100k inhabitants / 7days ################################
fig1, ax = plt.subplots()
plt.plot(x, positivetests)
positivetests = []

# Add X and y Label and limits
plt.xlabel('date')
plt.xlim(x[0], x[-1]) 
plt.ylabel('positive tests per 100k inhabitants in 7 days')
plt.ylim(bottom = 0)
#plt.ylim(0,450)

# add horizontal lines and surfaces
plt.fill_between(x, 0, 49, color='yellow', alpha=0.3, label='waakzaam')
plt.fill_between(x, 50, 149, color='orange', alpha=0.3, label='zorgelijk')
plt.fill_between(x, 150, 249, color='red', alpha=0.3, label='ernstig')
plt.fill_between(x, 250, 499, color='purple', alpha=0.3, label='zeer ernstig')
if Rnew>1:
    plt.fill_between(x, 500, 1000, color='grey', alpha=0.3, label='zeer zeer ernstig')

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
    ' // Rnew reached in ' + str(TURNINGPOINTDAY) + ' days (linear change)'  )
plt.title(titlex , fontsize=10)


# lay-out of the x axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.gca().set_title(titlex , fontsize=10)

st.pyplot(fig1)

################## HOSPITAL ##########################################
fig2, ax = plt.subplots()
plt.plot(x, inhospital)
inhospital = []

# Add X and y Label and limits
plt.xlabel('date')
plt.xlim(x[0], x[-1]) 
plt.ylabel('ziekenhuisopnames')
plt.ylim(bottom = 0)

# add horizontal lines and surfaces
plt.axhline(y=40, color='green', alpha=.6,linestyle='--', label = "signaalwaarde" )
plt.axvline(x=x[0]+35, color='purple', alpha=.6,linestyle='--',label = "19/01/2021")

# Add a grid
plt.grid(alpha=.4,linestyle='--')

#Add a Legend
fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend(  loc='upper right', prop=fontP)

# Add a title
titlex = (
    'Ziekenhuisopnames per dag.\n'
    'Number of cases on '+ str(STARTDATE) + ' = ' + str(numberofhospitaldayzero) + '\n'
    'Rold = ' + str(Rold) + 
    ' // Rnew reached in ' + str(TURNINGPOINTDAY) + ' days (linear change)'  )
plt.title(titlex , fontsize=10)

# lay-out of the x axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.gca().set_title(titlex , fontsize=10)

st.pyplot(fig2)

################## IC ##########################################
fig3, ax = plt.subplots()
plt.plot(x, inIC)
inIC = []

# Add X and y Label and limits
plt.xlabel('date')
plt.xlim(x[0], x[-1]) 
plt.ylabel('IC Opnames')
plt.ylim(bottom = 0)

# add horizontal lines and surfaces
plt.axhline(y=10, color='green', alpha=.6,linestyle='--', label = "signaalwaarde" )
plt.axvline(x=x[0]+35, color='purple', alpha=.6,linestyle='--',label = "19/01/2021")

# Add a grid
plt.grid(alpha=.4,linestyle='--')

#Add a Legend
fontP = FontProperties()
fontP.set_size('xx-small')
plt.legend(  loc='upper right', prop=fontP)

# Add a title
titlex = (
    'IC per dag.\n'
    'Number of cases on '+ str(STARTDATE) + ' = ' + str(numberofICdayzero) + '\n'
    'Rold = ' + str(Rold) + 
    ' // Rnew reached in ' + str(TURNINGPOINTDAY) + ' days (linear change)'  )
plt.title(titlex , fontsize=10)

# lay-out of the x axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.gcf().autofmt_xdate()
plt.gca().set_title(titlex , fontsize=10)

st.pyplot(fig3)


################################################

tekst = (
    '<hr>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\'>@rcsmit</a>) <br>'
    'Overdrachtstijd is 4 dagen. Disclaimer is following. Provided As-is etc.<br>'
    'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/edit/main/number_of_cases_interactive.py\">github.com/rcsmit</a>' )
links = (
'<h3>Useful dashboards</h3><ul>'

'<li><a href=\"https://datagraver.com/corona\">https://www.datagraver/corona/</a></li>'
'<li><a href=\"https://www.bddataplan.nl/corona\">https://www.bddataplan.nl/corona/</a></li>'
'<li><a href=\"https://renkulab.shinyapps.io/COVID-19-Epidemic-Forecasting/_w_ebc33de6/_w_dce98783/_w_0603a728/_w_5b59f69e/?tab=jhu_pred&country=France\">Dashboard by  Institute of Global Health, Geneve, Swiss</a></li>'
'<li><a href=\"https://coronadashboard.rijksoverheid.nl/\">Rijksoverheid NL</a></li>'

'</ul>')

st.markdown(tekst, unsafe_allow_html=True)
st.markdown(links, unsafe_allow_html=True)


