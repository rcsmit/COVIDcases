# Calculate the number of cases with a decreasing R-number

# INPUT
# numberofcasesdayzero = Total number of positive test 100k/7days
# numberofcasesdaytotzero = Total number of pos. test per day  
# numberofhospitaldayzero = Total number of hospital
# numberofICdayzero = Total number of IC
# startdate (mm/dd/yyyy)
# NUMBEROFDAYS = Number of days in graph
# Rold = R-number old
# Rnew = R-number new
# TURNINGPOINTDAY = Number of days needed to go to new R

# For information only. Provided "as-is" etc.

# The subplots are generated as 4 normal plots, thus repeating code :(
# 
# https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_decreasing_R.py
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
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# VARIABLES
# startdate in m/d/yyyy
# https://www.bddataplan.nl/corona/

date_format = "%m/%d/%Y"
b = datetime.today().strftime('%m/%d/%Y')

#values mid december 
# numberofcasesdaytotzero = 8306  
# numberofcasesdayzero = 277 
# numberofhospitaldayzero = 175
# numberofICdayzero = 34

#values 01/13/2021, according to https://www.bddataplan.nl/corona/
st.sidebar.title('Parameters')
numberofcasesdayzero = st.sidebar.number_input('Total number of positive test 100k/7days',None,None,277)
numberofcasesdaytotzero = st.sidebar.number_input('Total number of pos. test per day',None,None,6086)  

numberofhospitaldayzero = st.sidebar.number_input('Total number of hospital',None,None,75)
numberofICdayzero = st.sidebar.number_input('Total number of IC',None,None,34) 

st.markdown("<hr>", unsafe_allow_html=True)
a = st.sidebar.text_input('startdate (mm/dd/yyyy)',b)
NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 15, 150, 60)

Rold = st.sidebar.slider('R-number old', 0.1, 2.0, 0.93)
Rnew = st.sidebar.slider('R-number new', 0.1, 2.0, 0.85)
TURNINGPOINTDAY = st.sidebar.slider('Number of days needed to go to new R', 1, 30,10)
# Some manipulation of the x-values

try:
    startx = dt.datetime.strptime(a,'%m/%d/%Y').date()
except:
    st.markdown("Please make sure that the date is in format mm/dd/yyyy")
then = startx + dt.timedelta(days=NUMBEROFDAYS)
x = mdates.drange(startx,then,dt.timedelta(days=1))
# x = dagnummer gerekend vanaf 1 januari 1970 (?)
# y = aantal gevallen
# z = dagnummer van 1 tot NUMBEROFDAYS
z  = np.array(range(NUMBEROFDAYS))
positivetests = []
positiveteststot = []
inhospital = []
inIC=[]


# START CALCULATING --------------------------------------------------------------------

positiveteststot.append (numberofcasesdaytotzero)
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
    positiveteststot.append(positiveteststot[t-1] * (0.5**(1/thalf)))

st.title('Positive COVID-tests in NL')

# POS TESTS /100k inhabitants / 7days ################################
with _lock:
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
    #plt.axvline(x=x[0]+35, color='purple', alpha=.6,linestyle='--',label = "19/01/2021")

    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='upper right', prop=fontP)

    # Add a title
    titlex = (
        'Pos. tests per 100k inhabitants in 7 days.\n'
        'Number of cases on '+ str(a) + ' = ' + str(numberofcasesdayzero) + '\n'
        'Rold = ' + str(Rold) + 
        ' // Rnew (' + str(Rnew) + ') reached in ' + str(TURNINGPOINTDAY) + ' days (linear change)'  )
    plt.title(titlex , fontsize=10)


    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    plt.gca().set_title(titlex , fontsize=10)

    st.pyplot(fig1)



# POS TESTS / 7days ################################
with _lock:
    fig1b, ax = plt.subplots()
    plt.plot(x, positiveteststot)
    positiveteststot = []

    # Add X and y Label and limits
    plt.xlabel('date')
    plt.xlim(x[0], x[-1]) 
    plt.ylabel('new positive tests per day')
    plt.ylim(bottom = 0)
    #plt.ylim(0,450)

    # add horizontal lines and surfaces
    plt.fill_between(x, 0, 1250, color='#f392bd',  label='waakzaam')
    plt.fill_between(x, 1251, 3750, color='#db5b94',  label='zorgelijk')
    plt.fill_between(x, 3751, 6250, color='#bc2165',  label='ernstig')
    plt.fill_between(x, 6251, 10000, color='#68032f', label='zeer ernstig')
    #if Rnew>1:
    #    plt.fill_between(x, 500, 1000, color='grey', alpha=0.3, label='zeer zeer ernstig')

    #plt.axhline(y=0, color='green', alpha=.6,linestyle='--' )
    #plt.axhline(y=49, color='yellow', alpha=.6,linestyle='--')
    #plt.axhline(y=149, color='orange', alpha=.6,linestyle='--')
    #plt.axhline(y=249, color='red', alpha=.6,linestyle='--')
    #plt.axhline(y=499, color='purple', alpha=.6,linestyle='--')
    #plt.axvline(x=x[0]+35, color='purple', alpha=.6,linestyle='--',label = "19/01/2021")

    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='upper right', prop=fontP)

    # Add a title
    titlex = (
        'New pos. tests per day.\n'
        'Number on '+ str(a) + ' = ' + str(numberofcasesdaytotzero) + '\n'
        'Rold = ' + str(Rold) + 
        ' // Rnew (' + str(Rnew) + ') reached in ' + str(TURNINGPOINTDAY) + ' days (linear change)'  )
    plt.title(titlex , fontsize=10)


    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    plt.gca().set_title(titlex , fontsize=10)

    st.pyplot(fig1b)

################## HOSPITAL ##########################################
with _lock:
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
    #plt.axvline(x=x[0]+35, color='purple', alpha=.6,linestyle='--',label = "19/01/2021")

    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='upper right', prop=fontP)

    # Add a title
    titlex = (
        'Ziekenhuisopnames per dag.\n'
        'Number on '+ str(a) + ' = ' + str(numberofhospitaldayzero) + '\n'
        'Rold = ' + str(Rold) + 
        ' // Rnew (' + str(Rnew) + ') reached in ' + str(TURNINGPOINTDAY) + ' days (linear change)'  )
    plt.title(titlex , fontsize=10)

    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    plt.gca().set_title(titlex , fontsize=10)

    st.pyplot(fig2)

################## IC ##########################################
with _lock:
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
    #plt.axvline(x=x[0]+35, color='purple', alpha=.6,linestyle='--',label = "19/01/2021")

    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='upper right', prop=fontP)

    # Add a title
    titlex = (
        'IC per dag.\n'
        'Number on '+ str(a) + ' = ' + str(numberofICdayzero) + '\n'
        'Rold = ' + str(Rold) + 
        ' // Rnew (' + str(Rnew) + ')reached in ' + str(TURNINGPOINTDAY) + ' days (linear change)'  )
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

'<li><a href=\"https://datagraver.com/corona\" target=\"_blank\">https://www.datagraver/corona/</a></li>'
'<li><a href=\"https://www.bddataplan.nl/corona\" target=\"_blank\">https://www.bddataplan.nl/corona/</a></li>'
'<li><a href=\"https://renkulab.shinyapps.io/COVID-19-Epidemic-Forecasting/_w_ebc33de6/_w_dce98783/_w_0603a728/_w_5b59f69e/?tab=jhu_pred&country=France\" target=\"_blank\">Dashboard by  Institute of Global Health, Geneve, Swiss</a></li>'
'<li><a href=\"https://coronadashboard.rijksoverheid.nl/\" target=\"_blank\">Rijksoverheid NL</a></li>'
'<li><a href=\"https://www.corona-lokaal.nl/locatie/Nederland\" target=\"_blank\">Corona lokaal</a></li>'
'</ul>')

st.sidebar.markdown(tekst, unsafe_allow_html=True)
st.markdown(links, unsafe_allow_html=True)
