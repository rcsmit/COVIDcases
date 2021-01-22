# Calculate the number of cases with a decreasing R-number
# For information only. Provided "as-is" etc.

# The subplots are generated as 4 normal plots, thus repeating code :(
# 
# https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py
#

# Sorry for all the commented out code, maybe I will combine the old and new version(s) later

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

# variable1 = old variant
# variable2 = new variant
# variable 12 = old + new

# variablex = with vaccination



date_format = "%m/%d/%Y"
b = datetime.today().strftime('%m/%d/%Y')

#values 01/13/2021, according to https://www.bddataplan.nl/corona/
st.sidebar.title('Parameters')
numberofpositivetests = st.sidebar.number_input('Total number of positive tests',None,None,5600)

st.markdown("<hr>", unsafe_allow_html=True)
a = st.sidebar.text_input('startdate (mm/dd/yyyy)',b)
NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 15, 365, 60)
showcummulative = st.sidebar.checkbox("Show cummulative")
if showcummulative:
    numberofcasesdayz = (st.sidebar.text_input('Number cases on day zero', 130000))
    
    try:
        numberofcasesdayzero = int(numberofcasesdayz)
    except:
            st.title("Please enter a number for the number of cases on day zero")


vaccination = st.sidebar.checkbox("Vaccination")
if vaccination:
    VACTIME = st.sidebar.slider('Number of days needed for vaccination', 1, 365, 180)
percentagenewversion = (st.sidebar.slider('Percentage British variant at start', 0.0, 100.0, 10.0)/100)
#percentagenonvacc = (st.sidebar.slider('Percentage non-vaxx', 0.0, 100.0, 20.0)/100)

Rnew1 = st.sidebar.slider('R-number old variant', 0.1, 2.0, 0.95)
Rnew2 = st.sidebar.slider('R-number new British variant', 0.1, 2.0, 1.3)
Tg = st.sidebar.slider('Generation time', 2.0, 11.0, 4.0)


numberofpositivetests1 = numberofpositivetests*(1-percentagenewversion)
numberofpositivetests2 = numberofpositivetests*(percentagenewversion)
numberofpositivetests12 = numberofpositivetests

# Some manipulation of the x-values (the dates)

try:
    startx = dt.datetime.strptime(a,'%m/%d/%Y').date() 
except:
    st.title("Please make sure that the date is in format mm/dd/yyyy")
    pass

then = startx + dt.timedelta(days=NUMBEROFDAYS)
x = mdates.drange(startx,then,dt.timedelta(days=1)) 
# x = dagnummer gerekend vanaf 1 januari 1970 (?)
# y = aantal gevallen
# z = dagnummer van 1 tot NUMBEROFDAYS
z  = np.array(range(NUMBEROFDAYS))
positivetests1 = []
positivetests2 = []
positivetests12 = []
positivetestsper100k = []
cummulative1 = []
cummulative2 = []
cummulative12 = []
#walkingcummulative = []

walkingR=[]
if vaccination:
    Ry1x = []
    Ry2x = []

# START CALCULATING --------------------------------------------------------------------
positivetests1.append (numberofpositivetests1)
positivetests2.append (numberofpositivetests2)
positivetests12.append (numberofpositivetests12) 
positivetestsper100k.append ((numberofpositivetests12/25)) 
walkingR.append(1)
if showcummulative:
    cummulative1.append(numberofcasesdayzero*(1-percentagenewversion))
    cummulative2.append(numberofcasesdayzero*(percentagenewversion))
    cummulative12.append(numberofcasesdayzero)
#walkingcummulative.append(1)
if vaccination:
    Ry1x.append(Rnew1)
    Ry2x.append(Rnew2)

for t in range(1, NUMBEROFDAYS):
    if vaccination:
        if t<VACTIME :        
            Ry1 = Rnew1 * (1-(t/VACTIME))
            Ry2 = Rnew2 * (1-(t/VACTIME))
            #Ry1 = Rnew1 *     ((100-((100-percentagenonvacc)*   (t/VACTIME) )  )      /100)
            #Ry2 = Rnew2 *     ((100-((100-percentagenonvacc)*   (t/VACTIME) )  )      /100)
        else:
            Ry1 = Rnew1 * 0.0000001 
            Ry2 = Rnew2 * 0.0000001
    else:
            Ry1 = Rnew1 
            Ry2 = Rnew2

    if Ry1 == 1:
        # prevent an [divide by zero]-error
        Ry1 = 1.000001
    if Ry2 == 1:
        # prevent an [divide by zero]-error
        Ry2 = 1.000001

    thalf1 = Tg * math.log(0.5) / math.log(Ry1)  
    thalf2 = Tg * math.log(0.5) / math.log(Ry2)

    positivetests1.append(positivetests1[t-1] * (0.5**(1/thalf1)))
    positivetests2.append(positivetests2[t-1] * (0.5**(1/thalf2)))

    # This formula works also and gives same results 
    # https://twitter.com/hk_nien/status/1350953807933558792
    # positivetests1a.append(positivetests1a[t-1] * (Ry1**(1/Tg)))
    # positivetests2a.append(positivetests2a[t-1] * (Ry2**(1/Tg)))
    # positivetests12a.append(positivetests2a[t-1] * (Ry2**(1/Tg))+ positivetests1[t-1] * (Ry1**(1/Tg)))


    positivetests12.append(positivetests2[t-1] * (0.5**(1/thalf2)) + positivetests1[t-1] * (0.5**(1/thalf1)))
    if showcummulative:
        cummulative1.append   (cummulative1[t-1]+  (positivetests1[t-1] * (0.5**(1/thalf1))))
        cummulative2.append   (cummulative2[t-1]+  (positivetests2[t-1] * (0.5**(1/thalf2))) )
        cummulative12.append   (cummulative12[t-1]+  (positivetests2[t-1] * (0.5**(1/thalf2))) + (positivetests1[t-1] * (0.5**(1/thalf1))))
    
    #walkingcummulative.append((((cummulative[t-1]+  ((positivetests2[t-1] * (0.5**(1/thalf2))) + (positivetests1[t-1] * (0.5**(1/thalf1)))))/cummulative[t-1])))
    #walkingcummulative.append(    ( ((cummulative[t-1]+  (positivetests2[t-1] * (0.5**(1/thalf2)) + positivetests1[t-1] * (0.5**(1/thalf1))))/1)))

    ratio = ((positivetests2[t-1] * (0.5**(1/thalf2)) + positivetests1[t-1] * (0.5**(1/thalf1))) / (positivetests2[t-1]+positivetests1[t-1]))
    walkingR.append(ratio**4)

    positivetestsper100k.append((positivetests2[t-1] * (0.5**(1/thalf2)) + positivetests1[t-1] * (0.5**(1/thalf1)))/25)
    
    if vaccination:
        Ry1x.append(Ry1)
        Ry2x.append(Ry2)

st.title('Positive COVID-tests in NL')

disclaimernew=('Attention: these results are different from the official models like shown in https://twitter.com/gerardv/status/1351186187617185800')
st.markdown(disclaimernew,  unsafe_allow_html=True)

# POS TESTS /day ################################
with _lock:
    fig1, ax = plt.subplots()
    plt.plot(x, positivetests1, label='Old variant',  linestyle='--')
    plt.plot(x, positivetests2, label='New variant',  linestyle='--')
    plt.plot(x, positivetests12, label='Total')
    positivetests1 = []
    positivetests2 = []
    positivetest12 = []
    # Add X and y Label and limits
    plt.xlabel('date')
    plt.xlim(x[0], x[-1]) 
    plt.ylabel('positive tests per day')
    plt.ylim(bottom = 0)
 
    plt.fill_between(x, 0, 1250, color='#f392bd',  label='waakzaam')
    plt.fill_between(x, 1251, 3750, color='#db5b94',  label='zorgelijk')
    plt.fill_between(x, 3751, 6250, color='#bc2165',  label='ernstig')
    plt.fill_between(x, 6251, 10000, color='#68032f', label='zeer ernstig')
    if Rnew2>1:
       plt.fill_between(x, 10000, 20000, color='grey', alpha=0.3, label='zeer zeer ernstig')

    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='upper left', prop=fontP)

    # Add a title
    titlex = (
        'Pos. tests per day.\n'
        'Number of cases on '+ str(a) + ' = ' + str(numberofpositivetests) + '\n')
       
    plt.title(titlex , fontsize=10)
    
    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    plt.gca().set_title(titlex , fontsize=10)

    st.pyplot(fig1)
                   
# # POS TESTS per 100k per week ################################
with _lock:
    fig1d, ax = plt.subplots()
    plt.plot(x, positivetestsper100k)
    positivetestsper100k = []

    # Add X and y Label and limits
    plt.xlabel('date')
    plt.xlim(x[0], x[-1]) 
    plt.ylabel('new positive tests per 100k per week')
    plt.ylim(bottom = 0)


    # add horizontal lines and surfaces

    plt.fill_between(x, 0, 49, color='yellow', alpha=0.3, label='waakzaam')
    plt.fill_between(x, 50, 149, color='orange', alpha=0.3, label='zorgelijk')
    plt.fill_between(x, 150, 249, color='red', alpha=0.3, label='ernstig')
    plt.fill_between(x, 250, 499, color='purple', alpha=0.3, label='zeer ernstig')
    if Rnew2>1:
        plt.fill_between(x, 500, 1000, color='grey', alpha=0.3, label='zeer zeer ernstig')
 
    plt.axhline(y=0, color='green', alpha=.6,linestyle='--' )
    plt.axhline(y=49, color='yellow', alpha=.6,linestyle='--')
    plt.axhline(y=149, color='orange', alpha=.6,linestyle='--')
    plt.axhline(y=249, color='red', alpha=.6,linestyle='--')
    plt.axhline(y=499, color='purple', alpha=.6,linestyle='--')
  
    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='upper right', prop=fontP)

    # Add a title
    titlex = (
        'New pos. tests per 100k per week.\n'  )
    plt.title(titlex , fontsize=10)


    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    plt.gca().set_title(titlex , fontsize=10)

    st.pyplot(fig1d)

if showcummulative:
    with _lock:
        fig1e, ax = plt.subplots()
        plt.plot(x,cummulative1, label='Cummulative cases old variant',  linestyle='--')
        plt.plot(x,cummulative2, label='Cummulative cases new variant',  linestyle='--')
        plt.plot(x,cummulative12, label='Cummulative cases total',)
        
        cummulative1 = []
        cummulative2 = []
        cummulative12 = []
        # Add X and y Label and limits
        plt.xlabel('date')
        plt.xlim(x[0], x[-1]) 
        plt.ylabel('Total cases')
        plt.ylim(bottom = 0)

        #Add a Legend
        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(  loc='upper right', prop=fontP)

        # Add a title
        titlex = (
            'Cummulative cases.\n'
            )
        
        plt.title(titlex , fontsize=10)

        # Add a grid
        plt.grid(alpha=.4,linestyle='--')


        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        # lay-out of the x axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        st.markdown("Attention, people don't recover in this graph")
        st.pyplot(fig1e)


if vaccination:

    with _lock:
        fig1c, ax = plt.subplots()
        plt.plot(x, Ry1x, label='Old variant',  linestyle='--')
        plt.plot(x, Ry2x, label='New variant',  linestyle='--')
        Ry1x = []
        Ry2x = []
        
        # Add X and y Label and limits
        plt.xlabel('date')
        plt.xlim(x[0], x[-1]) 
        plt.ylabel('R')
        plt.ylim(bottom = 0)

        #Add a Legend
        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(  loc='upper right', prop=fontP)

        # Add a title
        titlex = (
            'R number over time due to vaccination.\n'
            )
        
        plt.title(titlex , fontsize=10)

        # Add a grid
        plt.grid(alpha=.4,linestyle='--')


        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        # lay-out of the x axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()

        st.pyplot(fig1c)




################################################

tekst = (
    '<style> .infobox {  background-color: lightblue; padding: 5px;}</style>'
    '<hr><div class=\'infobox\'>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\' target=\"_blank\">@rcsmit</a>) <br>'
    'Overdrachtstijd is 4 dagen. Disclaimer is following. Provided As-is etc.<br>'
    'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/edit/main/number_of_cases_interactive.py\" target=\"_blank\">github.com/rcsmit</a><br>'
    'How-to tutorial : <a href=\"https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d\" target=\"_blank\">rcsmit.medium.com</a><br>'
    'Inspired by <a href=\"https://twitter.com/mzelst/status/1350923275296251904\" target=\"_blank\">this tweet</a> of Marino van Zelst</div>')
links = (
'<h3>Useful dashboards</h3><ul>'

'<li><a href=\"https://datagraver.com/corona\" target=\"_blank\">https://www.datagraver/corona/</a></li>'
'<li><a href=\"https://www.bddataplan.nl/corona\" target=\"_blank\">https://www.bddataplan.nl/corona/</a></li>'
'<li><a href=\"https://renkulab.shinyapps.io/COVID-19-Epidemic-Forecasting/_w_ebc33de6/_w_dce98783/_w_0603a728/_w_5b59f69e/?tab=jhu_pred&country=France\" target=\"_blank\">Dashboard by  Institute of Global Health, Geneve, Swiss</a></li>'
'<li><a href=\"https://coronadashboard.rijksoverheid.nl/\" target=\"_blank\">Rijksoverheid NL</a></li>'
'<li><a href=\"https://www.corona-lokaal.nl/locatie/Nederland\" target=\"_blank\">Corona lokaal</a></li>'
'</ul>')

vaccinationdisclaimer = (
'<h3>Attention</h3>'
'The plot when having vaccination is very indicative and very simplified.'
' It assumes an uniform(?) distribution of the vaccins over the population, '
' that a person who had the vaccin can\'t be sick immediately, '
'that everybody takes the vaccin, the R is equal for everybody etc. etc.')

st.sidebar.markdown(tekst, unsafe_allow_html=True)
#st.sidebar.info(tekst)
if vaccination:
    st.markdown(vaccinationdisclaimer, unsafe_allow_html=True)
st.markdown(links, unsafe_allow_html=True)

