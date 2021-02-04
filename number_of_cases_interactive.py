# Calculate the number of cases with a decreasing R-number, 2 different variants and vaccination
# For information only. Provided "as-is" etc.

# The subplots are generated as 4 normal plots, thus repeating code :(
#
# https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py
#

# Sorry for all the commented out code, maybe I will combine the old and new version(s) later

# Import our modules that we are using
import math
from datetime import datetime

import streamlit as st
import numpy as np
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
_lock = RendererAgg.lock

# VARIABLES
# startdate in m/d/yyyy
# https://www.bddataplan.nl/corona/

# variable1 = old variant
# variable2 = new variant
# variable 12 = old + new

# variablex = with vaccination

DATE_FORMAT = "%m/%d/%Y"
b = datetime.today().strftime('%m/%d/%Y')

#values 01/13/2021, according to https://www.bddataplan.nl/corona/
st.sidebar.title('Parameters')
numberofpositivetests = st.sidebar.number_input('Total number of positive tests',None,None,3561)

st.markdown("<hr>", unsafe_allow_html=True)
a = st.sidebar.text_input('startdate (mm/dd/yyyy)',b)

try:
    startx = dt.datetime.strptime(a,'%m/%d/%Y').date()
except:
    st.error("Please make sure that the date is in format mm/dd/yyyy")
    st.stop()


NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 15, 365, 30)
if NUMBEROFDAYS >30:
    st.sidebar.text("Attention: Read the disclaimer")
Rnew1_ = st.sidebar.slider('R-number old variant', 0.1, 2.0, 0.85)

Rnew2_ = st.sidebar.slider('R-number new British variant', 0.1, 2.0, 1.3)

percentagenewversion = (st.sidebar.slider('Percentage British variant at start', 0.0, 100.0, 66.0)/100)

Tg = st.sidebar.slider('Generation time', 2.0, 11.0, 4.0)
global Tg_
Tg_=Tg
showcummulative = st.sidebar.checkbox("Show cummulative")
if NUMBEROFDAYS >30:
    st.sidebar.text("Attention: Read the disclaimer")
if showcummulative:
    numberofcasesdayz = (st.sidebar.text_input('Number active cases on day zero', 130000))

    try:
        numberofcasesdayzero = int(numberofcasesdayz)
    except:
        st.error("Please enter a number for the number of active cases on day zero")
        st.stop()


showimmunization = st.sidebar.checkbox("Immunization")
if showimmunization:
    totalimmunedayzero_ = (st.sidebar.text_input('Total immune persons day zero', 2_500_000))
    totalpopulation_ = (st.sidebar.text_input('Total population', 17_500_000))
    testimmunefactor = st.sidebar.slider('Test/immunityfactor', 0.0, 5.0, 2.5)
    try:
        totalimmunedayzero = int(totalimmunedayzero_)
    except:
        st.error("Please enter a number for the number of immune people on day zero")
        st.stop()

    try:
        totalpopulation = int(totalpopulation_)
    except:
        st.error("Please enter a number for the number ofpopulation")
        st.stop()

    st.sidebar.text("Attention: Read the disclaimer")
turning = st.sidebar.checkbox("Turning point")


if turning:
    #Rnew3 = st.sidebar.slider('R-number target British variant', 0.1, 2.0, 0.8)
    changefactor = st.sidebar.slider('Change factor (1.0 = no change)', 0.0, 3.0, 0.9)
    #turningpoint = st.sidebar.slider('Startday turning', 1, 365, 30)
    turningpointdate = st.sidebar.text_input('Turning point date (mm/dd/yyyy)', '02/28/2021')
    turningdays = st.sidebar.slider('Number of days needed to reach new R values', 1, 90, 10)
    try:
        starty = dt.datetime.strptime(turningpointdate,'%m/%d/%Y').date()
    except:
        st.error("Please make sure that the date is in format mm/dd/yyyy")
        st.stop()


    d1 = datetime.strptime(a, '%m/%d/%Y')
    d2 = datetime.strptime(turningpointdate,'%m/%d/%Y')
    if d2<d1:
        st.error("Turning point cannot be before startdate")
        st.stop()
    turningpoint =  abs((d2 - d1).days)

vaccination = st.sidebar.checkbox("Vaccination")

if vaccination:
    VACTIME = st.sidebar.slider('Number of days needed for vaccination', 1, 730, 365)
#percentagenonvacc = (st.sidebar.slider('Percentage non-vaxx', 0.0, 100.0, 20.0)/100)

if showimmunization and turning:
    st.error("Choose either Turning point OR Immunization")
    st.stop()
if showimmunization and vaccination:
    st.error("Choose either Vaccination OR Immunization")
    st.stop()

numberofpositivetests1 = numberofpositivetests*(1-percentagenewversion)
numberofpositivetests2 = numberofpositivetests*(percentagenewversion)
numberofpositivetests12 = numberofpositivetests

# Some manipulation of the x-values (the dates)
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
ratio=[]
walkingR=[]
actualR=[]
totalimmune=[]

#if vaccination:
ry1x = []
ry2x = []
if turning == False:
    label1= 'Old variant (R='+ str(Rnew1_) + ')'
    label2= 'New variant (R='+ str(Rnew2_) + ')'
else:
    label1= 'Old variant'
    label2= 'New variant'

# START CALCULATING --------------------------------------------------------------------
positivetests1.append (numberofpositivetests1)
positivetests2.append (numberofpositivetests2)
positivetests12.append (numberofpositivetests12)
positivetestsper100k.append ((numberofpositivetests12/25))

if showcummulative:
    cummulative1.append(numberofcasesdayzero*(1-percentagenewversion))
    cummulative2.append(numberofcasesdayzero*(percentagenewversion))
    cummulative12.append(numberofcasesdayzero)
ratio.append(percentagenewversion*100 )
#walkingcummulative.append(1)
#if vaccination:
ry1x.append(Rnew1_)
ry2x.append(Rnew2_)
immeratio_=[]
immeratio_.append(1)
walkingR.append((Rnew1_**(1-percentagenewversion))*(Rnew2_**(percentagenewversion)))
if showimmunization:
    totalimmune.append(totalimmunedayzero)

for t in range(1, NUMBEROFDAYS):


    if turning:
        if (t>=(turningpoint-1) and t<turningpoint+turningdays):
            Rnew31 = Rnew1_ * changefactor
            Rnew32 = Rnew2_ * changefactor
            Rnew1 = Rnew31+ ((Rnew1_ - Rnew31)*(1-((t-(turningpoint-1))/turningdays)))
            Rnew2 = Rnew32+ ((Rnew2_ - Rnew32)*(1-((t-(turningpoint-1))/turningdays)))
        elif t>=turningpoint-1+turningdays:
            Rnew1 = Rnew31
            Rnew2 = Rnew32
        else:
            Rnew1 = Rnew1_
            Rnew2 = Rnew2_
    else:
        Rnew1 = Rnew1_
        Rnew2 = Rnew2_

    if vaccination:
        if t>7:
            if t<(VACTIME+7) :
                ry1 = Rnew1 * ((1-((t-7)/(VACTIME))))
                ry2 = Rnew2 * ((1-((t-7)/(VACTIME))))
                #ry1 = Rnew1 *     ((100-((100-percentagenonvacc)*   (t/VACTIME) )  )      /100)
                #ry2 = Rnew2 *     ((100-((100-percentagenonvacc)*   (t/VACTIME) )  )      /100)

            else:
                ry1 = Rnew1 * 0.0000001
                ry2 = Rnew2 * 0.0000001
        else:
            ry1 = Rnew1
            ry2 = Rnew2
    else:
        ry1 = Rnew1
        ry2 = Rnew2

    if showimmunization:
        immeratio = (1-( (totalimmune[t-1]-totalimmune[0])/(totalpopulation-totalimmune[t-1])))
        #st.write (str((totalimmune[t-1]-totalimmune[0])) + " /  " + str((totalpopulation-totalimmune[t-1])) + "  =  "+ str(immeratio))
        ry1 = ry1x[0]*immeratio
        ry2 = ry2x[0]*immeratio
        immeratio_.append(immeratio)

        #st.write (str(ry1) + "  " + str(ry2))
    else:
        Rnew1 = Rnew1_
        Rnew2 = Rnew2_


    if ry1 == 1:
        # prevent an [divide by zero]-error
        ry1 = 1.000001
    if ry2 == 1:
        # prevent an [divide by zero]-error
        ry2 = 1.000001

    thalf1 = Tg * math.log(0.5) / math.log(ry1)
    thalf2 = Tg * math.log(0.5) / math.log(ry2)

    pt1 = (positivetests1[t-1] * (0.5**(1/thalf1)))
    pt2 = (positivetests2[t-1] * (0.5**(1/thalf2)))
    positivetests1.append(pt1)
    positivetests2.append(pt2)

    # This formula works also and gives same results
    # https://twitter.com/hk_nien/status/1350953807933558792
    # positivetests1a.append(positivetests1a[t-1] * (ry1**(1/Tg)))
    # positivetests2a.append(positivetests2a[t-1] * (ry2**(1/Tg)))
    # positivetests12a.append(positivetests2a[t-1] * (ry2**(1/Tg))+ positivetests1[t-1] * (ry1**(1/Tg)))

    positivetests12.append(pt1+pt2)

    if showcummulative:
        cpt1 = (cummulative1[t-1]+  pt1)
        cpt2 = (cummulative2[t-1]+  pt2 )
        cpt12 =  (cummulative12[t-1]+ pt1 + pt2)
        cummulative1.append   (cpt1)
        cummulative2.append   (cpt2 )
        cummulative12.append   (cpt12)
    ratio_ =  ((pt2/(pt1+pt2)))
    ratio.append   (100*ratio_)
    positivetestsper100k.append((pt1+pt2)/25)
    if showimmunization:
        totalimmune.append(totalimmune[t-1]+((pt1+pt2)*testimmunefactor))

    ry1x.append(ry1)
    ry2x.append(ry2)
    walkingR.append((ry1**(1-ratio_))*(ry2**(ratio_)))


st.title('Positive COVID-tests in NL')

disclaimernew=('<style> .infobox {  background-color: lightyellow; padding: 10px;margin: 20-px}</style>'
               '<div class=\"infobox\"><p>Attention: these results are different from the official models probably due to different (secret) parameters. </p>'
               '<p><b>This model is a simple growth model and doesn\'t take immunity into account like SEIR-models. </b></p>'
               '<p>The goal was/is to show the (big) influence of (small) changes in the R-number. '
               'In reality the curves will flatten and the numbers will drop due to measures, immunity and/or vaccination at a certain moment. '
               'At the bottom of the page are some links to SEIR models.</p></div>')
#like shown in https://twitter.com/gerardv/status/1351186187617185800<br>'
#'Parameters adapted at 24/01 to align with the graph shown in https://twitter.com/DanielTuijnman/status/1352250384077750274/photo/2')

st.markdown(disclaimernew,  unsafe_allow_html=True)
if showimmunization:
    disclaimerimm = ('<h1>Immunization<h1><p>The immunization is very speculative.'
        'Inspired by <a href=\'https://twitter.com/RichardBurghout/status/1357044694149128200\' target=\'_blank\'>this tweet</a>.<br> '
        'A lot of factors are not taken into account. Illustration puropose only.<br>'
        'The number of test is multiplied by' +str(testimmunefactor)+ ' to get the extra number of people immune <br><br>'
        '<font face=\'courier new\'>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        '   people immune<sub>t</sub> - people immune<sub>0</sub><br>'
        'R<sub>t</sub> =  R<sub>0</sub> * 1 - ( --------------------------------------<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                       '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'

        'total population - people immune<sub>0</sub>   </font> </p>')

    st.markdown(disclaimerimm, unsafe_allow_html=True)


def th2r(rz):
    Tg_=4
    th = int( Tg_ * math.log(0.5) / math.log(rz))
    return th

def r2th(th):
    Tg_=4
    r = int(10**((Tg_*mat.log(2))/th))
    # HK is using  r = 2**(Tg_/th)
    return r

def getsecondax():
    # get second y axis
    # Door Han-Kwang Nienhuys - MIT License
    # https://github.com/han-kwang/covid19/blob/master/nlcovidstats.py
    ax2 = ax.twinx()
    T2s = np.array([-2, -4,-7, -10, -11,-14, -21, -60, 9999, 60, 21, 14, 11,10, 7, 4, 2])
    y2ticks = 2**(Tg_/T2s)
    y2labels = [f'{t2 if t2 != 9999 else "∞"}' for t2 in T2s]
    ax2.set_yticks(y2ticks)
    ax2.set_yticklabels(y2labels)
    ax2.set_ylim(*ax.get_ylim())
    ax2.set_ylabel('Halverings-/verdubbelingstijd (dagen)')

def configgraph(titlex):
    plt.xlabel('date')
    plt.xlim(x[0], x[-1])

    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='best', prop=fontP)
    plt.title(titlex , fontsize=10)

    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()
    plt.gca().set_title(titlex , fontsize=10)

# POS TESTS /day ################################
with _lock:
    fig1, ax = plt.subplots()
    plt.plot(x, positivetests1, label=label1,  linestyle='--')
    plt.plot(x, positivetests2, label=label2,  linestyle='--')
    plt.plot(x, positivetests12, label='Total')
    positivetests1 = []
    positivetests2 = []
    positivetest12 = []
    # Add X and y Label and limits

    plt.ylabel('positive tests per day')
    plt.ylim(bottom = 0)

    plt.fill_between(x, 0, 875, color='#f392bd',  label='waakzaam')
    plt.fill_between(x, 876, 2500, color='#db5b94',  label='zorgelijk')
    plt.fill_between(x, 2501, 6250, color='#bc2165',  label='ernstig')
    plt.fill_between(x, 6251, 10000, color='#68032f', label='zeer ernstig')
    if Rnew2>1:
        plt.fill_between(x, 10000, 20000, color='grey', alpha=0.3, label='zeer zeer ernstig')

    # Add a title
    titlex = (
        'Pos. tests per day.\n'
        'Number of cases on '+ str(a) + ' = ' + str(numberofpositivetests) + '\n')
    configgraph(titlex)
    st.pyplot(fig1)

# # POS TESTS per 100k per week ################################
with _lock:
    fig1d, ax = plt.subplots()
    plt.plot(x, positivetestsper100k)
    positivetestsper100k = []

    # Add X and y Label and limits
    plt.ylabel('new positive tests per 100k per week')
    plt.ylim(bottom = 0)

    # add horizontal lines and surfaces
    plt.fill_between(x, 0, 35, color='yellow', alpha=0.3, label='waakzaam')
    plt.fill_between(x, 36, 100, color='orange', alpha=0.3, label='zorgelijk')
    plt.fill_between(x, 101, 250, color='red', alpha=0.3, label='ernstig')
    plt.fill_between(x, 251, 500, color='purple', alpha=0.3, label='zeer ernstig')
    if Rnew2>1:
        plt.fill_between(x, 501, 1000, color='grey', alpha=0.3, label='zeer zeer ernstig')

    plt.axhline(y=0, color='green', alpha=.6,linestyle='--' )
    plt.axhline(y=49, color='yellow', alpha=.6,linestyle='--')
    plt.axhline(y=149, color='orange', alpha=.6,linestyle='--')
    plt.axhline(y=249, color='red', alpha=.6,linestyle='--')
    plt.axhline(y=499, color='purple', alpha=.6,linestyle='--')

    # Add a title
    titlex = ( 'New pos. tests per 100k per week.\n'  )
    configgraph(titlex)

    st.pyplot(fig1d)

# Show cummulative cases
if showcummulative:
    with _lock:
        fig1e, ax = plt.subplots()
        plt.plot(x,cummulative1, label='Cummulative cases old variant',  linestyle='--')
        plt.plot(x,cummulative2, label='Cummulative cases new variant',  linestyle='--')
        plt.plot(x,cummulative12, label='Cummulative cases TOTAL',)

        cummulative1 = []
        cummulative2 = []
        cummulative12 = []
        # Add X and y Label and limits

        plt.ylabel('Total cases')
        plt.ylim(bottom = 0)

        # Add a title
        titlex = ('Cummulative cases\nNo recovery/death in this graph')

        configgraph(titlex)

        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        st.pyplot(fig1e)

# Show the percentage new variant
with _lock:
    fig1f, ax = plt.subplots()
    plt.plot(x, ratio, label='Ratio',  linestyle='--')
    ratio = []

    # Add  y Label and limits
    plt.ylabel('ratio')
    plt.ylim(bottom = 0)

    # Add a title
    titlex = ('Percentage new variant.\n')
    configgraph(titlex)
    plt.axhline(y=50, color='yellow', alpha=.6,linestyle='--')
    st.pyplot(fig1f)


# Show the R number in time
with _lock:
    fig1f, ax = plt.subplots()
    plt.plot(x, walkingR, label='Combined R number in time',  linestyle='--')
    plt.plot(x, ry1x, label='Old variant',  linestyle='--')
    plt.plot(x, ry2x, label='New variant',  linestyle='--')

    walkingR = []

    # Add X and y Label and limits
    plt.ylabel('R-number')
    #plt.ylim(bottom = 0)

    # Add a title
    titlex = ('R number in time.\n')
    plt.title(titlex , fontsize=10)
    configgraph(titlex)
    plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
    getsecondax()
    #secax = ax.secondary_yaxis('right', functions=(r2th,th2r))
    #secax.set_ylabel('Thalf')
    st.pyplot(fig1f)

if (vaccination or turning):
# Show the changing R number of the two variants
    with _lock:
        fig1c, ax = plt.subplots()
        plt.plot(x, ry1x, label='Old variant',  linestyle='--')
        plt.plot(x, ry2x, label='New variant',  linestyle='--')


        # Add X and y Label and limits
        plt.ylabel('R')
        #plt.ylim(bottom = 0)

        # Add a title
        titlex = ('The two R numbers in time.\n')

        configgraph(titlex)
        getsecondax()
        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')

        st.pyplot(fig1c)

ry1x = []
ry2x = []

################################################

tekst = (
    '<style> .infobox {  background-color: lightblue; padding: 5px;}</style>'
    '<hr><div class=\'infobox\'>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\' target=\"_blank\">@rcsmit</a>) <br>'
    'Overdrachtstijd is 4 dagen. Disclaimer is following. Provided As-is etc.<br>'
    'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/edit/main/number_of_cases_interactive.py\" target=\"_blank\">github.com/rcsmit</a><br>'
    'How-to tutorial : <a href=\"https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d\" target=\"_blank\">rcsmit.medium.com</a><br>'
    'Inspired by <a href=\"https://twitter.com/mzelst/status/1350923275296251904\" target=\"_blank\">this tweet</a> of Marino van Zelst.<br>'
    'With help of <a href=\"https://twitter.com/hk_nien" target=\"_blank\">Han-Kwang Nienhuys</a>.</div>')

links = (
'<h3>Useful dashboards</h3><ul>'
'<li><a href=\"https://allecijfers.nl/nieuws/statistieken-over-het-corona-virus-en-covid19/\" target=\"_blank\">Allecijfers.nl</a></li>'
'<li><a href=\"https://datagraver.com/corona\" target=\"_blank\">https://www.datagraver/corona/</a></li>'
'<li><a href=\"https://www.bddataplan.nl/corona\" target=\"_blank\">https://www.bddataplan.nl/corona/</a></li>'
'<li><a href=\"https://renkulab.shinyapps.io/COVID-19-Epidemic-Forecasting/_w_ebc33de6/_w_dce98783/_w_0603a728/_w_5b59f69e/?tab=jhu_pred&country=France\" target=\"_blank\">Dashboard by  Institute of Global Health, Geneve, Swiss</a></li>'
'<li><a href=\"https://coronadashboard.rijksoverheid.nl/\" target=\"_blank\">Rijksoverheid NL</a></li>'
'<li><a href=\"https://www.corona-lokaal.nl/locatie/Nederland\" target=\"_blank\">Corona lokaal</a></li>'
'</ul>'


'<h3>Other (SEIR) models</h3><ul>'
'<li><a href=\"http://gabgoh.github.io/COVID/index.html\" target=\"_blank\">Epidemic Calculator </a></li>'
'<li><a href=\"https://covid19-scenarios.org/\" target=\"_blank\">Covid scenarios</a></li>'
'<li><a href=\"https://share.streamlit.io/lcalmbach/pandemic-simulator/main/app.py\" target=\"_blank\">Pandemic simulator</a></li>'
'<li><a href=\"https://penn-chime.phl.io/\" target=\"_blank\">Hospital impact model</a></li>')


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
