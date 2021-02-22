# Calculate the number of cases with a decreasing R-number, 2 different variants and vaccination
# For information only. Provided "as-is" etc.

# https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py
# https://github.com/rcsmit/COVIDcases/blob/main/number_of_cases_interactive.py

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
from scipy.integrate import odeint
import pandas as pd


# VARIABLES
# startdate in mm/dd/yyyy
# variable1 = old variant
# variable2 = new variant
# variable 12 = old + new
# variablex = with vaccination

DATE_FORMAT = "%m/%d/%Y"
b = datetime.today().strftime('%m/%d/%Y')

#values 01/13/2021, according to https://www.bddataplan.nl/corona/
st.sidebar.title('Parameters')
numberofpositivetests = st.sidebar.number_input('Total number of positive tests',None,None,4600)

st.markdown("<hr>", unsafe_allow_html=True)
a = st.sidebar.text_input('startdate (mm/dd/yyyy)',"01/29/2021")

try:
    startx = dt.datetime.strptime(a,'%m/%d/%Y').date()
except:
    st.error("Please make sure that the date is in format mm/dd/yyyy")
    st.stop()

NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 15, 720, 60)
global numberofdays_
numberofdays_ = NUMBEROFDAYS

Rnew_1_ = st.sidebar.slider('R-number first variant', 0.1, 10.0, 0.84)
Rnew_2_ = st.sidebar.slider('R-number second variant', 0.1, 6.0, 1.15)
correction = st.sidebar.slider('Correction factor', 0.0, 2.0, 1.00)
Rnew1_= round(Rnew_1_ * correction,2)
Rnew2_= round(Rnew_2_ * correction,2)

percentagenewversion = (st.sidebar.slider('Percentage second variant at start', 0.0, 100.0, 43.0)/100)

Tg = st.sidebar.slider('Generation time', 2.0, 11.0, 4.0)
global Tg_
Tg_=Tg

lambdaa = st.sidebar.slider('Lambda / heterogeneity', 1.0, 6.0, 1.0)
averagedayssick = (st.sidebar.slider('Average days infectious', 1, 30, 20))
# https://www.medrxiv.org/content/10.1101/2020.09.13.20193896v1.full.pdf / page 4
showcummulative = st.sidebar.checkbox("Show cummulative / SIR", True)



showimmunization = st.sidebar.checkbox("Immunization", True)

showSIR = st.sidebar.checkbox("Show classical SIR-model based on 100% second variant",True)
#showSIR = False

if showcummulative or showSIR:
    numberofcasesdayz = (st.sidebar.text_input('Number infected persons on day zero', 130000))

    try:
        numberofcasesdayzero = int(numberofcasesdayz)
    except:
        st.error("Please enter a number for the number of active cases on day zero")
        st.stop()


if showcummulative or showSIR or showimmunization:
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
        st.error("Please enter a number for the number of population")
        st.stop()

turning = st.sidebar.checkbox("Turning point")

if turning:
    #Rnew3 = st.sidebar.slider('R-number target British variant', 0.1, 2.0, 0.8)
    changefactor = st.sidebar.slider('Change factor (1.0 = no change)', 0.0, 3.0, 0.9)
    #turningpoint = st.sidebar.slider('Startday turning', 1, 365, 30)
    turningpointdate = st.sidebar.text_input('Turning point date (mm/dd/yyyy)', b)
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

# I wanted to link the classical SIR model of Kermack & McKendrik but the R_0 in that 
# model isnt the same as the R_0 = beta / gamma from that model.
# See https://www.reddit.com/r/epidemiology/comments/lfk83s/real_r0_at_the_start_not_the_same_as_given_r0/


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

a_ = dt.datetime.strptime(a,'%m/%d/%Y').date()
b_ = dt.datetime.strptime(b,'%m/%d/%Y').date()
datediff = ( abs((a_ - b_).days))

# TODO:  Transform this in a multi dimensional list

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
hospital = []
infected = []
ic  = []
#if vaccination:
ry1x = []
ry2x = []

suspectible =[]
recovered = []
if showcummulative or showSIR or showimmunization:
    suspectible.append(totalpopulation -totalimmunedayzero)
    recovered.append(totalimmunedayzero )
if turning == False:
    label1= 'First variant (R='+ str(Rnew1_) + ')'
    label2= 'Second variant (R='+ str(Rnew2_) + ')'
else:
    label1= 'First variant'
    label2= 'Second variant'

# START CALCULATING --------------------------------------------------------------------
positivetests1.append (numberofpositivetests1)
positivetests2.append (numberofpositivetests2)
positivetests12.append (numberofpositivetests12)
positivetestsper100k.append ((numberofpositivetests12/25))

if showcummulative:
    cummulative1.append(numberofcasesdayzero*(1-percentagenewversion))
    cummulative2.append(numberofcasesdayzero*(percentagenewversion))
    cummulative12.append(numberofcasesdayzero)
    infected.append(numberofcasesdayzero)
ratio.append(percentagenewversion*100 )
#walkingcummulative.append(1)
#if vaccination:
ry1x.append(Rnew1_)
ry2x.append(Rnew2_)
immeratio_=[]
immeratio_.append(1)

hospital.append(None)
ic.append(None)
walkingR.append((Rnew1_**(1-percentagenewversion))*(Rnew2_**(percentagenewversion)))
if showimmunization:
    totalimmune.append(totalimmunedayzero)

for t in range(1, NUMBEROFDAYS):
    if showimmunization:
        immeratio = (1-( (totalimmune[t-1]-totalimmune[0])/(totalpopulation-totalimmune[0])))
        ry1_ = ry1x[0]*(immeratio**lambdaa)
        ry2_ = ry2x[0]*(immeratio**lambdaa)
        immeratio_.append(immeratio)
    else:
        ry1_ = ry1x[0]
        ry2_ = ry2x[0]

    if turning:
        if (t>=(turningpoint) and t<(turningpoint+turningdays)):
            if turningdays==0: 
                ry1__ = ry1_  * changefactor
                ry2__ = ry2_  * changefactor
            else:
                fraction =  (1-((t-(turningpoint))/turningdays))                
                ry1__ =(ry1_  * changefactor) + ((ry1_ -(ry1_  * changefactor))*fraction)
                ry2__ =(ry2_  * changefactor) + ((ry2_ -(ry2_  * changefactor))*fraction)
        elif t>=(turningpoint+turningdays):
            ry1__ = ry1_  * changefactor
            ry2__ = ry2_  * changefactor
        else:
            ry1__ = ry1_
            ry2__ = ry2_
    else:
        ry1__ = ry1_
        ry2__ = ry2_

    if vaccination:
        if t>7:
            if t<(VACTIME+7) :
                ry1 = ry1__ * ((1-((t-7)/(VACTIME))))
                ry2 = ry2__ * ((1-((t-7)/(VACTIME))))
            else:
                # vaccination is done, everybody is immune
                ry1 = ry1__ * 0.0000001
                ry2 = ry2__ * 0.0000001
        else:
            # it takes 7 days before vaccination works
            ry1 = ry1__
            ry2 = ry2__
    else:
        ry1 = ry1__
        ry2 = ry2__

    # prevent an [divide by zero]-error
    if ry1 == 1:    
        ry1 = 1.000001
    if ry2 == 1:
        ry2 = 1.000001
    if ry1 <= 0:
        ry1 = 0.000001
    if ry2 <= 0:
        ry2 = 0.000001

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
    # positivetests12a.append(positivetests2a[t-1] * (ry2**(1/Tg))+ positivetests1[t-1] 
    #                                              * (ry1**(1/Tg)))

    positivetests12.append(pt1+pt2)

    if showcummulative:
        cpt1 = (cummulative1[t-1]+  pt1)
        cpt2 = (cummulative2[t-1]+  pt2 )
        cpt12 =  (cummulative12[t-1]+ pt1 + pt2)
        
        if cpt1>=totalpopulation:
            cpt1 = totalpopulation
        if cpt2>=totalpopulation:
            cpt2 = totalpopulation
        if cpt12>=totalpopulation:
            cpt12 = totalpopulation

        cummulative1.append   (cpt1)
        cummulative2.append   (cpt2 )
        cummulative12.append   (cpt12)

    if (pt1+pt2)>0:
        ratio_ =  ((pt2/(pt1+pt2)))
    else:
        ratio_ = 1

    ratio.append   (100*ratio_)
    positivetestsper100k.append((pt1+pt2)/25)

    if showimmunization:
        totalimmune_ = totalimmune[t-1]+((pt1+pt2)*testimmunefactor)
        if totalimmune_>=totalpopulation:
            totalimmune_ = totalpopulation
        totalimmune.append(totalimmune_)   

    if showcummulative:
        if t>averagedayssick:
            infected.append (infected[t-1]+(((pt1+pt2))*testimmunefactor) -
                               (( positivetests1[t-averagedayssick]+ positivetests2[t-averagedayssick])*testimmunefactor )
                             ) 
            suspectible.append(suspectible[t-1]-(((pt1+pt2))*testimmunefactor) )
            recovered.append(recovered[t-1]+ 
                          (( positivetests1[t-averagedayssick]+positivetests2[t-averagedayssick])
                               * testimmunefactor ) ) 
        else:
            infected.append ( infected[t-1]+((pt1+pt2)*testimmunefactor) - 
                              (infected[0]/averagedayssick))  
            suspectible.append(suspectible[t-1]-(((pt1+pt2))*testimmunefactor) )
            recovered.append(recovered[t-1]+  (infected[0]/averagedayssick))
            
    ry1x.append(ry1)
    ry2x.append(ry2)
    walkingR.append((ry1**(1-ratio_))*(ry2**(ratio_)))
    if t>=7:
        hospital.append(positivetests12[t-7]*0.04)
    else:
        hospital.append(None)
    if t>=7:
        ic.append(positivetests12[t-7]*0.008)
    else:
        ic.append(None)

st.title('Positive COVID-tests in NL')

disclaimernew=('<style> .infobox {  background-color: lightyellow; padding: 10px;margin: 20-px}</style>'
               '<div class=\"infobox\"><h3>Disclaimer</h3><p>For illustration purpose only.</p>'
               '<p>Attention: these results are different from the official models'
               ' probably due to simplifications and different (secret) parameters.'
               '(<a href=\"https://archive.is/dqOjs\" target=\"_blank\">*</a>) '
                'The default parameters on this site are the latest known parameters of the RIVM'
                '</p><p>Forward-looking projections are estimates of what <em>might</em> occur. '
                'They are not predictions of what <em>will</em> occur. Actual results may vary substantially. </p>'
                 '<p>The goal was/is to show the (big) influence of (small) changes in the R-number. '
              'At the bottom of the page are some links to more advanced (SEIR) models.</p></div>')

st.markdown(disclaimernew,  unsafe_allow_html=True)
if showimmunization:
    disclaimerimm = ('<div class=\"infobox\"><p>The flattening  is very indicational. It is based on the principe R<sub>t</sub> = R<sub>start</sub> x (Suspectible / Population)<sup>λ</sup>. '
            'A lot of factors are not taken into account.'
        'The number of test is multiplied by ' +str(testimmunefactor)+ ' to get an estimation of the number of immune persons</div>'
        )

    st.markdown(disclaimerimm, unsafe_allow_html=True)
#        'Inspired by <a href=\'https://twitter.com/RichardBurghout/status/1357044694149128200\' target=\'_blank\'>this tweet</a>.<br> '

def th2r(rz):
    th = int( Tg_ * math.log(0.5) / math.log(rz))
    return th

def r2th(th):
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
    interval_ = int(numberofdays_ / 20)
    plt.xlabel('date')
    plt.xlim(x[0], x[-1])
    todaylabel = "Today ("+ b + ")"
    plt.axvline(x=x[0]+datediff, color='yellow', alpha=.6,linestyle='--',label = todaylabel)
    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='best', prop=fontP)
    plt.title(titlex , fontsize=10)

    # lay-out of the x axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval_))
    plt.gcf().autofmt_xdate()
    plt.gca().set_title(titlex , fontsize=10)





# POS TESTS /day ################################
with _lock:
    fig1, ax = plt.subplots()
    plt.plot(x, positivetests1, label=label1,  linestyle='--')
    plt.plot(x, positivetests2, label=label2,  linestyle='--')
    plt.plot(x, positivetests12, label='Total')
    
    # Add X and y Label and limits
    plt.ylabel('positive tests per day')
    plt.ylim(bottom = 0)

    plt.fill_between(x, 0, 875, color='#f392bd',  label='waakzaam')
    plt.fill_between(x, 876, 2500, color='#db5b94',  label='zorgelijk')
    plt.fill_between(x, 2501, 6250, color='#bc2165',  label='ernstig')
    plt.fill_between(x, 6251, 10000, color='#68032f', label='zeer ernstig')
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
    plt.fill_between(x, 501, 1000, color='grey', alpha=0.3, label='zeer zeer ernstig')

    plt.axhline(y=0, color='green', alpha=.6,linestyle='--' )
    plt.axhline(y=35, color='yellow', alpha=.6,linestyle='--')
    plt.axhline(y=100, color='orange', alpha=.6,linestyle='--')
    plt.axhline(y=250, color='red', alpha=.6,linestyle='--')
    plt.axhline(y=500, color='purple', alpha=.6,linestyle='--')

    # Add a title
    titlex = ( 'New pos. tests per 100k per week.\n'  )
    configgraph(titlex)

    st.pyplot(fig1d)


####### SHOW TOTAL  CASES PER WEEK
# inspired by https://twitter.com/steeph/status/1363865443467952128
output = pd.DataFrame(
    {'date': x,
     'First_variant': positivetests1,
     'Second_variant': positivetests2
    })
#st.write(output)


output['date'] =  pd.to_datetime(output['date'],  unit='D',
               origin=pd.Timestamp('1970-01-01'))
# output['week_number_of_year'] = output['date'].dt.week
# output['year'] = output['date'].dt.year
# output["weeknr"] = output["year"].astype(str) + ' ' + output["week_number_of_year"].astype(str)
# output['weekday'] = output['date'].dt.strftime('%U')
#output["weeknr"] = output['date'].dt.year.astype(str) + ' - ' +  output['date'].dt.isocalendar().week.astype(str)
output["weeknr"] =  output['date'].dt.isocalendar().week
     

output = output.groupby("weeknr").sum()


with st.beta_expander("Show bargraph per week - Attention - doesn't display well when there are two years involved and/or the weeks aren't complete. Weeks are Monday until Sunday"):
    fig1x = plt.figure()
    output.plot()
    plt.legend(loc='best')
    #st.pyplot(fig1x)
    #st.write(fig1x)
    titlex="Number of casees per week"
    configgraph(titlex)

    st.bar_chart(output)
    

positivetests1 = []
positivetests2 = []
positivetest12 = []
   

# Show cummulative cases
if showcummulative:
    with _lock:
        fig1e, ax = plt.subplots()
        plt.plot(x,cummulative1, label='Cummulative pos. test first variant',  linestyle='--')
        plt.plot(x,cummulative2, label='Cummulative pos. test second variant',  linestyle='--')
        plt.plot(x,cummulative12, label='Cummulative pos. test TOTAL',)

        cummulative1 = []
        cummulative2 = []
        cummulative12 = []
        # Add X and y Label and limits

        plt.ylabel('Total positive tests')
        plt.ylim(bottom = 0)

        # Add a title
        titlex = ('Cummulative positive tests\nNo recovery/death in this graph')

        configgraph(titlex)

        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        st.pyplot(fig1e)

    # Infected
    with _lock:
        fig1i, ax = plt.subplots()
        plt.plot(x, suspectible, label='Suspectible',  linestyle='--')
        plt.plot(x, infected, label='Infected',  linestyle='--')
        plt.plot(x, recovered, label='Recovered',  linestyle='--')
        infected = []

        # Add  y Label and limits
        plt.ylabel('No of cases')
        plt.ylim(bottom = 0)

        # Add a title
        titlex = ('Suspectible - Infected - Recovered.\nBased on positive tests.\n'
                  '(test/immunityfactor is taken in account)')
        configgraph(titlex)
        
        st.pyplot(fig1i)
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

# Ziekenhuis opnames
with _lock:
    fig1g, ax = plt.subplots()
    plt.plot(x, hospital, label='Ziekenhuis per dag')
    plt.plot(x, ic, label='IC per dag')
    # Add X and y Label and limits

    plt.ylabel('Ziekenhuis- en IC-opnames per day')
    plt.ylim(bottom = 0)
    plt.axhline(y=0, color='green', alpha=.6,linestyle='--' )
    plt.axhline(y=12, color='yellow', alpha=.6,linestyle='--')
    plt.axhline(y=40, color='orange', alpha=.6,linestyle='--')
    plt.axhline(y=80, color='red', alpha=.6,linestyle='--')
   
    # Add a title
    titlex = ('Ziekenhuis (4%) en IC (0.8%) opnames per day,\n7 dgn vertraging')
    configgraph(titlex)
    st.pyplot(fig1g)

ry1x = []
ry2x = []
hospital = []
ic = []
################################################
if showSIR:

    with st.beta_expander("Show classical SIR-graphs"):
   


        # https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

        # Total population, N.
        #N = int(input("Total population, N "))
        #if N == 0 :
        N = int(totalpopulation)
        
        # Initial number of infected and recovered individuals, I0 and R0.
        I0, R0 = int(numberofcasesdayz), totalimmunedayzero
        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 -  R0
        C0 = I0
        days = NUMBEROFDAYS

        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        #beta, gamma = 0.2, 1./10
        ##beta = float(input("Contact rate - beta [0-1] "))
        #gamma = 1/int(input("Mean recovery rate in days - 1/gamma  "))

        # Gamma is 1/serial interval
        # https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article

        gamma = 1./Tg_
        Rstart = Rnew2_
        #beta = Rstart*gamma
        beta = Rstart*gamma/(S0/N)
        #beta, gamma = 0.2, 1./20

        # reproductionrate = beta / gamma

        # β describes the effective contact rate of the disease: 
        # an infected individual comes into contact with βN other 
        # individuals per unit time (of which the fraction that are
        # susceptible to contracting the disease is S/N).
        # 1/gamma is recovery rate in days 

        # A grid of time points (in days)
        t = np.linspace(0, days, days)

        # The SIR model differential equations.
        # https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
        def deriv(y, t, N, beta, gamma):
            S, I, C, R = y
            dSdt = -beta * S * I / N   # aantal zieke mensen x aantal gezonde mensen x beta
            dIdt = beta * S * I / N - gamma * I
            dCdt = beta * S * I / N 
            dRdt = gamma * I
            
            #print (dIdt)

            # (n2/n1)^(Tg/t) 
            return dSdt, dIdt, dRdt, dCdt

        # Initial conditions vector
        y0 = S0, I0, C0, R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, C, R  = ret.T


        Tg = Tg_
        d = 1
        repr=[]
        repr.append(Rstart)
        repr_c=[]
        repr_i=[]
        repr_c.append(None)
        repr_i.append(None)
        t = np.linspace(0, days, days)
        #print (I)
        Cnew=[]
        #Cnew.append(int(numberofcasesdayz))
        Cnew.append(None)
        for time in range(1,days):
            Cnew.append(C[time]-C[time-1])
        
            #if Cnew[time-1] == None:
            if time == 1: 
                repr_ = None
                repr_c_ = None
                repr_i_ = None
            else:
                #repr_= (C[time]/C[time-1])**(Tg/d) 
                repr_= (Cnew[time]/Cnew[time-1])**(Tg/d) 
                repr_c_= (C[time]/C[time-1])**(Tg/d) 
                repr_i_= (I[time]/I[time-1])**(Tg/d) 
            #repr_= (I[time]/I[time-1])    
            repr.append(repr_)
            repr_c.append(repr_c_)
            repr_i.append(repr_i_)

        disclaimerSIR= ('<div class=\"infobox\"><h1>Classical SIR-graphs</h1>'
                        '<p>These graphs are based on classical SIR models.'
                        ' See <a href=\"https://web.stanford.edu/~jhj1/teachingdocs/Jones-on-R0.pdf\"'
                        ' target=\"_blank\">'
                        'here</a> for an explanation. '
                        'It is based on the number of immune peope at the start   ' +str(totalimmunedayzero)+ ' and '
                        'the population size</p>'
                        '<p>Beta : ' + str(round(beta,4)) + ' / Gamma : ' + str(gamma) + ' / R<sub>0</sub> : '+ str(Rstart) + '</p>'
                        '</div>'
            )

        st.markdown(disclaimerSIR, unsafe_allow_html=True)
    
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig2a = plt.figure(facecolor='w')
        #ax = fig2a.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax = fig2a.add_subplot(111, axisbelow=True)
        ax.plot(x, S, 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(x, I, 'r', alpha=0.5, lw=2, label='Infected')
        ax.plot(x, Cnew, 'yellow', alpha=0.5, lw=2, label='New Cases')
        ax.plot(x, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
        #ax.plot(t, repr, 'yellow', alpha=0.5, lw=2, label='R getal')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Number')
        #ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        #legend = ax.legend()
        #legend.get_frame().set_alpha(0.5)
        #for spine in ('top', 'right', 'bottom', 'left'):
        #    ax.spines[spine].set_visible(False)
        titlex = 'SIR based on cases first day'
        configgraph(titlex)
        plt.show()
        st.pyplot(fig2a)


        # New cases
        fig2c = plt.figure(facecolor='w')
        ax = fig2c.add_subplot(111,  axisbelow=True)
        #ax.plot(t, C, 'green', alpha=0.5, lw=2, label='Cases')
        ax.plot(x, Cnew, 'y', alpha=0.5, lw=2, label='New Cases')
        #ax.plot(t, repr, 'yellow', alpha=0.5, lw=2, label='R getal')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Number')
        #ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        # legend = ax.legend()
        # legend.get_frame().set_alpha(0.5)
        # for spine in ('top', 'right', 'bottom', 'left'):
        #     ax.spines[spine].set_visible(False)
        titlex = 'New cases'
        configgraph(titlex)
        plt.show()
        st.pyplot(fig2c)

        # Gliding R number
        fig2b = plt.figure(facecolor='w')
        ax = fig2b.add_subplot(111,  axisbelow=True)
        ax.plot(x, repr, 'b', alpha=0.5, lw=2, label='R getal_ based on Cnew')
        #ax.plot(t, repr_i, 'b', alpha=0.5, lw=2, label='R getal based on I')
        #ax.plot(t, repr_c, 'g', alpha=0.5, lw=2, label='R getal based on C')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('R getal')
        #ax.set_ylim(0,2)
        ax.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        # ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        # legend = ax.legend()
        titlex = "Gliding R-number"
        configgraph(titlex)
        plt.show()
        st.pyplot(fig2b)
        
        st.write  ("attack rate growth model : " +        str(round(100*((recovered[days-1])/N),2))+ " %")
        st.write  ("attack rate classical SIR model : " + str(round(100*((C[days-1])        /N),2))+ " %")
        st.markdown ("Theoretical herd immunity treshhold (HIT) (1 - [1/"+str(Rstart)+"]<sup>1/"+ str(lambdaa)+ "</sup>) : " + str(round(100*(1-((1/Rstart)**(1/lambdaa))),2))+ " % = " + str(round(N*(1-((1/Rstart)**(1/lambdaa))),0))+ " persons", unsafe_allow_html=True)
        st.write ("Attack rate = final size of the epidemic (FSE) ")
        repr=[]

#####################################################

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
'<li><a href=\"https://www.covidsim.org" target=\"_blank\">COVID-19 Scenario Analysis Tool (Imperial College London)</a></li>'
'<li><a href=\"http://www.covidsim.eu/" target=\"_blank\">CovidSIM/a></li>'
'<li><a href=\"https://covid19-scenarios.org/\" target=\"_blank\">Covid scenarios</a></li>'
'<li><a href=\"https://share.streamlit.io/lcalmbach/pandemic-simulator/main/app.py\" target=\"_blank\">Pandemic simulator</a></li>'
'<li><a href=\"https://penn-chime.phl.io/\" target=\"_blank\">Hospital impact model</a></li>'
'<li><a href=\"http://www.modelinginfectiousdiseases.org/\" target=\"_blank\">Code from the book Modeling Infectious Diseases in Humans and Animals '
'(Matt J. Keeling & Pejman Rohani)</a></li></ul>'
'<h3>Other sources/info</h3>'
'<ul><li><a href=\"https://archive.is/dqOjs\" target=\"_blank\">Waarom bierviltjesberekeningen over het virus niet werken</a></li>'
'<li><a href=\"https://www.scienceguide.nl/2020/03/modellen-geven-geen-absolute-zekerheid-maar-ze-zijn-ontontbeerlijk/\" target=\"_blank\">Modellen geven geen absolute zekerheid, maar ze zijn onontbeerlijk</a></li>'
'<li><a href=\"https://www.nature.com/articles/d41586-020-02009-ws\" target=\"_blank\">A guide to R — the pandemic’s misunderstood metric</a></li></ul>')

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
