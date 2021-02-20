# Make an interactive version of the SEIR model, inspired by Hobbeland
# https://twitter.com/MinaCoen/status/1362910764739231745



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

numberofcasesdayz = (st.sidebar.text_input('Number infected persons on day zero (I0)', 100))

try:
    numberofcasesdayzero = int(numberofcasesdayz)
except:
    st.error("Please enter a number for the number of active cases on day zero")
    st.stop()
    
totalimmunedayzero = 0 

st.markdown("<hr>", unsafe_allow_html=True)
a = st.sidebar.text_input('startdate (mm/dd/yyyy)',"03/01/2021")

try:
    startx = dt.datetime.strptime(a,'%m/%d/%Y').date()
except:
    st.error("Please make sure that the date is in format mm/dd/yyyy")
    st.stop()

NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 15, 720, 100)
global numberofdays_
numberofdays_ = NUMBEROFDAYS

Rstart = st.sidebar.slider('R-number first variant', 0.1, 10.0, 2.5)


incubationtime = (st.sidebar.slider('Incubatietijd (1/alfa)', 1, 30, 3))

infectioustime = (st.sidebar.slider('Average days infectious (1/gamma)', 1, 30, 2))




totalpopulation_ = (st.sidebar.text_input('Total population', 10_000_000))

try:
    totalpopulation = int(totalpopulation_)
except:
    st.error("Please enter a number for the number of population")
    st.stop()

Tg = st.sidebar.slider('Generation time (to calculate gliding Rt)', 2.0, 11.0, 4.0)
global Tg_
Tg_=Tg



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


walkingR=[]
actualR=[]
totalimmune=[]
hospital = []
infected = []
ic  = []
suspectible =[]
recovered = []

suspectible.append(totalpopulation -totalimmunedayzero)
#recovered.append(totalimmunedayzero )


# START CALCULATING --------------------------------------------------------------------


hospital.append(None)
ic.append(None)
 
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

################################################

   
# https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

# Total population, N.
#N = int(input("Total population, N "))
#if N == 0 :
N = int(totalpopulation)

# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = int(numberofcasesdayz), totalimmunedayzero
E0 = 0 


# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 -  R0 - E0

C0 = I0
days = NUMBEROFDAYS

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#beta, gamma = 0.2, 1./10
##beta = float(input("Contact rate - beta [0-1] "))
#gamma = 1/int(input("Mean recovery rate in days - 1/gamma  "))

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
alfa = 1./incubationtime
gamma = 1./infectioustime

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
    S, E, I, C, R = y
    dSdt = -beta * S * I / N   # aantal zieke mensen x aantal gezonde mensen x beta
    dEdt = beta * S * I / N  - alfa * E
    dIdt = alfa * E - gamma * I
    dCdt = alfa * E
    dRdt = gamma * I
    

    #print (dIdt)

    # (n2/n1)^(Tg/t) 
    return dSdt, dEdt, dIdt, dRdt, dCdt

# Initial conditions vector
y0 = S0, E0, I0, C0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, E, I, C, R  = ret.T


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

disclaimerSIR= ('<div class=\"infobox\"><h1>Classical SEIR-graphs</h1>'
                    '<p>These graphs are based on classical SEIR models.</p>'
                    '<p>The parameters are taken from'
                    '<a href=\"https://twitter.com/MinaCoen/status/1362910764739231745\" target=\"_blank\">'
                    ' the story from Willemijn Coene about Hobbeland</a></p>'

                    
                     '<p> See also <a href=\"https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py\"'
                    ' target=\"_blank\">'
                    'my illustrative and simple model about the Netherlands</a>. </p>'


                    
                    '<p>Alfa : ' + str(round(alfa,4)) + ' / Beta : ' + str(round(beta,4)) + ' / Gamma : ' + str(gamma) + ' / R<sub>0</sub> : '+ str(Rstart) + '</p>'
                    '</div>'
    )

st.markdown(disclaimerSIR, unsafe_allow_html=True)

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig2a = plt.figure(facecolor='w')
#ax = fig2a.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax = fig2a.add_subplot(111, axisbelow=True)
ax.plot(x, S, 'yellow', alpha=0.5, lw=2, label='Susceptible')
ax.plot(x, E, 'green', alpha=0.5, lw=2, label='Exposed')

ax.plot(x, I, 'r', alpha=0.5, lw=2, label='Infected / Ziek')
#ax.plot(x, Cnew, 'blue', alpha=0.5, lw=2, label='New Cases')
ax.plot(x, R, 'purple', alpha=0.5, lw=2, label='Recovered with immunity')
#ax.plot(t, repr, 'yellow', alpha=0.5, lw=2, label='R getal')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)

titlex = 'SIR based on cases first day'
configgraph(titlex)
plt.show()
st.pyplot(fig2a)


# New cases
fig2c = plt.figure(facecolor='w')
ax = fig2c.add_subplot(111,  axisbelow=True)
#ax.plot(t, C, 'green', alpha=0.5, lw=2, label='Cases')
ax.plot(x, Cnew, 'blue',  label='New Cases')
ax.plot(x, I, 'red', alpha=0.5,  label='Infected')

#ax.plot(t, repr, 'yellow', alpha=0.5, lw=2, label='R getal')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
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
lambdaa = 1
st.write  ("attack rate classical SIR model : " + str(round(100*((C[days-1])        /N),2))+ " %")
st.markdown ("Theoretical herd immunity treshhold (HIT) (1 - [1/"+str(Rstart)+"]<sup>1/"+ str(lambdaa)+ "</sup>) : " + str(round(100*(1-((1/Rstart)**(1/lambdaa))),2))+ " % = " + str(round(N*(1-((1/Rstart)**(1/lambdaa))),0))+ " persons", unsafe_allow_html=True)
st.write ("Attack rate = final size of the epidemic (FSE) ")
repr=[]

#####################################################

tekst = (
    '<style> .infobox {  background-color: lightblue; padding: 5px;}</style>'
    '<hr><div class=\'infobox\'>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\' target=\"_blank\">@rcsmit</a>) <br>'
    'Overdrachtstijd is 4 dagen. Disclaimer is following. Provided As-is etc.<br>'
    'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/blob/main/SEIR_hobbeland.py\" target=\"_blank\">github.com/rcsmit</a><br>'
    'How-to tutorial : <a href=\"https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d\" target=\"_blank\">rcsmit.medium.com</a><br>')

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
'<li><a href=\"https://covid19-scenarios.org/\" target=\"_blank\">Covid scenarios</a></li>'
'<li><a href=\"https://share.streamlit.io/lcalmbach/pandemic-simulator/main/app.py\" target=\"_blank\">Pandemic simulator</a></li>'
'<li><a href=\"https://penn-chime.phl.io/\" target=\"_blank\">Hospital impact model</a></li>'
'<li><a href=\"http://www.modelinginfectiousdiseases.org/\" target=\"_blank\">Code from the book Modeling Infectious Diseases in Humans and Animals '
'(Matt J. Keeling & Pejman Rohani)</a></li></ul>'
'<h3>Other sources/info</h3>'
'<ul><li><a href=\"https://archive.is/dqOjs\" target=\"_blank\">Waarom bierviltjesberekeningen over het virus niet werken</a></li>'
'<li><a href=\"https://www.scienceguide.nl/2020/03/modellen-geven-geen-absolute-zekerheid-maar-ze-zijn-ontontbeerlijk/\" target=\"_blank\">Modellen geven geen absolute zekerheid, maar ze zijn onontbeerlijk</a></li>'
'<li><a href=\"https://www.nature.com/articles/d41586-020-02009-ws\" target=\"_blank\">A guide to R — the pandemic’s misunderstood metric</a></li></ul>')


st.sidebar.markdown(tekst, unsafe_allow_html=True)
#st.sidebar.info(tekst)
st.markdown(links, unsafe_allow_html=True)
