# Playing with the parameters of a classical SEIR models
# René Smit, 20 February 2021, MIT-LICENSE
# The parameters are taken from the story from Willemijn Coene about Hobbeland
# https://twitter.com/MinaCoen/status/1362910764739231745
# Alfa : 0.3333 / Beta : 1.25 / Gamma : 0.5 / R0 : 2.5
# If there are strange results, just change the number of days a little bit. This is due to strange behavior of scipy's ODEINT. solve_ivp seems to be better

# replicating SIR model van www.beterdanhugo.nl

import math
from datetime import datetime
import copy
import streamlit as st
import numpy as np
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
_lock = RendererAgg.lock
from scipy.integrate import odeint



def th2r(rz):
    return int( Tg_ * math.log(0.5) / math.log(rz))

def r2th(th):
    # HK is using  r = 2**(Tg_/th)
    return int(10**((Tg_*mat.log(2))/th))

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

def configgraph(titlex,x,b,datediff):
    interval_ = int(numberofdays_ / 20)
    plt.xlabel('date')
    plt.xlim(x[0], x[-1])
    # todaylabel = "Today ("+ b + ")"
    # plt.axvline(x=x[0]+datediff, color='yellow', alpha=.6,linestyle='--',label = todaylabel)
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

def main():


    DATE_FORMAT = "%m/%d/%Y"
    today = datetime.today().strftime('%m/%d/%Y')

    st.sidebar.title('Parameters')

    numberofcasesdayz = (st.sidebar.text_input('Number infected persons on day zero (I0)', 1000))

    try:
        numberofcasesdayzero = int(numberofcasesdayz)
    except:
        st.error("Please enter a number for the number of active cases on day zero")
        st.stop()

    totalimmunedayzero = 0

    st.markdown("<hr>", unsafe_allow_html=True)
    startdate = st.sidebar.text_input('startdate (mm/dd/yyyy)',"03/01/2021")
numb
    try:
        startx = dt.datetime.strptime(startdate,'%m/%d/%Y').date()
    except:
        st.error("Please make sure that the date is in format mm/dd/yyyy")
        st.stop()

    NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 15, 720, 100)
    global numberofdays_
    numberofdays_ = NUMBEROFDAYS

    scenarioname = (st.sidebar.text_input('Scenarioname'))
    #Rstart = st.sidebar.slider('R-number variant', 0.1, 10.0, 2.5)
    Rstart =  st.sidebar.number_input('R number', 0.00, 10.00, 2.50)
    ifr = (st.sidebar.number_input('ifr in %', 0.0, 100.0, 0.9))/100

    b = (st.sidebar.slider('aantal dagen infectious', 1, 30, 5))

    c = (st.sidebar.slider('1/c', 1,1000, 525))



    totalpopulation_ = (st.sidebar.number_input('Total population', 0,100_000_000, 17_500_000))

    # Some manipulation of the x-values (the dates)
    then = startx + dt.timedelta(days=NUMBEROFDAYS)
    x = mdates.drange(startx,then,dt.timedelta(days=1))
    z  = np.array(range(NUMBEROFDAYS))

    a_ = dt.datetime.strptime(startdate,'%m/%d/%Y').date()
    b_ = dt.datetime.strptime(today,'%m/%d/%Y').date()
    datediff = ( abs((a_ - b_).days))

   
    # walkingR=[]
    # actualR=[]
    # totalimmune=[]
    # hospital = []
    # infected = []
    # ic  = []
    # suspectible =[]
    # recovered = []
    # vaccinated = []
    # suspectible.append(totalpopulation -totalimmunedayzero)
    #recovered.append(totalimmunedayzero )


    # START CALCULATING --------------------------------------------------------------------

  
    # https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

    # Total population, N.
    #N = int(input("Total population, N "))
    #if N == 0 :
    N = int(totalpopulation)

    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = int(numberofcasesdayz), totalimmunedayzero
   
    D0 = 0

    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 -  R0 - D0

    days = NUMBEROFDAYS

    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    #beta, gamma = 0.2, 1./10
    ##beta = float(input("Contact rate - beta [0-1] "))
    #gamma = 1/int(input("Mean recovery rate in days - 1/gamma  "))

    # Gamma is 1/serial interval
    # https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
    # alfa = 1./incubationtime
    # gamma = 1./infectioustime

    #beta = Rstart*gamma
    # beta = Rstart*gamma/(S0/N)
    #beta, gamma = 0.2, 1./20
    # zulu = 1/ days_needed_for_vaccination

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
    def deriv(y, t, N, b,c,  ifr):
        S,  I, R, D = y


        # dSdt = 0 if S<=0 else (-beta * S * I / N)  - (dVdt)

        # dIdt = alfa * E - gamma * I
        # dCdt = alfa * E
        # dDdt = (ifr*gamma) * I
        # dRdt = (gamma * I) -  (ifr*gamma) * I

        a = S/N  * b

        dSdt = R * c
        dIdt = a * S * I / N
        dRdt = b * I * (1 -ifr)
        dDdt = b * I * ifr

        return dSdt, dIdt, dRdt, dDdt

    # Initial conditions vector
    y0 = S0, I0, R0, D0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N,b,c, ifr))
    S, I, R, D  = ret.T

    Tg = Tg_
    d = 1
    lambdaa = 1
    t = np.linspace(0, days, days)


    disclaimerSIR= ('<div class=\"infobox\"><h1>Classical SEIR-graphs</h1>'
                        '<p>These graphs are based on classical SEIR models.</p>'




                        '<p>If there are strange results, just change the number of days/parameters a little bit. This is due to strange behavior of scipy\'s ODEINT. solve_ivp seems to be better</p>'
                        '</div>'
        )

    st.markdown(disclaimerSIR, unsafe_allow_html=True)

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig2a = plt.figure(facecolor='w')
    ax = fig2a.add_subplot(111, axisbelow=True)
    ax.plot(x, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    # ax.plot(x, E, 'purple', alpha=0.5, lw=2, label='Exposed')
    # ax.plot(x, V, 'yellow', alpha=0.5, lw=2, label='Vaccinated')

    ax.plot(x, I, 'r', alpha=0.5, lw=2, label='Infected')
    # ax.plot(x, Cnew, 'orange', alpha=0.5, lw=2, label='New Cases')
    ax.plot(x, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(x, D, 'black', alpha=0.5, lw=2, label='Death')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    titlex = (f'SEIR - {scenarioname}')
    configgraph(titlex,x,b,datediff)
    plt.show()
    st.pyplot(fig2a)

    # st.write  ("attack rate classical SIR model : " + str(int(C[days-1])) + " mensen / "+ str(round(100*((C[days-1])        /N),2))+ " %")
    # st.write (f"Number of deaths: {round(C[days-1])} * {ifr} = {round(C[days-1]*ifr)}")

    # st.markdown ("Theoretical herd immunity treshhold (HIT) (1 - [1/"+str(Rstart)+"]<sup>1/"+ str(lambdaa)+ "</sup>) : " + str(round(100*(1-((1/Rstart)**(1/lambdaa))),2))+ " % = " + str(round(N*(1-((1/Rstart)**(1/lambdaa))),0))+ " persons", unsafe_allow_html=True)
    # st.write ("Attack rate = final size of the epidemic (FSE) ")


    # # New cases
    # fig2c = plt.figure(facecolor='w')
    # ax = fig2c.add_subplot(111,  axisbelow=True)
    # ax.plot(x, Cnew, 'blue',  label='New Cases')
    # ax.plot(x, I, 'red', alpha=0.5,  label='Infected')
    # ax.set_xlabel('Time (days)')
    # ax.set_ylabel('Number')
    # ax.yaxis.set_tick_params(length=0)
    # ax.xaxis.set_tick_params(length=0)
    # titlex = 'New cases'
    # configgraph(titlex,x,b,datediff)
    # plt.show()
    # st.pyplot(fig2c)



    # st.write  ("attack rate classical SIR model : " + str(int(C[days-1])) + " mensen / "+ str(round(100*((C[days-1])        /N),2))+ " %")
    # st.markdown ("Theoretical herd immunity treshhold (HIT) (1 - [1/"+str(Rstart)+"]<sup>1/"+ str(lambdaa)+ "</sup>) : " + str(round(100*(1-((1/Rstart)**(1/lambdaa))),2))+ " % = " + str(round(N*(1-((1/Rstart)**(1/lambdaa))),0))+ " persons", unsafe_allow_html=True)
    # st.write ("Attack rate = final size of the epidemic (FSE) ")
    # repr=[]

    #####################################################

    tekst = (
        '<style> .infobox {  background-color: lightblue; padding: 5px;}</style>'
        '<hr><div class=\'infobox\'>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\' target=\"_blank\">@rcsmit</a>) <br>'
        'Overdrachtstijd is 4 dagen. Disclaimer is following. Provided As-is etc.<br>'
        'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/edit/main/number_of_cases_interactive.py\" target=\"_blank\">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href=\"https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d\" target=\"_blank\">rcsmit.medium.com</a><br>'
        'Inspired by <a href=\"https://twitter.com/mzelst/status/1350923275296251904\" target=\"_blank\">this tweet</a> of Marino van Zelst.<br>'
        'With help of <a href=\"https://twitter.com/hk_nien" target=\"_blank\">Han-Kwang Nienhuys</a>.</div>')



    st.sidebar.markdown(tekst, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
