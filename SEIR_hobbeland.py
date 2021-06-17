# Playing with the parameters of a classical SEIR models
# René Smit, 20 February 2021, MIT-LICENSE
# The parameters are taken from the story from Willemijn Coene about Hobbeland
# https://twitter.com/MinaCoen/status/1362910764739231745
# Alfa : 0.3333 / Beta : 1.25 / Gamma : 0.5 / R0 : 2.5
# If there are strange results, just change the number of days a little bit. This is due to strange behavior of scipy's ODEINT. solve_ivp seems to be better


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

def configgraph(titlex,x,b,datediff):
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

def main():


    DATE_FORMAT = "%m/%d/%Y"
    b = datetime.today().strftime('%m/%d/%Y')

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

    scenarioname = (st.sidebar.text_input('Scenarioname'))
    #Rstart = st.sidebar.slider('R-number variant', 0.1, 10.0, 2.5)
    Rstart =  st.sidebar.number_input('R number', 0.00, 10.00, 2.50)
    ifr = (st.sidebar.number_input('ifr in %', 0.0, 100.0, 0.60))/100

    incubationtime = (st.sidebar.slider('Incubatietijd (1/alfa)', 1, 30, 3))

    infectioustime = (st.sidebar.slider('Average days infectious (1/gamma)', 1, 30, 2))

    #start_day_vaccination = (st.sidebar.slider('Day on which the vaccination starts\n(set on max for no vaccination)', 1, NUMBEROFDAYS, int(NUMBEROFDAYS*0.2)))
    start_day_vaccination = (st.sidebar.slider('Day on which the vaccination starts\n(set on max for no vaccination)', 1, NUMBEROFDAYS,NUMBEROFDAYS ))

    days_needed_for_vaccination = (st.sidebar.slider('Days needed for vaccination', 1, 3650, 365))

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
    vaccinated = []
    suspectible.append(totalpopulation -totalimmunedayzero)
    #recovered.append(totalimmunedayzero )


    # START CALCULATING --------------------------------------------------------------------

    hospital.append(None)
    ic.append(None)
    # https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

    # Total population, N.
    #N = int(input("Total population, N "))
    #if N == 0 :
    N = int(totalpopulation)

    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = int(numberofcasesdayz), totalimmunedayzero
    E0 = 0
    V0 = 0
    D0 = 0

    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 -  R0 - E0 - D0

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
    zulu = 1/ days_needed_for_vaccination

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
    def deriv(y, t, N, beta, gamma, ifr):
        S, V, E, I, C, R, D = y
        if V >= N:
            dVdt = 0
        else:
            if t>start_day_vaccination:
                zulu_= copy.deepcopy(zulu)
                N_=copy.deepcopy(N)
                dVdt = zulu_ * N_
            else:
                dVdt = 0
        if S<=0:
            dSdt = 0
        else:
            dSdt = (-beta * S * I / N)  - (dVdt) # aantal zieke mensen x aantal gezonde mensen x beta

        dEdt = beta * S * I / N  - alfa * E
        dIdt = alfa * E - gamma * I
        dCdt = alfa * E
        dDdt = (ifr*gamma) * I
        dRdt = (gamma * I) -  (ifr*gamma) * I


        return dSdt, dVdt, dEdt, dIdt, dRdt, dDdt, dCdt

    # Initial conditions vector
    y0 = S0, V0, E0, I0, C0, R0, D0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, ifr))
    S, V, E, I, R, D,  C  = ret.T

    Tg = Tg_
    d = 1
    lambdaa = 1
    repr=[]
    repr.append(Rstart)
    repr_c=[]
    repr_i=[]
    repr_c.append(None)
    repr_i.append(None)
    t = np.linspace(0, days, days)
    Cnew=[]
    Cnew.append(None)
    for time in range(1,days):
        Cnew.append(C[time]-C[time-1])

        if time == 1:
            repr_ = None
            repr_c_ = None
            repr_i_ = None
        else:
            repr_= (Cnew[time]/Cnew[time-1])**(Tg/d)
            repr_c_= (C[time]/C[time-1])**(Tg/d)
            repr_i_= (I[time]/I[time-1])**(Tg/d)
        repr.append(repr_)
        repr_c.append(repr_c_)
        repr_i.append(repr_i_)

    disclaimerSIR= ('<div class=\"infobox\"><h1>Classical SEIR-graphs</h1>'
                        '<p>These graphs are based on classical SEIR models.</p>'
                        '<p>The parameters are taken from'
                        '<a href=\"https://twitter.com/MinaCoen/status/1362910764739231745\" target=\"_blank\">'
                        ' the story from Willemijn Coene about Hobbeland</a></p>'


                        '<p> See <a href=\"https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py\"'
                        ' target=\"_blank\">'
                        'for my illustrative and simple model about the Netherlands</a>. </p>'



                        '<p>Alfa : ' + str(round(alfa,4)) + ' / Beta : ' + str(round(beta,4)) + ' / Gamma : ' + str(gamma) + ' / R<sub>0</sub> : '+ str(Rstart) + '</p>'
                        '<p>If there are strange results, just change the number of days/parameters a little bit. This is due to strange behavior of scipy\'s ODEINT. solve_ivp seems to be better</p>'
                        '</div>'
        )

    st.markdown(disclaimerSIR, unsafe_allow_html=True)

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig2a = plt.figure(facecolor='w')
    ax = fig2a.add_subplot(111, axisbelow=True)
    ax.plot(x, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(x, E, 'purple', alpha=0.5, lw=2, label='Exposed')
    ax.plot(x, V, 'yellow', alpha=0.5, lw=2, label='Vaccinated')

    ax.plot(x, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(x, Cnew, 'orange', alpha=0.5, lw=2, label='New Cases')
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

    st.write  ("attack rate classical SIR model : " + str(int(C[days-1])) + " mensen / "+ str(round(100*((C[days-1])        /N),2))+ " %")
    st.write (f"Number of deaths: {round(C[days-1])} * {ifr} = {round(C[days-1]*ifr)}")

    st.markdown ("Theoretical herd immunity treshhold (HIT) (1 - [1/"+str(Rstart)+"]<sup>1/"+ str(lambdaa)+ "</sup>) : " + str(round(100*(1-((1/Rstart)**(1/lambdaa))),2))+ " % = " + str(round(N*(1-((1/Rstart)**(1/lambdaa))),0))+ " persons", unsafe_allow_html=True)
    st.write ("Attack rate = final size of the epidemic (FSE) ")


    # New cases
    fig2c = plt.figure(facecolor='w')
    ax = fig2c.add_subplot(111,  axisbelow=True)
    ax.plot(x, Cnew, 'blue',  label='New Cases')
    ax.plot(x, I, 'red', alpha=0.5,  label='Infected')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    titlex = 'New cases'
    configgraph(titlex,x,b,datediff)
    plt.show()
    st.pyplot(fig2c)

    # Gliding R number
    fig2b = plt.figure(facecolor='w')
    ax = fig2b.add_subplot(111,  axisbelow=True)
    ax.plot(x, repr, 'b', alpha=0.5, lw=2, label='R getal_ based on Cnew')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('R getal')
    ax.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
    titlex = "Gliding R-number"
    configgraph(titlex,x,b,datediff)
    plt.show()
    st.pyplot(fig2b)

    st.write  ("attack rate classical SIR model : " + str(int(C[days-1])) + " mensen / "+ str(round(100*((C[days-1])        /N),2))+ " %")
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



    st.sidebar.markdown(tekst, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
