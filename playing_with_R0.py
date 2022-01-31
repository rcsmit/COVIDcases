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

import copy
from scipy.integrate import odeint



def make_plot_math(x, title, datareeksen, showlogyaxis):
    def configgraph(titlex, showlogyaxis):
        interval_ = int(numberofdays_ / 20)
        plt.xlabel('date')
        plt.xlim(x[0], x[-1])
        # todaylabel = "Today ("+ b + ")"
        #plt.axvline(x=x[0]+datediff, color='yellow', alpha=.6,linestyle='--',label = todaylabel)
        # Add a grid
        plt.grid(alpha=.4,linestyle='--')

        #Add a Legend
        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(  loc='best', prop=fontP)
        plt.title(titlex , fontsize=10)
        #plt.ylim(bottom = 0)
        if showlogyaxis == "10":
            ax.semilogy()
        if showlogyaxis == "2":
            ax.semilogy(2)
        if showlogyaxis == "logit":
            ax.set_yscale("logit")
        # lay-out of the x axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval_))
        plt.gcf().autofmt_xdate()
        plt.gca().set_title(titlex , fontsize=10)

    # POS TESTS /day ################################
    with _lock:
        fig1, ax = plt.subplots()

        for a in datareeksen:
            plt.plot(x, a[0], label=f" {a[1]}", color = a[2] , linestyle='--')

        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        # Add X and y Label and limits
        configgraph(title,  showlogyaxis)
        plt.ylabel(title)
        st.pyplot(fig1)

def main_mathematisch(numberofpositivetests, NUMBEROFDAYS, datediff, Tg_A, R0, Rnew1_, Rnew2_, showimmunization, totalpopulation, total_immune_day_zero_A, testimmunefator, showlogyaxis, x):




    def calculate_cases(R, numberofpositivetests, total_immune_day_zero_A, Tg_A, NUMBEROFDAYS, showimmunization, totalpopulation, testimmunefator, ):


        positivetests1 = [numberofpositivetests]
        cummulative1 = [0]
        totalimmune_A=[total_immune_day_zero_A]
        r_A_in_de_tijd = [R]
        immeratio_A,immeratio_B = [],[]
        label1= f'Variant A | R0 = {R} / Tg = {Tg_A}'
        # START CALCULATING --------------------------------------------------------------------
        lambdaa = 1.0

        for t in range(1, NUMBEROFDAYS):
            if showimmunization:

                immeratio_A_ = (1-( (totalimmune_A[t-1]-totalimmune_A[0])/(totalpopulation-totalimmune_A[0])))
                ry_A = r_A_in_de_tijd[0]*(immeratio_A_**lambdaa)
                immeratio_A.append(immeratio_A_)
                r_A_in_de_tijd.append(ry_A)
            # prevent an [divide by zero]-error
            if ry_A == 1:
                ry_A = 1.000001
            if  ry_A <= 0:
                ry_A = 0.000001

            thalf1 = Tg_A * math.log(0.5) / math.log(ry_A)

            pt1 = round( (positivetests1[t-1] * (0.5**(1/thalf1))))
            positivetests1.append(pt1)
            totalimmune_A.append(totalimmune_A[t-1]+(pt1* testimmunefator))
            cpt1 = (cummulative1[t-1]+  pt1)

            if cpt1>=totalpopulation:
                cpt1 = totalpopulation
            cummulative1.append   (cpt1)

        return positivetests1


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



    ################################################
    positivetests0 = calculate_cases(R0, numberofpositivetests, total_immune_day_zero_A, Tg_A, NUMBEROFDAYS, showimmunization, totalpopulation, testimmunefator)
    positivetests1 = calculate_cases(Rnew1_, numberofpositivetests, total_immune_day_zero_A, Tg_A, NUMBEROFDAYS, showimmunization, totalpopulation, testimmunefator)
    positivetests2 = calculate_cases(Rnew2_, numberofpositivetests, total_immune_day_zero_A, Tg_A, NUMBEROFDAYS, showimmunization, totalpopulation, testimmunefator)

    datareeksen = [[positivetests0, R0, "red"],[positivetests1, Rnew1_, "green"],[positivetests2, Rnew2_, "blue"]]

    make_plot_math(x, "aantal cases mathematische groei", datareeksen, showlogyaxis)


def main_SIR(numberofpositivetests, NUMBEROFDAYS, datediff, b, R0_, factor1, factor2, totalpopulation, total_immune_day_zero_A,  x, incubationtime, infectioustime):
    suspectible =[]
    suspectible.append(totalpopulation - total_immune_day_zero_A)
    #recovered.append(totalimmunedayzero )


    # START CALCULATING --------------------------------------------------------------------

    # https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

    # Total population, N.
    #N = int(input("Total population, N "))
    #if N == 0 :
    N = int(totalpopulation)

    # Initial number of infected and recovered individuals, I0 and R0.
    I0 =  int(numberofpositivetests)
    E0 = 0
    V0 = 0
    D0 = 0
    R0= total_immune_day_zero_A
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
    beta = 1./incubationtime
    gamma = 1./infectioustime

    #beta = Rstart*gamma

    #beta, gamma = 0.2, 1./20

    # reproductionrate = beta / gamma

    # β describes the effective contact rate of the disease:
    # an infected individual comes into contact with βN other
    # individuals per unit time (of which the fraction that are
    # susceptible to contracting the disease is S/N).
    # 1/gamma is recovery rate in days

    # A grid of time points (in days)
    t = np.linspace(0, days, days)
    R0 =total_immune_day_zero_A

    I0_ = integrate(R0, N, I0, S0, beta, gamma, t, )
    I1_ = integrate(R0, N, I0, S0, beta*factor1, gamma, t, )
    I2_ = integrate(R0, N, I0, S0, beta*factor2, gamma, t, )

    datareeksen = [[I0_, f" = {R0_}", "red"],[I1_, f"*  {factor1}", "green"],[I2_, f"*  {factor2}", "blue"]]
    graph_SIR(datareeksen,"SIR model",x,b,datediff)


def graph_SIR(datareeksen,titlex,x,b,datediff):
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
    # New cases
    fig2c = plt.figure(facecolor='w')
    ax = fig2c.add_subplot(111,  axisbelow=True)
    for a in datareeksen:
        ax.plot(x, a[0], a[2], alpha=0.5,  label=f'Re {a[1]}')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    titlex = 'Infected'
    configgraph(titlex,x,b,datediff)
    plt.show()
    st.pyplot(fig2c)

def integrate(R0, N, I0,  S0, beta, gamma, t):
    #beta = R0_*gamma/(S0/N)
    # The SIR model differential equations.
    # https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/
    def deriv(y, t, N, beta, gamma):
        S,  I, R = y

        dSdt = 0 if S<=0 else (-beta * S * I / N)
        #dEdt = beta * S * I / N  - alfa * E
        dIdt = beta * S * I / N  - gamma * I
        #dCdt = alfa * E

        dRdt = (gamma * I)


        return dSdt,  dIdt, dRdt

    # Initial conditions vector
    y0 = S0,  I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R  = ret.T
    return I


def interface():
    DATE_FORMAT = "%m/%d/%Y"
    b = datetime.today().strftime('%m/%d/%Y')
    st.sidebar.title('Parameters')

    st.markdown("<hr>", unsafe_allow_html=True)
    a = st.sidebar.text_input('startdate (mm/dd/yyyy)',b)

    try:
        startx = dt.datetime.strptime(a,'%m/%d/%Y').date()
    except:
        st.error("Please make sure that the date is in format mm/dd/yyyy")
        st.stop()

    NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 20, 1440, 365)
    global numberofdays_
    numberofdays_ = NUMBEROFDAYS
    global Tg_A,Tg_B
    numberofpositivetests = st.sidebar.number_input('Total number of cases day 0',None,None,1_000)
    showimmunization = True # st.sidebar.checkbox("Immunization", True)
    totalpopulation = int(st.sidebar.number_input('Total population',0,1_000_000_000, 17_500_000))
    total_immune_day_zero_A = (st.sidebar.number_input('Total immune persons day zero var. A', 0, totalpopulation, 1_000))



    R0_number = st.sidebar.number_input('R0-number', 0.1, 10.0, 1.3)
    factor1 = st.sidebar.number_input('factor 1', 0.1, 10.0, 0.95)
    factor2 = st.sidebar.number_input('factor 2', 0.1, 10.0, 0.85)

    Tg_A = st.sidebar.slider('Generation time', 2.0, 11.0, 4.0)
    Rnew1_= round(R0_number * factor1,2)
    Rnew2_= round(R0_number * factor2,2)

    incubationtime = (st.sidebar.slider('Infection time (1/beta)', 1, 30, 2))
    infectioustime = (st.sidebar.slider('Recovery time (1/gamma)', 1, 30, 20))

    testimmunefator = 1 # st.sidebar.slider('cases/realityfactor', 0.0, 10.0, 2.50)
    showlogyaxis =  st.sidebar.selectbox("Y axis as log", ["No",  "10"], index=0)


    then = startx + dt.timedelta(days=NUMBEROFDAYS)
    x = mdates.drange(startx,then,dt.timedelta(days=1))
    z  = np.array(range(NUMBEROFDAYS))
    a_ = dt.datetime.strptime(a,'%m/%d/%Y').date()
    b_ = dt.datetime.strptime(b,'%m/%d/%Y').date()
    datediff = ( abs((a_ - b_).days))


    return numberofpositivetests,NUMBEROFDAYS, datediff, b, Tg_A, R0_number,Rnew1_,Rnew2_,factor1, factor2, showimmunization,totalpopulation,total_immune_day_zero_A,testimmunefator,showlogyaxis,x, incubationtime, infectioustime

def main():
    numberofpositivetests, NUMBEROFDAYS, datediff, b,  Tg_A, R0_number, Rnew1_, Rnew2_, factor1, factor2, showimmunization, totalpopulation, total_immune_day_zero_A, testimmunefator, showlogyaxis, x, incubationtime, infectioustime = interface()

    main_mathematisch(numberofpositivetests, NUMBEROFDAYS, datediff, Tg_A, R0_number, Rnew1_, Rnew2_, showimmunization, totalpopulation, total_immune_day_zero_A, testimmunefator, showlogyaxis, x)
    main_SIR(numberofpositivetests, NUMBEROFDAYS, datediff, b, R0_number, factor1, factor2, totalpopulation, total_immune_day_zero_A,  x, incubationtime, infectioustime)
if __name__ == "__main__":
    main()