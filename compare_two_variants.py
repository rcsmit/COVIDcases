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



def make_plot(x,A,B,title, voorvoegsel_label, label1, label2, showlogyaxis):
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
        plt.plot(x, A, label=f"{voorvoegsel_label} {label1}", color = 'blue' , linestyle='--')
        plt.plot(x, B, label=f"{voorvoegsel_label} {label2}", color='orange', linestyle='--')


        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        # Add X and y Label and limits
        configgraph(title,  showlogyaxis)
        plt.ylabel(title)
        st.pyplot(fig1)

def main():

    DATE_FORMAT = "%m/%d/%Y"
    b = datetime.today().strftime('%m/%d/%Y')
    st.sidebar.title('Parameters')
    numberofpositivetests = st.sidebar.number_input('Total number of positive tests',None,None,10)

    st.markdown("<hr>", unsafe_allow_html=True)
    a = st.sidebar.text_input('startdate (mm/dd/yyyy)',"12/6/2021")

    try:
        startx = dt.datetime.strptime(a,'%m/%d/%Y').date()
    except:
        st.error("Please make sure that the date is in format mm/dd/yyyy")
        st.stop()

    NUMBEROFDAYS = st.sidebar.slider('Number of days in graph', 20, 720, 60)
    global numberofdays_
    numberofdays_ = NUMBEROFDAYS
    global Tg_A,Tg_B

    Rnew_1_ = st.sidebar.number_input('R-number  variant A', 0.1, 10.0, 1.3)

    Tg_A = st.sidebar.slider('Generation time variant A', 2.0, 11.0, 4.0)
    Rnew_2_ = st.sidebar.number_input('R-number variant B', 0.1, 10.0, 5.0)


    Tg_B = st.sidebar.slider('Generation time variant B', 2.0, 11.0, 2.0)
    correction = 1 # st.sidebar.number_input('Correction factor', 0.0, 2.0, 1.00)
    Rnew1_= round(Rnew_1_ * correction,2)
    Rnew2_= round(Rnew_2_ * correction,2)

    # lambdaa = st.sidebar.number_input('Lambda / heterogeneity', 1.0, 10.0, 1.0)


    showimmunization = True # st.sidebar.checkbox("Immunization", True)
    totalpopulation = (st.sidebar.number_input('Total population',0,1000000000, 17_500_000))
    total_immune_day_zero_A = (st.sidebar.number_input('Total immune persons day zero var. A', 0, totalpopulation, 1_000))
    total_immune_day_zero_B = (st.sidebar.number_input('Total immune persons day zero var. B', 0,  totalpopulation, 1_000))

    testimmunefactor = st.sidebar.slider('cases/realityfactor', 0.0, 10.0, 2.50)
    showlogyaxis =  st.sidebar.selectbox("Y axis as log", ["No",  "10"], index=0)

    then = startx + dt.timedelta(days=NUMBEROFDAYS)
    x = mdates.drange(startx,then,dt.timedelta(days=1))

    z  = np.array(range(NUMBEROFDAYS))

    a_ = dt.datetime.strptime(a,'%m/%d/%Y').date()
    b_ = dt.datetime.strptime(b,'%m/%d/%Y').date()
    datediff = ( abs((a_ - b_).days))



    positivetests1 = [numberofpositivetests]
    positivetests2 = [numberofpositivetests]

    cummulative1 = [0]
    cummulative2 = [0]

    totalimmune_A=[total_immune_day_zero_A]
    totalimmune_B=[total_immune_day_zero_B]
    r_A_in_de_tijd = [Rnew1_]
    r_B_in_de_tijd = [Rnew2_]
    immeratio_A,immeratio_B = [],[]

    label1= f'Variant A | R0 = {Rnew_1_} / Tg = {Tg_A}'
    label2= f'Variant B | R0 = {Rnew_2_} / Tg = {Tg_B}'

    # START CALCULATING --------------------------------------------------------------------
    lambdaa = 1.0

    for t in range(1, NUMBEROFDAYS):
        if showimmunization:

            immeratio_A_ = (1-( (totalimmune_A[t-1]-totalimmune_A[0])/(totalpopulation-totalimmune_A[0])))
            immeratio_B_ = (1-( (totalimmune_B[t-1]-totalimmune_B[0])/(totalpopulation-totalimmune_B[0])))
            ry_A = r_A_in_de_tijd[0]*(immeratio_A_**lambdaa)
            ry_B = r_B_in_de_tijd[0]*(immeratio_B_**lambdaa)
            immeratio_A.append(immeratio_A_)
            immeratio_B.append(immeratio_B_)

            r_A_in_de_tijd.append(ry_A)
            r_B_in_de_tijd.append(ry_B)
        # prevent an [divide by zero]-error
        if ry_A == 1:
            ry_A = 1.000001
        if  ry_B == 1:
            ry_B = 1.000001
        if  ry_A <= 0:
             ry_A = 0.000001
        if ry_B <= 0:
            ry_B = 0.000001

        thalf1 = Tg_A * math.log(0.5) / math.log(ry_A)
        thalf2 = Tg_B * math.log(0.5) / math.log(ry_B)

        pt1 = round( (positivetests1[t-1] * (0.5**(1/thalf1))))
        pt2 = round((positivetests2[t-1] * (0.5**(1/thalf2))))
        positivetests1.append(pt1)
        positivetests2.append(pt2)
        totalimmune_A.append(totalimmune_A[t-1]+(pt1*testimmunefactor))
        totalimmune_B.append(totalimmune_B[t-1]+(pt2*testimmunefactor))

        cpt1 = (cummulative1[t-1]+  pt1)
        cpt2 = (cummulative2[t-1]+  pt2 )


        if cpt1>=totalpopulation:
            cpt1 = totalpopulation
        if cpt2>=totalpopulation:
            cpt2 = totalpopulation


        cummulative1.append   (cpt1)
        cummulative2.append   (cpt2 )




    st.title('Compare two variants with different R0 and Tg (not combined!)')
    st.write("Just as thinking excercise")

    make_plot (x,positivetests1, positivetests2,'positivetests per day', '', label1, label2, showlogyaxis)
    make_plot (x,cummulative1,cummulative2, "Cummulative cases", 'Cumm.', label1, label2, showlogyaxis)
    make_plot (x,r_A_in_de_tijd,r_B_in_de_tijd, "R(t)", 'R(t)', label1, label2, showlogyaxis)


if __name__ == "__main__":
    main()
