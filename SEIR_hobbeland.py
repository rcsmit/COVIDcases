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
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
# _lock = RendererAgg.lock
from scipy.integrate import odeint
import plotly.graph_objects as go


def show_disclaimer(Rstart, alfa, gamma, beta):
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

    #####################################################



def show_footer():
    tekst = (
        '<style> .infobox {  background-color: lightblue; padding: 5px;}</style>'
        '<hr><div class=\'infobox\'>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\' target=\"_blank\">@rcsmit</a>) <br>'
        'Overdrachtstijd is 4 dagen. Disclaimer is following. Provided As-is etc.<br>'
        'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/edit/main/number_of_cases_interactive.py\" target=\"_blank\">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href=\"https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d\" target=\"_blank\">rcsmit.medium.com</a><br>'
        'Inspired by <a href=\"https://twitter.com/mzelst/status/1350923275296251904\" target=\"_blank\">this tweet</a> of Marino van Zelst.<br>'
        'With help of <a href=\"https://twitter.com/hk_nien" target=\"_blank\">Han-Kwang Nienhuys</a>.</div>')



    st.sidebar.markdown(tekst, unsafe_allow_html=True)

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



def plot_seirv(data):
    # Create traces for each curve
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data["t"], y=data["S"], mode='lines', line=dict(color='blue', width=2), opacity=0.5, name='Susceptible'))
    fig.add_trace(go.Scatter(x=data["t"], y=data["E"], mode='lines', line=dict(color='purple', width=2), opacity=0.5, name='Exposed'))
    fig.add_trace(go.Scatter(x=data["t"], y=data["V"], mode='lines', line=dict(color='yellow', width=2), opacity=0.5, name='Vaccinated'))
    fig.add_trace(go.Scatter(x=data["t"], y=data["I"], mode='lines', line=dict(color='red', width=2), opacity=0.5, name='Infected'))
    fig.add_trace(go.Scatter(x=data["t"], y=data["Cnew"], mode='lines', line=dict(color='orange', width=2), opacity=0.5, name='New Cases'))
    fig.add_trace(go.Scatter(x=data["t"], y=data["R"], mode='lines', line=dict(color='green', width=2), opacity=0.5, name='Recovered with immunity'))
    fig.add_trace(go.Scatter(x=data["t"], y=data["D"], mode='lines', line=dict(color='black', width=2), opacity=0.5, name='Death'))

    # Set titles and labels
    fig.update_layout(
        title=f'SEIR - {data["scenarioname"]}',
        xaxis_title='Time (days)',
        yaxis_title='Number',
        xaxis=dict(showticklabels=True, ticklen=0),
        yaxis=dict(showticklabels=True, ticklen=0),
    )

    # Show the figure
    st.plotly_chart(fig) 
def plot_new_cases(data):
    # New Cases Plot
    fig_new_cases = go.Figure()

    # Add traces for New Cases and Infected
    fig_new_cases.add_trace(go.Scatter(x=data["t"], y=data["Cnew"], mode='lines', line=dict(color='blue', width=2), name='New Cases'))
    fig_new_cases.add_trace(go.Scatter(x=data["t"], y=data["I"], mode='lines', line=dict(color='red', width=2), opacity=0.5, name='Infected'))

    # Update layout for New Cases plot
    fig_new_cases.update_layout(
        title='New Cases',
        xaxis_title='Time (days)',
        yaxis_title='Number',
        xaxis=dict(showticklabels=True, ticklen=0),
        yaxis=dict(showticklabels=True, ticklen=0)
    )

    # Display figure in Streamlit
    st.plotly_chart(fig_new_cases)
    

def plot_r_eff(data):
    # Gliding R-number Plot
    fig_r_number = go.Figure()

    # Add trace for R number based on Cnew
    fig_r_number.add_trace(go.Scatter(x=data["t"], y=data["repr"], mode='lines', line=dict(color='blue', width=2), opacity=0.5, name='R getal based on Cnew'))

    # Add a horizontal line at y=1
    fig_r_number.add_hline(y=1, line=dict(color='yellow', width=2, dash='dash'), opacity=0.6)

    # Update layout for Gliding R-number plot
    fig_r_number.update_layout(
        title='R_eff through time',
        xaxis_title='Time (days)',
        yaxis_title='R getal',
        xaxis=dict(showticklabels=True, ticklen=0),
        yaxis=dict(showticklabels=True, ticklen=0)
    )

    # Display figure in Streamlit
    st.plotly_chart(fig_r_number)

################################################

def main():
    data_dict = interface()
    numberofcasesdayz = data_dict["number_of_cases_day_zero"]
    totalimmunedayzero = data_dict["total_immunity_day_zero"]
    NUMBEROFDAYS = data_dict["number_of_days"]
    scenarioname = data_dict["scenario_name"]
    Rstart = data_dict["R_start"]
    ifr = data_dict["infection_fatality_rate"]
    incubationtime = data_dict["incubation_time"]
    infectioustime = data_dict["infectious_time"]
    start_day_vaccination = data_dict["start_day_vaccination"]
    days_needed_for_vaccination = data_dict["days_needed_for_vaccination"]
    totalpopulation = data_dict["total_population"]

    # Some manipulation of the x-values (the dates)
    # then = startx + dt.timedelta(days=NUMBEROFDAYS)
    # x = mdates.drange(startx,then,dt.timedelta(days=1))
    x=None
    z  = np.array(range(NUMBEROFDAYS))

    # a_ = dt.datetime.strptime(a,'%m/%d/%Y').date()
    # b_ = dt.datetime.strptime(b,'%m/%d/%Y').date()
    # datediff = ( abs((a_ - b_).days))

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
        if V < N and t > start_day_vaccination:
            zulu_= copy.deepcopy(zulu)
            N_=copy.deepcopy(N)
            dVdt = zulu_ * N_
        else:
            dVdt = 0
        dSdt = 0 if S<=0 else (-beta * S * I / N)  - (dVdt)
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

    t, lambdaa, repr, Cnew, repr_c_, repr_i_ = calculate_r_eff(Rstart, days, I, C)

    show_disclaimer(Rstart, alfa, gamma, beta)

    data = {"x":x, "t":t,"S":S,"E":E,"V":V,"I":I,"Cnew":Cnew,
            "R":R,"D":D,"scenarioname":scenarioname, 
            "repr":repr, "repr_c_":repr_c_, "repr_i_":repr_i_}
    plot_seirv(data)



    st.write  ("attack rate classical SIR model : " + str(int(C[days-1])) + " mensen / "+ str(round(100*((C[days-1])        /N),2))+ " %")
    st.write (f"Number of deaths: {round(C[days-1])} * {ifr} = {round(C[days-1]*ifr)}")

    st.markdown ("Theoretical herd immunity treshhold (HIT) (1 - [1/"+str(Rstart)+"]<sup>1/"+ str(lambdaa)+ "</sup>) : " + str(round(100*(1-((1/Rstart)**(1/lambdaa))),2))+ " % = " + str(round(N*(1-((1/Rstart)**(1/lambdaa))),0))+ " persons", unsafe_allow_html=True)
    st.write ("Attack rate = final size of the epidemic (FSE) ")

    plot_new_cases(data)
    plot_r_eff(data)


    st.write  ("attack rate classical SIR model : " + str(int(C[days-1])) + " mensen / "+ str(round(100*((C[days-1])        /N),2))+ " %")
    st.markdown ("Theoretical herd immunity treshhold (HIT) (1 - [1/"+str(Rstart)+"]<sup>1/"+ str(lambdaa)+ "</sup>) : " + str(round(100*(1-((1/Rstart)**(1/lambdaa))),2))+ " % = " + str(round(N*(1-((1/Rstart)**(1/lambdaa))),0))+ " persons", unsafe_allow_html=True)
    st.write ("Attack rate = final size of the epidemic (FSE) ")
    st.write("Read also: 7 Reasons Not to Use ODEs for Epidemic Modeling https://gerritgr.medium.com/7-reasons-not-to-use-odes-for-epidemic-modeling-bf451037a97f")
    repr=[]
    show_footer()

def interface():
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
    # a = st.sidebar.text_input('startdate (mm/dd/yyyy)',"03/01/2021")

    # try:
    #     startx = dt.datetime.strptime(a,'%m/%d/%Y').date()
    # except:
    #     st.error("Please make sure that the date is in format mm/dd/yyyy")
    #     st.stop()

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

    Tg = st.sidebar.slider('Generation time (to calculate Reff)', 2.0, 11.0, 4.0)
    global Tg_
    Tg_=Tg

    data_dict = {
        "number_of_cases_day_zero": numberofcasesdayz,
        "total_immunity_day_zero": totalimmunedayzero,
        "number_of_days": NUMBEROFDAYS,
        "scenario_name": scenarioname,
        "R_start": Rstart,
        "infection_fatality_rate": ifr,
        "incubation_time": incubationtime,
        "infectious_time": infectioustime,
        "start_day_vaccination": start_day_vaccination,
        "days_needed_for_vaccination": days_needed_for_vaccination,
        "total_population": totalpopulation
    }
    return data_dict
def calculate_r_eff(Rstart, days, I, C):
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
    return t,lambdaa,repr,Cnew,repr_c_,repr_i_



if __name__ == "__main__":
   
    print(
        f"-----------------------------------{datetime.now()}-----------------------------------------------------"
    )

    main()
