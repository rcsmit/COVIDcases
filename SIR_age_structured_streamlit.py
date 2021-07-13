# MAKE AGE STRUCTURED SIR DIAGRAMS

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import pandas as pd

import streamlit as st
from streamlit import caching

@st.cache()
def get_contact_matrix(moment,contact_type):
    """Get the contactmatrix

    Args:
        moment (string):  ["2016/-17", "April2020", "June2020"]
        contact_type (string) : ["all", "community", "household"]
    """
    df= pd.read_csv(
            "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/contactmatrix.tsv",
            # "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\contactmatrix.tsv",
            comment="#",
            delimiter="\t",
            low_memory=False,
        )

    df = df.replace("[5,10)", "[05,10)")
    df = df.replace ("baseline", "2016/-17")
    df = df.rename(columns={'part_age':'participant_age'})
    df = df.rename(columns={'cont_age':'contact_age'})

    df_first =  df[(df['survey'] == moment) & (df['contact_type'] == contact_type)]
    df_first_pivot =  df_first.pivot_table(index='contact_age', columns='participant_age', values="m_est", margins = True, aggfunc=sum)
    # we assume that the contact rates for 0-5 are the same as 6-10
    # df_first_pivot = df_first_pivot.iloc[1: , :] # drop first row
    # df_first_pivot = df_first_pivot.iloc[: , 1:] # drop first column
    return df_first_pivot

def func(t, state, *argv):
    """The function with the formula

    Args:
        t (?): timepoints
        state (?): Numbers of S, I and R
        argv : groupsizes, beta's and gamma's
    Returns:
        [?]: the differences in each step (dSdt+ dIdt + dRdt)
    """
    lijst = list(state)
    arguments = [xy for xy in argv]
    df_contactrate = get_contact_matrix("2016/-17","all")

    S,E, I,R, C, H, IC, D =  [],[],[],[], [],[],[],[]
    N, alfa, beta,gamma, correction_per_age =  [],[],[],[],[]
    dSdt, dEdt, dIdt, dRdt, dCdt, dHdt, dICdt, dDdt = [], [],[], [], [],[],[],[]
    for i in range (len(lijst)):
        if i < number_of_agegroups:
            S.append(lijst[i])
        elif i>=number_of_agegroups and i < 2*number_of_agegroups:
            E.append(lijst [i])
        elif i>=2*number_of_agegroups and i < 3*number_of_agegroups:
            I.append(lijst[i])
        elif i>=3*number_of_agegroups and i < 4*number_of_agegroups:
            R.append(lijst[i])

        elif i>=4*number_of_agegroups and i < 5*number_of_agegroups:
            C.append(lijst[i])
        elif i>=5*number_of_agegroups and i < 6*number_of_agegroups:
            H.append(lijst[i])
        elif i>=6*number_of_agegroups and i < 7*number_of_agegroups:
            IC.append(lijst[i])
        elif i>=7*number_of_agegroups and i < 8*number_of_agegroups:
            D.append(lijst[i])

    for i in range (len(arguments)):
        if i < number_of_agegroups:
            N.append(arguments[i])
        elif i>=number_of_agegroups and  i < 2*number_of_agegroups:
            alfa.append(arguments [i])
        elif i>=2*number_of_agegroups and  i < 3*number_of_agegroups:
            beta.append(arguments [i])
        elif i >= 3*number_of_agegroups and i < 4*number_of_agegroups:
            gamma.append(arguments[i])
        elif i >= 4*number_of_agegroups and i < 5*number_of_agegroups:
            correction_per_age.append(arguments[i])
        else:
            print("error")

    # Normally I shoud define these in the main()
    names =     ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

    fraction =  [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]
    h_rivm  =   [0.00347, 0.000377, 0.000949, 0.00388, 0.00842,0.0165, 0.0251, 0.0494, 0.0463] # I -> H
    h_  =       [0.0015, 0.0001, 0.0002, 0.0007, 0.0013, 0.0028, 0.0044, 0.0097, 0.0107] # I -> H - time to go to hospital taken in account
    i1_ =       [0.0000, 0.0271, 0.0422, 0.0482, 0.0719, 0.0886, 0.0170, 0.0860, 0.0154] # H-> IC
    ic_opn = [ 0, 2.8402E-05, 0.000211306, 0.000609427, 0.001481364, 0.003788442, 0.006861962, 0.008609547, 0.00210745]

    i2_ =       [0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0531, 0.0080, 0.0367, 0.0356] # IC-> H
    d_  =       [0.0003, 0.0006, 0.0014, 0.0031, 0.0036, 0.0057, 0.0151, 0.0327, 0.0444] # H-> D
    dic_ =      [0.0071, 0.0071, 0.0071, 0.0071, 0.0071, 0.0090, 0.0463, 0.0225, 0.0234] # IC -> D
    dhic_ =     [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0010, 0.0040, 0.0120, 0.0290] # IC -> h -> D
    r_    =     [0.1263, 0.1260, 0.1254, 0.1238, 0.1234, 0.1215, 0.1131, 0.0976, 0.0872] # recovery rate from hospital (before IC)
    ric_  =     [0.0857, 0.0857, 0.0857, 0.0857, 0.0857, 0.0821, 0.0119, 0.0567, 0.0550] # recovery rate from hospital (after IC)
    LE_   =     [ 77.89,  67.93,  58.08,  48.48,  38.60,  29.22,  20.52,  12.76,  4.35 ] # Life expectancy
    rel_besmh  =     [ 3,  3,  3,  3,  3,  3,  1,  1,  1] # Relative infectiousness https://www.pnas.org/content/117/36/22430
    rel_vatbh_rivm  =     [ 1.000,  3.051,  5.751,  3.538,  3.705,  4.365,  5.688,  5.324,  7.211] # ik dnek dat deze matrix het totaal is van contactmatrix, besm, en vatbh

    rel_vatbh = [0.5,0.75,1,2.5,4,5,8,8,8]     # Relative suspceptibility fig 1b, op het ooghttps://www.nature.com/articles/s41591-020-0962-9

    #rel_besmh, rel_vatbh =   [1] * number_of_agegroups,   [1] * number_of_agegroups


                # Aantallen van https://www.rivm.nl/coronavirus-covid-19/grafieken geldeeld op 4836661 infecties (cummulatieve  prevalentie gedeeld door 8 )
    ifr_ =      [2.04658E-06, 3.78694E-06, 1.76088E-05, 5.45016E-05, 0.000156108, 0.000558534, 0.002271095, 0.009964733, 0.048248607 ]

    for i in range(number_of_agegroups):

        lambdaa = 1
        cumm_cfactor = 0
        for j in range(number_of_agegroups):
            cijt = df_contactrate.iat[j+1,i+1] * sum(N) / (N[i]*N[j])
            cumm_cfactor += cijt * rel_besmh[i]*I[j]
        lambdaa = (beta[i] * rel_vatbh[i] * cumm_cfactor ) / sum(N)
        # lambdaa = (beta[i] * rel_vatbh_rivm [i]  ) / sum(N)
        dSdt.append( - lambdaa * S[i] *  I[i] * correction_per_age[i]* rutte_factor )
        dEdt.append (( lambdaa * S[i] *  I[i] * correction_per_age[i]* rutte_factor) - (alfa[i] * E[i]))
        dIdt.append(                                     (alfa[i] * E[i]) - (gamma[i] * I[i]) - ( ifr_[i] *  I[i]))
        dRdt.append(                                                          gamma[i] * I[i]  - ( ifr_[i] * I[i]))
        dCdt.append(                                      (alfa[i] * E[i]))
        dHdt.append(                                       (alfa[i] * E[i]) * h_rivm[i])
        dICdt.append (                                       (alfa[i] * E[i]) * ic_opn[i])
        #dDdt.append( C[i]*ifr_[i])
        dDdt.append( ( ifr_[i]* (alfa[i] * E[i])))

    to_return = dSdt+ dEdt + dIdt + dRdt + dCdt + dHdt + dICdt + dDdt
    return to_return

def draw_graph_with_all_groups (result_odeint, names, beta, gamma, t, N):
    """Draws graphs with subgraphs of each agegroup and total

    Args:
        result_odeint (?): result of the ODEint

        names (list): names of the groups for the legenda
        beta (list): for the legenda
        gamma (list): for the legenda
        t (list): timevalues, for the number_of_agegroups-axis
    """
    show_all = True
    total_pop = sum(N)
    with _lock:

        fig = plt.figure()
        for i in range (number_of_agegroups):
            ax = fig.add_subplot((round(number_of_agegroups/2)+1), 2,i+1)
            ax.plot(t, result_odeint[:, i]/N[i], "black", lw=1.5, label="Susceptible")
            ax.plot(t, result_odeint[:, number_of_agegroups+i]/N[i], "purple", lw=1.5, label="Exposed")
            ax.plot(t, result_odeint[:, (2*number_of_agegroups)+i]/N[i], "orange", lw=1.5, label="Infected")
            ax.plot(t, result_odeint[:, (3*number_of_agegroups)+i]/N[i], "blue", lw=1.5, label="Recovered")
            ax.plot(t, result_odeint[:, (4*number_of_agegroups) +i]/N[i], "green", lw=1.5, label="Cases")
            ax.set_title(f"{ names[i]}",  fontsize=10)
            ax.set_ylim([0,1])
        #fig.tight_layout()
        plt.legend()
        plt.grid()
        ax.set_ylim([0,1])
        #plt.show()
        st.pyplot(fig)
    with _lock:
        fig = plt.figure()
        for i in range (number_of_agegroups):
            ax = fig.add_subplot((round(number_of_agegroups/2)+1), 2,i+1)
            ax.plot(t, result_odeint[:, (4*number_of_agegroups) +i]/N[i], "black", lw=1.5, label="Cases")
            ax.plot(t, result_odeint[:, (5*number_of_agegroups)+i]/N[i], "purple", lw=1.5, label="Hospital")
            ax.plot(t, result_odeint[:, (6*number_of_agegroups)+i]/N[i], "orange", lw=1.5, label="IC")
            ax.plot(t, result_odeint[:, (7*number_of_agegroups)+i]/N[i], "blue", lw=1.5, label="Death")
            ax.set_title(f"{ names[i]}",  fontsize=10)
            ax.set_ylim([0,1])
           #fig.tight_layout()
        plt.legend()
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ratio')
        plt.grid()
        #plt.show()
        st.pyplot(fig)


def plot_single_age_group(item, result_odeint, names,  t, N):
    # single age group
    i = names.index(item)
    with _lock:

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(t, result_odeint[:, i]/N[i], "black", lw=1.5, label="Susceptible")
        ax.plot(t, result_odeint[:, number_of_agegroups+i]/N[i], "purple", lw=1.5, label="Exposed")
        ax.plot(t, result_odeint[:, (2*number_of_agegroups)+i]/N[i], "orange", lw=1.5, label="Infected")
        ax.plot(t, result_odeint[:, (3*number_of_agegroups)+i]/N[i], "blue", lw=1.5, label="Recovered")
        ax.set_title(f"{ names[i]}",  fontsize=10)
        plt.legend()
        plt.grid()
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ratio')
        ax.set_ylim([0,1])
        #plt.show()
        st.pyplot(fig)

def plot_total_as_ratio(result_odeint, total_pop):
    S_tot_odeint, E_tot_odeint, I_tot_odeint, R_tot_odeint = 0.0,0.0,0.0,0.0
    C_tot_odeint, H_tot_odeint, IC_tot_odeint, D_tot_odeint = 0.0,0.0,0.0,0.0
    for i in range(number_of_agegroups):
        S_tot_odeint +=result_odeint[:, i]
        E_tot_odeint += result_odeint[:, number_of_agegroups+i]
        I_tot_odeint += result_odeint[:, (2*number_of_agegroups)+i]
        R_tot_odeint += result_odeint[:, (3*number_of_agegroups)+i]
        C_tot_odeint += result_odeint[:, (4*number_of_agegroups)+i]
        H_tot_odeint += result_odeint[:, (5*number_of_agegroups)+i]
        IC_tot_odeint += result_odeint[:, (6*number_of_agegroups)+i]
        D_tot_odeint += result_odeint[:, (7*number_of_agegroups)+i]
    with _lock:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(S_tot_odeint/total_pop, "black", lw=1.5, label="Susceptible")
        ax.plot(E_tot_odeint/total_pop, "purple", lw=1.5, label="Exposed")
        ax.plot(I_tot_odeint/total_pop, "orange", lw=1.5, label="Infected")
        ax.plot(R_tot_odeint/total_pop, "blue", lw=1.5, label="Recovered")
        ax.plot(C_tot_odeint/total_pop, "green", lw=1.5, label="Cases")
        ax.set_title("Totaal")
        #fig.tight_layout()
        ax.set_xlabel('Time (days)')
        if total_pop == 1 :
            ax.set_ylabel('Numbers')
        else:
            ax.set_ylabel('Ratio')
            ax.set_ylim([0,1])

        plt.legend()
        plt.grid()
        #plt.show()
        st.pyplot(fig)

def show_result(result_odeint, N):
    d = []
    tot_cases,tot_hosp, tot_ic, tot_deaths = 0,0,0,0

    for i in range(number_of_agegroups):
        tot_cases += int(result_odeint[-1, (4*number_of_agegroups) +i])
        tot_hosp  += int(result_odeint[-1, (5*number_of_agegroups) +i])
        tot_ic    += int(result_odeint[-1, (6*number_of_agegroups) +i])
        tot_deaths+= int(result_odeint[-1, (7*number_of_agegroups) +i])

        d.append((names[i],
            int(result_odeint[-1, (4*number_of_agegroups) + i]),
            round((result_odeint[-1, (4*number_of_agegroups) + i]/N[i] * 100), 2),
            round((result_odeint[-1, (5*number_of_agegroups)+i])),
            round((result_odeint[-1, (6*number_of_agegroups)+i])),
            round((result_odeint[-1, (7*number_of_agegroups)+i])),
            round((result_odeint[-1, (7*number_of_agegroups) + i] / (result_odeint[-1, (4*number_of_agegroups) + i]) * 100), 2)
            ))
    d.append(("TOTAL", tot_cases, round((tot_cases/sum(N)*100),1 ), tot_hosp, tot_ic, tot_deaths, round((tot_deaths/tot_cases*100),2)))
    df_result = pd.DataFrame(d, columns=('Agegroup', 'cases', 'attackrate (%)', 'hospital', 'ic', 'deaths', 'ifr (%)' ))
    st.write (df_result)


def show_toelichting():
    st.subheader ("TOELICHTING")
    st.write ("This model is a very, very simplified version of the RIVM model,")
    st.write ("'Unique' of  this model is that it has agegroups, keeps in account the contact rates between groups and the relative suspceptibility/infectiousness of the different ages")
    st.write ("Limits for now: There are only 8 compartments. People immediately go to hospital, IC or death after getting infected. No vaccination. No probabilities for outcomes. No seasonality")

    st.subheader ("Sources: ")

    st.write("* Beschrijving transmissiemodel berekening zorgbelasting Voorlopige versie, dd. 2 april 2020 https://www.rivm.nl/sites/default/files/2021-04/beschrijving_transmissiemodel%20beveiligd.pdf")
    st.write("* Ainslie et al. The expected outcome of COVID-19 vaccination strategies (https://www.rivm.nl/sites/default/files/2021-03/Modellingresults%20COVID19%20vaccination%20version1.0%2020210324_0.pdf) ")

    # ZIEKENHUIS-IC OPNAMES BEREKEND DOOR RENE
    # a. totaal aantal infected berekend mbp cummulatieve prevalentie geldeeld door 8
    # b. attack berekend: a/17.4mln
    # c. populatiefractie per leeftijd * b = aantal infected per leeftijdsgroep
    # d. De waardes van  "COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep" / c
def input_correction_age():
    st.sidebar.subheader("Correction per agegroup")
    a0 = st.sidebar.number_input(names[0], 0.0,  10.0, 3.0)
    a1 = st.sidebar.number_input(names[1], 0.0,  10.0, 1.0)
    a2 = st.sidebar.number_input(names[2], 0.0,  10.0, 1.0)
    a3 = st.sidebar.number_input(names[3], 0.0,  10.0, 1.0)
    a4 = st.sidebar.number_input(names[4], 0.0,  10.0, 1.0)
    a5 = st.sidebar.number_input(names[5], 0.0,  10.0, 1.0)
    a6 = st.sidebar.number_input(names[6], 0.0,  10.0, 1.0)
    a7 = st.sidebar.number_input(names[7], 0.0,  10.0, 1.0)
    a8 = st.sidebar.number_input(names[8], 0.0,  10.0, 3.0)
    return [a0,a1,a2,a3,a4,a5,a6,a7,a8]
def main():
    global number_of_agegroups, names
    names =     ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    number_of_agegroups = len (names) # number of agegroups

    N =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    total_pop = sum(N)
    I0 ,E0 = [], []
    initial_exposed_ratio, initial_infected_ratio = 0.03, 0.02
    for y in range (number_of_agegroups):
        E0.append (N[y]* initial_exposed_ratio)
        I0.append (N[y]* initial_infected_ratio)
    R0, S0,  C0 =   [0] * number_of_agegroups, [0] * number_of_agegroups, [0] * number_of_agegroups
    H0, IC0, D0 =   [0] * number_of_agegroups, [0] * number_of_agegroups, [0] * number_of_agegroups

    # IF YOU WANT TO START FROM AN R-NAUGHT
    # Rstart = [1.1]*number_of_agegroups
    # beta = []
    # for y in range (number_of_agegroups):
    #     beta[y] = Rstart[y]*gamma[y]/(S0[y]/N[y])

    st.sidebar.subheader("Parameters")

    incubationtime = (st.sidebar.slider('Incubatietijd (1/alfa)', 1, 30, 4))
    beta_ = st.sidebar.number_input(
                            "Contact rate (beta)",
                            min_value=0.0,
                            max_value=1.0,
                            step=1e-4,
                            value = 0.3100,
                            format="%.4f")
    infectioustime = (st.sidebar.slider('Average days infectious (1/gamma)', 1, 30, 8))
    alfa = [1/incubationtime]*number_of_agegroups # 1/incubation time
    beta = [beta_] * number_of_agegroups # contact rate
    gamma = [1/infectioustime] * number_of_agegroups # mean recovery rate (1/recovery days/infectious time)

    global rutte_factor
    rutte_factor = st.sidebar.slider('Rutte factor (seasonality, maatregelen, verspoepelingen', 0.0, 10.0, 1.0)
    correction_per_age = input_correction_age()

    for y in range(number_of_agegroups):
        S0[y] = N[y] - E0[y]- I0[y] - R0[y]

    y0 = tuple(S0 + E0 + I0 + R0 + C0 + H0 +IC0 + D0)
    p = tuple(N + alfa + beta + gamma + correction_per_age)
    n = 176 # number of time points
    t = np.linspace(0, n-1, n) # time points

    result_odeint = odeint(func, y0, t, p, tfirst=True)

    st.subheader("Totals")
    show_result(result_odeint, N)
    #  draw_graph_with_all_groups(result_odeint, names, beta, gamma, t,N)
    plot_total_as_ratio(result_odeint, 1)
    st.subheader("Per age group")
    for name in names:
        plot_single_age_group(name, result_odeint, names,  t, N)

    #plot_total_as_ratio(result_odeint, total_pop)

    st.subheader("Contact matrix")
    st.write(get_contact_matrix("2016/-17","all"))


    show_toelichting()



if __name__ == '__main__':
    main()
