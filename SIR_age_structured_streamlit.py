# MAKE AGE STRUCTURED SIR DIAGRAMS

from math import e
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
            # "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\contactmatrix.tsv",
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

    # drop first row and column (TOFIX : combine 0-5 and 5-10)
    df_first_pivot =df_first_pivot.iloc[: , 1:]
    df_first_pivot =df_first_pivot.iloc[1: , :]
    return df_first_pivot
def calculate_c_new(result_odeint, i):
    """Calculate new cases per day

    Args:
        result_odeint (array): The results
        i (int): which agegroup. None for total

    Returns:
        array : Array with cases per day
    """
    c_new_tot=[0]
    C_tot_day = 0

    for x in range(1,len (result_odeint)):
        if i == None:
            for i in range(number_of_agegroups):
                C_tot_day += result_odeint[x, (4*number_of_agegroups)+i] - result_odeint[x-1, (4*number_of_agegroups)+i]
            c_new_tot.append(C_tot_day)

        else:
            C_tot_day = result_odeint[x, (4*number_of_agegroups)+i] - result_odeint[x-1, (4*number_of_agegroups)+ i]
            c_new_tot.append(C_tot_day)
        C_tot_day = 0
    c_new_tot_as_array = np.array(c_new_tot)

    return c_new_tot_as_array


def plot_single_age_group(item, result_odeint, names,  t, N, what_to_show):
    # single age group
    i = names.index(item)
    with _lock:
        C_new_tot =  calculate_c_new(result_odeint, i)
        fig = plt.figure()
        ax = fig.add_subplot()

        ratio = False
        if ratio == True:
            noemer = N[i]
        else:
            noemer = 1 # aantallen

        if "S" in what_to_show : ax.plot(t, result_odeint[:, i]/noemer, "pink", lw=1.5, label="Susceptible")
        if "E" in what_to_show : ax.plot(t, result_odeint[:, number_of_agegroups+i]/noemer, "purple", lw=1.5, label="Exposed")
        if "I" in what_to_show : ax.plot(t, result_odeint[:, (2*number_of_agegroups)+i]/noemer, "orange", lw=1.5, label="Infected")
        if "R" in what_to_show : ax.plot(t, result_odeint[:, (3*number_of_agegroups)+i]/noemer, "blue", lw=1.5, label="Recovered")
        #if "C" in what_to_show : ax.plot(t, result_odeint[:, (4*number_of_agegroups)+i]/noemer, "green", lw=1.5, label="Cases cumm")
        if "H" in what_to_show : ax.plot(t, result_odeint[:, (5*number_of_agegroups)+i]/noemer, "yellow", lw=1.5, label="Hospital")
        if "IC" in what_to_show : ax.plot(t, result_odeint[:, (6*number_of_agegroups)+i]/noemer, "brown", lw=1.5, label="IC")
        if "D" in what_to_show : ax.plot(t, result_odeint[:, (7*number_of_agegroups)+i]/noemer, "black", lw=1.5, label="Death")

        if "C" in what_to_show : ax.plot(C_new_tot/noemer, "green", linestyle="--", lw=1.5, label="Cases")

         # ax.plot(result_solve_ivp.y[6+i, :], "blue", lw=1.5, label="Recovered")
        ax.set_title(f"{ names[i]}",  fontsize=10)
        plt.legend()
        #plt.grid()
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Ratio')
        #ax.set_ylim([0,1])
        #plt.show()
        st.pyplot(fig)

def plot_total(result_odeint, noemer, what_to_show):
    """Plot results. Total of all agegroups

    Args:
        result_odeint (array): [description]
        noemer (int): Total population. 1 if you want the numbers
        what_to_show (list): which compartments to show
    """
    S_tot_odeint, E_tot_odeint, I_tot_odeint, R_tot_odeint = 0.0,0.0,0.0,0.0
    C_tot_odeint, H_tot_odeint, IC_tot_odeint, D_tot_odeint = 0.0,0.0,0.0,0.0
    C_new_tot =  calculate_c_new(result_odeint, None)
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
        if "S" in what_to_show : ax.plot(S_tot_odeint/noemer, "pink", lw=1.5, label="Susceptible")
        if "E" in what_to_show : ax.plot(E_tot_odeint/noemer, "purple", lw=1.5, label="Exposed")
        if "I" in what_to_show : ax.plot(I_tot_odeint/noemer, "orange", lw=1.5, label="Infected")
        if "R" in what_to_show : ax.plot(R_tot_odeint/noemer, "blue", lw=1.5, label="Recovered")
        if "C" in what_to_show : ax.plot(C_tot_odeint/noemer, "green",  lw=1.5, label="Cases cumm")
        if "C" in what_to_show : ax.plot(C_new_tot/noemer, "green", linestyle="--", lw=1.5, label="Cases")
        if "H" in what_to_show : ax.plot(H_tot_odeint/noemer, "yellow", lw=1.5, label="Hospital")
        if "IC" in what_to_show : ax.plot(IC_tot_odeint/noemer, "brown", lw=1.5, label="IC")
        if "D" in what_to_show : ax.plot(D_tot_odeint/noemer, "black", lw=1.5, label="Death")
        ax.set_title("Totaal")
        #fig.tight_layout()
        ax.set_xlabel('Time (days)')
        if noemer == 1 :
            ax.set_ylabel('Numbers')
        else:
            ax.set_ylabel('Ratio')
            ax.set_ylim([0,1])

        plt.legend()
        #plt.grid()
        #plt.show()
        st.pyplot(fig)

def show_result(result_odeint, N):
    """Print dataframe/table with the results

    Args:
        result_odeint (array): Result of the solver
        N (List w/ integers): List with agegroup sizes
    """
    d = []
    tot_cases,tot_hosp, tot_ic, tot_deaths, tot_hic, tot_deaths_ifr = 0,0,0,0,0,0
    # ["S", "E",  "I", "R", "C", "H", "IC","HIC","DIFR", "D"] Hcumm ICcumm
    #   0    1     2    3    4    5    6    7     8       9   10     11
    for i in range(number_of_agegroups):
        tot_cases += int(result_odeint[-1, (4*number_of_agegroups) +i])
        tot_hosp  += int(result_odeint[-1, (10*number_of_agegroups) +i])
        tot_ic    += int(result_odeint[-1, (11*number_of_agegroups) +i])

        tot_deaths_ifr+= int(result_odeint[-1, (8*number_of_agegroups) +i])
        tot_deaths+= int(result_odeint[-1, (9*number_of_agegroups) +i])

        d.append((names[i],
            int(result_odeint[-1, (4*number_of_agegroups) + i]),
            round((result_odeint[-1, (4*number_of_agegroups) + i]/N[i] * 100), 2),
            round((result_odeint[-1, (10*number_of_agegroups)+i])),

            round((result_odeint[-1, (11*number_of_agegroups)+i])),
            round((result_odeint[-1, (8*number_of_agegroups)+i])),
            round((result_odeint[-1, (9*number_of_agegroups)+i])),
            round((result_odeint[-1, (9*number_of_agegroups) + i] / (result_odeint[-1, (4*number_of_agegroups) + i]) * 100), 2)
            ))
    d.append(("TOTAL", tot_cases, round((tot_cases/sum(N)*100),1 ), tot_hosp, tot_ic,  tot_deaths_ifr, tot_deaths, round((tot_deaths/tot_cases*100),2)))
    df_result = pd.DataFrame(d, columns=('Agegroup', 'cases', 'attackrate (%)', 'hospital', 'ic', 'deaths_from_ifr', 'deaths_model', 'ifr_model (%)' ))
    st.write (df_result)

def show_toelichting():
    """Generate footer
    """
    st.subheader ("TOELICHTING")
    st.write ("This model is a very, very simplified version of the RIVM model,")
    st.write ("'Unique' of  this model is that it has agegroups, keeps in account the contact rates between groups and the relative suspceptibility/infectiousness of the different ages")
    st.write ("Limits for now: There are only 8 compartments. People immediately go to hospital, IC or death after getting infected. No vaccination. No probabilities for outcomes. No seasonality")

    st.subheader ("Bronnen model: ")

    st.write("* Beschrijving transmissiemodel berekening zorgbelasting Voorlopige versie, dd. 2 april 2020 https://www.rivm.nl/sites/default/files/2021-04/beschrijving_transmissiemodel%20beveiligd.pdf")
    st.write("* Ainslie et al. The expected outcome of COVID-19 vaccination strategies (https://www.rivm.nl/sites/default/files/2021-03/Modellingresults%20COVID19%20vaccination%20version1.0%2020210324_0.pdf) ")

    st.subheader ("Bronnen parameters: ")
    st.write("* Ziekenhuis, IC en IFR: waardes uit ziekenhuis_ic_opnames_per_leeftijdsgroep.csv gedeeld op de cummulatieve prevalentie gedeeld door 8")
    st.write("* IFR: waardes van  https://www.rivm.nl/coronavirus-covid-19/grafieken (overleden naar leeftijd en geslacht) gedeeld op de cummulatieve prevalentie gedeeld door 8")
    st.write("* Relative infectiousness - Lau, 2020 : https://www.pnas.org/content/117/36/22430")
    st.write("* Relative suspceptibility fig 1b, op het oog - Davies, 2020 : https://www.nature.com/articles/s41591-020-0962-9")

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/blob/main/SIR_age_structured_streamlit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        'More scripts: <a href="https://share.streamlit.io/rcsmit/covidcases/main/covid_menu_streamlit.py</div>'
    )

    st.markdown(tekst, unsafe_allow_html=True)

def input_parameters(naam, defaults):
    resultaat = []
    a=[]

    for i in (range(len(defaults))):
        a.append(None)

    st.subheader(naam)
    for i in (range(len(defaults))):
        a[i] = st.number_input(naam + names[i], None, None, float(defaults[i]), format="%.4f")
        resultaat.append(a[i])
    return resultaat


def sublist(orig_list, list_of_subs, max_items_per_list):
    """Make a sublist ([a,a,b,b,c,c], [AX,BX,CX],2) -> AX = [a,a], BX = [b,b], CX = [c,c]]

    Args:
        orig_list ([type]): [description]
        list_of_subs ([type]): [description]
        max_items_per_list ([type]): [description]

    Yields:
        [type]: [description]
    """
    # https://stackoverflow.com/questions/52111627/how-to-split-a-list-into-multiple-lists-with-certain-ranges

    def sublist_generator():
        for sublist in list_of_subs:
            yield sublist

    sublist = sublist_generator()
    current_sublist = next(sublist)
    for i, element in enumerate(orig_list):
        current_sublist.append(element)

        if len(current_sublist) == max_items_per_list and (i != len(orig_list)-1): # current list is full
            current_sublist = next(sublist) # so let current point to the next list


def func(t, state, *argv):
    """The function with the formula

    Args:
        t (?): timepoints
        state (?): Numbers of S, I and R and other numbers
        argv : groupsizes, beta's and gamma's and other parameters
    Returns:
        [?]: the differences in each step (dSdt+ dIdt + dRdt)
    """
    lijst = list(state)
    arguments = [xy for xy in argv]
    df_contactrate = get_contact_matrix("2016/-17","all")

    S,E, I,R, C, H, HIC, IC, DIFR, D, Hcumm, ICcumm =  [],[],[],[], [],[],[],[],[],[],[],[]

    N, sigma, alfa, beta,gamma, correction_per_age =  [],[],[],[],[],[]
    rel_besmh, rel_vatbh, ifr, dfromi = [],[],[],[]
    h,  i1,  i2,  d,  dic,  dhic,  r,  ric =  [],[],[],[], [],[],[],[]
    dSdt, dEdt, dIdt, dRdt, dCdt, dHdt, dICdt, dHICdt, dDIFRdt, dDdt, dHcummdt, dICcummdt = [], [],[], [], [],[],[],[],[],[],[],[]
    sublist_compartments = [S,  E,  I, R, C, H, IC, HIC, DIFR, D, Hcumm, ICcumm]
    sublist_parameters = [N,  alfa,  beta,  gamma,  sigma,   rel_besmh,  rel_vatbh,  ifr,
                         h,  i1,  i2,  d,  dic,  dhic,  r,  ric, correction_per_age, dfromi]

    sublist(lijst, sublist_compartments,  number_of_agegroups)
    sublist(arguments, sublist_parameters,  number_of_agegroups)
    for i in range(number_of_agegroups):
        lambdaa = 1
        cumm_cfactor = 0
        for j in range(number_of_agegroups):
            cijt = df_contactrate.iat[j+1,i+1] * sum(N) / (N[i]*N[j])
            cumm_cfactor += cijt * rel_besmh[i]*I[j]
        lambdaa = (beta[i] * rel_vatbh[i] * cumm_cfactor ) / sum(N)
        # lambdaa = (beta[i] * rel_vatbh_rivm [i]  ) / sum(N)

        # h_  =    # I -> H
        # i1_ =    # H-> IC
        # i2_ =    # IC-> H
        # d_  =    # H-> D
        # dic_ =   # IC -> D
        # dhic_ =  # IC -> h -> D
        # r_    =  # recovery rate from hospital (before IC)
        # ric_  =  # recovery rate from hospital (after IC)

        dSdt.append( - lambdaa * S[i] *  I[i] * correction_per_age[i]* rutte_factor - S[i]*alfa[i] )
        dEdt.append (( lambdaa * S[i] *  I[i] * correction_per_age[i]* rutte_factor) - (sigma[i] * E[i]))
        dIdt.append(                                     (sigma[i] * E[i]) - ((gamma[i] + h[i])* I[i]) - dfromi[i]*I[i] )
        dHdt.append(    (h[i]*I[i])  - ((i1[i]+d[i]+r[i]) * H[i] )  )

        # There is a problem with HIC, gives negative values

        #dICdt.append ( i1[i]*H[i] - ((i2[i]+dic[i])* IC[i])                                  )
        #dHICdt.append( i2[i]*IC[i] - ((ric[i]+dhic[i])* IC[i])   )
        dICdt.append ( i1[i]*H[i] - ((dic[i]+dhic[i]+ ric[i])* IC[i])                                  )
        dHICdt.append(0)
        dDIFRdt.append( ( ifr[i]* (sigma[i] * E[i])))
        dDdt.append(  (d[i]*H[i]) + (dic[i]*IC[i]) + (dhic[i]*IC[i]) + dfromi[i]*I[i] )
        dRdt.append(   ( gamma[i] * I[i] )+  ( r[i]*H[i]) + (ric[i] * IC[i]))

        dCdt.append(                                      (sigma[i] * E[i]))
        dHcummdt.append( (h[i]*I[i]) )
        dICcummdt.append( i1[i]*H[i])
    to_return = dSdt+ dEdt + dIdt + dRdt + dCdt + dHdt + dICdt  +dHICdt + dDIFRdt+ dDdt + dHcummdt + dICcummdt

    return to_return


def main():
    global number_of_agegroups, names
    names =     ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    number_of_agegroups = len (names) # number of agegroups

    N =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    # h_rivm  =   [0.00347, 0.000377, 0.000949, 0.00388, 0.00842,0.0165, 0.0251, 0.0494, 0.0463] # I -> H
    # ic_opn = [ 0, 2.8402E-05, 0.000211306, 0.000609427, 0.001481364, 0.003788442, 0.006861962, 0.008609547, 0.00210745]
                # Aantallen van https://www.rivm.nl/coronavirus-covid-19/grafieken geldeeld op 4836661 infecties (cummulatieve  prevalentie gedeeld door 8 )
    ifr_ =      [2.04658E-06, 3.78694E-06, 1.76088E-05, 5.45016E-05, 0.000156108, 0.000558534, 0.002271095, 0.009964733, 0.048248607 ]

    h_  =   [0.0015, 0.0001, 0.0002, 0.0007, 0.0013, 0.0028, 0.0044, 0.0097, 0.0107] # I -> H
    i1_ =   [0.0000, 0.0271, 0.0422, 0.0482, 0.0719, 0.0886, 0.0170, 0.0860, 0.0154] # H-> IC
    i2_ =   [0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0531, 0.0080, 0.0367, 0.0356] # IC-> H
    d_  =   [0.0003, 0.0006, 0.0014, 0.0031, 0.0036, 0.0057, 0.0151, 0.0327, 0.0444] # H-> D
    dic_ =  [0.0071, 0.0071, 0.0071, 0.0071, 0.0071, 0.0090, 0.0463, 0.0225, 0.0234] # IC -> D
    dhic_ = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0010, 0.0040, 0.0120, 0.0290] # IC -> h -> D
    r_    = [0.1263, 0.1260, 0.1254, 0.1238, 0.1234, 0.1215, 0.1131, 0.0976, 0.0872] # recovery rate from hospital (before IC)
    ric_  = [0.0857, 0.0857, 0.0857, 0.0857, 0.0857, 0.0821, 0.0119, 0.0567, 0.0550] # recovery rate from hospital (after IC)

    with st.expander('Parameters',  expanded=False):
        col1, col2, col3, col4  = st.columns(4)
        with col1:
            initial_exposed_ratio = input_parameters("in. exp. ratio", [ 0.01, 0.06,     0.03,    0.01  , 0.0051   , 0.001   , 0.001  , 0.001 , 0.001])
        with col2:
            initial_infected_ratio = input_parameters("in. inf. ratio",  [ 0.01, 0.06,     0.03,    0.01  , 0.0051   , 0.001   , 0.001  , 0.001 , 0.001])
        with col3:
            rel_besmh  =  input_parameters("rel besm",[ 3,  3,  3,  3,  3,  3,  1,  1,  1]) # Relative infectiousness https://www.pnas.org/content/117/36/22430
        with col4:
            rel_vatbh = input_parameters("rel vatbaarh",[1,1.5,2,5,8,10,16,16,16] )     # Relative suspceptibility fig 1b, op het oog https://www.nature.com/articles/s41591-020-0962-9.
                                                                                        # Verdubbeld om alle waards => 1 te krijgen. (anders dooft het uit in die leeftijdsgroep)
        col1x, col2x, col3x, col4x  = st.columns(4)
        with col1x:
            correction_per_age = input_parameters("corr age/vax-eff",[1,1,1,1,1,1,1,1,1] )

    dfromi = []
    for x in range (number_of_agegroups):
        dfromi.append( ifr_[x] - ((h_[x]* d_[x]) + (h_[x]*i1_[x]*dic_[x] )))
    df_parameters = pd.DataFrame(
    {'Agegroup': names,
     'ifr':ifr_,
     'rel_besmh': rel_besmh,
     'rel_vatbaarh': rel_vatbh,
    })
    total_pop = sum(N)
    I0 ,E0 = [], []
                          #  ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    for y in range (number_of_agegroups):
        E0.append (N[y]* initial_exposed_ratio[y])
        I0.append (N[y]* initial_infected_ratio[y])
    R0, S0,  C0 =   [0] * number_of_agegroups, [None] * number_of_agegroups, [0] * number_of_agegroups
    H0, IC0, D0 =   [0] * number_of_agegroups, [0] * number_of_agegroups, [0] * number_of_agegroups
    DIFR0, HIC0, Hcumm0, ICcumm0  =   [0] * number_of_agegroups, [0] * number_of_agegroups,  [0] * number_of_agegroups, [0] * number_of_agegroups
    alfa =  [0] * number_of_agegroups # vaccination rate
    for y in range(number_of_agegroups):
        S0[y] = N[y] - E0[y]- I0[y] - R0[y]

    st.sidebar.subheader("Parameters")

    incubationtime = (st.sidebar.slider('Incubatietijd (1/sigma)', 1, 30, 2))
    beta_ = st.sidebar.number_input(
                            "Contact rate (beta)",
                            min_value=0.0,
                            max_value=1.0,
                            step=1e-4,
                            value = 0.03100,
                            format="%.4f")
    #R_start_ = (st.sidebar.slider('R-naugth', 0.0, 5.0, 1.1))
    infectioustime = (st.sidebar.slider('Average days infectious (1/gamma)', 1, 30, 2))
    sigma = [1/incubationtime]*number_of_agegroups # 1/incubation time - latent period
    beta = [beta_] * number_of_agegroups # contact rate
    gamma = [1/infectioustime] * number_of_agegroups # mean recovery rate (1/recovery days/infectious time)

     # IF YOU WANT TO START FROM AN R-NAUGHT
    # Rstart = [R_start_]*number_of_agegroups
    # beta = []
    # for y in range (number_of_agegroups):
    #     beta.append(Rstart[y]*gamma[y]/(S0[y]/N[y]))

    global rutte_factor
    rutte_factor = st.sidebar.slider('Rutte factor (seasonality, maatregelen (<1), verspoepelingen (>1)', 0.0, 10.0, 1.0)
    what_to_show_options = ["S", "E",  "I", "R", "C", "H", "IC","HIC","DIFR", "D", "Hcumm", "ICcumm"]
    what_to_show_options_default = [ "C"]

    what_to_show = st.sidebar.multiselect(
            "What to show", what_to_show_options, what_to_show_options_default)

    y0 = tuple(S0 + E0 + I0 + R0 + C0 + H0 +IC0+ HIC0 + DIFR0 + D0 + Hcumm0 + ICcumm0)
    parameters = tuple(N + alfa + beta + gamma + sigma  + rel_besmh + rel_vatbh + ifr_ + h_ + i1_ + i2_ + d_ + dic_ + dhic_ + r_ + ric_+ correction_per_age + dfromi)
    n = 176 # number of time points
    t = np.linspace(0, n-1, n) # time points

    result_odeint = odeint(func, y0, t, parameters, tfirst=True)

    st.subheader("Totals")
    show_result(result_odeint, N)
    #  draw_graph_with_all_groups(result_odeint, names, beta, gamma, t,N)
    plot_total(result_odeint, 1, what_to_show)

    with st.expander('Per leeftijdsgroep',  expanded=False):
        st.subheader("Per age group")
        for name in names:
            plot_single_age_group(name, result_odeint, names,  t, N, what_to_show)

    #plot_total_as_ratio(result_odeint, total_pop)

    st.subheader("Contact matrix")
    st.write(get_contact_matrix("2016/-17","all"))

    show_toelichting()



if __name__ == '__main__':
    main()
