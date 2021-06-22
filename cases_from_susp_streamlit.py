# FORMULAS AND VALUES TO MAKE A SIR MODEL FOR THE NETHERLANDS
# SOURCE : RIVM https://www.rivm.nl/sites/default/files/2021-03/Modellingresults%20COVID19%20vaccination%20version1.0%2020210324_0.pdf
# retreived 21st of April 2021
# Copied by René SMIT
# Fouten voorbehouden

import streamlit as st
import pandas as pd

def main():

    st.title( "Wat gebeurt er als je alles open gooit")
    st.write ("Dit is een simpel scriptje wat een bierviltjes berekening maakt om te zien wat er gebeurt als je alles open gooit zonder maatregelen." )
    st.write ("De belangrijkste parameter is de attackrate en de reeds opgebouwde immuniteit door doorgemaakte ziekte, vaccinatie of kruisimmuniteit")

    pop_ =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    fraction = [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]
 #           0-9     10-19  20-29  30-39    40-49   50-59   60-69   70-79  80+

    #x_tot = 0
    #for i in range (0, len(pop_)-1):

    #st.write (f"{wat} - {round(x_tot)}")
    st.sidebar.subheader ( "Attack rate & Long covid")
    attack_ = st.sidebar.number_input("Fraction of the suspectables that gets sick (equal for all ages)", 0.0, 1.0,0.2)
    long_ = st.sidebar.number_input("Fraction of the suspectables that gets long covid (equal for all ages)", 0.0, 1.0,0.1)
    header = [ "0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    one = [ 1.0,   1.0   ,    1.0      , 1.0      , 1.0    , 1.0      , 1.0    , 1.0 ,     1.0]

    attack = [ attack_,   attack_   ,    attack_      , attack_      , attack_    , attack_      , attack_    , attack_ ,     attack_]
    long_covid = [long_,    long_,      long_,      long_,     long_,      long_,     long_,      long_,     long_]
    # IFR van een Nature artikel
    #ifr_ =      [ 0.000002, 0.00002 , 0.000095 , 0.00032 , 0.00098  , 0.00265 , 0.007655 , 0.04877 , 0.08292 ]

    # Aantallen van https://www.rivm.nl/coronavirus-covid-19/grafieken geldeeld op 4836661 infecties (cummulatieve  prevalentie gedeeld door 8 )
    ifr_ =[2.04658E-06, 3.78694E-06, 1.76088E-05, 5.45016E-05, 0.000156108, 0.000558534, 0.002271095, 0.009964733, 0.048248607 ]

    # WAARDES VAN RIVM - levert te lage cijfers op.
    #h_  =   [0.0015, 0.0001, 0.0002, 0.0007, 0.0013, 0.0028, 0.0044, 0.0097, 0.0107] # I -> H
    #i1_ =   [0.0000, 0.0271, 0.0422, 0.0482, 0.0719, 0.0886, 0.0170, 0.0860, 0.0154] # H-> IC
    #ic_opn = [a * b for a, b in zip(h_, i1_)]

    # a. Totaal aantal gevallen per leeftijdsgrope berekend van  "cases_landelijk.csv",
    # b. Dit vermenidgvuldigd met een factor (4.8/1.7) om van cases- naar infected aantallen te komen
    # c. De waardes van  "COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep" gedeeld op [b]
    # DIT LEVERT TE HOGE CIJFERS OP


    h_ = [0.001935772, 0.000535596, 0.00123012, 0.003500636, 0.006540055, 0.014336461, 0.031555617, 0.065269381, 0.060888242]
    ic_opn = [0,2.300E-05,0.00015362,0.000541835,0.001243335,0.003305101,0.00881866,0.014365279,0.002328024]

    pop_ =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]

    # a. totaal aantal infected berekend mbp cummulatieve prevalentie geldeeld door 8
    # b. attack berekend: a/17.4mln
    # c. populatiefractie per leeftijd * b = aantal infected per leeftijdsgroep
    # d. De waardes van  "COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep" / c

    h_ = [0.000672671, 0.000661464, 0.001692048, 0.003937327, 0.007792106, 0.016433039, 0.024554007, 0.039117918, 0.055119236]
    ic_opn = [ 0, 2.8402E-05, 0.000211306, 0.000609427, 0.001481364, 0.003788442, 0.006861962, 0.008609547, 0.00210745]


    fraction = [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]
    st.sidebar.subheader("Fractie immuun")
    #attack = st.sidebar.number_input("Fraction of the suspectables that gets sick", 0.0, 1.0,0.2)
    s0 = st.sidebar.number_input(header[0], 0.0, 1.0,0.3)
    s1 = st.sidebar.number_input(header[1], 0.0, 1.0,0.3)
    s2 = st.sidebar.number_input(header[2], 0.0, 1.0,0.3)
    s3 = st.sidebar.number_input(header[3], 0.0, 1.0,0.4)
    s4 = st.sidebar.number_input(header[4], 0.0, 1.0,0.5)
    s5 = st.sidebar.number_input(header[5], 0.0, 1.0,0.8)
    s6 = st.sidebar.number_input(header[6], 0.0, 1.0,0.9)
    s7 = st.sidebar.number_input(header[7], 0.0, 1.0,0.95)
    s8 = st.sidebar.number_input(header[8], 0.0, 1.0,0.98)
    suspectible = [1-s0,1-s1,1-s2,1-s3,1-s4,1-s5,1-s6,1-s7,1-s8]
    suspect_nr = [round(a * b) for a, b in zip(pop_, suspectible)]

    sick_nr = [a* b for a, b in zip(attack, suspect_nr)]


    st.subheader ("Aantallen")
    df_calculated = pd.DataFrame()
    for i in range(len(pop_)):

        df_calculated = df_calculated.append(
            {
                "_agegroup": header[i],
                "_susp": round(pop_[i] * suspectible[i]),
                "_ziek": round(pop_[i] * suspectible[i] * attack [i]),
                "hosp": round( pop_[i] * suspectible[i] * attack [i] *h_[i]),
                "ic" :  round(pop_[i] * suspectible[i] * attack [i] * ic_opn[i]),
                "overl" : round( pop_[i] * suspectible[i] * attack [i] * ifr_[i]),
                "long covid":  round(pop_[i] * suspectible[i] * attack [i] * long_covid[i])
            },
            ignore_index=True,
        )
    df_calculated.loc['Totaal']= df_calculated.sum()
    df_calculated.loc['Totaal', "_agegroup"] = ""
    st.write (df_calculated)

    st.subheader ("suspectibles")
    susp  = pd.DataFrame(
        {'age': header,
        'groepsgrootte': pop_,
        'suspect. fractie': suspectible,
        'suspect aantal': suspect_nr,
        'attack rate': attack,
        'sick aantal': sick_nr
        })

    st.write (susp)
    st.subheader ("Kansenmatrix")
    kansenmatrix = pd.DataFrame(
    {'age': header,
     'ziekenhuisopn': h_,
     'IC opname': ic_opn,
     'IFR':ifr_
    })
    st.write("Deze cijfers zijn berekend aan de hand van ziekenhuisopnames, IC opnames en overlijdens per leeftijd waarbij aangenomen is dat de verschillende leeftijdsgroepen een gelijke attackrate ondervonden. (#) De waardes zijn uitgedrukt in fracties (niet in %)")
    st.write ("Gebruik is gemaakt van de volgende bestanden op https://data.rivm.nl/covid-19/:")
    st.write("* COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv")
    st.write("* COVID-19_casus_landelijk")
    st.write("* COVID-19_prevalentie.json (cummulatieve waarde gedeeld door 8)")
    st.write ("#) Gebruik maken van de totalen per leeftijdsgroep in cases_landelijk is geprobeerd maar geeft so wie so vertekening ondermeer doordat veel kinderen niet werden getest tot januari 2021")


    st.write (kansenmatrix)
    st.write ("Ter vergelijk: Normaal overlijden er per jaar ca. 5000 mensen  onder de 50 en 20.000 onder de 65. Er zijn ca 40.000 ziekenhuisbedden in NL en 2000 IC bedden. ")
    st.write ("Het aantal acceptabele ziekenhuis- en IC opnames, overlijdens en long covid gevallen is een morele en politieke keuze")


if __name__ == "__main__":
    main()
