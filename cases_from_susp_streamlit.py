# FORMULAS AND VALUES TO MAKE A SIR MODEL FOR THE NETHERLANDS
# SOURCE : RIVM https://www.rivm.nl/sites/default/files/2021-03/Modellingresults%20COVID19%20vaccination%20version1.0%2020210324_0.pdf
# retreived 21st of April 2021
# Copied by René SMIT
# Fouten voorbehouden

import streamlit as st
import pandas as pd


def calculate(wat, xxx,  attack):
    #               0-9     10-19   20-29  30-39      40-49   50-59   60-69   70-79  80+
    pop_ =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    fraction = [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]
    suspectible = [0.7,   0.7    ,   0.7 ,    0.7    , 0.7    , 0.5    , 0.1,    0.05,   0.02]

#           0-9     10-19  20-29  30-39    40-49   50-59   60-69   70-79  80+

    x_tot = 0
    for i in range (0, len(pop_)-1):
        x_tot += attack * pop_[i] * xxx[i] * suspectible[i]
    st.write (f"{wat} - {round(x_tot)}")

def main():
    header = [ "0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    besmet = [ 1,   1   ,    1      , 1      , 1    , 1      , 1    , 1 ,     1]
    h_  =   [0.0015, 0.0001, 0.0002, 0.0007, 0.0013, 0.0028, 0.0044, 0.0097, 0.0107] # I -> H
    i1_ =   [0.0000, 0.0271, 0.0422, 0.0482, 0.0719, 0.0886, 0.0170, 0.0860, 0.0154] # H-> IC
    ifr_ =      [ 0.000002, 0.00002 , 0.000095 , 0.00032 , 0.00098  , 0.00265 , 0.007655 , 0.04877 , 0.08292 ]
    long_covid = [0.05,    0.05,      0.05,      0.05,     0.05,      0.05,     0.05,      0.05,     0.05, ]
    ic_opn = [a * b for a, b in zip(h_, i1_)]
    pop_ =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    fraction = [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]
    suspectible = [0.7,   0.7    ,   0.7 ,    0.7    , 0.7    , 0.5    , 0.1,    0.05,   0.02]
    attack = st.sidebar.number_input("Fraction of the suspectables that gets sick", 0.0, 1.0,0.2)

    suspect_nr = [a * b for a, b in zip(pop_, suspectible)]
    sick_nr = [attack * b for b in suspect_nr]
    calculate("aantal besmettingen", besmet, attack)

    calculate("ziekenhuisopnames", h_, attack)
    calculate("IC opnames", ic_opn, attack)
    calculate ("Long COVID 5%", long_covid, attack)
    calculate("Overlijdens", ifr_, attack)
    st.subheader ("suspectibles")
    susp  = pd.DataFrame(
        {'age': header,
        'groepsgrootte': pop_,
        'suspect. fractie': suspectible,
        'suspect aantal': suspect_nr,
        'sick aantal': sick_nr
        })

    st.write (susp)
    st.subheader ("kansenmatrix")
    kansenmatrix = pd.DataFrame(
    {'age': header,
     'ziekenhuisopn': h_,
     'IC opname': ic_opn,
     'IFR':ifr_
    })
    st.write (kansenmatrix)
    st.write ("BRON:  https://www.rivm.nl/sites/default/files/2021-03/Modellingresults%20COVID19%20vaccination%20version1.0%2020210324_0.pdf, p. 70.\nLeeftijdsgerichte IFR komen van een Lancet onderzoek.\nAangenomen is dat <50 jaar 30% ziek is geweest. Daarboven is men (deels) gevaccinneerd en ziek geweest")

if __name__ == "__main__":
    main()
