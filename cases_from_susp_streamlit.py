import streamlit as st
import pandas as pd

def calculate_aantallen(ziek, header, h_, ic_opn, ifr_, long_covid):
    st.subheader ("Aantallen")
    df_calculated = pd.DataFrame()
    for i in range(len(ziek)):

        df_calculated = df_calculated.append(
            {
                "_agegroup": header[i],
                "_ziek": round(ziek[i] ),
                "hosp": round( ziek[i] *h_[i]),
                "ic" :  round(ziek[i] *  ic_opn[i]),
                "overl" : round( ziek[i] *  ifr_[i]),
                "long covid":  round(ziek[i]  * long_covid[i])
            },
            ignore_index=True,
        )
    df_calculated.loc['Totaal']= df_calculated.sum()
    df_calculated.loc['Totaal', "_agegroup"] = ""
    df_calculated=df_calculated.astype(str)
    st.write (df_calculated)

def main():

    st.title( "Wat gebeurt er als je alles open gooit")
    st.write ("Dit is een simpel scriptje wat een bierviltjes berekening maakt om te zien wat er gebeurt als je alles open gooit zonder maatregelen." )
    st.write ("De belangrijkste parameter is de attackrate en de reeds opgebouwde immuniteit door doorgemaakte ziekte, vaccinatie of kruisimmuniteit")

    pop_ =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    fraction = [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]
 #           0-9     10-19  20-29  30-39    40-49   50-59   60-69   70-79  80+

    # WAARDES VAN RIVM
    h_rivm  =   [0.00347, 0.000377, 0.000949, 0.00388, 0.00842,0.0165, 0.0251, 0.0494, 0.0463] # I -> H
    # SOURCE : RIVM https://www.rivm.nl/sites/default/files/2021-03/Modellingresults%20COVID19%20vaccination%20version1.0%2020210324_0.pdf

    # ZIEKENHUISOPNAMES BEREKEND DOOR RENE
    # a. totaal aantal infected berekend mbp cummulatieve prevalentie geldeeld door 8
    # b. attack berekend: a/17.4mln
    # c. populatiefractie per leeftijd * b = aantal infected per leeftijdsgroep
    # d. De waardes van  "COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep" / c

    h_rene =       [0.000672671, 0.000661464, 0.001692048, 0.003937327, 0.007792106, 0.016433039, 0.024554007, 0.039117918, 0.055119236]

    # Q2-2021. Alle stappen voor deze maanden:
    # a. totaal aantal cases per leeftijdsgroep uit casus_landelijk
    # b. totaalziekenhuisopnames per leeftijdsgroep uit _ziekenhuis_ic_opnames_per_leeftijdsgroep.csv
    # c. b/a
    # d. totaal aantal infected  mbp cummulatieve prevalentie geldeeld door 8 (onderrapportagefactor)
    # e. c / d


    h_rene_q2_2021 = [0.000798129, 0.000798129,0.001296388,0.00508778,0.009016154,0.020276935,0.042398226,0.084732437,0.121956144]

    ic_opn = [ 0, 2.8402E-05, 0.000211306, 0.000609427, 0.001481364, 0.003788442, 0.006861962, 0.008609547, 0.00210745]


    # https://twitter.com/YorickB/status/1412453783754481666
    h_yorick = [0.0014, 0.0014, 0.0020, 0.0064, 0.0098, 0.0259, 0.0380, 0.100027, 0.1350]

    st.sidebar.subheader ( "Long covid & hospital rates")
    long_ = st.sidebar.number_input("Fraction of the suspectables that gets long covid (equal for all ages)", 0.0, 1.0,0.1)
    which_hospital = st.sidebar.selectbox("which hospital rates ", ["yorick", "rene", "rene_q2_2021" , "rivm"], index=1)
    if which_hospital == "yorick": h_ = h_yorick
    if which_hospital == "rene": h_ = h_rene
    if which_hospital == "rivm": h_ = h_rivm
    if which_hospital == "rene_q2_2021": h_ = h_rene_q2_2021

    header = [ "0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    one = [ 1.0,   1.0   ,    1.0      , 1.0      , 1.0    , 1.0      , 1.0    , 1.0 ,     1.0]
    long_covid = [long_,    long_,      long_,      long_,     long_,      long_,     long_,      long_,     long_]

    # Aantallen van https://www.rivm.nl/coronavirus-covid-19/grafieken geldeeld op 4836661 infecties (cummulatieve  prevalentie gedeeld door 8 )
    ifr_ =[2.04658E-06, 3.78694E-06, 1.76088E-05, 5.45016E-05, 0.000156108, 0.000558534, 0.002271095, 0.009964733, 0.048248607 ]

    st.sidebar.subheader("Method")
    method =  st.sidebar.selectbox("Method ", ["Suspectables_attackrate", "cases","ratios"], index=1)


    if method == "Suspectables_attackrate":

        # Values from Sanquin bloedonderzoek
        # https://twitter.com/mr_Smith_Econ/status/1413158835817259013/photo/1

        st.sidebar.subheader("Fractie immuun per agegroup")
        s0 = st.sidebar.number_input(header[0], 0.0, 1.0,0.4)
        s1 = st.sidebar.number_input(header[1], 0.0, 1.0,0.4)
        s2 = st.sidebar.number_input(header[2], 0.0, 1.0,0.4)
        s3 = st.sidebar.number_input(header[3], 0.0, 1.0,0.4)
        s4 = st.sidebar.number_input(header[4], 0.0, 1.0,0.55)
        s5 = st.sidebar.number_input(header[5], 0.0, 1.0,0.8)
        s6 = st.sidebar.number_input(header[6], 0.0, 1.0,0.95)
        s7 = st.sidebar.number_input(header[7], 0.0, 1.0,0.98)
        s8 = st.sidebar.number_input(header[8], 0.0, 1.0,0.98)
        suspectible = [1-s0,1-s1,1-s2,1-s3,1-s4,1-s5,1-s6,1-s7,1-s8]
        suspect_nr = [round(a * b) for a, b in zip(pop_, suspectible)]
        st.sidebar.subheader("Attackrate per agegroup")
        a0 = st.sidebar.number_input(header[0], 0.0,  100.0, 20.0)
        a1 = st.sidebar.number_input(header[1], 0.0,  100.0, 20.0)
        a2 = st.sidebar.number_input(header[2], 0.0,  100.0, 20.0)
        a3 = st.sidebar.number_input(header[3], 0.0,  100.0, 20.0)
        a4 = st.sidebar.number_input(header[4], 0.0,  100.0, 20.0)
        a5 = st.sidebar.number_input(header[5], 0.0,  100.0, 20.0)
        a6 = st.sidebar.number_input(header[6], 0.0,  100.0, 20.0)
        a7 = st.sidebar.number_input(header[7], 0.0,  100.0, 20.0)
        a8 = st.sidebar.number_input(header[8], 0.0,  100.0, 20.0)
        attack =  [a0/100,a1/100,a2/100,a3/100,a4/100,a5/100,a6/100,a7/100,a8/100]

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
        df_calculated=df_calculated.astype(str).copy(deep = True)
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

    if method == "ratios":
        number_of_cases = st.sidebar.number_input("Number of cases", 0, 5_000_000,5_000)
        st.sidebar.subheader ("Ratio of each agegroup")
        #   0-9     10-19  20-29  30-39    40-49   50-59   60-69   70-79  80+
        r0 = st.sidebar.number_input(header[0],  0.0, 100.0, 1.0)
        r1 = st.sidebar.number_input(header[1],  0.0, 100.0, 26.0)
        r2 = st.sidebar.number_input(header[2],  0.0, 100.0,59.0)
        r3 = st.sidebar.number_input(header[3],  0.0, 100.0,8.0)
        r4 = st.sidebar.number_input(header[4],  0.0, 100.0,3.0)
        r5 = st.sidebar.number_input(header[5],  0.0, 100.0,2.5)
        r6 = st.sidebar.number_input(header[6],  0.0, 100.0,0.2)
        r7 = st.sidebar.number_input(header[7],  0.0, 100.0,0.2)
        r8 = st.sidebar.number_input(header[8],  0.0, 100.0, 0.1)
        totaal = r0+r1+r2+r3+r4+r5+r6+r7+r8
        st.sidebar.write(f"Totaal {totaal}")

        ziek = [r0*number_of_cases/100,
        r1*number_of_cases/100,
        r2*number_of_cases/100,
        r3*number_of_cases/100,
        r4*number_of_cases/100,
        r5*number_of_cases/100,
        r6*number_of_cases/100,
        r7*number_of_cases/100,
        r8*number_of_cases/100       ]

        calculate_aantallen(ziek, header, h_, ic_opn, ifr_, long_covid)

    elif method == "cases":
        st.sidebar.subheader ("Number of cases per agegroup")
        z0 = st.sidebar.number_input(header[0],  0, 10000, 33)
        z1 = st.sidebar.number_input(header[1],  0, 10000, 1068)
        z2 = st.sidebar.number_input(header[2],  0, 10000, 2380)
        z3 = st.sidebar.number_input(header[3],  0, 10000, 316)
        z4 = st.sidebar.number_input(header[4],  0, 10000, 124)
        z5 = st.sidebar.number_input(header[5],  0, 10000, 105)
        z6 = st.sidebar.number_input(header[6],  0, 10000, 36)
        z7 = st.sidebar.number_input(header[7],  0, 10000, 8)
        z8 = st.sidebar.number_input(header[8],  0, 10000, 9)
        totaal_ziek = z0 +z1 +z2 +z3 +z4 +z5 +z6 +z7 +z8
        st.sidebar.write(f"Totaal {totaal_ziek}")
        ziek = [z0,z1,z2,z3,z4,z5,z6,z7,z8]
        totaal = 100

        if totaal == 100:
            calculate_aantallen(ziek, header, h_, ic_opn,ifr_, long_covid)

        else:
            st.error(f"Zorg ervoor dat de fracties in totaal 100 zijn in plaats van {totaal}")

    st.subheader ("Kansenmatrix")
    kansenmatrix = pd.DataFrame(
    {'age': header,
     'ziekenhuisopn yorick': h_yorick,
     'ziekenhuisopn rene': h_rene,
     'ziekenhuisopn rivm': h_rivm,
     'IC opname': ic_opn,
     'IFR':ifr_
    })

    st.write ("Ziekenhuisopname kans is berekend door Yorick Bleijenberg adhv Sanquin, mijzelf (zie broncode) en RIVM data.")
    st.write(" De categorieen 0-9 en 10-19 zijn gelijk getrokken 80-89 en 90+ zijn gemiddeld")

    st.write (kansenmatrix)
    st.write ("Ter vergelijk: Normaal overlijden er per jaar ca. 5000 mensen  onder de 50 en 20.000 onder de 65. Er zijn ca 40.000 ziekenhuisbedden in NL en 2000 IC bedden. ")
    st.write ("Het aantal acceptabele ziekenhuis- en IC opnames, overlijdens en long covid gevallen is een morele en politieke keuze")


if __name__ == "__main__":
    main()
