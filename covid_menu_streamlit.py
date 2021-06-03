import streamlit as st
import importlib
def dynamic_import(module):
    return importlib.import_module(module)



def main():
    query_params = st.experimental_get_query_params()
    keuze = int(query_params["keuze"][0]) if "keuze" in query_params else 0

    lijst = [["0. welcome","welcome"],
        ["1. covid dashboard","covid dashboard rcsmit"],
        ["2. plot hosp ic","plot hosp ic"],
        ["3. false positive rate covid test","calculate false positive rate covid test"],
        ["4. number of cases interactive","number of cases interactive"],
        ["5. ifr from prevalence","ifr from prevalence streamlit"],
        ["6. fit to data","fit to data streamlit"],
        ["7. SEIR hobbeland","SEIR hobbeland"],
        ["8. show contactmatrix","show contactmatrix"],
        ["9. r getal per provincie","r getal per provincie"],
        ["10. grafiek pos testen per leeftijdscat","grafiek pos testen per leeftijdscat",],
        ["11. per provincie per leeftijd","perprovincieperleeftijd"]
        ]
    menukeuzelijst = []
    for l in lijst:
        menukeuzelijst.append[l[0]]

    with st.sidebar.beta_expander('MENU: Choose a script',  expanded=True):
        menu_keuze = st.radio("",menukeuzelijst, index=keuze)

    st.sidebar.markdown("<h1>- - - - - - - - - - - - - - - - - - </h1>", unsafe_allow_html=True)
    st.experimental_set_query_params(keuze=lijst.index(menu_keuze))
    for n, l in enumerate(lijst):
        if menu_keuze == lijst[n][0]:
            module = dynamic_import(lijst[n][1])
            module.main()

    # if menu_keuze == lijst[0] or keuze == 0:
    #     import welcome
    #     welcome.main()

    # elif menu_keuze == lijst[1] or keuze == 1:
    #     import covid_dashboard_rcsmit
    #     covid_dashboard_rcsmit.main()

    # elif menu_keuze == lijst[2] or keuze == 2:
    #     import plot_hosp_ic_streamlit
    #     plot_hosp_ic_streamlit.main()

    # elif menu_keuze == lijst[3] or keuze == 3:
    #     import calculate_false_positive_rate_covid_test_streamlit
    #     calculate_false_positive_rate_covid_test_streamlit.main()


    # elif menu_keuze ==lijst[4] or keuze == 4:
    #     import number_of_cases_interactive
    #     number_of_cases_interactive.main()

    # elif menu_keuze ==lijst[5] or keuze == 5:
    #     import calculate_ifr_from_prevalence_streamlit
    #     calculate_ifr_from_prevalence_streamlit.main()

    # elif menu_keuze ==lijst[6] or keuze == 6:
    #     import fit_to_data_streamlit
    #     fit_to_data_streamlit.main()

    # elif menu_keuze == lijst[7] or keuze == 7:
    #     import SEIR_hobbeland
    #     SEIR_hobbeland.main()

    # elif menu_keuze == lijst[8] or keuze == 8:
    #     import show_contactmatrix
    #     show_contactmatrix.main()
    # elif menu_keuze  == lijst[9] or keuze == 9:
    #     import r_getal_per_provincie
    #     r_getal_per_provincie.main()
    # elif menu_keuze == lijst[10] or keuze == 10:
    #     import grafiek_pos_testen_per_leeftijdscategorie_streamlit
    #     grafiek_pos_testen_per_leeftijdscategorie_streamlit.main()

    # elif menu_keuze == lijst[11] or keuze == 11:
    #     import perprovincieperleeftijd
    #     perprovincieperleeftijd.main_per_province_per_leeftijd()

if __name__ == "__main__":
    main()