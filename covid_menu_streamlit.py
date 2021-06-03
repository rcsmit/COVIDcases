import streamlit as st
#from multiapp import MultiApp
#import welcome, fit_to_data_streamlit,covid_dashboard_rcsmit, plot_hosp_ic_streamlit, SEIR_hobbeland
#import grafiek_pos_testen_per_leeftijdscategorie_streamlit, perprovincieperleeftijd, number_of_cases_interactive
#import calculate_false_positive_rate_covid_test_streamlit


def main_new():
    app = MultiApp()

    # Add all your application here
    app.add_app("Home",welcome.main())
    app.add_app("Fit to data",fit_to_data_streamlit.main())
    app.add_app("Covid dashboard",covid_dashboard_rcsmit.main())
    app.add_app("Hosp/IC opnames",plot_hosp_ic_streamlit.main())
    app.add_app("SEIR Hobbeland",SEIR_hobbeland.main())
    app.add_app("Pos testen per leeftijdscat",grafiek_pos_testen_per_leeftijdscategorie_streamlit.main())
    app.add_app("Per provincie per leeftijd",perprovincieperleeftijd.main())
    app.add_app("Number of cases",number_of_cases_interactive.main())
    app.add_app("False positive calculator",calculate_false_positive_rate_covid_test_streamlit.main())

    # The main app
    app.run()


def main_oud():
    query_params = st.experimental_get_query_params()

        # Query parameters are returned as a list to support multiselect.
        # Get the first item in the list if the query parameter exists.
    keuze = int(query_params["keuze"][0]) if "keuze" in query_params else 0
    lijst = [
        "0. welcome",
        "1. covid dashboard",
        "2. plot hosp ic",
        "3. false positive rate covid test",
        "4. number of cases interactive",
        "5. ifr from prevalence",
        "6. fit to data",
        "7. SEIR hobbeland",
        "8. show contactmatrix",
        "9. r getal per provincie",
        "10. grafiek pos testen per leeftijdscat",
        "11. per provincie per leeftijd",
    ]


    with st.sidebar.beta_expander('MENU: Choose a script',  expanded=True):
        menu_keuze = st.radio("",lijst, index=keuze)

    st.sidebar.markdown("<h1>- - - - - - - - - - - - - - - - - - </h1>", unsafe_allow_html=True)
    if menu_keuze == lijst[0] or keuze == 0:
        import welcome
        welcome.main()

    elif menu_keuze == lijst[1] or keuze == 1:
        import covid_dashboard_rcsmit
        covid_dashboard_rcsmit.main()

    elif menu_keuze == lijst[2] or keuze == 2:
        import plot_hosp_ic_streamlit
        plot_hosp_ic_streamlit.main()

    elif menu_keuze == lijst[3] or keuze == 3:
        import calculate_false_positive_rate_covid_test_streamlit
        calculate_false_positive_rate_covid_test_streamlit.main()


    elif menu_keuze ==lijst[4] or keuze == 4:
        import number_of_cases_interactive
        number_of_cases_interactive.main()

    elif menu_keuze ==lijst[5] or keuze == 5:
        import calculate_ifr_from_prevalence_streamlit
        calculate_ifr_from_prevalence_streamlit.main()

    elif menu_keuze ==lijst[6] or keuze == 6:
        import fit_to_data_streamlit
        fit_to_data_streamlit.main()

    elif menu_keuze == lijst[7] or keuze == 7:
        import SEIR_hobbeland
        SEIR_hobbeland.main()

    elif menu_keuze == lijst[8] or keuze == 8:
        import show_contactmatrix
        show_contactmatrix.main()
    elif menu_keuze  == lijst[9] or keuze == 9:
        import r_getal_per_provincie
        r_getal_per_provincie.main()
    elif menu_keuze == lijst[10] or keuze == 10:
        import grafiek_pos_testen_per_leeftijdscategorie_streamlit
        grafiek_pos_testen_per_leeftijdscategorie_streamlit.main()

    elif menu_keuze == lijst[11] or keuze == 11:
        import perprovincieperleeftijd
        perprovincieperleeftijd.main_per_province_per_leeftijd()


if __name__ == "__main__":
    main_oud()
