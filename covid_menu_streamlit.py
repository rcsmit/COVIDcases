import streamlit as st


def main():
    lijst = [
        "welcome",
        "fit_to_data",
        "covid_dashboard_rcsmit",
        "plot_hosp_ic",
        "calculate_false_positive_rate_covid_test",
        "number_of_cases_interactive",
        "calculate_ifr_from_prevalence",
        "SEIR_hobbeland",
        "grafiek_pos_testen_per_leeftijdscategorie",
        "perprovincieperleeftijd",

    ]

    menu_keuze = st.sidebar.selectbox(
        "Choose a script",
        lijst,
        index=0,
    )

    if menu_keuze == "welcome":
        import welcome

        welcome.main()

    elif menu_keuze == "fit_to_data":
        import fit_to_data_streamlit

        fit_to_data_streamlit.main()

    elif menu_keuze == "covid_dashboard_rcsmit":
        import covid_dashboard_rcsmit

        covid_dashboard_rcsmit.main()
    elif menu_keuze == "plot_hosp_ic":
        import plot_hosp_ic_streamlit

        plot_hosp_ic_streamlit.main()
    elif menu_keuze == "SEIR_hobbeland":
        import SEIR_hobbeland

        SEIR_hobbeland.main()
    elif menu_keuze == "grafiek_pos_testen_per_leeftijdscategorie":
        import grafiek_pos_testen_per_leeftijdscategorie_streamlit

        grafiek_pos_testen_per_leeftijdscategorie_streamlit.main()
    elif menu_keuze == "perprovincieperleeftijd":
        import perprovincieperleeftijd

        perprovincieperleeftijd.main_per_province_per_leeftijd()
    elif menu_keuze == "number_of_cases_interactive":
        import number_of_cases_interactive

        number_of_cases_interactive.main()
    elif menu_keuze == "calculate_ifr_from_prevalence":
        import calculate_ifr_from_prevalence_streamlit

        calculate_ifr_from_prevalence_streamlit.main()
    elif menu_keuze == "calculate_false_positive_rate_covid_test":
        import calculate_false_positive_rate_covid_test_streamlit

        calculate_false_positive_rate_covid_test_streamlit.main()


if __name__ == "__main__":
    main()
