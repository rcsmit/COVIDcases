import fit_to_data_streamlit
import covid_dashboard_rcsmit
import plot_hosp_ic_streamlit
import number_of_cases_interactive
import SEIR_hobbeland
import grafiek_pos_testen_per_leeftijdscategorie_streamlit
import perprovincieperleeftijd
import welcome
import calculate_ifr_from_prevalence_streamlit


import streamlit as st

def main():
    lijst = ["welcome", "fit_to_data_streamlit", "covid_dashboard_rcsmit", "plot_hosp_ic_streamlit", "number_of_cases_interactive", "SEIR_hobbeland",
     "grafiek_pos_testen_per_leeftijdscategorie_streamlit", "perprovincieperleeftijd", "calculate_ifr_from_prevalence_streamlit", "calculate_false_positive_rate_covid_test_streamlit"]

    what_to_do =  st.sidebar.selectbox(
            "What to dp", lijst,
            index=0,
        )

    if if what_to_do == "welcome":
        welcome.main()

    elif what_to_do == "fit_to_data_streamlit":
        fit_to_data_streamlit.main()
    elif  what_to_do == "covid_dashboard_rcsmit" :
        covid_dashboard_rcsmit.main()
    elif  what_to_do == "plot_hosp_ic_streamlit":
        plot_hosp_ic_streamlit.main()
    elif what_to_do == "SEIR_hobbeland":
        SEIR_hobbeland.main()
    elif what_to_do == "grafiek_pos_testen_per_leeftijdscategorie_streamlit":
        grafiek_pos_testen_per_leeftijdscategorie_streamlit.main()
    elif what_to_do == "perprovincieperleeftijd":
        perprovincieperleeftijd.main_per_province_per_leeftijd()
    elif what_to_do == "number_of_cases_interactive":
        number_of_cases_interactive.main()
    elif what_to_do == "calculate_ifr_from_prevalence_streamlit":
        calculate_ifr_from_prevalence_streamlit.main()
    elif what_to_do == "calculate_false_positive_rate_covid_test_streamlit":
        calculate_false_positive_rate_covid_test_streamlit.main()

if __name__ == "__main__":
    main()
