import fit_to_data_streamlit
import covid_dashboard_rcsmit
import plot_hosp_ic_streamlit
import SEIR_hobbeland
import grafiek_pos_testen_per_leeftijdscategorie_streamlit
import perprovincieperleeftijd

import streamlit as st

def main():
    lijst = ["fit_to_data_streamlit", "covid_dashboard_rcsmit", "plot_hosp_ic_streamlit", "SEIR_hobbeland",
     "grafiek_pos_testen_per_leeftijdscategorie_streamlit", "perprovincieperleeftijd"]

    what_to_do =  st.sidebar.selectbox(
            "What to dp", lijst,
            index=3,
        )


    if what_to_do == "fit_to_data_streamlit":
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

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
