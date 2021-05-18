import streamlit as st


def main():
    st.header ("Welcome!")
    toelichting = (
        "<p>Her you the scripts I made in the last months regarding to COVID-19 in the Netherlands.</p>"
        "<br><i>covid_dashboard_rcsmit</i> - - aggregates a lot of information and statistics from the Netherlands. Shows correlations and graphs."
        "<br><i>plot_hosp_ic_streamlit",
        "<br><i>calculate_false_positive_rate_covid_test_streamlit</i> -  HOE BETROUWBAAR ZIJN DE TESTEN ?"
        "<br><i>number_of_cases_interactive</i> -  - Plotting the number of COVID cases with different values. Contains a SIR-graph and a classical SIR-model. Including immunity"
        "<br><i>calculate_ifr_from_prevalence_streamlit</i> -  calculate percentage of population who had covid and the IFR from the prevalence"
        "<br><i>fit_to_data_streamlit</i> -  FIT THE DATA
        "<br><i>SEIR_hobbeland</i> - - Make an interactive version of the SEIR model, inspired by Hobbeland - https://twitter.com/MinaCoen/status/1362910764739231745"

        "<br><i>grafiek_pos_testen_per_leeftijdscategorie_streamlit</i> -  draw graphs of positieve cases per age in time. DATA NOT UPDATED"
        "<br><i>perprovincieperleeftijd</i> - Zijn kinderen de redenen dat het percentage positief daalt? DATA NOT UPDATED"
    )
    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        'With support of : @hk_nien, @mr_Smith_Econ, @dimgrr, @JosetteSchoenma e.a'
    )
    st.markdown(toelichting, unsafe_allow_html=True)
    st.markdown(tekst, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
