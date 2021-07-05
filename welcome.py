import streamlit as st


def main():
    st.header ("Welcome!")
    toelichting = (
        "<p>Here you'll find the scripts I made in the last months regarding to COVID-19 in the Netherlands.</p>"
        "<br><br><i>1. covid_dashboard_rcsmit</i> - aggregates a lot of information and statistics from the Netherlands. Shows correlations and graphs."
        "<br><img src='https://user-images.githubusercontent.com/1609141/112730553-8b1cf680-8f32-11eb-83f6-1569f5114678.png' width=400>"
        "<br><br><i>2. plot_hosp_ic_streamlit</i> - Plot the number of hospital or ICU admissions per agegroup"
        "<br><img src='https://user-images.githubusercontent.com/1609141/118802804-e02a1880-b8a2-11eb-8772-cc495bf7bca8.png' width=400>"
        "<br><br><i>3. calculate_false_positive_rate_covid_test_streamlit</i> -  HOE BETROUWBAAR ZIJN DE TESTEN ?"
        "<br><img src='https://user-images.githubusercontent.com/1609141/115085050-14a85e80-9f0a-11eb-9732-87a78ffa73d3.png' width=400>"
        "<br><br><i>4. number_of_cases_interactive</i> - Plotting the number of COVID cases with different values. Contains a SIR-graph and a classical SIR-model. Including immunity"
        "<br><img src='https://user-images.githubusercontent.com/1609141/112731094-945b9280-8f35-11eb-8c3d-a99e5f48487d.png' width=400>"
        "<br><br><i>5. calculate_ifr_from_prevalence_streamlit</i> - calculate percentage of population who had covid and the IFR from the prevalence"
        "<br><img src='https://user-images.githubusercontent.com/1609141/115160069-8f05e980-a096-11eb-87f4-106738c6feed.png' width=400>"
        "<br><br><i>6. fit_to_data_streamlit</i> - Fit the various curves to a derivate formula. Make an animation to see how the maximum predicted value changes in time"
        "<br><img src='https://user-images.githubusercontent.com/1609141/115085210-651fbc00-9f0a-11eb-99e6-6aa4504fd325.png' width=400>"
        "<br><br><i>7. SEIR_hobbeland</i> - Make an interactive version of the SEIR model, inspired by Hobbeland - https://twitter.com/MinaCoen/status/1362910764739231745"
        "<br><img src='https://user-images.githubusercontent.com/1609141/112730583-adaf0f80-8f32-11eb-9517-0b2fd6443c42.png' width=400>"

        "<br><br><i>8. show contactmatrix</i> - Laat de contactmatrix voor en na covid (1e golf) zien en laat zien welke cellen het meest zijn veranderd "
        "<br><br><i>9. r getal per provincie</i> - Berekent het R getal per provincie vanuit casus_landelijk (duurt lang om te laden!) "
        "<br><br><i>10. Cases from suspectibles</i> - Een simpel scriptje wat een bierviltjes berekening maakt om te zien wat er gebeurt als je alles open gooit zonder maatregelen. "
        "<br><br><i>11. Fit to data OWID</i> - Fitting data from Our World in Data to formulas "
        "<br><br><i>12. Calculate R per country owid</i> - Fitting data from Our World in Data to formula and calculate R-number for each country "
        "<br><br><i>13. Covid dashboard OWID/Google or Waze</i> - Show the data from Our World in Data. Link it to the Google and Waze-info. Calcuate which one has a bigger correlation with the R-number "
        "<br><br><i>14. Dag verschillen per leeftijd</i> - Calculate the differences of cases per age between a date-frame "
        "<br><br><i>15. Calculate spec./abs. humidity from rel. hum</i> - Calculate specific and absolute humidity from relative humidity and temperature "
        "<br><br><i>16. R getal per leeftijdscategorie</i> -  - Berekent het R getal per leeftijdscategorie vanuit casus_landelijk (duurt lang om te laden!) "
        "<br><br><i>17.grafiek_pos_testen_per_leeftijdscategorie_streamlit</i> -  draw graphs of positieve cases per age in time. DATA NOT UPDATED"
        "<br><img src='https://user-images.githubusercontent.com/1609141/112730260-e0f09f00-8f30-11eb-9bff-a835c2f965f7.png' width=400>"
        "<br><br><i>18.perprovincieperleeftijd</i> - Zijn kinderen de redenen dat het percentage positief daalt? DATA NOT UPDATED"



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
