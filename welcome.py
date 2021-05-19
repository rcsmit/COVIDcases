import streamlit as st
from PIL import Image
image = Image.open('sunrise.jpg')
st.image(image, caption='')

def main():
    st.header ("Welcome!")
    toelichting = (
        "<p>Here you'll find the scripts I made in the last months regarding to COVID-19 in the Netherlands.</p>"
        "<br><br><i>covid_dashboard_rcsmit</i> - aggregates a lot of information and statistics from the Netherlands. Shows correlations and graphs."
        image = Image.open('sunrise.jpg')
        st.image(image, caption='')
        "<br><br><i>plot_hosp_ic_streamlit</i> - Plot the number of hospital or ICU admissions per agegroup"
        image = Image.open('sunrise.jpg')
        st.image(image, caption='')
        "<br><br><i>calculate_false_positive_rate_covid_test_streamlit</i> -  HOE BETROUWBAAR ZIJN DE TESTEN ?"
        image = Image.open('sunrise.jpg')
        st.image(image, caption='')
        "<br><br><i>number_of_cases_interactive</i> - Plotting the number of COVID cases with different values. Contains a SIR-graph and a classical SIR-model. Including immunity"
        image = Image.open('https://user-images.githubusercontent.com/1609141/112731094-945b9280-8f35-11eb-8c3d-a99e5f48487d.png')
        st.image(image, caption='')
        "<br><br><i>calculate_ifr_from_prevalence_streamlit</i> - calculate percentage of population who had covid and the IFR from the prevalence"
        
        "<br><br><i>fit_to_data_streamlit</i> - Fit the various curves to a derivate formula. Make an animation to see how the maximum predicted value changes in time"
        "<br><br><i>SEIR_hobbeland</i> - Make an interactive version of the SEIR model, inspired by Hobbeland - https://twitter.com/MinaCoen/status/1362910764739231745"
        image = Image.open('sunrise.jpg')
        st.image(image, caption='')
        "<br><br><i>grafiek_pos_testen_per_leeftijdscategorie_streamlit</i> -  draw graphs of positieve cases per age in time. DATA NOT UPDATED"
        "<br><br><i>perprovincieperleeftijd</i> - Zijn kinderen de redenen dat het percentage positief daalt? DATA NOT UPDATED"
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
