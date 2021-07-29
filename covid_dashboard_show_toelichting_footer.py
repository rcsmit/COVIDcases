import streamlit as st
def show_toelichting_footer():
    toelichting = (
        "<h2>Toelichting bij de keuzevelden:</h2>"
        "<p>Order may/might have been changed</p>"
        "<i>IC_Bedden_COVID</i> - Aantal bezette bedden met COVID patienten (hospital_intake_rivm)"
        "<br><i>IC_Bedden_Non_COVID</i> - Totaal aantal bezette bedden (hospital_intake_rivm) "
        "<br><i>Kliniek_Bedden</i> - Totaal aantal ziekenhuisbedden (hospital_intake_rivm)"
        "<br><i>IC_Nieuwe_Opnames_COVID</i> - Nieuwe opnames op de IC "
        "<br><br><i>Hospital_admission_hospital_intake_rivm</i> - Nieuwe opnames in de ziekenhuizen hospital_intake_rivm. Vanaf oktober 2020. Verzameld op geaggreerd niveau en gericht op bezetting "
        "<br><i>hospital_intake_rivm</i> - Nieuwe opnames in de ziekenhuizen RIVM door NICE. Is in principe gelijk aan het officiele dashboard. Bevat ook mensen die wegens een andere reden worden opgenomen maar positief getest zijn."
        "<br><i>Hospital_admission_GGD</i> - Nieuwe opnames in de ziekenhuizen GGD, lager omdat niet alles vanuit GGD wordt doorgegeven "
        "<br><br><i>new.infection</i> - Totaal aantal gevallen (GGD + ..?.. ) "
        "<br><i>new.deaths</i> - Totaal overledenen "
        "<br><i>Rt_avg</i> - Rt-getal berekend door RIVM"
        "<br><i>Tested_with_result</i> - Totaal aantal testen bij GGD "
        "<br><i>Tested_positive</i> - Totaal aantal positief getesten bij GGD "
        "<br><i>Percentage_positive</i> - Percentage positief getest bij de GGD "
        "<br><i>prev_avg</i> - Aantal besmettelijke mensen."
        "<br><br><i>total_vaccinations</i> - aantal doses geinjecteerd"
        "<br><i>people_vaccinated</i> - aantal mensen dat tenminste een prik heeft ontvangen"
        "<br><i>people_fully_vaccinated</i> - aantal mensen volledig gevaccineerd"
        "<br><i>*_diff</i> - * per day"

        "<br><br><i>retail_and_recreation, grocery_and_pharmacy, parks, transit_stations, workplaces, "
        "residential</i> - Mobiliteitsdata van Google"

        "<br><br><i>temp_etmaal</i> - Etmaalgemiddelde temperatuur (in graden Celsius)"
        "<br><i>temp_max</i> - Maximale temperatuur (in graden Celsius)"
        "<br><br><i>Zonneschijnduur</i> - Zonneschijnduur (uur) berekend uit de globale straling"
        "<br><i>Globale straling</i> - Globale straling in (in J//cm2) "
        "<br><i>Neerslag</i> - Etmaalsom van de neerslag (mm) (-1 voor  minder dan 0.05 mm) "
        "<br><i>Specific_humidity_KNMI_derived</i> - Specific humidity in g/kg, calculated with the 24-hours values of <i>De Bilt</i> from the KNMI : RH<sub>min</sub> and Temp<sub>max</sub>  with the formulas : <br><i>es = 6.112 * exp((17.67 * t)/(t + 243.5))<br>e = es * (rh / 100)<br>q = (0.622 * e)/(p - (0.378 * e)) * 1000 // [p = 1020]"
        "<br><i>Absolute_humidity_KNMI_derived</i> - Absolute humidity in g/kg, calculated with the 24-hours values of <i>De Bilt</i> from the KNMI : RH<sub>min</sub> and Temp<sub>max</sub>  with the formulas : <br><i>Absolute Humidity (grams/m3) = (6.112 × e^[(17.67 × T)/(T+243.5)] × rh × 2.1674) / (273.15+T)"


        "<br><i>RH_avg, RH_max, RH_min</i> - Relatieve luchtvochtigheid - 24 uurs gemiddelde, minimaal en maximaal"
        "<br><br><i>RNA_per_ml</i> - Rioolwater tot 9/9/2020"
        "<br><i>RNA_flow_per_100000</i> - Rioolwater vanaf 9/9/2020"
        "<br><i>RNA_per_reported</i> - (RNA_flow_per_100000/1e15)/ (new.infection * 100)"

        "<br><br><i>hosp_adm_per_reported</i> - Percentage hospital admissions "
        "<br><i>IC_adm_per_reported</i> - Percentage ICU admissions"
        "<br><i>new.deaths_per_reported</i> - Percentage hospital admissions "

        "<br><i>hosp_adm_per_reported_moved_5</i> - Percentage hospital admissions, total reported moved 5 days"
        "<br><i>IC_adm_per_reported_moved_5</i>  - Percentage hospital admissions, total reported moved 5 days - "
        "<br><i>new.deaths_per_reported_moved_14</i> - Percentage hospital admissions, total reported moved 14 days "

        "<br><br><i>*_cumm</i> - cummulative numbers, from the start"
        "<br><i>*_cumm_period</i> - cummulative numbers for the chosen period"

        "<br><br><i>prev_div_days_contagious</i> - Prevalentie gedeeld door [number_days_contagious] (aantal dagen dat men besmettelijk is) "
        "<br><i>prev_div_days_contagious_cumm</i> -"
        "<br><i>prev_div_days_contagious_cumm_period</i> -"
        "<br><i>new.deaths_per_prev_div_days_contagious</i> -"
        "<br><i>new.deaths_cumm_div_prev_div_days_contagious_cumm</i> -"
        "<br><i>new.deaths_cumm_period_div_prev_div_days_contagious_cumm_period</i> -"

        "<br><br><i>reported_corrected</i> - new.infection * (getest_positief / 12.8) - waarbij 12.8% het percentage positief was in week 1 van 2021"
        "<br><i>reported_corrected</i> - new.infection * (getest_positief / 1e waarde van getest_postief in tijdsperiode) "

        "<br><i>onderrapportagefactor</i> - prev_div_days_contagious_cumm / new.infection_cumm"
        "<br><br><i>*__diff_n_days</i> - Verschil tov een n dagen  terug in procenten [((nieuw-oud)/oud)*100]"
        "<br><i>*__diff_n_days_index</i> - Verschil tov n dagen terug als index [(nieuw/oud)*100] -> NB: Om rekenen naar R getal : [(nieuw/oud)^(4/7)]"
        "<br><br><i>pos_test_x-y, hosp_x-y, new.deaths_x-y</i> - Number of positive tests, hospital admissions and new.deaths by agecategory. Attention, the date is mostly the date of disease onset, so the first day of desease and given with a delay! These numbers are updated manually."
        "<br><br><i>Rt_corr_transit</i> -  Rt_avg * (1/ (1- transit_stations)). What would the R-number be if people don't change the number of contacts? Assumed is that the number of contacts between people is in direct correlation with the Google Transit. "
        "<br><i>Rt_corr_transit_period</i> -  Rt_avg * (1/ (1- transit_stations)) for the period chosen"
        "<h2>Toelichting bij de opties</h2>"
        "<h3>What to plot</h3>"
        "<i>Line</i> - Line graph"
        "<br><i>Linemax</i> - Indexed line grap. Maximum (smoothed) value is 100"
        "<br><i>Linefirst</i> - Indexed line graph. First (smoothed) value is 100"
        "<br><i>Bar</i> - Bar graph for the left axis, line graph for the right ax"
        "<h3>How to smooth</h3>"
        "<i>SMA</i> - Smooth moving average. <br><i>savgol</i> - <a href='https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter' target='_bank'>Savitzky-Golay filter</a>"
        "<h2>Hidden features</h2>"
        "<h3>Calculate R value</h3>"
        "Choose (what to plot)[bar] to calculate the R value."
        "<h3>Correlation</h3>"
        "If you have chosen one field on the left side and one for the right side, correlation of the fields are shown. Attention: <i>correlation is not causation</i>!"
        "<h3>Find correlations</h3>"
        "After clicking this button, you can choose your treshold. Fields with correlations above this treshold are shown"
        "<h3>Move curves at right axis (days)</h3>"
        "You can move the curves at the right ax to see possible cause-effect relations."
        "<h3>Show Scenario</h3>"
        "You are able to calculate a scenario based on two R-numbers, their ratio, a correction factor (to put in effect measures) and add extra days. Works only with [total reported]. The current values 'work' when the period starts at 2021/1/1. "
        "You can calculate scenarios with more options and graphs at my other webapp <a href='https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py' target='_blank'>https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py</a>"
        "<h2>Show specific weekday</h2>"
        "When you choose (day or week)[week] and (what to plot)[bar], you can choose one specific weekday to compare them more easily"
        "<h2>Datasource</h2>"
        "Data is scraped from https://data.rivm.nl/covid-19/ and hospital_intake_rivm and cached. "
        ' <a href=/"https://coronadashboard.rijksoverheid.nl/verantwoording#ziekenhuizen/" target=/"_blank/">Info here</a>.<br>'
        "For the moment most of the data is be updated automatically every 24h."
        ' The <a href=/"https://www.knmi.nl/nederland-nu/klimatologie/daggegevens/" target=/"_blank/">KNMI</a> and  Google  data will be updated manually at a lower frequency.<br><br>'
        "<b>Feel free to contact me for any wishes, added fields or calculations</b>"
    )

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/blob/main/covid_dashboard_rcsmit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        'Restrictions by <a href="https://twitter.com/hk_nien" target="_blank">Han-Kwang Nienhuys</a> (MIT-license).</div>'
    )

    st.markdown(toelichting, unsafe_allow_html=True)
    st.sidebar.markdown(tekst, unsafe_allow_html=True)
    now = UPDATETIME
    UPDATETIME_ = now.strftime("%d/%m/%Y %H:%M:%S")
    st.write(f"\n\n\nData last updated : {str(UPDATETIME_)}")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.image(
        "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png"
    )

    st.markdown(
        '<a href="https://www.buymeacoffee.com/rcsmit" target="_blank">If you are happy with this dashboard, you can buy me a coffee</a>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<br><br><a href="https://www.linkedin.com/in/rcsmit" target="_blank">Contact me for custom dashboards and infographics</a>',
        unsafe_allow_html=True,
    )
