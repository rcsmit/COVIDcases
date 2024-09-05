import streamlit as st
import importlib
import traceback


st.set_page_config(page_title="COVID SCRIPTS of René Smit", layout="wide")

def show_info():
    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/blob/main/covid_dashboard_rcsmit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
    )

    st.sidebar.markdown(tekst, unsafe_allow_html=True)
def dynamic_import(module):
    """Import a module stored in a variable
    Args:
        module (string): The module to import
    Returns:
        the module you want
    """
    return importlib.import_module(module)

def main():
    st.title ("COVID SCRIPTS of René Smit -")

       #    [n. name in menu, module name]
    options = [["[0] welcome","welcome"],
            ["[X] covid dashboard","covid dashboard rcsmit"],
            ["[X] plot hosp ic per age","plot hosp ic streamlit"],
            ["[3] false positive rate covid test","calculate false positive rate covid test streamlit"],
            ["[4] number of cases interactive","number of cases interactive"],
            ["[5] ifr from prevalence","calculate_ifr_from_prevalence_streamlit"],
            ["[6] fit to data","fit to data streamlit"],
            ["[7] SEIR hobbeland","SEIR hobbeland"],
            ["[8] show contactmatrix","show contactmatrix"],
            ["[9] r getal per provincie","r getal per provincie"],
            ["[XX] Cases from suspectibles", "cases_from_susp_streamlit"],
            ["[11] Fit to data OWID", "fit_to_data_owid_streamlit_animated"],
            ["[12] Calculate R per country owid", "calculate_r_per_country_owid_streamlit"],
            ["[13] Covid dashboard OWID/Google or Waze","covid dashboard owid"],
            ["[14] Dag verschillen per leeftijd", "dag_verschillen_casus_landelijk"],
            ["[15] Calculate spec]/abs. humidity from rel. hum", "rh2q"],
            ["[16] R getal per leeftijdscategorie", "r_number_by_age"],
            ["[17] Show rioolwaardes", "show_rioolwater"],
            ["[18] SIR model met leeftijdsgroepen","SIR_age_structured_streamlit"],
            ["[19] grafiek pos testen per leeftijdscat","grafiek pos testen per leeftijdscategorie streamlit"],
            ["[20] per provincie per leeftijd","perprovincieperleeftijd"],
            ["[21] kans om covid op te lopen","kans_om_covid_op_te_lopen"],
            ["[22] Data per gemeente", "vacc_inkomen_cases"] ,
            ["[23] VE Israel", "israel_zijlstra"],
            ["[24] Hosp/death NL", "cases_hospital_decased_NL"],
            ["[25] VE Nederland", "VE_nederland_"],
            ["[26] Scatterplots QoG OWID", "qog_owid"],
            ["[27] VE & CI calculations", "VE_CI_calculations"],
            ["[28] VE scenario calculator", "VE_scenario_calculator"],
            ["[29] VE vs inv. odds", "VE_vs_inv_odds"],
            ["[30] Fit to data Levitt", "fit_to_data_owid_levitt_streamlit_animated"],
            ["[31] Aerosol concentration in room by @hk_nien", "aerosol_in_room_streamlit"],
            ["[32] Compare two variants", "compare_two_variants"],
            ["[33] Scatterplot OWID", "scatterplots_owid"],
            ["[34] Playing with R0", "playing_with_R0"],
            ["[35] Calculate Se & Sp Rapidtest" , "calculate_se_sp_rapidtest_streamlit"],
            ["[36] Oversterfte gemeente" , "oversterfte_gemeente"],
            ["[37] Oversterfte" , "oversterfte"],
            ["[38] Bayes Lines tools" , "bayes_lines_tools"],
            ["[39] Oversterfte (CBS Odata)" , "oversterfte_compleet"],
            ["[40] Bayes berekeningen IC ziekenh" , "bayes_prob_ic_hosp"],
            ["[41] Disabled by Long covid" , "disabled_by_longcovid"],
            ["[42] Oversterfte 5yrs groeps Eurostat week" , "oversterfte_eurostats"],
            ["[43] Oversterfte 5yrs groeps Eurostat maand" , "oversterfte_eurostats_maand"],
            ["[44] Rioolwaarde vs ziekenhuis" , "rioolwater_vs_ziekenhuis"],
            ["[45] Rioolwaarde vs overleden CBS", "overledenen_rioolwaardes"],
            ["[46] Mortality yearly per capita", "mortality_yearly_per_capita"],
            ["[47] Deltavax", "deltavax"],
            ]

    #query_params = st.experimental_get_query_params() # reading  the choice from the URL..
    # query_params = st.query_params["choice"] 
    try:
        choice = int(st.query_params["choice"]) 
    except:
        choice = 0 
    # if "choice" in query_params else 0 # .. and make it the default value

    if choice == 99:  #sandbox
        try:
            module = dynamic_import("various_test_and_sandbox")
        except Exception as e:
            st.error(f"Module not found or error in the script\n")
            st.warning(f"{e}")
            st.stop()
        try:
            module.main()
        except Exception as e:
            st.error(f"Function 'main()' in module '{module}' not found or error in the script")
            st.warning(f"{e}")
            st.warning(traceback.format_exc())
            st.stop()
        st.stop()

    menuchoicelist = [options[n][0] for n, l in enumerate(options)]

    with st.sidebar.expander('MENU: Choose a script | scroll down for options/parameters',  expanded=True):
        try:
            menu_choice = st.radio(".",menuchoicelist, index=choice,  label_visibility='hidden')
        except:
            st.error("ERROR. Choice not in menulist")
            st.stop()
    st.sidebar.markdown("<h1>- - - - - - - - - - - - - - - - - </h1>", unsafe_allow_html=True)
    #st.experimental_set_query_params(choice=menuchoicelist.index(menu_choice)) # setting the choice in the URL
    st.query_params.choice = menuchoicelist.index(menu_choice)
    for n, l in enumerate(options):
        if menu_choice == options[n][0]:
            m = options[n][1].replace(" ","_") # I was too lazy to change it in the list
            try:
                module = dynamic_import(m)
            except Exception as e:
                st.error(f"Module '{m}' not found or error in the script\n")
                st.warning(f"{e}")
                st.warning(traceback.format_exc())
                st.stop()
            try:
                module.main()
            except Exception as e:
                st.error(f"Function 'main()' in module '{m}' not found or error in the script")
                st.warning(f"{e}")
                st.warning(traceback.format_exc())
                st.stop()

if __name__ == "__main__":
    main()
    show_info()
