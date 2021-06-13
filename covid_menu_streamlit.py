import streamlit as st
import importlib

def dynamic_import(module):
    """Import a module stored in a variable

    Args:
        module (string): The module to import

    Returns:
        the module you want
    """
    return importlib.import_module(module)

def main():

       #    [n. name in menu, module name]
    options = [["0. welcome","welcome"],
            ["1. covid dashboard","covid dashboard rcsmit"],
            ["2. plot hosp ic per age","plot hosp ic streamlit"],
            ["3. false positive rate covid test","calculate false positive rate covid test streamlit"],
            ["4. number of cases interactive","number of cases interactive"],
            ["5. ifr from prevalence","calculate_ifr_from_prevalence_streamlit"],
            ["6. fit to data","fit to data streamlit"],
            ["7. SEIR hobbeland","SEIR hobbeland"],
            ["8. show contactmatrix","show contactmatrix"],
            ["9. r getal per provincie","r getal per provincie"],
            ["10. Cases from suspectibles", "cases_from_susp_streamlit"],
            ["11. Fit to data OWID", "fit_to_data_owid_streamlit_animated"],
            ["12. Calculate R per country owid", "calculate_r_per_country_owid_streamlit.py"]
            ["13. grafiek pos testen per leeftijdscat","grafiek pos testen per leeftijdscategorie streamlit"],
            ["14. per provincie per leeftijd","perprovincieperleeftijd"]]


    query_params = st.experimental_get_query_params() # reading  the choice from the URL..
    choice = int(query_params["choice"][0]) if "choice" in query_params else 0 # .. and make it the default value

    menuchoicelist = [] # making another list with the options to be shown

    for n, l in enumerate(options):
        menuchoicelist.append(options[n][0])

    with st.sidebar.beta_expander('MENU: Choose a script',  expanded=True):
        menu_choice = st.radio("",menuchoicelist, index=choice)

    st.sidebar.markdown("<h1>- - - - - - - - - - - - - - - - - - </h1>", unsafe_allow_html=True)
    st.experimental_set_query_params(choice=menuchoicelist.index(menu_choice)) # setting the choice in the URL

    for n, l in enumerate(options):
        if menu_choice == options[n][0]:
            m = options[n][1].replace(" ","_") # I was too lazy to change it in the list
            try:
                module = dynamic_import(m)
            except Exception as e:
                st.error(f"Module '{m}' not found or error in the script\n{e}")
                st.stop()
            try:
                module.main()
            except Exception as e:
                st.error(f"Function 'main()' in module '{m}' not found or error in the script\n{e}")
                st.stop()

if __name__ == "__main__":
    main()