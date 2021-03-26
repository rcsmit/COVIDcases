# Grafiek positief getest naar leeftijd door de tijd heen, per leeftijdscategorie
# René Smit, (@rcsmit) - MIT Licence

# IN: tabel met positief aantal testen en totaal aantal testen per week, gecategoriseerd naar leeftijd
#     handmatig overgenomen uit Tabel 14 vh wekelijkse rapport van RIVM
#     Wekelijkse update epidemiologische situatie COVID-19 in Nederland
#     https://www.rivm.nl/coronavirus-covid-19/actueel/wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland

# Uitdagingen : Kopieren en bewerken Tabel 14. 3 verschillende leeftijdsindelingen. Tot dec. 2020 alles
# cummulatief. X-as in de grafiek

# TODO : - Nog enkele weken toevoegen voorafgaand het huidige begin in de datafile (waren weken met weinig besmettingen).
#        - integreren in het dashboard
#        - 'Total reported' toevoegen


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from labellines import *  #https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
import streamlit as st
from streamlit import caching
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock
def save_df(df,name):
    """  _ _ _ """
    OUTPUT_DIR = 'C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\'

    name_ =  OUTPUT_DIR + name+'.csv'
    compression_opts = dict(method=None,
                            archive_name=name_)
    df.to_csv(name_, index=False,
            compression=compression_opts)

    print ("--- Saving "+ name_ + " ---" )

@st.cache(ttl=60 * 60 * 24)
def read_df():
    #url = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\covid19_seir_models\input\pos_test_leeftijdscat_wekelijks.csv"
    url= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/pos_test_leeftijdscat_wekelijks.csv"
    #id;datum;leeftijdscat;methode;mannen_pos;mannen_getest;vrouwen_pos ;vrouwen_getest ;
    # totaal_pos;totaal_getest;weeknr2021;van2021;tot2021

    df   = pd.read_csv(url,
                        delimiter=";",
                        low_memory=False)
    return df

def action(to_show_in_graph):

    df_= read_df()
    df = df_.copy(deep=False)
    df["datum"]=pd.to_datetime(df["datum"], format='%d-%m-%Y')

    list_dates = df["datum"].unique()
    cat_oud = [ "0-4",  "05-09",  "10-14",  "15-19",  "20-24",  "25-29",  "30-34",  "35-39",
                "40-44",  "45-49",  "50-54",  "55-59",  "60-64",  "65-69",  "70-74",  "75-79",  "80-84",  "85-89",  "90-94",  "95+" ]
    cat_vervanging = [ "0-4",  "05-09",  "10-14",  "15-19",  "20-29",  "20-29",  "30-39",  "30-39",
                "40-49",  "40-49",  "50-59",  "50-59",  "60-69",  "60-69",   "70+",   "70+",      "70+",   "70+",    "70+",   "70+" ]

    cat_nieuw = [ "0-12", "13-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld"]
    cat_nieuwst_code =["A",  "B",    "C",     "D",    "E",       "F"     "G",     "H",     "I",    "J",     "K"]
    cat_nieuwst= ["0-3", "04-12", "13-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld"]

    # Deze grafieken komen uiteindelijk voort in de grafiek
    cat_nieuwstx= ["0-12", "0-03", "04-12", "13-17", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+", "Niet vermeld"]

    color_list = [  "#ff6666",  # reddish 0
                    "#ac80a0",  # purple 1
                    "#3fa34d",  # green 2
                    "#EAD94C",  # yellow 3
                    "#EFA00B",  # orange 4
                    "#7b2d26",  # red 5
                    "#3e5c76",  # blue 6
                    "#e49273" , # dark salmon 7
                    "#1D2D44",  # 8
                    "#02A6A8",
                    "#4E9148",
                    "#F05225",
                    "#024754",
                    "#FBAA27",
                    "#302823",
                    "#F07826",
                     ]
    #####################################################
    df_new= pd.DataFrame({'date': [],'cat_oud': [],
                    'cat_nieuw': [], "positief_testen": [],"totaal_testen": [], "methode":[]})

    for i in range(0, len(df)):
        d =  df.loc[i, "datum"]
        for x in range(0,len(cat_oud)-1):
            c_o,c,p,t,m = None,None,None,None,None
            if df.loc[i, "methode"] == "oud":
                # print (df.loc[i, "leeftijdscat"])
                # print (f"----{df.loc[i, 'leeftijdscat']}----{cat_oud[x]}----")
                if df.loc[i, "leeftijdscat"] == cat_oud[x]:

                    c_o =  cat_oud[x]
                    c = cat_vervanging[x]
                    # print (f"{c} - {i} - {x} ")
                    # print (f"----{df.loc[i, 'leeftijdscat']}----{cat_oud[x]}----")
                    p =df.loc[i, "totaal_pos"]
                    t = df.loc[i, "totaal_getest"]
                    m = df.loc[i, "methode"] == "oud"
                    df_new = df_new.append({ 'date': d, 'cat_oud': c_o, 'cat_nieuw': c,  "positief_testen": p,"totaal_testen":t, "methode": m}, ignore_index= True)
                    c_o,c,p,t,m = None,None,None,None,None
            else:
                if x <= len(cat_nieuwstx)-1 :
                    if df.loc[i, "leeftijdscat"] == cat_nieuwstx[x]:
                        c_o =  df.loc[i, "leeftijdscat"]
                        c =  df.loc[i, "leeftijdscat"]
                        p =df.loc[i, "totaal_pos"]
                        t = df.loc[i, "totaal_getest"]
                        m = df.loc[i, "methode"]
                        df_new = df_new.append({ 'date': d, 'cat_oud': c_o, 'cat_nieuw': c,  "positief_testen": p,"totaal_testen":t, "methode": m}, ignore_index= True)
                        c_o,c,p,t,m = None,None,None,None,None

    df_new = df_new.groupby(['date','cat_nieuw'], sort=True).sum().reset_index()

    df_new['percentage'] =  round((df_new['positief_testen']/df_new['totaal_testen']*100),1)


    show_from = "2020-1-1"
    show_until = "2030-1-1"

    startdate = pd.to_datetime(show_from).date()
    enddate = pd.to_datetime(show_until).date()
    datumveld = 'date'
    mask = (df_new[datumveld].dt.date >= startdate) & (df_new[datumveld].dt.date <= enddate)
    df_new = df_new.loc[mask]

    print (f'Totaal aantal positieve testen : {df_new["positief_testen"].sum()}')
    print (f'Totaal aantal testen : {df_new["totaal_testen"].sum()}')
    print (f'Percentage positief  : {  round (( 100 * df_new["positief_testen"].sum() /  df_new["totaal_testen"].sum() ),2)    }')

    list_age_groups =  df_new["cat_nieuw"].unique()


    with _lock:
        fig1x, ax = plt.subplots(1,1)
        #for l in to_show_in_graph:
        for i,l in enumerate(to_show_in_graph):
            df_temp = df_new[df_new['cat_nieuw']==l]
            list_percentage = df_temp["percentage"].tolist()
            list_dates = df_temp["date"].tolist()

            plt.plot(list_dates, list_percentage, color = color_list[i], label = l)
        #labelLines(plt.gca().get_lines(),align=False,fontsize=8)

        ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',
                    transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
        plt.title("Percentage Positieve testen per agegroup" , fontsize=10)
        plt.legend(bbox_to_anchor=(1.3, 1),loc="best")
        plt.tight_layout()
        #plt.show()
        st.pyplot(fig1x)

def main():
    lijst = ["0-12", "0-03", "04-12", "13-17",  "0-4",  "05-09",  "10-14",  "15-19", "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    to_show_in_graph = st.sidebar.multiselect(
                "What to show  (multiple possible)", lijst, ["30-39", "40-49", "50-59", "60-69"])
    st.sidebar.write("Onder de 18 heeft men 3x de groepsindeling veranderd.")
    st.header ("Percentage positieve testen per leeftijdsgroep door de tijd heen")
    action(to_show_in_graph)
    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/grafiek_pos_testen_per_leeftijdscategorie_streamlit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
        )

    st.sidebar.markdown(tekst, unsafe_allow_html=True)

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

main()
