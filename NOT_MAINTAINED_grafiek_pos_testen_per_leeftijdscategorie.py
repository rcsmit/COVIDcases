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

def save_df(df,name):
    """  _ _ _ """
    OUTPUT_DIR = 'C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\'

    name_ =  OUTPUT_DIR + name+'.csv'
    compression_opts = dict(method=None,
                            archive_name=name_)
    df.to_csv(name_, index=False,
            compression=compression_opts)

    print ("--- Saving "+ name_ + " ---" )

def main():

    #url = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\covid19_seir_models\input\pos_test_leeftijdscat_wekelijks.csv"
    url= "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/pos_test_leeftijdscat_wekelijks.csv"
    to_show_in_graph = [ "18-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    #id;datum;leeftijdscat;methode;mannen_pos;mannen_getest;vrouwen_pos ;vrouwen_getest ;
    # totaal_pos;totaal_getest;weeknr2021;van2021;tot2021

    df   = pd.read_csv(url,
                        delimiter=";",
                        low_memory=False)

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


    #####################################################
    df_new= pd.DataFrame({'date': [],'cat_oud': [],
                    'cat_nieuw': [], "positief_testen": [],"totaal_testen": [], "methode":[]})

    for i in range(len(df)):
        d =  df.loc[i, "datum"]
        for x in range(len(cat_oud)-1):
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
            elif (
                x <= len(cat_nieuwstx) - 1
                and df.loc[i, "leeftijdscat"] == cat_nieuwstx[x]
            ):
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



    fig1x, ax = plt.subplots(1,1)
    for l in to_show_in_graph:
        df_temp = df_new[df_new['cat_nieuw']==l]
        list_percentage = df_temp["percentage"].tolist()
        list_dates = df_temp["date"].tolist()

        plt.plot(list_dates, list_percentage, label = l)

    ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',
                transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
    plt.title("Percentage Positieve testen per agegroup" , fontsize=10)
    plt.legend(bbox_to_anchor=(1.3, 1),loc="best")
    plt.tight_layout()
    plt.show()



main()
