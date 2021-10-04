import pandas as pd
from functools import reduce

# Antwoord op de vraag
#
# Als de hele wereld relatief gezien net zo veel (geregistreerde) coronadoden had als NL,"
# hoeveel waren dat er wereldwijd dan minder/meer geweest dan het huidige aantal van 4,6 miljoen?
#
# https://twitter.com/rcsmit/status/1434497100411293700

def save_df(df, name):

    """  Saves the dataframe """
    name_ =  name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)
    print("--- Saving " + name_ + " ---")

def prepare_cases_landelijk():
    """Berekent aantal overleden per leeftijdsgroep van casus_landelijk.csv (data.rivm.nl)

    Returns:
        df: df met aantal overleden per leeftijdsgroep
    """
    url1 = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
    df = pd.read_csv(url1, delimiter=";", low_memory=False)
    df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")
    df.rename(
        columns={
            "Date_file": "cases_nl",
        },
        inplace=True,
    )

    df_deceased = df[df["Deceased"] == "Yes"].copy(deep=False)

    df_deceased = df_deceased.groupby(["Agegroup"], sort=True).count().reset_index()
    df_deceased = df_deceased[["Agegroup", "cases_nl"]]
    agegrpstart_old = ["<50", "50-59", "60-69", "70-79", "80-89", "90+"]
    agegrpstart_new = [0, 50,60,70,80,90]

    for i in range(len(df_deceased)):
        for x,y in zip(agegrpstart_old,agegrpstart_new):
                if df_deceased.loc[i,"Agegroup"] == x : df_deceased.loc[i, "AgeGrp2"] = y
    print (df_deceased)
    print (df_deceased["cases_nl"].sum())
    return df_deceased

def haal_data_op():
    """Filtert .csv bestand; alleen data van landen en uit 2021
    https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2019_PopulationByAgeSex_Medium.csv

    Maakt gebruik van een lijst met landennamen.

    Returns:
        df : dataframe met populatiedata van landen (geen regios) in 2021
    """
    df = pd.read_csv("C:\\Users\\rcxsm\\Documents\\phyton_scripts\\in\\WPP2019_PopulationByAgeSex_Medium.csv")
    df_countries = pd.read_csv("C:\\Users\\rcxsm\\Documents\\phyton_scripts\\in\\countrynames.csv")
    df = df[df["Time"] == 2021]
    df["AgeGrp2"] = None

    df_temp = pd.merge(
                df, df_countries, how="right", left_on="Location", right_on="Countryname"
            )

    return df_temp

def maak_groeps_colom(df_temp):
    """Voegt een extra kolom toe zodat je twee groepen kunt samenvoegen (bijv. 50-54 en 55-59 naar 50-59)

    Args:
        df_temp (df):

    Returns:
        df: df met de nieuwe kolom
    """
    for i in range(len(df_temp)):
        agegrpstart_old = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
        # agegrpstart_new = [0,0,10,10,20,20,30,30,40,40,50,50,60,60,70,70,80,80,90,90,90]
        agegrpstart_new = [0,0,0,0,0,0,0,0,0,0,50,50,60,60,70,70,80,80,90,90,100]
        for x,y in zip(agegrpstart_old,agegrpstart_new):
            if df_temp.loc[i,"AgeGrpStart"] == x : df_temp.loc[i, "AgeGrp2"] = y
    return df_temp

def  leeftijdsopbouw():
    df = haal_data_op()
    df = maak_groeps_colom(df)
    df_leeftijdsopbouw_aarde = df.groupby('AgeGrp2', sort=False).sum().copy(deep=False)
    print (df_leeftijdsopbouw_aarde.dtypes)

    df_leeftijdsopbouw_aarde["PopTotal_aarde"] = df_leeftijdsopbouw_aarde["PopTotal"]*1000
    df_leeftijdsopbouw_aarde = df_leeftijdsopbouw_aarde[["PopTotal_aarde"]]
    df_nl = df[df["Location"]== "Netherlands"].copy(deep=False)
    df_leeftijdsopbouw_nl = df_nl.groupby('AgeGrp2', sort=False).sum()
    df_leeftijdsopbouw_nl["PopTotal_nl"] = df_leeftijdsopbouw_nl["PopTotal"]*1000
    df_leeftijdsopbouw_nl  = df_leeftijdsopbouw_nl[["PopTotal_nl"]]
    print ("NEDERLAND")
    print (df_leeftijdsopbouw_nl)
    print (f"Totale bevolking NL : {df_leeftijdsopbouw_nl['PopTotal_nl'].sum()}")

    print ("WERELD")
    print (df_leeftijdsopbouw_aarde)
    print (f"Totale bevolking  aarde :  {df_leeftijdsopbouw_aarde['PopTotal_aarde'].sum()}")


    return df_leeftijdsopbouw_aarde, df_leeftijdsopbouw_nl

def main():

    df_aarde, df_nl = leeftijdsopbouw()
    df_deceased = prepare_cases_landelijk()
    dfs = [df_aarde, df_nl, df_deceased]
    df_final = reduce(lambda left,right: pd.merge(left,right,on='AgeGrp2'), dfs)

    df_final["cases_per_inw_nl"] = df_final["cases_nl"] / df_final["PopTotal_nl"]
    df_final["cases_totaal_aarde"] = df_final["PopTotal_aarde"] * df_final["cases_per_inw_nl"]
    print (df_final)
    save_df(df_final, "df_final")
    print (f"TOTAAL AANTAL CASES COVID OP AARDE : {round(df_final['cases_totaal_aarde'].sum())}")

main()