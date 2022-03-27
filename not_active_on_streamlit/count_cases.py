
# BEREKEN HET TOTAAL AANTAL GEVALLEN PER LEEFTIJDSGROPE IN casus_landelijk.csv

import pandas as pd

url =  "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\covid19_seir_models\\input\\COVID-19_casus_landelijk.csv"
df_temp = pd.read_csv(url, delimiter=";", low_memory=False)
df = df_temp.groupby(pd.Grouper(key="Agegroup")).count().reset_index()
print (df)

