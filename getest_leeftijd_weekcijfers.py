# IS ER EEN VERBAND MET HET PERCENTAGE POSITIEF PER LEEFTIJDSGROEP
# EN DE ZIEKENHUISOPNAMES?
#
# René SMIT (@rcsmit) - MIT-licence
# Geinspireerd door Maarten van den Berg (@mr_Smith_Econ)

# IN: - tabel met positief aantal testen en totaal aantal testen per week, gecategoriseerd naar leeftijd
#       handmatig overgenomen uit Tabel 14 vh wekelijkse rapport van RIVM
#       Wekelijkse update epidemiologische situatie COVID-19 in Nederland
#       https://www.rivm.nl/coronavirus-covid-19/actueel/wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland
#       Ivm het simpel houden van de code enkele aanpassingen:
#       0 -3 is toegevoegd bij 0 gevallen vóór de opsplitsing
#       70+ => 70-97
#       onbekend => 97-99
#
#     - tabel met ziekenhuisopnames per dag


# UIT : grafiek : https://twitter.com/rcsmit/status/1374425588409114629
#
# TODO : - één bar, één lijn grafiek
#        - weeknummers

import pandas as pd
import matplotlib.pyplot as plt

url1 = 'https://raw.githubusercontent.com/rcsmit/COVIDcases/main/postestennaarleeftijd2.csv'
df1   = pd.read_csv(url1,
                    delimiter=";",
                    low_memory=False)
url2 = 'https://raw.githubusercontent.com/rcsmit/COVIDcases/main/weektabel.csv'
df2   = pd.read_csv(url2,
                    delimiter=",",
                    low_memory=False)
print (df1.dtypes)
print (df2.dtypes)

dfx = pd.merge(
                df1, df2, how="outer", left_on="weeknr", right_on="weeknr"
            )
df_actual = dfx[dfx['tot']==59].copy(deep=False)
print (df_actual.dtypes)

print (df_actual)
df_actual["perc_pos_getest"] = round(
        ((df_actual["positief"] / (df_actual["getest"]) *100 )), 2)

x = [1,2,3,4,5,6,7,8,9,10,11]

fig1x, ax = plt.subplots(1,1)

df_actual.Hospital_admission_RIVM.plot(ax=ax, label="ziekenh opn", style='b-')
df_actual.perc_pos_getest.plot(ax=ax, style='r-', label="perc pos getest", secondary_y=True)

print(df_actual["perc_pos_getest"])
print(df_actual["Hospital_admission_RIVM"])
ax.set_ylabel('percentage')

handles,labels = [],[]
for ax in fig1x.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
plt.legend(handles,labels)
ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',
            transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')

plt.legend()
plt.title("ZIekenhuisopnames (allen) vs getest positief 50-59" , fontsize=10)
plt.show()
