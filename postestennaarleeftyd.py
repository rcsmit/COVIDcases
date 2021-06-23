# ZIJN KINDEREN DE REDEN DAT HET PERCENTAGE POSITIEF DAALT ?
#
# René SMIT (@rcsmit) - MIT-licence
# Geinspireerd door Maarten van den Berg (@mr_Smith_Econ)

# IN: tabel met positief aantal testen en totaal aantal testen per week, gecategoriseerd naar leeftijd
#     handmatig overgenomen uit Tabel 14 vh wekelijkse rapport van RIVM
#     Wekelijkse update epidemiologische situatie COVID-19 in Nederland
#     https://www.rivm.nl/coronavirus-covid-19/actueel/wekelijkse-update-epidemiologische-situatie-covid-19-in-nederland

#     Ivm het simpel houden van de code enkele aanpassingen:
#     0 -3 is toegevoegd bij 0 gevallen vóór de opsplitsing
#     70+ => 70-97
#     onbekend => 97-99

# UIT : tabel   | weeknr    getest  positief  percentage  getest_weg  positief_weg  getest_metkids  positief_metkids  percentage_metkids|
#       grafiek : https://twitter.com/rcsmit/status/1373308226624716806
#       (één grafiek met de kinderen inbegrepen en één grafiek zonder kinderen inbegrepen)
#       NB door een bugje ergens laat hij ook 2x een lege grafiek zien

# TODO : functie maken voor het laten zien vd grafiek.
#        lege grafiek weghalen

import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/rcsmit/COVIDcases/main/postestennaarleeftijd2.csv'
df   = pd.read_csv(url,
                    delimiter=";",
                    low_memory=False)

uitslag= pd.DataFrame({'weeknr': [],
                'getest': [],'positief': [], 'percentage': [],
                'getest_weg': [],'positief_weg': [],
                'getest_metkids': [],'positief_metkids': [], 'percentage_metkids': []})

actuele_week = 0

for n in range (len(df)):
    huidige_week = df.loc[n]['weeknr']
    if huidige_week != actuele_week:

        # uitslagen zonder kinderen
        cumm_getest = 0
        cumm_positief = 0

        # niet getelde testen/uitslagen (WEGgelaten)
        getest_weg = 0
        positief_weg = 0

        # uitslagen met kinderen (dus alles)
        cumm_getest_metkids = 0
        cumm_positief_metkids = 0
        actuele_week = df.loc[n]['weeknr']

    cumm_getest_metkids +=  (df.loc[n]['getest'])
    cumm_positief_metkids += (df.loc[n]['positief'])

    if df.loc[n]['tot'] in [3, 12, 99]:
        getest_weg +=  (df.loc[n]['getest'])
        positief_weg += (df.loc[n]['positief'])
    else:
        cumm_getest +=  (df.loc[n]['getest'])
        cumm_positief += (df.loc[n]['positief'])
    if df.loc[n]['tot'] == 99:
        uitslag = uitslag.append({'weeknr':df.loc[n]['weeknr'],
            'getest': cumm_getest,'positief': cumm_positief, 'percentage': round((cumm_positief/cumm_getest*100),1),
            'getest_weg': getest_weg,'positief_weg': positief_weg,
            'getest_metkids': cumm_getest_metkids,'positief_metkids': cumm_positief_metkids, 'percentage_metkids': round((cumm_positief_metkids/cumm_getest_metkids*100),1)  },ignore_index=True)

print (uitslag)

# GRAFIEK ZONDER KINDEREN

fig1x, ax = plt.subplots(1,1)
#ax = uitslag.plot(x="weeknr", y=["getest", "positief"], kind="bar")
ax = uitslag.plot(x="weeknr", y=[ "positief"], kind="bar")
ax3=uitslag["percentage"].plot(secondary_y=True,linestyle='--', label="Percentage positive")

ax.set_ylabel('aantal')
ax3.set_ylabel('percentage')
handles,labels = [],[]
for ax in fig1x.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
plt.legend(handles,labels)
ax3.set_ylim(0,15)
ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',
            transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
plt.title("Getest boven de 12 jaar" , fontsize=10)
plt.show()
exit()
# GRAFIEK MET KINDEREN

fig1y = plt.subplots()
ax = uitslag.plot(x="weeknr", y=["getest_metkids", "positief_metkids"], kind="bar")
ax3=uitslag["percentage_metkids"].plot(secondary_y=True,linestyle='--', label="Percentage positive_metkids")

ax.set_ylabel('aantal')
ax3.set_ylabel('percentage')
#ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',fontsize='xx-small', va='top', ha='right')
handles,labels = [],[]
for ax in fig1y.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
plt.legend(handles,labels)
ax3.set_ylim(0,15)

plt.title("Getest alles" , fontsize=10)
plt.show()
