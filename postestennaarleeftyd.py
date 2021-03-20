import pandas as pd
import matplotlib.pyplot as plt
url = 'C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\postestennaarleeftijd2.csv'
df   = pd.read_csv(url,
                    delimiter=";",
                    low_memory=False)
print (df)
uitslag= pd.DataFrame({'weeknr': [],
                'getest': [],'positief': [], 'percentage': [],
                 'getest_metkids': [],'positief_metkids': [], 'percentage_metkids': []})
actuele_week = 0

for n in range (len(df)):

    huidige_week = df.loc[n]['weeknr']
    if huidige_week != actuele_week:
        cumm_getest = 0
        cumm_positief = 0

        cumm_getest_metkids = 0
        cumm_positief_metkids = 0
        actuele_week = df.loc[n]['weeknr']

    cumm_getest_metkids +=  (df.loc[n]['getest'])
    cumm_positief_metkids += (df.loc[n]['positief'])
    if df.loc[n]['tot'] ==3 or  df.loc[n]['tot'] ==12:
        pass

    else:
        #print (df.loc[n]['tot'])
        #print (df.loc[n]['id'])
        cumm_getest +=  (df.loc[n]['getest'])
        cumm_positief += (df.loc[n]['positief'])
        if df.loc[n]['tot'] ==99:
            print (cumm_getest)
            uitslag = uitslag.append({'weeknr':df.loc[n]['weeknr'],
                'getest': cumm_getest,'positief': cumm_positief, 'percentage': round((cumm_positief/cumm_getest*100),1),
                 'getest_metkids': cumm_getest_metkids,'positief_metkids': cumm_positief_metkids, 'percentage_metkids': round((cumm_positief_metkids/cumm_getest_metkids*100),1)  },ignore_index=True)
            cumm_getest = 0
            cumm_positief =0
print (uitslag)

fig1x = plt.figure()
ax = fig1x.add_subplot(111)
plt.xlabel('week')

# ax=uitslag["getest"].plot.bar(label="getest")
# ax=uitslag["positief"].plot.bar(label="positief")
ax = uitslag.plot(x="weeknr", y=["getest", "positief"], kind="bar")

ax3=uitslag["percentage"].plot(secondary_y=True,linestyle='--', label="Percentage positive")

ax.set_ylabel('aantal')
ax3.set_ylabel('percentage')
handles,labels = [],[]
for ax in fig1x.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
plt.legend(handles,labels)
# Add a grid
ax3.set_ylim(0,15)
plt.title("Getest boven de 12 jaar" , fontsize=10)
plt.show()



fig1x = plt.figure()
ax = fig1x.add_subplot(111)
plt.xlabel('week')

# ax=uitslag["getest"].plot.bar(label="getest")
# ax=uitslag["positief"].plot.bar(label="positief")
ax = uitslag.plot(x="weeknr", y=["getest_metkids", "positief_metkids"], kind="bar")

ax3=uitslag["percentage_metkids"].plot(secondary_y=True,linestyle='--', label="Percentage positive_metkids")

ax.set_ylabel('aantal')
ax3.set_ylabel('percentage')
handles,labels = [],[]
for ax in fig1x.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
plt.legend(handles,labels)
# Add a grid
ax3.set_ylim(0,15)
plt.title("Getest alles" , fontsize=10)
plt.show()