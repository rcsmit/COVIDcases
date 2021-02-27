import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sn
from scipy import stats
    
def configgraph(titlex):
    interval_ = 20
    plt.xlabel('date')
    
    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    plt.legend(  loc='best')
    plt.title(titlex , fontsize=10)

# R-numbers from 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
# Google mobilty from https://www.google.com/covid19/mobility/?hl=nl
# Merged in one file in Excel and saved to CSV

df = pd.read_csv(
                r'covid19_seir_models\mobilityR.csv',
                delimiter=';',
                parse_dates=["date"]
            )

# 7 days average
WDW = 7

# remove outliers - NOTHING WORKS

# for r in range(2,7):
#     print (r)
#     df = df[(np.abs(stats.zscore(df[r])) < 3)]
# #df[np.abs(df.grocery_and_pharmacy.mean()) <= (3*df.grocery_and_pharmacy.std())]
# keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
#df[((df.grocery_and_pharmacy - df.grocery_and_pharmacy.mean()) / df.grocery_and_pharmacy.std()).abs() < 1]

df['retail_and_recreation_SMA'] = df.iloc[:,2].rolling(window=WDW).mean()
df['grocery_and_pharmacy_SMA'] = df.iloc[:,3].rolling(window=WDW).mean()
df['parks_SMA'] = df.iloc[:,4].rolling(window=WDW).mean()
df['transit_stations_SMA'] = df.iloc[:,5].rolling(window=WDW).mean()
df['workplaces_SMA'] = df.iloc[:,6].rolling(window=WDW).mean()
df['residential_SMA'] = df.iloc[:,7].rolling(window=WDW).mean()
df['Rt_avg_SMA'] = df.iloc[:,9].rolling(window=WDW).mean()
# url = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'

# remove unneccessary columns
df = df.drop(columns=['retail_and_recreation'],axis=1)
df = df.drop(columns=['grocery_and_pharmacy'],axis=1)
df = df.drop(columns=['parks'],axis=1)
df = df.drop(columns=['transit_stations'],axis=1)
df = df.drop(columns=['workplaces'],axis=1)
df = df.drop(columns=['residential'],axis=1)
df = df.drop(columns=['Rt_low'],axis=1)
df = df.drop(columns=['Rt_up'],axis=1)

# CALCULATE CORRELATION
corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()


# SHOW A GRAPH IN TIME
fig1x = plt.figure()
ax = fig1x.add_subplot(111, facecolor='#dddddd', axisbelow=True)
df["workplaces_SMA"].plot(ax=ax, x_compat=True)
plt.legend(loc='best')
ax.set_xticklabels(df.date)
titlex = ('G&P')
configgraph(titlex)
plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
plt.show()