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
                parse_dates=["date"],
                low_memory=False
            )
print (df.dtypes)


# SPLIT UP IN WEEKDAY AND WEEKEND
# https://stackoverflow.com/posts/56336718/revisions
df['WEEKDAY'] = pd.to_datetime(df['date']).dt.dayofweek  # monday = 0, sunday = 6
df['weekend'] = 0          # Initialize the column with default value of 0
df.loc[df['WEEKDAY'].isin([5, 6]), 'weekend'] = 1  # 5 and 6 correspond to Sat and Sun



# remove outliers - NOTHING WORKS

# for r in range(2,7):
#     print (r)
#     df = df[(np.abs(stats.zscore(df[r])) < 3)]
# #df[np.abs(df.grocery_and_pharmacy.mean()) <= (3*df.grocery_and_pharmacy.std())]
# keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
#df[((df.grocery_and_pharmacy - df.grocery_and_pharmacy.mean()) / df.grocery_and_pharmacy.std()).abs() < 1]

# SMOOTH ROLLING AVERAGES
# 7 days average
WDW = 7
df['retail_and_recreation_SMA'] = df.iloc[:,2].rolling(window=WDW).mean()
df['grocery_and_pharmacy_SMA'] = df.iloc[:,3].rolling(window=WDW).mean()
df['parks_SMA'] = df.iloc[:,4].rolling(window=WDW).mean()
df['transit_stations_SMA'] = df.iloc[:,5].rolling(window=WDW).mean()
df['workplaces_SMA'] = df.iloc[:,6].rolling(window=WDW).mean()
df['residential_SMA'] = df.iloc[:,7].rolling(window=WDW).mean()
df['Rt_avg_SMA'] = df.iloc[:,9].rolling(window=WDW).mean()
df['Rt_mvd_SMA'] = df.iloc[:,15].rolling(window=WDW).mean()
df['apple_driving_SMA'] = df.iloc[:,12].rolling(window=WDW).mean()
df['apple_transit_SMA'] = df.iloc[:,13].rolling(window=WDW).mean()
df['apple_walking_SMA'] = df.iloc[:,14].rolling(window=WDW).mean()


# remove unneccessary columns
#df = df.drop(columns=['retail_and_recreation'],axis=1)
#df = df.drop(columns=['grocery_and_pharmacy'],axis=1)
df = df.drop(columns=['parks'],axis=1)
#df = df.drop(columns=['transit_stations'],axis=1)
#df = df.drop(columns=['workplaces'],axis=1)
df = df.drop(columns=['residential'],axis=1)
df = df.drop(columns=['Rt_low'],axis=1)
df = df.drop(columns=['Rt_up'],axis=1)
df = df.drop(columns=['apple_driving'],axis=1)
df = df.drop(columns=['apple_transit'],axis=1)
df = df.drop(columns=['apple_walking'],axis=1)

# Two different dataframes for workdays/weekend
werkdagen = df.loc[(df['weekend'] == 0)] 
weekend_ = df.loc[(df['weekend'] == 1) ]

# #move Rt r days, because a maybe change has effect after r days
# Tested - Conclusion : It has no effect
# for r in range(1,11,2):
#     name = "Rt_moved" + str(r)
#     df[name] = df['Rt_avg_SMA'].shift(r)

#print (df.dtypes)
# CALCULATE CORRELATION
corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()


corrMatrix = werkdagen.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

corrMatrix = weekend_.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

# MAKE A SCATTERPLOT
#sn.regplot(y="Rt_avg_SMA", x="transit_stations_SMA", data=df)
#plt.show()


# SHOW A GRAPH IN TIME
# fig1x = plt.figure()
# ax = fig1x.add_subplot(111, facecolor='#dddddd', axisbelow=True)
# df["workplaces_SMA"].plot(ax=ax, x_compat=True)
# plt.legend(loc='best')
# ax.set_xticklabels(df.date)
# titlex = ('G&P')
# configgraph(titlex)
# plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
# plt.show()




# https://www.medrxiv.org/content/10.1101/2020.05.06.20093039v3.full
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7729173/
