# Calculate the relation between gliding R and mobility (Apple and Google)
# Calculate the corelation with hospital admissions and factors mentioned above
# Merging files on date in different date formats
# Remove outliers (doesnt work)
# Calculating moving avarages
# Make different statistics for weekdays and weekend
# Scraping statistics from RIVM
# Plotting a heatmap with correlations
# Plotting a scattermap
# Plotting a graph in time, with an adjusted x-axis


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sn
from scipy import stats
from datetime import datetime
    
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

_lock = RendererAgg.lock


import urllib
import urllib.request
from pathlib import Path

def configgraph(titlex):
    interval_ = 20
    plt.xlabel('date')
    
    # Add a grid
    plt.grid(alpha=.4,linestyle='--')


     #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='best', prop=fontP)
    plt.title(titlex , fontsize=10)

    # lay-out of the x axis
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval_))
    # plt.gcf().autofmt_xdate()

    # ax.set_xticklabels(df["date"], rotation=0)
    # xticks = ax.xaxis.get_major_ticks()
    # for i,tick in enumerate(xticks):
    #     if i%5 != 0:
    #         tick.label1.set_visible(False)
    plt.gca().set_title(titlex , fontsize=10)


# R-numbers from 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
# Google mobilty from https://www.google.com/covid19/mobility/?hl=nl
# Apple mobility from https://covid19.apple.com/mobility
# # Merged in one file in Excel and saved to CSV
# Hospitals from RIVM 'https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv


df_mobR = pd.read_csv(
                r'covid19_seir_models\mobilityR.csv',
                delimiter=';',
                low_memory=False
            )

# datum is 16-2-2020
df_mobR['date']=pd.to_datetime(df_mobR['date'], format="%d-%m-%Y")
print (df_mobR.dtypes)

                
df_mobR.set_index('date') 
print (df_mobR)

######################

download = False
if download == True :
    # Code by Han-Kwang Nienhuys - MIT License
    url='https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv'
    #url="https://lcps.nu/wp-content/uploads/covid-19.csv"
    fpath = Path('covid19_seir_models\COVID-19_ziekenhuisopnames.csv')
    print(f'Getting new daily case statistics file...')
    with urllib.request.urlopen(url) as response:
        data_bytes = response.read()
        fpath.write_bytes(data_bytes)
        print(f'Wrote {fpath} .')

df_ziekenhuis = pd.read_csv(
                r'covid19_seir_models\COVID-19_ziekenhuisopnames.csv',
                delimiter=';',
                #delimiter=',',
                low_memory=False
                
            )

# datum is 2020-02-27
df_ziekenhuis['Date_of_statistics'] = df_ziekenhuis['Date_of_statistics'].astype('datetime64[D]')

df_ziekenhuis = df_ziekenhuis.groupby(["Date_of_statistics"], sort=True).sum().reset_index()


compression_opts = dict(method=None,
                        archive_name='out.csv')  
df_ziekenhuis.to_csv('outhospital.csv', index=False,
          compression=compression_opts)


df = pd.merge(df_mobR, df_ziekenhuis, how='outer', left_on = 'date', right_on="Date_of_statistics")


# SPLIT UP IN WEEKDAY AND WEEKEND
# https://stackoverflow.com/posts/56336718/revisions
df['WEEKDAY'] = pd.to_datetime(df['date']).dt.dayofweek  # monday = 0, sunday = 6
df['weekend'] = 0          # Initialize the column with default value of 0
df.loc[df['WEEKDAY'].isin([5, 6]), 'weekend'] = 1  # 5 and 6 correspond to Sat and Sun



# remove outliers - doesnt work
# df = df[(np.abs(stats.zscore(df['retail_and_recreation'])) < 3)]
# df = df[(np.abs(stats.zscore(df['transit_stations'])) < 3)]
# df = df[(np.abs(stats.zscore(df['workplaces'])) < 3)]
# df = df[(np.abs(stats.zscore(df['grocery_and_pharmacy'])) < 3)]    

# SMOOTH ROLLING AVERAGES
# 7 days average
WDW = 3
WDW2=3
df['Rt_avg_SMA'] = df.iloc[:,9].rolling(window=WDW).mean()
# df['retail_and_recreation_SMA'] = df.iloc[:,2].rolling(window=WDW).mean()
# df['grocery_and_pharmacy_SMA'] = df.iloc[:,3].rolling(window=WDW).mean()
# df['parks_SMA'] = df.iloc[:,4].rolling(window=WDW).mean()
df['transit_stations_SMA'] = df.iloc[:,5].rolling(window=WDW).mean()
# df['workplaces_SMA'] = df.iloc[:,6].rolling(window=WDW).mean()
# df['residential_SMA'] = df.iloc[:,7].rolling(window=WDW).mean()
# #df['Rt_mvd_SMA'] = df.iloc[:,15].rolling(window=WDW).mean()
# #df['apple_driving_SMA'] = df.iloc[:,12].rolling(window=WDW).mean()
# df['apple_transit_SMA'] = df.iloc[:,13].rolling(window=WDW).mean()
# #df['apple_walking_SMA'] = df.iloc[:,14].rolling(window=WDW).mean()
# df['hosp_adm_notif_SMA'] = df.iloc[:,17].rolling(window=WDW2).mean()
# df['hosp_admission_SMA'] = df.iloc[:,18].rolling(window=WDW2).mean()

# remove unneccessary columns
#df = df.drop(columns=['retail_and_recreation'],axis=1)
#df = df.drop(columns=['grocery_and_pharmacy'],axis=1)
df = df.drop(columns=['parks'],axis=1)
#df = df.drop(columns=['transit_stations'],axis=1)
#df = df.drop(columns=['workplaces'],axis=1)
df = df.drop(columns=['residential'],axis=1)
df = df.drop(columns=['Rt_low'],axis=1)
df = df.drop(columns=['Rt_up'],axis=1)
# df = df.drop(columns=['apple_driving'],axis=1)
# df = df.drop(columns=['apple_transit'],axis=1)
# df = df.drop(columns=['apple_walking'],axis=1)
df = df.drop(columns=['Version'],axis=1)

# Two different dataframes for workdays/weekend
werkdagen = df.loc[(df['weekend'] == 0)] 
weekend_ = df.loc[(df['weekend'] == 1) ]
df = df.drop(columns=['WEEKDAY'],axis=1)
df = df.drop(columns=['weekend'],axis=1)
werkdagen = werkdagen.drop(columns=['WEEKDAY'],axis=1)
werkdagen = werkdagen.drop(columns=['weekend'],axis=1)
weekend_ = weekend_.drop(columns=['WEEKDAY'],axis=1)
weekend_ = weekend_.drop(columns=['weekend'],axis=1)


# #move Rt r days, because a maybe change has effect after r days
# Tested - Conclusion : It has no effect
# for r in range(0,8,2):
#     name = "Rt_moved" + str(r)
#     df[name] = df['Rt_avg_SMA'].shift(r)
r=10
name = "Rt_moved" + str(r)
#df[name] = df['Rt_avg_SMA'].shift(r)

print (df.dtypes)

compression_opts = dict(method=None,
                        archive_name='outalles.csv')  
df.to_csv('outalles.csv', index=False,
          compression=compression_opts)


# CALCULATE CORRELATION

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True, annot_kws={"fontsize":7})
plt.title("ALL DAYS", fontsize =20)
plt.show()


corrMatrix = werkdagen.corr()
sn.heatmap(corrMatrix, annot=True)
plt.title("WORKING DAYS", fontsize =20)
plt.show()

corrMatrix = weekend_.corr()
sn.heatmap(corrMatrix, annot=True)
plt.title("WEEKEND", fontsize =20)
plt.show()

#MAKE A SCATTERPLOT

sn.regplot(y="Rt_avg", x="Hospital_admission", data=df)
plt.show()


# SHOW A GRAPH IN TIME
fig1x = plt.figure()
ax = fig1x.add_subplot(111)

plt.title('Transit_stations compared to gliding Rt-number for the Netherlands')
#df["hosp_admission_SMA"].plot(label="hospital admission SMA")
#df["Hospital_admission"].plot.bar(label="hospital admission")
df["transit_stations_SMA"].plot(label="Transit_stations_SMA")
ax3=df["Rt_avg"].plot(secondary_y=True, label="Gliding Rt")
#ax3=df[name].plot(secondary_y=True, label=name)
ax3.set_ylabel('Rt')
ax.set_ylabel('Numbers')
ax.xaxis.grid(True, which='minor')
ax.xaxis.set_major_locator(MultipleLocator(1))
#ax.xaxis.set_major_formatter()

# layout of the x-axis
ax.set_xticklabels(df["date"].dt.date,fontsize=6, rotation=45)
xticks = ax.xaxis.get_major_ticks()

for i,tick in enumerate(xticks):
    if i%10 != 0:
        tick.label1.set_visible(False)
plt.xticks()
plt.legend(loc='best')

# everything in legend
# https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
handles,labels = [],[]
for ax in fig1x.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.legend(handles,labels)

# titlex = ('_')
# configgraph(titlex)
plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
plt.show()

# https://www.medrxiv.org/content/10.1101/2020.05.06.20093039v3.full
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7729173/