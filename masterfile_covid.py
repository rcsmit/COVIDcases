# 27/28 feb 2021
# Calculate the relation between gliding R and mobility (Apple and Google)
# Calculate the corelation with hospital admissions and factors mentioned above
# Plotting a heatmap with correlations
# Plotting a scattermap
# Plotting a graph in time, with an adjusted x-

# 1 maart 2021
# Merging files on date in different date formats
# Remove outliers (doesnt work)
# Calculating moving avarages
# Make different statistics for weekdays and weekend
# Scraping statistics from RIVM

# 2 maart
# R van ziekenhuisopnames
# weekgrafiek
# corrigeren vd merge functie 

# 3 maart
# downloaden en mergen hospital admission
# downloaden en mergen r-getal RIVM
# alles omgezet in functies



# I used iloc.  Iterating through pandas objects is generally slow. 
# In many cases, iterating manually over the rows is not needed and 
# can be avoided with one of the following approaches:
# http://pandas-docs.github.io/pandas-docs-travis/getting_started/basics.html#iteration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sn
from scipy import stats
from datetime import datetime
import datetime as dt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

_lock = RendererAgg.lock
from scipy.signal import savgol_filter

import urllib
import urllib.request
from pathlib import Path

from inspect import currentframe, getframeinfo
global log 
log = False
# R-numbers from 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
# Google mobilty from https://www.google.com/covid19/mobility/?hl=nl
# Apple mobility from https://covid19.apple.com/mobility
# # Merged in one file in Excel and saved to CSV
# Hospitals from RIVM 'https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv

 
    
def download_mobR():


    df_mobR = pd.read_csv(
                    r'covid19_seir_models\mobilityR.csv',
                    delimiter=';',
                    low_memory=False
                )

    # datum is 16-2-2020
    df_mobR['date']=pd.to_datetime(df_mobR['date'], format="%d-%m-%Y")
                     
    df_mobR.set_index('date')

    # SMOOTH ROLLING AVERAGES
    # 7 days average
    WDW = 3
    df_mobR['Rt_avg_SMA'] =df_mobR.iloc[:,9].rolling(window=WDW).mean()
    # df['retail_and_recreation_SMA'] = df.iloc[:,2].rolling(window=WDW).mean()
    # df['grocery_and_pharmacy_SMA'] = df.iloc[:,3].rolling(window=WDW).mean()
    # df['parks_SMA'] = df.iloc[:,4].rolling(window=WDW).mean()
    df_mobR['transit_stations_SMA'] = df_mobR.iloc[:,5].rolling(window=WDW).mean()
    # df['workplaces_SMA'] = df.iloc[:,6].rolling(window=WDW).mean()
    # df['residential_SMA'] = df.iloc[:,7].rolling(window=WDW).mean()
    # #df['Rt_mvd_SMA'] = df.iloc[:,15].rolling(window=WDW).mean()
    df_mobR['apple_driving_SMA'] = df_mobR.iloc[:,12].rolling(window=WDW).mean()
    # df['apple_transit_SMA'] = df.iloc[:,13].rolling(window=WDW).mean()
    # #df['apple_walking_SMA'] = df.iloc[:,14].rolling(window=WDW).mean()
    
    return df_mobR
def download_hospital_admissions():
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

    df_hospital = pd.read_csv(
                    r'covid19_seir_models\COVID-19_ziekenhuisopnames.csv',
                    delimiter=';',
                    #delimiter=',',
                    low_memory=False
                    
                )

    # datum is 2020-02-27
    df_hospital['Date_of_statistics'] = df_hospital['Date_of_statistics'].astype('datetime64[D]')

    df_hospital = df_hospital.groupby(["Date_of_statistics"], sort=True).sum().reset_index()

    # compression_opts = dict(method=None,
    #                         archive_name='out.csv')  
    # df_hospital.to_csv('outhospital.csv', index=False,
    #         compression=compression_opts)
    WDW2 = 7
    #print (df_hospital)
    df_hospital['hosp_adm_notif_SMA'] = df_hospital.iloc[:,2].rolling(window=WDW2).mean()
    df_hospital['hosp_admission_SMA'] = df_hospital.iloc[:,3].rolling(window=WDW2).mean()
    df_hospital['hosp_admission_savgol'] = df_hospital['Hospital_admission'].transform(lambda x: savgol_filter(x, 7,2))
    return df_hospital
def download_lcps():

    download = False
    if download == True :
        # Code by Han-Kwang Nienhuys - MIT License
        url='https://lcps.nu/wp-content/uploads/covid-19.csv'
        #url="https://lcps.nu/wp-content/uploads/covid-19.csv"
        fpath = Path('covid19_seir_models\LCPS.csv')
        print(f'Getting new daily case statistics file LCPS...')
        with urllib.request.urlopen(url) as response:
            data_bytes = response.read()
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')

    df_lcps = pd.read_csv(
                    r'covid19_seir_models\LCPS.csv',
                    delimiter=',',
                    #delimiter=',',
                    low_memory=False
                    
                )
    print (df_lcps)
    print (df_lcps.dtypes)
    # Datum,IC_Bedden_COVID,IC_Bedden_Non_COVID,Kliniek_Bedden,IC_Nieuwe_Opnames_COVID,Kliniek_Nieuwe_Opnames_COVID
    # datum is 2020-02-27
    df_lcps['Datum']=pd.to_datetime(df_lcps['Datum'], format="%d-%m-%Y")

    #df_lcps = df_lcps.groupby(["Datum"], sort=True).sum().reset_index()

    # compression_opts = dict(method=None,
    #                         archive_name='out.csv')  
    # df_hospital.to_csv('outhospital.csv', index=False,
    #         compression=compression_opts)
    WDW2 = 7
    print (df_lcps)
    print (df_lcps.dtypes)
    return df_lcps

def download_reproductiegetal():
    #https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json
    download = True
    if download == True:
        df_reprogetal = pd.read_json (r'covid19_seir_models\COVID-19_reproductiegetal.json')
        # url = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
        
        compression_opts = dict(method=None,
                            archive_name='reprogetal.csv')  
        df_reprogetal.to_csv('covid19_seir_models\\reprogetal.csv', index=False,
            compression=compression_opts)
        df_reprogetal.set_index("Date")
    else:
        df_reprogetal = pd.read_csv(
                    r'covid19_seir_models\reprogetal.csv',
                    delimiter=',',
                    #delimiter=',',
                    low_memory=False)
    print (df_reprogetal.dtypes)
    return df_reprogetal

    
def download_testen():
    download = False
    if download == True :
        # Code by Han-Kwang Nienhuys - MIT License
        url='https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv'
       
        # Het werkelijke aantal COVID-19 patiënten opgenomen in het ziekenhuis is hoger dan 
        # het aantal opgenomen patiënten gemeld in de surveillance, omdat de GGD niet altijd op 
        # de hoogte is van ziekenhuisopname als deze na melding plaatsvindt.
        # Daarom benoemt het RIVM sinds 6 oktober actief de geregistreerde ziekenhuisopnames 
        # van Stichting NICE
       
        #url="https://lcps.nu/wp-content/uploads/covid-19.csv"

        fpath = Path('covid19_seir_models\COVID-19_aantallen_gemeente_per_dag.csv')
        print(f'Getting new daily case statistics file...')
        with urllib.request.urlopen(url) as response:
            data_bytes = response.read()
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')

    df_testen = pd.read_csv(
                    r'covid19_seir_models\COVID-19_aantallen_gemeente_per_dag.csv',
                    delimiter=';',
                    #delimiter=',',
                    low_memory=False)

    df_testen['Date_of_publication'] = df_testen['Date_of_publication'].astype('datetime64[D]')

    df_testen = df_testen.groupby(["Date_of_publication"], sort=True).sum().reset_index()

    return df_testen
def splitupweekweekend(df):
    # SPLIT UP IN WEEKDAY AND WEEKEND
    # https://stackoverflow.com/posts/56336718/revisions
    df['WEEKDAY'] = pd.to_datetime(df['date']).dt.dayofweek  # monday = 0, sunday = 6
    df['weekend'] = 0          # Initialize the column with default value of 0
    df.loc[df['WEEKDAY'].isin([5, 6]), 'weekend'] = 1  # 5 and 6 correspond to Sat and Sun
    return(df)


    # remove outliers - doesnt work
    # df = df[(np.abs(stats.zscore(df['retail_and_recreation'])) < 3)]
    # df = df[(np.abs(stats.zscore(df['transit_stations'])) < 3)]
    # df = df[(np.abs(stats.zscore(df['workplaces'])) < 3)]
    # df = df[(np.abs(stats.zscore(df['grocery_and_pharmacy'])) < 3)]    
    

    
    
def walkingR(df):
    #print(df)
    # Calculate walking R from hosp admission
    slidingRdf= pd.DataFrame({'date_sR': [],
            'Rvalue_from_hosp_adm_SMA': [],
            'Rvalue_from_hosp_adm_notif_SMA': []})
    Tg = 4
    d= 7
    for i in range(0, len(df)):
        if df.iloc[i]['hosp_admission_SMA'] != None:
            date_ = pd.to_datetime(df.iloc[i]['Date_of_statistics'], format="%Y-%m-%d")
            date_ = df.iloc[i]['Date_of_statistics']
            
            slidingR_= round(((df.iloc[i]['hosp_admission_SMA']/df.iloc[i-d]['hosp_admission_SMA'])**(Tg/d) ),2)  
            slidingRnotif_= round(((df.iloc[i]['hosp_adm_notif_SMA']/df.iloc[i-d]['hosp_adm_notif_SMA'])**(Tg/d) ),2)  
            
            slidingRdf=slidingRdf.append({'date_sR':date_,
            
            'Rvalue_from_hosp_adm_SMA':slidingR_,
            'Rvalue_from_hosp_adm_notif_SMA':slidingRnotif_
            },ignore_index=True)



    slidingRdf['Rt_hosp_adm_SMA_SMA'] = round(slidingRdf.iloc[:,1].rolling(window=7).mean(),2)
    slidingRdf['Rt_hosp_adm_notif_SMA_SMA'] = round(slidingRdf.iloc[:,2].rolling(window=7).mean(),2)
    


    df = df.reset_index()
    slidingRdf = slidingRdf.reset_index()
    return slidingRdf

def agg_week(df):

    df.loc[df["date"].isnull(),'date'] = df["Date_of_statistics"] 


    df["weeknr"] =  df['date'].dt.isocalendar().week
    df["yearnr"] =  df['date'].dt.isocalendar().year

    df["weekalt"]  = df['date'].dt.isocalendar().year.astype(str) + "-"+ df['date'].dt.isocalendar().week.astype(str)

    dfweek = df.groupby("weekalt", sort=False).mean()
    # dfd = df.copy()
    # df.iloc[i]['week_alt']=str(dfd.iloc[i]["yearnr"]) + '-' + str(dfd.iloc[i]["weeknr"]).astype(str)

    # for i in range(0, len(df)):    
    #     if df.iloc[i]["weeknr"]<9:
    #         df.iloc[i]['week_alt']=df.iloc[i]["yearnr"].astype(str) + '-0' + df.iloc[i]["weeknr"].astype(str)
    #     else:
    #         df.iloc[i]['week_alt']=df.iloc[i]["yearnr"].astype(str) + '-' + df.iloc[i]["weeknr"].astype(str)

    return dfweek
def last_manipulations(df):
    
    # compression_opts = dict(method=None,
    #                         archive_name='outweek.csv')  
    # dfweek.to_csv('outweek2.csv', index=False,
    #         compression=compression_opts)


    # compression_opts = dict(method=None,
    #                         archive_name='outalles.csv')  
    # df.to_csv('outalles.csv', index=False,
    #         compression=compression_opts)


    # remove unneccessary columns
    #df = df.drop(columns=['retail_and_recreation'],axis=1)
    #df = df.drop(columns=['grocery_and_pharmacy'],axis=1)
    df = df.drop(columns=['parks'],axis=1)
    #df = df.drop(columns=['transit_stations'],axis=1)
    #df = df.drop(columns=['workplaces'],axis=1)
    df = df.drop(columns=['residential'],axis=1)
    # df = df.drop(columns=['Rt_low'],axis=1)
    # df = df.drop(columns=['Rt_up'],axis=1)
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
    df[name] = df['Rt_avg_SMA'].shift(r)

    mask = (df['date'] > '2020-12-31') & (df['date'] <= '2021-6-1')
    df = (df.loc[mask])

    df = df.reset_index()
    #df = df[:-2] #drop last row
    print (df)

    return df, werkdagen, weekend_
def save_df(df,name):
    name_ = name+'.csv'
    compression_opts = dict(method=None,
                            archive_name=name_)  
    df.to_csv(name_, index=False,
            compression=compression_opts)
def get_data():

    df_hospital = download_hospital_admissions()
    slidingRdf = walkingR(df_hospital)
    df_lcps = download_lcps()
    df_mobR = download_mobR()  
    df_testen = download_testen()
    df_reprogetal = download_reproductiegetal()

    
    df = pd.merge(df_mobR, df_hospital, how='outer', left_on = 'date', right_on="Date_of_statistics")
    df.loc[df["date"].isnull(),'date'] = df["Date_of_statistics"] 
    df = pd.merge(df, df_lcps, how='outer', left_on = 'date', right_on="Datum")
    df.loc[df["date"].isnull(),'date'] = df["Datum"] 
    df = pd.merge(df, slidingRdf, how='outer', left_on = 'date', right_on="date_sR", left_index=True )
    
    df = pd.merge(df, df_testen, how='outer', left_on = 'date', right_on="Date_of_publication", left_index=True )
    df = pd.merge(df, df_reprogetal, how='outer', left_on = 'date', right_on="Date", left_index=True )
    
    
    df = df.sort_values(by=['date'])
    df = splitupweekweekend(df)

    df, werkdagen, weekend_ = last_manipulations(df)

    dfweek = agg_week(df)
    print (df)
    print (df.dtypes)
    save_df(df, "algemeen")
    
    return df, dfweek, werkdagen, weekend_
def correlation_matrix(df, werkdagen, weekend_):
    print("x")
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

    sn.regplot(y="Rt_avg_x", x="Hospital_admission_x", data=df)
    plt.show()
def graph_day(df):

    # SHOW A GRAPH IN TIME / DAY
    fig1x = plt.figure()
    ax = fig1x.add_subplot(111)
    #print(df["Rt_hosp_adm_SMA_SMA"])
    #plt.title('Transit_stations compared to gliding Rt-number for the Netherlands')
    #plt.title('Hospital admissions (smoothed)/  sliding R hosp adm (smoothed')
    
    #df["hosp_admission_SMA"].plot(label="hospital admission SMA")

    #df["hosp_admission_SMA"].plot(color="blue", label="hospital admission SMA")
    #df["hosp_admission_savgol"].plot(color = "red", label="hospital admission Savgol")
    df["IC_Nieuwe_Opnames_COVID"].plot(color= "yellow", label="IC admissionX")
    df["Hospital_admission_x"].plot(color= "purple", label="Hospital admission")
    #df["transit_stations_SMA"].plot(label="Transit_stations_SMA")
    #df["apple_driving_SMA"].plot(label="Apple driving smoothened")
    ax3=df["Rt_avg_x"].plot(secondary_y=True, label="Gliding Rt RIVM")
    #ax3=df["Rt_moved10"].plot(secondary_y=True, label="Gliding Rt RIVM Moved 10 days")


    #ax3=df["Rt_hosp_adm_SMA_SMA"].plot(secondary_y=True, color = 'r', label="Gliding Rt from hospadm")
    #ax3=df["Rt_hosp_adm_notif_SMA_SMA"].plot(secondary_y=True, color = 'g', label="Gliding Rt from hospadm notif")



    ax3.set_ylabel('Rt')
    ax.set_ylabel('Numbers')
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    #ax.xaxis.set_major_formatter()

    # layout of the x-axis

    
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_major_locator(MultipleLocator(1))
   
    ax.set_xticklabels(df["date"].dt.date,fontsize=6, rotation=45)
    xticks = ax.xaxis.get_major_ticks()

    for i,tick in enumerate(xticks):
        if i%10 != 0:
            tick.label1.set_visible(False)
    plt.xticks()
    plt.legend(loc='best')


    
    

    plt.xlabel('date')
    
    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

     #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='best', prop=fontP)
    titlex = "____"
    plt.title(titlex , fontsize=10)

    #ax.xaxis.set_major_formatter()

    # everything in legend
    # https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
    handles,labels = [],[]
    for ax in fig1x.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)

    plt.legend(handles,labels)

    titlex = "Hospital admissions (unedited/smoothed/Savgol)"

    
    # configgraph(titlex)
    plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')



    # Add restrictions
    # From Han-Kwang Nienhuys - MIT-licence
    
    df_restrictions = pd.read_csv(
                    r'covid19_seir_models\restrictions.csv',
                    delimiter=',',
                    low_memory=False
                    
            )
    a = ((df["date"].dt.date.min()))
    #print (a)
    # b_ ="2021-01-22"
    # #b = datetime.today().strftime('%m/%d/%Y')
    ymin, ymax = ax.get_ylim()
    y_lab = ymin
    for i in range(0, len(df_restrictions)):    
        d_ = df_restrictions.iloc[i]['Date'] #string
        d__ = dt.datetime.strptime(d_,'%Y-%m-%d').date()  # to dateday

        diff = (d__ - a)
        #print (diff)
        if diff.days >0 :
            #print (diff.days)
            ax.text(diff.days, y_lab, f'  {df_restrictions.iloc[i]["Description"]}', rotation=90, fontsize=6,horizontalalignment='center')
            plt.axvline(x=diff.days, color='yellow', alpha=.3,linestyle='--')
    

    plt.show()
   
def graph_week(dfweek):
        
    # SHOW A GRAPH IN TIME / WEEK
    fig1x = plt.figure()
    ax = fig1x.add_subplot(111)

    ax.set_xticklabels(dfweek["weeknr"],fontsize=6, rotation=45)
    #print(df["Rt_hosp_adm_SMA_SMA"])
    #plt.title('Transit_stations compared to gliding Rt-number for the Netherlands')
    #plt.title('Hospital admissions (unedited/smoothed/Savgol)'
    # /  sliding R hosp adm (smoothed')
    dfweek["hosp_admission_SMA"].plot(label="hospital admission SMA")
    #df["Hospital_admission"].plot.bar(label="hospital admission")
    #df["transit_stations_SMA"].plot(label="Transit_stations_SMA")
    dfweek["apple_driving_SMA"].plot(label="Apple driving smoothened")
    #ax3=df["Rt_avg"].plot(secondary_y=True, label="Gliding Rt RIVM")
    #ax3=df["Rt_moved10"].plot(secondary_y=True, label="Gliding Rt RIVM Moved 10 days")
    ax3=dfweek["Rt_hosp_adm_SMA_SMA"].plot(secondary_y=True, color = 'r', label="Gliding Rt from hospadm")
    #print (dfweek["Rt_hosp_adm_SMA_SMA"])
    #ax3=df[name].plot(secondary_y=True, label=name)
    ax3.set_ylabel('Rt')
    ax.set_ylabel('Numbers')
    
    titlex = "Hospital admissions (week)"
    plt.xlabel('date')
    
    # Add a grid
    plt.grid(alpha=.4,linestyle='--')

    #Add a Legend
    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend(  loc='best', prop=fontP)
    plt.title(titlex , fontsize=10)

    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    #ax.xaxis.set_major_formatter()
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
def main():
    

    
    df, dfweek, werkdagen, weekend_ = get_data()
    
    #correlation_matrix(df,werkdagen, weekend_)
    graph_day(df)
    #graph_week(dfweek)

main()

# https://www.medrxiv.org/content/10.1101/2020.05.06.20093039v3.full
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7729173/

