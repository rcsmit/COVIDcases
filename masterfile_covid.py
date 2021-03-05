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
# added restrictions (file van @HK_nien, MIT-licence)
# downloaden en mergen hospital admission
# downloaden en mergen r-getal RIVM
# alles omgezet in functies

# 4 maart
# meer onderverdeling in functies. Alles aan te roepen vanuit main() met parameters

# 5 maart
# custom colors
# weekend different color in barplot
# annoying problem met een join (van outer naar inner naar outer en toen werkte het weer)

# I used iloc.  Iterating through pandas objects is generally slow. 
# In many cases, iterating manually over the rows is not needed and 
# can be avoided with one of the following approaches:
# http://pandas-docs.github.io/pandas-docs-travis/getting_started/basics.html#iteration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sn
from scipy import stats
from datetime import datetime
import datetime as dt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker

_lock = RendererAgg.lock
from scipy.signal import savgol_filter

import urllib
import urllib.request
from pathlib import Path

from inspect import currentframe, getframeinfo

# R-numbers from 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
# Google mobilty from https://www.google.com/covid19/mobility/?hl=nl
# Apple mobility from https://covid19.apple.com/mobility
# # Merged in one file in Excel and saved to CSV
# Hospitals from RIVM 'https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv

 
    
def download_mobR():
    """  _ _ _ """
    df_mobR = pd.read_csv(
                    r'covid19_seir_models\mobility.csv',
                    delimiter=';',
                    low_memory=False
                )
    # datum is 16-2-2020
    df_mobR['date']=pd.to_datetime(df_mobR['date'], format="%d-%m-%Y")                   
    df_mobR.set_index('date')
    return df_mobR

def download_hospital_admissions():
    """  _ _ _ """
    if download == True :
        # Code by Han-Kwang Nienhuys - MIT License
        url='https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv'
        #url="https://lcps.nu/wp-content/uploads/covid-19.csv"
        fpath = Path('covid19_seir_models\COVID-19_ziekenhuisopnames.csv')
        print(f'Getting new daily case statistics file ziekenhuisopnames. ..')
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
    
    #print (df_hospital)
    
    return df_hospital

def download_lcps():
    """Download data from LCPS"""
    
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
    # print (df_lcps)
    # print (df_lcps.dtypes)
    # Datum,IC_Bedden_COVID,IC_Bedden_Non_COVID,Kliniek_Bedden,IC_Nieuwe_Opnames_COVID,Kliniek_Nieuwe_Opnames_COVID
    # datum is 2020-02-27
    df_lcps['Datum']=pd.to_datetime(df_lcps['Datum'], format="%d-%m-%Y")
 
    #df_lcps = df_lcps.groupby(["Datum"], sort=True).sum().reset_index()

    # compression_opts = dict(method=None,
    #                         archive_name='out.csv')  
    # df_hospital.to_csv('outhospital.csv', index=False,
    #         compression=compression_opts)
    
    return df_lcps

def download_reproductiegetal():
    """  _ _ _ """
    #https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json
    
    if download == True:
        print ("Download reproductiegetal-file")
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

                    
    df_reprogetal['Date']=pd.to_datetime(df_reprogetal['Date'], format="%Y-%m-%d")
    
    #print (df_reprogetal.dtypes)
    return df_reprogetal
    
def download_testen():
    """  _ _ _ """
   
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
        print(f'Getting new daily case statistics file - testen - ...')
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
    save_df(df_testen,"COVID-19_aantallen_per_dag")
    return df_testen
###################################################################
def get_data():
    """  _ _ _ """

    df_hospital = download_hospital_admissions()
    #slidingRdf = walkingR(df_hospital, "Hospital_admission")
    df_lcps = download_lcps()
    df_mobR = download_mobR()  
    df_testen = download_testen()
    df_reprogetal = download_reproductiegetal()

    
    df = pd.merge(df_mobR, df_hospital, how='outer', left_on = 'date', right_on="Date_of_statistics")
    df.loc[df["date"].isnull(),'date'] = df["Date_of_statistics"] 
    df = pd.merge(df, df_lcps, how='outer', left_on = 'date', right_on="Datum")
    df.loc[df["date"].isnull(),'date'] = df["Datum"] 
    #df = pd.merge(df, slidingRdf, how='outer', left_on = 'date', right_on="date_sR", left_index=True )
    
    df = pd.merge(df, df_testen, how='outer', left_on = 'date', right_on="Date_of_publication", left_index=True )
    df = pd.merge(df, df_reprogetal, how='outer', left_on = 'date', right_on="Date", left_index=True )
    
    
    df = df.sort_values(by=['date'])
    df = splitupweekweekend(df)
    # last_manipulations(df, what_to_drop, show_from, show_until,drop_last):
    df, werkdagen, weekend_ = last_manipulations(df, None,FROM, UNTIL, None)

    # prepare an aggregated weektable
    df, new_column = add_SMA(df,"Rt_avg")  #added for the weekgraph, can be deleted
    dfweek = agg_week(df)

    # for debug purposes
    # save_df(df, "algemeen")

    print ("=== AFTER GET DATA ==== ")
    print (df.dtypes)

    
    return df, dfweek, werkdagen, weekend_
  
###################################################
def add_SMA(df,columns):   
    """  _ _ _ """
    if type(columns) == list:
        columns_=columns
    else:
        columns_=[columns]

    for c in columns_:
        
        new_column = c + '_SMA_' + str(WDW2)
        print ('Generating ' + new_column+ '...')
        df[new_column] = df.iloc[:,df.columns.get_loc(c)].rolling(window=WDW2).mean()
        
    return df, new_column

def add_savgol(df,columns): 
    """  _ _ _ """  
    if type(columns) == list:
        columns_=columns
    else:
        columns_=[columns]

    for c in columns_:
        w = 7
        savgol_column = c + '_savgol_' + str(w)
        print ('Generating ' + savgol_column + '...')
        df[savgol_column] = df[c].transform(lambda x: savgol_filter(x, w,2))
    return df, savgol_column


def splitupweekweekend(df):
    """  _ _ _ """
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
        
def add_walkingR(df, base, how_to_smooth):
    """  _ _ _ """
    #print(df)
    # Calculate walking R from a certain base

    column_name_R = 'R_value_from_'+ base 

    if how_to_smooth == "savgol":
        df, new_column = add_savgol(df,base)
        column_name_R_SMA = 'R_value_from_'+ base + '_savgol_' + str(WDW2)
    else:
        df, new_column = add_SMA(df,base)
        column_name_R_SMA = 'R_value_from_'+ base + '_SMA_' + str(WDW2)
    
    base2= new_column
    
    #df[SMA1] = df.iloc[:,df.columns.get_loc(base)].rolling(window=WDW2).mean()
    

    slidingRdf= pd.DataFrame({'date_sR': [],
            column_name_R: []})
    Tg = 4
    d= 1
    
    for i in range(0, len(df)):
        if df.iloc[i][base] != None:
            date_ = pd.to_datetime(df.iloc[i]['Date_of_statistics'], format="%Y-%m-%d")
            date_ = df.iloc[i]['Date_of_statistics']
            if df.iloc[i-d][base2] != 0 or df.iloc[i-d][base2] !=None:
                slidingR_= round(((df.iloc[i][base2]/df.iloc[i-d][base2])**(Tg/d) ),2)  
            else:
                slidingR_ = None
            slidingRdf=slidingRdf.append({'date_sR':date_,
            column_name_R : slidingR_ },ignore_index=True)

    # je zou deze nog kunnen smoothen, maar levert een extra vertraging in de resultaten op, dus wdw=1
    slidingRdf[column_name_R_SMA] = round(slidingRdf.iloc[:,1].rolling(window=1).mean(),2)
    
    # WHY DOES IT CHANGE MY DF[base]. Inner = ok / outer = ERR
    df = pd.merge(df, slidingRdf, how='outer', left_on = 'date', right_on="date_sR", left_index=True )
    
    R_SMA = column_name_R_SMA
    #df = df.reset_index()
    #slidingRdf = slidingRdf.reset_index()
    return df, R_SMA

def agg_week(df):
    """  _ _ _ """

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
    #dfweek = dfweek.sort_values(by=['yearnr','weekalt'])
    return dfweek

def move_column(df, column,days):
    """  _ _ _ """
    # #move Rt r days, because a maybe change has effect after r days
    # Tested - Conclusion : It has no effect
    r=days
    new_column = column + "_moved_" + str(r)
    df[new_column] = df[column].shift(r)
    print ("Name moved column : " + new_column)
    return df

def drop_columns(what_to_drop):
    """  _ _ _ """
    
    if what_to_drop != None:
        print ("dropping " + what_to_drop)
        for d in what_to_drop:
            df = df.drop(columns=['d'],axis=1)

def last_manipulations(df, what_to_drop, show_from, show_until,drop_last):
    """  _ _ _ """
    print ("Doing some last minute manipulations")
    drop_columns(what_to_drop) 


    if show_from == None:
        show_from = '2020-1-1'

    if show_until == None:
        show_until = '2030-1-1'
    mask = (df['date'] >= show_from) & (df['date'] <= show_until)
    df = (df.loc[mask])

    df = df.reset_index()

    # Two different dataframes for workdays/weekend

    werkdagen = df.loc[(df['weekend'] == 0)] 
    weekend_ = df.loc[(df['weekend'] == 1) ]
    #df = df.drop(columns=['WEEKDAY'],axis=1)
    df = df.drop(columns=['weekend'],axis=1)
    werkdagen = werkdagen.drop(columns=['WEEKDAY'],axis=1)
    werkdagen = werkdagen.drop(columns=['weekend'],axis=1)
    weekend_ = weekend_.drop(columns=['WEEKDAY'],axis=1)
    weekend_ = weekend_.drop(columns=['weekend'],axis=1)
       

    if drop_last != None:
        df = df[:drop_last] #drop last row
    print ("=== AFTER LAST MANIPULATIONS ==== ")
    print (df.dtypes)

    return df, werkdagen, weekend_
    #return df

def save_df(df,name):
    """  _ _ _ """
    name_ = name+'.csv'
    compression_opts = dict(method=None,
                            archive_name=name_)  
    df.to_csv(name_, index=False,
            compression=compression_opts)
##########################################################
def correlation_matrix(df, werkdagen, weekend_):
    """  _ _ _ """
    print("x")
    # CALCULATE CORRELATION

    # corrMatrix = df.corr()
    # sn.heatmap(corrMatrix, annot=True, annot_kws={"fontsize":7})
    # plt.title("ALL DAYS", fontsize =20)
    # plt.show()


    # corrMatrix = werkdagen.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # plt.title("WORKING DAYS", fontsize =20)
    # plt.show()

    # corrMatrix = weekend_.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # plt.title("WEEKEND", fontsize =20)
    # plt.show()

    #MAKE A SCATTERPLOT

    #sn.regplot(y="Rt_avg", x="Kliniek_Nieuwe_Opnames_COVID", data=df)
    #plt.show()

def graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title,t):
    """  _ _ _ """
    if type(what_to_show_l) == list:
        what_to_show_l_=what_to_show_l
    else:
        what_to_show_l_=[what_to_show_l]
    aantal = len(what_to_show_l_)
    # SHOW A GRAPH IN TIME / DAY
    fig1x = plt.figure()
    ax = fig1x.add_subplot(111)
    

    bittersweet = "#ff6666"  # reddish 0
    operamauve = "#ac80a0" # purple 1
    green_pigment = "#3fa34d" #green 2
    minion_yellow = "#EAD94C" # yellow 3
    mariagold = "#EFA00B" # orange 4
    falu_red= "#7b2d26" # red 5
    COLOR_weekday = "#3e5c76" # blue 6
    COLOR_weekend = "#e49273" # dark salmon 7
    prusian_blue = "#1D2D44" # 8
    
    color_list = [ operamauve,green_pigment, bittersweet, minion_yellow,mariagold,falu_red]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_list)
    n = 0 
    for a in what_to_show_l_:   
        if type(a) == list:
            a_=a
        else:
            a_=[a] 
        for b in a_: 
                n +=1 
                if t == "bar":         
                    if how_to_smooth == "savgol":
                        df, R_SMA = add_walkingR(df, b, "savgol")
                        column_name_R_savgol = 'R_value_from_'+ b + '_savgol_' + str(WDW2)
                        ax3=df[column_name_R_savgol].plot(secondary_y=True,  label=column_name_R_savgol)
                        c = str(b)+ "_savgol_" + str(WDW2)
                    elif how_to_smooth == "SMA":
                        
                        df, R_SMA = add_walkingR(df, b, "SMA")
                        column_name_R_SMA = 'R_value_from_'+ b + '_SMA_' + str(WDW2)
                        print (R_SMA)
                        print (column_name_R_SMA)
                        #ax3=df[column_name_R_SMA].plot(secondary_y=True,  label=column_name_R_SMA)
                        c = str(b)+ "_SMA_" + str(WDW2)

                    # if b == "Total_reported":
                    #     z = df[b].index
                        
                    #     plt.fill_between(z, 0, 875, color='#f392bd',  label='waakzaam')
                    #     plt.fill_between(z, 876, 2500, color='#db5b94',  label='zorgelijk')
                    #     plt.fill_between(z, 2501, 6250, color='#bc2165',  label='ernstig')
                    #     plt.fill_between(z, 6251, 10000, color='#68032f', label='zeer ernstig')
                    #     plt.fill_between(z, 10000, 20000, color='grey', alpha=0.3, label='zeer zeer ernstig')
                    print(df.iloc[0]['date'])
                    firstday = df.iloc[0]['WEEKDAY']  # monday = 0
                   
                    if firstday == 0:
                        color_x = [COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekend, COLOR_weekend]
                    elif firstday == 1:
                        color_x = [COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekend, COLOR_weekend, COLOR_weekday]
                    elif firstday == 2:
                        color_x = [COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekend, COLOR_weekend, COLOR_weekday, COLOR_weekday]
                    elif firstday == 3:
                        color_x = [COLOR_weekday, COLOR_weekday, COLOR_weekend, COLOR_weekend, COLOR_weekday, COLOR_weekday, COLOR_weekday]
                    elif firstday == 4:
                        color_x = [COLOR_weekday, COLOR_weekend, COLOR_weekend, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekday]
                    elif firstday == 5:
                        color_x = [COLOR_weekend, COLOR_weekend, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekday]
                    elif firstday == 6:
                        color_x = [COLOR_weekend, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekday, COLOR_weekend]
                                  
                    ax = df[b].plot.bar(label=b, color = color_x)     # number of cases
                    ax = df[c].plot(label=c, color = color_list[3],linewidth=.8)         # SMA

                else:
                    # t = line -> no smoothing
                    #print(df[b])
                    #df, R_SMA = add_walkingR(df, b, "SMA")
                    
                    df[b].plot(label=b, color = color_list[n],linewidth=.8)
                    #ax3=df[R_SMA].plot(secondary_y=True, label=R_SMA)
                    #ax3.set_ylabel('Rt')

    ax.set_ylabel('Numbers')
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_major_locator(MultipleLocator(1))

    if what_to_show_r != None:
        n = len (color_list)
        x = n
        for a in what_to_show_r:
            x -=1
        
            ax3=df[a].plot(secondary_y=True, color = color_list[x], linewidth=.8, label=a)
            ax3.set_ylabel('Rt')

    # layout of the x-axis
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(df["date"].index)
    ax.set_xticklabels(df["date"].dt.date,fontsize=6, rotation=90)
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
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

    plt.title(title , fontsize=10)

    # everything in legend
    # https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
    handles,labels = [],[]
    for ax in fig1x.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)

    plt.legend(handles,labels)
    ax.text(1, 1.1, 'Created by: Rene Smit — @rcsmit',
            transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
    # configgraph(titlex)
    plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
    add_restrictions(df,ax)
    plt.show()

def add_restrictions(df,ax):
    """  _ _ _ """
    # Add restrictions
    # From Han-Kwang Nienhuys - MIT-licence
    
    df_restrictions = pd.read_csv(
                    r'covid19_seir_models\restrictions.csv',
                    delimiter=',',
                    low_memory=False)

    
    a = (min(df['date'].tolist())).date()
    #a = ((df["date_sR"].dt.date.min()))  # werkt niet, blijkbaar NaN values
   
    ymin, ymax = ax.get_ylim()
    y_lab = ymin
    for i in range(0, len(df_restrictions)):    
        d_ = df_restrictions.iloc[i]['Date'] #string
        d__ = dt.datetime.strptime(d_,'%Y-%m-%d').date()  # to dateday

        diff = (d__ - a)
        
        if diff.days >0 :
            # no idea why diff.days-2
            ax.text((diff.days), y_lab, f'  {df_restrictions.iloc[i]["Description"]}', rotation=90, fontsize=4,horizontalalignment='center')
            #plt.axvline(x=(diff.days), color='yellow', alpha=.3,linestyle='--')
      
def graph_week(dfweek):
    """  _ _ _ """
    save_df(dfweek,"weektabel")
    # SHOW A GRAPH IN TIME / WEEK
    fig1x = plt.figure()
    ax = fig1x.add_subplot(111)
  
    ax.set_xticklabels(dfweek["weeknr"],fontsize=6, rotation=45)
    #plt.title('Transit_stations compared to gliding Rt-number for the Netherlands')
    dfweek["Hospital_admission_x"].plot(label="Hospital_admission_x")
    #df["Hospital_admission"].plot.bar(label="hospital admission")
    ax3=dfweek["Rt_avg_SMA_7"].plot(secondary_y=True, color = 'r', label="Rt_avg_SMA_7")
 
    ax3.set_ylabel('Rt')
    ax.set_ylabel('Numbers')
    
    titlex = "Hospital admissions (week)"
    plt.xlabel('week')
    
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

def show_me_graph(df, what_to_show_l, what_to_show_r,how_to_smooth,t):
    """  _ _ _ """
    
    if t == "line":    
       
        title = "Aantal gevallen"
        graph_day(df, what_to_show_l,what_to_show_r , how_to_smooth, title, t)
    else:
        for c in what_to_show_l:
            what_to_show_l= ([c])      
            what_to_show_r = None
            title = c
            graph_day(df, what_to_show_l,what_to_show_r , how_to_smooth, title, t)

def smooth_columnlist(df,columnlist_):
    c_sma = []
    for c in columnlist_:
        df, newcolumn = add_SMA(df,c)
        c_sma.append(newcolumn)
    return df, c_sma

###################################################################       
def init():
    """  _ _ _ """

    global download 
    global WDW 
    global WDW2 
    global FROM
    global UNTIL


    # GLOBAL SETTINGS
    download = False
    WDW = 7
    WDW2 = 7
    FROM = '2020-2-1'
    UNTIL = '2022-1-1'
    # attention, minimum period between FROM and UNTIL = wdw days!
    
    
def main():
    """  _ _ _ """
    init()
    # LET'S GET AND PREPARE THE DATA
    df, dfweek, werkdagen, weekend_ = get_data()
    # WHAT YOU CAN DO :

    # show correlation matrixes - no options
    # correlation_matrix(df,werkdagen, weekend_)
   
    #   SHOW DAILY GRAPHS
    # show_me_graph(df, what_to_show_l, what_to_show_r,how_to_smooth,t):
   
    # Attention : the columns to be included have to be a list, even if it's only one item!
    # Options :  
    # - how to smooth : 'savgol' or 'SMA'
    # - type (t) :      'mixed' for bar+line in seperate graphs, 
    #                   'line' for all lines in one graph                  
   
   
    #                   RIVM gem/dag                  LCPS                          LCPS                
    # show_me_graph(df,["Total_reported","Kliniek_Nieuwe_Opnames_COVID","IC_Nieuwe_Opnames_COVID"],"savgol", "line")
    #add_SMA(df,["Total_reported", "Deceased"])

    #move_column(df, "Total_reported_SMA_7",20 )
    #show_me_graph(df,["Total_reported_SMA_7", "Total_reported_SMA_7_moved_20"],["Deceased_SMA_7"],"savgol", "line")
    
    # show a weekgraph, options have to put in the function self, no options her (for now)
    # graph_week(dfweek)
    
  

                       # RIVM,                 gemeentes             ,LCPS
    #columnlist2= ["Hospital_admission_x", "Hospital_admission_y", "Kliniek_Nieuwe_Opnames_COVID"]
    
    # DAILY STATISTICS ################
    columnlist2= ["Kliniek_Nieuwe_Opnames_COVID","IC_Nieuwe_Opnames_COVID", "Deceased"]
    columnlist1= ["Total_reported"]
    
    df, c_sma1 = smooth_columnlist(df, columnlist1)
    df, c_sma2 = smooth_columnlist(df, columnlist2)
    show_me_graph(df,c_sma2,c_sma1,"SMA", "line")
    


main()

# https://www.medrxiv.org/content/10.1101/2020.05.06.20093039v3.full
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7729173/

