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
# alles omgeFzet in functies

# 4 maart
# meer onderverdeling in functies. Alles aan te roepen vanuit main() met parameters

# 5 maart
# custom colors
# weekend different color in barplot
# annoying problem met een join (van outer naar inner naar outer en toen werkte het weer)
# R value (14 days back due to smoothing)

#6 maart
# last row in bar-graph was omitted due to ["date of statistics"] instead of ["date"] in addwalkingR
# Bug wit an reset.index() somewhere. Took a long time to find out
# Tried to first calculate SMA and R, and then cut of FROM/UNTIL. Doesnt
# work. Took also a huge amount of time. Reversed everything afterwards

# 7 maart
# weekgraph function with parameters

# 8 maart
# find columns with max correlation
# find the timelag between twee columns
# added a second way to calculate and display R

# 9-11 maart: Grafieken van Dissel : bezetting bedden vs R

# 12 maart
# Genormeerde grafiek (max = 1 of begin = 100)
# Various Tg vd de R-number-curves

#14 maart
# Streamlit :)


#15 maart
# weekgrafieken
#  Series.dt.isocalendar().week
# first value genormeerde grafiek
# bredere grafiek
# leegmaken vd cache DONE
# kiezen welke R je wilt in de bargraph
# waarde dropdown anders dan zichtbaar

#16 maart


# I used iloc.  Iterating through pandas objects is generally slow.
# In many cases, iterating manually over the rows is not needed and
# can be avoided with one of the following approaches:
# http://pandas-docs.github.io/pandas-docs-travis/getting_started/basics.html#iteration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
#import seaborn as sn
from scipy import stats
import datetime as dt
from datetime import datetime, timedelta

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
import math
_lock = RendererAgg.lock
from scipy.signal import savgol_filter
import streamlit as st
import urllib
import urllib.request
from pathlib import Path
from streamlit import caching
from inspect import currentframe, getframeinfo

# R-numbers from 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
# Google mobilty from https://www.google.com/covid19/mobility/?hl=nl
# Apple mobility from https://covid19.apple.com/mobility
# # Merged in one file in Excel and saved to CSV
# Hospitals from RIVM 'https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv

###################################################################
@st.cache(ttl=60*60*24)
def download_csv_file(url, csv,delimiter_):
    #df_temp = None
    print (f"Downloading {url}...")
    download= True
    with st.spinner(f'Wait for it...{url}'):
        if download :
            if url[-3:]=='csv' :
                df_temp = pd.read_csv(url,
                        delimiter=delimiter_,
                        low_memory=False)
            elif url[-4:]=='json':
                print (f"Download {url}")
                df_temp = pd.read_json (url)
            else:
                print ("Error in URL")
                st.stop()
            df_temp = df_temp.drop_duplicates()
            df_temp = df_temp.replace({pd.np.nan: None})
            save_df(df_temp,csv)
        return df_temp

@st.cache(ttl=60*60*24)
def get_data():
    """  _ _ _ """
    data =  [

        {'url'       :'https://lcps.nu/wp-content/uploads/covid-19.csv',
        'name'       :'LCPS',
        'delimiter'  :',',
        'key'        : 'Datum',
        'dateformat' :'%d-%m-%Y',
        'groupby'    :None},

        # ERROR IN THIS FILE
        {'url'     :'https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv',
        'name'       :'COVID-19_ziekenhuisopname',
        'delimiter'  :';',
        'key'        : 'Date_of_statistics',     #] = df_hospital['Date_of_statistics'].astype('datetime64[D]')
        'dateformat' : '%Y-%m-%d',                       # 'dateformat'
        'groupby'    : 'Date_of_statistics'},



        {'url'       :'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json',
        'name'       :'reprogetal',
        'delimiter'  :',',
        'key'        : 'Date',
        'dateformat' :'%Y-%m-%d',
        'groupby'    :None},

        {'url'       :'https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv',
        'name'       :'COVID-19_aantallen_gemeente_per_dag',
        'delimiter'  :';',
        'key'        :  'Date_of_publication',  #.astype('datetime64[D]')
        'dateformat' :'%Y-%m-%d',
        'groupby'    :'Date_of_publication'},     #] , sort=True).sum().reset_index()

        {'url'       :'https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv',
        'name'       :'COVID-19_uitgevoerde_testen',
        'delimiter'  :';',
        'key'        : 'Date_of_statistics', #] = df_uitgevoerde_testen['Date_of_statistics'].astype('datetime64[D]')
        'dateformat' : '%Y-%m-%d',
        'groupby'    :'Date_of_statistics'}, #] , sort=True).sum().reset_index()

        {'url'       :'https://data.rivm.nl/covid-19/COVID-19_prevalentie.json',
        'name'       :'prevalentie',
        'delimiter'  :',',
        'key'        : 'Date',   #] = df_prevalentie['Date'].astype('datetime64[D]')
        'dateformat' : '%Y-%m-%d',
        'groupby'    :None},

        {'url'       :'https://raw.githubusercontent.com/rcsmit/COVIDcases/main/mobility.csv',
        'name'       :'mobility',
        'delimiter'  : ';',
        'key'        : 'Date',
        'dateformat' : '%d-%m-%Y',
        'groupby'    :None},

        {'url'       :'https://raw.githubusercontent.com/rcsmit/COVIDcases/main/knmi2.csv',
        'name'       :'knmi',
        'delimiter'  :';',
        'key'        : 'Datum',
        'dateformat' : '%d-%m-%Y',
        'groupby'    :None},

        {'url'       :'https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.json',
        'name'       :'rioolwater',
        'delimiter'  :',',
        'key'        :'Date_measurement',
        'dateformat' : '%Y-%m-%d',
        'groupby'    :'Date_measurement'}
        ]

    type_of_join="outer"
    d=0
    #st.write(data[d]['url'], data[d]['name'],data[d]['delimiter'])
    df_temp_x = download_csv_file(data[d]['url'], data[d]['name'],data[d]['delimiter'])
    df_temp_x = df_temp_x.replace({pd.np.nan: None})
    #st.write(df_temp_x)

    df_temp_x[data[d]['key']]=pd.to_datetime(df_temp_x[data[d]['key']], format=data[d]['dateformat'])
    #st.write(df_temp_x.dtypes)
    firstkey = data[d]['key']
    df_temp = df_temp_x

    if data[d]['groupby'] != None:
        df_temp_x = df_temp_x.groupby([data[d]['key']] , sort=True).sum().reset_index()
    for d in range (1,len(data)):
       # st.write(data[d]['url'], data[d]['name'],data[d]['delimiter'])
        df_temp_x = download_csv_file(data[d]['url'], data[d]['name'],data[d]['delimiter'])
        df_temp_x = df_temp_x.replace({pd.np.nan: None})
        #st.write(df_temp_x)
        oldkey = data[d]['key']
        newkey = "key"+ str(d)
        df_temp_x = df_temp_x.rename(columns = {oldkey: newkey})
        df_temp_x[newkey]=pd.to_datetime(df_temp_x[newkey], format=data[d]['dateformat'])
       # st.write(df_temp_x.dtypes)
        if data[d]['groupby'] != None:
            df_temp_x = df_temp_x.groupby([newkey] , sort=True).sum().reset_index()
        #df_temp_x = df_temp_x.replace({pd.np.nan: None})
        df_temp = pd.merge(df_temp, df_temp_x, how=type_of_join, left_on = firstkey, right_on=newkey)
        #df_temp_x = df_temp_x.replace({pd.np.nan: None})
        df_temp.loc[df_temp[firstkey].isnull(),firstkey] = df_temp[newkey]
        df_temp = df_temp.sort_values(by=firstkey)

        #st.write(df_temp)
        save_df(df_temp,"waaromowaarom")
    df_temp = df_temp.rename(columns = {firstkey: "date"})  #the tool is build around "date"


    global UPDATETIME
    UPDATETIME = datetime.now()
    df = splitupweekweekend(df_temp)
    df = extra_bewerkingen(df)
    return df, UPDATETIME #, werkdagen, weekend_

def extra_bewerkingen(df):

    df['Percentage_positive'] = round((df['Tested_positive'] /
                                                df['Tested_with_result'] * 100),2 )


    df['temp_etmaal'] = df['temp_etmaal']  / 10
    df['temp_max'] = df['temp_max']  / 10
    df['RNA_per_reported'] = round(((df['RNA_flow_per_100000']/1e15)/df[ 'Total_reported']* 100),2 )
    print (df.dtypes)
    return df

###################################################
def calculate_cases(df, ry1, ry2,  total_cases_0, sec_variant,extra_days):
    column = df["date"]
    b_= column.max().date()
    # #fr = '2021-1-10' #doesnt work
    # fr = FROM
    # a_ = dt.datetime.strptime(fr,'%Y-%m-%d').date()

    a_ = FROM

    #b_ = dt.datetime.strptime(UNTIL,'%Y-%m-%d').date()
    datediff = ( abs((a_ - b_).days))+1+extra_days
    # f = 1
    # ry1 = 0.8 * f
    # ry2 = 1.15 * f
    # total_cases_0 = 7500
    # sec_variant = 10
    population = 17_500_000
    immune_day_zero = 4_000_000

    suspectible_0 = population - immune_day_zero
    cumm_cases = 0

    cases_1 = ((100-sec_variant)/100)* total_cases_0
    cases_2 = (sec_variant/100)* total_cases_0
    temp_1 = cases_1
    temp_2 = cases_2
    r_temp_1 = ry1
    r_temp_2 = ry2

    immeratio = 1
    df_calculated = pd.DataFrame({'date_calc': a_,
                'variant_1': cases_1,'variant_2' :cases_2, 'variant_12' : int(cases_1+cases_2)}, index=[0])
    Tg = 4

    #print (df_calculated.dtypes)
    #a_ = dt.datetime.strptime(a,'%m/%d/%Y').date()
    column = df["date"]
    max_value = column. max()

    for day in range (1, datediff):

        thalf1 = Tg * math.log(0.5) / math.log(immeratio*ry1)
        thalf2 = Tg * math.log(0.5) / math.log(immeratio*ry2)
        day = a_ + timedelta(days=day)
        pt1 = (temp_1 * (0.5**(1/thalf1)))
        pt2 = (temp_2* (0.5**(1/thalf2)))
        day_ = day.strftime("%Y-%m-%d") # FROM object TO string
        day__ =  dt.datetime.strptime(day_,'%Y-%m-%d') # from string to daytime

        df_calculated =df_calculated .append({'date_calc':day_,
                'variant_1'  : int(pt1),
                'variant_2' : int(pt2) , 'variant_12' : int(pt1+pt2) },ignore_index=True)

        temp_1 = pt1
        temp_2 = pt2

        cumm_cases += pt1 + pt2
        cumm_cases_corr = cumm_cases * 2.5
        immeratio = (1-(cumm_cases_corr/suspectible_0 ))

    df_calculated['date_calc'] = pd.to_datetime( df_calculated['date_calc'])

    df = pd.merge(df, df_calculated, how='outer', left_on = 'date', right_on="date_calc",
                        left_index=True )
    print (df.dtypes)
    df.loc[df['date'].isnull(),'date'] = df['date_calc']
    return df

def splitupweekweekend(df):
    """  _ _ _ """
    # SPLIT UP IN WEEKDAY AND WEEKEND
    # https://stackoverflow.com/posts/56336718/revisions
    df['WEEKDAY'] = pd.to_datetime(df['date']).dt.dayofweek  # monday = 0, sunday = 6
    df['weekend'] = 0          # Initialize the column with default value of 0
    df.loc[df['WEEKDAY'].isin([5, 6]), 'weekend'] = 1  # 5 and 6 correspond to Sat and Sun
    return df

    # remove outliers - doesnt work
    # df = df[(np.abs(stats.zscore(df['retail_and_recreation'])) < 3)]
    # df = df[(np.abs(stats.zscore(df['transit_stations'])) < 3)]
    # df = df[(np.abs(stats.zscore(df['workplaces'])) < 3)]
    # df = df[(np.abs(stats.zscore(df['grocery_and_pharmacy'])) < 3)]

def add_walking_r(df, smoothed_columns, how_to_smooth, tg):
    """  _ _ _ """
    #print(df)
    # Calculate walking R from a certain base. Included a second methode to calculate R
    # de rekenstappen: (1) n=lopend gemiddelde over 7 dagen; (2) Rt=exp(Tc*d(ln(n))/dt)
    # met Tc=4 dagen, (3) data opschuiven met rapportagevertraging (10 d) + vertraging
    # lopend gemiddelde (3 d).
    # https://twitter.com/hk_nien/status/1320671955796844546
    # https://twitter.com/hk_nien/status/1364943234934390792/photo/1
    column_list_r_smoothened = []
    column_list_r_sec_smoothened = []
    for base in smoothed_columns:
        column_name_R = 'R_value_from_'+ base +'_tg'+str(tg)
        column_name_R_sec = 'R_value_(hk)_from_'+ base

        #df, new_column = smooth_columnlist(df,[base],how_to_smooth)
        column_name_r_smoothened = 'R_value_from_'+ base +'_tg'+str(tg) + '_'+ how_to_smooth + '_' + str(WDW3)
        column_name_r_sec_smoothened = 'R_value_sec_from_'+ base +'_tg'+str(tg) + '_'+ how_to_smooth + '_' + str(WDW3)
        #df[SMA1] = df.iloc[:,df.columns.get_loc(base)].rolling(window=WDW2).mean()

        sliding_r_df= pd.DataFrame({'date_sR': [],
                column_name_R: [],column_name_R_sec: []})

        d= 7
        d2=2
        r_sec = []
        for i in range(0, len(df)):
            if df.iloc[i][base] != None:
                date_ = pd.to_datetime(df.iloc[i]['date'], format="%Y-%m-%d")
                date_ = df.iloc[i]['date']
                if df.iloc[i-d][base] != 0 or df.iloc[i-d][base] is not None:
                    slidingR_= round(((df.iloc[i][base]/df.iloc[i-d][base])**(tg/d) ),2)
                    slidingR_sec = round(math.exp((tg *(math.log(df.iloc[i][base])- math.log(df.iloc[i-d2][base])))/d2),2)
                else:
                    slidingR_ = None
                    slidingR_sec = None
                sliding_r_df=sliding_r_df.append({'date_sR':date_,
                column_name_R    : slidingR_,
                column_name_R_sec : slidingR_sec },ignore_index=True)

        # je zou deze nog kunnen smoothen, maar levert een extra vertraging in de resultaten op,
        # dus wdw=1
        sliding_r_df[column_name_r_smoothened] = round(sliding_r_df.iloc[:,1].rolling(window=WDW3,  center=True).mean(),2)
        sliding_r_df[column_name_r_sec_smoothened] = round(sliding_r_df.iloc[:,2].rolling(window=WDW3,  center=True).mean(),2)

        # df = df.reset_index()
        # df.set_index('date')

        sliding_r_df = sliding_r_df.reset_index()
        #save_df(sliding_r_df ,"hadhkgelijk")
        # WHY DOES IT CHANGE MY DF[base]. Inner = ok / outer = ERR
        # when having this error "posx and posy should be finite values"
        df = pd.merge(df, sliding_r_df, how='outer', left_on = 'date', right_on="date_sR",
                        left_index=True )
        column_list_r_smoothened.append(column_name_r_smoothened)
        column_list_r_sec_smoothened.append(column_name_r_sec_smoothened)
        #R_SMA = column_name_r_smoothened
        # df = df.reset_index()
        # df.set_index('date')

        sliding_r_df = sliding_r_df.reset_index()
        #save_df(df,"lastchrismas")
    return df, column_list_r_smoothened, column_list_r_sec_smoothened

def agg_week(df, how):
    """  _ _ _ """

    #df.loc[df['date'].isnull(),'date'] = df['Date_of_statistics']

    df['weeknr']  =  df['date'].dt.week
    df['yearnr']  =  df['date'].dt.year



    df['weekalt']   = (df['date'].dt.year.astype(str) + "-"+
                         df['date'].dt.week.astype(str))

    for i in range(0,len (df)):
        if df.iloc[i]['weekalt'] == "2021-53":
            df.iloc[i]['weekalt'] = "2020-53"

    #how = "mean"
    if how == "mean":
        dfweek = df.groupby(['weeknr', 'yearnr', 'weekalt'], sort=False).mean().reset_index()
    elif how == "sum" :
        dfweek = df.groupby(['weeknr', 'yearnr', 'weekalt'], sort=False).sum().reset_index()
    else:
        print ("error agg_week()")
        st.stop()
    return df, dfweek

def move_column(df, column_,days):
    """  _ _ _ """
    # #move Rt r days, because a maybe change has effect after r days
    # Tested - Conclusion : It has no effect
    r=days
    if type(column_) == list:
        column_=column_
    else:
        column_=[column_]

    for column in column_:
        #print (column)
        new_column = column + "_moved_" + str(r)
        df[new_column] = df[column].shift(r)
    #print ("Name moved column : " + new_column)
    return df, new_column

def drop_columns(what_to_drop):
    """  _ _ _ """

    if what_to_drop != None:
        print ("dropping " + what_to_drop)
        for d in what_to_drop:
            df = df.drop(columns=['d'],axis=1)
def select_period(df, show_from, show_until):
    #st.write("PERIOD CHANGED" + str(show_from) + "-"+ str(show_until))
    #st.write(show_from)
    if show_from == None:
        show_from = '2020-1-1'

    if show_until == None:
        show_until = '2030-1-1'
    # show_from_ = dt.datetime(show_from)
    # show_until_ = dt.datetime.strptime(show_until,'%Y-%m-%d')
    mask = (df['date'].dt.date >= show_from) & (df['date'].dt.date <= show_until)
    df = (df.loc[mask])

    df = df.reset_index()

    return df

def last_manipulations(df, what_to_drop, drop_last):
    """  _ _ _ """
    #print ("Doing some last minute manipulations")
    drop_columns(what_to_drop)

    #

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
        df = df[:drop_last] #drop last row(s)
    #print ("=== AFTER LAST MANIPULATIONS ==== ")
    #print (df.dtypes)

    # df['weeknr']  =  df['date'].dt.isocalendar().week
    # df['yearnr']  =  df['date'].dt.isocalendar().year


    # df['weekalt']   = (df['date'].dt.isocalendar().year.astype(str) + "-"+
    #                      df['date'].dt.isocalendar().week.astype(str))

    df['weeknr']  =  df['date'].dt.week
    df['yearnr']  =  df['date'].dt.year


    df['weekalt']   = (df['date'].dt.year.astype(str) + "-"+
                         df['date'].dt.week.astype(str))


    return df, werkdagen, weekend_
    #return df

def save_df(df,name):
    """  _ _ _ """
    #name_ = OUTPUT_DIR + name+'.csv'
    name_ =  OUTPUT_DIR + name+'.csv'
    compression_opts = dict(method=None,
                            archive_name=name_)
    df.to_csv(name_, index=False,
            compression=compression_opts)

    print ("--- Saving "+ name_ + " ---" )
##########################################################
def correlation_matrix(df, werkdagen, weekend_):
    """  _ _ _ """
    print("x")
    # CALCULATE CORRELATION

    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True, annot_kws={"fontsize":7})
    plt.title("ALL DAYS", fontsize =20)
    plt.show()

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
def normeren(df, what_to_norm):
    """ In : columlijst
    Bewerking : max = 1
    Out : columlijst met genormeerde kolommen """
    #print(df.dtypes)

    normed_columns = []

    for column in what_to_norm:
        #print(df[column])
        maxvalue = (df[column].max())/100
        firstvalue = df[column].iloc[int(WDW2/2)]/100
        name = (f"{column}_normed")
        #st.write (firstvalue)
        #st.write (column)
        for i in range(0,len(df)):
            if how_to_norm == "max":

                df.loc[i, name] = df.loc[i,column]/maxvalue
            else:
                #df.name.loc[~df.name.isnull()].iloc[0]
                df.loc[i, name] = df.loc[i,column]/firstvalue
        normed_columns.append(name)
        print (f"{name} generated")
    return df, normed_columns

def graph_daily_normed(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display):
    """ IN : df, de kolommen die genormeerd moeten worden
    ACTION : de grafieken met de genormeerde kolommen tonen """

    if what_to_show_day_l is None:
        st.warning ("Choose something")
        st.stop()

    df, smoothed_columns_l = smooth_columnlist(df,what_to_show_day_l,how_to_smoothen)
    df, normed_columns_l = normeren(df, smoothed_columns_l )

    df, smoothed_columns_r = smooth_columnlist(df,what_to_show_day_r,how_to_smoothen)
    df, normed_columns_r = normeren(df, smoothed_columns_r )

    graph_daily(df,normed_columns_l,normed_columns_r, None , how_to_display)

def graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title,t):

    df_temp = pd.DataFrame(columns = ['date'])
    if what_to_show_l is None:
        st.warning ("Choose something")
        st.stop()

    # print (df)

    # if show_variant == True:
    #     df, ry1, ry2 = calculate_cases(df)
    #print (df.dtypes)
    """  _ _ _ """
    if type(what_to_show_l) == list:
        what_to_show_l_=what_to_show_l
    else:
        what_to_show_l_=[what_to_show_l]
    aantal = len(what_to_show_l_)
    #st.write(aantal)
    # SHOW A GRAPH IN TIME / DAY
    #st.write(df)

    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)
        # Some nice colors chosen with coolors.com
        bittersweet = "#ff6666"  # reddish 0
        operamauve = "#ac80a0" # purple 1
        green_pigment = "#3fa34d" #green 2
        minion_yellow = "#EAD94C" # yellow 3
        mariagold = "#EFA00B" # orange 4
        falu_red= "#7b2d26" # red 5
        COLOR_weekday = "#3e5c76" # blue 6
        COLOR_weekend = "#e49273" # dark salmon 7
        prusian_blue = "#1D2D44" # 8

        #{"May Green":"4E9148","Red Orange Color Wheel":"F05225","Midnight Green Eagle Green":"024754",
        # "Bright Yellow Crayola":"FBAA27","Black Coffee":"302823","Pumpkin":"F07826","Verdigris":"02A6A8"}

        color_list =["#02A6A8","#4E9148","#F05225","#024754","#FBAA27","#302823","#F07826"]

        #color_list = [ operamauve, bittersweet, minion_yellow, COLOR_weekday, mariagold,falu_red,'y',  COLOR_weekend ,green_pigment]
        #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
        #mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=color_list)
        n = 0  # counter to walk through the colors-list

        df, columnlist_sm_l = smooth_columnlist(df,what_to_show_l_,how_to_smooth)
        print (df)
        # df, R_smooth = add_walking_r(df, columnlist, how_to_smooth)
        # stackon=""
        # if len(what_to_show_l_)>1:
        #     w = ["Datum"]
        #     for s in what_to_show_l_:
        #         w.append(s)
        #     #st.write(w)
        #     df_stacked = df[w].copy()
        #     #print (df_stacked.dtypes)
        #     #df_stacked.set_index('Datum')


            #st.write(df_stacked)
            #if t == "bar":
                #ax = df_stacked.plot.bar(stacked=True)
                #ax = df_stacked.plot(rot=0)
                #st.bar_chart(df_stacked)
         #ax = df[c_smooth].plot(label=c_smooth, color = color_list[2],linewidth=1.5)         # SMA

        for b in what_to_show_l_:
        # if type(a) == list:
        #     a_=a
        # else:
        #     a_=[a]

            # PROBEERSEL OM WEEK GEMIDDELDES MEE TE KUNNEN PLOTTEN IN DE DAGELIJKSE GRAFIEK

            # dfweek_ = df.groupby('weekalt', sort=False).mean().reset_index()
            # save_df(dfweek_,"whatisdftemp1")
            # w = b + "_week"
            # print ("============="+ w)
            # df_temp = dfweek_[["weekalt",b ]]
            # df_temp = df_temp(columns={b: w})

            # print (df_temp.dtypes)
            # #df_temp is suddenly a table with all the rows
            # print (df_temp)
            # save_df(df_temp,"whatisdftemp2")

            if t == "bar":

                # if b == "Total_reported":
                #     z = df[b].index

                #     plt.fill_between(z, 0, 875, color='#f392bd',  label='waakzaam')
                #     plt.fill_between(z, 876, 2500, color='#db5b94',  label='zorgelijk')
                #     plt.fill_between(z, 2501, 6250, color='#bc2165',  label='ernstig')
                #     plt.fill_between(z, 6251, 10000, color='#68032f', label='zeer ernstig')
                #     plt.fill_between(z, 10000, 20000, color='grey', alpha=0.3, label='zeer zeer ernstig')
                #print(df.iloc[0]['date'])

                # weekends have a different color
                firstday = df.iloc[0]['WEEKDAY']  # monday = 0
                if  firstday == 0:
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
                #color_x = ["white", "red", "yellow", "blue", "purple", "green", "black"]
                white = "#eeeeee"
                if showoneday:
                    if showday == 6:
                        color_x = [white, white, white, white, white,white, bittersweet]
                    elif  showday == 5:
                        color_x = [white, white, white, white, white, bittersweet, white]
                    elif showday == 4:
                        color_x = [white, white, white, white, bittersweet, white, white]
                    elif showday == 3:
                        color_x = [white, white, white, bittersweet, white, white, white]
                    elif showday == 2:
                        color_x = [white, white, bittersweet, white, white, white, white]
                    elif showday == 1:
                        color_x = [white, bittersweet, white, white, white, white, white]
                    elif showday == 0:
                        color_x = [bittersweet, white, white, white, white, white, white]
                # if showoneday = True:
                #     color_x = [bittersweet, white, white,white, white, white, white]
                # MAYBE WE CAN LEAVE THIS OUT HERE
                df, columnlist = smooth_columnlist(df,[b],how_to_smooth)

                df.set_index('date')

                #x= 4+ (WDW2+WDW3)*-1 # 4 days correction factor, dont know why
                #x=0
                #df, new_column = move_column(df,R_smooth,x)
                df_temp = df
                if len(what_to_show_l_)==1:
                    #st.write(what_to_show_l)
                    ax = df_temp[b].plot.bar(label=b, color = color_x, alpha=.6)     # number of cases

                    for c_smooth in columnlist:
                        # print (c_smooth)
                        # print (df[c_smooth])
                        ax = df[c_smooth].plot(label=c_smooth, color = color_list[2],linewidth=1.5)         # SMA

                    #df_temp = select_period(df, FROM, UNTIL)
                    #df_temp.set_index('date')

                    #print(df_temp.dtypes)
                    #c = str(b)+ "_"+how_to_smooth+ "_" + str(WDW2)

                    if showR :
                        ax3=df["Rt_avg"].plot(secondary_y=True,linestyle='--', label="Rt RIVM",color=green_pigment, alpha=.8,linewidth=1)
                        ax3.fill_between(df['date'].index, df["Rt_low"], df["Rt_up"],
                                    color=green_pigment, alpha=0.2, label='_nolegend_')
                        tgs = [3.5,4,5]

                        #print(df['Rt_low'].index)
                        teller=0
                        dfmin = ""
                        dfmax = ""
                        for TG in tgs:
                            df, R_smooth, R_smooth_sec = add_walking_r(df, columnlist, how_to_smooth,TG)

                            for R in R_smooth:
                                # correctie R waarde, moet naar links ivm 2x smoothen
                                df, Rn = move_column(df,R,MOVE_WR)

                                if teller == 1 :
                                    ax3=df[Rn].plot(secondary_y=True, label=Rn,linestyle='--',color=falu_red, linewidth=1.2)
                                else:
                                    if teller == 0 :
                                        #print ("-" + str(teller)+ "/ " + Rn)
                                        dfmin = Rn
                                    if teller == 2 :
                                        dfmax = Rn
                                        #print ("-" + str(teller)+ "/ " + Rn)
                                        print (dfmax)
                                    #ax3=df[Rn].plot(secondary_y=True,label='_nolegend_', linestyle='dotted',color=falu_red, linewidth=0.8)
                                teller += 1
                                #print ("TELLER" + str(teller))
                            for R in R_smooth_sec:  # SECOND METHOD TO CALCULATE R
                                # correctie R waarde, moet naar links ivm 2x smoothen
                                df, Rn = move_column(df,R,MOVE_WR)
                                #ax3=df[Rn].plot(secondary_y=True, label=Rn,linestyle='--',color=operamauve, linewidth=1)
                        ax3.fill_between(df['date'].index, df[dfmin], df[dfmax],  color=falu_red, alpha=0.3, label='_nolegend_')

                        #ax3.fill_between(x, y1, y2)

                        #ax3.set_ylabel('Rt')

                #left, right = ax.get_xlim()
                #ax.set_xlim(left - 0.5, right + 0.5)
                #ax3.set_ylim(0.6,1.5)


            else: # t = line

                #df, R_SMA = add_walking_r(df, b, "SMA")

                #df_temp = select_period(df, FROM, UNTIL)
                df_temp = df

                if how_to_smooth == None:
                    how_to_smooth_ = "unchanged_"
                else:
                    how_to_smooth_ = how_to_smooth + "_" + str(WDW2)
                b_ = str(b)+ "_"+how_to_smooth_
                df_temp[b_].plot(label=b, color = color_list[n], linewidth=1.1) # label = b_ for uitgebreid label
                df_temp[b].plot(label='_nolegend_', color = color_list[n],linestyle='dotted',alpha=.9,linewidth=.8)
            # else:
            #     print ("ERROR in graph_day")
            n +=1
        if show_scenario == True:
            df = calculate_cases(df, ry1, ry2, total_cases_0, sec_variant, extra_days)
            print (df.dtypes)
            l1 = (f"R = {ry1}")
            l2 = (f"R = {ry2}")
            ax = df["variant_1"].plot(label=l1, color = color_list[4],linestyle='dotted',linewidth=1, alpha=1)
            ax = df["variant_2"].plot(label=l2, color = color_list[5],linestyle='dotted',linewidth=1, alpha=1)
            ax = df["variant_12"].plot(label='TOTAL', color = color_list[6],linestyle='--',linewidth=1, alpha=1)


        if what_to_show_r != None:
            if type(what_to_show_r) == list:
                what_to_show_r=what_to_show_r
            else:
                what_to_show_r=[what_to_show_r]

            n = len (color_list)

            x = n
            for a in what_to_show_r:
                x -=1
                lbl = a + " (right ax)"
                #lbl2 = a + "_" + how_to_smooth + "_" + str(WDW2)

                df, columnlist = smooth_columnlist(df,[a],how_to_smooth)
                #df_temp = select_period(df, FROM, UNTIL)
                #df_temp = df
                for c_ in columnlist:
                    #smoothed
                    lbl2 = a + " (right ax)"
                    ax3 = df_temp[c_].plot(secondary_y=True, label=lbl2, color = color_list[x], linestyle='--', linewidth=1.1) #abel = lbl2 voor uitgebreid label
                ax3=df_temp[a].plot(secondary_y=True, linestyle='dotted', color = color_list[x], linewidth=1, alpha=.9, label='_nolegend_')
                ax3.set_ylabel('_')
        if t != "bar":

            pass



        if what_to_show_r is not None:
            if len( what_to_show_l) ==1 and len( what_to_show_r)==1:
                correlation = find_correlation_pair(df, what_to_show_l, what_to_show_r)
                correlation_sm= find_correlation_pair(df, b_, c_)

        if what_to_show_r is not None:
            if len( what_to_show_l) ==1 and len( what_to_show_r)==1:
                title = (f"{title} \nCorrelation = {correlation}\nCorrelation smoothed = {correlation_sm}")
        plt.title(title , fontsize=10)

    a__ = (max(df_temp['date'].tolist())).date() - (min(df_temp['date'].tolist())).date()
    freq = int(a__.days/10)
    #ax.xaxis.set_minor_locator(MultipleLocator())
    ax.xaxis.set_major_locator(MultipleLocator(freq))

    ax.set_xticks(df_temp['date'].index)
    ax.set_xticklabels(df_temp['date'].dt.date,fontsize=6, rotation=90)
    xticks = ax.xaxis.get_major_ticks()
    for i,tick in enumerate(xticks):
        if i%10 != 0:
            tick.label1.set_visible(False)
    plt.xticks()


    # layout of the x-axis
    ax.xaxis.grid(True, which='major',alpha=.4,linestyle='--')
    ax.yaxis.grid(True, which='major',alpha=.4,linestyle='--')

    left, right = ax.get_xlim()
    ax.set_xlim(left, right)
    fontP = FontProperties()
    fontP.set_size('xx-small')

    plt.xlabel('date')
    # everything in legend
    # https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
    handles,labels = [],[]
    for ax in fig1x.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    #plt.xlim(FROM, UNTIL)
    #ax.set_xlim([datetime.date(2021, 1, 26), datetime.date(2024, 2, 1)])
    #ax.set_xlim(153,500)
    #ax.set_xlim(pd.to_datetime(FROM), pd.to_datetime(UNTIL))
    plt.legend(handles,labels)
    #ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4)) #here is the magic
    #plt.legend( bbox_to_anchor=(0.5, -0.4), loc=2,fontsize=6, prop=fontP)
    ax.text(1, 1.1, 'Created by Rene Smit — @rcsmit',
            transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
    # configgraph(titlex)
    #plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
    add_restrictions(df,ax)
    #plt.show()
    #fig1x.tight_layout()
    if t == "line":
        set_xmargin(ax, left=-0.04, right=-0.04)
    st.pyplot(fig1x)
    #st.write(df)
def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta*left
    right = lim[1] + delta*right
    ax.set_xlim(left,right)



def add_restrictions(df,ax):
    """  _ _ _ """
    # Add restrictions
    # From Han-Kwang Nienhuys - MIT-licence

    df_restrictions = pd.read_csv('https://raw.githubusercontent.com/rcsmit/COVIDcases/main/restrictions.csv',
                    comment='#',
                    delimiter=',',
                    low_memory=False,
                    )
    #df_restrictions =   select_period(df_restrictions, FROM, UNTIL)
    #mask = (df_restrictions['Date'].date >= show_from) & (df_restrictions['Date'].date <= show_until)
    #df_restrictions = (df_restrictions.loc[mask])

    a = (min(df['date'].tolist())).date()
    b = (max(df['date'].tolist())).date()
    #a = ((df['date_sR'].dt.date.min()))  # werkt niet, blijkbaar NaN values

    ymin, ymax = ax.get_ylim()
    y_lab = ymin
    for i in range(0, len(df_restrictions)):
        d_ = df_restrictions.iloc[i]['Date'] #string
        d__ = dt.datetime.strptime(d_,'%Y-%m-%d').date()  # to dateday

        diff = (d__ - a)
        diff2 = (b - d__ )

        if diff.days >0 and diff2.days >0:
            # no idea why diff.days-2
            ax.text((diff.days), y_lab, f'  {df_restrictions.iloc[i]["Description"] }', rotation=90, fontsize=4,horizontalalignment='center')
            #plt.axvline(x=(diff.days), color='yellow', alpha=.3,linestyle='--')

def graph_week(df, what_to_show_l, how_l, what_to_show_r, how_r):
    """  _ _ _ """
    #save_df(dfweek,"weektabel")
    # SHOW A GRAPH IN TIME / WEEK
    df_l, dfweek_l = agg_week(df, how_l)
    #st.write(df_l)
    #st.write(FROM)
    if str(FROM) is not '2021-01-01':
        st.info("To match the weeknumbers on the ax with the real weeknumbers, please set the startdate at 2021-1-1")
    if what_to_show_r != None:
        df_r, dfweek_r = agg_week (df, how_r)

    if type(what_to_show_l) == list:
            what_to_show_l = what_to_show_l
    else:
            what_to_show_l=[what_to_show_l]

    for show_l in what_to_show_l:

        fig1y = plt.figure()
        ax = fig1y.add_subplot(111)
        ax.set_xticks(dfweek_l['weeknr'])
        ax.set_xticklabels(dfweek_l['weekalt'] ,fontsize=6, rotation=45)
        label_l = show_l+ " ("+ how_l + ")"
        dfweek_l[show_l].plot.bar(label=label_l, color="#F05225")

        if what_to_show_r != None:
            for what_to_show_r_ in what_to_show_r:
                label_r = what_to_show_r_+ " ("+ how_r + ")"
                ax3=dfweek_r[what_to_show_r_].plot(secondary_y=True, color = 'r', label=label_r)

        #ax3=dfweek['Rt_avg_SMA_7'] .plot(secondary_y=True, color = 'r', label="Rt_avg_SMA_7")

        #ax3.set_ylabel('Rt')
        #ax.set_ylabel('Numbers')



        # Add a grid
        plt.grid(alpha=.2,linestyle='--')

        #Add a Legend
        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(  loc='best', prop=fontP)
        #titlex = "Weekly numbers - " + show_l + "(" + how  + ")"
        #plt.title(titlex , fontsize=10)

        #ax.xaxis.grid(True, which='minor')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        #ax.xaxis.set_major_formatter()
        # everything in legend
        # https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
        handles,labels = [],[]
        for ax in fig1y.axes:
            for h,l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)

        plt.legend(handles,labels)
        plt.xlabel('Week counted from '+ str(FROM))
        # configgraph(titlex)
        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        st.pyplot(fig1y)
        #plt.show()

def graph_daily(df, what_to_show_l, what_to_show_r,how_to_smooth,t):
    """  _ _ _ """
    if t == "bar":
        if type(what_to_show_l) == list:
            what_to_show_l=what_to_show_l
        else:
            what_to_show_l=[what_to_show_l]
        title = ""
        for c in what_to_show_l:

        #    what_to_show_r = what_to_show_r
            title += str(c) + " "
        graph_day(df, what_to_show_l,what_to_show_r , how_to_smooth, title, t)
    else:
        tl =""
        tr=""
        i=0
        j=0
        if what_to_show_l is not None:
            for l in what_to_show_l:
                if i != len(what_to_show_l) - 1:
                    tl += l + " / "
                    i+=1
                else:
                    tl += l
        if what_to_show_r is not None:
            if type(what_to_show_r) == list:
                what_to_show_r=what_to_show_r
            else:
                what_to_show_r=[what_to_show_r]
            tl += " - "
            for r in what_to_show_r:
                if j != len(what_to_show_r) - 1:
                    tl += r + " / "
                    j+=1
                else:
                    tl += r
        tl = tl.replace("_", " ")

        title = (f"{tl}")
        graph_day(df, what_to_show_l,what_to_show_r , how_to_smooth, title, t)

    # else:
    #     print ("ERROR IN graph_daily")
    #     st.stop()

def smooth_columnlist(df,columnlist,t):
    """  _ _ _ """
    c_smoothen = []
    wdw_savgol = 7
    #print (columnlist_)


    if columnlist is not None:
        if type(columnlist) == list:
            columnlist_=columnlist
        else:
            columnlist_=[columnlist]
            print (columnlist)
        for c in columnlist_:
            print (f"Smoothening {c}")
            if t=='SMA':
                new_column = c + '_SMA_' + str(WDW2)
                print ('Generating ' + new_column+ '...')
                df[new_column] = df.iloc[:,df.columns.get_loc(c)].rolling(window=WDW2,  center=True).mean()

            elif t=='savgol':
                new_column = c + '_savgol_' + str(WDW2)
                print ('Generating ' + new_column + '...')
                df[new_column] = df[c].transform(lambda x: savgol_filter(x, WDW2,2))
            # elif t=="None": DOESNT WORK YET
            #     new_column = c + '_'+ '_None_' + str(WDW2)
            #     df[new_column] = df[c]
            elif t == None:
                new_column = c + "_unchanged_"
                df[new_column] = df[c]
                print ('Added ' + new_column + '...~')
            else:
                print ("ERROR in smooth_columnlist")
                st.stop()
            c_smoothen.append(new_column)
        #save_df (df, "aftersmoothen")
    return df, c_smoothen

###################################################################
def find_correlations(df):
    al_gehad = []
    paar = []
    column_list = list(df.columns)
    # print (column_list)


    for i in column_list:
        for j in column_list:
            #paar = [j, i]
            paar = str(i)+str(j)
            if paar not in al_gehad:
                if i == j:
                    pass
                else:
                    try:
                        c = round(df[i].corr(df[j]),3)
                        if c > 0.8:
                            print (f"{i} - {j} - {str(c)}")

                    except:
                        pass
            else:
                #print ("algehad")
                pass
            al_gehad.append(str(j)+str(i))
            #print ("toegevoegd" )
            #print (paar)
    #print (al_gehad)
                 # first # second

def find_correlation_pair(df, first, second):
    al_gehad = []
    paar = []
    if type(first) == list:
            first = first
    else:
            first = [first]
    if type(second) == list:
            second = second
    else:
            second = [second]
    for i in first:
        for j in second:
            c = round(df[i].corr(df[j]),3)
    return c

def find_lag_time(df, what_happens_first,what_happens_second ,r1,r2):
    b = what_happens_first
    a = what_happens_second
    x = []
    y = []
    max = 0
    max_column = None
    for n in range (r1,(r2+1)):
        df,m  = move_column(df,b,n)
        c = round(df[m].corr(df[a]),3)
        if c > max :
            max =c
            max_column = m
            m_days = n
        #print (f"{a} - {m} - {str(c)}")
        x.append(n)
        y.append(c)
    title = (f"Correlation between : {a} - {b} ")
    title2 = (f" {a} - b - moved {m_days} days ")

    fig1x = plt.figure()
    ax = fig1x.add_subplot(111)
    plt.xlabel('shift in days')
    plt.plot(x, y)
    plt.axvline(x=0, color='yellow', alpha=.6,linestyle='--')
    # Add a grid
    plt.grid(alpha=.2,linestyle='--')
    plt.title(title , fontsize=10)
    plt.show()
    graph_daily(df,[a],[b], "SMA","line")
    graph_daily(df,[a],[max_column], "SMA","line")
    # if the optimum is negative, the second one is that x days later

def init():
    """  _ _ _ """

    global download


    global INPUT_DIR
    global OUTPUT_DIR

    # INPUT_DIR = 'C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\'
    OUTPUT_DIR = 'C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\'
    INPUT_DIR = ''
    #OUTPUT_DIR = ''

    # GLOBAL SETTINGS
    download = True
    # De open data worden om 15.15 uur gepubliceerd

    #WDW2 = 7 # for all values except R values
    #WDW3 = 3 # for R values in add_walking_r
    #MOVE_WR = -7 # Rapportagevertraging

    # # attention, minimum period between FROM and UNTIL = wdw days!

def main():
    """  _ _ _ """
    global FROM
    global UNTIL
    global WDW2
    global WDW3

    global MOVE_WR
    # FROM = '2020-1-1'
    # UNTIL = '2021-5-1'

    df_getdata, UPDATETIME = get_data()
    df = df_getdata.copy(deep=False)
    df, werkdagen, weekend_ = last_manipulations(df, None, None)
    df.rename(columns={"Hospital_admission_x" : "Hospital_admission_RIVM",
    "Hospital_admission_y" : "Hospital_admission_GGD",
    "Kliniek_Nieuwe_Opnames_COVID" : "Hospital_admission_LCPS"}, inplace=True)
    st.title("Interactive Corona Dashboard")
    #st.header("")
    st.subheader("Under construction - Please send feedback to @rcsmit")

    # LET'S GET AND PREPARE THE DATA

    # what_to_show_day_r = None
    # what_to_show_week_r = None
    # COLUMNS
    # index,country_region_code,date,retail_and_recreation,grocery_and_pharmacy,parks,
    # transit_stations,workplaces,residential,apple_driving,apple_transit,apple_walking,
    # Date_of_statistics_x,Version_x,Hospital_admission_notification,Hospital_admission_x,
    # Datum,IC_Bedden_COVID,IC_Bedden_Non_COVID,Kliniek_Bedden,IC_Nieuwe_Opnames_COVID,
    # Kliniek_Nieuwe_Opnames_COVID,Date_of_publication,Total_reported,Hospital_admission_y,
    # Deceased,Date,Rt_low,Rt_avg,Rt_up,population,version,Date_of_statistics_y,Version_y,
    # Tested_with_result,Tested_positive,Percentage_positive,WEEKDAY
    # Date, prev_low, prev_avg , prev_up, population"

    # WHAT YOU CAN DO :
    # show correlation matrixes - no options
    # correlation_matrix(df,werkdagen, weekend_)

    #   SHOW DAILY GRAPHS
    # graph_daily(df, what_to_show_l, how, what_to_show_r,how_to_smooth,t):

    # Attention : the columns to be included have to be a LIST, even if it's only one item!
    # Options :
    # - how to smooth : 'savgol' or 'SMA'
    # - type (t) :      'mixed' for bar+line in seperate graphs,
    #                   'line' for all lines in one graph

    #                   RIVM gem/dag                  LCPS                          LCPS
    # graph_daily(df,['Total_reported","Kliniek_Nieuwe_Opnames_COVID","IC_Nieuwe_Opnames_COVID'] ,"savgol", "line")
    #add_sma(df,['Total_reported", "Deceased'] )

    #move_column(df, "Total_reported_SMA_7",20 )
    #graph_daily(df,['Total_reported_SMA_7", "Total_reported_SMA_7_moved_20'] ,['Deceased_SMA_7'] ,"savgol", "line")

                       # RIVM [USE THIS],                 gemeentes             ,LCPS
    #columnlist2= ["Hospital_admission_x", "Hospital_admission_y", "Kliniek_Nieuwe_Opnames_COVID"]

    # DAILY STATISTICS ################
    df_temp = None
    what_to_show_day_l = None

    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2021-01-01"
    today = datetime.today().strftime('%Y-%m-%d')

    #values 01/13/2021, according to https://www.bddataplan.nl/corona/
    from_ = st.sidebar.text_input('startdate (yyyy-mm-dd)',start_)

    try:
        FROM = dt.datetime.strptime(from_,'%Y-%m-%d').date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input('enddate (yyyy-mm-dd)',today)

    try:
        UNTIL = dt.datetime.strptime(until_,'%Y-%m-%d').date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

       # a_ = dt.datetime.strptime(fr,'%Y-%m-%d').date()

    if FROM >= UNTIL :
        st.warning("Make sure that the end date is not before the start date")
        st.stop()


    if until_ == "2023-08-23":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
            caching.clear_cache()
            until_ = "2021-01-01"

    df = select_period(df, FROM, UNTIL)
    #st.write(f"Before : {len(df)}")
    df = df.drop_duplicates()
    #st.write(f"After : {len(df)}")
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    week_or_day = st.sidebar.selectbox('Day or Week', ["day", "week"], index=0)
    if week_or_day != "week":
    #how_to_display = st.sidebar.selectbox('What to plot (line/bar)', ["line", "line_scaled_to_peak", "bar"], index=0)
        how_to_display = st.sidebar.selectbox('What to plot (line/bar)', ["line", "line_scaled_to_peak", "line_first_is_100", "bar"], index=0)
    else:
        how_to_display = "bar"



    global showR
    global WDW3
    global MOVE_WR
    global lijst   # Lijst in de pull down menu's voor de assen
    lijst = ['IC_Bedden_COVID', 'IC_Bedden_Non_COVID', 'Kliniek_Bedden',
        'IC_Nieuwe_Opnames_COVID',"Hospital_admission_RIVM",
        "Hospital_admission_LCPS",  "Hospital_admission_GGD",
        'Total_reported', 'Deceased',
        'Rt_avg',
        'Tested_with_result', 'Tested_positive', 'Percentage_positive',
        'prev_avg',
        "retail_and_recreation", "grocery_and_pharmacy", "parks", "transit_stations", "workplaces",
        "residential", "apple_driving", "apple_transit", "apple_walking",
        "temp_etmaal","temp_max","zonneschijnduur","globale_straling","specific_humidity","neerslag","RNA_per_ml",
   "RNA_flow_per_100000" , 'RNA_per_reported'     ]
    #print (lijst)
    if how_to_display != "bar":
        what_to_show_day_l = st.sidebar.multiselect('What to show left-axis (multiple possible)', lijst, ["Total_reported"]  )
        what_to_show_day_r = st.sidebar.multiselect('What to show right-axis (multiple possible)', lijst)
        if what_to_show_day_l == None:
            st.error ("CHoose something")
            st.stop()
        move_right = st.sidebar.slider('Move curves at right axis (days)', -14, 14, 0)

    else:
        move_right = 0
    showR = False
    if how_to_display == "bar":
        what_to_show_day_l = st.sidebar.selectbox('What to show left-axis (bar -one possible)',lijst, index=5  )
        #what_to_show_day_l = st.sidebar.multiselect('What to show left-axis (multiple possible)', lijst, ["Total_reported"]  )

        showR = st.sidebar.selectbox('Show R number',[True, False], index=0)
        if what_to_show_day_l == []:
            st.error ("Choose something for the left-axis")
        if showR == False:
            what_to_show_day_r =  st.sidebar.multiselect('What to show right-axis (multiple possible)', lijst, ["Total_reported"])
        else:
            what_to_show_day_r = None
            pass # what_to_show_day_r = st.sidebar.selectbox('What to show right-axis (line - one possible)',lijst, index=6)
        lijst_x = [0,1,2,3,4,5,6]
        showoneday = st.sidebar.selectbox('Show one day',[True, False], index=0)
        if showoneday:
            showday= st.sidebar.selectbox('Show which day',lijst_x, index=0  )


    how_to_smoothen = st.sidebar.selectbox('How to smooth (SMA/savgol)', ["SMA", "savgol"], index=0)
    WDW2 = st.sidebar.slider('Window smoothing curves (days)', 1, 14, 7)
    if how_to_smoothen== "savgol" and int(WDW2/2)==(WDW2/2):
        st.warning ("When using Savgol, the window has to be uneven")
        st.stop()
    if showR == True:
        #st.sidebar.multiselect('What to show left-axis (multiple possible)', lijst, ["Total_reported"]
        WDW3 =  st.sidebar.slider('Window smoothing R-number', 1, 14, 7)
        MOVE_WR = st.sidebar.slider('Move the R-curve', -20, 10, -8)
    else:
        showR = False


    #week_or_day = "day"
    if week_or_day == "week":
        how_to_agg_l = st.sidebar.selectbox('How to agg left (sum/mean)', ["sum", "mean"], index=0)
        how_to_agg_r = st.sidebar.selectbox('How to agg right (sum/mean)', ["sum", "mean"], index=0)

    global show_scenario

    show_scenario = st.sidebar.selectbox('Show Scenario',[True, False], index=1)
    if show_scenario:

        global Rnew1_, Rnew2_
        global ry1, ry2,  total_cases_0, sec_variant,extra_days

        total_cases_0 = st.sidebar.number_input('Total number of positive tests',None,None,8000)

        Rnew_1_ = st.sidebar.slider('R-number first variant', 0.1, 10.0, 0.84)
        Rnew_2_ = st.sidebar.slider('R-number second variant', 0.1, 6.0, 1.15)
        f = st.sidebar.slider('Correction factor', 0.0, 2.0, 1.00)
        ry1 = round(Rnew_1_ * f,2)
        ry2 = round(Rnew_2_ * f,2)
        sec_variant = (st.sidebar.slider('Percentage second variant at start', 0.0, 100.0, 43.0))
        extra_days = st.sidebar.slider('Extra days', 0, 60, 0)
        #calculate_cases(df, Rnew1_, Rnew2_, correction, numberofpositivetests, percentagenewversion)
    global how_to_norm
    #how_to_agg_l = "sum"
    #how_to_agg_r = "mean"
    if what_to_show_day_l == []:
            st.error ("Choose something for the left-axis")
            st.stop()

    if what_to_show_day_l is not None:

        if week_or_day == "day"  :
            if move_right != 0 and  len(what_to_show_day_r) != 0:
                df, what_to_show_day_r = move_column(df, what_to_show_day_r,move_right  )
            if how_to_display == "line":
                graph_daily       (df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
            elif how_to_display == "line_scaled_to_peak":
                how_to_norm = "max"
                print(how_to_norm)
                graph_daily_normed(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
            elif how_to_display == "line_first_is_100":
                how_to_norm = "first"
                print(how_to_norm)
                graph_daily_normed(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
            elif how_to_display == "bar":
                #st.write(what_to_show_day_l)
                graph_daily        (df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)

        else:
            if showR == True:
                if what_to_show_day_r != None:
                    st.warning ("On the right axis the R number will shown")
                graph_week(df, what_to_show_day_l , how_to_agg_l, None , how_to_agg_r)
            else:
                graph_week(df, what_to_show_day_l , how_to_agg_l, what_to_show_day_r , how_to_agg_r)

    else:
        st.error ("Choose what to show")
    # show a weekgraph, options have to put in the function self, no options her (for now)


    #find_correlations(df)
    #find_lag_time(df,"transit_stations","Rt_avg", 0,10)


    toelichting = ('<h2>Toelichting bij de keuzevelden</h2>'
    '<i>IC_Bedden_COVID</i> - Aantal bezette bedden met COVID patienten (LCPS)'
     '<br><i>IC_Bedden_Non_COVID</i> - Totaal aantal bezette bedden (LCPS) '
     '<br><i>Kliniek_Bedden</i> - Totaal aantal ziekenhuisbedden (LCPS)'
       '<br><i>IC_Nieuwe_Opnames_COVID</i> - Nieuwe opnames op de IC '
        '<br><br><i>Hospital_admission_LCPS</i> - Nieuwe opnames in de ziekenhuizen LCPS. Vanaf oktober 2020. Verzameld op geaggreerd niveau en gericht op bezetting '

        '<br><i>Hospital_admission_RIVM</i> - Nieuwe opnames in de ziekenhuizen RIVM door NICE. Is in principe gelijk aan het officiele dashboard. Bevat ook mensen die wegens een andere reden worden opgenomen maar positief getest zijn.'
        '<br><i>Hospital_admission_GGD</i> - Nieuwe opnames in de ziekenhuizen GGD, lager omdat niet alles vanuit GGD wordt doorgegeven '

       '<br><br><i>Total_reported</i> - Totaal aantal gevallen (GGD + ..?.. ) '

       '<br><i>Deceased</i> - Totaal overledenen '
       '<br><i>Rt_avg</i> - Rt-getal berekend door RIVM'
       '<br><i>Tested_with_result</i> - Totaal aantal testen bij GGD '
       '<br><i>Tested_positive</i> - Totaal aantal positief getesten bij GGD '
       '<br><i>Percentage_positive</i> - Percentage positief getest bij de GGD '
       '<br><i>prev_avg</i> - Aantal besmettelijke mensen.'
       '<br><br><i>retail_and_recreation, grocery_and_pharmacy, parks, transit_stations, workplaces, '
        'residential</i> - Mobiliteitsdata van Google'
        '<br><i>apple_driving, apple_transit, apple_walking</i> - Mobiliteitsdata van Apple'
        '<br><br><i>temp_etmaal</i> - Etmaalgemiddelde temperatuur (in graden Celsius)'
       '<br><i>temp_max</i> - Maximale temperatuur (in graden Celsius)'
        '<br><br><i>Zonneschijnduur</i> - Zonneschijnduur (in 0.1 uur) berekend uit de globale straling'
        '<br><i>Globale straling</i> - Globale straling in (in J//cm2) '
        '<br><i>Neerslag</i> - Etmaalsom van de neerslag (in 0.1 mm) (-1 voor  minder dan 0.05 mm) '
        '<br><i>Specific Humidity</i> -  - Specific Humidity in (g/kg)'
        '<br><br><i>RNA_per_ml</i> - Rioolwater tot 9/9/2020'
        '<br><i>RNA_flow_per_100000</i> - Rioolwater vanaf 9/9/2020'
       '<h2>Toelichting bij de opties</h2>'
       '<h3>What to plot</h3>'
       '<i>Line</i> - Line graph'
       '<br><i>Linemax</i> - Indexed line grap. Maximum (smoothed) value is 100'
       '<br><i>Linefirst</i> - Indexed line graph. First (smoothed) value is 100'
       '<br><i>Bar</i> - Bar graph for the left axis, line graph for the right ax'
        '<h3>How to smooth</h3>'
       '<i>SMA</i> - Smooth moving average. <br><i>savgol</i> - <a href=\'https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter\' target=\'_bank\'>Savitzky-Golay filter</a>'
       '<h3>Correlation</h3>'
       'If you have chosen one field on the left side and one for the right side, correlation of the fields are shown. Attention: <i>correlation is not causation</i>!'

       '<h3>Move curves at right axis (days)</h3>'
       'You can move the curves at the right ax to see possible cause-effect relations.'
       '<h3>Show Scenario</h3>'
       'You are able to calculate a scenario based on two R-numbers, their ratio, a correction factor (to put in effect measures) and add extra days. Works only with [total reported].'
       'You can calculate scenarios with more options and graphs at my other webapp <a href=\'https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py\' target=\'_blank\'>https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py</a>'
       '<h2>Datasource</h2>'
       'Data is scraped from https://data.rivm.nl/covid-19/ and LCPS and cached. '
       ' <a href=/"https://coronadashboard.rijksoverheid.nl/verantwoording#ziekenhuizen/" target=/"_blank/">Info here</a>.<br>'
       'For the moment most of the data is be updated automatically every 24h.'
       ' The KNMI, Google and Apple data will be updated manually at a lower frequency.<br><br>')

    tekst = (
    '<style> .infobox {  background-color: lightblue; padding: 5px;}</style>'
    '<hr><div class=\'infobox\'>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\' target=\"_blank\">@rcsmit</a>) <br>'
    'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/edit/main/covid_dashboard_rcsmit.py\" target=\"_blank\">github.com/rcsmit</a><br>'
    'How-to tutorial : <a href=\"https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d\" target=\"_blank\">rcsmit.medium.com</a><br>'
    'Restrictions by <a href=\"https://twitter.com/hk_nien" target=\"_blank\">Han-Kwang Nienhuys</a> (MIT-license).</div>')

    st.markdown(toelichting, unsafe_allow_html=True)
    st.sidebar.markdown(tekst, unsafe_allow_html=True)
    now = UPDATETIME
    UPDATETIME_ = now.strftime("%d/%m/%Y %H:%M:%S")
    st.write(f"\n\n\nData last updated : {str(UPDATETIME_)}")
    st.markdown('<hr>', unsafe_allow_html=True)

    st.image('https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png')

    st.markdown('<a href=\"https://www.buymeacoffee.com/rcsmit" target=\"_blank\">If you are happy with this dashboard, you can buy me a coffee</a>', unsafe_allow_html=True)

    st.markdown('<br><br><a href=\"https://www.linkedin.com/in/rcsmit" target=\"_blank\">Contact me for custom dashboards and infographics</a>', unsafe_allow_html=True)

init()
main()
# https://www.medrxiv.org/content/10.1101/2020.05.06.20093039v3.full
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7729173/

# 'With help of <a href=\"https://twitter.com/hk_nien" target=\"_blank\">Han-Kwang Nienhuys</a> and others<br>.'
#
