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

import urllib
import urllib.request
from pathlib import Path

from inspect import currentframe, getframeinfo

# R-numbers from 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
# Google mobilty from https://www.google.com/covid19/mobility/?hl=nl
# Apple mobility from https://covid19.apple.com/mobility
# # Merged in one file in Excel and saved to CSV
# Hospitals from RIVM 'https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv



def download_mob_r():
    """  _ _ _ """
    df_mob_r = pd.read_csv(
                    r'covid19_seir_models\input\mobility.csv',
                    delimiter=';',
                    low_memory=False
                )
    # datum is 16-2-2020
    df_mob_r['date']=pd.to_datetime(df_mob_r['date'], format="%d-%m-%Y")
    df_mob_r.set_index('date')
    return df_mob_r

def download_hospital_admissions():
    """  _ _ _ """
    # THIS ARE THE SAME NUMBERS AS ON THE DASHBOARD

    if download :
        # Code by Han-Kwang Nienhuys - MIT License
        url='https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv'
        #url="https://lcps.nu/wp-content/uploads/covid-19.csv"
        fpath = Path('covid19_seir_models\input\COVID-19_ziekenhuisopnames.csv')
        print(f'Getting new daily case statistics file ziekenhuisopnames. ..')
        with urllib.request.urlopen(url) as response:
            data_bytes = response.read()
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')

    df_hospital = pd.read_csv(
                    r'covid19_seir_models\input\COVID-19_ziekenhuisopnames.csv',
                    delimiter=';',
                    #delimiter=',',
                    low_memory=False

                )

    # datum is 2020-02-27
    df_hospital['Date_of_statistics'] = df_hospital['Date_of_statistics'].astype('datetime64[D]')

    df_hospital = df_hospital.groupby(['Date_of_statistics'] , sort=True).sum().reset_index()
    #print ("Last hospital admissions :
    #       " (df_hospital.iloc[len (df_hospital)]['Date_of_statistics'])
    # compression_opts = dict(method=None,
    #                         archive_name='out.csv')
    # df_hospital.to_csv('outhospital.csv', index=False,
    #         compression=compression_opts)

    #print (df_hospital)
    #save_df(df_hospital,"ziekenhuisopnames_RIVM")
    return df_hospital

def download_lcps():
    """Download data from LCPS"""

    if download :
        # Code by Han-Kwang Nienhuys - MIT License
        url='https://lcps.nu/wp-content/uploads/covid-19.csv'
        #url="https://lcps.nu/wp-content/uploads/covid-19.csv"
        fpath = Path('covid19_seir_models\input\LCPS.csv')
        print(f'Getting new daily case statistics file LCPS...')
        with urllib.request.urlopen(url) as response:
            data_bytes = response.read()
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')

    df_lcps = pd.read_csv(
                    r'covid19_seir_models\input\LCPS.csv',
                    delimiter=',',
                    #delimiter=',',
                    low_memory=False

                )
    # print (df_lcps)
    # print (df_lcps.dtypes)
    # Datum,IC_Bedden_COVID,IC_Bedden_Non_COVID,Kliniek_Bedden,IC_Nieuwe_Opnames_COVID,
    # Kliniek_Nieuwe_Opnames_COVID
    # datum is 2020-02-27
    df_lcps['Datum']=pd.to_datetime(df_lcps['Datum'], format="%d-%m-%Y")

    #df_lcps = df_lcps.groupby(['Datum'] , sort=True).sum().reset_index()

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
        #df_reprogetal = pd.read_json (r'covid19_seir_models\input\COVID-19_reproductiegetal.json')
        df_reprogetal = pd.read_json (r'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json')
        # url = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'

        compression_opts = dict(method=None,
                            archive_name='reprogetal.csv')
        df_reprogetal.to_csv('covid19_seir_models\\input\\reprogetal.csv', index=False,
            compression=compression_opts)
        df_reprogetal.set_index("Date")
    else:
        df_reprogetal = pd.read_csv(
                    r'covid19_seir_models\input\reprogetal.csv',
                    delimiter=',',
                    #delimiter=',',
                    low_memory=False)


    df_reprogetal['Date']=pd.to_datetime(df_reprogetal['Date'], format="%Y-%m-%d")

    # als er nog geen reprogetal berekend is, neem dan het gemiddeld van low en up
    # vanaf half juni is dit altijd het geval geweest (0,990 en 1,000)

    #df_reprogetal.loc[df_reprogetal["Rt_avg"].isnull(),'Rt_avg'] = round(((df_reprogetal["Rt_low"] + df_reprogetal["Rt_up"])/2),2)

    #print (df_reprogetal)
    return df_reprogetal

def download_gemeente_per_dag():
    """  _ _ _ """

    if download :
        # Code by Han-Kwang Nienhuys - MIT License
        url='https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv'

        # Het werkelijke aantal COVID-19 patiënten opgenomen in het ziekenhuis is hoger dan
        # het aantal opgenomen patiënten gemeld in de surveillance, omdat de GGD niet altijd op
        # de hoogte is van ziekenhuisopname als deze na melding plaatsvindt.
        # Daarom benoemt het RIVM sinds 6 oktober actief de geregistreerde ziekenhuisopnames
        # van Stichting NICE

        #url="https://lcps.nu/wp-content/uploads/covid-19.csv"

        fpath = Path('covid19_seir_models\input\COVID-19_aantallen_gemeente_per_dag.csv')
        print(f'Getting new daily case statistics file - aantallen-gemeente-per-dag - ...')
        with urllib.request.urlopen(url) as response:
            data_bytes = response.read()
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')

    df_gemeente_per_dag = pd.read_csv(
                    r'covid19_seir_models\input\COVID-19_aantallen_gemeente_per_dag.csv',
                    delimiter=';',
                    #delimiter=',',
                    low_memory=False)

    df_gemeente_per_dag['Date_of_publication'] = df_gemeente_per_dag['Date_of_publication'].astype('datetime64[D]')

    df_gemeente_per_dag = df_gemeente_per_dag.groupby(['Date_of_publication'] , sort=True).sum().reset_index()
    #save_df(df_gemeente_per_dag,"COVID-19_aantallen_per_dag")
    return df_gemeente_per_dag

def download_uitgevoerde_testen():
    """  _ _ _ """
    # Version;Date_of_report;Date_of_statistics;Security_region_code;
    # Security_region_name;Tested_with_result;Tested_positive

    if download :
        # Code by Han-Kwang Nienhuys - MIT License
        url='https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv'


        fpath = Path('covid19_seir_models\input\COVID-19_uitgevoerde_testen.csv')
        print(f'Getting new daily case statistics file - testen - ...')
        with urllib.request.urlopen(url) as response:
            data_bytes = response.read()
            fpath.write_bytes(data_bytes)
            print(f'Wrote {fpath} .')

    df_uitgevoerde_testen = pd.read_csv(
                    r'covid19_seir_models\input\COVID-19_uitgevoerde_testen.csv',
                    delimiter=';',
                    #delimiter=',',
                    low_memory=False)

    #df_uitgevoerde_testen['Date_of_publication'] = df_uitgevoerde_testen['Date_of_publication'].astype('datetime64[D]')
    df_uitgevoerde_testen['Date_of_statistics'] = df_uitgevoerde_testen['Date_of_statistics'].astype('datetime64[D]')

    df_uitgevoerde_testen = df_uitgevoerde_testen.groupby(['Date_of_statistics'] , sort=True).sum().reset_index()
    df_uitgevoerde_testen['Percentage_positive'] = round((df_uitgevoerde_testen['Tested_positive'] /
                                                  df_uitgevoerde_testen['Tested_with_result'] * 100),2 )

    #save_df(df_uitgevoerde_testen,"COVID-19_uitgevoerde_testen")
    return df_uitgevoerde_testen
###################################################################
def get_data():
    """  _ _ _ """

    df_hospital = download_hospital_admissions()
    #sliding_r_df = walkingR(df_hospital, "Hospital_admission")
    df_lcps = download_lcps()
    df_mob_r = download_mob_r()
    df_gemeente_per_dag = download_gemeente_per_dag()
    df_reprogetal = download_reproductiegetal()
    df_uitgevoerde_testen = download_uitgevoerde_testen()

    type_of_join = "outer"
    df = pd.merge(df_mob_r, df_hospital, how=type_of_join, left_on = 'date',
                    right_on="Date_of_statistics")
    #df = df_hospital
    df.loc[df['date'].isnull(),'date'] = df['Date_of_statistics']
    df = pd.merge(df, df_lcps, how=type_of_join, left_on = 'date', right_on="Datum")
    df.loc[df['date'].isnull(),'date'] = df['Datum']
    #df = pd.merge(df, sliding_r_df, how=type_of_join, left_on = 'date', right_on="date_sR", left_index=True )

    df = pd.merge(df, df_gemeente_per_dag, how=type_of_join, left_on = 'date', right_on="Date_of_publication",
                    left_index=True )

    df = pd.merge(df, df_reprogetal, how=type_of_join, left_on = 'date', right_on="Date",
                    left_index=True )
    df = pd.merge(df, df_uitgevoerde_testen, how=type_of_join, left_on = 'date', right_on="Date_of_statistics",
                    left_index=True )


    df = df.sort_values(by=['date'])
    df = splitupweekweekend(df)
    df, werkdagen, weekend_ = last_manipulations(df, None, None)
    df.set_index('date')

    return df, werkdagen, weekend_

###################################################
def calculate_cases(df):
    column = df["date"]
    b_= column.max().date()
    #fr = '2021-1-10' #doesnt work
    fr = FROM
    a_ = dt.datetime.strptime(fr,'%Y-%m-%d').date()

    #b_ = dt.datetime.strptime(UNTIL,'%Y-%m-%d').date()
    datediff = ( abs((a_ - b_).days))+1+30
    f = 1
    ry1 = 0.8 * f
    ry2 = 1.15 * f
    total_cases_0 = 7500
    sec_variant = 10
    population = 17_500_000
    immune_day_zero = 2_500_000

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
        immeratio = (1-(cumm_cases/suspectible_0 ))

    df_calculated['date_calc'] = pd.to_datetime( df_calculated['date_calc'])

    df = pd.merge(df, df_calculated, how='outer', left_on = 'date', right_on="date_calc",
                        left_index=True )
    print (df.dtypes)
    df.loc[df['date'].isnull(),'date'] = df['date_calc']
    return df, ry1, ry2

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

        d= 1
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
        sliding_r_df[column_name_r_smoothened] = round(sliding_r_df.iloc[:,1].rolling(window=WDW3).mean(),2)
        sliding_r_df[column_name_r_sec_smoothened] = round(sliding_r_df.iloc[:,2].rolling(window=WDW3).mean(),2)

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

    df['weeknr']  =  df['date'].dt.isocalendar().week
    df['yearnr']  =  df['date'].dt.isocalendar().year

    df['weekalt']   = (df['date'].dt.isocalendar().year.astype(str) + "-"+
                         df['date'].dt.isocalendar().week.astype(str))
    #how = "mean"
    if how == "mean":
        dfweek = df.groupby('weekalt', sort=False).mean()
    elif how == "sum" :
        dfweek = df.groupby('weekalt', sort=False).sum()
    else:
        print ("error agg_week()")
        exit()
    return df, dfweek

def move_column(df, column,days):
    """  _ _ _ """
    # #move Rt r days, because a maybe change has effect after r days
    # Tested - Conclusion : It has no effect
    r=days
    print (column)
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

    if show_from == None:
        show_from = '2020-1-1'

    if show_until == None:
        show_until = '2030-1-1'
    mask = (df['date'] >= show_from) & (df['date'] <= show_until)
    df = (df.loc[mask])

    df = df.reset_index()

    return df

def last_manipulations(df, what_to_drop, drop_last):
    """  _ _ _ """
    print ("Doing some last minute manipulations")
    drop_columns(what_to_drop)

    #
    # select_period(df, FROM, UNTIL)
    df = select_period(df, FROM, UNTIL)

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

    df['weeknr']  =  df['date'].dt.isocalendar().week
    df['yearnr']  =  df['date'].dt.isocalendar().year

    df['weekalt']   = (df['date'].dt.isocalendar().year.astype(str) + "-"+
                         df['date'].dt.isocalendar().week.astype(str))

    return df, werkdagen, weekend_
    #return df

def save_df(df,name):
    """  _ _ _ """
    name_ = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\" + name+'.csv'
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
        maxvalue = (df[column].max())
        firstvalue = df[column].iloc[0]/100
        name = (f"{column}_normed")

        for i in range(0,len(df)):
            df.loc[i, name] = df.loc[i,column]/firstvalue
        normed_columns.append(name)
        print (f"{name} generated")
    return df, normed_columns

def graph_daily_normed(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display):
    """ IN : df, de kolommen die genormeerd moeten worden
    ACTION : de grafieken met de genormeerde kolommen tonen """
    df, normed_columns = normeren(df, what_to_show_day_l)
    graph_daily(df,normed_columns, what_to_show_day_r, how_to_smoothen, how_to_display)

def graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title,t):
    # print (df)

    show_variant = False  # show lines coming from the growth formula
    if show_variant == True:
        df, ry1, ry2 = calculate_cases(df)
    #print (df.dtypes)
    """  _ _ _ """
    if type(what_to_show_l) == list:
        what_to_show_l_=what_to_show_l
    else:
        what_to_show_l_=[what_to_show_l]
    aantal = len(what_to_show_l_)

    # SHOW A GRAPH IN TIME / DAY
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
    # df, R_smooth = add_walking_r(df, columnlist, how_to_smooth)
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

            # MAYBE WE CAN LEAVE THIS OUT HERE
            df, columnlist = smooth_columnlist(df,[b],how_to_smooth)

            df.set_index('date')


            #x= 4+ (WDW2+WDW3)*-1 # 4 days correction factor, dont know why
            #x=0
            #df, new_column = move_column(df,R_smooth,x)
            df_temp = df
            ax = df_temp[b].plot.bar(label=b, color = color_x, alpha=.6)     # number of cases
            for c_smooth in columnlist:
                # print (c_smooth)
                # print (df[c_smooth])
                ax = df[c_smooth].plot(label=c_smooth, color = color_list[2],linewidth=1)         # SMA


            #df_temp = select_period(df, FROM, UNTIL)
            #df_temp.set_index('date')

            #print(df_temp.dtypes)
            #c = str(b)+ "_"+how_to_smooth+ "_" + str(WDW2)

            ax3=df["Rt_avg"].plot(secondary_y=True,linestyle='--', label="Rt RIVM",color=green_pigment, alpha=.8,linewidth=1)
            tgs = [3.5,4,5]
            teller=0
            for TG in tgs:

                df, R_smooth, R_smooth_sec = add_walking_r(df, columnlist, how_to_smooth,TG)



                for R in R_smooth:
                    # correctie R waarde, moet naar links ivm 2x smoothen
                    df, Rn = move_column(df,R,MOVE_WR)
                    if teller == 1 :
                        ax3=df[Rn].plot(secondary_y=True, label=Rn,linestyle='--',color=falu_red, linewidth=1.2)
                    else:
                        ax3=df[Rn].plot(secondary_y=True,label='_nolegend_', linestyle='dotted',color=falu_red, linewidth=0.8)
                    teller += 1

                for R in R_smooth_sec:  # SECOND METHOD TO CALCULATE R
                    # correctie R waarde, moet naar links ivm 2x smoothen
                    df, Rn = move_column(df,R,MOVE_WR)
                    #ax3=df[Rn].plot(secondary_y=True, label=Rn,linestyle='--',color=operamauve, linewidth=1)

            #ax3.fill_between(x, y1, y2)

            ax3.set_ylabel('Rt')

            left, right = ax.get_xlim()
            ax.set_xlim(left - 0.5, right + 0.5)
            #ax3.set_ylim(0.6,1.5)

        elif t== "line":
            #df, R_SMA = add_walking_r(df, b, "SMA")

            #df_temp = select_period(df, FROM, UNTIL)
            df_temp = df
            #print (df_temp)
            #df_temp = df
            b_ = str(b)+ "_"+how_to_smooth+ "_" + str(WDW2)
            df_temp[b_].plot(label=b, color = color_list[n], linewidth=1.1) # label = b_ for uitgebreid label
            df_temp[b].plot(label='_nolegend_', color = color_list[n],linestyle='dotted',alpha=.8,linewidth=.8)
        else:
            print ("ERROR in graph_day")
        n +=1
    if show_variant == True:
        l1 = (f"R = {ry1}")
        l2 = (f"R = {ry2}")
        ax = df["variant_1"].plot(label=l1, color = color_list[4],linestyle='dotted',linewidth=1, alpha=1)
        ax = df["variant_2"].plot(label=l2, color = color_list[5],linestyle='dotted',linewidth=1, alpha=1)
        ax = df["variant_12"].plot(label='TOTAL', color = color_list[6],linestyle='--',linewidth=1, alpha=1)

    df["testvshospital"] = (df["Hospital_admission_x"]/df["Total_reported"]*100)
    df["testvsIC"] = (df["IC_Nieuwe_Opnames_COVID"]/df["Total_reported"]*100)
    #print (df["testvsIC"])
    #save_df(df, "testvsIC")
    #ax3 = df["testvsIC"].plot(label="testvic", color = color_list[4],linestyle='dotted',linewidth=1, alpha=1)

    ax.set_ylabel('Numbers')
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_major_locator(MultipleLocator(1))

    if what_to_show_r != None:
        n = len (color_list)

        x = n
        for a in what_to_show_r:
            x -=1
            lbl = a + " (right ax)"
            #lbl2 = a + "_" + how_to_smooth + "_" + str(WDW2)

            df, columnlist = smooth_columnlist(df,[a],how_to_smooth)
            df_temp = select_period(df, FROM, UNTIL)
            #df_temp = df
            for b_ in columnlist: #smoothed
                lbl2 = b_ + " (right ax)"
                ax3 = df_temp[b_].plot(secondary_y=True, label=lbl, color = color_list[x], linestyle='--', linewidth=.8) #abel = lbl2 voor uitgebreid label
            pass # ax3=df_temp[a].plot(secondary_y=True, linestyle='dotted', color = color_list[x], linewidth=.8, '_nolegend_')
            ax3.set_ylabel('_')

    # layout of the x-axis
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(df_temp['date'].index)
    ax.set_xticklabels(df_temp['date'].dt.date,fontsize=6, rotation=90)
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
    xticks = ax.xaxis.get_major_ticks()

    for i,tick in enumerate(xticks):
        if i%10 != 0:
            tick.label1.set_visible(False)
    plt.xticks()

    fontP = FontProperties()
    fontP.set_size('xx-small')
    plt.legend( bbox_to_anchor=(1.05, 1), loc=2,fontsize=6, prop=fontP)
    plt.xlabel('date')

    # Add a grid
    plt.grid(alpha=.4,linestyle='--')


    plt.title(title , fontsize=10)

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
                    r'covid19_seir_models\input\restrictions.csv',
                    delimiter=',',
                    low_memory=False)


    a = (min(df['date'].tolist())).date()
    #a = ((df['date_sR'].dt.date.min()))  # werkt niet, blijkbaar NaN values

    ymin, ymax = ax.get_ylim()
    y_lab = ymin
    for i in range(0, len(df_restrictions)):
        d_ = df_restrictions.iloc[i]['Date'] #string
        d__ = dt.datetime.strptime(d_,'%Y-%m-%d').date()  # to dateday

        diff = (d__ - a)

        if diff.days >0 :
            # no idea why diff.days-2
            ax.text((diff.days), y_lab, f'  {df_restrictions.iloc[i]["Description"] }', rotation=90, fontsize=4,horizontalalignment='center')
            #plt.axvline(x=(diff.days), color='yellow', alpha=.3,linestyle='--')


def graph_week(df, what_to_show_l, how_l, what_to_show_r, how_r):
    """  _ _ _ """
    #save_df(dfweek,"weektabel")
    # SHOW A GRAPH IN TIME / WEEK
    df_l, dfweek_l = agg_week(df, how_l)
    if what_to_show_r != None:
        df_r, dfweek_r = agg_week (df, how_r)
    for show_l in what_to_show_l:

        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)
        ax.set_xticks(dfweek_l['weeknr'])
        ax.set_xticklabels(dfweek_l['weeknr'] ,fontsize=6, rotation=45)
        label_l = show_l+ " ("+ how_l + ")"
        dfweek_l[show_l].plot.bar(label=label_l)

        if what_to_show_r != None:
            for what_to_show_r_ in what_to_show_r:
                label_r = what_to_show_r_+ " ("+ how_r + ")"
                ax3=dfweek_r[what_to_show_r_].plot(secondary_y=True, color = 'r', label=label_r)

        #ax3=dfweek['Rt_avg_SMA_7'] .plot(secondary_y=True, color = 'r', label="Rt_avg_SMA_7")

        #ax3.set_ylabel('Rt')
        #ax.set_ylabel('Numbers')

        plt.xlabel('week')

        # Add a grid
        plt.grid(alpha=.4,linestyle='--')

        #Add a Legend
        fontP = FontProperties()
        fontP.set_size('xx-small')
        plt.legend(  loc='best', prop=fontP)
        #titlex = "Weekly numbers - " + show_l + "(" + how  + ")"
        #plt.title(titlex , fontsize=10)

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


        # configgraph(titlex)
        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')

        plt.show()

def graph_daily(df, what_to_show_l, what_to_show_r,how_to_smooth,t):
    """  _ _ _ """

    if t == "line":
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
    elif t == "bar":
        for c in what_to_show_l:
            what_to_show_l= (c)
            what_to_show_r = what_to_show_r
            title = c
            graph_day(df, what_to_show_l,what_to_show_r , how_to_smooth, title, t)
    else:
        print ("ERROR IN graph_daily")
        exit()

def smooth_columnlist(df,columnlist_,t):
    """  _ _ _ """
    c_smoothen = []
    wdw_savgol = 7
    #print (columnlist_)
    for c in columnlist_:
        print (c)
        if t=='SMA':
            new_column = c + '_SMA_' + str(WDW2)
            print ('Generating ' + new_column+ '...')
            df[new_column] = df.iloc[:,df.columns.get_loc(c)].rolling(window=WDW2).mean()

        elif t=='savgol':
            new_column = c + '_savgol_' + str(wdw_savgol)
            print ('Generating ' + new_column + '...')
            df[new_column] = df[c].transform(lambda x: savgol_filter(x, wdw_savgol,2))
        else:
            print ("ERROR in smooth_columnlist")
            exit()
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
    plt.grid(alpha=.4,linestyle='--')
    plt.title(title , fontsize=10)
    plt.show()
    graph_daily(df,[a],[b], "SMA","line")
    graph_daily(df,[a],[max_column], "SMA","line")
    # if the optimum is negative, the second one is that x days later

def init():
    """  _ _ _ """

    global download

    global WDW2
    global WDW3
    global FROM
    global UNTIL
    global MOVE_WR

    # GLOBAL SETTINGS
    download = False
    # De open data worden om 15.15 uur gepubliceerd

    WDW2 = 7 # for all values except R values
    WDW3 = 3 # for R values in add_walking_r
    MOVE_WR = -10 # How many days the R value has to move to correct SMA
    FROM = '2020-12-1'
    UNTIL = '2025-5-1'
    # attention, minimum period between FROM and UNTIL = wdw days!


def main():
    """  _ _ _ """
    init()
    # LET'S GET AND PREPARE THE DATA
    df, werkdagen, weekend_ = get_data()
    what_to_show_day_r = None
    what_to_show_week_r = None
    # COLUMNS
    # index,country_region_code,date,retail_and_recreation,grocery_and_pharmacy,parks,
    # transit_stations,workplaces,residential,apple_driving,apple_transit,apple_walking,
    # Date_of_statistics_x,Version_x,Hospital_admission_notification,Hospital_admission_x,
    # Datum,IC_Bedden_COVID,IC_Bedden_Non_COVID,Kliniek_Bedden,IC_Nieuwe_Opnames_COVID,
    # Kliniek_Nieuwe_Opnames_COVID,Date_of_publication,Total_reported,Hospital_admission_y,
    # Deceased,Date,Rt_low,Rt_avg,Rt_up,population,version,Date_of_statistics_y,Version_y,
    # Tested_with_result,Tested_positive,Percentage_positive,WEEKDAY

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
    columnlist1= ['Total_reported']
    #columnlist1= ['Kliniek_Nieuwe_Opnames_COVID','Total_reported']

    # save_df(df,"thisisallfolks")

    # FILL HERE WHAT YOU WANT TO SEE // PARAMETERS
    LCPS = ["Total_reported"]
    #what_to_show_day_r = ["IC_Nieuwe_Opnames_COVID","Kliniek_Nieuwe_Opnames_COVID"]
    #what_to_show_day_l = ["Total_reported" ,"Hospital_admission_x","Kliniek_Nieuwe_Opnames_COVID",    "Kliniek_Bedden","IC_Bedden_COVID", "IC_Nieuwe_Opnames_COVID", "Deceased"]
    #what_to_show_day_l = ["Total_reported", "Tested_with_result"]
    #what_to_show_day_l = LCPS

    #what_to_show_day_l =["Kliniek_Bedden","Total_reported"]
    #what_to_show_day_l = ["Kliniek_Nieuwe_Opnames_COVID"]
    #what_to_show_day_l= ["Hospital_admission_x"]
    what_to_show_day_l =["Total_reported" ] # shown n seperate graphs if it's bar
    #what_to_show_day_l = ["Total_reported", "IC_Nieuwe_Opnames_COVID"] #""Hospital_admission_x","Kliniek_Nieuwe_Opnames_COVID", "IC_Nieuwe_Opnames_COVID", "Deceased"]

    #what_to_show_day_r = ["Percentage_positive"] # always a line.
    #what_to_show_day_r = ["Rt_avg"]
    #what_to_show_day_r = None
    # Max 7 items
    how_to_smoothen = "SMA"         # "SMA" or "savgol"
    how_to_display = "bar"          # "line" (all in 1 linegraph) or "bar" (different bargraphs)

    #what_to_show_week_l = ["Hospital_admission_x"]
    #what_to_show_week_l = ["Total_reported" ,"Hospital_admission_x","Kliniek_Nieuwe_Opnames_COVID","Kliniek_Bedden","IC_Bedden_COVID", "IC_Nieuwe_Opnames_COVID", "Deceased"]

    how_to_agg_l = "sum"
    what_to_show_week_r = ["Percentage_positive"]
    how_to_agg_r = "mean"

    if how_to_display == "bar":
        what_to_show_day_r = None

    graph_daily(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
    #graph_daily_normed(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
    #normeren(df, ["Total_reported"])

    # show a weekgraph, options have to put in the function self, no options her (for now)
    #graph_week(df, what_to_show_week_l , how_to_agg_l, what_to_show_week_r , how_to_agg_r)
    #find_correlations(df)
    #find_lag_time(df,"transit_stations","Rt_avg", 0,10)

main()

# https://www.medrxiv.org/content/10.1101/2020.05.06.20093039v3.full
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7729173/
