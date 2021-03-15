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


# TO DO
# weekgrafieken
#  Series.dt.isocalendar().week
# first value genormeerde grafiek
# bredere grafiek
# leegmaken vd cache
# kiezen welke R je wilt in de bargraph
# waarde dropdown anders dan zichtbaar

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


@st.cache()
def download_hospital_admissions():
    """  _ _ _ """
    # THIS ARE THE SAME NUMBERS AS ON THE DASHBOARD

    url='https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv'
    #url="https://lcps.nu/wp-content/uploads/covid-19.csv"
    csv = INPUT_DIR + 'COVID-19_ziekenhuisopnames.csv'
    delimiter_ = ";"

    df_hospital = download_csv_file(url, csv,delimiter_)
    # datum is 2020-02-27
    df_hospital['Date_of_statistics'] = df_hospital['Date_of_statistics'].astype('datetime64[D]')

    df_hospital = df_hospital.groupby(['Date_of_statistics'] , sort=True).sum().reset_index()

    #save_df(df_hospital,"ziekenhuisopnames_RIVM")
    return df_hospital

@st.cache()
def download_lcps():
    """Download data from LCPS"""

    url='https://lcps.nu/wp-content/uploads/covid-19.csv'
    csv = INPUT_DIR + 'LCPS.csv'
    delimiter_ = ','

    df_lcps = download_csv_file(url, csv,delimiter_)

    # Datum,IC_Bedden_COVID,IC_Bedden_Non_COVID,Kliniek_Bedden,IC_Nieuwe_Opnames_COVID,
    # Kliniek_Nieuwe_Opnames_COVID
    # datumformat  is 2020-02-27
    df_lcps['Datum']=pd.to_datetime(df_lcps['Datum'], format="%d-%m-%Y")

    return df_lcps
@st.cache()
def download_reproductiegetal():
    """  _ _ _ """
    #https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json

    url = 'https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json'
    csv = 'covid19_seir_models\\input\\reprogetal.csv'
    delimiter_=','

    df_reprogetal = download_csv_file(url, csv, delimiter_)
    print (df_reprogetal)
    df_reprogetal['Date']=pd.to_datetime(df_reprogetal['Date'], format="%Y-%m-%d")

    # als er nog geen reprogetal berekend is, neem dan het gemiddeld van low en up
    # vanaf half juni is dit altijd het geval geweest (0,990 en 1,000)

    #df_reprogetal.loc[df_reprogetal["Rt_avg"].isnull(),'Rt_avg'] = round(((df_reprogetal["Rt_low"] + df_reprogetal["Rt_up"])/2),2)

    return df_reprogetal
@st.cache()
def download_gemeente_per_dag():
    """  _ _ _ """
    # Code by Han-Kwang Nienhuys - MIT License
    # Het werkelijke aantal COVID-19 patiënten opgenomen in het ziekenhuis is hoger dan
    # het aantal opgenomen patiënten gemeld in de surveillance, omdat de GGD niet altijd op
    # de hoogte is van ziekenhuisopname als deze na melding plaatsvindt.
    # Daarom benoemt het RIVM sinds 6 oktober actief de geregistreerde ziekenhuisopnames
    # van Stichting NICE

    url='https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv'
    csv = INPUT_DIR + 'COVID-19_aantallen_gemeente_per_dag.csv'
    delimiter_ =';'

    df_gemeente_per_dag = download_csv_file(url, csv, delimiter_)

    df_gemeente_per_dag['Date_of_publication'] = df_gemeente_per_dag['Date_of_publication'].astype('datetime64[D]')

    df_gemeente_per_dag = df_gemeente_per_dag.groupby(['Date_of_publication'] , sort=True).sum().reset_index()
    #save_df(df_gemeente_per_dag,"COVID-19_aantallen_per_dag")
    return df_gemeente_per_dag
@st.cache()
def download_uitgevoerde_testen():
    """  _ _ _ """
    # Version;Date_of_report;Date_of_statistics;Security_region_code;
    # Security_region_name;Tested_with_result;Tested_positive

    url='https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv'
    csv = INPUT_DIR + 'COVID-19_uitgevoerde_testen.csv'
    delimiter_ = ';'
    df_uitgevoerde_testen = download_csv_file(url, csv,delimiter_)

    df_uitgevoerde_testen['Date_of_statistics'] = df_uitgevoerde_testen['Date_of_statistics'].astype('datetime64[D]')
    df_uitgevoerde_testen = df_uitgevoerde_testen.groupby(['Date_of_statistics'] , sort=True).sum().reset_index()
    df_uitgevoerde_testen['Percentage_positive'] = round((df_uitgevoerde_testen['Tested_positive'] /
                                                  df_uitgevoerde_testen['Tested_with_result'] * 100),2 )

    #save_df(df_uitgevoerde_testen,"COVID-19_uitgevoerde_testen")
    return df_uitgevoerde_testen
@st.cache()
def dowload_nice():
    url= 'https://stichting-nice.nl/covid-19/public/intake-count/'
    csv = INPUT_DIR + 'intake_count.csv'
    delimiter_=','
@st.cache()
def download_prevalentie():
    url = 'https://data.rivm.nl/covid-19/COVID-19_prevalentie.json'
    csv = INPUT_DIR + 'prevalentie.csv'
    delimiter_=','
    df_prevalentie = download_csv_file(url, csv,delimiter_)
    df_prevalentie['Date'] = df_prevalentie['Date'].astype('datetime64[D]')
    return df_prevalentie
###################################################################
@st.cache()
def download_csv_file(url, csv,delimiter_):
    #df_temp = None
    print (f"Downloading {url}...")
    with st.spinner(f'Wait for it...{url}'):
        if download :
            if url[-3:]=='csv' :

                # fpath = Path(str(csv))
                # # Code by Han-Kwang Nienhuys - MIT License
                # print(f'Getting new daily case statistics file - {url} - ...')
                # with urllib.request.urlopen(url) as response:
                #     data_bytes = response.read()
                #     fpath.write_bytes(data_bytes)
                #     print(f'Wrote {fpath} .')

                df_temp = pd.read_csv(url,
                        delimiter=delimiter_,
                        low_memory=False)
            elif url[-4:]=='json':
                print (f"Download {url}")
                df_temp = pd.read_json (url)

                # compression_opts = dict(method=None,
                #                     archive_name=csv)
                # df_json.to_csv(csv, index=False,
                #     compression=compression_opts)

            else:
                print ("Error in URL")
                st.stop()
        return df_temp
@st.cache()
def get_data():
    """  _ _ _ """

    df_hospital = download_hospital_admissions()
    #sliding_r_df = walkingR(df_hospital, "Hospital_admission")
    df_lcps = download_lcps()

    df_gemeente_per_dag = download_gemeente_per_dag()
    df_reprogetal = download_reproductiegetal()
    df_uitgevoerde_testen = download_uitgevoerde_testen()
    df_prevalentie = download_prevalentie()
    type_of_join = "outer"
    df = pd.merge(df_lcps, df_hospital, how=type_of_join, left_on = 'Datum',
                    right_on="Date_of_statistics")
    #df = df_hospital
    df.loc[df['Datum'].isnull(),'date'] = df['Date_of_statistics']

    #df.loc[df['date'].isnull(),'date'] = df['Datum']
    #df = pd.merge(df, sliding_r_df, how=type_of_join, left_on = 'date', right_on="date_sR", left_index=True )

    df = pd.merge(df, df_gemeente_per_dag, how=type_of_join, left_on = 'Datum', right_on="Date_of_publication")

    df = pd.merge(df, df_reprogetal, how=type_of_join, left_on = 'Datum', right_on="Date")

    df = pd.merge(df, df_uitgevoerde_testen, how=type_of_join, left_on = 'Datum', right_on="Date_of_statistics")

    df = pd.merge(df, df_prevalentie, how=type_of_join, left_on = 'Datum', right_on="Date")

    df["date"]=df["Datum"]

    df = df.sort_values(by=['date'])
    df = splitupweekweekend(df)

    # df.set_index('date')

    return df #, werkdagen, weekend_

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

    df['weeknr']  =  df['date'].dt.week
    df['yearnr']  =  df['date'].dt.year

    df['weekalt']   = (df['date'].dt.year.astype(str) + "-"+
                         df['date'].dt.week.astype(str))
    #how = "mean"
    if how == "mean":
        dfweek = df.groupby('weekalt', sort=False).mean()
    elif how == "sum" :
        dfweek = df.groupby('weekalt', sort=False).sum()
    else:
        print ("error agg_week()")
        st.stop()
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
    name_ = OUTPUT_DIR + name+'.csv'
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
        maxvalue = (df[column].max())/100
        firstvalue = df[column].iloc[0]/100
        name = (f"{column}_normed")

        for i in range(0,len(df)):
            if how_to_norm == "max":

                df.loc[i, name] = df.loc[i,column]/maxvalue
            else:
                df.name.loc[~df.name.isnull()].iloc[0]
                #df.loc[i, name] = df.loc[i,column]/firstvalue
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

    if what_to_show_r is not None:
        if len( what_to_show_l) ==1 and len( what_to_show_r)==1:
            correlation = find_correlation_pair(df, what_to_show_l, what_to_show_r)

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

                if showR :
                    ax3=df["Rt_avg"].plot(secondary_y=True,linestyle='--', label="Rt RIVM",color=green_pigment, alpha=.8,linewidth=1)
                    ax3.fill_between(df['date'].index, df["Rt_low"], df["Rt_up"],
                                color=green_pigment, alpha=0.3, label='_nolegend_')
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

            #elif t== "line":
            else:
                #df, R_SMA = add_walking_r(df, b, "SMA")

                #df_temp = select_period(df, FROM, UNTIL)
                df_temp = df

                if how_to_smooth == None:
                    how_to_smooth_ = "unchanged_"
                else:
                    how_to_smooth_ = how_to_smooth + "_" + str(WDW2)
                b_ = str(b)+ "_"+how_to_smooth_
                df_temp[b_].plot(label=b, color = color_list[n], linewidth=1.1) # label = b_ for uitgebreid label
                df_temp[b].plot(label='_nolegend_', color = color_list[n],linestyle='dotted',alpha=.7,linewidth=.8)
            # else:
            #     print ("ERROR in graph_day")
            n +=1
        if show_variant == True:
            l1 = (f"R = {ry1}")
            l2 = (f"R = {ry2}")
            ax = df["variant_1"].plot(label=l1, color = color_list[4],linestyle='dotted',linewidth=1, alpha=1)
            ax = df["variant_2"].plot(label=l2, color = color_list[5],linestyle='dotted',linewidth=1, alpha=1)
            ax = df["variant_12"].plot(label='TOTAL', color = color_list[6],linestyle='--',linewidth=1, alpha=1)

        #df["testvshospital"] = (df["Hospital_admission_x"]/df["Total_reported"]*100)
        #df["testvsIC"] = (df["IC_Nieuwe_Opnames_COVID"]/df["Total_reported"]*100)
        #print (df["testvsIC"])
        #save_df(df, "testvsIC")
        #ax3 = df["testvsIC"].plot(label="testvic", color = color_list[4],linestyle='dotted',linewidth=1, alpha=1)

        #ax.set_ylabel('Numbers')
        ax.xaxis.grid(True, which='major')
        ax.xaxis.set_major_locator(MultipleLocator(1))

        if what_to_show_r != None:
            n = len (color_list)

            x = n
            for a in what_to_show_r:
                x -=1
                lbl = a + " (right ax)"
                #lbl2 = a + "_" + how_to_smooth + "_" + str(WDW2)

                df, columnlist = smooth_columnlist(df,[a],how_to_smooth)
                #df_temp = select_period(df, FROM, UNTIL)
                #df_temp = df
                for b_ in columnlist:
                    #smoothed
                    lbl2 = a + " (right ax)"
                    ax3 = df_temp[b_].plot(secondary_y=True, label=lbl2, color = color_list[x], linestyle='--', linewidth=.8) #abel = lbl2 voor uitgebreid label
                ax3=df_temp[a].plot(secondary_y=True, linestyle='dotted', color = color_list[x], linewidth=.8, alpha=.7, label='_nolegend_')
                ax3.set_ylabel('_')

        # layout of the x-axis
        #ax.xaxis.grid(True, which='minor')
        ax.yaxis.grid(True, which='major')
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xticks(df['date'].index)
        ax.set_xticklabels(df['date'].dt.date,fontsize=6, rotation=90)
        # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=10))
        xticks = ax.xaxis.get_major_ticks()

        for i,tick in enumerate(xticks):
            if i%10 != 0:
                tick.label1.set_visible(False)
        plt.xticks()

        fontP = FontProperties()
        fontP.set_size('xx-small')

        plt.xlabel('date')

        # Add a grid
        plt.grid(alpha=.4,linestyle='--')
        if what_to_show_r is not None:
            if len( what_to_show_l) ==1 and len( what_to_show_r)==1:
                title = (f"{title} \nCorrelation = {correlation}")
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
        #ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4)) #here is the magic
        #plt.legend( bbox_to_anchor=(0.5, -0.4), loc=2,fontsize=6, prop=fontP)
        ax.text(1, 1.1, 'Created by: Rene Smit — @rcsmit',
                transform=ax.transAxes, fontsize='xx-small', va='top', ha='right')
        # configgraph(titlex)
        plt.axhline(y=1, color='yellow', alpha=.6,linestyle='--')
        #add_restrictions(df,ax)
        #plt.show()
        st.pyplot(fig1x)

def add_restrictions(df,ax):
    """  _ _ _ """
    # Add restrictions
    # From Han-Kwang Nienhuys - MIT-licence

    df_restrictions = pd.read_csv('https://github.com/rcsmit/COVIDcases/blob/main/restrictions.csv',
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

        #ax.xaxis.grid(True, which='minor')
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
    if t == "bar":
        if type(what_to_show_l) == list:
            what_to_show_l=what_to_show_l
        else:
            what_to_show_l=[what_to_show_l]
        for c in what_to_show_l:
            what_to_show_l= (c)
            what_to_show_r = what_to_show_r
            title = c
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
                new_column = c + '_savgol_' + str(wdw_savgol)
                print ('Generating ' + new_column + '...')
                df[new_column] = df[c].transform(lambda x: savgol_filter(x, wdw_savgol,2))
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
    plt.grid(alpha=.4,linestyle='--')
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

    INPUT_DIR = 'C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\'
    OUTPUT_DIR = 'C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\'

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

    df_getdata = get_data()
    df = df_getdata.copy(deep=False)
    df, werkdagen, weekend_ = last_manipulations(df, None, None)
    st.title("Interactive Corona Dashboard")
    st.header("Under construction")
    st.subheader("Please send feedback to @rcsmit")

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

    if until_ == "2030-01-01":
        st.sidebar.write("Clear cache")
        caching.clear_cache()
        until_ = "2021-01-01"
        st.sidebar.write("Change the date!")

    df = select_period(df, FROM, UNTIL)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    how_to_display = st.sidebar.selectbox('What to plot (line/bar)', ["line", "linemax", "bar"], index=0)
    #how_to_display = st.sidebar.selectbox('What to plot (line/bar)', ["line", "linemax", "linefirst", "bar"], index=0)
    lijst = ['IC_Bedden_COVID', 'IC_Bedden_Non_COVID', 'Kliniek_Bedden',
        'IC_Nieuwe_Opnames_COVID', 'Kliniek_Nieuwe_Opnames_COVID',
        'Hospital_admission_notification', 'Hospital_admission_x',
        'Total_reported', 'Hospital_admission_y', 'Deceased',
        'Rt_avg',
        'Tested_with_result', 'Tested_positive', 'Percentage_positive',
        'prev_avg']
    global showR
    global WDW3
    global MOVE_WR

    #print (lijst)
    if how_to_display is not "bar":
        what_to_show_day_l = st.sidebar.multiselect('What to show left-axis (multiple possible)', lijst, ["Total_reported"]  )
        what_to_show_day_r = st.sidebar.multiselect('What to show right-axis (multiple possible)', lijst)
        if what_to_show_day_l == None:
            st.error ("CHoose something")
            st.stop()

    showR = False
    if how_to_display == "bar":
        what_to_show_day_l = st.sidebar.selectbox('What to show left-axis (one possible)',lijst, index=7  )
        if what_to_show_day_l == None:
            st.error ("Choose something")
        what_to_show_day_r = None
        showR = st.sidebar.selectbox('Show R number',[True, False], index=0)

    how_to_smoothen = st.sidebar.selectbox('How to smooth (SMA/savgol)', ["SMA", "savgol"], index=0)
    WDW2 = st.sidebar.slider('Window smoothing curves (days)', 1, 14, 7)
    if showR == True:
        WDW3 =  st.sidebar.slider('Window smoothing R-number', 1, 14, 7)
        MOVE_WR = st.sidebar.slider('Move the R-curve', -20, 10, -10)
    else:
        showR = False


      # week_day = st.sidebar.selectbox('Day or Week', ["day", "week"], index=0)
    week_day = "day"
    # if week_day == "week":
    #     how_to_agg_l = st.sidebar.selectbox('How to agg left (sum/mean)', ["sum", "mean"], index=0)
    #     how_to_agg_r = st.sidebar.selectbox('How to agg right (sum/mean)', ["sum", "mean"], index=0)



    global how_to_norm
    #how_to_agg_l = "sum"
    #how_to_agg_r = "mean"

    if what_to_show_day_l is not None:

        if week_day == "day"  :
            if how_to_display == "line":
                graph_daily       (df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
            elif how_to_display == "linemax":
                how_to_norm = "max"
                print(how_to_norm)
                graph_daily_normed(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
            elif how_to_display == "linefirst":
                how_to_norm = "first"
                print(how_to_norm)
                graph_daily_normed(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
            elif how_to_display == "bar":
                graph_daily       (df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)

        else:
            #graph_daily_normed(df,what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display)
            graph_week(df, what_to_show_day_l , how_to_agg_l, what_to_show_day_r , how_to_agg_r)
    else:
        st.error ("Choose what to show")
    # show a weekgraph, options have to put in the function self, no options her (for now)


    #find_correlations(df)
    #find_lag_time(df,"transit_stations","Rt_avg", 0,10)
    tekst = (
    '<style> .infobox {  background-color: lightblue; padding: 5px;}</style>'
    '<hr><div class=\'infobox\'>Made by Rene Smit. (<a href=\'http://www.twitter.com/rcsmit\' target=\"_blank\">@rcsmit</a>) <br>'
    'Sourcecode : <a href=\"https://github.com/rcsmit/COVIDcases/edit/main/covid_dashboard_rcsmit.py\" target=\"_blank\">github.com/rcsmit</a><br>'
    'How-to tutorial : <a href=\"https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d\" target=\"_blank\">rcsmit.medium.com</a><br>')

    st.sidebar.markdown(tekst, unsafe_allow_html=True)
init()
main()
# https://www.medrxiv.org/content/10.1101/2020.05.06.20093039v3.full
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7729173/

# 'With help of <a href=\"https://twitter.com/hk_nien" target=\"_blank\">Han-Kwang Nienhuys</a> and others<br>.'
#    'Restrictions by <a href=\"https://twitter.com/hk_nien" target=\"_blank\">Han-Kwang Nienhuys</a> (MIT-license).</div>')#
