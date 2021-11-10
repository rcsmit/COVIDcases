import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from textwrap import wrap
import matplotlib.cm as cm
import seaborn as sn
from scipy import stats
import datetime as dt
from datetime import datetime, timedelta

from streamlit.errors import NoSessionContext

import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as ticker
import math
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import streamlit as st
import urllib
import urllib.request
from pathlib import Path
from streamlit import caching
from inspect import currentframe, getframeinfo
from helpers import *
import covid_dashboard_show_toelichting_footer
###################################################################
@st.cache(ttl=60 * 60 * 24)
def download_data_file(url, filename, delimiter_, fileformat):
    """Download the external datafiles
    IN :  url : the url
          filename : the filename (without extension) to export the file
          delimiter : delimiter
          fileformat : fileformat
    OUT : df_temp : the dataframe
    """

    # df_temp = None
    download = True
    with st.spinner(f"Downloading...{url}"):
        if download:  # download from the internet
            url = url
        elif fileformat == "json":
            url = INPUT_DIR + filename + ".json"
        else:
            url = INPUT_DIR + filename + ".csv"

        if fileformat == "csv":
            df_temp = pd.read_csv(url, delimiter=delimiter_, low_memory=False)
        elif fileformat == "json":
            df_temp = pd.read_json(url)

        # elif fileformat =='json_x':   # workaround for NICE IC data
        #     pass
        #     # with urllib.request.urlopen(url) as url_x:
        #     #     datajson = json.loads(url_x.read().decode())
        #     #     for a in datajson:
        #     #         df_temp = pd.json_normalize(a)
        else:
            st.error("Error in fileformat")
            st.stop()
        df_temp = df_temp.drop_duplicates()
        # df_temp = df_temp.replace({pd.np.nan: None})  Let it to test
        save_df(df_temp, filename)
        return df_temp


@st.cache(ttl=60 * 60 * 24)
def get_data():
    """Get the data from various sources
    In : -
    Out : df        : dataframe
         UPDATETIME : Date and time from the last update"""
    with st.spinner(f"GETTING ALL DATA ..."):
        init()
        # #CONFIG
        data = [
            # {
            #     "url": "https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv",
            #     "name": "COVID-19_ziekenhuisopname",
            #     "delimiter": ";",
            #     "key": "Date_of_statistics",
            #     "dateformat": "%Y-%m-%d",
            #     "groupby": "Date_of_statistics",
            #     "fileformat": "csv",
            # },
            #  {
            #     "url": "https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv",
            #     "name": "COVID-19_aantallen_gemeente_per_dag",
            #     "delimiter": ";",
            #     "key": "Date_of_publication",
            #     "dateformat": "%Y-%m-%d",
            #     "groupby": "Date_of_publication",
            #     "fileformat": "csv",
            # },
            {
                "url": "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.json",
                "name": "rioolwater",
                "delimiter": ",",
                "key": "Date_measurement",
                "dateformat": "%Y-%m-%d",
                "groupby": "Date_measurement",
                "fileformat": "json",
            },
            # {
            #     "url": "https://hospital_intake_rivm.nu/wp-content/uploads/covid-19.csv",
            #     "name": "hospital_intake_rivm",
            #     "delimiter": ",",
            #     "key": "Datum",
            #     "dateformat": "%d-%m-%Y",
            #     "groupby": None,
            #     "fileformat": "csv",
            # },

            {
                "url": "https://raw.githubusercontent.com/Sikerdebaard/vaccinatie-orakel/main/data/ensemble.csv",
                "name": "vaccinatie",
                "delimiter": ",",
                "key": "date",
                "dateformat": "%Y-%m-%d",
                "groupby": None,
                "fileformat": "csv",
            },
            {
                "url": "https://data.rivm.nl/covid-19/COVID-19_reproductiegetal.json",
                "name": "reprogetal",
                "delimiter": ",",
                "key": "Date",
                "dateformat": "%Y-%m-%d",
                "groupby": None,
                "fileformat": "json",
            },

            {
                "url": "https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv",
                "name": "COVID-19_uitgevoerde_testen",
                "delimiter": ";",
                "key": "Date_of_statistics",
                "dateformat": "%Y-%m-%d",
                "groupby": "Date_of_statistics",
                "fileformat": "csv",
            },
            {
                "url": "https://data.rivm.nl/covid-19/COVID-19_prevalentie.json",
                "name": "prevalentie",
                "delimiter": ",",
                "key": "Date",
                "dateformat": "%Y-%m-%d",
                "groupby": None,
                "fileformat": "json",
            },

 {
                "url": "https://raw.githubusercontent.com/mzelst/covid-19/master/data-rivm/ic-datasets/ic_daily_2021-05-06.csv",
                "name": "ic_daily",
                "delimiter": ",",
                "key": "Date_of_statistics",
                "dateformat": "%Y-%m-%d",
                "groupby": None,
                "fileformat": "csv",
            },


            {

            #     #"url": "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/cases_hospital_new.deaths__ages.csv",
                "url": "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/final_result_vanuit_casus_landelijk.csv",
                "name": "cases_hospital_new.deaths__ages",
                "delimiter": ",",
                "key": "pos_test_Date_statistics",
                "dateformat": "%Y-%m-%d",
                "groupby": None,
                "fileformat": "csv",
            },


            {
                "url": "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/mobility.csv",
                "name": "mobility",
                "delimiter": ",",
                "key": "date",
                "dateformat": "%Y-%m-%d",
                "groupby": None,
                "fileformat": "csv",
            },

            # {
            #     "url": "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/knmi3.csv",
            #     # https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_260.zip
            #     "name": "knmi",
            #     "delimiter": ",",
            #     "key": "Datum",
            #     "dateformat": "%Y%m%d",
            #     "groupby": None,
            #     "fileformat": "csv",
            # },
            {
                "url": "https://raw.githubusercontent.com/mzelst/covid-19/master/data/all_data.csv",
                "name": "all_mzelst",
                "delimiter": ",",
                "key": "date",
                "dateformat": "%Y-%m-%d",
                "groupby": None,
                "fileformat": "csv",
            },


            {
                "url": "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/waze.csv",
                #  # https://raw.githubusercontent.com/ActiveConclusion/COVID19_mobility/master/waze_reports/waze_mobility.csv
                "name": "waze",
                "delimiter": ",",
                "key": "date",
                "dateformat":  "%Y-%m-%d",
                "groupby": None,
                "fileformat": "csv",
            },


            # {'url'       : 'https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/SWEDEN_our_world_in_data.csv',
            # 'name'       : 'sweden',
            # 'delimiter'  : ';',
            # 'key'        : 'Day',
            # 'dateformat' : '%d-%m-%Y',
            # 'groupby'    : None,
            # 'fileformat' : 'csv'},
            #  {'url'      : 'https://stichting-nice.nl/covid-19/public/new-intake/',
            # 'name'       : 'IC_opnames_hospital_intake_rivm',
            # 'delimiter'  : ',',
            # 'key'        : 'date',
            # 'dateformat' : '%Y-%m-%d',
            # 'groupby'    : 'date',
            # 'fileformat' : 'json_x'}
            # {'url'       : 'C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\download_NICE.json',
            # 'name'       : 'IC_opnames_hospital_intake_rivm',
            # 'delimiter'  : ',',
            # 'key'        : 'date',
            # 'dateformat' : '%Y-%m-%d',
            # 'groupby'    : 'date',
            # 'fileformat' : 'json_y'}
        ]

        type_of_join = "outer"
        d = 0

        # Read first datafile
        df_temp_x = download_data_file(
            data[d]["url"], data[d]["name"], data[d]["delimiter"], data[d]["fileformat"]
        )
        # df_temp_x = df_temp_x.replace({pd.np.nan: None})
        df_temp_x[data[d]["key"]] = pd.to_datetime(
            df_temp_x[data[d]["key"]], format=data[d]["dateformat"]
        )
        firstkey = data[d]["key"]

        if data[d]["groupby"] != None:
            df_temp_x = (
                df_temp_x.groupby([data[d]["key"]], sort=True).sum().reset_index()
            )
            df_ungrouped = df_temp_x.reset_index()
            firstkey_ungrouped = data[d]["key"]
        else:
            df_temp_x = df_temp_x.sort_values(by=firstkey)
            df_ungrouped = None

        df_temp = (
            df_temp_x  # df_temp is the base to which the other databases are merged to
        )
        # Read the other files

        for d in range(1, len(data)):

            df_temp_x = download_data_file(
                data[d]["url"],
                data[d]["name"],
                data[d]["delimiter"],
                data[d]["fileformat"],
            )
            # df_temp_x = df_temp_x.replace({pd.np.nan: None})
            oldkey = data[d]["key"]
            newkey = "key" + str(d)
            df_temp_x = df_temp_x.rename(columns={oldkey: newkey})
            #st.write (df_temp_x.dtypes)
            try:
                df_temp_x[newkey] = pd.to_datetime(df_temp_x[newkey], format=data[d]["dateformat"]           )
            except:
                st.error(f"error in {oldkey} {newkey}")
                st.stop()
            if data[d]["groupby"] != None:
                if df_ungrouped is not None:
                    df_ungrouped = df_ungrouped.append(df_temp_x, ignore_index=True)
                    print(df_ungrouped.dtypes)
                    print(firstkey_ungrouped)
                    print(newkey)
                    df_ungrouped.loc[
                        df_ungrouped[firstkey_ungrouped].isnull(), firstkey_ungrouped
                    ] = df_ungrouped[newkey]

                else:
                    df_ungrouped = df_temp_x.reset_index()
                    firstkey_ungrouped = newkey
                df_temp_x = df_temp_x.groupby([newkey], sort=True).sum().reset_index()

            df_temp = pd.merge(
                df_temp, df_temp_x, how=type_of_join, left_on=firstkey, right_on=newkey
            )
            df_temp.loc[df_temp[firstkey].isnull(), firstkey] = df_temp[newkey]
            df_temp = df_temp.sort_values(by=firstkey)
        # the tool is build around "date"
        df_temp = df_temp.rename(columns={firstkey: "date"})

        df_knmi_data = getdata_knmi()
        df_temp = pd.merge(
                df_temp, df_knmi_data, how=type_of_join, left_on='date', right_on='date_knmi'
            )

        UPDATETIME = datetime.now()
        df = splitupweekweekend(df_temp)

        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        print (df)
        return df, df_ungrouped, UPDATETIME


def week_to_week(df, column_, aantal_dagen):
    column_ = column_ if type(column_) == list else [column_]
    newcolumns = []
    newcolumns2 = []

    for c in column_:
        newname = str(c) + "_diff_" + str(aantal_dagen) + "days"
        newname2 = str(c) + "_weekdiff_index" + str(aantal_dagen) + "days"
        newcolumns.append(newname)
        newcolumns2.append(newname2)
        df[newname] = np.nan
        df[newname2] = np.nan
        for n in range(aantal_dagen, len(df)):
            vorige_week = df.iloc[n - aantal_dagen][c]
            nu = df.iloc[n][c]
            waarde = round((((nu - vorige_week) / vorige_week) * 100), 2)
            waarde2 = round((((nu) / vorige_week) * 100), 2)
            df.at[n, newname] = waarde
            df.at[n, newname2] = waarde2
    return df, newcolumns, newcolumns2

def rh2ah(rh, t ):
    """[summary]

    Args:
        rh ([type]): rh min
        t ([type]): temp max

    Returns:
        [type]: [description]
    """
    return (6.112 * math.exp((17.67 * t) / (t + 243.5)) * rh * 2.1674) / (273.15 + t )

def rh2q(rh, t, p ):
    """[summary]

    Args:
        rh ([type]): rh min
        t ([type]): temp max

    Returns:
        [type]: [description]
    """
    # https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html

    #Td = math.log(e/6.112)*243.5/(17.67-math.log(e/6.112))
    es = 6.112 * math.exp((17.67 * t)/(t + 243.5))
    e = es * (rh / 100)
    q_ = (0.622 * e)/(p - (0.378 * e)) * 1000
    return round(q_,2)


def extra_calculations(df):
    """Extra calculations
    IN  : df
    OUT : df with extra columns"""
    #st.write(df.dtypes)
    df["Percentage_positive"] = round(
        (df["Tested_positive"] / df["Tested_with_result"] * 100), 2
    )

    #st.write(df.dtypes)
    try:
        df["RNA_per_reported"] = round(
            ((df["RNA_flow_per_100000"] / 1e15) / df["positivetests"] * 100), 2
        )
    except:
        pass
    df["reported_corrected"] = round(
        (df["positivetests"] * (df["Percentage_positive"] / 12.8)), 2
    # 12.8 is percentage positief getest in week 1-2021
    )

    df["reported_div_tested"] =  round((df["positivetests"] / df["Tested_with_result"]),4)
    df["positivetests_moved_12"] = df["positivetests"].shift(12)
    df["positivetests_moved_-12"] = df["positivetests"].shift(-12)
    df["positivetests_moved_6"] = df["positivetests"].shift(6)
    df["Reported_min_positive"] = df["positivetests"]-df["Tested_positive"]
    df["positivetests_moved_14"] = df["positivetests"].shift(14)
    df["hosp_adm_per_reported"] = round(
            ((df["Kliniek_Nieuwe_Opnames_COVID"] ) / df["positivetests"] * 100), 2
        )

    df["IC_adm_per_reported"] = round(
            ((df["IC_Nieuwe_Opnames_COVID"] ) / df["positivetests"] * 100), 2
        )
    df["new.deaths_per_reported"] = round(
            ((df["new.deaths"] ) / df["positivetests"] * 100), 2)
    df["hosp_adm_per_reported_moved_6"] = round(
            ((df["Kliniek_Nieuwe_Opnames_COVID"] ) / df["positivetests_moved_6"] * 100), 2
        )
    df["hosp_adm_per_reported_moved_12"] = round(
            ((df["Kliniek_Nieuwe_Opnames_COVID"] ) / df["positivetests_moved_12"] * 100), 2
        )
    df["hosp_adm_per_reported_moved_-12"] = round(
            ((df["Kliniek_Nieuwe_Opnames_COVID"] ) / df["positivetests_moved_-12"] * 100), 2
        )

    df["IC_adm_per_reported_moved_6"] = round(
            ((df["IC_Nieuwe_Opnames_COVID"] ) / df["positivetests_moved_6"] * 100), 2
        )
    try:
        df["prev_div_days_contagious"] = round ((df["prev_avg"] ) / number_days_contagious)
    except:
        df["prev_div_days_contagious"] = round ((df["prev_avg"] ) / 8)
    df["prev_div_days_contagious_cumm"] = df["prev_div_days_contagious"].cumsum()


    df["new.deaths_per_prev_div_days_contagious"] = ( df["new.deaths"] / df["prev_div_days_contagious"] )*100

    df["new.deaths_per_reported_moved_14"] = round(
            ((df["new.deaths"] ) / df["positivetests_moved_14"] * 100), 2)

    df["spec_humidity_knmi_derived"] = df.apply(lambda x: rh2q(x['RH_min'],x['temp_max'], 1020),axis=1)
    df["abs_humidity_knmi_derived"] =df.apply(lambda x: rh2ah(x['RH_min'],x['temp_max']),axis=1)
    df["positivetests_cumm"] = df["positivetests"].cumsum()
    df["positivetests_log10"] = np.log10(df["positivetests"])
    df["globale_straling_log10"] = np.log10(df["globale_straling"])
    df["onderrapportagefactor"] = df["prev_div_days_contagious_cumm"] / df["positivetests_cumm"]
    df["new.deaths_cumm"] = df["new.deaths"].cumsum()
    df["new.deaths_cumm_div_prev_div_days_contagious_cumm"] =  df["new.deaths_cumm"] / df["prev_div_days_contagious_cumm"]  * 100
    df["IC_Nieuwe_Opnames_COVID_cumm"] = df["IC_Nieuwe_Opnames_COVID"].cumsum()
    df["Kliniek_Nieuwe_Opnames_COVID_cumm"] = df["Kliniek_Nieuwe_Opnames_COVID"].cumsum()
    #df["total_vaccinations_diff"]=df["total_vaccinations"].diff()
    df["people_vaccinated_diff"]=df["people_vaccinated"].diff()
    df["people_fully_vaccinated_diff"]= df["people_fully_vaccinated"].diff()

    df["hosp_0-49"] = df["hosp_0-9"] + df["hosp_10-19"] + df["hosp_20-29"] + df["hosp_30-39"] + df["hosp_40-49"]
    df["hosp_50-79"] =  df["hosp_50-59"] + df["hosp_60-69"]
    df["hosp_70+"] =   df["hosp_70-79"]  + df["hosp_80-89"] + df["hosp_90+"]
    df["Rt_corr_transit"] = df["Rt_avg"] * (1/ (1-(-1* df["transit_stations"]/100) ))

    return df

def extra_calculations_period(df):
    df["positivetests_cumm_period"] = df["positivetests"].cumsum()
    df["new.deaths_cumm_period"] = df["new.deaths"].cumsum()
    df["IC_Nieuwe_Opnames_COVID_cumm_period"] = df["IC_Nieuwe_Opnames_COVID"].cumsum()
    df["Kliniek_Nieuwe_Opnames_COVID_cumm_period"] = df["Kliniek_Nieuwe_Opnames_COVID"].cumsum()
    df["prev_div_days_contagious_cumm_period"] = df["prev_div_days_contagious"].cumsum()
    df["new.deaths_cumm_period_div_prev_div_days_contagious_cumm_period"] =  df["new.deaths_cumm_period"] / df["prev_div_days_contagious_cumm_period"]  * 100
    first_value_transit = df["transit_stations"].values[0]  # first value in the chosen period
    df["Rt_corr_transit_period"] =df["Rt_avg"] * (1/ (1-( 1* (df["transit_stations"] - first_value_transit)/first_value_transit) ))
    df["reported_corrected2"] = round(
        (df["positivetests"] * (df["Percentage_positive"] / df["Percentage_positive"].values[0])), 2
    )

    return df

###################################################
def calculate_cases(df, ry1, ry2, total_cases_0, sec_variant, extra_days):
    """Add an curve to the graph with a predicted growth model
    IN :  ry1, ry2      : R numbers of two variants
          total_cases_0 : Number of cases on the start date
          sec_variant   : Fraction of the second variant in % (0-100)
          extra_days    : Extra days to plot after the end date
    OUT : df            : dataframe with extra columns (coming from df_calculated)"""

    column = df["date"]
    b_ = column.max().date()
    a_ = FROM
    datediff = (abs((a_ - b_).days)) + 1 + extra_days
    population = 17_500_000
    immune_day_zero = 5_000_000
    Tg = 4
    #st.write (df.dtypes)
    #df.set_index("date")
    suspectible_0 = population - immune_day_zero
    cumm_cases = 0
    #df["date"]= df["date"].strftime("%Y-%m-%d")

    cases_1 = ((100 - sec_variant) / 100) * total_cases_0
    cases_2 = (sec_variant / 100) * total_cases_0
    temp_1 = cases_1
    temp_2 = cases_2
    r_temp_1 = ry1
    r_temp_2 = ry2

    immeratio = 1
    df_calculated = pd.DataFrame(
        {
            "date_calc": a_,
            "variant_1": cases_1,
            "variant_2": cases_2,
            "variant_12": int(cases_1 + cases_2),
        },
        index=[0],
    )

    column = df["date"]
    max_value = column.max()
    #min_index = df.idxmin()
    df = df.fillna(0)

    for day_x in range(1, datediff):
        #print (f"{day_x}  - {immeratio = }")
        if day_x>dag_versoepelingen1:
            versoepeling_factor1 = versoepeling_factor1_
        else:
            versoepeling_factor1 = 1
        if day_x>dag_versoepelingen2:
            versoepeling_factor2 = versoepeling_factor2_
            #versoepeling_factor1 = 1
        else:
            versoepeling_factor2 = 1
        thalf1 = Tg * math.log(0.5) / math.log(versoepeling_factor1 *versoepeling_factor2* immeratio * ry1)
        thalf2 = Tg * math.log(0.5) / math.log(versoepeling_factor1* versoepeling_factor2 * immeratio * ry2)
        day = a_ + timedelta(days=day_x)
        pt1 = temp_1 * (0.5 ** (1 / thalf1))
        pt2 = temp_2 * (0.5 ** (1 / thalf2))
        day_ = day.strftime("%Y-%m-%d")  # FROM object TO string
        day__ = dt.datetime.strptime(day_, "%Y-%m-%d")  # from string to daytime

        df_calculated = df_calculated.append(
            {
                "date_calc": day_,
                "variant_1": round(pt1),
                "variant_2": round(pt2),
                "variant_12": round(pt1 + pt2),
            },
            ignore_index=True,
        )

        temp_1 = pt1
        temp_2 = pt2

        cumm_cases += pt1 + pt2
        cumm_cases_corr = cumm_cases * 2.5

        if day_x>15:
            # we assume that the vaccinations work after 15 days
            day_xx = day_x-15
            people_vaccinated  = df.at[day_xx, 'people_vaccinated']
            people_fully_vaccinated =  df.at[day_xx, 'people_fully_vaccinated']
        else:
            people_vaccinated = 0
            people_fully_vaccinated = 0



        if show_vaccination:
            immeratio = 1 - ((cumm_cases_corr +((people_vaccinated-people_fully_vaccinated)*0.5)+(people_fully_vaccinated*0.95)) / suspectible_0)
        else:
            immeratio = 1 - (cumm_cases_corr  / suspectible_0)

        #immeratio = 1 - ((cumm_cases_corr +  (people_vaccinated*0.5)) / suspectible_0)
    df_calculated["date_calc"] = pd.to_datetime(df_calculated["date_calc"])

    df = pd.merge(
        df,
        df_calculated,
        how="outer",
        left_on="date",
        right_on="date_calc",
        #left_index=True,
    )

    df.loc[df["date"].isnull(), "date"] = df["date_calc"]
    return df

def isNaN(num):
    return num!= num

def splitupweekweekend(df):
    """SPLIT UP IN WEEKDAY AND WEEKEND
    IN  : df
    OUT : df"""
    # SPLIT UP IN WEEKDAY AND WEEKEND
    # https://stackoverflow.com/posts/56336718/revisions
    df["WEEKDAY"] = pd.to_datetime(df["date"]).dt.dayofweek  # monday = 0, sunday = 6
    df["weekend"] = 0  # Initialize the column with default value of 0
    df.loc[
        df["WEEKDAY"].isin([5, 6]), "weekend"
    ] = 1  # 5 and 6 correspond to Sat and Sun
    return df


def add_walking_r(df, smoothed_columns, how_to_smooth, tg):
    """  _ _ _ """
    # Calculate walking R from a certain base. Included a second methode to calculate R
    # de rekenstappen:
    # (1) X =  gemiddelde over 7 dagen over parameter X

    # (2) Rt =   (X(t) / X(t-d) ^ (tg / d))   (d = 7, tg = 4)

    # (3) data opschuiven met 7 dagen - (tijd tussen ziekworden en testen, rapportagevertraging, vertraging, lopend gemiddelde (3 d)
    # https://twitter.com/hk_nien/status/1320671955796844546
    # https://twitter.com/hk_nien/status/1364943234934390792/photo/1
    column_list_r_smoothened = []
    column_list_r_sec_smoothened = []
    d2 = 2
    r_sec = []
    for base in smoothed_columns:
        column_name_R = "R_value_from_" + base + "_tg" + str(tg)
        column_name_R_sec = "R_value_(hk)_from_" + base

        column_name_r_smoothened = (
            "R_value_from_"
            + base
            + "_tg"
            + str(tg)
            + "_"
            + "over_" + str(WDW4) + "_days_"
            + how_to_smooth
            + "_"
            + str(WDW3)
        )
        column_name_r_sec_smoothened = (
            "R_value_sec_from_"
            + base
            + "_tg"
            + str(tg)
            + "_"
            + "over_" + str(WDW4) + "_days_"

            + how_to_smooth
            + "_"
            + str(WDW3)
        )

        sliding_r_df = pd.DataFrame(
            {"date_sR": [], column_name_R: [], column_name_R_sec: []}
        )

        d = WDW4
        for i in range(len(df)):
            if df.iloc[i][base] != None:
                date_ = pd.to_datetime(df.iloc[i]["date"], format="%Y-%m-%d")
                date_ = df.iloc[i]["date"]
                if df.iloc[i - d][base] != 0 or df.iloc[i - d][base] is not None:
                    slidingR_ = round(
                        ((df.iloc[i][base] / df.iloc[i - d][base]) ** (tg / d)), 2
                    )

                else:
                    slidingR_ = None
                slidingR_sec = None  # slidingR_sec = round(math.exp((tg *(math.log(df.iloc[i][base])- math.log(df.iloc[i-d2][base])))/d2),2)
                sliding_r_df = sliding_r_df.append(
                    {
                        "date_sR": date_,
                        column_name_R: slidingR_,
                        column_name_R_sec: slidingR_sec,
                    },
                    ignore_index=True,
                )
        #st.warning(column_name_r_smoothened)
        sliding_r_df = sliding_r_df.fillna(0)
        #st.write (df[column_name_r_smoothened])
        #st.write(sliding_r_df)

        sliding_r_df[column_name_r_smoothened] = round(
            sliding_r_df.iloc[:, 1].rolling(window=WDW3, center=True).mean(), 2
        )



        sliding_r_df[column_name_r_sec_smoothened] = round(
            sliding_r_df.iloc[:, 2].rolling(window=WDW3, center=True).mean(), 2
        )

        sliding_r_df = sliding_r_df.reset_index()
        df = pd.merge(
            df,
            sliding_r_df,
            how="outer",
            left_on="date",
            right_on="date_sR",
            #left_index=True,
        )
        column_list_r_smoothened.append(column_name_r_smoothened)
        column_list_r_sec_smoothened.append(column_name_r_sec_smoothened)

        sliding_r_df = sliding_r_df.reset_index()
    return df, column_list_r_smoothened, column_list_r_sec_smoothened


def move_column(df, column_, days):
    """Move/shift a column

    Args:
        df (df): df
        column_ (string): which column to move
        days (int): how many days

    Returns:
        df: df
        new_column : name of the new column
    """    """  _ _ _ """
    column_ = column_ if type(column_) == list else [column_]
    for column in column_:
        new_column = column + "_moved_" + str(days)
        df[new_column] = df[column].shift(days)
    return df, new_column


def drop_columns(df, what_to_drop):
    """  _ _ _ """
    if what_to_drop != None:
        for d in what_to_drop:
            print("dropping " + d)

            df = df.drop(columns=[d], axis=1)
    return df

def select_period_oud(df, field, show_from, show_until):
    """Shows two inputfields (from/until and Select a period in a df (helpers.py).
    Args:
        df (df): dataframe
        field (string): Field containing the date
    Returns:
        df: filtered dataframe
    """

    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df



def agg_week(df, how):
    """  _ _ _ """
    # #TODO
    # HERE ARE SOME PROBLEMS DUE TO MISSING .isotype()
    # FutureWarning: Series.dt.weekofyear and Series.dt.week have been deprecated.
    #  Please use Series.dt.isocalendar().week instead.
    df["weeknr"] = df["date"].dt.week
    df["yearnr"] = df["date"].dt.year

    df["weekalt"] = (
        df["date"].dt.year.astype(str) + "-" + df["date"].dt.week.astype(str)
    )

    for i in range(len(df)):
        if df.iloc[i]["weekalt"] == "2021-53":
            df.iloc[i]["weekalt"] = "2020-53"

    # how = "mean"
    if how == "mean":
        dfweek = (
            df.groupby(["weeknr", "yearnr", "weekalt"], sort=False).mean().reset_index()
        )
    elif how == "sum":
        dfweek = (
            df.groupby(["weeknr", "yearnr", "weekalt"], sort=False).sum().reset_index()
        )
    else:
        print("error agg_week()")
        st.stop()
    return df, dfweek


def last_manipulations(df, what_to_drop, drop_last):
    """  _ _ _ """
    df = drop_columns(df, what_to_drop)

    # Two different dataframes for workdays/weekend

    werkdagen = df.loc[(df["weekend"] == 0)]
    weekend_ = df.loc[(df["weekend"] == 1)]
    df = df.drop(columns=["weekend"], axis=1)
    werkdagen = werkdagen.drop(columns=["WEEKDAY"], axis=1)
    werkdagen = werkdagen.drop(columns=["weekend"], axis=1)
    weekend_ = weekend_.drop(columns=["WEEKDAY"], axis=1)
    weekend_ = weekend_.drop(columns=["weekend"], axis=1)

    if drop_last != None:
        df = df[:drop_last]  # drop last row(s)

    df.insert(df.shape[1], "q_biggerthansix", None)

    df.insert(df.shape[1], "q_smallerthansix", None)

    return df, werkdagen, weekend_


def save_df(df, name):
    """  _ _ _ """
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")


##########################################################
def correlation_matrix(df, werkdagen, weekend_):
    """  _ _ _ """

    # CALCULATE CORRELATION

    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True, annot_kws={"fontsize": 7})
    plt.title("ALL DAYS", fontsize=20)
    plt.show()

    # corrMatrix = werkdagen.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # plt.title("WORKING DAYS", fontsize =20)
    # plt.show()

    # corrMatrix = weekend_.corr()
    # sn.heatmap(corrMatrix, annot=True)
    # plt.title("WEEKEND", fontsize =20)
    # plt.show()

    # MAKE A SCATTERPLOT

    # sn.regplot(y="Rt_avg", x="Kliniek_Nieuwe_Opnames_COVID", data=df)
    # plt.show()


def normeren(df, what_to_norm):
    """In : columlijst
    Bewerking : max = 1
    Out : columlijst met genormeerde kolommen"""
    # print(df.dtypes)

    normed_columns = []

    for column in what_to_norm:
        maxvalue = (df[column].max()) / scale_to_x
        firstvalue = df[column].iloc[int(WDW2 / 2)] / scale_to_x
        name = f"{column}_normed"
        for i in range(len(df)):
            if how_to_norm == "max":
                df.loc[i, name] = df.loc[i, column] / maxvalue
            else:
                df.loc[i, name] = df.loc[i, column] / firstvalue
        normed_columns.append(name)
        print(f"{name} generated")
    return df, normed_columns


def graph_daily_normed(
    df, what_to_show_day_l, what_to_show_day_r, how_to_smoothen, how_to_display
):
    """IN : df, de kolommen die genormeerd moeten worden
    ACTION : de grafieken met de genormeerde kolommen tonen"""

    if what_to_show_day_l is None:
        st.warning("Choose something")
        st.stop()

    df, smoothed_columns_l = smooth_columnlist(df, what_to_show_day_l, how_to_smoothen,WDW2, centersmooth)
    df, normed_columns_l = normeren(df, smoothed_columns_l)

    df, smoothed_columns_r = smooth_columnlist(df, what_to_show_day_r, how_to_smoothen, WDW2, centersmooth)
    df, normed_columns_r = normeren(df, smoothed_columns_r)

    graph_daily(df, normed_columns_l, normed_columns_r, None, how_to_display, showday)


def graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t,showday):
    """  _ _ _ """
    #st.write(f"t = {t}")
    df_temp = pd.DataFrame(columns=["date"])
    if what_to_show_l is None:
        st.warning("Choose something")
        st.stop()

    if type(what_to_show_l) == list:
        what_to_show_l_ = what_to_show_l
    else:
        what_to_show_l_ = [what_to_show_l]
    aantal = len(what_to_show_l_)
    # SHOW A GRAPH IN TIME / DAY
    title_plotly = title


    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)
        # Some nice colors chosen with coolors.com

        # #CONFIG

        operamauve = "#ac80a0"  # purple 1
        green_pigment = "#3fa34d"  # green 2
        minion_yellow = "#EAD94C"  # yellow 3
        mariagold = "#EFA00B"  # orange 4
        falu_red = "#7b2d26"  # red 5

        prusian_blue = "#1D2D44"  # 8

        color_list = [
            "#02A6A8",
            "#4E9148",
            "#F05225",
            "#024754",
            "#FBAA27",
            "#302823",
            "#F07826",
             "#ff6666",  # reddish 0
         "#ac80a0",  # purple 1
         "#3fa34d",  # green 2
         "#EAD94C",  # yellow 3
         "#EFA00B",  # orange 4
         "#7b2d26",  # red 5
         "#3e5c76",  # blue 6
         "#e49273",  # dark salmon 7
         "#1D2D44",  # 8

        ]

        n = 0  # counter to walk through the colors-list

        df, columnlist_sm_l = smooth_columnlist(df, what_to_show_l_, how_to_smooth, WDW2, centersmooth)

        # CODE TO MAKE STACKED BARS - DOESNT WORK
        # stackon=""
        # if len(what_to_show_l_)>1:
        #     w = ["Datum"]
        #     for s in what_to_show_l_:
        #         w.append(s)
        #     #st.write(w)
        #     df_stacked = df[w].copy()
        #     #print (df_stacked.dtypes)
        #     #df_stacked.set_index('Datum')

        # st.write(df_stacked)
        # if t == "bar":
        # ax = df_stacked.plot.bar(stacked=True)
        # ax = df_stacked.plot(rot=0)
        # st.bar_chart(df_stacked)
        # ax = df[c_smooth].plot(label=c_smooth, color = color_list[2],linewidth=1.5)         # SMA

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
                # weekends have a different color
                firstday = df.iloc[0]["WEEKDAY"]  # monday = 0
                color_x = find_color_x(firstday, showoneday, showday)
                # MAYBE WE CAN LEAVE THIS OUT HERE
                df, columnlist = smooth_columnlist(df, [b], how_to_smooth, WDW2, centersmooth)

                df.set_index("date")

                df_temp = df
                if len(what_to_show_l_) == 1:
                    ax = df_temp[b].plot.bar(
                        label=b, color=color_x, alpha=0.6
                    )  # number of cases

                    for c_smooth in columnlist:
                        ax = df[c_smooth].plot(
                            label=c_smooth, color=color_list[2], linewidth=1.5
                        )  # SMA

                    if showR:
                        if show_R_value_RIVM:
                            ax3 = df["Rt_avg"].plot(
                                secondary_y=True,
                                linestyle="--",
                                label="Rt RIVM",
                                color=green_pigment,
                                alpha=0.8,
                                linewidth=1,
                            )
                            ax3.fill_between(
                                df["date"].index,
                                df["Rt_low"],
                                df["Rt_up"],
                                color=green_pigment,
                                alpha=0.2,
                                label="_nolegend_",
                            )
                        tgs = [3.5, 4, 5]

                        teller = 0
                        dfmin = ""
                        dfmax = ""
                        for TG in tgs:
                            df, R_smooth, R_smooth_sec = add_walking_r(
                                df, columnlist, how_to_smooth, TG
                            )

                            for R in R_smooth:
                                # correctie R waarde, moet naar links ivm 2x smoothen
                                df, Rn = move_column(df, R, MOVE_WR)

                                if teller == 0:
                                    dfmin = Rn
                                elif teller == 1:
                                    if show_R_value_graph:
                                        ax3 = df[Rn].plot(
                                            secondary_y=True,
                                            label=Rn,
                                            linestyle="--",
                                            color=falu_red,
                                            linewidth=1.2,
                                        )
                                elif teller == 2:
                                    dfmax = Rn
                                teller += 1
                            for R in R_smooth_sec:  # SECOND METHOD TO CALCULATE R
                                # correctie R waarde, moet naar links ivm 2x smoothen
                                df, Rn = move_column(df, R, MOVE_WR)
                                # ax3=df[Rn].plot(secondary_y=True, label=Rn,linestyle='--',color=operamauve, linewidth=1)
                        if show_R_value_graph:
                            ax3.fill_between(
                                df["date"].index,
                                df[dfmin],
                                df[dfmax],
                                color=falu_red,
                                alpha=0.3,
                                label="_nolegend_",
                            )

            else:  # t = line
                df_temp = df

                if how_to_smooth == None:
                    how_to_smooth_ = "unchanged_"
                else:
                    how_to_smooth_ = how_to_smooth + "_" + str(WDW2)
                b_ = str(b) + "_" + how_to_smooth_
                df_temp[b_].plot(
                    label=b, color=color_list[n], linewidth=1.1
                )  # label = b_ for uitgebreid label
                df_temp[b].plot(
                    label="_nolegend_",
                    color=color_list[n],
                    linestyle="dotted",
                    alpha=0.9,
                    linewidth=0.8,
                )
            n += 1
        if show_scenario == True:
            df = calculate_cases(df, ry1, ry2, total_cases_0, sec_variant, extra_days)
            # print (df.dtypes)
            l1 = f"R = {ry1}"
            l2 = f"R = {ry2}"
            ax = df["variant_1"].plot(
                label=l1, color=color_list[4], linestyle="dotted", linewidth=1, alpha=1
            )
            ax = df["variant_2"].plot(
                label=l2, color=color_list[5], linestyle="dotted", linewidth=1, alpha=1
            )
            ax = df["variant_12"].plot(
                label="TOTAL", color=color_list[6], linestyle="--", linewidth=1, alpha=1
            )

        if what_to_show_r != None:
            if type(what_to_show_r) == list:
                what_to_show_r = what_to_show_r
            else:
                what_to_show_r = [what_to_show_r]

            n = len(color_list)
            x = n
            for a in what_to_show_r:
                x -= 1
                lbl = a + " (right ax)"
                df, columnlist = smooth_columnlist(df, [a], how_to_smooth, WDW2, centersmooth)
                for c_ in columnlist:
                    # smoothed
                    lbl2 = a + " (right ax)"
                    ax3 = df_temp[c_].plot(
                        secondary_y=True,
                        label=lbl2,
                        color=color_list[x],
                        linestyle="--",
                        linewidth=1.1,
                    )  # abel = lbl2 voor uitgebreid label
                ax3 = df_temp[a].plot(
                    secondary_y=True,
                    linestyle="dotted",
                    color=color_list[x],
                    linewidth=1,
                    alpha=0.9,
                    label="_nolegend_",
                )
                ax3.set_ylabel("_")

            if len(what_to_show_l) == 1 and len(what_to_show_r) == 1:  # add correlation
                correlation = find_correlation_pair(df, what_to_show_l, what_to_show_r)
                correlation_sm = find_correlation_pair(df, b_, c_)
                title_scatter =  f"{title}({str(FROM)} - {str(UNTIL)})\nCorrelation = {correlation}"
                title = f"{title} \nCorrelation = {correlation}\nCorrelation smoothed = {correlation_sm}"
                title_plotly = f"{title}<br>Correlation = {correlation}<br>Correlation smoothed = {correlation_sm}"


            if len(what_to_show_r) == 1:
                mean = df[what_to_show_r].mean()
                std =df[what_to_show_r].std()
                # print (f"mean {mean}")
                # print (f"st {std}")
                low = mean -2*std
                up = mean +2*std
                #ax3.set_ylim = (-100, 100)
        plt.title(title, fontsize=10)

        a__ = (max(df_temp["date"].tolist())).date() - (
            min(df_temp["date"].tolist())
        ).date()
        freq = int(a__.days / 10)
        ax.xaxis.set_major_locator(MultipleLocator(freq))
        if what_to_show_l == ["reported_div_tested"]:
            ax.set_ylim(0,0.3)
        ax.set_xticks(df_temp["date"].index)
        ax.set_xticklabels(df_temp["date"].dt.date, fontsize=6, rotation=90)
        xticks = ax.xaxis.get_major_ticks()
        if groupby_timeperiod == "none":
            for i, tick in enumerate(xticks):
                if i % 10 != 0:
                    tick.label1.set_visible(False)
        plt.xticks()

        # layout of the x-axis
        ax.xaxis.grid(True, which="major", alpha=0.4, linestyle="--")
        ax.yaxis.grid(True, which="major", alpha=0.4, linestyle="--")
        if showlogyaxis == "10":
            ax.semilogy()
        if showlogyaxis == "2":
            ax.semilogy(2)
        if showlogyaxis == "logit":
            ax.set_yscale("logit")

        left, right = ax.get_xlim()
        ax.set_xlim(left, right)
        fontP = FontProperties()
        fontP.set_size("xx-small")

        plt.xlabel("date")
        # everything in legend
        # https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
        handles, labels = [], []
        for ax in fig1x.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.append(h)
                labels.append(l)
        # plt.legend(handles,labels)
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
        plt.legend(handles, labels, bbox_to_anchor=(0, -0.5), loc="lower left", ncol=2)
        ax.text(1,1.1,"Created by Rene Smit — @rcsmit",transform=ax.transAxes,
            fontsize="xx-small",va="top",ha="right",)
        if show_R_value_graph or show_R_value_RIVM:
            plt.axhline(y=1, color="yellow", alpha=0.6, linestyle="--")
        if groupby_timeperiod == "none":
            add_restrictions(df, ax)

        plt.axhline(y=horiz_line, color="black", alpha=0.6, linestyle="--")
        if t == "line":
            set_xmargin(ax, left=-0.04, right=-0.04)
        st.pyplot(fig1x)

        fig = go.Figure()
        # TOFIX
        fig.add_trace(go.Scatter(x=df_temp["date"], y= df_temp[b], mode='lines', name=b, line=dict(
            color='LightSkyBlue')))
        fig.add_trace(go.Scatter(x=df_temp["date"], y=df_temp[b], mode='markers', name = b, showlegend=False,marker=dict(
            color='LightSkyBlue',
            size=2)))

        fig.update_layout(

            yaxis=dict(
                title=b,
                titlefont=dict(
                    color="#1f77b4"
                ),
                tickfont=dict(
                    color="#1f77b4"
                )
            ),
            title=dict(
                    text=title_plotly,
                    x=0.5,
                    y=0.95,
                    font=dict(
                        family="Arial",
                        size=14,
                        color='#000000'
                    )
                ),


        )

        if type(what_to_show_r) == list:
            what_to_show_r = what_to_show_r
        else:
            what_to_show_r = [what_to_show_r]

        n = len(color_list)
        x = n
        for a in what_to_show_r:
            x -= 1
            lbl = a + " (right ax)"
            df, columnlist = smooth_columnlist(df, [a], how_to_smooth, WDW2, centersmooth)
            for c_ in columnlist:
                # smoothed
                lbl2 = a + " (right ax)"
                fig.add_trace(go.Scatter(x=df_temp["date"], y=df_temp[c_],  name=a, mode='lines',   line=dict(color='red'),yaxis="y2"))

            fig.add_trace(go.Scatter(x=df_temp["date"], y=df_temp[a], mode='markers', name = a, showlegend=False,   yaxis="y2", marker=dict(
            color='red',
            size=2)))
            fig.update_layout(
                yaxis2=dict(
                    title=a,
                    titlefont=dict(
                        color="red"
                    ),
                    tickfont=dict(
                        color="red"
                    ),

                    overlaying="y",
                    side="right",
                    position=1.0
                )

            )

        # Create axis objects




        st.plotly_chart(fig, use_container_width=True)




    #if len(what_to_show_l) >= 1 and len(what_to_show_r) >= 1:  # add scatter plot
    if what_to_show_l is not None and what_to_show_r is not None:
        for l in what_to_show_l:
            for r in what_to_show_r:

                left_sm = str(l) + "_" + how_to_smooth_
                right_sm = str(r) + "_" + how_to_smooth_
                make_scatterplot(df_temp, l,r, FROM, UNTIL,  True, False)
                make_scatterplot(df_temp,left_sm, right_sm, FROM, UNTIL, True, True)


def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_xmargin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta * left
    right = lim[1] + delta * right
    ax.set_xlim(left, right)


def add_restrictions(df, ax):
    pass

def add_restrictions_original(df, ax):


    """  _ _ _ """
    # Add restrictions
    # From Han-Kwang Nienhuys - MIT-licence
    df_restrictions = pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/restrictions.csv",
        comment="#",
        delimiter=",",
        low_memory=False,
    )

    a = (min(df["date"].tolist())).date()
    b = (max(df["date"].tolist())).date()

    ymin, ymax = ax.get_ylim()
    y_lab = ymin
    for i in range(0, len(df_restrictions)):
        d_ = df_restrictions.iloc[i]["Date"]  # string
        d__ = dt.datetime.strptime(d_, "%Y-%m-%d").date()  # to dateday

        diff = d__ - a
        diff2 = b - d__

        if diff.days > 0 and diff2.days > 0:

            ax.text(
                (diff.days),
                0,
                f'  {df_restrictions.iloc[i]["Description"] }',
                rotation=90,
                fontsize=4,
                horizontalalignment="center",
            )
            # plt.axvline(x=(diff.days), color='yellow', alpha=.3,linestyle='--')


def graph_week(df, what_to_show_l, how_l, what_to_show_r, how_r):
    """  _ _ _ """

    # SHOW A GRAPH IN TIME / WEEK
    df_l, dfweek_l = agg_week(df, how_l)

    if str(FROM) != "2021-01-01":
        st.info(
            "To match the weeknumbers on the ax with the real weeknumbers, please set the startdate at 2021-1-1"
        )
    if what_to_show_r != None:
        df_r, dfweek_r = agg_week(df, how_r)

    if type(what_to_show_l) == list:
        what_to_show_l = what_to_show_l
    else:
        what_to_show_l = [what_to_show_l]

    for show_l in what_to_show_l:
        label_l = show_l + " (" + how_l + ")"
        label_r = None
        if graph_how == "pyplot":

            fig1y = plt.figure()
            ax = fig1y.add_subplot(111)
            ax.set_xticks(dfweek_l["weeknr"])
            ax.set_xticklabels(dfweek_l["weekalt"], fontsize=6, rotation=45)

            dfweek_l[show_l].plot.bar(label=label_l, color="#F05225")

            if what_to_show_r != None:
                for what_to_show_r_ in what_to_show_r:
                    label_r = what_to_show_r_ + " (" + how_r + ")"
                    ax3 = dfweek_r[what_to_show_r_].plot(
                        secondary_y=True, color="r", label=label_r
                    )

            # Add a grid
            plt.grid(alpha=0.2, linestyle="--")

            # Add a Legend
            fontP = FontProperties()
            fontP.set_size("xx-small")
            plt.legend(loc="best", prop=fontP)

            ax.xaxis.set_major_locator(MultipleLocator(1))
            # ax.xaxis.set_major_formatter()
            # everything in legend
            # https://stackoverflow.com/questions/33611803/pyplot-single-legend-when-plotting-on-secondary-y-axis
            handles, labels = [], []
            for ax in fig1y.axes:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    handles.append(h)
                    labels.append(l)

            plt.legend(handles, labels)
            plt.xlabel("Week counted from " + str(FROM))
            # configgraph(titlex)
            if show_R_value_graph or show_R_value_RIVM:
                pass
                #ax3.axhline(y=1, color="yellow", alpha=0.6, linestyle="--")
            st.pyplot(fig1y)
            # plt.show()

        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add traces
            fig.add_trace(
                go.Bar(x=dfweek_l["weeknr"], y=dfweek_l[show_l], name=show_l),
                secondary_y=False,
            )
            if what_to_show_r != None:
                for what_to_show_r_ in what_to_show_r:
                    label_r = what_to_show_r_ + " (" + how_r + ")"
                    fig.add_trace(
                        go.line(x=dfweek_l["weeknr"], y= dfweek_r[what_to_show_r_], name=label_r),
                        secondary_y=True,
                                )



            # Add figure title
            fig.update_layout(
                title_text=f"{show_l} - {label_r}"
            )

            # Set x-axis title
            fig.update_xaxes(title_text="Week counted from " + str(FROM))

            # Set y-axes titles
            fig.update_yaxes(title_text=show_l, secondary_y=False)
            fig.update_yaxes(title_text=label_r, secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)


def graph_daily(df, what_to_show_l, what_to_show_r, how_to_smooth, t, showday):
    """  _ _ _ """

    if type(what_to_show_l) == list:
        what_to_show_l = what_to_show_l
    else:
        what_to_show_l = [what_to_show_l]
    title = ""
    if t == "bar":
        for c in what_to_show_l:

            #    what_to_show_r = what_to_show_r


            title += str(c) + " "

        t1 =wrap(title, 40)
        title = ""
        #st.write (t1)
        for tx in t1:
            title += tx + "\n"
        print (f"titel 1277{title}")

        graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t, showday)
    else:
        tl = ""
        tr = ""
        i = 0
        j = 0
        if what_to_show_l is not None:
            for l in what_to_show_l:
                if i != len(what_to_show_l) - 1:
                    if groupby_how == "sum":
                        tl += l+" (sum) /"
                    elif groupby_how == "mean":
                        tl += l+" (mean) /"
                    elif groupby_how == "max":
                        tl += l+" (max) /"
                    else:
                        tl += l + " / "
                    i += 1
                else:

                    if groupby_how == "sum":
                        tl += " (sum) "
                    elif groupby_how == "mean":
                        tl += " (mean) "
                    elif groupby_how == "max":
                        tl += l+" (max) "
                    else:
                        tl += l

        if what_to_show_r is not None:
            if type(what_to_show_r) == list:
                what_to_show_r = what_to_show_r
            else:
                what_to_show_r = [what_to_show_r]
            tl += " - \n"
            for r in what_to_show_r:
                if j != len(what_to_show_r) - 1:
                    if groupby_how == "sum":
                        tl += r+" (sum) /"
                    elif groupby_how == "mean":
                        tl += r+" (mean) /"
                    elif groupby_how == "max":
                        tl += r+" (max) /"
                    else:
                        tl += r + " / "
                    j += 1
                else:

                    if groupby_how == "sum":
                        tl += r+" (sum) "
                    elif groupby_how == "mean":
                        tl += r+" (mean) "
                    elif groupby_how == "max":
                        tl += r+" (max) "
                    else:
                        tl +=r
        tl = tl.replace("_", " ")

        #title = f"{tl}"
        t1 =wrap(tl, 80)
        title = ""

        for t in t1:
            title += t + "\n"
        graph_day(df, what_to_show_l, what_to_show_r, how_to_smooth, title, t, showday)


def smooth_columnlist(df, columnlist, t, WDW2, centersmooth):
    """  _ _ _ """
    c_smoothen = []
    wdw_savgol = 7
    #if __name__ = "covid_dashboard_rcsmit":
    # global WDW2, centersmooth, show_scenario
    # WDW2=7
    # st.write(__name__)
    # centersmooth = False
    # show_scenario = False
    if columnlist is not None:
        if type(columnlist) == list:
            columnlist_ = columnlist
        else:
            columnlist_ = [columnlist]
            # print (columnlist)
        for c in columnlist_:
            print(f"Smoothening {c}")
            if t == "SMA":
                new_column = c + "_SMA_" + str(WDW2)
                print("Generating " + new_column + "...")
                df[new_column] = (
                    df.iloc[:, df.columns.get_loc(c)]
                    .rolling(window=WDW2, center=centersmooth)
                    .mean()
                )

            elif t == "savgol":
                new_column = c + "_savgol_" + str(WDW2)
                print("Generating " + new_column + "...")
                df[new_column] = df[c].transform(lambda x: savgol_filter(x, WDW2, 2))

            elif t == None:
                new_column = c + "_unchanged_"
                df[new_column] = df[c]
                print("Added " + new_column + "...~")
            else:
                print("ERROR in smooth_columnlist")
                st.stop()
            c_smoothen.append(new_column)
    return df, c_smoothen


###################################################################
def find_correlations(df, treshold, fields):
    al_gehad = []
    paar = []
    # column_list = list(df.columns)
    column_list = fields
    # print (column_list)
    st.header("Found correlations in the data :")
    for i in column_list:
        for j in column_list:
            # paar = [j, i]
            paar = str(i) + str(j)
            if paar not in al_gehad:
                if i == j:
                    pass
                else:
                    try:
                        c = round(df[i].corr(df[j]), 3)
                        if c >= treshold or c <= (treshold * -1):
                            st.write(f"{i} - {j} - {str(c)}")

                    except:
                        pass
            else:
                pass  # ("algehad")
            al_gehad.append(str(j) + str(i))


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
            c = round(df[i].corr(df[j]), 3)
    return c


def find_lag_time(df, what_happens_first, what_happens_second, r1, r2):
    b = what_happens_first
    a = what_happens_second  # shape (266,1)
    x = []
    y = []
    y_sma =[]

    max = 0
    max_sma = 0
    df, b_sma = smooth_columnlist(df, b, "SMA", 7, True )
    df, a_sma = smooth_columnlist(df, a, "SMA", 7, True )


    df, nx = move_column(df, a, 0) #strange way to prevent error
    df, nx_sma = move_column(df, a_sma, 0) #strange way to prevent error

    max_column = None
    for n in range(r1, (r2 + 1)):

        df, m = move_column(df, b, n) #(shape (266,)
        c = round(df[m].corr(df[nx]), 3)
        if c > max:
            max = c

        x.append(n)
        y.append(c)

        df, m_sma = move_column(df, b_sma, n) #(shape (266,)
        c_sma = round(df[m_sma].corr(df[nx_sma]), 3)
        if c_sma > max_sma:
            max_sma = c_sma
        y_sma.append(c_sma)


    title = f"Correlation between : {a} - {b} with moved days"

    with _lock:
        fig1x = plt.figure()
        ax = fig1x.add_subplot(111)
        plt.xlabel("shift in days")
        plt.plot(x, y, label = "Values")
        plt.plot(x, y_sma, label = "Smoothed")
        #plt.axvline(x=0, color="yellow", alpha=0.6, linestyle="--")
        # Add a grid
        plt.legend()
        plt.grid(alpha=0.2, linestyle="--")
        plt.title(title, fontsize=10)
        st.pyplot(fig1x)
    # plt.show()

    # graph_daily(df, [a], [b], "SMA", "line", showday)
    # graph_daily(df, [a], [max_column], "SMA", "line", showday)
    # if the optimum is negative, the second one is that x days later


def init():
    """  _ _ _ """

    global download

    global INPUT_DIR
    global OUTPUT_DIR

    INPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\input\\"
    )
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\"
    )

    # GLOBAL SETTINGS
    download = True  # True : download from internet False: download from INPUT_DIR
    # De open data worden om 15.15 uur gepubliceerd


def get_locations(df_ungrouped, field):
    """ Get a list of the Municipalities """
    return df_ungrouped[field].unique()
    # Municipality_name;
    # Province;
    # Security_region_code;
    # Security_region_name;
    # Municipal_health_service;
    # ROAZ_region
    print(df_ungrouped)

def get_duplicate_cols(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.columns).value_counts()[lambda x: x>1]

def main():
    """  _ _ _ """
    global FROM
    global UNTIL
    global WDW2
    global WDW3, WDW4
    global number_days_contagious
    global showoneday
    global showday
    global MOVE_WR
    global showR
    global lijst  # Lijst in de pull down menu's voor de assen
    global show_scenario, show_vaccination
    global how_to_norm
    global Rnew1_, Rnew2_
    global ry1, ry2, total_cases_0, sec_variant, extra_days
    global show_R_value_graph, show_R_value_RIVM, centersmooth
    global dag_versoepelingen1 , versoepeling_factor1_
    global dag_versoepelingen2 , versoepeling_factor2_
    global showlogyaxis
    global OUTPUT_DIR
    global INPUT_DIR
    init()

    df_getdata, df_ungrouped_, UPDATETIME = get_data()
    df = df_getdata.copy(deep=False)
    df_ungrouped = df_ungrouped_.copy(deep=False)

    # rioolwaterplaatsen = (get_locations(df_ungrouped, "RWZI_AWZI_name"))

    df, werkdagen, weekend_ = last_manipulations(df, None, None)

    # #CONFIG
    df.rename(
        columns={
            "Hospital_admission_x": "hospital_intake_rivm",
            "IC_admission": "IC_admission_RIVM",
            "Hospital_admission_y": "Hospital_admission_GGD",
            #"Kliniek_Nieuwe_Opnames_COVID_x": "Hospital_admission_hospital_intake_rivm",
            #   "value"                             : "IC_opnames_NICE",
            "IC_Nieuwe_Opnames_COVID_x": "IC_Nieuwe_Opnames_COVID",
            "IC_Nieuwe_Opnames_COVID_y": "IC_Nieuwe_Opnames_COVID",
            "IC_Bedden_COVID_x": "IC_Bedden_COVID",
            "IC_Bedden_Non_COVID_x":"IC_Bedden_Non_COVID",
            "Kliniek_Bedden_x":"Kliniek_Bedden",
            "retail_and_recreation_percent_change_from_baseline":  "retail_and_recreation",
            "grocery_and_pharmacy_percent_change_from_baseline": "grocery_and_pharmacy",
            "parks_percent_change_from_baseline" :  "parks",
            "transit_stations_percent_change_from_baseline" : "transit_stations",
            "workplaces_percent_change_from_baseline":   "workplaces",
            "residential_percent_change_from_baseline":  "residential",
            "UG": "RH_avg",
            "UX": "RH_max",
            "UN": "RH_min",
        },
        inplace=True,
    )

    df = extra_calculations(df)
    save_df(df, "EINDTABELx")
    lijst = []
    lijst_oud = [
        # "IC_Bedden_COVID",
        # "IC_Bedden_Non_COVID",
        # "Kliniek_Bedden",
        # "IC_Nieuwe_Opnames_COVID",
        # "IC_admission_RIVM",
        # "IC_Intake_Proven",
        # "hospital_intake_rivm",
        # "Hospital_admission_hospital_intake_rivm",
        # "Hospital_admission_GGD",
        # "positivetests",
        # "new.deaths",
        "Rt_avg",
        "Tested_with_result",
        "Tested_positive",
        "Percentage_positive",
        "reported_div_tested",
        "Reported_min_positive",
        "prev_avg",
        "total_vaccinations",
        "people_vaccinated",
        "people_fully_vaccinated",
        "total_vaccinations_diff",
        "people_vaccinated_diff",
        "people_fully_vaccinated_diff",
        "retail_and_recreation",
        "grocery_and_pharmacy",
        "parks",
        "transit_stations",
        "workplaces",
        "residential",
        "driving_waze",
        "temp_min",
        "temp_etmaal",
        "temp_max",
        "zonneschijnduur",
        "globale_straling",
        "globale_straling_log10",
        "spec_humidity_knmi_derived",
        "abs_humidity_knmi_derived",
        "RH_avg",
        "RH_min",
        "RH_max",
        "neerslag",
        "RNA_per_ml",
        "RNA_flow_per_100000",
        "RNA_per_reported",
        "hosp_adm_per_reported",
        "IC_adm_per_reported",
        "new.deaths_per_reported",
        "hosp_adm_per_reported_moved_6",
        "hosp_adm_per_reported_moved_12",
        "hosp_adm_per_reported_moved_-12",
        "IC_adm_per_reported_moved_6",
        "new.deaths_per_reported_moved_14",


        "positivetests_cumm",
        "Kliniek_Nieuwe_Opnames_COVID_cumm",
        "new.deaths_cumm",
        "IC_Nieuwe_Opnames_COVID_cumm",
        "positivetests_cumm_period",
        "Kliniek_Nieuwe_Opnames_COVID_cumm_period",
        "new.deaths_cumm_period",
        "IC_Nieuwe_Opnames_COVID_cumm_period",
        "prev_div_days_contagious",
        "prev_div_days_contagious_cumm",
        "prev_div_days_contagious_cumm_period",
        "new.deaths_per_prev_div_days_contagious",
        "new.deaths_cumm_div_prev_div_days_contagious_cumm",
        "new.deaths_cumm_period_div_prev_div_days_contagious_cumm_period",

        "reported_corrected",
        "reported_corrected2",
        "onderrapportagefactor",
        "positivetests_log10",
        "Rt_corr_transit",
        "Rt_corr_transit_period"
    ]
    # "SWE_retail_and_recreation", "SWE_grocery_and_pharmacy", "SWE_residential",
    # "SWE_transit_stations", "SWE_parks", "SWE_workplaces", "SWE_total_cases",
    # "SWE_new_cases", "SWE_total_deaths", "SWE_new_deaths", "SWE_total_cases_per_million",
    # "SWE_new_cases_per_million", "SWE_reproduction_rate", "SWE_icu_patients",
    # "SWE_hosp_patients", "SWE_new_tests", "SWE_total_tests", "SWE_stringency_index"]

    st.title("Interactive Corona Dashboard")
    # st.header("")
    st.subheader("Under construction - Please send feedback to @rcsmit")

    # DAILY STATISTICS ################
    df_temp = None
    what_to_show_day_l = None

    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2021-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    if until_ == "2023-08-23":
        st.sidebar.error("Do you really, really, wanna do this?")
        if st.sidebar.button("Yes I'm ready to rumble"):
            caching.clear_cache()
            st.success("Cache is cleared, please reload to scrape new values")


    mzelst = ["date","cases","hospitalization","deaths","positivetests","hospital_intake_rivm","Hospital_Intake_Proven",
    "Hospital_Intake_Suspected","IC_Intake_Proven","IC_Intake_Suspected","IC_Current","ICs_Used","IC_Cumulative",
    "Hospital_Currently","IC_Deaths_Cumulative","IC_Discharge_Cumulative","IC_Discharge_InHospital","Hospital_Cumulative",
    "Hospital_Intake","IC_Intake","Hosp_Intake_Suspec_Cumul","IC_Intake_Suspected_Cumul","IC_Intake_Proven_Cumsum",
    "IC_Bedden_COVID","IC_Bedden_Non_COVID","Kliniek_Bedden","IC_Nieuwe_Opnames_COVID","Kliniek_Nieuwe_Opnames_COVID",
    "Totaal_bezetting","IC_Opnames_7d","Kliniek_Opnames_7d","Totaal_opnames","Totaal_opnames_7d","Totaal_IC","IC_opnames_14d",
    "Kliniek_opnames_14d","OMT_Check_IC","OMT_Check_Kliniek","positivetests","corrections.cases","net.infection","new.hospitals",
    "corrections.hospitals","net.hospitals","new.deaths","corrections.deaths","net.deaths","positive_7daverage","infections.today.nursery",
    "infections.total.nursery","deaths.today.nursery","deaths.total.nursery","mutations.locations.nursery","total.current.locations.nursery",
    "values.tested_total","values.infected","values.infected_percentage","pos.rate.3d.avg"]

    lijst.extend(mzelst)
    lijst.extend(lijst_oud)
    df = select_period_oud(df, "date", FROM, UNTIL)
    df = extra_calculations_period(df)

    df = drop_columns(df,["Version_x", "Version_y"])
    df = df.drop_duplicates()
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)


    # df,newcolumns = week_to_week(df,["positivetests"])

    # show_R_value_graph, show_R_value_RIVM, show_scenario = False, False, False
    # WDW2=7
    # st.write(df.dtypes)

    w2w = [
        "positivetests",
        "Kliniek_Nieuwe_Opnames_COVID",
        "new.deaths",
    #     "spec_humidity_knmi_derived"
    ]

    how_to_smoothen = "SMA"
    WDW2 = 7
    centersmooth = True


    #st.write(get_duplicate_cols(df))
    df, newcolumns_w2w7, newcolumns2_w2w7 = week_to_week(df, ["positivetests", "Kliniek_Nieuwe_Opnames_COVID"], 7)
    df, newcolumns_w2w14, newcolumns2_w2w14 = week_to_week(df,["positivetests", "Kliniek_Nieuwe_Opnames_COVID"], 14)
    lijst.extend(newcolumns_w2w7) # percentage
    lijst.extend(newcolumns2_w2w7) # index

    lijst.extend(newcolumns_w2w14) # percentage
    lijst.extend(newcolumns2_w2w14) # index

    df, smoothed_columns_w2w0 = smooth_columnlist(df, w2w, how_to_smoothen, WDW2, centersmooth)
    df, newcolumns_w2w, newcolumns2_w2w = week_to_week(df, smoothed_columns_w2w0, 7)

    lijst.extend(newcolumns_w2w) # percentage
    lijst.extend(newcolumns2_w2w) # index

    # diff of diff

    df, smoothed_columns_w2w1 = smooth_columnlist(df, newcolumns_w2w, how_to_smoothen, WDW2, centersmooth)
    df, newcolumns_w2w2, newcolumns2_w2w2 = week_to_week(df, smoothed_columns_w2w1, 7)

    lijst.extend(newcolumns_w2w2) # percentage

    #chd = ["pos_test_0-9", "pos_test_10-19", "pos_test_20-29", "pos_test_30-39", "pos_test_40-49", "pos_test_50-59", "pos_test_60-69", "pos_test_70-79", "pos_test_80-89", "pos_test_90+", "pos_test_20-99","pos_test_0-99", "hosp_0-9", "hosp_10-19", "hosp_20-29", "hosp_30-39", "hosp_40-49", "hosp_50-59", "hosp_60-69", "hosp_70-79", "hosp_80-89", "hosp_90+", "hosp_0-49","hosp_50-79","hosp_70+", "hosp_0-90", "new.deaths_<50", "new.deaths_50-59", "new.deaths_60-69", "new.deaths_70-79", "new.deaths_80-89", "new.deaths_90+", "new.deaths_0-99"]

    #lijst.extend(chd)
    # for n in newcolumns:
    #     .write(df[n])
    # graph_daily       (df,newcolumns,None, "SMA", "line")
    # st.stop()
    global scale_to_x

    week_or_day = st.sidebar.selectbox("Day or Week", ["day", "week"], index=0)
    showday = 0
    if week_or_day != "week":
        how_to_display = st.sidebar.selectbox(
            "What to plot (line/bar)",
            ["line", "line_scaled_to_peak", "line_first_is_1", "bar"],
            index=0,
        )
        if how_to_display == "line_scaled_to_peak" or how_to_display == "line_first_is_1":
            scale_to_x = st.sidebar.selectbox(
                "First / max = ",
                [1, 100],
                index=1,
            )
    else:
        how_to_display = "bar"

    if how_to_display != "bar":
        what_to_show_day_l = st.sidebar.multiselect(
            "What to show left-axis (multiple possible)", lijst, ["positivetests"]
        )
        what_to_show_day_r = st.sidebar.multiselect(
            "What to show right-axis (multiple possible)", lijst, ["Kliniek_Nieuwe_Opnames_COVID"]
        )
        if what_to_show_day_l == None:
            st.warning("Choose something")
            st.stop()
        move_right = st.sidebar.slider("Move curves at right axis (days)", -14, 14, 0)
    else:
        move_right = 0

    showR = False
    if how_to_display == "bar":
        what_to_show_day_l = st.sidebar.selectbox(
            "What to show left-axis (bar -one possible)", lijst, index=4
        )
        # what_to_show_day_l = st.sidebar.multiselect('What to show left-axis (multiple possible)', lijst, ["positivetests"]  )

        showR = st.sidebar.selectbox("Show R number", [True, False], index=0)
        if what_to_show_day_l == []:
            st.error("Choose something for the left-axis")
        if showR == False:
            what_to_show_day_r = st.sidebar.multiselect(
                "What to show right-axis (multiple possible)", lijst, ["positivetests"]
            )
            show_R_value_graph = False
            show_R_value_RIVM = False
        else:

            show_R_value_graph = st.sidebar.checkbox(
                f"Show R from {what_to_show_day_l}", value=True
            )
            show_R_value_RIVM = st.sidebar.checkbox("Show R-value RIVM", value=True)
            what_to_show_day_r = None
            pass  # what_to_show_day_r = st.sidebar.selectbox('What to show right-axis (line - one possible)',lijst, index=6)
        lijst_x = [0, 1, 2, 3, 4, 5, 6]
    else:
        show_R_value_graph = False
        show_R_value_RIVM = False
    if week_or_day == "day" and how_to_display == "bar":
        firstday = int(df.iloc[0]["WEEKDAY"])  # monday = 0
        dagenvdweek = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        showonedaylabel = "Show which day (0 = " + dagenvdweek[firstday] + ")"
        showoneday = st.sidebar.selectbox("Show one day", [True, False], index=1)
        if showoneday:
            showday = st.sidebar.selectbox(showonedaylabel, lijst_x, index=0)
    else:
        showoneday = False
        showday = 0
    global groupby_timeperiod
    global groupby_how
    global graph_how
    # graph_how  = st.sidebar.selectbox("Plotly (interactive with info on hoover) or pyplot (static - easier to copy/paste)", ["plotly", "pyplot"], index=1)
    graph_how = "pyplot"
    showlogyaxis =  st.sidebar.selectbox("Y axis as log", ["No", "2", "10", "logit"], index=0)
    groupby_timeperiod =  st.sidebar.selectbox("GROUPBY : none, week or month", ["none", "1W", "1M"], index=0)
    if groupby_timeperiod != "none":
        groupby_how = st.sidebar.selectbox("GROUPBY : Sum / mean / max", ["sum", "mean"], index=0)
    else:
        groupby_how = "None"

        if groupby_how == "sum":
            df = df.groupby(pd.Grouper(key="date", freq=groupby_timeperiod)).sum().reset_index()
        elif groupby_how == "mean":
            df = df.groupby(pd.Grouper(key="date", freq=groupby_timeperiod)).mean().reset_index()
        elif groupby_how == "max" :
            # TOFIX : gives error
            df = df.groupby(pd.Grouper(key="date", freq=groupby_timeperiod)).max() # .reset_index()

    how_to_smoothen = st.sidebar.selectbox(
        "How to smooth (SMA/savgol)", ["SMA", "savgol"], index=0
    )
    centersmooth =  st.sidebar.selectbox(
        "Smooth in center", [True, False], index=0
    )
    if groupby_timeperiod == "none":
        WDW2 = st.sidebar.slider("Window smoothing curves (days)", 1, 45, 7)
    else:
        WDW2 = st.sidebar.slider("Window smoothing curves (days)", 1, 45, 1)
    if how_to_smoothen == "savgol" and int(WDW2 / 2) == (WDW2 / 2):
        st.warning("When using Savgol, the window has to be uneven")
        st.stop()
    if showR == True:
        WDW3 = st.sidebar.slider("Window smoothing R-number", 1, 14, 7)
        WDW4 = st.sidebar.slider("Calculate R-number over .. days", 1, 14, 4)

        MOVE_WR = st.sidebar.slider("Move the R-curve", -20, 10, -11)
    else:
        showR = False

    if week_or_day == "week":
        how_to_agg_l = st.sidebar.selectbox(
            "How to agg left (sum/mean)", ["sum", "mean"], index=0
        )
        how_to_agg_r = st.sidebar.selectbox(
            "How to agg right (sum/mean)", ["sum", "mean"], index=0
        )
    number_days_contagious = st.sidebar.slider("Aantal dagen besmettelijk", 1, 21, 8)
    global horiz_line
    horiz_line = st.sidebar.number_input(
            "Horizontal line", None, None, 0
        )

    show_scenario = st.sidebar.selectbox("Show Scenario", [True, False], index=1)
    if show_scenario:

        total_cases_0 = st.sidebar.number_input(
            "Total number of positive tests", None, None, 8000
        )

        Rnew_1_ = st.sidebar.slider("R-number first variant", 0.1, 10.0, 0.84)
        Rnew_2_ = st.sidebar.slider("R-number second variant", 0.1, 6.0, 1.16)
        f = st.sidebar.slider("Correction factor", 0.0, 2.0, 1.00)
        ry1 = round(Rnew_1_ * f, 2)
        ry2 = round(Rnew_2_ * f, 2)
        sec_variant = st.sidebar.slider(
            "Percentage second variant at start", 0.0, 100.0, 10.0
        )
        extra_days = st.sidebar.slider("Extra days", 0, 60, 0)
        show_vaccination = st.sidebar.selectbox("Vaccination", [True, False], index=1)
        # avondklok 23 jan
        dag_versoepelingen1 = st.sidebar.slider("Verandering 1 op dag", 0, 300,23)
        versoepeling_factor1_= st.sidebar.slider("Veranderingsfactor 1", 0.0, 2.0, 0.95)

        # Kappers, contactberoepen en winkels open (2mrt)
        dag_versoepelingen2 = st.sidebar.slider("Verandering 2  op dag ", 0, 300, 60)
        versoepeling_factor2_= st.sidebar.slider("Veranderingsfactor2", 0.0, 2.0, 1.03)


    if what_to_show_day_l == []:
        st.error("Choose something for the left-axis")
        st.stop()

    if what_to_show_day_l is not None:

        if week_or_day == "day":
            if move_right != 0 and len(what_to_show_day_r) != 0:
                df, what_to_show_day_r = move_column(df, what_to_show_day_r, move_right)
            if how_to_display == "line":
                graph_daily(
                    df,
                    what_to_show_day_l,
                    what_to_show_day_r,
                    how_to_smoothen,
                    how_to_display, showday
                )
                if len(what_to_show_day_l) > 1:
                    for xx in what_to_show_day_l:
                        graph_daily(df, [xx], None, how_to_smoothen, how_to_display, showday)

            elif how_to_display == "line_scaled_to_peak":
                how_to_norm = "max"
                graph_daily_normed(
                    df,
                    what_to_show_day_l,
                    what_to_show_day_r,
                    how_to_smoothen,
                    how_to_display,
                )
                if len(what_to_show_day_l) > 1:
                    for xx in what_to_show_day_l:
                        graph_daily_normed(
                            df, [xx], None, how_to_smoothen, how_to_display
                        )
            elif how_to_display == "line_first_is_1":
                how_to_norm = "first"
                graph_daily_normed(
                    df,
                    what_to_show_day_l,
                    what_to_show_day_r,
                    how_to_smoothen,
                    how_to_display,
                )
                if len(what_to_show_day_l) > 1:
                    for xx in what_to_show_day_l:
                        graph_daily_normed(
                            df, [xx], None, how_to_smoothen, how_to_display
                        )

            elif how_to_display == "bar":
                # st.write(what_to_show_day_l)
                graph_daily(
                    df,
                    what_to_show_day_l,
                    what_to_show_day_r,
                    how_to_smoothen,
                    how_to_display, showday
                )

        else:
            if showR == True:
                if what_to_show_day_r != None:
                    st.warning("On the right axis the R number will shown")
                graph_week(df, what_to_show_day_l, how_to_agg_l, None, how_to_agg_r)
            else:
                graph_week(
                    df,
                    what_to_show_day_l,
                    how_to_agg_l,
                    what_to_show_day_r,
                    how_to_agg_r,
                )
                if len(what_to_show_day_r) > 0:
                    for xx in what_to_show_day_r:
                        graph_daily_normed(
                            df, [xx], None, how_to_smoothen, how_to_display
                        )

    else:
        st.error("Choose what to show")

    # EXTRA POSSIBLE CALCULATIONS - INTERFACE HAS TO BE WRITTEN

    if st.sidebar.button("Find Correlations"):
        treshhold = st.sidebar.slider("R-number first variant", 0.0, 1.0, 0.8)
        find_correlations(df, treshhold, lijst)
    # import base64
    # coded_data = base64.b64encode(df.to_csv(index=False).encode()).decode()
    # st.markdown(f'<a href="data:file/csv;base64,{coded_data}" download="data.csv">Download Data<a>', unsafe_allow_html = True)
    if len(what_to_show_day_l) == 1 and len(what_to_show_day_r) == 1:  # add lagtime
        #if st.sidebar.button("Find lagtime"):
        find_lag_time(df, what_to_show_day_l, what_to_show_day_r, 0,14)
    # correlation_matrix(df,werkdagen, weekend_)

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)

    # st.download_button(
    #     label="Download data as CSV",
    #     data=csv,

    #     mime='text/csv',
    # )

    covid_dashboard_show_toelichting_footer.show_toelichting_footer()
if __name__ == "__main__":
    #caching.clear_cache()
    main()