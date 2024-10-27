from mortality_yearly_per_capita import get_bevolking
import streamlit as st
from typing import List, Tuple
import pandas as pd
import platform
# import random
import datetime as dt
import pandas as pd
import streamlit as st
# import numpy as np
# from covid_dashboard_rcsmit import find_lag_time
# import plotly.express as px

# from scipy.stats import linregress
# import statsmodels.api as sm
# from scipy import stats
from oversterfte_compleet import  get_sterftedata, get_data_for_series_wrapper,make_df_quantile #, layout_annotations_fig

from oversterfte_eurostats_maand import get_data_eurostat
import pandas as pd

import streamlit as st

@st.cache_data()
def get_rioolwater_oud() -> pd.DataFrame:
    """
    Fetch and process historical wastewater data.

    Returns:
        pd.DataFrame: Processed wastewater data with year, week, and RNA flow per 100,000 people.
    """
    with st.spinner("GETTING ALL DATA ..."):
        url1 = "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
        df = pd.read_csv(url1, delimiter=";", low_memory=False)
        df["Date_measurement"] = pd.to_datetime(df["Date_measurement"], format="%Y-%m-%d")

        # Create 'year' and 'week' columns from the 'Date_measurement' column
        df['jaar'] = df['Date_measurement'].dt.year
        df['week'] = df['Date_measurement'].dt.isocalendar().week

        # df=df[ (df["jaar"] == 2022) & (df["week"] >= 9)& (df["week"] <= 29)]

        # Group by 'year' and 'week', then sum 'RNA_flow_per_100000'
        df = df.groupby(['jaar', 'week'], as_index=False)['RNA_flow_per_100000'].sum()

        # OLS goes wrong with high numbers
        # https://github.com/statsmodels/statsmodels/issues/9258
        df['RNA_flow_per_100000'] = df['RNA_flow_per_100000'] / 10**17
        return df
@st.cache_data()
def get_oversterfte(opdeling):
    # if platform.processor() != "":
    #     file = f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\basevalues_sterfte.csv"
    #     file = f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\basevalues_sterfte_Y0-120_T.csv"
    
    # else:
    #     #file = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/basevalues_sterfte.csv"
    #     file = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/basevalues_sterfte_Y0-120_T.csv"
    # # Load the CSV file
    # df_ = pd.read_csv(file)
    df__ = get_sterftedata(2015, "m_v_0_999")

    df_data= get_data_for_series_wrapper(df__,"m_v_0_999",2015)
    df_, df_corona, df_quantile = make_df_quantile("m_v_0_999", df_data, "week") 
    #df_to_export = df_data[["weeknr", "avg", "aantal_overlijdens"]].copy()
    df_["age_sex"] = "Y0-120_T"

    df_ = df_.assign(
        jaar_week=df_["periodenr"],
        base_value=df_["avg"],
        OBS_VALUE_=df_["m_v_0_999"]
    )


    df_ = df_[["jaar_week","base_value","OBS_VALUE_"]]
    df_["age_sex"]= "Y0-120_T"

    df_["jaar"] = (df_["jaar_week"].str[:4]).astype(int)
    df_["week"] = (df_["jaar_week"].str[5:]).astype(int)
    df_["YearWeekISO"] = df_["jaar"].astype(int).astype(str) + "-W"+ df_["week"].astype(int).astype(str)
    df_["TIME_PERIOD"] = df_["jaar"].astype(int).astype(str) + "-W"+ df_["week"].astype(int).astype(str)

    # Function to extract age_low and age_high based on patterns
    def extract_age_ranges(age: str) -> Tuple[int, int]:
        """
        Extract age ranges from age string.

        Args:
            age (str): Age string.

        Returns:
            Tuple[int, int]: Lower and upper age range.
        """
        if age == "TOTAL":
            return 999, 999
        elif age == "UNK":
            return 9999, 9999
        elif age == "Y_LT5":
            return 0, 4
        elif age == "Y_GE90":
            return 90, 120
        else:
            # Extract the numeric part from the pattern 'Y10-14'
            parts = age[1:].split('-')
            return int(parts[0]), int(parts[1])

    # Apply the function to create the new columns
    #df_['age_low'], df_['age_high'] = zip(*df_['age'].apply(extract_age_ranges))
    
    df_["jaar"] = df_["jaar"].astype(int)
    df_["week"] = df_["week"].astype(int)
    #df_ = df_[df_["sex"] == "T"]

    def add_custom_age_group_deaths(df: pd.DataFrame, min_age: int, max_age: int) -> pd.DataFrame:
        """
        Add custom age group deaths to the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            min_age (int): Minimum age for the group.
            max_age (int): Maximum age for the group.

        Returns:
            pd.DataFrame: Dataframe with added custom age group.
        """
        # Filter the data based on the dynamic age range
        df_filtered = df_[(df_['age_low'] >= min_age) & (df_['age_high'] <= max_age)]

        # Group by TIME_PERIOD (week), sex, and sum the deaths (OBS_VALUE)
        totals = df_filtered.groupby(['TIME_PERIOD', 'sex'], observed=False)['avg'].sum().reset_index()

        # Assign a new label for the age group (dynamic)
        totals['age'] = f'Y{min_age}-{max_age}'
        totals["age_sex"] = totals["age"] + "_" +totals["sex"]
        totals["jaar"] = (totals["TIME_PERIOD"].str[:4]).astype(int)
        totals["week"] = (totals["TIME_PERIOD"].str[6:]).astype(int)
        return totals


    # for i in opdeling:
    #     custom_age_group = add_custom_age_group_deaths(df_, i[0], i[1])
    #     df_ = pd.concat([df_, custom_age_group], ignore_index=True)
    #df_["age_sex"] = df_["age_group"] + "_" +df_["geslacht"]
  
    df_bevolking = get_bevolking("NL", opdeling)
    
    df__ = pd.merge(df_, df_bevolking, on=['jaar', 'age_sex'], how='outer')

    
    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["base_value"].notna()]
    #df__ = df__[df__["jaar"] != 2024]
    df__["per100k"] = round(df__["OBS_VALUE_"]/df__["aantal"]*100000,1)


    df__["oversterfte"] = df__["OBS_VALUE_"] - df__["base_value"]
    df__["p_score"] = ( df__["OBS_VALUE_"]- df__["base_value"]) /   df__["base_value"]
   
    return df__

@st.cache_data()
def get_maandelijkse_overlijdens(oorzaak):

    if platform.processor() != "":
        file = f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\overlijdens_{oorzaak}.csv"
    else:
        file = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_{oorzaak}.csv"
 
    # Load the CSV file
    df = pd.read_csv(file)

    # Melt the dataframe
    df_melted = df.melt(id_vars=['maand'], var_name='year', value_name=f'OBS_VALUE_{oorzaak}')

    # Map Dutch month names to their numerical equivalent
    month_map = {
        "Januari": "01", "Februari": "02", "Maart": "03", "April": "04", 
        "Mei": "05", "Juni": "06", "Juli": "07", "Augustus": "08", 
        "September": "09", "Oktober": "10", "November": "11", "December": "12"
    }

    # Apply mapping and create YearMonth column in the format YYYY-MM
    df_melted['month'] = df_melted['maand'].map(month_map)
    df_melted['YearMonth'] = df_melted['year'] + '-' + df_melted['month']

    # Drop extra columns and keep only relevant ones
    df_melted_clean = df_melted[['YearMonth', f'OBS_VALUE_{oorzaak}']].dropna()
    return df_melted_clean

@st.cache_data()
def get_sterfte(opdeling: List[Tuple[int, int]], country: str = "NL") -> pd.DataFrame:
    """
    Fetch and process mortality data for a given country.

    Args:
        opdeling (List[Tuple[int, int]]): List of age ranges to process.
        country (str, optional): Country code. Defaults to "NL".

    Returns:
        pd.DataFrame: Processed mortality data.
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true

    if 1==2:
        if country == "NL":
            if platform.processor() != "":
                file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_NL.csv"
            else:
                file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_NL.csv"
        elif country == "BE":
            if platform.processor() != "":
                file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_BE.csv"
            else:
                file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_BE.csv"
        else:
            st.error(f"Error in country {country}")
        
        df_ = pd.read_csv(
            file,
            delimiter=",",
                low_memory=False,
                )
    df_ = get_data_eurostat()
   

    df_=df_[df_["geo"] == country]

    df_["age_sex"] = df_["age"] + "_" +df_["sex"]

    # Function to extract age_low and age_high based on patterns
    def extract_age_ranges(age: str) -> Tuple[int, int]:
        """
        Extract age ranges from age string.

        Args:
            age (str): Age string.

        Returns:
            Tuple[int, int]: Lower and upper age range.
        """
        if age == "TOTAL":
            return 999, 999
        elif age == "UNK":
            return 9999, 9999
        elif age == "Y_LT5":
            return 0, 4
        elif age == "Y_GE90":
            return 90, 120
        else:
            # Extract the numeric part from the pattern 'Y10-14'
            parts = age[1:].split('-')
            return int(parts[0]), int(parts[1])

    # Apply the function to create the new columns
    df_['age_low'], df_['age_high'] = zip(*df_['age'].apply(extract_age_ranges))
    
    df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["week"] = (df_["TIME_PERIOD"].str[6:]).astype(int)

    df_ = df_[df_["sex"] == "T"]

    def add_custom_age_group_deaths(df: pd.DataFrame, min_age: int, max_age: int) -> pd.DataFrame:
        """
        Add custom age group deaths to the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            min_age (int): Minimum age for the group.
            max_age (int): Maximum age for the group.

        Returns:
            pd.DataFrame: Dataframe with added custom age group.
        """
        # Filter the data based on the dynamic age range
        df_filtered = df_[(df_['age_low'] >= min_age) & (df_['age_high'] <= max_age)]

        # Group by TIME_PERIOD (week), sex, and sum the deaths (OBS_VALUE)
        totals = df_filtered.groupby(['TIME_PERIOD', 'sex'], observed=False)['OBS_VALUE'].sum().reset_index()

        # Assign a new label for the age group (dynamic)
        totals['age'] = f'Y{min_age}-{max_age}'
        totals["age_sex"] = totals["age"] + "_" +totals["sex"]
        totals["jaar"] = (totals["TIME_PERIOD"].str[:4]).astype(int)
        totals["week"] = (totals["TIME_PERIOD"].str[6:]).astype(int)
        return totals

    for i in opdeling:
        custom_age_group = add_custom_age_group_deaths(df_, i[0], i[1])
        df_ = pd.concat([df_, custom_age_group], ignore_index=True)

    df_bevolking = get_bevolking("NL", opdeling)

    df__ = pd.merge(df_, df_bevolking, on=['jaar', 'age_sex'], how='outer')
    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["OBS_VALUE"].notna()]
    #df__ = df__[df__["jaar"] != 2024]
    df__["per100k"] = round(df__["OBS_VALUE"]/df__["aantal"]*100000,1)

    return df__

@st.cache_data()
def get_rioolwater():
    # https://www.rivm.nl/corona/actueel/weekcijfers

    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rioolwater_2024okt.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwater_2024okt.csv"
    df = pd.read_csv(
        file,
        delimiter=";",

        low_memory=False,
    )

    return df

# Function to convert date to YearWeekISO
def date_to_yearweekiso(date):
    #date = dt.datetime.strptime(date_str, '%Y-%m-%d')
    # Convert to YearWeekISO format (ISO year and ISO week)
    return date.strftime('%G-W%V')

@st.cache_data()
def get_vaccinaties_owid():
    # https://ourworldindata.org/grapher/daily-covid-19-vaccination-doses?tab=chart&country=~NLD
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\vaccinations_OWOD_NL_daily.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/vaccinations_OWOD_NL_daily.csv"
        
    df = pd.read_csv(
        file,
        delimiter=",",

        low_memory=False,
    )

    df['age_sex'] ='TOTAL_T'


    # Convert 'datum' to datetime if it's not already
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')

    # Get the range of dates from the minimum date in the 'datum' column to today
    start_date = df['datum'].min()
    end_date = dt.datetime.now()

    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a DataFrame from the date range
    date_df = pd.DataFrame(date_range, columns=['datum'])

    # Merge the original dataframe with the new date dataframe to fill in missing dates
    df_filled = pd.merge(date_df, df, on='datum', how='left')
    df_filled["TotalDoses"] = df_filled["TotalDoses"].fillna(0)
    df_filled["age_sex"] = df_filled["age_sex"].fillna("TOTAL_T")
    
    # If you want to keep any existing data in df, you can use:
    df_filled = pd.concat([df, df_filled]).drop_duplicates(subset='datum').sort_values('datum').reset_index(drop=True)

    # Apply the conversion function to the 'datum' column
    df_filled['YearWeekISO'] = df_filled['datum'].apply(date_to_yearweekiso)

    df_filled["jaar"] = (df_filled["YearWeekISO"].str[:4]).astype(int)
    df_filled["week"] = (df_filled["YearWeekISO"].str[6:]).astype(int)
    df_filled = df_filled.groupby(['jaar','week','YearWeekISO']).sum(numeric_only=True).reset_index()
    
    return df_filled

@st.cache_data()
def get_vaccinaties():
    # https://www.ecdc.europa.eu/en/publications-data/data-covid-19-vaccination-eu-eea

    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\vaccinaties_NL_2023.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/vaccinaties_NL_2023.csv"
    df = pd.read_csv(
        file,
        delimiter=",",

        low_memory=False,
    )

    df['age_sex'] =df['age_sex']+'_T'

    df = df.groupby(['YearWeekISO', 'age_sex']).sum(numeric_only=True).reset_index()
    df['TotalDoses'] = df[['FirstDose', 'SecondDose', 'DoseAdditional1', 'DoseAdditional2',
                       'DoseAdditional3', 'DoseAdditional4', 'DoseAdditional5', 'UnknownDose']].sum(axis=1)

    df["jaar"] = (df["YearWeekISO"].str[:4]).astype(int)
    df["week"] = (df["YearWeekISO"].str[6:]).astype(int)

    df["periodenr"] = df["jaar"].astype(str) + "_" + df["week"].astype(str).str.zfill(2)
    

    return df

def get_ziekenhuis_ic() -> pd.DataFrame:
    """
    Fetch and process historical wastewater data.

    Returns:
        pd.DataFrame: Processed wastewater data with year, week, and RNA flow per 100,000 people.
    """
    with st.spinner("GETTING ALL DATA ..."):
     
        # URLs for the datasets
        url1 = "https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep_tm_03102021.csv"
        url2 = "https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv"

        # Load the two CSV files into dataframes
        df1 = pd.read_csv(url1, delimiter=';')
        df2 = pd.read_csv(url2, delimiter=';')

        # Combine the two dataframes
        df_combined = pd.concat([df1, df2], ignore_index=True)

        # Ensure the Date_of_statistics_week_start column is in datetime format
        df_combined['Date_of_statistics_week_start_nr'] =  pd.to_datetime(df_combined["Date_of_statistics_week_start"], format="%Y-%m-%d")
        
        # Create 'year' and 'week' columns from the 'Date_measurement' column
        df_combined['jaar'] = df_combined['Date_of_statistics_week_start_nr'].dt.year
        df_combined['week'] = df_combined['Date_of_statistics_week_start_nr'].dt.isocalendar().week
        df_combined["periodenr"] = df_combined["jaar"].astype(str) + "_" + df_combined["week"].astype(str).str.zfill(2)
   
        # df=df[ (df["jaar"] == 2022) & (df["week"] >= 9)& (df["week"] <= 29)]

        # Group by 'year' and 'week', then sum 'RNA_flow_per_100000'
        df_combined = df_combined.groupby(['jaar', 'week'], as_index=False).sum(["Hospital_admission", "IC_admission"])


        
        return df_combined


def yearweek_to_yearmonth(yearweek: str) -> str:
    """
    Convert YearWeekISO to YearMonth format.

    Args:
        yearweek (str): Year and week in ISO format (e.g., "2023-W01").

    Returns:
        str: Year and month in format "YYYY-MM".
    """
    year, week = yearweek.split('-W')
    # Calculate the Monday of the given ISO week
    date = dt.datetime.strptime(f'{year} {week} 1', '%G %V %u')
    # Extract year and month from the date
    return date.strftime('%Y-%m')


def from_week_to_month(rw, how):
    rw["YearWeekISO"] = rw["jaar"].astype(int).astype(str) + "-W"+ rw["week"].astype(int).astype(str)

    # Apply the conversion function to the YearWeekISO column
    rw['YearMonth'] = rw['YearWeekISO'].apply(yearweek_to_yearmonth)
    if how == "sum":
        rw = rw.groupby(['YearMonth'], as_index=False).sum( numeric_only=True)
    else:
        rw = rw.groupby(['YearMonth'], as_index=False).mean( numeric_only=True)
    return rw
