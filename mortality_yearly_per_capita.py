
import pandas as pd

import plotly.graph_objects as go
import eurostat
import platform
import streamlit as st
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
try:
    st.set_page_config(layout="wide")
except:
    pass

def get_bevolking(country, opdeling):
    """Gegt beolking, 1960-2024 naar geslacht en leeftijd

    Args:
        country (_type_): _description_
        opdeling (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if country == "NL":
        if platform.processor() != "":
            #https://opendata.cbs.nl/#/CBS/nl/dataset/03759ned/table?dl=39E0B
            #file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_NL.csv"
            file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_nl_crosstable.csv"

        else: 
            #file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_nl_crosstable.csv"
        
        df = pd.read_csv(file, sep=';', decimal=',')

        df_long = df.melt(id_vars=['Geslacht', 'Leeftijd'], var_name='jaar', value_name='aantal')

        df_long = df_long.rename(columns={'Geslacht': 'geslacht', 'Leeftijd': 'leeftijd'})
        df_long = df_long[['leeftijd', 'geslacht', 'jaar', 'aantal']]
        df_long['jaar'] = df_long['jaar'].astype(int)
        df_long['aantal'] = df_long['aantal'].fillna(0).astype(int)
        data = df_long.sort_values(['leeftijd', 'geslacht', 'jaar']).reset_index(drop=True)
     
        #print(df_long.to_csv(sep=';', index=False))
    elif country == "BE":
        # https://ec.europa.eu/eurostat/databrowser/view/demo_pjan__custom_12780094/default/table?lang=en
        if platform.processor() != "":
            file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_BE.csv"
        else: 
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_BE.csv"

        data = pd.read_csv(
            file,
            delimiter=";",
            
            low_memory=False,
        )
    else:
        st.error(f"Error in country {country}")
   
   

    # nieuwe dataset bevat al gemiddelde bevolking ipv bevolking op 1 januari

    # Sorteren op leeftijd (en eventueel op geslacht en jaar)
    df_bevolking = data.sort_values(by=['geslacht', 'jaar', 'leeftijd'])

    # Hernoemen van 'aantal' naar 'aantal_original'
    df_bevolking.rename(columns={'aantal': 'aantal_original'}, inplace=True)

    # Aantal verschuiven om de waarde voor de volgende leeftijd (x+1) te krijgen
    df_bevolking['aantal_shifted'] = df_bevolking['aantal_original'].shift(-1)

    # Corrected aantal: als leeftijd < 99, gemiddelde van aantal_original en aantal_shifted
    # Anders blijft aantal gelijk aan aantal_original
    df_bevolking['aantal'] = df_bevolking.apply(
        lambda row: row['aantal_original'] if row['leeftijd'] >= 99 else (row['aantal_original'] + row['aantal_shifted']) / 2,
        axis=1
    )

    # Verwijderen van kolom 'aantal_shifted' omdat deze niet meer nodig is
    df_bevolking.drop(columns=['aantal_shifted'], inplace=True)
    data['leeftijd'] = data['leeftijd'].astype(int)
    
    # Define age bins and labels
    bins = list(range(0, 95, 5)) + [1000]  # [0, 5, 10, ..., 90, 1000]
    labels = [f'Y{i}-{i+4}' for i in range(0, 90, 5)] + ['Y90-120']


    # Create a new column for age bins
    data['age_group'] = pd.cut(data['leeftijd'], bins=bins, labels=labels, right=False)
   

    # Group by year, gender, and age_group and sum the counts
    grouped_data = data.groupby(['jaar', 'geslacht', 'age_group'], observed=False)['aantal'].sum().reset_index()

    # Save the resulting data to a new CSV file
    # grouped_data.to_csv('grouped_population_by_age_2010_2024.csv', index=False, sep=';')

    # print("Grouping complete and saved to grouped_population_by_age_2010_2024.csv")
    grouped_data["age_sex"] = grouped_data['age_group'].astype(str) +"_"+grouped_data['geslacht'].astype(str)
    
    
    for s in ["M", "F", "T"]:
        grouped_data.replace(f'Y0-4_{s}', f'Y_LT5_{s}', inplace=True)
        grouped_data.replace(f'Y90-120_{s}',f'Y_GE90_{s}', inplace=True)
    

    # Calculate totals per year and gender (geslacht)
    totals = grouped_data.groupby(['jaar', 'geslacht'], observed=False)['aantal'].sum().reset_index()


    # Assign 'Total' as the age group for these sums
    totals['age_group'] = 'TOTAL'
    totals['age_sex'] = "TOTAL_" + totals['geslacht'].astype(str)

    # Concatenate the original grouped data with the totals
    final_data = pd.concat([grouped_data, totals], ignore_index=True)
  

    def add_custom_age_group(data, min_age, max_age):
        # Find the age group labels that fit within the specified min and max age
        valid_age_groups = [f'Y{i}-{i+4}' for i in range(min_age, max_age + 1, 5) if i < 90]
        
        # Include edge cases for Y_LT5 and Y_GE90 if they fall within the range
        if min_age <= 4:
            valid_age_groups.append('Y_LT5')
        if max_age >= 90:
            valid_age_groups.append('Y_GE90')

        # Filter the grouped data based on these age groups and sum
        custom_age_group = data[data['age_group'].isin(valid_age_groups)].groupby(['jaar', 'geslacht'], observed=False)['aantal'].sum().reset_index()

        # Assign the label for the new age group
        custom_age_group['age_group'] = f'Y{min_age}-{max_age}'
        custom_age_group['age_sex'] = f'Y{min_age}-{max_age}_' + custom_age_group['geslacht'].astype(str)

        return custom_age_group
    # Concatenate the original grouped data with the totals

    for i in opdeling:
        
        custom_age_group = add_custom_age_group(data, i[0], i[1])
        final_data = pd.concat([final_data, custom_age_group], ignore_index=True)

    return final_data

@st.cache_data()
def get_sterfte(opdeling, country="NL"):
    """Fetch and process weekly mortality data from Eurostat for a given country.

    Args:
        opdeling (list[list[int]]): List of [min_age, max_age] pairs defining custom age groups.
        country (str): Country code, either 'NL' or 'BE'. Defaults to 'NL'.

    Returns:
        pd.DataFrame: DataFrame with yearly mortality, population, and per-100k figures
                      for each age_sex combination.
    """
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
        st.stop()

    df_ = pd.read_csv(file, delimiter=",", low_memory=False)
    df_ = df_[df_["geo"] == country]
    df_["age_sex"] = df_["age"] + "_" + df_["sex"]

    def extract_age_ranges(age: str) -> tuple[int, int]:
        """Extract (age_low, age_high) from Eurostat age group string.

        Args:
            age (str): Age group string e.g. 'Y10-14', 'TOTAL', 'Y_GE90'.

        Returns:
            tuple[int, int]: Lower and upper age bounds.
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
            try:
                parts = age.lstrip('Y').split('-')
                return int(parts[0]), int(parts[1])
            except (IndexError, ValueError):
                return 9999, 9999  # malformed → excluded from custom groups

    df_['age_low'], df_['age_high'] = zip(*df_['age'].apply(extract_age_ranges))
    df_["jaar"] = df_["TIME_PERIOD"].str[:4].astype(int)
    df_["weeknr"] = df_["TIME_PERIOD"].str[6:].astype(int)

    def add_custom_age_group_deaths(df_source: pd.DataFrame, min_age: int, max_age: int) -> pd.DataFrame:
        """Sum weekly deaths across all raw age groups within [min_age, max_age].

        Args:
            df_source (pd.DataFrame): Source DataFrame with raw age groups.
            min_age (int): Lower bound of the custom age group (inclusive).
            max_age (int): Upper bound of the custom age group (inclusive).

        Returns:
            pd.DataFrame: Aggregated DataFrame with custom age group label.
        """
     
        df_filtered = df_source[
            (df_source['age_low'] >= min_age) &
            (df_source['age_high'] <= max_age) &
            (df_source['age_low'] != 999) &    # exclude TOTAL
            (df_source['age_low'] != 9999)     # exclude UNK / malformed
        ]
       
        totals = df_filtered.groupby(
            ['TIME_PERIOD', 'sex'], observed=False
        )['OBS_VALUE'].sum().reset_index()
        totals['age'] = f'Y{min_age}-{max_age}'
        totals['age_sex'] = totals['age'] + '_' + totals['sex']
        totals['jaar'] = totals['TIME_PERIOD'].str[:4].astype(int)
        totals['age_low'] = min_age
        totals['age_high'] = max_age
        return totals

    # Build all custom groups from the original df_ BEFORE any concat
    df_custom = pd.concat(
        [add_custom_age_group_deaths(df_, i[0], i[1]) for i in opdeling],
        ignore_index=True
    )

    # FIX: keep only original rows NOT already covered by a custom group
    # to prevent double-counting when e.g. Y80-120 and Y80-84/Y85-89/Y90-120 coexist
    custom_labels = {f'Y{i[0]}-{i[1]}' for i in opdeling}
    df_originals_only = df_[~df_['age'].isin(custom_labels)]
    df_ = pd.concat([df_custom, df_originals_only], ignore_index=True)
    df_["TIMEPERIODEAGESEX"] = df_['TIME_PERIOD'] + '_' + df_['age_sex']
    #st.write(df_)
    # Sanity check: no age_sex should appear more than once per TIME_PERIOD
    dupe_check = df_.groupby(['TIME_PERIOD', 'age_sex']).size()
    if (dupe_check > 1).any():
        pass # st.warning(f"⚠️ Duplicate age_sex entries detected after concat — check opdeling logic / {dupe_check}")

    df_bevolking = get_bevolking(country, opdeling)
    #st.write(df_)
    summed_per_year = df_.groupby(["jaar", 'age_sex'])['OBS_VALUE'].sum()

    df__ = pd.merge(summed_per_year, df_bevolking, on=['jaar', 'age_sex'], how='outer')

    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["OBS_VALUE"].notna()]
    df__["per100k"] = round(df__["OBS_VALUE"] / df__["aantal"] * 100000, 1)
    #st.write(df__)
    return df__

def get_sterfte_oud(opdeling,country="NL"):
    """_summary_

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
          

    if country == "NL": 

        # https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_r_mwk_05/1.0/*.*.*.*.*?c[freq]=W&c[age]=TOTAL,Y_LT5,Y5-9,Y10-14,Y15-19,Y20-24,Y25-29,Y30-34,Y35-39,Y40-44,Y45-49,Y50-54,Y55-59,Y60-64,Y65-69,Y70-74,Y75-79,Y80-84,Y85-89,Y_GE90,UNK&c[sex]=T,M,F&c[unit]=NR&c[geo]=NL&c[TIME_PERIOD]=2026-W05,2026-W04,2026-W03,2026-W02,2026-W01,2025-W52,2025-W51,2025-W50,2025-W49,2025-W48,2025-W47,2025-W46,2025-W45,2025-W44,2025-W43,2025-W42,2025-W41,2025-W40,2025-W39,2025-W38,2025-W37,2025-W36,2025-W35,2025-W34,2025-W33,2025-W32,2025-W31,2025-W30,2025-W29,2025-W28,2025-W27,2025-W26,2025-W25,2025-W24,2025-W23,2025-W22,2025-W21,2025-W20,2025-W19,2025-W18,2025-W17,2025-W16,2025-W15,2025-W14,2025-W13,2025-W12,2025-W11,2025-W10,2025-W09,2025-W08,2025-W07,2025-W06,2025-W05,2025-W04,2025-W03,2025-W02,2025-W01,2024-W52,2024-W51,2024-W50,2024-W49,2024-W48,2024-W47,2024-W46,2024-W45,2024-W44,2024-W43,2024-W42,2024-W41,2024-W40,2024-W39,2024-W38,2024-W37,2024-W36,2024-W35,2024-W34,2024-W33,2024-W32,2024-W31,2024-W30,2024-W29,2024-W28,2024-W27,2024-W26,2024-W25,2024-W24,2024-W23,2024-W22,2024-W21,2024-W20,2024-W19,2024-W18,2024-W17,2024-W16,2024-W15,2024-W14,2024-W13,2024-W12,2024-W11,2024-W10,2024-W09,2024-W08,2024-W07,2024-W06,2024-W05,2024-W04,2024-W03,2024-W02,2024-W01,2023-W52,2023-W51,2023-W50,2023-W49,2023-W48,2023-W47,2023-W46,2023-W45,2023-W44,2023-W43,2023-W42,2023-W41,2023-W40,2023-W39,2023-W38,2023-W37,2023-W36,2023-W35,2023-W34,2023-W33,2023-W32,2023-W31,2023-W30,2023-W29,2023-W28,2023-W27,2023-W26,2023-W25,2023-W24,2023-W23,2023-W22,2023-W21,2023-W20,2023-W19,2023-W18,2023-W17,2023-W16,2023-W15,2023-W14,2023-W13,2023-W12,2023-W11,2023-W10,2023-W09,2023-W08,2023-W07,2023-W06,2023-W05,2023-W04,2023-W03,2023-W02,2023-W01,2022-W52,2022-W51,2022-W50,2022-W49,2022-W48,2022-W47,2022-W46,2022-W45,2022-W44,2022-W43,2022-W42,2022-W41,2022-W40,2022-W39,2022-W38,2022-W37,2022-W36,2022-W35,2022-W34,2022-W33,2022-W32,2022-W31,2022-W30,2022-W29,2022-W28,2022-W27,2022-W26,2022-W25,2022-W24,2022-W23,2022-W22,2022-W21,2022-W20,2022-W19,2022-W18,2022-W17,2022-W16,2022-W15,2022-W14,2022-W13,2022-W12,2022-W11,2022-W10,2022-W09,2022-W08,2022-W07,2022-W06,2022-W05,2022-W04,2022-W03,2022-W02,2022-W01,2021-W52,2021-W51,2021-W50,2021-W49,2021-W48,2021-W47,2021-W46,2021-W45,2021-W44,2021-W43,2021-W42,2021-W41,2021-W40,2021-W39,2021-W38,2021-W37,2021-W36,2021-W35,2021-W34,2021-W33,2021-W32,2021-W31,2021-W30,2021-W29,2021-W28,2021-W27,2021-W26,2021-W25,2021-W24,2021-W23,2021-W22,2021-W21,2021-W20,2021-W19,2021-W18,2021-W17,2021-W16,2021-W15,2021-W14,2021-W13,2021-W12,2021-W11,2021-W10,2021-W09,2021-W08,2021-W07,2021-W06,2021-W05,2021-W04,2021-W03,2021-W02,2021-W01,2020-W53,2020-W52,2020-W51,2020-W50,2020-W49,2020-W48,2020-W47,2020-W46,2020-W45,2020-W44,2020-W43,2020-W42,2020-W41,2020-W40,2020-W39,2020-W38,2020-W37,2020-W36,2020-W35,2020-W34,2020-W33,2020-W32,2020-W31,2020-W30,2020-W29,2020-W28,2020-W27,2020-W26,2020-W25,2020-W24,2020-W23,2020-W22,2020-W21,2020-W20,2020-W19,2020-W18,2020-W17,2020-W16,2020-W15,2020-W14,2020-W13,2020-W12,2020-W11,2020-W10,2020-W09,2020-W08,2020-W07,2020-W06,2020-W05,2020-W04,2020-W03,2020-W02,2020-W01,2019-W52,2019-W51,2019-W50,2019-W49,2019-W48,2019-W47,2019-W46,2019-W45,2019-W44,2019-W43,2019-W42,2019-W41,2019-W40,2019-W39,2019-W38,2019-W37,2019-W36,2019-W35,2019-W34,2019-W33,2019-W32,2019-W31,2019-W30,2019-W29,2019-W28,2019-W27,2019-W26,2019-W25,2019-W24,2019-W23,2019-W22,2019-W21,2019-W20,2019-W19,2019-W18,2019-W17,2019-W16,2019-W15,2019-W14,2019-W13,2019-W12,2019-W11,2019-W10,2019-W09,2019-W08,2019-W07,2019-W06,2019-W05,2019-W04,2019-W03,2019-W02,2019-W01,2018-W52,2018-W51,2018-W50,2018-W49,2018-W48,2018-W47,2018-W46,2018-W45,2018-W44,2018-W43,2018-W42,2018-W41,2018-W40,2018-W39,2018-W38,2018-W37,2018-W36,2018-W35,2018-W34,2018-W33,2018-W32,2018-W31,2018-W30,2018-W29,2018-W28,2018-W27,2018-W26,2018-W25,2018-W24,2018-W23,2018-W22,2018-W21,2018-W20,2018-W19,2018-W18,2018-W17,2018-W16,2018-W15,2018-W14,2018-W13,2018-W12,2018-W11,2018-W10,2018-W09,2018-W08,2018-W07,2018-W06,2018-W05,2018-W04,2018-W03,2018-W02,2018-W01,2017-W52,2017-W51,2017-W50,2017-W49,2017-W48,2017-W47,2017-W46,2017-W45,2017-W44,2017-W43,2017-W42,2017-W41,2017-W40,2017-W39,2017-W38,2017-W37,2017-W36,2017-W35,2017-W34,2017-W33,2017-W32,2017-W31,2017-W30,2017-W29,2017-W28,2017-W27,2017-W26,2017-W25,2017-W24,2017-W23,2017-W22,2017-W21,2017-W20,2017-W19,2017-W18,2017-W17,2017-W16,2017-W15,2017-W14,2017-W13,2017-W12,2017-W11,2017-W10,2017-W09,2017-W08,2017-W07,2017-W06,2017-W05,2017-W04,2017-W03,2017-W02,2017-W01,2016-W52,2016-W51,2016-W50,2016-W49,2016-W48,2016-W47,2016-W46,2016-W45,2016-W44,2016-W43,2016-W42,2016-W41,2016-W40,2016-W39,2016-W38,2016-W37,2016-W36,2016-W35,2016-W34,2016-W33,2016-W32,2016-W31,2016-W30,2016-W29,2016-W28,2016-W27,2016-W26,2016-W25,2016-W24,2016-W23,2016-W22,2016-W21,2016-W20,2016-W19,2016-W18,2016-W17,2016-W16,2016-W15,2016-W14,2016-W13,2016-W12,2016-W11,2016-W10,2016-W09,2016-W08,2016-W07,2016-W06,2016-W05,2016-W04,2016-W03,2016-W02,2016-W01,2015-W53,2015-W52,2015-W51,2015-W50,2015-W49,2015-W48,2015-W47,2015-W46,2015-W45,2015-W44,2015-W43,2015-W42,2015-W41,2015-W40,2015-W39,2015-W38,2015-W37,2015-W36,2015-W35,2015-W34,2015-W33,2015-W32,2015-W31,2015-W30,2015-W29,2015-W28,2015-W27,2015-W26,2015-W25,2015-W24,2015-W23,2015-W22,2015-W21,2015-W20,2015-W19,2015-W18,2015-W17,2015-W16,2015-W15,2015-W14,2015-W13,2015-W12,2015-W11,2015-W10,2015-W09,2015-W08,2015-W07,2015-W06,2015-W05,2015-W04,2015-W03,2015-W02,2015-W01,2014-W52,2014-W51,2014-W50,2014-W49,2014-W48,2014-W47,2014-W46,2014-W45,2014-W44,2014-W43,2014-W42,2014-W41,2014-W40,2014-W39,2014-W38,2014-W37,2014-W36,2014-W35,2014-W34,2014-W33,2014-W32,2014-W31,2014-W30,2014-W29,2014-W28,2014-W27,2014-W26,2014-W25,2014-W24,2014-W23,2014-W22,2014-W21,2014-W20,2014-W19,2014-W18,2014-W17,2014-W16,2014-W15,2014-W14,2014-W13,2014-W12,2014-W11,2014-W10,2014-W09,2014-W08,2014-W07,2014-W06,2014-W05,2014-W04,2014-W03,2014-W02,2014-W01,2013-W52,2013-W51,2013-W50,2013-W49,2013-W48,2013-W47,2013-W46,2013-W45,2013-W44,2013-W43,2013-W42,2013-W41,2013-W40,2013-W39,2013-W38,2013-W37,2013-W36,2013-W35,2013-W34,2013-W33,2013-W32,2013-W31,2013-W30,2013-W29,2013-W28,2013-W27,2013-W26,2013-W25,2013-W24,2013-W23,2013-W22,2013-W21,2013-W20,2013-W19,2013-W18,2013-W17,2013-W16,2013-W15,2013-W14,2013-W13,2013-W12,2013-W11,2013-W10,2013-W09,2013-W08,2013-W07,2013-W06,2013-W05,2013-W04,2013-W03,2013-W02,2013-W01,2012-W52,2012-W51,2012-W50,2012-W49,2012-W48,2012-W47,2012-W46,2012-W45,2012-W44,2012-W43,2012-W42,2012-W41,2012-W40,2012-W39,2012-W38,2012-W37,2012-W36,2012-W35,2012-W34,2012-W33,2012-W32,2012-W31,2012-W30,2012-W29,2012-W28,2012-W27,2012-W26,2012-W25,2012-W24,2012-W23,2012-W22,2012-W21,2012-W20,2012-W19,2012-W18,2012-W17,2012-W16,2012-W15,2012-W14,2012-W13,2012-W12,2012-W11,2012-W10,2012-W09,2012-W08,2012-W07,2012-W06,2012-W05,2012-W04,2012-W03,2012-W02,2012-W01,2011-W52,2011-W51,2011-W50,2011-W49,2011-W48,2011-W47,2011-W46,2011-W45,2011-W44,2011-W43,2011-W42,2011-W41,2011-W40,2011-W39,2011-W38,2011-W37,2011-W36,2011-W35,2011-W34,2011-W33,2011-W32,2011-W31,2011-W30,2011-W29,2011-W28,2011-W27,2011-W26,2011-W25,2011-W24,2011-W23,2011-W22,2011-W21,2011-W20,2011-W19,2011-W18,2011-W17,2011-W16,2011-W15,2011-W14,2011-W13,2011-W12,2011-W11,2011-W10,2011-W09,2011-W08,2011-W07,2011-W06,2011-W05,2011-W04,2011-W03,2011-W02,2011-W01,2010-W52,2010-W51,2010-W50,2010-W49,2010-W48,2010-W47,2010-W46,2010-W45,2010-W44,2010-W43,2010-W42,2010-W41,2010-W40,2010-W39,2010-W38,2010-W37,2010-W36,2010-W35,2010-W34,2010-W33,2010-W32,2010-W31,2010-W30,2010-W29,2010-W28,2010-W27,2010-W26,2010-W25,2010-W24,2010-W23,2010-W22,2010-W21,2010-W20,2010-W19,2010-W18,2010-W17,2010-W16,2010-W15,2010-W14,2010-W13,2010-W12,2010-W11,2010-W10,2010-W09,2010-W08,2010-W07,2010-W06,2010-W05,2010-W04,2010-W03,2010-W02,2010-W01,2009-W53,2009-W52,2009-W51,2009-W50,2009-W49,2009-W48,2009-W47,2009-W46,2009-W45,2009-W44,2009-W43,2009-W42,2009-W41,2009-W40,2009-W39,2009-W38,2009-W37,2009-W36,2009-W35,2009-W34,2009-W33,2009-W32,2009-W31,2009-W30,2009-W29,2009-W28,2009-W27,2009-W26,2009-W25,2009-W24,2009-W23,2009-W22,2009-W21,2009-W20,2009-W19,2009-W18,2009-W17,2009-W16,2009-W15,2009-W14,2009-W13,2009-W12,2009-W11,2009-W10,2009-W09,2009-W08,2009-W07,2009-W06,2009-W05,2009-W04,2009-W03,2009-W02,2009-W01,2008-W52,2008-W51,2008-W50,2008-W49,2008-W48,2008-W47,2008-W46,2008-W45,2008-W44,2008-W43,2008-W42,2008-W41,2008-W40,2008-W39,2008-W38,2008-W37,2008-W36,2008-W35,2008-W34,2008-W33,2008-W32,2008-W31,2008-W30,2008-W29,2008-W28,2008-W27,2008-W26,2008-W25,2008-W24,2008-W23,2008-W22,2008-W21,2008-W20,2008-W19,2008-W18,2008-W17,2008-W16,2008-W15,2008-W14,2008-W13,2008-W12,2008-W11,2008-W10,2008-W09,2008-W08,2008-W07,2008-W06,2008-W05,2008-W04,2008-W03,2008-W02,2008-W01,2007-W52,2007-W51,2007-W50,2007-W49,2007-W48,2007-W47,2007-W46,2007-W45,2007-W44,2007-W43,2007-W42,2007-W41,2007-W40,2007-W39,2007-W38,2007-W37,2007-W36,2007-W35,2007-W34,2007-W33,2007-W32,2007-W31,2007-W30,2007-W29,2007-W28,2007-W27,2007-W26,2007-W25,2007-W24,2007-W23,2007-W22,2007-W21,2007-W20,2007-W19,2007-W18,2007-W17,2007-W16,2007-W15,2007-W14,2007-W13,2007-W12,2007-W11,2007-W10,2007-W09,2007-W08,2007-W07,2007-W06,2007-W05,2007-W04,2007-W03,2007-W02,2007-W01,2006-W52,2006-W51,2006-W50,2006-W49,2006-W48,2006-W47,2006-W46,2006-W45,2006-W44,2006-W43,2006-W42,2006-W41,2006-W40,2006-W39,2006-W38,2006-W37,2006-W36,2006-W35,2006-W34,2006-W33,2006-W32,2006-W31,2006-W30,2006-W29,2006-W28,2006-W27,2006-W26,2006-W25,2006-W24,2006-W23,2006-W22,2006-W21,2006-W20,2006-W19,2006-W18,2006-W17,2006-W16,2006-W15,2006-W14,2006-W13,2006-W12,2006-W11,2006-W10,2006-W09,2006-W08,2006-W07,2006-W06,2006-W05,2006-W04,2006-W03,2006-W02,2006-W01,2005-W52,2005-W51,2005-W50,2005-W49,2005-W48,2005-W47,2005-W46,2005-W45,2005-W44,2005-W43,2005-W42,2005-W41,2005-W40,2005-W39,2005-W38,2005-W37,2005-W36,2005-W35,2005-W34,2005-W33,2005-W32,2005-W31,2005-W30,2005-W29,2005-W28,2005-W27,2005-W26,2005-W25,2005-W24,2005-W23,2005-W22,2005-W21,2005-W20,2005-W19,2005-W18,2005-W17,2005-W16,2005-W15,2005-W14,2005-W13,2005-W12,2005-W11,2005-W10,2005-W09,2005-W08,2005-W07,2005-W06,2005-W05,2005-W04,2005-W03,2005-W02,2005-W01,2004-W53,2004-W52,2004-W51,2004-W50,2004-W49,2004-W48,2004-W47,2004-W46,2004-W45,2004-W44,2004-W43,2004-W42,2004-W41,2004-W40,2004-W39,2004-W38,2004-W37,2004-W36,2004-W35,2004-W34,2004-W33,2004-W32,2004-W31,2004-W30,2004-W29,2004-W28,2004-W27,2004-W26,2004-W25,2004-W24,2004-W23,2004-W22,2004-W21,2004-W20,2004-W19,2004-W18,2004-W17,2004-W16,2004-W15,2004-W14,2004-W13,2004-W12,2004-W11,2004-W10,2004-W09,2004-W08,2004-W07,2004-W06,2004-W05,2004-W04,2004-W03,2004-W02,2004-W01,2003-W52,2003-W51,2003-W50,2003-W49,2003-W48,2003-W47,2003-W46,2003-W45,2003-W44,2003-W43,2003-W42,2003-W41,2003-W40,2003-W39,2003-W38,2003-W37,2003-W36,2003-W35,2003-W34,2003-W33,2003-W32,2003-W31,2003-W30,2003-W29,2003-W28,2003-W27,2003-W26,2003-W25,2003-W24,2003-W23,2003-W22,2003-W21,2003-W20,2003-W19,2003-W18,2003-W17,2003-W16,2003-W15,2003-W14,2003-W13,2003-W12,2003-W11,2003-W10,2003-W09,2003-W08,2003-W07,2003-W06,2003-W05,2003-W04,2003-W03,2003-W02,2003-W01,2002-W52,2002-W51,2002-W50,2002-W49,2002-W48,2002-W47,2002-W46,2002-W45,2002-W44,2002-W43,2002-W42,2002-W41,2002-W40,2002-W39,2002-W38,2002-W37,2002-W36,2002-W35,2002-W34,2002-W33,2002-W32,2002-W31,2002-W30,2002-W29,2002-W28,2002-W27,2002-W26,2002-W25,2002-W24,2002-W23,2002-W22,2002-W21,2002-W20,2002-W19,2002-W18,2002-W17,2002-W16,2002-W15,2002-W14,2002-W13,2002-W12,2002-W11,2002-W10,2002-W09,2002-W08,2002-W07,2002-W06,2002-W05,2002-W04,2002-W03,2002-W02,2002-W01,2001-W52,2001-W51,2001-W50,2001-W49,2001-W48,2001-W47,2001-W46,2001-W45,2001-W44,2001-W43,2001-W42,2001-W41,2001-W40,2001-W39,2001-W38,2001-W37,2001-W36,2001-W35,2001-W34,2001-W33,2001-W32,2001-W31,2001-W30,2001-W29,2001-W28,2001-W27,2001-W26,2001-W25,2001-W24,2001-W23,2001-W22,2001-W21,2001-W20,2001-W19,2001-W18,2001-W17,2001-W16,2001-W15,2001-W14,2001-W13,2001-W12,2001-W11,2001-W10,2001-W09,2001-W08,2001-W07,2001-W06,2001-W05,2001-W04,2001-W03,2001-W02,2001-W01,2000-W52,2000-W51,2000-W50,2000-W49,2000-W48,2000-W47,2000-W46,2000-W45,2000-W44,2000-W43,2000-W42,2000-W41,2000-W40,2000-W39,2000-W38,2000-W37,2000-W36,2000-W35,2000-W34,2000-W33,2000-W32,2000-W31,2000-W30,2000-W29,2000-W28,2000-W27,2000-W26,2000-W25,2000-W24,2000-W23,2000-W22,2000-W21,2000-W20,2000-W19,2000-W18,2000-W17,2000-W16,2000-W15,2000-W14,2000-W13,2000-W12,2000-W11,2000-W10,2000-W09,2000-W08,2000-W07,2000-W06,2000-W05,2000-W04,2000-W03,2000-W02,2000-W01&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=both
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
   
    df_=df_[df_["geo"] == country]

    df_["age_sex"] = df_["age"] + "_" +df_["sex"]

    
    # Function to extract age_low and age_high based on patterns
    def extract_age_ranges(age):
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
    df_["weeknr"] = (df_["TIME_PERIOD"].str[6:]).astype(int)


    def add_custom_age_group_deaths(df_, min_age, max_age):
        # Filter the data based on the dynamic age range
        df_filtered = df_[(df_['age_low'] >= min_age) & (df_['age_high'] <= max_age)]

        # Group by TIME_PERIOD (week), sex, and sum the deaths (OBS_VALUE)
        totals = df_filtered.groupby(['TIME_PERIOD', 'sex'], observed=False)['OBS_VALUE'].sum().reset_index()

        # Assign a new label for the age group (dynamic)
        totals['age'] = f'Y{min_age}-{max_age}'
        totals["age_sex"] = totals["age"] + "_" +totals["sex"]
        totals["jaar"] = (totals["TIME_PERIOD"].str[:4]).astype(int)
        return totals
    
    for i in opdeling:
        custom_age_group = add_custom_age_group_deaths(df_, i[0], i[1])
        df_ = pd.concat([df_, custom_age_group], ignore_index=True)


    df_bevolking = get_bevolking(country, opdeling)
   
    summed_per_year = df_.groupby(["jaar", 'age_sex'])['OBS_VALUE'].sum() # .reset_index()
  
    df__ = pd.merge(summed_per_year, df_bevolking, on=['jaar', 'age_sex'], how='outer')
    
    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["OBS_VALUE"].notna()]
  
    df__["per100k"] = round(df__["OBS_VALUE"]/df__["aantal"]*100000,1)
    
    return df__

def plot(df, category, value_field, countries):
    if value_field == "percentage":
        value_field_ ="per100k"
    else:
        value_field_=value_field
    # Filter the data
    df_before_2020 = df[df["jaar"] < 2020]
    df_2020_and_up = df[df["jaar"] >= 2020]
    
   
    trendline_info = ""  # Initialize a string to store trendline info
    for country in countries:
        if country == "BE":
            color_before_2020 = '#B22222'  # Dark Red
            color_2020_and_up = '#DC143C'  # Crimson
            trendline_color = '#FFA07A'    # Light Salmon
        elif country == "NL":
            color_before_2020 = '#00008B'  # Dark Blue
            color_2020_and_up = '#1E90FF'  # Dodger Blue
            trendline_color = '#87CEFA'    # Light Sky Blue
    
        df_country_before_2020 = df_before_2020[df_before_2020["country"] == country]
        df_country_2020_and_up = df_2020_and_up[df_2020_and_up["country"] == country]
        
        sd = df_country_before_2020[value_field_].std()
        
        
        # Calculate the trendline for each country before 2020
        X = df_country_before_2020["jaar"]
        y = df_country_before_2020[value_field_]
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        # Define the extended range of years
        extended_years = np.arange(df_country_before_2020["jaar"].min(), 2025)

        try:
            model = sm.OLS(y, X).fit()
            trendline = model.predict(X)
            
            # Add the trendline for the original years to the DataFrame
            if value_field == 'OBS_VALUE':
                df_country_before_2020['predicted_deaths'] = trendline
            else:
                df_country_before_2020['predicted_per100k'] = trendline
            
            # Create a DataFrame for the extended years
            extended_X = sm.add_constant(extended_years)
            
              # Predict the trendline and bounds for the extended years
            trendline_extended = model.predict(extended_X)
            upper_bound_extended = trendline_extended + 2 * sd
            lower_bound_extended = trendline_extended - 2 * sd
             # Calculate the upper and lower bounds of the shaded area
            upper_bound = trendline + 2 * sd
            lower_bound = trendline - 2 * sd

           
            # Calculate R² value
            r2 = r2_score(y, trendline)
            # trendline_info += f"{country}\nTrendline formula: y = {model.params[1]:.4f}x + {model.params[0]:.4f}\nR² value: {r2:.4f}\n\n"

            # Adjusted code with .iloc for position-based access
            trendline_info += f"{country}\nTrendline formula: y = {model.params.iloc[1]:.4f}x + {model.params.iloc[0]:.4f}\nR² value: {r2:.4f}\n\n"
        
            # # Print the formula and R² value
            # st.write(f"Trendline formula: y = {model.params[1]:.4f}x + {model.params[0]:.4f}")
            # st.write(f"R² value: {r2:.4f}")

                       
        except:
            pass
       
        if value_field == 'OBS_VALUE':
            df_extended = pd.merge(df_country_2020_and_up, pd.DataFrame({
                    'jaar': extended_years,
                    'predicted_deaths': trendline_extended
                    }), on='jaar')
        else:
            df_extended = pd.merge(df_country_2020_and_up, pd.DataFrame({
                    'jaar': extended_years,
                    'predicted_per100k': trendline_extended
                    }), on='jaar')

        # Concatenate the original and extended DataFrames
        df_diff = pd.concat([df_country_before_2020, df_extended], ignore_index=True)
        
        # Optionally, sort by year
        df_diff = df_diff.sort_values(by='jaar').reset_index(drop=True)
        if value_field_ == 'per100k':
            df_diff['predicted_deaths'] = df_diff['predicted_per100k']*df_diff['aantal']/100000

        df_diff['oversterfte'] = round(df_diff['OBS_VALUE'] - df_diff['predicted_deaths']) 
        df_diff['aantal']=round(df_diff['aantal'])
        df_diff['percentage'] = round(((df_diff['OBS_VALUE'] - df_diff['predicted_deaths'])/df_diff['predicted_deaths'])*100,1)
        df_diff = df_diff[['jaar', 'aantal', 'per100k', 'oversterfte', 'percentage']]
        
         # Create the scatter plot with Plotly Express for values before 2020
        fig = go.Figure()
        if value_field == 'percentage':
             # Plot before 2020
            # Filter the data for the current country
            df_country_before_2020 = df_diff[df_diff['jaar']<2020]
            df_country_2020_and_up = df_diff[df_diff['jaar']>=2020]

            # Plot bars before 2020
            fig.add_trace(go.Bar(
                x=df_country_before_2020["jaar"],
                y=df_country_before_2020[value_field],
                name=f'{country} - Before 2020',
                marker=dict(color=color_before_2020)
            ))

            # Plot bars for 2020 and up
            fig.add_trace(go.Bar(
                x=df_country_2020_and_up["jaar"],
                y=df_country_2020_and_up[value_field],
                name=f'{country} - 2020 and up',
                marker=dict(color='red')  # Set the color to red for years >= 2020
            ))
            # # Plot 2020 and up
            # fig.add_trace(go.Scatter(x=df_country_2020_and_up["jaar"], y=df_country_2020_and_up[value_field], 
            #                         mode='markers', name=f'{country} - 2020 and up', marker=dict(color=color_2020_and_up)))
           
        else:
            # Plot before 2020
            fig.add_trace(go.Scatter(x=df_country_before_2020["jaar"], y=df_country_before_2020[value_field], 
                                    mode='markers', name=f'{country} - Before 2020', marker=dict(color=color_before_2020)))

            # Plot 2020 and up
            fig.add_trace(go.Scatter(x=df_country_2020_and_up["jaar"], y=df_country_2020_and_up[value_field], 
                                    mode='markers', name=f'{country} - 2020 and up', marker=dict(color=color_2020_and_up)))
            

            # Add the shaded area to the plot
            fig.add_trace(go.Scatter(
                x=df_country_before_2020["jaar"].tolist() + df_country_before_2020["jaar"].tolist()[::-1],
                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.3)',  # Adjust the color and opacity as needed
                line=dict(color='rgba(255,255,255,0)'),  # Invisible line
                name=f'±2 SD {country} till 2019'
            ))
            # Add the trendline to the plot
            fig.add_trace(go.Scatter(x=df_country_before_2020["jaar"], y=trendline, 
                                    mode='lines', name=f'Trendline {country} till 2019', line=dict(color=trendline_color)))
                # Add the shaded area to the plot
            fig.add_trace(go.Scatter(
                x=np.concatenate([extended_years, extended_years[::-1]]),
                y=np.concatenate([upper_bound_extended, lower_bound_extended[::-1]]),
                fill='toself',  # Corrected fill mode
                fillcolor='rgba(128, 128, 128, 0.3)',  # Adjust the color and opacity as needed
                line=dict(color='rgba(255,255,255,0)'),  # Invisible line
                name=f'±2 SD {country} until 2024'
            ))

            # Add the trendline to the plot
            fig.add_trace(go.Scatter(
                x=extended_years,
                y=trendline_extended,
                mode='lines',
                name=f'Trendline {country} until 2024',
                line=dict(color=trendline_color)
            ))
        
        fig.update_layout(
                title=category,
                xaxis_title="Year",
                yaxis_title=value_field,
            )
        # Show the plot
    st.plotly_chart(fig)
    with st.expander(f"Trendline/oversterfte info - {category}"):
        st.write(trendline_info)
        #if value_field == 'OBS_VALUE':
        st.write(df_diff) 

def plot_wrapper(df, t2, value_field, countries):
    df_ = df[df["age_sex"] == t2]
    if len(df_) > 0:
        plot(df_, t2, value_field, countries)
    else:
        st.info(f"No data - {t2}")

   
def interface_opdeling():
    def ends_in_4_9_or_120(number):
    # Check if the number ends in 4 or 9, or is exactly 120
        return number % 10 in {4, 9} or number == 120

    def ends_in_5_0_or_120(number):
    # Check if the number ends in 4 or 9, or is exactly 120
        return number % 10 in {5, 0} or number == 120
    # Get data for all selected countries and concatenate them
    
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        b1,b2 = st.columns(2)
        with b1:
            l1 = st.number_input("low1", 0,120,0)
        with b2:
            h1 = st.number_input("high1", 0,120,14)
    with col2:
        c1,c2 = st.columns(2)
        with c1:
            l2 = st.number_input("low2", 0,120,15)
        with c2:
            h2 = st.number_input("high2", 0,120,64)
    with col3:
        d1,d2 = st.columns(2)
        with d1:
            l3 = st.number_input("low3", 0,120,65)
        with d2:
            h3 = st.number_input("high3", 0,120,79)
    with col4:
        e1,e2 = st.columns(2)
        with e1:
            l4 = st.number_input("low4", 0,120,80)
        with e2:
            h4 = st.number_input("high4", 0,120,120)

    
    fout = False
    for l in [l1,l2,l3,l4]:
        if not ends_in_5_0_or_120(l):
            st.error(f"low number **{l}** is not compatible")
            fout = True
    for h in [h1,h2,h3,h4]:
        if not ends_in_4_9_or_120(h):
            st.error(f"high number **{h}** is not compatible")
            fout = True
    if fout:
        st.info("Please correct values")
        st.stop()

    opdeling = [[l1,h1],[l2,h2],[l3,h3],[l4,h4]]
    return opdeling

def main():
    st.title("Deaths in age groups ")
    
    # Let the user select one or both countries
    countries = ["NL"] # st.multiselect("Country [NL | BE]", ["NL", "BE"], default=["NL", "BE"])
 
    opdeling = interface_opdeling()

    df_list = []
    for country in countries:
        df = get_sterfte(opdeling, country)
        df["country"] = country  # Add a column to distinguish the countries
        df_list.append(df)
    
    df_combined = pd.concat(df_list)
    
    # Plot the data for both countries
    to_do = unique_values = df_combined["age_sex"].unique()
    #labels = ['TOTAL']+["Y0-19"]+["Y20-64"]+["Y65-79"]+["Y80-120"] + ['Y_LT5'] + [f'Y{i}-{i+4}' for i in range(5, 90, 5)] + ['Y_GE90']
    labels = ['TOTAL'] + [f'Y{start}-{end}' for start, end in opdeling] + ['Y_LT5'] + [f'Y{i}-{i+4}' for i in range(5, 90, 5)] + ['Y_GE90']
  
    colx, coly = st.columns(2)
    with colx:
        value_field = st.selectbox("Value field (per 100.000 | absolute value| percentage (based on per 100k))", ["per100k", "OBS_VALUE","percentage"], 0)
    with coly:
        how = st.selectbox("How (all from one year | compare startyears)", ["all from one year", "compare startyears"], 1)
    
    if how == "all from one year":
        start = st.number_input("Startjaar", 2000, 2020, 2000)
        df_combined = df_combined[df_combined["jaar"] >= start]
        
        for t in labels:
            col1, col2, col3 = st.columns(3)
            with col1:
                t2 = f"{t}_T"
                plot_wrapper(df_combined, t2, value_field, countries)
            with col2:
                t2 = f"{t}_M"
                plot_wrapper(df_combined, t2, value_field, countries)
            with col3:
                t2 = f"{t}_F"
                plot_wrapper(df_combined, t2, value_field, countries)
    else:
        y = st.selectbox("Which category (T=all, M=Male, F=Female)", ["T", "M", "F"], 0)
        for x in labels:
            col1, col2, col3 = st.columns(3)
            t2 = f"{x}_{y}"
            with col1:
                df_ = df_combined[df_combined["jaar"] >= 2000]
                plot_wrapper(df_, t2, value_field, countries)
            with col2:
                df_ = df_combined[df_combined["jaar"] >= 2010]
                plot_wrapper(df_, t2, value_field, countries)
            with col3:
                df_ = df_combined[df_combined["jaar"] >= 2015]
                plot_wrapper(df_, t2, value_field, countries)


    st.subheader("Databronnen")
    st.info("Bevolkingsgrootte NL: https://opendata.cbs.nl/#/CBS/nl/dataset/03759ned/table?dl=39E0B")
    st.info("Bevolkingsgrootte BE:https://ec.europa.eu/eurostat/databrowser/view/demo_pjan__custom_12780094/default/table?lang=en")
    st.info("Sterfte: https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")
    st.info("See also https://www.mortality.watch/explorer/?c=NLD&t=cmr&e=1&df=2010&dt=2023&ag=all&ce=0&st=1&pi=0&p=1")


if __name__ == "__main__":
    print ("gooo")
    main()
