# EXPORT FROM THE BASEVALUES AND OBSERVED VALUES
# Used in correlatie_sterfte_rioolwater_vaccins.py


# Imitating RIVM overstefte grafieken
# overlijdens per week: https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1655808656887

# Het verwachte aantal overledenen wanneer er geen COVID-19-epidemie was geweest, is
# geschat op basis van de waargenomen sterfte in 2015–2019. Eerst wordt voor elk jaar de
# sterfte per week bepaald. Vervolgens wordt per week de gemiddelde sterfte in die week
# en de zes omliggende weken bepaald. Deze gemiddelde sterfte per week levert een
# benadering van de verwachte wekelijkse sterfte. Er is dan nog geen rekening gehouden
# met de trendmatige vergrijzing van de bevolking. Daarom is de sterfte per week nog
# herschaald naar de verwachte totale sterfte voor het jaar. Voor 2020 is de verwachte sterfte
# 153 402 en voor 2021 is deze 154 887. Het aantal voor 2020 is ontleend aan de
# Kernprognose 2019–2060 en het aantal voor 2021 aan de Bevolkingsprognose 2020–2070
# CBS en RIVM | Sterfte en oversterfte in 2020 en 2021 | Juni 2022 15
# (exclusief de aanname van extra sterfgevallen door de corona-epidemie). De marges rond
# de verwachte sterfte zijn geschat op basis van de waargenomen spreiding in de sterfte per
# week in de periode 2015–2019. Dit 95%-interval geeft de bandbreedte weer van de
# gewoonlijk fluctuaties in de sterfte. 95 procent van de sterfte die in eerdere jaren is
# waargenomen, valt in dit interval. Er wordt van oversterfte gesproken wanneer de sterfte
# boven de bovengrens van dit interval ligt.

# TO DO: https://towardsdatascience.com/using-eurostat-statistical-data-on-europe-with-python-2d77c9b7b02b

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import platform
import plotly.express as px
import datetime
import eurostat

def get_sterfte(country):
    """_summary_

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
    do_local = True
    if do_local:
        st.warning("STATIC DATA dd 23/06/2024")
        if platform.processor() != "":
            file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_NL.csv"
        
        else:
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_NL.csv"
                  
        #file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_new.csv"
        df_ = pd.read_csv(
            file,
            delimiter=",",
            low_memory=False,
            )  
     
    else:
        try:
            df_ = get_data_eurostat()
            
        except:
            st.warning("STATIC DATA dd 23/06/2024")
            if platform.processor() != "":
                file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_new.csv"
            
            else:
                file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_new.csv"
                
            df_ = pd.read_csv(
                file,
                delimiter=",",
                low_memory=False,
                )

    
    df_=df_[df_["geo"] == country]
    
    df_["age_sex"] = df_["age"] + "_" +df_["sex"]
    df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["weeknr"] = (df_["TIME_PERIOD"].str[6:]).astype(int)
    
    df_bevolking = get_bevolking()
    
    df__= df_.merge(df_bevolking, on="age_sex", how="outer")
    df__["per100k"] = df__["OBS_VALUE"] / df__["aantal"]
    
    df__.columns = df__.columns.str.replace('jaar_x', 'jaar', regex=False)
    #df__.to_csv(r"C:\Users\rcxsm\Documents\endresult.csv")
    return df__

def get_bevolking():
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_NL.csv"
    else: 
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
    data = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
  
    data['leeftijd'] = data['leeftijd'].astype(int)
    #st.write(data)



    # Define age bins and labels
    bins = list(range(0, 95, 5)) + [1000]  # [0, 5, 10, ..., 90, 1000]
    labels = [f'Y{i}-{i+4}' for i in range(0, 90, 5)] + ['90-120']


    # Create a new column for age bins
    data['age_group'] = pd.cut(data['leeftijd'], bins=bins, labels=labels, right=False)


    # Group by year, gender, and age_group and sum the counts
    grouped_data = data.groupby(['jaar', 'geslacht', 'age_group'])['aantal'].sum().reset_index()

    # Save the resulting data to a new CSV file
    # grouped_data.to_csv('grouped_population_by_age_2010_2024.csv', index=False, sep=';')

    # print("Grouping complete and saved to grouped_population_by_age_2010_2024.csv")
    grouped_data["age_sex"] = grouped_data['age_group'].astype(str) +"_"+grouped_data['geslacht'].astype(str)
    
    
    for s in ["M", "F", "T"]:
        grouped_data.replace(f'0-4_{s}', f'Y_LT5_{s}', inplace=True)
        grouped_data.replace(f'90-120_{s}',f'Y_GE90_{s}', inplace=True)
    #st.write(grouped_data)
    # grouped_data.to_csv(r"C:\Users\rcxsm\Documents\per5jaar.csv")
    
    return grouped_data


def get_data_for_series(df_, seriename, vanaf_jaar):
  
    if seriename == "TOTAL_T":
       # df = df_[["jaar","weeknr","aantal_dgn", seriename]].copy(deep=True)
        df = df_[["jaar","weeknr", seriename]].copy(deep=True)

    else:
       # df = df_[["jaar","weeknr","aantal_dgn","totaal_m_v_0_120", seriename]].copy(deep=True)
        df = df_[["jaar","weeknr","TOTAL_T", seriename]].copy(deep=True)
   
    df = df[ (df["jaar"] >= vanaf_jaar)] 
   
    df = df.sort_values(by=['jaar','weeknr']).reset_index()
 
    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_120","totaal_m_0_120","totaal_v_0_120","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_120","totaal_m_80_120","totaal_v_80_120"]
    som_2015_2019 = 0
    som =  149832
    noemer = 149832
    for y in range (int(vanaf_jaar),2020):
        # df_year = df[(df["jaar"] == y)]
        # som = df_year["TOTAL_T"].sum()

        # som = df_year["TOTAL_T"].sum()
        # som_2015_2019 +=som

        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        # https://www.cbs.nl/nl-nl/nieuws/2023/05/eind-januari-geen-oversterfte-meer/oversterfte-en-verwachte-sterfte
       

        factors = {
            2014: 1,
            2015: 1,
            2016: 1,
            2017: 1,
            2018: 1,
            2019: 1,
            2020: 153402 / noemer,
            2021: 154887 / noemer,
            2022: 155494 / noemer,
            2023: 156666 / noemer,  # or 169333 / som if you decide to use the updated factor
            2024: 157846 / noemer,
        }
        for year in range(2014, 2025):
            new_column_name = f"{seriename}_factor_{year}"
            factor = factors[year]
            # factor=1
            df[new_column_name] = df[seriename] * factor
    return df

def plot_graph_oversterfte(how, df, df_corona, df_boosters, df_herhaalprik, df_rioolwater, series_name, rightax, mergetype, show_scatter, sma, sma_center):
    """_summary_

    Args:
        how (_type_): _description_
        df (_type_): _description_
        df_corona (_type_): _description_
        df_boosters (_type_): _description_
        df_herhaalprik (_type_): _description_
        series_name (_type_): _description_
        rightax (_type_): _description_
        mergetype (_type_): _description_
    """
    booster_cat = ["m_v_0_120","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_120"]
   
    df_oversterfte = pd.merge(df, df_corona, left_on = "weeknr", right_on="weeknr", how = "outer")
    
    what_to_sma_ = ["low05", "high95"]
    for what_to_sma in what_to_sma_:
        # sma = 6 is also in de procedure of CBS
        df_oversterfte[what_to_sma] = df_oversterfte[what_to_sma].rolling(window=6, center=True).mean()
   
    df_oversterfte["over_onder_sterfte"] =  0
    df_oversterfte["meer_minder_sterfte"] =  0
    
    df_oversterfte["year_minus_high95"] = df_oversterfte[series_name] - df_oversterfte["high95"]
    df_oversterfte["year_minus_avg"] = df_oversterfte[series_name]- df_oversterfte["avg"]
    df_oversterfte["year_minus_avg_only_pos"] =  df_oversterfte["year_minus_avg"] 
    df_oversterfte["year_minus_avg_cumm"] = df_oversterfte["year_minus_avg_only_pos"].cumsum()
    df_oversterfte["p_score"] = ( df_oversterfte[series_name]- df_oversterfte["avg"]) /   df_oversterfte["avg"]
    
    affected_columns = [
        "year_minus_high95",
        "year_minus_avg",
        "year_minus_avg_only_pos",
        "year_minus_avg_cumm",
        "p_score"
    ]
    for a in affected_columns:
        df_oversterfte[a] = df_oversterfte[a].rolling(window=sma, center=sma_center).mean()
    
    df_oversterfte.year_minus_avg_only_pos=df_oversterfte.year_minus_avg_only_pos.mask(df_oversterfte.year_minus_avg_only_pos.lt(0),0)
    
    for i in range( len (df_oversterfte)):
        if df_oversterfte.loc[i,series_name ] >  df_oversterfte.loc[i,"high95"] :
            df_oversterfte.loc[i,"over_onder_sterfte" ] =  df_oversterfte.loc[i,series_name ]  -  df_oversterfte.loc[i,"avg"] #["high95"]
            df_oversterfte.loc[i,"meer_minder_sterfte" ] =  df_oversterfte.loc[i,series_name ]  -  df_oversterfte.loc[i,"high95"] 
        elif df_oversterfte.loc[i,series_name ] <  df_oversterfte.loc[i,"low05"]:
            df_oversterfte.loc[i,"over_onder_sterfte" ] =     df_oversterfte.loc[i,series_name ] - df_oversterfte.loc[i,"avg"] #["low05"]
            df_oversterfte.loc[i,"meer_minder_sterfte" ] =     df_oversterfte.loc[i,series_name ] - df_oversterfte.loc[i,"low05"]
    name_ = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\oversterfte"+series_name+".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df_oversterfte.to_csv(name_, index=False, compression=compression_opts)
    print("--- Saving " + name_ + " ---")
    
   
def plot( how, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center, country):
    """_summary_

    Args:
        series_names (_type_): _description_
        how (_type_): ["quantiles", "Lines", "over_onder_sterfte", "meer_minder_sterfte","year_minus_avg","year_minus_avg_cumm", "p_score"]
        yaxis_to_zero (_type_): _description_
        rightax (_type_): _description_
        mergetype (_type_): _description_
    """    
    df_ = get_data(country)
   
    #series_names  = df_['age_sex'].drop_duplicates().sort_values()

    series_names_ = df_.columns.tolist()
    series_names_ = series_names_[3:]
    series_names = st.sidebar.multiselect("Which ages to show", series_names_, ["TOTAL_T", "Y50-54_M"])
 
    series_to_show = [  "Y0_49_T",
                        "Y_LT5_T",
                        "Y5-9_T",
                        "Y10-14_T",
                        "Y15-19_T",
                        "Y20-24_T",
                        "Y25-29_T",
                        "Y30-34_T",
                        "Y35-39_T",
                        "Y40-44_T",
                        "Y45-49_T",
                        "Y25_49_T",
                        "Y25-29_T",
                        "Y30-34_T",
                        "Y35-39_T",
                        "Y40-44_T",
                        "Y45-49_T",
                        "Y50_64_T",
                        "Y50-54_T",
                        "Y55-59_T",
                        "Y60-64_T",
                        "Y65_79_T",
                        "Y65-69_T",
                        "Y70-74_T",
                        "Y75-79_T",
                        "Y50_59_T",
                        "Y50-54_T",
                        "Y55-59_T",
                        "Y60_69_T",
                        "Y60-64_T",
                        "Y65-69_T",
                        "Y70_79_T",
                        "Y70-74_T",
                        "Y75-79_T",
                        "Y80_89_T",
                        "Y80-84_T",
                        "Y85-89_T",
                        "Y_GE90_T",
                        "Y0_120_T",
                        "Y0_49_T",
                        "Y50_64_T",
                        "Y65_79_T",
                        "Y80_120_T",
                        "Y80_89_T",
                         "Y90_120_T",
]

    df_endresult = pd.DataFrame()
    for col, series_name in enumerate(series_to_show):
        
        st.write(f"{col+1} / {len(series_to_show)} - {series_name}")
        if how =="quantiles":
            df_data, df_corona, df_quantile = make_df_data_corona_quantile(vanaf_jaar, df_, series_name)
           
            columnlist = ["q05","q25","q50","avg","q75","q95", "low05", "high95"]
            for what_to_sma in columnlist:
                df_quantile[what_to_sma] = df_quantile[what_to_sma].rolling(window=6, center=sma_center).mean()
            df_corona = df_corona[df_corona["weeknr"] != "2015_53"]
            
            df_quantile = df_quantile.sort_values(by=['jaar','week_'])

            # # Number of rows to drop
            # n = 52-18
            

            # # Dropping last n rows using drop
            # df_quantile.drop(df_quantile.tail(n),inplace = True)
            st.write("x282")
            # st.write(df_quantile)
            # st.write(df_corona)
            # st.write(df_data)
            df_quantile = df_quantile[[ "weeknr","seriesname", "avg"]] 
            df_data["weeknr"] = df_data["jaar"].astype(int).astype(str)+"_"+ df_data["weeknr"].astype(int).astype(str).str.zfill(2)
            df_data["OBS_VALUE_"] =  df_data[series_name]
            df_data = df_data[["weeknr", "OBS_VALUE_"]]

            df_result = pd.merge(df_data,df_quantile, on="weeknr", how="outer")
            st.write(df_result)
            df_endresult=pd.concat([df_endresult,df_result], axis=0)
    st.write(df_endresult)
    df_endresult.to_csv('sterfte_avg.csv', index=False, sep=',')

def make_df_data_corona_quantile(vanaf_jaar, df_, series_name):
   
    st.subheader(series_name)
    df_data = get_data_for_series(df_, series_name, vanaf_jaar).copy(deep=True)
    df_corona, df_quantile = make_df_qantile(series_name, df_data)
    return df_data,df_corona,df_quantile
            
#@st.cache_data
def get_data(country):
    
    df__ = get_sterfte(country)
  
    df__ = df__[df__['age'] !="UNK"]
    df__["age_group"] = df__["age_group"].astype(str)
   
    df__ = df__.fillna(0)
    df__ = df__[df__['OBS_VALUE'] !=None]
    value_to_do = st.sidebar.selectbox("Value to do [OBS_VALUE | per 100k]", ["OBS_VALUE", "per100k"], 0)
    #value_to_do = "OBS_VALUE"
    #value_to_do = "per100k"
    df__["jaar_week"] = df__["jaar"].astype(int).astype(str)  +"_" + df__["weeknr"].astype(int).astype(str).str.zfill(2)
   
    try:
        df_ = df__.pivot(index=["jaar_week", "jaar", "weeknr"], columns='age_sex', values=value_to_do).reset_index()
    except:
                # Aggregating duplicate entries by summing the 'OBS_VALUE'
        df_aggregated = df__.groupby(["jaar_week", "jaar", "weeknr", 'age_sex'])[value_to_do].sum().reset_index()

        # Pivot the aggregated dataframe
        df_ = df_aggregated.pivot(index=["jaar_week", "jaar", "weeknr"], columns='age_sex', values=value_to_do).reset_index()

    #df_ = df__.pivot(index="jaar_week", columns='age_sex', values='OBS_VALUE').reset_index()
    
    df_["m_v_0_49"] = df_["Y_LT5_T"] + df_["Y5-9_T"] + df_["Y10-14_T"]+ df_["Y15-19_T"]+ df_["Y20-24_T"] +  df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]
    df_["m_v_25_49"] = df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]

    opdeling = [[0,120],[15,17],[18,24], [25,49],[50,59],[60,69],[70,79],[80,120]]

    df_["m_v_50_64"] = df_["Y50-54_T"]+ df_["Y55-59_T"] + df_["Y60-64_T"]
    df_["m_v_65_79"] =+ df_["Y65-69_T"]+ df_["Y70-74_T"] + df_["Y75-79_T"]
    df_["m_v_50_59"] = df_["Y50-54_T"] + df_["Y55-59_T"]
    df_["m_v_60_69"] = df_["Y60-64_T"] + df_["Y65-69_T"]
    df_["m_v_70_79"] = df_["Y70-74_T"] + df_["Y75-79_T"]
    df_["m_v_80_89"] = df_["Y80-84_T"] + df_["Y85-89_T"]
    df_["m_v_90_120"] = df_["Y_GE90_T"]
    df_["m_v_0_120"] = df_["m_v_0_49"] + df_["m_v_50_64"] + df_["m_v_65_79"]  + df_["m_v_80_89"]+  df_["m_v_90_120"] 
    df_["m_v_80_120"] = df_["m_v_80_89"] + df_["m_v_90_120"]
    
    df_["Y0_49_T"] = df_["Y_LT5_T"] + df_["Y5-9_T"] + df_["Y10-14_T"]+ df_["Y15-19_T"]+ df_["Y20-24_T"] +  df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]
    df_["Y25_49_T"] = df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]

    df_["Y50_64_T"] = df_["Y50-54_T"]+ df_["Y55-59_T"] + df_["Y60-64_T"]
    df_["Y65_79_T"] =+ df_["Y65-69_T"]+ df_["Y70-74_T"] + df_["Y75-79_T"]
    df_["Y50_59_T"] = df_["Y50-54_T"] + df_["Y55-59_T"]
    df_["Y60_69_T"] = df_["Y60-64_T"] + df_["Y65-69_T"]
    df_["Y70_79_T"] = df_["Y70-74_T"] + df_["Y75-79_T"]
    df_["Y80_89_T"] = df_["Y80-84_T"] + df_["Y85-89_T"]
    df_["Y90_120_T"] = df_["Y_GE90_T"]
    df_["Y0_120_T"] = df_["Y0_49_T"] + df_["Y50_64_T"] + df_["Y65_79_T"] +  df_["Y80_89_T"] + df_["Y90_120_T"]
    df_["Y80_120_T"] = df_["Y80_89_T"] + df_["Y90_120_T"]
    
    return df_

def make_df_qantile(series_name, df_data):
    """_summary_

    Args:
        series_name (_type_): _description_
        df_data (_type_): _description_

    Returns:
        _type_: _description_
    """    
   
    df_corona = df_data[df_data["jaar"].between(2015, 2025)]
    df_corona["weeknr"] = df_corona["jaar"].astype(int).astype(str) +"_" + df_corona["weeknr"].astype(int).astype(str).str.zfill(2)
  
    # List to store individual quantile DataFrames
    df_quantiles = []

    # Loop through the years 2014 to 2024
    for year in range(2015, 2025):
        df_quantile_year = make_df_quantile_year(series_name, df_data, year)
        df_quantiles.append(df_quantile_year)

    # Concatenate all quantile DataFrames into a single DataFrame
    df_quantile = pd.concat(df_quantiles, axis=0)
    df_quantile["seriesname"] = series_name
    df_quantile["weeknr"] = (
        df_quantile["jaar"].astype(int).astype(str)
        + "_"
        + df_quantile["week_"].astype(int).astype(str).str.zfill(2)
    )

    #df = pd.merge(df_corona, df_quantile, on="weeknr")
    return df_corona, df_quantile

def make_row_df_quantile(series_name, year, df_to_use, w_):
    """Calculate the percentiles of a certain week

    Args:
        series_name (_type_): _description_
        year (_type_): _description_
        df_to_use (_type_): _description_
        w_ (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if w_ == 53:
        w = 52
    else:
        w = w_
    df_to_use_ = df_to_use[(df_to_use["weeknr"] == w)].copy(deep=True)
    column_to_use = series_name +  "_factor_" + str(year)
    data = df_to_use_[column_to_use ] #.tolist()
               
    q05 = np.percentile(data, 5)
    q25 = np.percentile(data, 25)
    q50 = np.percentile(data, 50)
    q75 = np.percentile(data, 75)
    q95 = np.percentile(data, 95)
               
    avg = round(data.mean(),0)
    sd = round(data.std(),0)
    low05 = round(avg - (2*sd),0)
    high95 = round(avg +(2*sd),0)
    
    df_quantile_ =  pd.DataFrame(
                   [ {
                        "week_": w_,
                        "jaar":year,
                        "q05": q05,
                        "q25": q25,
                        "q50": q50,
                        "avg": avg,
                        "q75": q75,
                        "q95": q95,
                        "low05":low05,
                        "high95":high95,
                       
                        }]
                )
            
    return df_quantile_

def make_df_quantile_year(series_name, df_data, year):

    """ Calculate the quantiles

    Returns:
        _type_: _description_
    """    
    #df_to_use = df_data[(df_data["jaar"] !=2020) & (df_data["jaar"] !=2021) & (df_data["jaar"] !=2022) & (df_data["jaar"] !=2023) & (df_data["jaar"] !=2024)].copy(deep=True)
    df_to_use = df_data[(df_data["jaar"] >= 2015) & (df_data["jaar"] < 2020)].copy(
            deep=True
        )
    df_quantile =None
           
    week_list = df_to_use['weeknr'].unique().tolist()
            # week_list = week_list.sort()
          
            
            #for w in week_list:  #puts week 1 at the end
    for w in range(1,53):
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, w)
        df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)
    if year==2020:
        #2020 has a week 53
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, 53)
        df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)

    
    return df_quantile
        


def footer(vanaf_jaar):
    st.write("Voor de correctiefactor voor 2020, 2021 en 2022 is uitgegaan van de factor over de gehele populatie. *")
    st.write(f"Het 95%-interval is berekend aan de hand van het gemiddelde en standaarddeviatie (z=2)  over de waardes per week van {vanaf_jaar} t/m 2019")
    # st.write("Week 53 van 2020 heeft een verwachte waarde en 95% interval van week 52")
    #st.write("Enkele andere gedeeltelijke weken zijn samengevoegd conform het CBS bestand")
    st.write("Bron data: Eurostats https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en")
    st.write("Code: https://github.com/rcsmit/COVIDcases/blob/main/oversterfte_eurostats.py")
    st.write("P score = (verschil - gemiddelde) / gemiddelde, gesmooth over 6 weken")
    st.write()
    st.write("*. https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493")

def interface():
    how = st.sidebar.selectbox("How", ["quantiles", "Lines", "over_onder_sterfte", "meer_minder_sterfte","year_minus_avg","year_minus_avg_cumm", "p_score"], index = 0)
    yaxis_to_zero = st.sidebar.selectbox("Y as beginnen bij 0", [False, True], index = 0)
    if (how == "year_minus_avg") or (how=="year_minus_avg_cumm") or (how == "p_score"):
        rightax = st.sidebar.selectbox("Right-ax", ["boosters", "herhaalprik","rioolwater", None], index = 1, key = "aa")
        mergetype = st.sidebar.selectbox("How to merge", ["inner", "outer"], index = 1, key = "bb")
        show_scatter = st.sidebar.selectbox("Show_scatter", [False, True], index = 0)
        
    else:
        rightax, mergetype, show_scatter = None, None, None
    vanaf_jaar = st.sidebar.number_input ("Beginjaar voor CI-interv. (incl.)", 2000, 2022,2015)
    sma= st.sidebar.number_input ("Smooth moving average", 0, 100,6)
    sma_center = st.sidebar.selectbox("SMA center", [True, False], index = 0, key = "cc")
    country = "NL" #  st.sidebar.selectbox("country",["NL", "BE", "DE", "DK", "FR", "ES", "IT", "UK"], index=0)
    return how,yaxis_to_zero,rightax,mergetype, show_scatter, vanaf_jaar,sma, sma_center, country

def main():
    # import cbsodata
    # data_ruw = pd.DataFrame(cbsodata.get_data("03759ned"))
    # st.write(data_ruw)
    st.header("(Over)sterfte per week per geslacht per 5 jaars groep")
   
    how, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center, country = interface()
    plot(how, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center, country)
    footer(vanaf_jaar)



if __name__ == "__main__":
  
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()
