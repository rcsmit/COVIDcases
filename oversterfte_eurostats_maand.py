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

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import platform
import plotly.express as px
import datetime
#from oversterfte_eurostats import get_data_eurostat
import eurostat

@st.cache_data()
def get_data_eurostat():
    """Get from Eurostat : Deaths by week, sex and 5-year age group
    Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
   
    Returns:
        df: dataframe with weekly mortality in 5 year tranches

                                DATAFLOW        LAST UPDATE freq     age sex  ... OBS_VALUE OBS_FLAG   age_sex  jaar weeknr
            0      ESTAT:DEMO_R_MWK_05(1.0)  24/06/24 23:00:00    W   TOTAL   F  ...    1868.0        p   TOTAL_F  2000      1
            1      ESTAT:DEMO_R_MWK_05(1.0)  24/06/24 23:00:00    W   TOTAL   M  ...    1699.0        p   TOTAL_M  2000      1
            2      ESTAT:DEMO_R_MWK_05(1.0)  24/06/24 23:00:00    W   TOTAL   T  ...    3567.0        p   TOTAL_T  2000      1
            ...
            80071  ESTAT:DEMO_R_MWK_05(1.0)  24/06/24 23:00:00    W   Y_LT5   M  ...       8.0        p   Y_LT5_M  2024     19
            80072  ESTAT:DEMO_R_MWK_05(1.0)  24/06/24 23:00:00    W   Y_LT5   T  ...      12.0        p   Y_LT5_T  2024     19
 
                W  = Weekly
                NL = Netherlands
                NR = Number
                P  = Provisory
    """
    
    code = "DEMO_R_MWK_05"
    # ['freq', 'age', 'sex', 'unit', 'geo']

    # pars = eurostat.get_pars(code)
    # result : ['freq', 'age', 'sex', 'unit', 'geo']
    # for p in pars:
    #     par_values = eurostat.get_par_values(code,p)
    #     print (f"{p} ------------------------------")
    #     print (par_values)
        
    my_filter_pars = {'beginPeriod': 2015, 'geo': ['NL']} # beginPeriod is ignored somehow
    
    flags = True

    if flags:
        df = eurostat.get_data_df(code, flags=True, filter_pars=my_filter_pars, verbose=True, reverse_time=False)
        print (df)
        # Identify value and flag columns
        value_columns = [col for col in df.columns if col.endswith('_value')]
        flag_columns = [col for col in df.columns if col.endswith('_flag')]

        # Melt the value columns
        df_values = df.melt(id_vars=['freq', 'age', 'sex', 'unit', 'geo\\TIME_PERIOD'],
                            value_vars=value_columns,
                            var_name='TIME_PERIOD', value_name='OBS_VALUE')

        # Remove '_value' suffix from TIME_PERIOD column
        df_values['TIME_PERIOD'] = df_values['TIME_PERIOD'].str.replace('_value', '')

        # Melt the flag columns
        df_flags = df.melt(id_vars=['freq', 'age', 'sex', 'unit', 'geo\\TIME_PERIOD'],
                        value_vars=flag_columns,
                        var_name='TIME_PERIOD', value_name='OBS_FLAG')

        # Remove '_flag' suffix from TIME_PERIOD column
        df_flags['TIME_PERIOD'] = df_flags['TIME_PERIOD'].str.replace('_flag', '')

        # Merge the values and flags dataframes
        df_long = pd.merge(df_values, df_flags, on=['freq', 'age', 'sex', 'unit', 'geo\\TIME_PERIOD', 'TIME_PERIOD'])

        # Add additional columns
        df_long['DATAFLOW'] = 'ESTAT:DEMO_R_MWK_05(1.0)'
        df_long['LAST UPDATE'] = '14/06/24 23:00:00'

        # Rename the columns to match the desired output
        df_long.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)

        # Filter out rows with None values in OBS_VALUE
        df_long = df_long[df_long['OBS_VALUE'].notna()]

        # Reorder the columns
        df_long = df_long[['DATAFLOW', 'LAST UPDATE', 'freq', 'age', 'sex', 'unit', 'geo', 'TIME_PERIOD', 'OBS_VALUE', 'OBS_FLAG']]
    else:
        df = eurostat.get_data_df(code, flags=False, filter_pars=my_filter_pars, verbose=True, reverse_time=False)
        print (df)

        # Melt the dataframe to long format
        df_long = df.melt(id_vars=['freq', 'age', 'sex', 'unit', r'geo\TIME_PERIOD'],
                        var_name='TIME_PERIOD', value_name='OBS_VALUE')

        # Add additional columns, made to be reverse compatible with older code
        df_long['DATAFLOW'] = 'ESTAT:DEMO_R_MWK_05(1.0)'
        df_long['LAST UPDATE'] = '24/06/24 23:00:00'
        #df_long['OBS_FLAG'] = 'p'

        # Rename the columns to match the desired output
        df_long.rename(columns={'geo\\TIME_PERIOD': 'geo'}, inplace=True)


        # Filter out rows with None values in OBS_VALUE
        df_long = df_long[df_long['OBS_VALUE'].notna()]

        # Reorder the columns
        df_long = df_long[['DATAFLOW', 'LAST UPDATE', 'freq', 'age', 'sex', 'unit', 'geo', 'TIME_PERIOD', 'OBS_VALUE', 'OBS_FLAG']]
    
    df_long["age_sex"] = df_long["age"] + "_" +df_long["sex"]
    df_long["jaar"] = (df_long["TIME_PERIOD"].str[:4]).astype(int)
    df_long["weeknr"] = (df_long["TIME_PERIOD"].str[6:]).astype(int)

    # Display the resulting dataframe
    print (df_long)
    return (df_long)


def get_sterfte_DELETE():
    """_summary_

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
    # if platform.processor() != "":
    #     #file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_per_week_meer_leeftijdscat.csv"
    #     file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats.csv"
    #     # file = r"C:\Users\rcxsm\Downloads\demo_r_mwk_05__custom_3595567_linear.csv"
    #     # file = r"C:\Users\rcxsm\Downloads\demo_r_mwk_05__custom_3595595_linear.csv"
    # else:
    #     file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats.csv"
    #     # file = r"C:\Users\rcxsm\Downloads\demo_r_mwk_05__custom_3595567_linear.csv"
    #     # file = r"C:\Users\rcxsm\Downloads\demo_r_mwk_05__custom_3595595_linear.csv"
    # df_ = pd.read_csv(
    #     file,
    #     delimiter=",",
        
    #     low_memory=False,
    #     )
   
    # df_["age_sex"] = df_["age"] + "_" +df_["sex"]
    # df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    # df_["weeknr"] = (df_["TIME_PERIOD"].str[6:]).astype(int)
    df_ = get_data_eurostat()
    st.write(df_)
    return df_

def get_boosters():
    """_summary_

    Returns:
        _type_: _description_
    """   
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\boosters_per_week_per_leeftijdscat.csv"
    else: 
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/boosters_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
    df_["weeknr"] = df_["jaar"].astype(str) +"_" + df_["weeknr"].astype(str).str.zfill(2)
    df_ = df_.drop('jaar', axis=1)
    return df_
def get_herhaalprik():
    """_summary_

    Returns:
        _type_: _description_
    """    
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\herhaalprik_per_week_per_leeftijdscat.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herhaalprik_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
    df_["weeknr"] = df_["jaar"].astype(str) +"_" + df_["weeknr"].astype(str).str.zfill(2)
    df_ = df_.drop('jaar', axis=1)

    return df_

def plot_boosters(df_boosters, series_name):
    """_summary_

    Args:
        df_boosters (_type_): _description_
        series_name (_type_): _description_
    """   
    
    booster_cat = ["m_v_0_999","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_999"]

    if series_name in booster_cat:
       
        fig = make_subplots()
        b= "boosters_"+series_name
        fig.add_trace(  go.Scatter(
                name='boosters',
                x=df_boosters["weeknr"],
                y=df_boosters[b],
                mode='lines',
                
                line=dict(width=2,
                        color="rgba(255, 0, 255, 1)")
                )  )                  
   
                
        title = f"Boosters for {series_name}"
        layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                                title=title,)
             
        fig.update_yaxes(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

def plot_herhaalprik(df_herhaalprik, series_name):
    """_summary_

    Args:
        df_herhaalprik (_type_): _description_
        series_name (_type_): _description_
    """   
    
    booster_cat = ["m_v_0_999","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_999"]

    if series_name in booster_cat:
       
        fig = make_subplots()
        b= "herhaalprik_"+series_name
        fig.add_trace(  go.Scatter(
                name='herhaalprik',
                x=df_herhaalprik["weeknr"],
                y=df_herhaalprik[b],
                mode='lines',
                
                line=dict(width=2,
                        color="rgba(255, 0, 255, 1)")
                )  )                  
   
                
        title = f"herhaalprik for {series_name}"
        layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                                title=title,)
             
        fig.update_yaxes(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

def get_data_for_series(df_, seriename, vanaf_jaar, period):
    period_ = "weeknr" if period=="week" else "maandnr"
    
    if seriename == "TOTAL_T":
       # df = df_[["jaar","weeknr","aantal_dgn", seriename]].copy(deep=True)
        df = df_[["jaar","periode_", seriename]].copy(deep=True)

    else:
       # df = df_[["jaar","weeknr","aantal_dgn","totaal_m_v_0_999", seriename]].copy(deep=True)
        df = df_[["jaar","periode_","TOTAL_T", seriename]].copy(deep=True)
    #df = df[(df["aantal_dgn"] == 7) & (df["jaar"] > 2014)]
    df = df[ (df["jaar"] >= vanaf_jaar)]  #& (df["weeknr"] != 53)]
    #df = df[df["jaar"] > 2014 | (df["weeknr"] != 0) | (df["weeknr"] != 53)]
    df = df.sort_values(by=['jaar',"periode_"]).reset_index()
  
    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]

    noemer = 149832 # average deaths per year 2015-2019
    for y in range(2015, 2020):
        df_year = df[(df["jaar"] == y)]
        # som = df_year["m_v_0_999"].sum()
        # som_2015_2019 +=som

        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        #  https://www.cbs.nl/nl-nl/nieuws/2024/06/sterfte-in-2023-afgenomen/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2023%20is%20deze%20156%20666.
        # https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85753NED/table?dl=A787C
        # Define the factors for each year
        

        # 2.5.1 Verwachte sterfte en oversterfte
        # De oversterfte is het verschil tussen het waargenomen aantal overledenen en een verwacht 
        # aantal overledenen in dezelfde periode. Het verwachte aantal overledenen wanneer er geen 
        # COVID-19-epidemie zou zijn geweest, wordt geschat op basis van de waargenomen sterfte in 
        # 2015–2019 in twee stappen. Eerst wordt voor elk jaar de sterfte per week bepaald.
        # Vervolgens wordt per week de gemiddelde sterfte in die week en de zes omliggende weken bepaald. 
        # Deze gemiddelde sterfte per week levert een benadering van de verwachte wekelijkse sterfte. 
        # Er is dan nog geen rekening gehouden met de ontwikkeling van de bevolkingssamenstelling. 

        # Daarom is de sterfte per week nog herschaald naar de verwachte totale sterfte voor het jaar. 
        # Het verwachte aantal overledenen in het hele jaar wordt bepaald op basis van de prognoses 
        # die het CBS jaarlijks maakt. Deze prognoses geven de meest waarschijnlijke toekomstige 
        # ontwikkelingen van de bevolking en de sterfte. De prognoses houden rekening met het feit 
        # dat de bevolking continu verandert door immigratie en vergrijzing. Het CBS gebruikt voor 
        # de prognose van de leeftijds- en geslachtsspecifieke sterftekansen een extrapolatiemodel 
        # (L. Stoeldraijer, van Duin et al., 2013
        # https://pure.rug.nl/ws/portalfiles/portal/13869387/stoeldraijer_et_al_2013_DR.pdf
        # ): er wordt van uitgegaan dat de toekomstige trends 
        # een voortzetting zijn van de trends uit het verleden. In het model wordt niet alleen 
        # uitgegaan van de trends in Nederland, maar ook van de meer stabiele trends in andere 
        # West-Europese landen. Tijdelijke versnellingen en vertragingen die voorkomen in de 
        # Nederlandse trends hebben zo een minder groot effect op de toekomstverwachtingen. 
        # Het model houdt ook rekening met het effect van rookgedrag op de sterfte, wat voor 
        # Nederland met name belangrijk is om de verschillen tussen mannen en vrouwen in sterftetrends 
        # goed te beschrijven.
        # https://www.cbs.nl/nl-nl/longread/rapportages/2023/oversterfte-en-doodsoorzaken-in-2020-tot-en-met-2022?onepage=true
        # Op basis van de geprognosticeerde leeftijds- en geslachtsspecifieke sterftekansen en de verwachte bevolkingsopbouw in dat jaar, wordt het verwachte aantal overledenen naar leeftijd en geslacht berekend voor een bepaald jaar. Voor 2020 is de verwachte sterfte 153 402, voor 2021 is deze 154 887 en voor 2022 is dit 155 493. 
        # Op basis van de geprognosticeerde leeftijds- en geslachtsspecifieke sterftekansen en de verwachte
        #  bevolkingsopbouw in dat jaar, wordt het verwachte aantal overledenen naar leeftijd en geslacht 
        # berekend voor een bepaald jaar. Voor 2020 is de verwachte sterfte 153 402, 
        # voor 2021 is deze 154 887 en voor 2022 is dit 155 493. 
        # Het aantal voor 2020 is ontleend aan de Kernprognose 2019-2060 
        # (L. Stoeldraijer, van Duin, C., Huisman, C., 2019), het aantal voor 2021 aan de
        #  Bevolkingsprognose 2020-2070 exclusief de aanname van extra sterfgevallen door de 
        # COVID-19-epidemie; (L. Stoeldraijer, de Regt et al., 2020) en het aantal voor 2022 
        # aan de Kernprognose 2021-2070 (exclusief de aanname van extra sterfgevallen door de coronapandemie) (L. Stoeldraijer, van Duin et al., 2021). 
        # https://www.cbs.nl/nl-nl/longread/rapportages/2023/oversterfte-en-doodsoorzaken-in-2020-tot-en-met-2022/2-data-en-methode
        
        # geen waarde voor 2024, zie https://twitter.com/Cbscommunicatie/status/1800505651833270551
        # huidige waarde 2024 is geexptrapoleerd 2022-2023
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

            # 2015	16,9	0,5
            # 2016	17	0,6
            # 2017	17,1	0,6
            # 2018	17,2	0,6
            # 2019	17,3	0,6
            # 2020	17,4	0,7
            # 2021	17,5	0,4
            # 2022	17,6	0,7
            # 2023	17,8	1,3
            # 2024	17,9	0,7
    # avg_overledenen_2015_2019 = (som_2015_2019/5)
    # st.write(avg_overledenen_2015_2019)
    # Loop through the years 2014 to 2024 and apply the factors
    for year in range(2014, 2025):
        new_column_name = f"{seriename}_factor_{year}"
        factor = factors[year]
        # factor=1
        df[new_column_name] = df[seriename] * factor

    return df


def plot_graph_oversterfte(how, df, df_corona, df_boosters, df_herhaalprik, series_name, rightax, mergetype, show_scatter,wdw):
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
    booster_cat = ["m_v_0_999","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_999"]
    print (df)
    print (df_corona)
    df["weeknr"] = df["week_"] 
    df_corona["weeknr"] = df_corona["periode_"] 
    df_oversterfte = pd.merge(df, df_corona, left_on = "week_", right_on="weeknr", how = "outer")
    if rightax == "boosters":
        df_oversterfte = pd.merge(df_oversterfte, df_boosters, on="weeknr", how = mergetype)
    if rightax == "herhaalprik":
        df_oversterfte = pd.merge(df_oversterfte, df_herhaalprik, on="weeknr", how = mergetype)
    what_to_sma_ = ["low05", "high95"]
    for what_to_sma in what_to_sma_:
        # sma = 6 is also in de procedure of CBS
        df_oversterfte[what_to_sma] = df_oversterfte[what_to_sma].rolling(window=wdw, center=True).mean()

    df_oversterfte["over_onder_sterfte"] =  0
    df_oversterfte["meer_minder_sterfte"] =  0
    
    df_oversterfte["year_minus_high95"] = df_oversterfte[series_name] - df_oversterfte["high95"]
    df_oversterfte["year_minus_avg"] = df_oversterfte[series_name]- df_oversterfte["avg"]
    df_oversterfte["p_score"] = ( df_oversterfte[series_name]- df_oversterfte["avg"]) /   df_oversterfte["avg"]
    df_oversterfte["p_score"] = df_oversterfte["p_score"].rolling(window=wdw, center=True).mean()

    for i in range( len (df_oversterfte)):
        if df_oversterfte.loc[i,series_name ] >  df_oversterfte.loc[i,"high95"] :
            df_oversterfte.loc[i,"over_onder_sterfte" ] =  df_oversterfte.loc[i,series_name ]  -  df_oversterfte.loc[i,"avg"] #["high95"]
            df_oversterfte.loc[i,"meer_minder_sterfte" ] =  df_oversterfte.loc[i,series_name ]  -  df_oversterfte.loc[i,"high95"] 
        elif df_oversterfte.loc[i,series_name ] <  df_oversterfte.loc[i,"low05"]:
            df_oversterfte.loc[i,"over_onder_sterfte" ] =     df_oversterfte.loc[i,series_name ] - df_oversterfte.loc[i,"avg"] #["low05"]
            df_oversterfte.loc[i,"meer_minder_sterfte" ] =     df_oversterfte.loc[i,series_name ] - df_oversterfte.loc[i,"low05"]
    # name_ = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\oversterfte"+series_name+".csv"
    # compression_opts = dict(method=None, archive_name=name_)
    # df_oversterfte.to_csv(name_, index=False, compression=compression_opts)
  
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace( go.Scatter(x=df_oversterfte['week_'],
                            y=df_oversterfte[how],
                            #line=dict(width=2), opacity = 1, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                            line=dict(width=2,color='rgba(205, 61,62, 1)'),
                            mode='lines',
                            name=how,
                           ))
    
    if how == "p_score":
       # the p-score is already plotted
       pass
    elif how == "year_minus_avg": 
        show_avg = False
        if show_avg:   
            grens = "avg"
            fig.add_trace(go.Scatter(
                    name=grens,
                    x=df_oversterfte["weeknr"],
                    y=df_oversterfte[grens],
                    mode='lines',
                    line=dict(width=1,color='rgba(205, 61,62, 1)'),
                    ))
    else:
        grens = "95%_interval"
        
        fig.add_trace( go.Scatter(
                name='low',
                x=df_oversterfte["week_"],
                y=df_oversterfte["low05"],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)',
               ))
        fig.add_trace(go.Scatter(
                name='high',
                x=df_oversterfte["week_"],
                y=df_oversterfte["high95"],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"), fill='tonexty'
                ))
        
       
        #data = [high, low, fig_, sterfte ]
        fig.add_trace( go.Scatter(
                    name="Verwachte Sterfte",
                    x=df_oversterfte["week_"],
                    y=df_oversterfte["avg"],
                    mode='lines',
                    line=dict(width=.5,color='rgba(204, 63, 61, .8)'),
                    )) 
        fig.add_trace( go.Scatter(
                    name="Sterfte",
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[series_name],
                    mode='lines',
                    line=dict(width=1,color='rgba(204, 63, 61, 1)'),
                    )) 
    # rightax = "boosters" # "herhaalprik"
    
    if rightax == "boosters":
        if series_name in booster_cat:
            b= "boosters_"+series_name
            fig.add_trace(  go.Scatter(
                    name='boosters',
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[b],
                    mode='lines',
                    
                    line=dict(width=2,
                            color="rgba(94, 172, 219, 1)")
                    )  ,secondary_y=True) 
            corr = df_oversterfte[b].corr(df_oversterfte[how]) 
            st.write(f"Correlation = {round(corr,3)}")     
    elif rightax == "herhaalprik" :          
        if series_name in booster_cat:
            b= "herhaalprik_"+series_name
            fig.add_trace(  go.Scatter(
                    name='herhaalprik',
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[b],
                    mode='lines',
                    
                    line=dict(width=2,
                            color="rgba(94, 172, 219, 1)")
                    )  ,secondary_y=True) 
          
            corr = df_oversterfte[b].corr(df_oversterfte[how]) 
            st.write(f"Correlation = {round(corr,3)}")  

    #data.append(booster)  
    
    #layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
    #                        title=title,)
             
    fig.add_hline(y=0)

    fig.update_yaxes(rangemode='tozero')

    st.plotly_chart(fig, use_container_width=True)
    if series_name in booster_cat and show_scatter == True and rightax !=None:
                
        title = f"{how} vs {b} - corr = {round(corr,3)}"
        fig1xy = px.scatter(df_oversterfte, x=b, y=how, trendline="ols", title = title)
        st.plotly_chart(fig1xy, use_container_width=True)

def getMonth(year: int, week: int) -> int:
    """Return the month number in the given week in the given year."""
    return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w").month
   
def plot( how,period, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center):
    """_summary_

    Args:
        series_names (_type_): _description_
        how (_type_): _description_
        yaxis_to_zero (_type_): _description_
        rightax (_type_): _description_
        mergetype (_type_): _description_
    """    
    df_boosters = get_boosters()
    df_herhaalprik = get_herhaalprik()
    df__ = get_data_eurostat()
    df__ = df__[df__['age'] !="UNK"]
    #serienames = ["m_v_0_999","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_999" ,"m__0_99","m_0_49","m_50_64","m_65_79","m_80_89","m_90_999","v_0_999","v_0_49","v_50_64","v_65_79","v_80_89","v_90_999"]
    #series_names = df__["age_sex"].unique().sort()
    series_name_new  = ["Y0-49_T", "Y50-64_T", "Y65-79_T", "Y25-49_T", "Y50-59_T", "Y60-69_T", "Y70-79_T", "Y80-89_T", "Y80-120_T", "Y90-120_T", "Y0-120_T"] 
    series_from_db = df__['age_sex'].drop_duplicates().sort_values()
    series_names = list( series_name_new) + list(series_from_db)
    series_to_show = st.sidebar.multiselect("Series to show", series_names,["TOTAL_T"])# series_names)  
    #series_to_show = series_names # ["Y50-54_M","Y50-54_F"]
    #series_to_show = ["Y50-54_M","Y50-54_F"]
    wdw=sma 
    # wdw=6 if period =="week" else 2
    if period == "week":
        df__["jaar_week"] = df__["jaar"].astype(str)  +"_" + df__["weeknr"].astype(str).str.zfill(2)
        df__['periode_']  = df__["jaar"].astype(str)  +"_" + df__["weeknr"].astype(str).str.zfill(2)
        df_ = df__.pivot(index=["jaar_week","periode_", "jaar", "weeknr"], columns='age_sex', values='OBS_VALUE').reset_index()
        df_ = df_[df_["weeknr"] !=53]
    elif period == "month":
        df__['maandnr'] = df__.apply(lambda x: getMonth(x['jaar'], x['weeknr']),axis=1)
        df__['periode_'] = df__["jaar"].astype(str)  +"_" + df__['maandnr'].astype(str).str.zfill(2)
        df__['maandnr_'] = df__["jaar"].astype(str)  +"_" + df__['maandnr'].astype(str).str.zfill(2)
        #df__["jaar_maand"] = df__["jaar"].astype(str)  +"_" + df__["maandnr"].astype(str).str.zfill(2)
        #df__ = df__.groupby(["periode_",'jaar', 'maandnr_', 'age_sex'], as_index=False)
        
        df_ = df__.pivot_table(index=["periode_", "jaar", "maandnr_"], columns='age_sex', values='OBS_VALUE',  aggfunc='sum',).reset_index()
        
    df_["Y0-49_T"] = df_["Y_LT5_T"] + df_["Y5-9_T"] + df_["Y10-14_T"]+ df_["Y15-19_T"]+ df_["Y20-24_T"] +  df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]

    df_["Y50-64_T"] = df_["Y50-54_T"]+ df_["Y55-59_T"] + df_["Y60-64_T"]
    df_["Y65-79_T"] =+ df_["Y65-69_T"]+ df_["Y70-74_T"] + df_["Y75-79_T"]
    df_["Y25-49_T"] = df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]


    df_["Y50-59_T"] = df_["Y50-54_T"] + df_["Y55-59_T"]
    df_["Y60-69_T"] = df_["Y60-64_T"] + df_["Y65-69_T"]
    df_["Y70-79_T"] = df_["Y70-74_T"] + df_["Y75-79_T"]
    df_["Y80-89_T"] = df_["Y80-84_T"] + df_["Y85-89_T"]
    df_["Y80-120_T"] = df_["Y80-84_T"] + df_["Y85-89_T"] + df_["Y_GE90_T"]
    df_["Y90-120_T"] = df_["Y_GE90_T"]
    df_["Y0-120_T"] = df_["Y0-49_T"] + df_["Y50-64_T"] + df_["Y65-79_T"] + df_["Y80-89_T"] + df_["Y90-120_T"]
       
          
    for col, series_name in enumerate(series_to_show):
        print (f"---{series_name}----")
        
        df_data = get_data_for_series(df_, series_name, vanaf_jaar, period).copy(deep=True)
        df_data["maandnr"] = (df_["periode_"].str[5:]).astype(int)
         
        df_corona, df_quantile = make_df_quantile(series_name, df_data, period)
        st.subheader(series_name)
        
        if how =="quantiles":
            oversterfte = int(df_corona[series_name].sum() - df_quantile["avg"].sum())
            st.info(f"Oversterfte simpel = Sterfte minus baseline = {oversterfte}")

            columnlist = ["q05","q25","q50","avg","q75","q95", "low05", "high95"]
            for what_to_sma in columnlist:
                df_quantile[what_to_sma] = df_quantile[what_to_sma].rolling(window=wdw, center=sma_center).mean()

                 
            
            df_quantile["periode_"] = df_quantile["jaar"].astype(str) +"_"+ df_quantile["period_"].astype(str).str.zfill(2)
            df_quantile = df_quantile.sort_values(by=['jaar','periode_'])
            

           
            fig = go.Figure()
            low05 = go.Scatter(
                name='low',
                x=df_quantile["periode_"],
                y=df_quantile['low05'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.1)', fill='tonexty')

            q05 = go.Scatter(
                name='q05',
                x=df_quantile["periode_"],
                y=df_quantile['q05'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)', fill='tonexty')

            q25 = go.Scatter(
                name='q25',
                x=df_quantile["periode_"],
                y=df_quantile['q25'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty')

            avg = go.Scatter(
                name='gemiddeld',
                x=df_quantile["periode_"],
                y=df_quantile["avg"],
                mode='lines',
                line=dict(width=0.75,color='rgba(68, 68, 68, 0.8)'),
                )
            col_sma = series_name +"_sma"
            df_corona[col_sma] =  df_corona[series_name].rolling(window = int(sma), center = True).mean()
            sterfte = go.Scatter(
                name="Sterfte",
                x=df_corona["periode_"],
                y=df_corona[series_name],)
                #mode='lines',
                #line=dict(width=2,color='rgba(255, 0, 0, 0.8)'),
                
            sterfte_sma = go.Scatter(
                name="Sterfte sma",
                x=df_corona["periode_"],
                y=df_corona[col_sma],
                mode='lines',
                line=dict(width=2,color='rgba(255, 0, 0, 0.8)'),
                )

            q75 = go.Scatter(
                name='q75',
                x=df_quantile["periode_"],
                y=df_quantile['q75'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty')

            q95 = go.Scatter(
                name='q95',
                x=df_quantile["periode_"],
                y=df_quantile['q95'],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)'
            )
            high95 = go.Scatter(
                name='high',
                x=df_quantile["periode_"],
                y=df_quantile['high95'],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)'
             )
            
            #data = [ q95, high95, q05,low05,avg, sterfte] #, value_in_year_2021 ]
            data = [ high95,low05,avg, sterfte, sterfte_sma] #, value_in_year_2021 ]
            title = f"Overleden {series_name}"
            layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                            title=title,)
                
    
            fig = go.Figure(data=data, layout=layout)
            fig.update_layout(xaxis=dict(tickformat="%d-%m"))
            
            #             — eerste oversterftegolf: week 13 tot en met 18 van 2020 (eind maart–eind april 2020);
            # — tweede oversterftegolf: week 39 van 2020 tot en met week 3 van 2021 (eind
            # september 2020–januari 2021);
            # — derde oversterftegolf: week 33 tot en met week 52 van 2021 (half augustus 2021–eind
            # december 2021).
            # De hittegolf in 2020 betreft week 33 en week 34 (half augustus 2020).
            if period == "week":
                fig.add_vrect(x0="2020_13", x1="2020_18", 
                annotation_text="Eerste golf", annotation_position="top left",
                fillcolor="pink", opacity=0.25, line_width=0)
                fig.add_vrect(x0="2020_39", x1="2021_03", 
                annotation_text="Tweede golf", annotation_position="top left",
                fillcolor="pink", opacity=0.25, line_width=0)
                fig.add_vrect(x0="2021_33", x1="2021_52", 
                annotation_text="Derde golf", annotation_position="top left",
                fillcolor="pink", opacity=0.25, line_width=0)
                fig.add_vrect(x0="2020_33", x1="2020_34", 
                annotation_text="Hitte golf", annotation_position="top left",
                fillcolor="orange", opacity=0.25, line_width=0)
                fig.add_vrect(x0="2022_32", x1="2022_33", 
                annotation_text="Hitte golf", annotation_position="top left",
                fillcolor="orange", opacity=0.25, line_width=0)
            if yaxis_to_zero:
                fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)

        elif (how == "year_minus_avg") or (how == "over_onder_sterfte") or (how == "meer_minder_sterfte") or (how == "p_score"):
            plot_graph_oversterfte(how, df_quantile, df_corona, df_boosters, df_herhaalprik, series_name, rightax, mergetype, show_scatter,wdw)
           

        else:
            #fig = plt.figure()
            
            year_list = df_data['jaar'].unique().tolist()
            data = []
            for idx, year in enumerate(year_list):
                df = df_data[df_data['jaar'] == year].copy(deep=True)  # [['weeknr', series_name]].reset_index()

                #df = df.sort_values(by=['weeknr'])
                if year == 2020 or year ==2021:
                    width = 3
                    opacity = 1
                else:
                    width = .7
                    opacity = .3
                
                fig_ = go.Scatter(x=df['weeknr'],
                            y=df[series_name],
                            line=dict(width=width), opacity = opacity, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                            mode='lines',
                            name=year,
                            legendgroup=str(year))
                    
                data.append(fig_)
            
            title = f"Stefte - {series_name}"
            layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                            title=title,)
                
    
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig, use_container_width=True)

def make_df_quantile(series_name, df_data, period):
    """_summary_

    Args:
        series_name (_type_): _description_
        df_data (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # df_corona_20 = df_data[(df_data["jaar"] ==2020)].copy(deep=True)
    # df_corona_21 = df_data[(df_data["jaar"] ==2021)].copy(deep=True)
    # df_corona_22 = df_data[(df_data["jaar"] ==2022)].copy(deep=True)
    # df_corona = pd.concat([df_corona_20, df_corona_21,  df_corona_22],axis = 0)
    
    # df_quantile_2020 = make_df_quantile_jaar(series_name, df_data, 2020, period)
    # df_quantile_2021 = make_df_quantile_jaar(series_name, df_data, 2021, period)
    # df_quantile_2022 = make_df_quantile_jaar(series_name, df_data, 2022, period)
    # df_quantile = pd.concat([df_quantile_2020, df_quantile_2021,  df_quantile_2022],axis = 0)
    df_quantiles = []
    df_coronas = []
    # Loop through the years 2014 to 2024
    for year in range(2015, 2025):
        df_corona_year = df_data[(df_data["jaar"] ==year)].copy(deep=True)
        df_quantile_year = make_df_quantile_jaar(series_name, df_data, year, period)
        df_quantiles.append(df_quantile_year)
        df_coronas.append(df_corona_year)
    # Concatenate all quantile DataFrames into a single DataFrame
    df_quantile = pd.concat(df_quantiles, axis=0)
    df_corona = pd.concat(df_coronas,axis=0)
    if period == "week":
            
        df_corona["periodenr"] = df_corona["jaar"].astype(str) +"_" + df_corona["maandnr"].astype(str).str.zfill(2)
        df_quantile["periode_"]= df_quantile["jaar"].astype(str) +"_" + df_quantile['period_'].astype(str).str.zfill(2)
    elif period == "maand":
        df_corona["periodenr"] = df_corona["jaar"].astype(str) +"_" + df_corona["maandnr"].astype(str).str.zfill(2)
        df_quantile["periode_"]= df_quantile["jaar"].astype(str) +"_" + df_quantile['period_'].astype(str).str.zfill(2)
    return df_corona,df_quantile

def make_row_df_quantile(series_name, year, df_to_use, w_, period):
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
  
    if period == "week":
        df_to_use_ = df_to_use[(df_to_use["maandnr"] == w)].copy(deep=True)
    elif period == "month":
        df_to_use_ = df_to_use[(df_to_use["maandnr"] == w)].copy(deep=True)
   

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
                        "period_": w_,
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

def make_df_quantile_jaar(series_name, df_data, year,period):

    """ Calculate the quantiles

    Returns:
        _type_: _description_
    """    
    df_to_use = df_data[(df_data["jaar"] >= 2015) & (df_data["jaar"] < 2020)].copy(deep=True)
    df_quantile =None
           
    #week_list = df_to_use['weeknr'].unique().tolist()
            # week_list = week_list.sort()
          
            
            #for w in week_list:  #puts week 1 at the end
    end = 53 if period == "week" else 13
        
    
    for w in range(1,end):
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, w, period)
        df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)
    if year==2020 and period == "week":
        #2020 has a week 53
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, 53, period)
        df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)

        
    return df_quantile
        


def footer(vanaf_jaar):

    st.write("Voor de correctiefactor voor 2020, 2021 en 2022 is uitgegaan van de factor over de gehele populatie. *")
    st.write(f"Het 95%-interval is berekend aan de hand van het gemiddelde en standaarddeviatie (z=2)  over de waardes per week van {vanaf_jaar} t/m 2019, er wordt een lopend gemiddelde berekend per 2 maand")
    # st.write("Week 53 van 2020 heeft een verwachte waarde en 95% interval van week 52")
    #st.write("Enkele andere gedeeltelijke weken zijn samengevoegd conform het CBS bestand")
    st.write("Bron data: Eurostats https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en")
    st.write("Code: https://github.com/rcsmit/COVIDcases/blob/main/oversterfte_eurostats.py")
    st.write("P score = (verschil - gemiddelde) / gemiddelde, gesmooth over 6 weken")
    st.write()
    st.write("*. https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493")

def interface():
    #period = "month" #= 
    period = st.sidebar.selectbox("Period", ["week", "month"], index = 0)
    if period == "month":
        how = "quantiles"
    else:
        how = st.sidebar.selectbox("How", ["quantiles", "Lines", "over_onder_sterfte", "meer_minder_sterfte","year_minus_avg", "p_score"], index = 0)
    
    yaxis_to_zero = st.sidebar.selectbox("Y as beginnen bij 0", [False, True], index = 0)
    if (how == "year_minus_avg") or (how == "p_score"):
        rightax = st.sidebar.selectbox("Right-ax", ["boosters", "herhaalprik", None], index = 1, key = "aa")
        mergetype = st.sidebar.selectbox("How to merge", ["inner", "outer"], index = 0, key = "aa")
        show_scatter = st.sidebar.selectbox("Show_scatter", [False, True], index = 0)
        
    else:
        rightax, mergetype, show_scatter = None, None, None
    vanaf_jaar = st.sidebar.number_input ("Beginjaar voor CI-interv. (incl.)", 2000, 2022,2015)
    if how == "quantiles":
        if period == "month":
            sma_def = 3
        else:
            sma_def = 6
        sma= st.sidebar.number_input ("Smooth moving average", 0, 100,sma_def)
        sma_center = st.sidebar.selectbox("SMA center", [True, False], index = 0, key = "bb")
    else:
        sma, sma_center = None, None
    return how,period,yaxis_to_zero,rightax,mergetype, show_scatter, vanaf_jaar,sma, sma_center

def main():
    st.header("(Over)sterfte per week of  maand per geslacht per 5 jaars groep")
    
    how, period, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center = interface()
    if period == "month":
        st.write("Er wordt gekeken naar de begindatum vd week voor toewijzing per maand")
    
    plot(how,  period, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center)
    footer(vanaf_jaar)

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()
