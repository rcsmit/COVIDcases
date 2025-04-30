import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime

import eurostat
from oversterfte_cbs_functions import get_data_for_series, get_data_for_series_wrapper, make_df_quantile
from oversterfte_plot_functions import layout_annotations_fig 
from oversterfte_get_data import get_herhaalprik,get_boosters 

# (Over)sterfte per week of  maand per geslacht per 5 jaars groep"


def get_dataframe(file, delimiter=";"):
    """Get data from a file and return as a pandas DataFrame.

    Args:
        file (str): url or path to the file.
        delimiter (str, optional): _description_. Defaults to ";".

    Returns:
        pd.DataFrame: dataframe
    """   
    
    data = pd.read_csv(
        file,
        delimiter=delimiter,
        low_memory=False,
         encoding='utf-8',
          on_bad_lines='skip'
    )
    return data



def calculate_per_100k(df_long):
    # calculte per 100k
    # only valid when plotting 5-year-bins. (je kan geen fracties bij elkaar optellen denk ik)
    # Display the resulting dataframe
   
    population = get_dataframe(r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv")   
    
    # Step 1: Define bin edges and labels to match df_long
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            55, 60, 65, 70, 75, 80, 85, 90, 999]
    labels = ['Y_LT5_T'] + [f'Y{i}-{i+4}_T' for i in range(5, 85, 5)] + ['Y85-89_T', 'Y_GE90_T']

    # Step 2: Filter and bin ages
    pop = population[population['geslacht'] == 'T'].copy()
    pop['leeftijd'] = pd.to_numeric(pop['leeftijd'], errors='coerce')
    pop = pop.dropna(subset=['leeftijd'])  # drop unknowns
    pop['age_sex'] = pd.cut(pop['leeftijd'], bins=bins, labels=labels, right=False)

    # Step 3: Group and sum
    pop_grouped = pop.groupby(['jaar', 'age_sex'], as_index=False)['aantal'].sum()
    # Add TOTAL_T rows: total per year
    total = pop.groupby('jaar', as_index=False)['aantal'].sum()
    total['age_sex'] = 'TOTAL_T'

    # Match column order
    total = total[['jaar', 'age_sex', 'aantal']]

    # Append to pop_grouped
    pop_grouped = pd.concat([pop_grouped, total], ignore_index=True)
   
    # Step 4: Merge with df_long
    df_long = df_long.merge(pop_grouped, on=['jaar', 'age_sex'], how='left')
    df_long = df_long.rename(columns={'aantal': 'population'})
   
    df_long["OBS_VALUE"] = df_long["OBS_VALUE"]/df_long["population"]*100000
    return (df_long)

#@st.cache_data()
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
    df_long["week"] = (df_long["TIME_PERIOD"].str[6:]).astype(int)

    # df_long=calculate_per_100k(df_long)

    return (df_long)

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
                x=df_boosters["periodnr"],
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
                x=df_herhaalprik["periodnr"],
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

def get_data_for_series_wrapper(df_, seriename, vanaf_jaar, period):
    
    if seriename == "TOTAL_T":
        df = df_[["jaar","periodenr", seriename]].copy(deep=True)

    else:
        df = df_[["jaar","periodenr","TOTAL_T", seriename]].copy(deep=True)
    
    df = df[ (df["jaar"] >= vanaf_jaar)] 
    df = df.sort_values(by=['jaar',"periodenr"]).reset_index()
  
    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]

    df = get_data_for_series(df, seriename, vanaf_jaar)
    return df

def plot_graph_oversterfte_eurostats(how, df, df_corona, df_boosters, df_herhaalprik, series_name, rightax, mergetype, show_scatter,wdw):
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
   
    
    df_oversterfte = pd.merge(df, df_corona, left_on = "periodenr", right_on="periodenr", how = "outer")
    if rightax == "boosters":
        df_oversterfte = pd.merge(df_oversterfte, df_boosters, on="periodenr", how = mergetype)
    if rightax == "herhaalprik":
        df_oversterfte = pd.merge(df_oversterfte, df_herhaalprik, on="periodenr", how = mergetype)
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
    fig.add_trace( go.Scatter(x=df_oversterfte['periodenr'],
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
                    x=df_oversterfte["periodenr"],
                    y=df_oversterfte[grens],
                    mode='lines',
                    line=dict(width=1,color='rgba(205, 61,62, 1)'),
                    ))
    else:
        grens = "95%_interval"
        
        fig.add_trace( go.Scatter(
                name='low',
                x=df_oversterfte["periodenr"],
                y=df_oversterfte["low05"],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)',
               ))
        fig.add_trace(go.Scatter(
                name='high',
                x=df_oversterfte["periodenr"],
                y=df_oversterfte["high95"],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"), fill='tonexty'
                ))
        
       
        #data = [high, low, fig_, sterfte ]
        fig.add_trace( go.Scatter(
                    name="Verwachte Sterfte",
                    x=df_oversterfte["periodenr"],
                    y=df_oversterfte["avg"],
                    mode='lines',
                    line=dict(width=.5,color='rgba(204, 63, 61, .8)'),
                    )) 
        fig.add_trace( go.Scatter(
                    name="Sterfte",
                    x=df_oversterfte["periodenr"],
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
                    x=df_oversterfte["periodenr"],
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
                    x=df_oversterfte["periodenr"],
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
    USED

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
    series_name_new  = ["Y0-49_T","Y0-64_T", "Y50-64_T", "Y65-79_T", "Y25-49_T", "Y50-59_T", "Y60-69_T", "Y70-79_T", "Y80-89_T", "Y80-120_T", "Y90-120_T", "Y0-120_T"] 
    series_from_db = df__['age_sex'].drop_duplicates().sort_values()
    series_names = list( series_name_new) + list(series_from_db)
    series_to_show = st.sidebar.multiselect("Series to show", series_names,["TOTAL_T"])# series_names)  
    #series_to_show = series_names # ["Y50-54_M","Y50-54_F"]
    #series_to_show = ["Y50-54_M","Y50-54_F"]
    wdw=sma 
    wdw=6 if period =="week" else 2
    if period == "week":
        df__["jaar_week"] = df__["jaar"].astype(str)  +"_" + df__["week"].astype(str).str.zfill(2)
        df__['periodenr']  = df__["jaar"].astype(str)  +"_" + df__["week"].astype(str).str.zfill(2)
        df_ = df__.pivot(index=["periodenr", "jaar", "week"], columns='age_sex', values='OBS_VALUE').reset_index()
        df_ = df_[df_["week"] !=53]
    elif period == "maand":
        df__['maand'] = df__.apply(lambda x: getMonth(x['jaar'], x['week']),axis=1)
        df__['periodenr'] = df__["jaar"].astype(str)  +"_" + df__['maand'].astype(str).str.zfill(2)
        #df__['maandnr_'] = df__["jaar"].astype(str)  +"_" + df__['maandnr'].astype(str).str.zfill(2)
        #df__["jaar_maand"] = df__["jaar"].astype(str)  +"_" + df__["maandnr"].astype(str).str.zfill(2)
        #df__ = df__.groupby(["periodenr",'jaar', 'maandnr_', 'age_sex'], as_index=False)
        
        df_ = df__.pivot_table(index=["periodenr", "jaar", "maand"], columns='age_sex', values='OBS_VALUE',  aggfunc='sum',).reset_index()
        
    df_["Y0-49_T"] = df_["Y_LT5_T"] + df_["Y5-9_T"] + df_["Y10-14_T"]+ df_["Y15-19_T"]+ df_["Y20-24_T"] +  df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]
    df_["Y0-64_T"] = df_["Y_LT5_T"] + df_["Y5-9_T"] + df_["Y10-14_T"]+ df_["Y15-19_T"]+ df_["Y20-24_T"] +  df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]+ df_["Y50-54_T"]+ df_["Y55-59_T"] + df_["Y60-64_T"]

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
        st.subheader(series_name)
        df_data = get_data_for_series_wrapper(df_, series_name, vanaf_jaar, period).copy(deep=True)
        if period == "week":
            df_data["week"] = (df_["periodenr"].str[5:]).astype(int)
        elif period == "maand":
            df_data["maand"] = (df_["periodenr"].str[5:]).astype(int)
        
        df, df_corona, df_quantile = make_df_quantile(series_name, df_data, period)
        
        
        if how =="quantiles":
            oversterfte = int(df_corona[series_name].sum() - df_quantile["avg"].sum())
            st.info(f"Oversterfte simpel = Sterfte minus baseline = {oversterfte}")

            columnlist = ["q05","q25","q50","avg","q75","q95", "low05", "high95"]
            for what_to_sma in columnlist:
                df_quantile[what_to_sma] = df_quantile[what_to_sma].rolling(window=wdw, center=sma_center).mean()

      
            
            #df_quantile["periodenr"] #["jaar"].astype(str) +"_"+ df_quantile["period_"].astype(str).str.zfill(2)
            df_quantile = df_quantile.sort_values(by=['jaar','periodenr'])
            
            
           
            make_plot_eurostats(period, yaxis_to_zero, sma, series_name, df_corona, df_quantile)

        elif (how == "year_minus_avg") or (how == "over_onder_sterfte") or (how == "meer_minder_sterfte") or (how == "p_score"):
            plot_graph_oversterfte_eurostats(how, df_quantile, df_corona, df_boosters, df_herhaalprik, series_name, rightax, mergetype, show_scatter,wdw)
           
        else:
            #fig = plt.figure()
            
            year_list = df_data['jaar'].unique().tolist()
            data = []
            for idx, year in enumerate(year_list):
                df = df_data[df_data['jaar'] == year].copy(deep=True) 
                if year == 2020 or year ==2021:
                    width = 3
                    opacity = 1
                else:
                    width = .7
                    opacity = .3
                
                fig_ = go.Scatter(x=df['periodenr'],
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

def make_plot_eurostats(period, yaxis_to_zero, sma, series_name, df_corona, df_quantile):
    fig = go.Figure()
    low05 = go.Scatter(
                name='low',
                x=df_quantile["periodenr"],
                y=df_quantile['low05'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.1)', fill='tonexty')

    q05 = go.Scatter(
                name='q05',
                x=df_quantile["periodenr"],
                y=df_quantile['q05'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)', fill='tonexty')

    q25 = go.Scatter(
                name='q25',
                x=df_quantile["periodenr"],
                y=df_quantile['q25'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty')

    avg = go.Scatter(
                name='gemiddeld',
                x=df_quantile["periodenr"],
                y=df_quantile["avg"],
                mode='lines',
                line=dict(width=0.75,color='rgba(68, 68, 68, 0.8)'),
                )
    col_sma = series_name +"_sma"
    df_corona[col_sma] =  df_corona[series_name].rolling(window = int(sma), center = True).mean()
    sterfte = go.Scatter(
                name="Sterfte",
                x=df_corona["periodenr"],
                y=df_corona[series_name],)
                #mode='lines',
                #line=dict(width=2,color='rgba(255, 0, 0, 0.8)'),
                
    sterfte_sma = go.Scatter(
                name="Sterfte sma",
                x=df_corona["periodenr"],
                y=df_corona[col_sma],
                mode='lines',
                line=dict(width=2,color='rgba(255, 0, 0, 0.8)'),
                )

    q75 = go.Scatter(
                name='q75',
                x=df_quantile["periodenr"],
                y=df_quantile['q75'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty')

    q95 = go.Scatter(
                name='q95',
                x=df_quantile["periodenr"],
                y=df_quantile['q95'],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)'
            )
    high95 = go.Scatter(
                name='high',
                x=df_quantile["periodenr"],
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
        fig = layout_annotations_fig(fig)
            
    if yaxis_to_zero:
        fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)


   



        

def footer(vanaf_jaar):

    #st.write("Voor de correctiefactor voor 2020, tot en met  is uitgegaan van de factor over de gehele populatie. *")
    st.write(f"Het 95%-interval is berekend aan de hand van het gemiddelde en standaarddeviatie (z=2)  over de waardes per week van {vanaf_jaar} t/m 2019, er wordt een lopend gemiddelde berekend per 2 maand")
    # st.write("Week 53 van 2020 heeft een verwachte waarde en 95% interval van week 52")
    #st.write("Enkele andere gedeeltelijke weken zijn samengevoegd conform het CBS bestand")
    st.write("Bron data: Eurostats https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en")
    st.write("Code: https://github.com/rcsmit/COVIDcases/blob/main/oversterfte_eurostats.py")
    st.write("P score = (verschil - gemiddelde) / gemiddelde, gesmooth over 6 weken")
    st.write()
    st.write("*. https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493")

def interface():
    #period = "maand" #= 
    period = st.sidebar.selectbox("Period", ["week", "maand"], index = 0)
    if period == "maand":
        how = "quantiles"
    else:
        how = st.sidebar.selectbox("How", ["quantiles", "Lines", "over_onder_sterfte", "meer_minder_sterfte","year_minus_avg", "p_score"], index = 0)
    
    yaxis_to_zero = st.sidebar.selectbox("Y as beginnen bij 0", [False, True], index = 0)
    if (how == "year_minus_avg") or (how == "p_score"):
        rightax = st.sidebar.selectbox("Right-ax", ["boosters", "herhaalprik", None], index = 1, key = "aa")
        mergetype = st.sidebar.selectbox("How to merge", ["inner", "outer"], index = 0, key = "bb")
        show_scatter = st.sidebar.selectbox("Show_scatter", [False, True], index = 0)
        
    else:
        rightax, mergetype, show_scatter = None, None, None
    vanaf_jaar = st.sidebar.number_input ("Beginjaar voor CI-interv. (incl.)", 2000, 2022,2015)
    if how == "quantiles":
        if period == "maand":
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
    if period == "maand":
        st.write("Er wordt gekeken naar de begindatum vd week voor toewijzing per maand")
    
    plot(how,  period, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center)
    footer(vanaf_jaar)

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()
