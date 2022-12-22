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
import get_rioolwater

def get_sterfte(country):
    """_summary_

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
    if platform.processor() != "":
        #file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_per_week_meer_leeftijdscat.csv"
        file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_new.csv"
        # file = r"C:\Users\rcxsm\Downloads\demo_r_mwk_05__custom_3595567_linear.csv"
        # file = r"C:\Users\rcxsm\Downloads\demo_r_mwk_05__custom_3595595_linear.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_new.csv"
        # file = r"C:\Users\rcxsm\Downloads\demo_r_mwk_05__custom_3595567_linear.csv"
        # file = r"C:\Users\rcxsm\Downloads\demo_r_mwk_05__custom_3595595_linear.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        
        low_memory=False,
        )
    #print (df_.dtypes)
    #st.write(df_)
    #st.write(df_["geo"].unique())
    df_=df_[df_["geo"] == country]
    
    df_["age_sex"] = df_["age"] + "_" +df_["sex"]
    #st.write(df_["age_sex"].unique())
    df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["weeknr"] = (df_["TIME_PERIOD"].str[6:]).astype(int)
    

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

def get_data_for_series(df_, seriename, vanaf_jaar):
  
    if seriename == "TOTAL_T":
       # df = df_[["jaar","weeknr","aantal_dgn", seriename]].copy(deep=True)
        df = df_[["jaar","weeknr", seriename]].copy(deep=True)

    else:
       # df = df_[["jaar","weeknr","aantal_dgn","totaal_m_v_0_999", seriename]].copy(deep=True)
        df = df_[["jaar","weeknr","TOTAL_T", seriename]].copy(deep=True)
    #df = df[(df["aantal_dgn"] == 7) & (df["jaar"] > 2014)]
    df = df[ (df["jaar"] >= vanaf_jaar)]  #& (df["weeknr"] != 53)]
    print (f"Lengte df ({len(df)})")
    #df = df[df["jaar"] > 2014 | (df["weeknr"] != 0) | (df["weeknr"] != 53)]
    df = df.sort_values(by=['jaar','weeknr']).reset_index()
 
    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]

    for y in range (int(vanaf_jaar),2020):
        df_year = df[(df["jaar"] == y)]
        som = df_year["TOTAL_T"].sum()
        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        factor_2020 = 153402 / som
        factor_2021 = 154887 / som
        factor_2022 = 155494 / som
       
        for i in range(len(df)):
            
            
            if df.loc[i,"jaar"] == y:
                #for s in serienames:
                new_column_name_2020 = seriename + "_factor_2020"
                new_column_name_2021 = seriename + "_factor_2021"
                new_column_name_2022 = seriename + "_factor_2022"
                df.loc[i,new_column_name_2020] = df.loc[i,seriename] * factor_2020
                df.loc[i,new_column_name_2021] = df.loc[i,seriename] * factor_2021               
                df.loc[i,new_column_name_2022] = df.loc[i,seriename] * factor_2022
    return df

def plot_graph_oversterfte(how, df, df_corona, df_boosters, df_herhaalprik, df_rioolwater, series_name, rightax, mergetype, show_scatter):
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

    df_oversterfte = pd.merge(df, df_corona, left_on = "week_", right_on="weeknr", how = "outer")
    if rightax == "boosters":
        df_oversterfte = pd.merge(df_oversterfte, df_boosters, on="weeknr", how = mergetype)
    if rightax == "herhaalprik":
        df_oversterfte = pd.merge(df_oversterfte, df_herhaalprik, on="weeknr", how = mergetype)
    if rightax == "rioolwater":
        df_oversterfte = pd.merge(df_oversterfte, df_rioolwater, on="weeknr", how = mergetype)
    what_to_sma_ = ["low05", "high95"]
    for what_to_sma in what_to_sma_:
        # sma = 6 is also in de procedure of CBS
        df_oversterfte[what_to_sma] = df_oversterfte[what_to_sma].rolling(window=6, center=True).mean()

    df_oversterfte["over_onder_sterfte"] =  0
    df_oversterfte["meer_minder_sterfte"] =  0
    
    df_oversterfte["year_minus_high95"] = df_oversterfte[series_name] - df_oversterfte["high95"]
    df_oversterfte["year_minus_avg"] = df_oversterfte[series_name]- df_oversterfte["avg"]
    df_oversterfte["year_minus_avg_only_pos"] =  df_oversterfte["year_minus_avg"] 
    df_oversterfte.year_minus_avg_only_pos=df_oversterfte.year_minus_avg_only_pos.mask(df_oversterfte.year_minus_avg_only_pos.lt(0),0)
    df_oversterfte["year_minus_avg_cumm"] = df_oversterfte["year_minus_avg_only_pos"].cumsum()
    df_oversterfte["p_score"] = ( df_oversterfte[series_name]- df_oversterfte["avg"]) /   df_oversterfte["avg"]
    df_oversterfte["p_score"] = df_oversterfte["p_score"].rolling(window=6, center=True).mean()

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
    # print("--- Saving " + name_ + " ---")
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
    elif how == "year_minus_avg_cumm": 
        show_avg = False
        if show_avg:   
            grens = "year_minus_avg_cumm"
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
                    x=df_oversterfte["weeknr"],
                    y=df_oversterfte["avg"],
                    mode='lines',
                    line=dict(width=.5,color='rgba(204, 63, 61, .8)'),
                    )) 
        fig.add_trace( go.Scatter(
                    name="Sterfte",
                    x=df_oversterfte["weeknr"],
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
    
    elif rightax == "rioolwater" :          
        
            
            b= "value_rivm_official_sma"
            fig.add_trace(  go.Scatter(
                    name='rioolwater',
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

   
def plot( how, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center, country):
    """_summary_

    Args:
        series_names (_type_): _description_
        how (_type_): ["quantiles", "Lines", "over_onder_sterfte", "meer_minder_sterfte","year_minus_avg","year_minus_avg_cumm", "p_score"]
        yaxis_to_zero (_type_): _description_
        rightax (_type_): _description_
        mergetype (_type_): _description_
    """    
    df_boosters, df_herhaalprik, df_ = get_data(country)
    df_rioolwater_dag, df_rioolwater = get_rioolwater.scrape_rioolwater()
   
    #series_names  = df_['age_sex'].drop_duplicates().sort_values()

    series_names_ = df_.columns.tolist()
    series_names_ = series_names_[3:]
    series_names = st.sidebar.multiselect("Which ages to show", series_names_, ["TOTAL_T"])
    print (series_names)
    series_to_show = series_names # ["Y50-54_M","Y50-54_F"]
 
    for col, series_name in enumerate(series_to_show):
        
        
        if how =="quantiles":
            df_data, df_corona, df_quantile = make_df_data_corona_quantile(vanaf_jaar, df_, series_name)
            
            columnlist = ["q05","q25","q50","avg","q75","q95", "low05", "high95"]
            for what_to_sma in columnlist:
                df_quantile[what_to_sma] = df_quantile[what_to_sma].rolling(window=6, center=sma_center).mean()

                 
            #print (df_quantile)
            df_quantile = df_quantile.sort_values(by=['jaar','week_'])
            fig = go.Figure()
            low05 = go.Scatter(
                name='low',
                x=df_quantile["week_"],
                y=df_quantile['low05'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.1)', fill='tonexty')

            q05 = go.Scatter(
                name='q05',
                x=df_quantile["week_"],
                y=df_quantile['q05'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)', fill='tonexty')

            q25 = go.Scatter(
                name='q25',
                x=df_quantile["week_"],
                y=df_quantile['q25'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty')

            avg = go.Scatter(
                name='gemiddeld',
                x=df_quantile["week_"],
                y=df_quantile["avg"],
                mode='lines',
                line=dict(width=0.75,color='rgba(68, 68, 68, 0.8)'),
                )
            if sma == 1:
                sterfte = go.Scatter(
                    name="Sterfte",
                    x=df_corona["weeknr"],
                    y=df_corona[series_name],
                    mode='lines',
                    line=dict(width=2,color='rgba(255, 0, 0, 0.8)'),)

            else:
                sterfte = go.Scatter(
                    name="Sterfte",
                    x=df_corona["weeknr"],
                    y=df_corona[series_name],
                    mode='lines',
                    line=dict(width=1,color='rgba(255, 0, 255, 0.6)'),)

                col_sma = series_name +"_sma"
                df_corona[col_sma] =  df_corona[series_name].rolling(window = int(sma), center = True).mean()
            
                
                sterfte_sma = go.Scatter(
                    name="Sterfte sma",
                    x=df_corona["weeknr"],
                    y=df_corona[col_sma],
                    mode='lines',
                    line=dict(width=2,color='rgba(255, 0, 0, 1)'),
                    )

            q75 = go.Scatter(
                name='q75',
                x=df_quantile["week_"],
                y=df_quantile['q75'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty')

            q95 = go.Scatter(
                name='q95',
                x=df_quantile["week_"],
                y=df_quantile['q95'],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)'
            )
            high95 = go.Scatter(
                name='high',
                x=df_quantile["week_"],
                y=df_quantile['high95'],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)'
             )
            
            #data = [ q95, high95, q05,low05,avg, sterfte] #, value_in_year_2021 ]
            if sma>1:
                data = [ high95,low05,avg, sterfte, sterfte_sma] #, value_in_year_2021 ]
            else:
                data = [ high95,low05,avg, sterfte] #, value_in_year_2021 ]
            title = f"Overleden {series_name} (marges {vanaf_jaar}-2019)"
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
            fig.add_vline(x="2021_01",  line_width=3, line_dash="dash", line_color="green")
            fig.add_vline(x="2022_01",  line_width=3, line_dash="dash", line_color="green")




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

        elif (how == "year_minus_avg") or (how == "year_minus_avg_cumm")  or (how == "over_onder_sterfte") or (how == "meer_minder_sterfte") or (how == "p_score"):
            print (series_name[-1:]) 
            #if series_name[:4] == "m_v_":
            if (series_name[-1:] == "T" or series_name[:4] == "m_v_"):
                df_data, df_corona, df_quantile = make_df_data_corona_quantile(vanaf_jaar, df_, series_name)
                
                plot_graph_oversterfte(how, df_quantile, df_corona, df_boosters, df_herhaalprik, df_rioolwater, series_name, rightax, mergetype, show_scatter)
           

        elif (how == "lines"):
            df_data, df_corona, df_quantile = make_df_data_corona_quantile(vanaf_jaar, df_, series_name)
            
            #fig = plt.figure()
            
            year_list = df_data['jaar'].unique().tolist()
            data = []
            for idx, year in enumerate(year_list):
                df = df_data[df_data['jaar'] == year].copy(deep=True)  # [['weeknr', series_name]].reset_index()

                #df = df.sort_values(by=['weeknr'])
                if year == 2020 or year ==2021   or year ==2022:
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
        else:
            st.error("ERROR in [how]")
            st.stop()

def make_df_data_corona_quantile(vanaf_jaar, df_, series_name):
    print (f"---{series_name}----")
    st.subheader(series_name)
    df_data = get_data_for_series(df_, series_name, vanaf_jaar).copy(deep=True)
    df_corona, df_quantile = make_df_qantile(series_name, df_data)
    return df_data,df_corona,df_quantile
            
@st.cache 
def get_data(country):
    df_boosters = get_boosters()
    df_herhaalprik = get_herhaalprik()
    df__ = get_sterfte(country)
    df__ = df__[df__['age'] !="UNK"]
    df__["jaar_week"] = df__["jaar"].astype(str)  +"_" + df__["weeknr"].astype(str).str.zfill(2)
    df_ = df__.pivot(index=["jaar_week", "jaar", "weeknr"], columns='age_sex', values='OBS_VALUE').reset_index()
    #df_ = df__.pivot(index="jaar_week", columns='age_sex', values='OBS_VALUE').reset_index()
    
    df_["m_v_0_49"] = df_["Y_LT5_T"] + df_["Y5-9_T"] + df_["Y10-14_T"]+ df_["Y15-19_T"]+ df_["Y20-24_T"] +  df_["Y25-29_T"]+ df_["Y30-34_T"]+ df_["Y35-39_T"]+ df_["Y40-44_T"] + df_["Y45-49_T"]

    df_["m_v_50_64"] = df_["Y50-54_T"]+ df_["Y55-59_T"] + df_["Y60-64_T"]
    df_["m_v_65_79"] =+ df_["Y65-69_T"]+ df_["Y70-74_T"] + df_["Y75-79_T"]
    df_["m_v_80_89"] = df_["Y80-84_T"] + df_["Y85-89_T"]
    df_["m_v_90_999"] = df_["Y_GE90_T"]
    df_["m_v_0_999"] = df_["m_v_0_49"] + df_["m_v_50_64"] + df_["m_v_65_79"] + df_["m_v_80_89"] + df_["m_v_90_999"]
    return df_boosters,df_herhaalprik,df_

def make_df_qantile(series_name, df_data):
    """_summary_

    Args:
        series_name (_type_): _description_
        df_data (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df_corona_20 = df_data[(df_data["jaar"] ==2020)].copy(deep=True)
    df_corona_21 = df_data[(df_data["jaar"] ==2021)].copy(deep=True)
    df_corona_22 = df_data[(df_data["jaar"] ==2022)].copy(deep=True)
    df_corona = pd.concat([df_corona_20, df_corona_21,  df_corona_22],axis = 0)
    df_corona["weeknr"] = df_corona["jaar"].astype(str) +"_" + df_corona["weeknr"].astype(str).str.zfill(2)
  
    df_quantile_2020 = make_df_quantile(series_name, df_data, 2020)
    df_quantile_2021 = make_df_quantile(series_name, df_data, 2021)
    df_quantile_2022 = make_df_quantile(series_name, df_data, 2022)
    df_quantile = pd.concat([df_quantile_2020, df_quantile_2021,  df_quantile_2022],axis = 0)
    df_quantile["week_"]= df_quantile["jaar"].astype(str) +"_" + df_quantile['week_'].astype(str).str.zfill(2)
        
    return df_corona,df_quantile

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

def make_df_quantile(series_name, df_data, year):

    """ Calculate the quantiles

    Returns:
        _type_: _description_
    """    
    df_to_use = df_data[(df_data["jaar"] !=2020) & (df_data["jaar"] !=2021) & (df_data["jaar"] !=2022)].copy(deep=True)
    print (f"Lengte df_to_use = {len(df_to_use)}")
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
        mergetype = st.sidebar.selectbox("How to merge", ["inner", "outer"], index = 0, key = "bb")
        show_scatter = st.sidebar.selectbox("Show_scatter", [False, True], index = 0)
        
    else:
        rightax, mergetype, show_scatter = None, None, None
    vanaf_jaar = st.sidebar.number_input ("Beginjaar voor CI-interv. (incl.)", 2000, 2022,2015)
    if how == "quantiles":
        sma= st.sidebar.number_input ("Smooth moving average", 0, 100,1)
        sma_center = st.sidebar.selectbox("SMA center", [True, False], index = 0, key = "bb")
    else:
        sma, sma_center = None, None
    country = "NL" #  st.sidebar.selectbox("country",["NL", "BE", "DE", "DK", "FR", "ES", "IT", "UK"], index=0)
    return how,yaxis_to_zero,rightax,mergetype, show_scatter, vanaf_jaar,sma, sma_center, country

def main():
    st.header("(Over)sterfte per week per geslacht per 5 jaars groep")
   
    how, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center, country = interface()
    plot(how, yaxis_to_zero, rightax, mergetype, show_scatter, vanaf_jaar,sma, sma_center, country)
    footer(vanaf_jaar)

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()
