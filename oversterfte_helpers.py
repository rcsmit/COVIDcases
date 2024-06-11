import pandas as pd
import cbsodata


import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
# import platform



def get_boosters():
    """_summary_

    Returns:
        _type_: _description_
    """   
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\boosters_per_week_per_leeftijdscat.csv"
    # else: 
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/boosters_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
    df_["weeknr"] = df_["jaar"].astype(str) +"_" + df_["weeknr"].astype(str).str.zfill(2)
    df_ = df_.drop('jaar', axis=1)
   
    df_['boosters_m_v_0_64'] = df_['boosters_m_v_0_49']+df_['boosters_m_v_50_64']
    df_['boosters_m_v_80_999'] = df_['boosters_m_v_80_89']+df_['boosters_m_v_90_999']
   
    return df_
def get_herhaalprik():
    """_summary_

    Returns:
        _type_: _description_
    """    
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\herhaalprik_per_week_per_leeftijdscat.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herhaalprik_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
    df_['herhaalprik_m_v_0_64'] = df_['herhaalprik_m_v_0_49']+df_['herhaalprik_m_v_50_64']
    df_['herhaalprik_m_v_80_999'] = df_['herhaalprik_m_v_80_89']+df_['herhaalprik_m_v_90_999']
  
    df_["weeknr"] = df_["jaar"].astype(str) +"_" + df_["weeknr"].astype(str).str.zfill(2)
    df_ = df_.drop('jaar', axis=1)
 
    return df_



def get_kobak():
    """_summary_

    Returns:
        _type_: _description_
    """    
    # if platform.processor() != "":
    #     # C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\excess-mortality-timeseries_NL_kobak.csv

    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\excess-mortality-timeseries_NL_kobak.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/excess-mortality-timeseries_NL_kobak.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        
        low_memory=False,
    )
  
    return df_

def get_herfstprik():
    """_summary_

    Returns:
        _type_: _description_
    """    
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\herfstprik_per_week_per_leeftijdscat.csv"
    # else:
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herfstprik_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        
        low_memory=False,
    )
    df_['herfstprik_m_v_0_64'] = df_['herfstprik_m_v_0_49']+df_['herfstprik_m_v_50_64']
    df_['herfstprik_m_v_80_999'] = df_['herfstprik_m_v_80_89']+df_['herfstprik_m_v_90_999']
  
    df_["weeknr"] = df_["jaar"].astype(str) +"_" + df_["weeknr"].astype(str).str.zfill(2)
    df_ = df_.drop('jaar', axis=1)
  
    return df_

def get_data_for_series(df_, seriename):
    
    if seriename == "m_v_0_999":
       # df = df_[["jaar","weeknr","aantal_dgn", seriename]].copy(deep=True)
        df = df_[["jaar","weeknr","week", seriename]].copy(deep=True)
        
    else:
       # df = df_[["jaar","weeknr","aantal_dgn","totaal_m_v_0_999", seriename]].copy(deep=True)
        df = df_[["jaar","week","weeknr","m_v_0_999", seriename]].copy(deep=True)

   
    df = df[ (df["jaar"] > 2014)]  #& (df["weeknr"] != 53)]
    #df = df[df["jaar"] > 2014 | (df["weeknr"] != 0) | (df["weeknr"] != 53)]
    df = df.sort_values(by=['jaar','weeknr']).reset_index()
    
    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]
    for y in range (2015,2020):
        df_year = df[(df["jaar"] == y)]
        som = df_year["m_v_0_999"].sum()
        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        #  https://www.cbs.nl/nl-nl/nieuws/2024/06/sterfte-in-2023-afgenomen/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2023%20is%20deze%20156%20666.
        # https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85753NED/table?dl=A787C
        factor_2020 = 153402 / som
        factor_2021 = 154887 / som
        factor_2022 = 155494 / som
        factor_2023 = 156666 / som # 169333 / som  # was 156 666, wellicht excl. de sterfte door corona ?
        factor_2024 = 157846 / som # berekend door de intervallen 2023/2022 op elkaar te delen // 169521 / som  #NOG OPZOEKEN
        for i in range(len(df)):
            
            if df.loc[i,"jaar"] == y:
                #for s in serienames:
                new_column_name_2020 = seriename + "_factor_2020"
                new_column_name_2021 = seriename + "_factor_2021"
                new_column_name_2022 = seriename + "_factor_2022"
                new_column_name_2023 = seriename + "_factor_2023"
                new_column_name_2024 = seriename + "_factor_2024"
                df.loc[i,new_column_name_2020] = df.loc[i,seriename] * factor_2020
                df.loc[i,new_column_name_2021] = df.loc[i,seriename] * factor_2021               
                df.loc[i,new_column_name_2022] = df.loc[i,seriename] * factor_2022
                df.loc[i,new_column_name_2023] = df.loc[i,seriename] * factor_2023
                df.loc[i,new_column_name_2024] = df.loc[i,seriename] * factor_2024
        
    return df


def plot_graph_oversterfte(how, df, df_corona, df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater,df_kobak, series_name, rightax, mergetype, sec_y):
            
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
    booster_cat = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999"]

    df_oversterfte = pd.merge(df, df_corona, left_on = "week_", right_on="weeknr", how = "outer")
    
    if rightax == "boosters":
        df_oversterfte = pd.merge(df_oversterfte, df_boosters, on="weeknr", how = mergetype)
    if rightax == "herhaalprik":
        df_oversterfte = pd.merge(df_oversterfte, df_herhaalprik, on="weeknr", how = mergetype)
    if rightax == "herfstprik":
        df_oversterfte = pd.merge(df_oversterfte, df_herfstprik, on="weeknr", how = mergetype)
    if rightax == "rioolwater":
        df_oversterfte = pd.merge(df_oversterfte, df_rioolwater, on="weeknr", how = mergetype)
    if rightax == "kobak":
        df_oversterfte = pd.merge(df_oversterfte, df_kobak, on="weeknr", how = mergetype)
    # print (df_oversterfte)
    what_to_sma_ = ["low05", "high95"]
    for what_to_sma in what_to_sma_:
        df_oversterfte[what_to_sma] = df_oversterfte[what_to_sma].rolling(window=6, center=True).mean()

    df_oversterfte["over_onder_sterfte"] =  0
    df_oversterfte["meer_minder_sterfte"] =  0
    
    df_oversterfte["year_minus_high95"] = df_oversterfte[series_name] - df_oversterfte["high95"]
    df_oversterfte["year_minus_avg"] = df_oversterfte[series_name]- df_oversterfte["avg"]
    df_oversterfte["p_score"] = ( df_oversterfte[series_name]- df_oversterfte["avg"]) /   df_oversterfte["avg"]
    df_oversterfte["p_score"] = df_oversterfte["p_score"].rolling(window=6, center=True).mean()

    for i in range( len (df_oversterfte)):
        if df_oversterfte.loc[i,series_name ] >  df_oversterfte.loc[i,"high95"] :
            df_oversterfte.loc[i,"over_onder_sterfte" ] =  df_oversterfte.loc[i,series_name ]  -  df_oversterfte.loc[i,"avg"] #["high95"]
            df_oversterfte.loc[i,"meer_minder_sterfte" ] =  df_oversterfte.loc[i,series_name ]  -  df_oversterfte.loc[i,"high95"] 
        elif df_oversterfte.loc[i,series_name ] <  df_oversterfte.loc[i,"low05"]:
            df_oversterfte.loc[i,"over_onder_sterfte" ] =     df_oversterfte.loc[i,series_name ] - df_oversterfte.loc[i,"avg"] #["low05"]
            df_oversterfte.loc[i,"meer_minder_sterfte" ] =     df_oversterfte.loc[i,series_name ] - df_oversterfte.loc[i,"low05"]

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
    if series_name in booster_cat or rightax == "rioolwater" :
        if rightax == "boosters":
            
            b= "boosters_"+series_name
            fig.add_trace(  go.Scatter(
                    name='boosters',
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[b],
                    mode='lines',
                    
                    line=dict(width=2,
                            color="rgba(94, 172, 219, 1)")
                    )  ,secondary_y=sec_y) 
            corr = df_oversterfte[b].corr(df_oversterfte[how]) 
            st.write(f"Correlation = {round(corr,3)}")     
        elif rightax == "herhaalprik" :          
           
                
            b= "herhaalprik_"+series_name
            fig.add_trace(  go.Scatter(
                    name='herhaalprik',
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[b],
                    mode='lines',
                    
                    line=dict(width=2,
                            color="rgba(94, 172, 219, 1)")
                    )  ,secondary_y=sec_y) 
        
            corr = df_oversterfte[b].corr(df_oversterfte[how])
            
            st.write(f"Correlation = {round(corr,3)}")  
        elif rightax == "herfstprik" :          
        
            
            b= "herfstprik_"+series_name
            fig.add_trace(  go.Scatter(
                    name='herfstprik',
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[b],
                    mode='lines',
                    
                    line=dict(width=2,
                            color="rgba(94, 172, 219, 1)")
                    )  ,secondary_y=sec_y) 
        
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
                    )  ,secondary_y=sec_y) 
        
            corr = df_oversterfte[b].corr(df_oversterfte[how])
            
            st.write(f"Correlation = {round(corr,3)}")  
        elif rightax == "kobak" :          
        
            # print (df_oversterfte)
            b= "excess deaths"
            fig.add_trace(  go.Scatter(
                    name='excess deaths(kobak)',
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[b],
                    mode='lines',
                    
                    line=dict(width=2,
                            color="rgba(94, 172, 219, 1)")
                    )  ,secondary_y=sec_y) 
        
            corr = df_oversterfte[b].corr(df_oversterfte[how])
            
            st.write(f"Correlation = {round(corr,3)}")  
   
    #data.append(booster)  
            
    title = how
    layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                            title=title,)
             
    fig.add_hline(y=0)

    fig.update_yaxes(rangemode='tozero')

    st.plotly_chart(fig, use_container_width=True)
    #plot(df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_sterfte, df_kobak, serienames, how, yaxis_to_zero, rightax, mergetype)
def plot(df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_,        df_kobak, series_names, how, yaxis_to_zero, rightax, mergetype, sec_y):
    """_summary_

    Args:
        df_ : df_sterfte
        series_names (_type_): _description_
        how (_type_): _description_
        yaxis_to_zero (_type_): _description_
        rightax (_type_): _description_
        mergetype (_type_): _description_
    """    
    print("plot is called")
    
    for col, series_name in enumerate(series_names):
        print (f"---{series_name}----")
        df_data = get_data_for_series(df_, series_name).copy(deep=True)
        
        df_corona, df_quantile = make_df_qantile(series_name, df_data)
        st.subheader(series_name)
        if how =="quantiles":
            
            plot_quantiles(yaxis_to_zero, series_name, df_corona, df_quantile)

        elif (how == "year_minus_avg")  or (how == "over_onder_sterfte") or (how == "meer_minder_sterfte") or (how == "p_score"):
            plot_graph_oversterfte(how, df_quantile, df_corona, df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_kobak, series_name, rightax, mergetype, sec_y)
            #plot_graph_oversterfte(how, df_quantile, df_corona,  series_name, rightax, mergetype)
           

        else:
            plot_lines(series_name, df_data)
def plot_lines(series_name, df_data):
    #fig = plt.figure()
            
    year_list = df_data['jaar'].unique().tolist()
            
    data = []
    # print (year_list)
            
    for idx, year in enumerate(year_list):
        df = df_data[df_data['jaar'] == year].copy(deep=True)  # [['weeknr', series_name]].reset_index()

                #df = df.sort_values(by=['weeknr'])
        if year == 2020 or year ==2021  or year ==2022 or year ==2023 or year ==2024:
            width = 3
            opacity = 1
        else:
            width = .7
            opacity = .3
        # print (df)
               
        fig_ = go.Scatter(x=df['week'],
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

def plot_quantiles(yaxis_to_zero, series_name, df_corona, df_quantile):
    columnlist = ["q05","q25","q50","avg","q75","q95", "low05", "high95"]
    for what_to_sma in columnlist:
        df_quantile[what_to_sma] = df_quantile[what_to_sma].rolling(window=6, center=True).mean()

                 

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

    sterfte = go.Scatter(
                name="Sterfte",
                x=df_corona["weeknr"],
                y=df_corona[series_name],
                mode='lines',
                line=dict(width=2,color='rgba(255, 0, 0, 0.8)'),
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
    data = [ high95,low05,avg, sterfte] #, value_in_year_2021 ]
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

    fig.add_vrect(x0="2020_13", x1="2020_18", 
              annotation_text="Eerste golf", annotation_position="top left",
              fillcolor="pink", opacity=0.25, line_width=0)
    fig.add_vrect(x0="2020_39", x1="2021_03", 
              annotation_text="Tweede golf", annotation_position="top left",
              fillcolor="pink", opacity=0.25, line_width=0)
    fig.add_vrect(x0="2021_33", x1="2021_52", 
              annotation_text="Derde golf", annotation_position="top left",
              fillcolor="pink", opacity=0.25, line_width=0)

    # hittegolven
    fig.add_vrect(x0="2020_33", x1="2020_34", 
              annotation_text=" ", annotation_position="top left",
              fillcolor="yellow", opacity=0.35, line_width=0)

              
    fig.add_vrect(x0="2022_32", x1="2022_33", 
              annotation_text=" ", annotation_position="top left",
              fillcolor="yellow", opacity=0.35, line_width=0)
    
    fig.add_vrect(x0="2023_23", x1="2023_24", 
              annotation_text=" ", annotation_position="top left",
              fillcolor="yellow", opacity=0.35, line_width=0)
    fig.add_vrect(x0="2023_36", x1="2023_37", 
              annotation_text="Geel = Hitte golf", annotation_position="top left",
              fillcolor="yellow", opacity=0.35, line_width=0)
    

    if yaxis_to_zero:
        fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)
    return df_quantile

def make_df_qantile(series_name, df_data):
    """_summary_

    Args:
        series_name (_type_): _description_
        df_data (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # df_data =  filter_rivm(df_data, series_name)
    df_corona_20 = df_data[(df_data["jaar"] ==2020)].copy(deep=True)
    df_corona_21 = df_data[(df_data["jaar"] ==2021)].copy(deep=True)
    df_corona_22 = df_data[(df_data["jaar"] ==2022)].copy(deep=True)
    df_corona_23 = df_data[(df_data["jaar"] ==2023)].copy(deep=True)
    df_corona_24 = df_data[(df_data["jaar"] ==2024)].copy(deep=True)
    df_corona = pd.concat([df_corona_20, df_corona_21,  df_corona_22,df_corona_23,df_corona_24],axis = 0)
    #df_corona["weeknr"] = df_corona["jaar"].astype(str) +"_" + df_corona["weeknr"].astype(str).str.zfill(2)
   
    df_quantile_2020 = make_df_quantile(series_name, df_data, 2020)
    df_quantile_2021 = make_df_quantile(series_name, df_data, 2021)
    df_quantile_2022 = make_df_quantile(series_name, df_data, 2022)
    df_quantile_2023 = make_df_quantile(series_name, df_data, 2023)
    df_quantile_2024 = make_df_quantile(series_name, df_data, 2024)
    
    
    df_quantile = pd.concat([df_quantile_2020, df_quantile_2021,  df_quantile_2022,  df_quantile_2023,  df_quantile_2024],axis = 0)
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
    
    df_to_use_ = df_to_use[(df_to_use["week"] == w)].copy(deep=True)
    
    
    column_to_use = series_name +  "_factor_" + str(year)
    data = df_to_use_[column_to_use ] #.tolist()
    try:           
        q05 = np.percentile(data, 5)
        q25 = np.percentile(data, 25)
        q50 = np.percentile(data, 50)
        q75 = np.percentile(data, 75)
        q95 = np.percentile(data, 95)
    except:
        q05, q25,q50,q75,q95 = 0,0,0,0,0
                
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
    df_to_use = df_data[(df_data["jaar"] > 2014 ) & (df_data["jaar"] !=2020) & (df_data["jaar"] !=2021) & (df_data["jaar"] !=2022)& (df_data["jaar"] !=2023)& (df_data["jaar"] !=2024)].copy(deep=True)
    
    
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
        

def footer():
    st.write("De correctiefactor voor 2020, 2021 en 2022 is berekend over de gehele populatie.")
    st.write("Het 95%-interval is berekend aan de hand van het gemiddelde en standaarddeviatie (z=2)  over de waardes per week van 2015 t/m 2019")
    # st.write("Week 53 van 2020 heeft een verwachte waarde en 95% interval van week 52")
    st.write("Enkele andere gedeeltelijke weken zijn samengevoegd conform het CBS bestand")
    st.write("Code: https://github.com/rcsmit/COVIDcases/blob/main/oversterfte.py")
    st.write("P score = (verschil - gemiddelde) / gemiddelde, gesmooth over 6 weken")

