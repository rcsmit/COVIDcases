import pandas as pd
import cbsodata


import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
# import platform
import scipy.stats as stats
from scipy.signal import savgol_filter


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
    """Load the csv with the excess mortality as calculated by Ariel Karlinsky and Dmitry Kobak
    https://elifesciences.org/articles/69336#s4
    https://github.com/dkobak/excess-mortality/
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

def get_all_data():
    """_summary_

    Returns:
        _type_: df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater,df_
    """    
    df_boosters = get_boosters()
    df_herhaalprik = get_herhaalprik()
    df_herfstprik = get_herfstprik()
    #df_rioolwater_dag, df_rioolwater = None, None # get_rioolwater.scrape_rioolwater()
    df_kobak = get_kobak()
    df_rioolwater = get_rioolwater_simpel()
    df_ = get_sterftedata()


    return df_, df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater, df_kobak

@st.cache_data(ttl=60 * 60 * 24)
def get_sterftedata():
    data = pd.DataFrame(cbsodata.get_data('70895ned'))
 
    data[['jaar','week']] = data.Perioden.str.split(" week ",expand=True,)
    #data['week_'] = data['week_']+" _"
    #data['week_'] = data['week_'].replace(' dagen','_',  regex=True)
    #data['week_'] = data['week_'].replace('\)_',')',  regex=True)
    # Remove rows where 'Perioden' contains 'dagen'
    data = data[~data['Perioden'].str.contains('dagen')]
    data = data[~data['Perioden'].str.contains('dag')]
    
    data = data.reset_index()

    #data[['week','aantal_dagen']] = data.week_.str.split(" ",expand=True,)
    data = data[data['week'].notnull()] #remove the year-totals
    # data['aantal_dagen'] = data['aantal_dagen'].replace('_','7')
    # data['aantal_dagen'] = data['aantal_dagen'].replace('\(','', regex=True)
    # data['aantal_dagen'] = data['aantal_dagen'].replace('_\)','', regex=True)
    # data = data[data['aantal_dagen'] == '7'] # only complete weeks
    data["weeknr"] = data["jaar"].astype(str) +"_" + data["week"].astype(str).str.zfill(2)
    import math
    data["week_int"]=data['week'].astype(int)
    #data["week_int"].apply(lambda x: float(x))
    data["virtuele_maand"] = ((data["week_int"]-1)/4)+1
    data["virtuele_maand"]=data['virtuele_maand'].astype(int)
    data["virtuele_maandnr"] = data["jaar"].astype(str) +"_" + data["virtuele_maand"].astype(str).str.zfill(2)
    #data = data.round({'virtuele_maand': 0})

    data['Geslacht'] = data['Geslacht'].replace(['Totaal mannen en vrouwen'],'m_v_')
    data['Geslacht'] = data['Geslacht'].replace(['Mannen'],'m_')
    data['Geslacht'] = data['Geslacht'].replace(['Vrouwen'],'v_')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['Totaal leeftijd'],'0_999')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['0 tot 65 jaar'],'0_64')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['65 tot 80 jaar'],'65_79')
    data['LeeftijdOp31December'] = data['LeeftijdOp31December'].replace(['80 jaar of ouder'],'80_999')
    data['categorie'] = data['Geslacht']+data['LeeftijdOp31December']

    df = data.pivot(index=['weeknr', "jaar", "week"], columns='categorie', values = 'Overledenen_1').reset_index()
    df["week"] = df["week"].astype(int)
    df["jaar"] = df["jaar"].astype(int)
    
    return df

def get_rioolwater_simpel():
    # if platform.processor() != "":
    #     file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rioolwaarde2024.csv"
    # else: 
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwaarde2024.csv"
    df_rioolwater = pd.read_csv(
        file,
        delimiter=",",
        low_memory=False,
        )
    df_rioolwater["weeknr"] = df_rioolwater["jaar"].astype(int).astype(str) +"_"+df_rioolwater["week"].astype(int).astype(str)
    df_rioolwater["rioolwater_sma"] =  df_rioolwater["rioolwaarde"].rolling(window = 5, center = False).mean().round(1)
    
    return df_rioolwater


def get_df_offical():
    """Laad de waardes zoals door RIVM en CBS is bepaald. Gedownload dd 11 juni 2024
    Returns:
        _df
    """    
    file="C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\overl_cbs_vs_rivm.csv"
    # else: 
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overl_cbs_vs_rivm.csv"
    df_ = pd.read_csv(
        file,
        delimiter=",",
        
        low_memory=False,
    )
    df_["weeknr_z"] = df_["jaar_z"].astype(str) +"_" + df_["week_z"].astype(str).str.zfill(2)
    df_["verw_rivm_official"] = (df_["low_rivm_official"] + df_["high_rivm_official"])/2

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
    som_2015_2019 = 0
    for y in range (2015,2020):
        df_year = df[(df["jaar"] == y)]
        som = df_year["m_v_0_999"].sum()
        som_2015_2019 +=som

        # https://www.cbs.nl/nl-nl/nieuws/2022/22/in-mei-oversterfte-behalve-in-de-laatste-week/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2022%20is%20deze%20155%20493.
        #  https://www.cbs.nl/nl-nl/nieuws/2024/06/sterfte-in-2023-afgenomen/oversterfte-en-verwachte-sterfte#:~:text=Daarom%20is%20de%20sterfte%20per,2023%20is%20deze%20156%20666.
        # https://opendata.cbs.nl/statline/#/CBS/nl/dataset/85753NED/table?dl=A787C
        # Define the factors for each year
        factors = {
            2014: 1,
            2015: 1,
            2016: 1,
            2017: 1,
            2018: 1,
            2019: 1,
            2020: 153402 / (som_2015_2019/5),
            2021: 154887 / (som_2015_2019/5),
            2022: 155494 / (som_2015_2019/5),
            2023: 156666 / (som_2015_2019/5),  # or 169333 / som if you decide to use the updated factor
            2024: 157846 / (som_2015_2019/5)
        }
    avg_overledenen_2015_2019 = (som_2015_2019/5)
    
    # Loop through the years 2014 to 2024 and apply the factors
    for year in range(2014, 2025):
        new_column_name = f"{seriename}_factor_{year}"
        factor = factors[year]
        #factor=1
        df[new_column_name] = df[seriename] * factor

            
    return df

def rolling (df, what):
   
    df[f'{what}_sma'] = df[what].rolling(window=6, center=True).mean()
    
    df[what] = df[what].rolling(window=6, center=False).mean()
    #df[f'{what}_sma'] = savgol_filter(df[what], 7,2)
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
    
    df_oversterfte["over_onder_sterfte"] =  0
    df_oversterfte["meer_minder_sterfte"] =  0
    
    df_oversterfte["year_minus_high95"] = df_oversterfte[series_name] - df_oversterfte["high95"]
    df_oversterfte["year_minus_avg"] = df_oversterfte[series_name]- df_oversterfte["avg"]
    df_oversterfte["p_score"] = ( df_oversterfte[series_name]- df_oversterfte["avg"]) /   df_oversterfte["avg"]
    df_oversterfte= rolling(df_oversterfte, "p_score")
    
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
            b= "rioolwater_sma"
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
    
    for idx, year in enumerate(year_list):
        df = df_data[df_data['jaar'] == year].copy(deep=True)  # [['weeknr', series_name]].reset_index()

                #df = df.sort_values(by=['weeknr'])
        if year == 2020 or year ==2021  or year ==2022 or year ==2023 or year ==2024:
            width = 3
            opacity = 1
        else:
            width = .7
            opacity = .3
      
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
                y=df_quantile["q50"],
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



def show_difference_testing(df__, date_field, show_official):
    """Function to show the difference between the two methods quickly

    ONLY BASSELINE CBS MODEL AND BASELINE CBS OFFICIAL FOR TESTING PURPOSES
    """
    df_baseline_kobak = get_baseline_kobak()
    df = pd.merge(df__, df_baseline_kobak,on="weeknr")
    rolling(df, 'baseline_kobak')
    fig = go.Figure()
   
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['baseline_kobak'],
        mode='lines',
        name='Baseline Kobak'
        ))

    # Voeg de voorspelde lijn toe
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['verw_cbs_official'],
        mode='lines',
        name='Baseline model cbs  official'
        ))

        # Voeg de voorspelde lijn toe
   
   # Voeg de voorspelde lijn toe
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['verw_cbs_sma'],
        mode='lines',
        name='Baseline model cbs q50'))
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['verw_cbs_avg_sma'],
        mode='lines',
        name='Baseline model cbs avg'))


    # fig.add_trace(go.Scatter(
    #         x=df[date_field],
    #         y=df['avg'],
    #         mode='lines',
    #         name='Baseline model cbs'))

    # Voeg de voorspelde lijn RIVM toe
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['aantal_overlijdens'],
        mode='lines',
        name='Werkelijk overleden'
    ))
    # Titel en labels toevoegen
    fig.update_layout(
        title='Vergelijking CBS vs RIVM',
        xaxis_title='Tijd',
        yaxis_title='Aantal Overledenen'
    )

    st.plotly_chart(fig)

def get_baseline_kobak():
    """Load the csv with the baseline as calculated by Ariel Karlinsky and Dmitry Kobak
        https://elifesciences.org/articles/69336#s4
        https://github.com/dkobak/excess-mortality/

    Returns:
        _type_: _description_
    """
    
    url ="C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\kobak_baselines.csv"     # Maak een interactieve plot met Plotly
    df_ = pd.read_csv(
        url,
        delimiter=",",
        low_memory=False,
    )

   
    df_["weeknr"] = df_["jaar"].astype(str) +"_" + df_["week"].astype(str).str.zfill(2)
    df_ = df_[["weeknr", "baseline_kobak"]]
    return df_



def show_difference(df, date_field, show_official):
    """Function to show the difference between the two methods quickly
    """
    
    # df_baseline_kobak = get_baseline_kobak()
    # df = pd.merge(df__, df_baseline_kobak,on="weeknr")
    # rolling(df, 'baseline_kobak')

    
   # Maak een interactieve plot met Plotly
    fig = go.Figure()

    # fig.add_trace(go.Scatter(
    #     x=df[date_field],
    #     y=df['baseline_kobak'],
    #     mode='lines',
    #     name='Baseline Kobak'
    #     ))
    
        # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['high_rivm'],
        mode='lines',
        fill=None,
        line_color='yellow',
        name='high rivm'
    ))

    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['low_rivm'],
        mode='lines',
        fill='tonexty',  # Vul het gebied tussen de lijnen
        line_color='yellow',
        name='low rivm'
    ))

  
    
    # Voeg de betrouwbaarheidsinterval toe
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['low_cbs_sma'],
        mode='lines',
        fill=None,
        line_color='lightgrey',
        name='low cbs'
    ))

    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['high_cbs_sma'],
        mode='lines',
        fill='tonexty',  # Vul het gebied tussen de lijnen
        line_color='lightgrey',
        name='high cbs'
    ))
    if show_official:
        fig.add_trace(go.Scatter(
            x=df[date_field],
            y=df['high_rivm_official'],
            mode='lines',
            fill=None,
            line_color='orange',
            name='high rivm official'
        ))

        fig.add_trace(go.Scatter(
            x=df[date_field],
            y=df['low_rivm_official'],
            mode='lines',
            fill='tonexty',  # Vul het gebied tussen de lijnen
            line_color='orange',
            name='low rivm  official'
        ))
 
     
        # Voeg de betrouwbaarheidsinterval toe
        fig.add_trace(go.Scatter(
            x=df[date_field],
            y=df['low_cbs_official'],
            mode='lines',
            fill=None,
            line_color='lightblue',
            name='low cbs  official'
        ))

        fig.add_trace(go.Scatter(
            x=df[date_field],
            y=df['high_cbs_official'],
            mode='lines',
            fill='tonexty',  # Vul het gebied tussen de lijnen
            line_color='lightblue',
            name='high cbs  official'
        ))
       # Voeg de voorspelde lijn toe
        fig.add_trace(go.Scatter(
            x=df[date_field],
            y=df['verw_rivm_official'],
            mode='lines',
            name='Baseline model rivm  official'
        ))
    # Voeg de voorspelde lijn toe
        fig.add_trace(go.Scatter(
            x=df[date_field],
            y=df['verw_cbs_official'],
            mode='lines',
            name='Baseline model cbs  official'
        ))
    
        # Voeg de voorspelde lijn toe
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['verw_rivm'],
        mode='lines',
        name='Baseline model rivm'
    ))


    # Voeg de voorspelde lijn toe
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['verw_cbs_sma'],
        mode='lines',
        name='Baseline model cbs'))

    
    # Voeg de voorspelde lijn RIVM toe
    fig.add_trace(go.Scatter(
        x=df[date_field],
        y=df['aantal_overlijdens'],
        mode='lines',
        name='Werkelijk overleden'
    ))
    # Titel en labels toevoegen
    fig.update_layout(
        title='Vergelijking CBS vs RIVM',
        xaxis_title='Tijd',
        yaxis_title='Aantal Overledenen'
    )

    st.plotly_chart(fig)



def make_df_qantile(series_name, df_data):
    """_summary_

    Args:
        series_name (_type_): _description_
        df_data (_type_): _description_

    Returns:
        df_corona: df with baseline
        df_quantiles : df with quantiles
    """    
   
    df_corona = df_data[df_data["jaar"].between(2015, 2025)]

    # List to store individual quantile DataFrames
    df_quantiles = []

    # Loop through the years 2014 to 2024
    for year in range(2015, 2025):
        df_quantile_year = make_df_quantile_year(series_name, df_data, year)
        df_quantiles.append(df_quantile_year)

    # Concatenate all quantile DataFrames into a single DataFrame
    df_quantile = pd.concat(df_quantiles, axis=0)    
    
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
                        "avg_": avg,
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
    df_to_use = df_data[(df_data["jaar"] >= 2015 ) & (df_data["jaar"] <2020)].copy(deep=True)
   
    
    df_quantile =None
  
           
    week_list = df_to_use['weeknr'].unique().tolist()
    for w in range(1,53):
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, w)
        df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)
    # if year==2020:
    #     #2020 has a week 53
    #     df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, 53)
    #     df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)
  
        
    return df_quantile
        
def duplicate_row(df, from_,to):
    """Duplicates a row

    Args:
        df (df): df
        from_ (str): oorspronkelijke rij eg. '2022_51'
        to (str): bestemmingsrij eg. '2022_52'
    """    
     # Find the row where weeknr is '2022_51' and duplicate it
    row_to_duplicate = df[df['weeknr'] == from_].copy()

    # Update the weeknr value to '2022_52' in the duplicated row
    row_to_duplicate['weeknr'] = to
    row_to_duplicate['week'] =int(to.split('_')[1])
    #row_to_duplicate['m_v_0_999'] = 0
    # df_merged['week'] = np.where(df_merged['weeknr'] == '2021_52', 52, df_merged['week'])
    # df_merged['week'] = np.where(df_merged['weeknr'] == '2022_52', 52, df_merged['week'])
    # df_merged['week'] = np.where(df_merged['weeknr'] == '2019_01', 1, df_merged['week'])
    # df_merged['week'] = np.where(df_merged['weeknr'] == '2015_01', 1, df_merged['week'])
    
    # Append the duplicated row to the DataFrame
    df = pd.concat([df,row_to_duplicate], ignore_index=True)


    df = df.sort_values(by=['weeknr']).reset_index(drop=True)
     
    return df

def footer():
    st.write("De correctiefactor voor 2020, 2021 en 2022 is berekend over de gehele populatie.")
    st.write("Het 95%-interval is berekend aan de hand van het gemiddelde en standaarddeviatie (z=2)  over de waardes per week van 2015 t/m 2019")
    # st.write("Week 53 van 2020 heeft een verwachte waarde en 95% interval van week 52")
    st.write("Enkele andere gedeeltelijke weken zijn samengevoegd conform het CBS bestand")
    st.write("Code: https://github.com/rcsmit/COVIDcases/blob/main/oversterfte.py")
    st.write("P score = (verschil - gemiddelde) / gemiddelde, gesmooth over 6 weken")