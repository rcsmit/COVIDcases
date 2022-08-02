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

def get_boosters():
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/boosters_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
    df_["weeknr"] = df_["jaar"].astype(str) +"_" + df_["weeknr"].astype(str).str.zfill(2)
    return df_
def get_herhaalprik():
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herhaalprik_per_week_per_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
    df_["weeknr"] = df_["jaar"].astype(str) +"_" + df_["weeknr"].astype(str).str.zfill(2)
    return df_

def plot_boosters(df_boosters, series_name):
   
    
    booster_cat = ["m_v_0_999","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_999"]

    if series_name in booster_cat:
       
        fig = make_subplots()
        b= "boosters_"+series_name
        fig.add_trace(  go.Scatter(
                name='boosters',
                x=df_boosters["weeknr"],
                y=df_boosters[b],
                mode='lines',
                
                line=dict(width=0.5,
                        color="rgba(255, 0, 255, 0.5)")
                )  )                  
   
                
        title = f"Boosters for {series_name}"
        layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                                title=title,)
             
        fig.update_yaxes(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

def plot_herhaalprik(df_herhaalprik, series_name):
   
    
    booster_cat = ["m_v_0_999","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_999"]

    if series_name in booster_cat:
       
        fig = make_subplots()
        b= "herhaalprik_"+series_name
        fig.add_trace(  go.Scatter(
                name='herhaalprik',
                x=df_herhaalprik["weeknr"],
                y=df_herhaalprik[b],
                mode='lines',
                
                line=dict(width=0.5,
                        color="rgba(255, 0, 255, 0.5)")
                )  )                  
   
                
        title = f"herhaalprik for {series_name}"
        layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                                title=title,)
             
        fig.update_yaxes(title_text=title)
        st.plotly_chart(fig, use_container_width=True)

def get_data_for_series(seriename):
    #file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_per_week.csv"
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_per_week_meer_leeftijdscat.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
    print (df_)
    if seriename == "m_v_0_999":
       # df = df_[["jaar","weeknr","aantal_dgn", seriename]].copy(deep=True)
        df = df_[["jaar","weeknr", seriename]].copy(deep=True)

    else:
       # df = df_[["jaar","weeknr","aantal_dgn","totaal_m_v_0_999", seriename]].copy(deep=True)
        df = df_[["jaar","weeknr","m_v_0_999", seriename]].copy(deep=True)
    #df = df[(df["aantal_dgn"] == 7) & (df["jaar"] > 2014)]
    df = df[ (df["jaar"] > 2014)]  #& (df["weeknr"] != 53)]
    #df = df[df["jaar"] > 2014 | (df["weeknr"] != 0) | (df["weeknr"] != 53)]
    df = df.sort_values(by=['jaar','weeknr']).reset_index()
 
    # Voor 2020 is de verwachte sterfte 153 402 en voor 2021 is deze 154 887.
    # serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]

    for y in range (2015,2020):
        df_year = df[(df["jaar"] == y)]
        som = df_year["m_v_0_999"].sum()
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


def plot_graph_oversterfte(how, df, df_corona, series_name):
    df_boosters = get_boosters()
    df_herhaalprik = get_herhaalprik()
    booster_cat = ["m_v_0_999","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_999"]

    df_oversterfte = pd.merge(df, df_corona, left_on = "week_", right_on="weeknr")
    df_oversterfte = pd.merge(df_oversterfte, df_boosters, on="weeknr")
    st.write(df_oversterfte)
    df_oversterfte["over_onder_sterfte"] =  0
    df_oversterfte["year_minus_high95"] = df_oversterfte[series_name] - df_oversterfte["high95"]
    df_oversterfte["year_minus_avg"] = df_oversterfte[series_name]- df_oversterfte["avg"]
    for i in range( len (df_oversterfte)):
        if df_oversterfte.loc[i,series_name ] >  df_oversterfte.loc[i,"high95"] :
            df_oversterfte.loc[i,"over_onder_sterfte" ] =  df_oversterfte.loc[i,series_name ] -  df_oversterfte.loc[i,"high95"] 
        elif df_oversterfte.loc[i,series_name ] <  df_oversterfte.loc[i,"low05"]:
            df_oversterfte.loc[i,"over_onder_sterfte" ] =     df_oversterfte.loc[i,series_name ] - df_oversterfte.loc[i,"low05"]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace( go.Scatter(x=df_oversterfte['week_'],
                            y=df_oversterfte[how],
                            line=dict(width=2), opacity = 1, # PLOT_COLORS_WIDTH[year][1] , color=PLOT_COLORS_WIDTH[year][0]),
                            mode='lines',
                            name=how,
                           ))
    
  
    if how == "year_minus_avg":    
        grens = "avg"
        fig.add_trace(go.Scatter(
                name=grens,
                x=df_oversterfte["weeknr"],
                y=df_oversterfte[grens],
                mode='lines',
                line=dict(width=1,color='rgba(0, 0,255, 0.8)'),
                ))
        #data = [fig_, avg, sterfte ]

    else:
        grens = "95%_interval"
       
        fig.add_trace( go.Scatter(
                name='low',
                x=df_oversterfte["week_"],
                y=df_oversterfte["low05"],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.3)',
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
                name="Sterfte",
                x=df_oversterfte["weeknr"],
                y=df_oversterfte[series_name],
                mode='lines',
                line=dict(width=1,color='rgba(255, 0, 0, 0.8)'),
                )) 
    rightax = "herhaalprik"
    if rightax == "boosters":
        if series_name in booster_cat:
            b= "boosters_"+series_name
            fig.add_trace(  go.Scatter(
                    name='boosters',
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[b],
                    mode='lines',
                    
                    line=dict(width=0.5,
                            color="rgba(255, 0, 255, 0.5)")
                    )  ,secondary_y=True)       
    elif rightax == "herhaalprik" :          
        if series_name in herhaalprik_cat:
            b= "herhaalprik_"+series_name
            fig.add_trace(  go.Scatter(
                    name='herhaalprik',
                    x=df_oversterfte["week_"],
                    y=df_oversterfte[b],
                    mode='lines',
                    
                    line=dict(width=0.5,
                            color="rgba(255, 0, 255, 0.5)")
                    )  ,secondary_y=True)                  
   
    #data.append(booster)  
            
    title = how
    layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                            title=title,)
             
    # fig = go.Figure(data=data, layout=layout) 
    
    # fig.add_trace(booster,
    #         secondary_y=True,
    #     )
    fig.add_hline(y=0)

    fig.update_yaxes(rangemode='tozero')
    # fig.update_yaxes(title_text="Boosters", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    plot_boosters(df_boosters, series_name)
    plot_herhaalprik(df_herhaalprik, series_name)

def plot(series_names, how, yaxis_to_zero):

    for col, series_name in enumerate(series_names):
        print (f"---{series_name}----")
        df_data = get_data_for_series(series_name).copy(deep=True)
        df_corona, df_quantile = make_df_qantile(series_name, df_data)
        st.subheader(series_name)
        if how =="quantiles":
            
            columnlist = ["q05","q25","q50","avg","q75","q95", "low05", "high95"]
            for what_to_sma in columnlist:
                df_quantile[what_to_sma] = df_quantile[what_to_sma].rolling(window=6, center=True).mean()

                 
            print (df_quantile)
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
            fig.add_vrect(x0="2020_33", x1="2020_34", 
              annotation_text="Hitte golf", annotation_position="top left",
              fillcolor="orange", opacity=0.25, line_width=0)
            if yaxis_to_zero:
                fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)


        elif (how == "year_minus_avg") or (how == "over_onder_sterfte"):
            plot_graph_oversterfte(how, df_quantile, df_corona, series_name)


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

def make_df_qantile(series_name, df_data):
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
    df_to_use = df_data[(df_data["jaar"] !=2020) & (df_data["jaar"] !=2021) & (df_data["jaar"] !=2022)].copy(deep=True)
    df_quantile =None
           
    week_list = df_to_use['weeknr'].unique().tolist()
            # week_list = week_list.sort()
          
            
            #for w in week_list:  #puts week 1 at the end
    for w in range(1,53):
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, w)

        df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)
    if year==2020:
        df_quantile_ = make_row_df_quantile(series_name, year, df_to_use, 53)
        df_quantile = pd.concat([df_quantile, df_quantile_],axis = 0)

        
    return df_quantile
        
def main():
    #serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]
    serienames = ["m_v_0_999","m_v_0_49","m_v_50_64","m_v_65_79","m_v_80_89","m_v_90_999","m__0_99","m_0_49","m_50_64","m_65_79","m_80_89","m_90_999","v_0_999","v_0_49","v_50_64","v_65_79","v_80_89","v_90_999"]

    #serienames = ["totaal_m_v_0_999"]
    how = st.sidebar.selectbox("How", ["quantiles", "Lines", "over_onder_sterfte", "year_minus_avg"], index = 0)
    yaxis_to_zero = st.sidebar.selectbox("Y as beginnen bij 0", [False, True], index = 0)
    plot(serienames, how, yaxis_to_zero)
    st.write("De correctiefactor voor 2020, 2021 en 2022 is berekend over de gehele populatie.")
    st.write("Het 95%-interval is berekend aan de hand van het gemiddelde en standaarddeviatie (z=2)  over de waardes per week van 2015 t/m 2019")
    # st.write("Week 53 van 2020 heeft een verwachte waarde en 95% interval van week 52")
    st.write("Enkele andere gedeeltelijke weken zijn samengevoegd conform het CBS bestand")
    st.write("Code: https://github.com/rcsmit/COVIDcases/blob/main/oversterfte.py")
if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()
