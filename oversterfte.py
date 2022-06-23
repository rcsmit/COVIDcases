# Imitating RIVM overstefte grafieken
# overlijdens per week: https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1655808656887

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

def get_data_for_series(seriename):
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_per_week.csv"
    try:
        df_ = pd.read_csv(
            file,
            delimiter=";",
            
            low_memory=False,
        )
        df = df_[["jaar","weeknr","aantal_dgn",seriename]].copy(deep=True)
        df = df[df["aantal_dgn"] == 7]
    except:
        print("Error reading data")
        st.stop()
    return df


def plot(series_names, how):

   

    for col, series_name in enumerate(series_names):
        print (f"---{series_name}----")
        df_data = get_data_for_series(series_name)
        
        if how =="quantiles":
            df_corona_20 = df_data[(df_data["jaar"] ==2020)].copy(deep=True)
            df_corona_21 = df_data[(df_data["jaar"] ==2021)].copy(deep=True)
            
            df_to_use = df_data[(df_data["jaar"] !=2020) & (df_data["jaar"] !=2021)].copy(deep=True)
            df_quantile = pd.DataFrame(
                {"": [],  "q10": [], "q25": [], "q50":[] ,"avg": [], "q75": [], "q90": []}    )
            
            week_list = df_to_use['weeknr'].unique().tolist()
            
            for w in week_list:

                df_to_use_ = df_to_use[(df_to_use["weeknr"] == w)].copy(deep=True)
                data = df_to_use_[series_name] #.tolist()
                q05 = np.percentile(data, 5)
                q25 = np.percentile(data, 25)
                q50 = np.percentile(data, 50)
                q75 = np.percentile(data, 75)
                q95 = np.percentile(data, 95)
               
                avg = data.mean()


                df_quantile = df_quantile.append(
                    {
                        "week_": w,
                        "q05": q05,
                        "q25": q25,
                        "q50": q50,
                        "avg": avg,

                        "q75": q75,

                        "q95": q95,
                        },
                    ignore_index=True,
                )
            print (df_quantile)
            fig = go.Figure()
            q05 = go.Scatter(
                name='q05',
                x=df_quantile["week_"],
                y=df_quantile['q05'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.1)', fill='tonexty')

            q25 = go.Scatter(
                name='q25',
                x=df_quantile["week_"],
                y=df_quantile['q25'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty')

            avg = go.Scatter(
                name='gemiddeld',
                x=df_quantile["week_"],
                y=df_quantile["avg"],
                mode='lines',
                line=dict(width=0.75,color='rgba(68, 68, 68, 0.8)'),
                )

            value_in_year_2020 = go.Scatter(
                name="2020",
                x=df_corona_20["weeknr"],
                y=df_corona_20[series_name],
                mode='lines',
                line=dict(width=2,color='rgba(255, 0, 0, 0.8)'),
                )
            value_in_year_2021 = go.Scatter(
                name="2021",
                x=df_corona_21["weeknr"],
                y=df_corona_21[series_name],
                mode='lines',
                line=dict(width=2,color='rgba(0, 0, 255, 0.8)'),
                )

            q75 = go.Scatter(
                name='q75',
                x=df_quantile["week_"],
                y=df_quantile['q75'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.1)',
                fill='tonexty')


            q95 = go.Scatter(
                name='q95',
                x=df_quantile["week_"],
                y=df_quantile['q95'],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.1)'
            )

            data = [q95, q75, q25, q05,avg, value_in_year_2020, value_in_year_2021 ]
            title = f"Overleden {series_name}"
            layout = go.Layout(xaxis=dict(title="Weeknumber"),yaxis=dict(title="Number of persons"),
                            title=title,)
                
    
            fig = go.Figure(data=data, layout=layout)
            fig.update_layout(xaxis=dict(tickformat="%d-%m"))
            st.plotly_chart(fig, use_container_width=True)

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
        
def main():
    serienames = ["totaal_m_v_0_999","totaal_m_0_999","totaal_v_0_999","totaal_m_v_0_65","totaal_m_0_65","totaal_v_0_65","totaal_m_v_65_80","totaal_m_65_80","totaal_v_65_80","totaal_m_v_80_999","totaal_m_80_999","totaal_v_80_999"]
    how = st.sidebar.selectbox("How", ["quantiles", "Lines"], index = 0)
    plot(serienames, how)

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()