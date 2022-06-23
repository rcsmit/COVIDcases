# Imitating RIVM overstefte grafieken
# overlijdens per week: https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1655808656887

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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


def plot(series_names):

    for col, series_name in enumerate(series_names):
        #fig = plt.figure()
        print (f"---{series_name}----")
        df_data = get_data_for_series(series_name)
        year_list = df_data['jaar'].unique().tolist()
        data = []
        for idx, year in enumerate(year_list):
            df = df_data[df_data['jaar'] == year][['weeknr', series_name]].reset_index()

            df = df.sort_values(by=['weeknr'])
            if year == 2020 or year ==2021:
                width = 2
                opacity = 1
            else:
                width = 1
                opacity = .5

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
    plot(serienames)

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()