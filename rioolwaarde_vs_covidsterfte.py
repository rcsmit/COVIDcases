import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

# Define a function to calculate the month from the selected year and week
def calculate_month(row):
    first_day_of_year = datetime(row['selected_year'], 1, 1)
    first_day_of_week = first_day_of_year + timedelta(weeks=row['week'] - 1)
    return first_day_of_week.month

def calculate_month_oversterfte(row):
    first_day_of_year = datetime(int(row['jaar_x_x']), 1, 1)
    first_day_of_week = first_day_of_year + timedelta(weeks=int(row['week_x_x']) - 1)
    return first_day_of_week.month

# Define a function to select the appropriate year based on the week
def select_year(row):
    if row['week'] >= 40:
        return int(row['year'].split('/')[0])
    else:
        return int(row['year'].split('/')[1])


def make_lineplot(merged_df):
    # Create a line plot with two axes using Plotly
    fig = go.Figure()

    # Add the rioolwater data to the plot
    fig.add_trace(go.Scatter(x=merged_df['selected_year'].astype(str) + '-' + merged_df['month'].astype(str),
                            y=merged_df['value_rioolwater'],
                            mode='lines',
                            name='Rioolwater',
                            yaxis='y1'))

    # Add the covid sterfte data to the plot
    fig.add_trace(go.Scatter(x=merged_df['selected_year'].astype(str) + '-' + merged_df['month'].astype(str),
                            y=merged_df['value_covid_sterfte'],
                            mode='lines',
                            name='Covid Sterfte',
                            yaxis='y1'))

    # Add the oversterfte data to the plot
    fig.add_trace(go.Scatter(x=merged_df['selected_year'].astype(str) + '-' + merged_df['month'].astype(str),
                            y=merged_df['oversterfte_cbs'],
                            mode='lines',
                            name='Oversterfte',
                            #  yaxis='y3'
                            ))

    # Update the layout to include three y-axes
    fig.update_layout(
        title='Rioolwater vs Covid Sterfte vs Oversterfte',
        xaxis_title='Year-Month',
        yaxis=dict(
            title='Rioolwater Value',
        
        ),
    
    )

    # Display the plot
    st.plotly_chart(fig)



def make_scatterplot(df_temp, what_to_show_l, what_to_show_r,):
    
    what_to_show_l = what_to_show_l if type(what_to_show_l) == list else [what_to_show_l]
    what_to_show_r = what_to_show_r if type(what_to_show_r) == list else [what_to_show_r]
    x_ = np.array(df_temp[what_to_show_l])
    y_ = np.array(df_temp[what_to_show_r])
            

    idx = np.isfinite(x_) & np.isfinite(y_)
    m, b = np.polyfit(x_[idx], y_[idx], 1)
    model = np.polyfit(x_[idx], y_[idx], 1)

    predict = np.poly1d(model)
    from sklearn.metrics import r2_score

    r2 = r2_score  (y_[idx], predict(x_[idx]))

    fig1xyz = px.scatter(df_temp, x=what_to_show_l[0], y=what_to_show_r[0], 
                        trendline="ols", trendline_scope = 'overall',trendline_color_override = 'black'
            )

    correlation_sp = round(df_temp[what_to_show_l[0]].corr(df_temp[what_to_show_r[0]], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
    correlation_p = round(df_temp[what_to_show_l[0]].corr(df_temp[what_to_show_r[0]], method='pearson'), 3)

    title_scatter_plotly = (f"{what_to_show_l[0]} -  {what_to_show_r[0]}<br>Correlation spearman = {correlation_sp} - Correlation pearson = {correlation_p}<br>y = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

    fig1xyz.update_layout(
        title=dict(
            text=title_scatter_plotly,
            x=0.5,
            y=0.95,
            font=dict(
                family="Arial",
                size=14,
                color='#000000'
            )
        ),
        xaxis_title=what_to_show_l[0],
        yaxis_title=what_to_show_r[0],

    )


    st.plotly_chart(fig1xyz, use_container_width=True)

def main():
    st.write("We zetten de rioolwater af tegen de COVID-sterfte zoals door het CBS is geregistreerd en de oversterfte volgens de CBS methode.")
    st.write("https://www.rivm.nl/corona/actueel/weekcijfers")
    st.write("https://www.cbs.nl/nl-nl/reeksen/tijd/doodsoorzaken")

    # Load the CSV files
    rioolwater_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwater_2025mrt.csv"
    covid_sterfte_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_covid_months_as_int.csv"
    oversterfte_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/refs/heads/main/input/oversterfte_mrt2025.csv"
    # https://www.rivm.nl/corona/actueel/weekcijfers
    # https://www.cbs.nl/nl-nl/reeksen/tijd/doodsoorzaken

    rioolwater = pd.read_csv(rioolwater_url, delimiter=';')
    covid_sterfte = pd.read_csv(covid_sterfte_url, delimiter=',')
    oversterfte = pd.read_csv(oversterfte_url, delimiter=',')


    # Process oversterfte data
    oversterfte['month'] = oversterfte.apply(calculate_month_oversterfte, axis=1)
    #oversterfte['oversterfte_cbs'] = oversterfte['Overledenen_z'] - oversterfte['verw_cbs_official']

    oversterfte['oversterfte_cbs'] = oversterfte['aantal_overlijdens'] - oversterfte['avg']
    grouped_oversterfte = oversterfte.groupby(['jaar_z', 'month'])['oversterfte_cbs'].mean().reset_index()

    # Process rioolwater data
    df_melted_rioolwater = rioolwater.melt(id_vars=['week'], var_name='year', value_name='value')
    df_melted_rioolwater['selected_year'] = df_melted_rioolwater.apply(select_year, axis=1)
    df_melted_rioolwater['month'] = df_melted_rioolwater.apply(calculate_month, axis=1)
    grouped_df_rioolwater = df_melted_rioolwater.groupby(['selected_year', 'month'])['value'].mean().reset_index()

    # Process covid sterfte data
    df_melted_covid_sterfte = covid_sterfte.melt(id_vars=['month'], var_name='year', value_name='value')
    final_df_covid_sterfte = df_melted_covid_sterfte[['year', 'month', 'value']]
    final_df_covid_sterfte['year'] = final_df_covid_sterfte['year'].astype(int)

    # Merge the DataFrames
    merged_df = pd.merge(grouped_df_rioolwater, final_df_covid_sterfte, left_on=['selected_year', 'month'], right_on=['year', 'month'], suffixes=('_rioolwater', '_covid_sterfte'))
    merged_df = pd.merge(merged_df, grouped_oversterfte, left_on=['selected_year', 'month'], right_on=['jaar_z', 'month'])

    make_lineplot(merged_df)
    make_scatterplot(merged_df, "value_rioolwater", "value_covid_sterfte")
    make_scatterplot(merged_df, "value_rioolwater", "oversterfte_cbs")

if __name__ == "__main__":
    main()
