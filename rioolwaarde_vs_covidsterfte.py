import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go

# Define a function to calculate the month from the selected year and week
def calculate_month(row):
    first_day_of_year = datetime(row['selected_year'], 1, 1)
    first_day_of_week = first_day_of_year + timedelta(weeks=row['week'] - 1)
    return first_day_of_week.month

def calculate_month_oversterfte(row):
    first_day_of_year = datetime(row['jaar_z'], 1, 1)
    first_day_of_week = first_day_of_year + timedelta(weeks=row['week_z'] - 1)
    return first_day_of_week.month

# Define a function to select the appropriate year based on the week
def select_year(row):
    if row['week'] >= 40:
        return int(row['year'].split('/')[0])
    else:
        return int(row['year'].split('/')[1])

# Load the CSV files
rioolwater_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwater_2025mrt.csv"
covid_sterfte_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_covid_months_as_int.csv"
oversterfte_url =  "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overl_cbs_vs_rivm.csv"
#oversterfte_url=r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\oversterfte_mrt2025.csv"

rioolwater = pd.read_csv(rioolwater_url, delimiter=';')
covid_sterfte = pd.read_csv(covid_sterfte_url, delimiter=',')
oversterfte = pd.read_csv(oversterfte_url, delimiter=';')
# oversterfte= oversterfte[["jaar_z","week_z","datum","Overledenen_z","verw_cbs_official","aantal_overlijdens_z"]]
# oversterfte["jaar_z"]=oversterfte["jaar_z"].astype(int)
# oversterfte["week_z"]=oversterfte["week_z"].astype(int)

print (oversterfte.dtypes)
print (covid_sterfte)
# Process oversterfte data
oversterfte['month'] = oversterfte.apply(calculate_month_oversterfte, axis=1)
oversterfte['oversterfte_cbs'] = oversterfte['Overledenen_z'] - oversterfte['verw_cbs_official']
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

# # Add the oversterfte data to the plot
# fig.add_trace(go.Scatter(x=merged_df['selected_year'].astype(str) + '-' + merged_df['month'].astype(str),
#                          y=merged_df['oversterfte_cbs'],
#                          mode='lines',
#                          name='Oversterfte',
#                          yaxis='y3'))

# Update the layout to include three y-axes
fig.update_layout(
    title='Rioolwater vs Covid Sterfte vs Oversterfte',
    xaxis_title='Year-Month',
    yaxis=dict(
        title='Rioolwater Value',
        titlefont=dict(
            color='#1f77b4'
        ),
        tickfont=dict(
            color='#1f77b4'
        )
    ),
    # yaxis2=dict(
    #     title='Covid Sterfte Value',
    #     # titlefont=dict(
    #     #     color='#ff7f0e'
    #     # ),
    #     # tickfont=dict(
    #     #     color='#ff7f0e'
    #     # ),
    #     overlaying='y',
    #     side='right'
    # ),
    # yaxis3=dict(
    #     title='Oversterfte Value',
    #     titlefont=dict(
    #         color='#2ca02c'
    #     ),
    #     tickfont=dict(
    #         color='#2ca02c'
    #     ),
    #     overlaying='y',
    #     side='right',
    #     anchor='free',
    #     position=0.85
    # )
)

# Display the plot
st.plotly_chart(fig)