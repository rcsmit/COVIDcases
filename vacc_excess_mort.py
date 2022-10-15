import pandas as pd
import datetime
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import r2_score
import plotly.express as px
def left(s, amount):
    return s[:amount]

def right(s, amount):
    return s[-amount:]


def mid(s, offset, amount):
    return s[offset-1:offset+amount-1]

def getMonth(year: int, week: int) -> int:
    """Return the month number in the given week in the given year."""
    return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w").month

def make_barchart(df_temp, what_to_show_l):
    fig1xy = px.bar(df_temp, x="ReportingCountry", y=what_to_show_l,
                    hover_name="ReportingCountry")
    st.plotly_chart(fig1xy, use_container_width=True)

def make_scatterplot(df_temp, what_to_show_l, what_to_show_r):
    """Scatterplot maken
    """

    x_ = np.array(df_temp[what_to_show_l])
    y_ = np.array(df_temp[what_to_show_r])
    #obtain m (slope) and b(intercept) of linear regression line
    idx = np.isfinite(x_) & np.isfinite(y_)
    m, b = np.polyfit(x_[idx], y_[idx], 1)
    model = np.polyfit(x_[idx], y_[idx], 1)

    predict = np.poly1d(model)
    r2 = r2_score  (y_[idx], predict(x_[idx]))


    fig1xy = px.scatter(df_temp, x=what_to_show_l, y=what_to_show_r, size='Population', trendline="ols",
                    hover_name="ReportingCountry")


    #add linear regression line to scatterplot


    correlation_sp = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='spearman'), 3) #gebruikt door HJ Westeneng, rangcorrelatie
    correlation_p = round(df_temp[what_to_show_l].corr(df_temp[what_to_show_r], method='pearson'), 3)

   
    title_scatter = (f"{what_to_show_l} -  {what_to_show_r}<br>Correlation spearman = {correlation_sp} - Correlation pearson = {correlation_p}<br>y = {round(m,2)}*x + {round(b,2)} | r2 = {round(r2,4)}")

    fig1xy.update_layout(
        title=dict(
            text=title_scatter,
            x=0.5,
            y=0.95,
            font=dict(
                family="Arial",
                size=14,
                color='#000000'
            )
        ),
        xaxis_title=what_to_show_l,
        yaxis_title=what_to_show_r,
        font=dict(
            family="Courier New, Monospace",
            size=12,
            color='#000000'
        )
    )


    st.plotly_chart(fig1xy, use_container_width=True)

# https://ec.europa.eu/eurostat/databrowser/view/DEMO_MEXRT__custom_3576968/default/table?lang=en
# https://www.ecdc.europa.eu/en/publications-data/data-covid-19-vaccination-eu-eea     
# 
# Numbers do not correspond with https://vaccinetracker.ecdc.europa.eu/public/extensions/COVID-19/vaccine-tracker.html#uptake-tab       
vacc_url = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\vaccinations_europe.csv"
excess_mort_url = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\excess_mortality.csv"

df_vacc = pd.read_csv(vacc_url, delimiter=',')
df_excess_mort = pd.read_csv(excess_mort_url, delimiter=',')


df_excess_mort["Year"] = (df_excess_mort["TIME_PERIOD"].str[:4]).astype(int)
df_excess_mort["Month"] = (df_excess_mort["TIME_PERIOD"].str[5:]).astype(int)


df_vacc = df_vacc[df_vacc['TargetGroup'] == "ALL"]
df_vacc["Year"] = (df_vacc["YearWeekISO"].str[:4]).astype(int)
df_vacc["Week"] = (df_vacc["YearWeekISO"].str[6:]).astype(int)
df_vacc['Month'] = df_vacc.apply(lambda x: getMonth(x['Year'], x['Week']),axis=1)
df_vacc = (df_vacc.groupby(['Year', 'Month', 'ReportingCountry'], as_index=False)
       .agg({
            'Denominator':'mean',
            'NumberDosesReceived':'sum',
            'NumberDosesExported':'sum',
            'FirstDose':'sum',
            'FirstDoseRefused':'sum',
            'SecondDose':'sum',
            'DoseAdditional1':'sum',
            'DoseAdditional2':'sum',
            'UnknownDose':'sum',
            'Population':'mean',}))
                

df  = pd.merge(
                df_vacc, df_excess_mort, how="inner", left_on=["ReportingCountry","Year", "Month"], right_on=["geo","Year", "Month"]
            )

print (df)
countries = df.ReportingCountry.unique()
list_values =  ['NumberDosesReceived', 'NumberDosesExported', 'FirstDose', 'FirstDoseRefused', 'SecondDose', 'DoseAdditional1', 'DoseAdditional2', 'UnknownDose']
list_values_cum,list_values_cum_per_capita =[],[]

for l in list_values:
    new_column_name_cum = l + '_cum'
    df[new_column_name_cum] = df.groupby(['ReportingCountry'])[l].cumsum()
 
    new_column_cum_capita = l + '_cum_per_capita'
    df[new_column_cum_capita] = df[new_column_name_cum]/df["Population"]
    list_values_cum.append(new_column_name_cum)
    list_values_cum_per_capita.append(new_column_cum_capita)

#list_values_per_capita=  ['NumberDosesReceived_per_capita', 'NumberDosesExported_per_capita', 'FirstDose_per_capita', 'FirstDoseRefused_per_capita', 'SecondDose_per_capita', 'DoseAdditional1_per_capita', 'DoseAdditional2_per_capita', 'UnknownDose']
menu = list_values_cum + list_values_cum_per_capita
print (countries)

what_to_show_l =   st.sidebar.selectbox("What to show x-ax", menu, index=0)
y_ =  st.sidebar.selectbox("Year", [2020,2021,2022], index=2)
m_ =  st.sidebar.number_input("month",1,12,7)

df = df[((df["Year"] == y_) & (df["Month"] == m_))]
if len(df) == 0:
    st.error("No data available")
    st.stop()
print (df)
make_scatterplot(df, what_to_show_l, "OBS_VALUE")

make_barchart(df, what_to_show_l)
