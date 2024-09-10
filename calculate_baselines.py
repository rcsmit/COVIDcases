import pandas as pd
import streamlit as st
import numpy as np

import datetime
import plotly.express as px
import statsmodels.api as sm

import statsmodels.formula.api as smf

import plotly.graph_objs as go
import plotly.express as px

# Later aan te vullen met andere methodes
# https://www.zonmw.nl/sites/zonmw/files/2023-04/Rapportage-oversterfte.pdf
# In 39 studies werd een statistisch model gebruikt om de verwachte oversterfte te modelleren (Tabel 3
# en Bijlage 5). Vaak werd hiervoor een statistisch model met een Poisson-verdeling gebruikt (15 studies),
# hoewel ook binnen de groep Poisson-modellen verschillende keuzes werden gemaakt. Meestal kozen
# onderzoekers voor een overdispersed (quasi-)Poisson-model waarbij er ruimte is voor meer variatie
# tussen de datapunten dan het model normaal toestaat. Een andere vaak gebruikte methode was het
# autoregressive integrated moving average (ARIMA) model of het seasonal ARIMA (SARIMA) model (7
# studies). Dit is een methode die gebruikt wordt om voorspellingen te doen op basis van een dataset
# waarbij je metingen hebt op verschillende momenten in de tijd (time series data). Het SARIMA-model
# bevat daarnaast ook een seizoenscomponent. Regelmatig werd er gekozen voor lineare regressie (6
# studies) en ook andere generalized linear models met verschillende verdelingen werden gebruikt.


# This function fits a Poisson and a quasi-Poisson model to observed death data from 2015 to 2019, and then predicts 
# expected deaths for the year 2020 based on the fitted models. It calculates excess deaths by comparing observed deaths 
# to the expected deaths from both models.


# In R, the main difference between a Poisson regression model and a quasi-Poisson regression model 
# is how they handle overdispersion. Overdispersion occurs when the variance of the response variable
#  is greater than the mean, which is a common issue in count data. 

# https://chatgpt.com/c/7d57f4eb-20fe-47d6-9f47-0f48295eeff0

# 70895ned = https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1659307527578
# Overledenen; geslacht en leeftijd, per week

# Downloaden van tabeloverzicht
# toc = pd.DataFrame(cbsodata.get_table_list())
try:
    st.set_page_config(layout="wide")
except:
    pass

# TO SUPRESS
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
pd.options.mode.chained_assignment = None


def adjust_overledenen(df):
        """# Adjust "Overledenen_1" based on the week number
        # if week = 0, overledenen_l : add to week 52 of the year before
        # if week = 53: overleden_l : add to week 1 to the year after
        """

        for index, row in df.iterrows():
            if row["week"] == 0:
                previous_year = row["year"] - 1
                df.loc[
                    (df["year"] == previous_year) & (df["week"] == 52), "Overledenen_1"
                ] += row["Overledenen_1"]
            elif row["week"] == 53:
                next_year = row["year"] + 1
                df.loc[
                    (df["year"] == next_year) & (df["week"] == 1), "Overledenen_1"
                ] += row["Overledenen_1"]
        # Filter out the rows where week is 0 or 53 after adjusting
        df = df[~df["week"].isin([0, 53])]
        return df


def get_sterfte_data_fixed():
    #C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_per_week.csv
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_per_week.csv"
    df_ = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )   
    df_ = df_.rename(columns={"weeknr": "week"})
    df_ = df_.rename(columns={"jaar": "year"})
    df_ = df_[df_['year'] >2014]
    return df_



def do_poisson_original(df):

    """
    This function fits a Poisson and a quasi-Poisson model to observed death data from 2015 to 2019, and then predicts 
    expected deaths for the year 2020 based on the fitted models. It calculates excess deaths by comparing observed deaths 
    to the expected deaths from both models.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'jaar': The year of observation.
        - 'week': The week number of the year.
        - 'observed_deaths': The observed number of deaths.

    Returns:
    --------
    None
        The function does not return any values but prints out a DataFrame showing the week, year, observed deaths, 
        expected deaths from both models (Poisson and quasi-Poisson), and excess deaths calculated from both models 
        using Streamlit's st.write() function.
    
    Notes:
    ------
    - The function adds an intercept column to the DataFrame for modeling purposes.
    - The quasi-Poisson model is implemented using the Tweedie family with a power parameter set to 1.
    - The results are printed for inspection within the Streamlit application using `st.write`.
    """


    # Voeg een constante toe aan het model (intercept)
    df['intercept'] = 1

    # Selecteer de jaren 2015-2019 voor het trainen van het model
    train_data = df[df['jaar'] < 2020]
    st.write(train_data)
    # Poisson model
    poisson_model = sm.GLM(train_data['observed_deaths'], train_data[['intercept', 'week']], family=sm.families.Poisson())
    poisson_results = poisson_model.fit()

    # Voor een quasi-Poisson model gebruik je de Tweedie familie met power=1
    quasi_poisson_model = sm.GLM(train_data['observed_deaths'], train_data[['intercept', 'week']], family=sm.families.Tweedie(var_power=1))
    quasi_poisson_results = quasi_poisson_model.fit()

    # Voorspellingen voor 2020 maken
    predict_data = df[df['year'] == 2020]
    predict_data['expected_deaths'] = poisson_results.predict(predict_data[['intercept', 'week']])
    # Voor quasi-Poisson model:
    predict_data['expected_deaths_quasi_poisson'] = quasi_poisson_results.predict(predict_data[['intercept', 'week']])

    predict_data['excess_deaths'] = predict_data['observed_deaths'] - predict_data['expected_deaths']
    predict_data['excess_deaths_quasi_poisson'] = predict_data['observed_deaths'] - predict_data['expected_deaths_quasi_poisson']

    # Bekijk de resultaten
    st.write(predict_data[['week', 'year', 'observed_deaths', 'expected_deaths', 'excess_deaths', "expected_deaths_quasi_poisson", "excess_deaths_quasi_poisson"]])


    # Plotly visualisatie
    fig = px.line(
        predict_data,
        x='week',
        y=['observed_deaths', 'expected_deaths', 'expected_deaths_quasi_poisson'],
        labels={
            'week': 'Week',
            'value': 'Number of Deaths',
            'variable': 'Death Type'
        },
        title='Observed vs Expected Deaths (Poisson and Quasi-Poisson Models) - 2020'
    )
    
    fig.add_scatter(
        x=predict_data['week'], 
        y=predict_data['excess_deaths'], 
        mode='lines', 
        name='Excess Deaths (Poisson)',
        line=dict(dash='dash', color='red')
    )
    
    fig.add_scatter(
        x=predict_data['week'], 
        y=predict_data['excess_deaths_quasi_poisson'], 
        mode='lines', 
        name='Excess Deaths (Quasi-Poisson)',
        line=dict(dash='dash', color='blue')
    )

    st.plotly_chart(fig)


def do_poisson(df):

    """
    This function fits a Poisson and a quasi-Poisson model to observed death data from 2015 to 2019, and then predicts 
    expected deaths for the year 2020 based on the fitted models. It calculates excess deaths by comparing observed deaths 
    to the expected deaths from both models.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing the following columns:
        - 'jaar': The year of observation.
        - 'week': The week number of the year.
        - 'observed_deaths': The observed number of deaths.

    Returns:
    --------
    None
        The function does not return any values but prints out a DataFrame showing the week, year, observed deaths, 
        expected deaths from both models (Poisson and quasi-Poisson), and excess deaths calculated from both models 
        using Streamlit's st.write() function.
    
    Notes:
    ------
    - The function adds an intercept column to the DataFrame for modeling purposes.
    - The quasi-Poisson model is implemented using the Tweedie family with a power parameter set to 1.
    - The results are printed for inspection within the Streamlit application using `st.write`.
    """


    # Voeg een constante toe aan het model (intercept)
    df['intercept'] = 1

    # Selecteer de jaren 2015-2019 voor het trainen van het model
    train_data = df[df['jaar'] < 2020]
    data_2020 =  df[df['jaar'] == 2020]
    st.write(train_data)
    # Poisson model


    # Fit a quasi-Poisson model on the data from 2015 to 2019
    model = smf.poisson('observed_deaths ~ week + C(year)', data=df).fit(scale='X2')

    # Prepare data for 2020
    weeks_2020 = pd.DataFrame({
        'week': np.arange(1, 53),
        'year': 2020
    })

    # Predict expected deaths for 2020
    weeks_2020['expected_deaths'] = model.predict(weeks_2020)

        
    # Merge the predicted and actual data for 2020
    df_merged_2020 = pd.merge(weeks_2020, data_2020, on=['week', 'year'])

    # Calculate excess mortality
    df_merged_2020['excess_deaths'] = df_merged_2020['Overledenen_1'] - df_merged_2020['expected_deaths']

    
    st.write(df_merged_2020)


    fig = go.Figure()

    # Add observed deaths trace
    fig.add_trace(go.Scatter(
        x=df_merged_2020['week'],
        y=df_merged_2020['Overledenen_1'],
        mode='lines+markers',
        name='Observed Deaths',
        line=dict(color='blue')
    ))

    # Add expected deaths trace
    fig.add_trace(go.Scatter(
        x=df_merged_2020['week'],
        y=df_merged_2020['expected_deaths'],
        mode='lines+markers',
        name='Expected Deaths',
        line=dict(color='green')
    ))

    # Add excess deaths trace
    fig.add_trace(go.Scatter(
        x=df_merged_2020['week'],
        y=df_merged_2020['excess_deaths'],
        mode='lines+markers',
        name='Excess Deaths',
        line=dict(color='red')
    ))

    # Update layout
    fig.update_layout(
        title='Observed vs Expected vs Excess Deaths in 2020',
        xaxis_title='Week',
        yaxis_title='Number of Deaths',
        legend_title='Legend',
        template='plotly_white'
    )

    # Show the figure
    st.plotly_chart(fig)
def main():
    st.info("""This function fits a Poisson and a quasi-Poisson model to observed death data from 2015 to 2019, and then predicts 
    expected deaths for the year 2020 based on the fitted models. It calculates excess deaths by comparing observed deaths 
    to the expected deaths from both models.
    """)
    st.warning("Results are strange, they dont follow the seasons, but it is just a straight line")
    serienames_ = [
        "m_v_0_999",
        "m_v_0_64",
        "m_v_65_79",
        "m_v_80_999",
        "m_0_999",
        "m_0_64",
        "m_65_79",
        "m_80_999",
        "v_0_999",
        "v_0_64",
        "v_65_79",
        "v_80_999",
    ]

    series_name = st.sidebar.selectbox("Leeftijden", serienames_, 0)
    
  
    df_data = get_sterfte_data_fixed()
    
    df_data = df_data.sort_values(by=["year", "week"])
    df_data["observed_deaths"]= df_data[f"totaal_{series_name}"]
    df_data["Overledenen_1"] = df_data[f"totaal_{series_name}"]
    df_data["jaar"] = df_data["year"]
    df_data["weeknr"] = (
        df_data["jaar"].astype(str) + "_" + df_data["week"].astype(str).str.zfill(2)
    )
   
    df_data = adjust_overledenen(df_data)
    do_poisson(df_data)
    
if __name__ == "__main__":
    import datetime

    print(
        f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------"
    )

    main()
