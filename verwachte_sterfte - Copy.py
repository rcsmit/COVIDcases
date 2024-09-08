
import pandas as pd

import plotly.graph_objects as go
import eurostat
import platform
import streamlit as st
import plotly.express as px

import statsmodels.api as sm
from sklearn.metrics import r2_score

import pandas as pd
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor, LinearRegression

import numpy as np


try:
    st.set_page_config(layout="wide")
except:
    pass

@st.cache_data()
def get_bevolking(gevraagde_jaar, land):
    if land == "NL":
        if platform.processor() != "":
            file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_NL.csv"
        else: 
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
    elif land == "BE":
        # https://ec.europa.eu/eurostat/databrowser/view/demo_pjan__custom_12780094/default/table?lang=en
        if platform.processor() != "":
            file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_BE.csv"
        else: 
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_BE.csv"
    else:
        st.error(f"Error in land {land}")
    data = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
   
    data['leeftijd'] = data['leeftijd'].astype(int)
    
    # Define age bins and labels
    bins = list(range(0, 95, 5)) + [1000]  # [0, 5, 10, ..., 90, 1000]
    labels = [f'Y{i}-{i+4}' for i in range(0, 90, 5)] + ['90-999']


    # Create a new column for age bins
    data['age_group'] = pd.cut(data['leeftijd'], bins=bins, labels=labels, right=False)


    # Group by year, gender, and age_group and sum the counts
    grouped_data = data.groupby(['jaar', 'geslacht', 'age_group'], observed=False)['aantal'].sum().reset_index()

    # Save the resulting data to a new CSV file
    # grouped_data.to_csv('grouped_population_by_age_2010_2024.csv', index=False, sep=';')

    # print("Grouping complete and saved to grouped_population_by_age_2010_2024.csv")
    grouped_data["age_sex"] = grouped_data['age_group'].astype(str) +"_"+grouped_data['geslacht'].astype(str)
    
    
    for s in ["M", "F", "T"]:
        grouped_data.replace(f'Y0-4_{s}', f'Y_LT5_{s}', inplace=True)
        grouped_data.replace(f'90-999_{s}',f'Y_GE90_{s}', inplace=True)
   
    grouped_data_gevraagde_jaar = grouped_data[grouped_data["jaar"] ==gevraagde_jaar]

    return grouped_data, grouped_data_gevraagde_jaar


# Function to adjust the year and week
def adjust_year_week(row):
    if row['weeknr'] >= 40:
        adjusted_year = row['jaar'] + 1
        adjusted_week = row['weeknr'] - 39  # Weeks start from 1 after week 39
    else:
        adjusted_year = row['jaar']
        adjusted_week = row['weeknr'] + 13  # Weeks 1-39 shift to 40+
    
    return pd.Series([adjusted_year, adjusted_week])


# Function to determine the season
def determine_season(adjusted_weeknr):
    if adjusted_weeknr <= 26:
        return 'winter'
    else:
        return 'summer'

@st.cache_data()
def get_sterfte(gevraagde_jaar, land, split_season):
    """_summary_

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
          

    if land == "NL": 
        if platform.processor() != "":
            file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_NL.csv"
        else:
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_NL.csv"
    elif land == "BE":
        if platform.processor() != "":
            file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_BE.csv"
        else:
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_BE.csv"
    else:
        st.error(f"Error in land {land}")          
    df_ = pd.read_csv(
        file,
        delimiter=",",
            low_memory=False,
            )  
   
    df_=df_[df_["geo"] == land]
    
    df_["age_sex"] = df_["age"] + "_" +df_["sex"]
    df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["weeknr"] = (df_["TIME_PERIOD"].str[6:]).astype(int)


    if split_season:
        # Apply the function to adjust year and week number
        df_[['adjusted_jaar', 'adjusted_weeknr']] = df_.apply(adjust_year_week, axis=1)

        # Apply the function to create the season column
        df_['season'] = df_['adjusted_weeknr'].apply(determine_season)


  
    df_bevolking, df_bevolking_gevraagde_jaar = get_bevolking(gevraagde_jaar, land)
    if split_season:
        summed_per_year = df_.groupby(["jaar", 'age_sex', 'season'])['OBS_VALUE'].sum().reset_index()
    else:
        summed_per_year = df_.groupby(["jaar", 'age_sex'])['OBS_VALUE'].sum().reset_index()
  
    df__ = pd.merge(summed_per_year, df_bevolking, on=['jaar', 'age_sex'], how='outer')
    
    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["OBS_VALUE"].notna()]
    df__ = df__[df__["jaar"] != 2024]
    df__["per100k"] = df__["OBS_VALUE"]/(df__["aantal"]/100_000)
  
    
    return df__, df_bevolking_gevraagde_jaar

def bereken_verwachte_sterfte_kalenderjaar(countries, start, gevraagde_jaar, to_plot=False):
    """bereken de verwachte sterfte uitgaande van kalenderjaren. 

    Args:
        countries (_type_): _description_
        start (_type_): jaar waarin de lineaire regressie begint
        gevraagde_jaar (_type_): jaar waar je de verwachte sterfte van wilt weten
        to_plot (bool, optional): Plot maken?  Defaults to False.

    Returns:
        _type_: _description_
    """    
    # Get data for all selected countries and concatenate them
    df_list = []
    for land in countries:
        print (land)
        df, df_bevolking_gevraagde_jaar = get_sterfte(gevraagde_jaar,land, False)

        
        df["land"] = land  # Add a column to distinguish the countries
        df_list.append(df)
    
    df_combined = pd.concat(df_list)
    
 
    to_do = unique_values = df_combined["age_sex"].unique()
    labels = ['Y_LT5'] + [f'Y{i}-{i+4}' for i in range(5, 90, 5)] + ['Y_GE90']
    
   
    df_combined= df_combined[(df_combined["jaar"] >= start) &(df_combined["jaar"] <2020)]
    
    # Assuming df_combined is your dataframe
    df_combined['jaar'] = pd.to_numeric(df_combined['jaar'])

    # Initialize an empty list to store results
    results = []

    # Loop through each age_sex group
    for age_sex_group, group_data in df_combined.groupby('age_sex'):
        # Define X (independent variable) and y (dependent variable)
        #X = group_data[['jaar']]  # Reshape X as a 2D array
        X= group_data['jaar'].values.reshape(-1, 1)
        y = group_data['per100k']
        
        # # Create and fit the linear regression model
        # model = LinearRegression()
        # model.fit(X, y)
        
        # # Predict per100k for gevraagde_jaar
        # predicted_value = model.predict(np.array([[gevraagde_jaar]]))
        
       
        # Example using RANSAC with a linear regression model
        ransac = RANSACRegressor(LinearRegression())
        ransac.fit(X, y)
        predicted_value = ransac.predict(np.array([[gevraagde_jaar]]))
        
        # Append the result as a dictionary
        results.append({
            'age_sex': age_sex_group,
             'jaar': gevraagde_jaar,
            'per100k': predicted_value[0]
        })

        # Convert the results into a DataFrame
        # Add actual data points to the results to include in the graph
        for _, row in group_data.iterrows():
            results.append({
                'age_sex': age_sex_group,
                'jaar': row['jaar'],
                'per100k': row['per100k']
            })

    # Convert the results into a DataFrame
    predictions_gevraagde_jaar = pd.DataFrame(results)

    result_gevraagde_jaar = predictions_gevraagde_jaar[predictions_gevraagde_jaar["jaar"] == gevraagde_jaar]
      
    endresult_gevraagde_jaar=   pd.merge(result_gevraagde_jaar, df_bevolking_gevraagde_jaar, on=['age_sex'], how='outer') 
    endresult_gevraagde_jaar = endresult_gevraagde_jaar[endresult_gevraagde_jaar["geslacht"] != "T"]
    endresult_gevraagde_jaar["aantal_overleden_voorspelling"] = round( endresult_gevraagde_jaar["per100k"] * endresult_gevraagde_jaar["aantal"] / 100_000,1)


    if (gevraagde_jaar==2024) & (start ==2015):
        to_plot = True
    else:
        to_plot = False

    if to_plot:
        # Plot the results with Plotly
        fig = px.scatter(predictions_gevraagde_jaar, x='jaar', y='per100k', color='age_sex',
                    title=f'Linear Regression from {start} - Predictions for {gevraagde_jaar} by Age and Sex',
                    labels={'jaar': 'Year', 'per100k': 'per100k'},
                    trendline= "ols"
                    )
        # Update trendline traces to have a lighter color (reduce opacity)
        fig.update_traces(selector=dict(mode='lines'), line=dict(width=2, color='rgba(0,0,0,0.2)'))

        # Show the plot
        st.plotly_chart(fig)
        st.write(endresult_gevraagde_jaar)
    # Show the predicted values for gevraagde_jaar

    result_gevraagde_jaar = predictions_gevraagde_jaar[predictions_gevraagde_jaar["jaar"] == gevraagde_jaar]
      
    endresult_gevraagde_jaar=   pd.merge(result_gevraagde_jaar, df_bevolking_gevraagde_jaar, on=['age_sex'], how='outer') 
    endresult_gevraagde_jaar = endresult_gevraagde_jaar[endresult_gevraagde_jaar["geslacht"] != "T"]
    endresult_gevraagde_jaar["aantal_overleden_voorspelling"] = round( endresult_gevraagde_jaar["per100k"] * endresult_gevraagde_jaar["aantal"] / 100_000,1)

    verw_overleden = int(endresult_gevraagde_jaar["aantal_overleden_voorspelling"].sum())
    bevolkingsgrootte = df_bevolking_gevraagde_jaar["aantal"].sum()  # delen door 2 ivm de T-waardes
    #st.info(f"Totaal aantal verwachte overledenen {gevraagde_jaar} = **{verw_overleden}**  / Bevolkingsgrootte {bevolkingsgrootte}")
    return verw_overleden, bevolkingsgrootte

    

def bereken_verwachte_sterfte_seizoen(countries, start, gevraagde_jaar,  to_plot=False):
    """bereken de verwachte sterfte uitgaande van seizoenen. Het jaar loopt van week 40 van j-1 tot week 39. 

    Args:
        countries (_type_): _description_
        start (_type_): jaar waarin de lineaire regressie begint
        gevraagde_jaar (_type_): jaar waar je de verwachte sterfte van wilt weten
        to_plot (bool, optional): Plot maken?  Defaults to False.

    Returns:
        _type_: _description_
    """    
    # Get data for all selected countries and concatenate them
    df_list = []
    for land in countries:
        df, df_bevolking_gevraagde_jaar = get_sterfte(gevraagde_jaar, land, True)

        df["land"] = land  # Add a column to distinguish the countries
        df_list.append(df)
    
    df_combined = pd.concat(df_list)
    
    # Filter the data within the required time range
    df_combined = df_combined[(df_combined["jaar"] >= start) & (df_combined["jaar"] < 2020)]
   
    
    
 
    # Initialize an empty list to store results
    results = []
    
    # Loop through each group of age_sex and season
    for (age_sex_group, season), group_data in df_combined.groupby(['age_sex', 'season']):
        # Define X (independent variable) and y (dependent variable)
        X = group_data['jaar'].values.reshape(-1, 1)
        y = group_data['per100k']
        
        # # Create and fit the linear regression model
        # model = LinearRegression()
        # model.fit(X, y)
        
        # # Predict per100k for gevraagde_jaar
        # predicted_value = model.predict(np.array([[gevraagde_jaar]]))
        
       
        # Example using RANSAC with a linear regression model
        ransac = RANSACRegressor(LinearRegression())
        ransac.fit(X, y)
        predicted_value = ransac.predict(np.array([[gevraagde_jaar]]))
        
        # Append the result as a dictionary
        results.append({
            'age_sex': age_sex_group,
            'season': season,
            'jaar': gevraagde_jaar,
            'per100k': predicted_value[0]
        })

        # Add actual data points to the results to include in the graph
        for _, row in group_data.iterrows():
            results.append({
                'age_sex': age_sex_group,
                'season': season,
                'jaar': row['jaar'],
                'per100k': row['per100k']
            })

    # Convert the results into a DataFrame
    predictions_gevraagde_jaar = pd.DataFrame(results)

    # Merge with population data and calculate predictions
    result_gevraagde_jaar = predictions_gevraagde_jaar[predictions_gevraagde_jaar["jaar"] == gevraagde_jaar]
    endresult_gevraagde_jaar = pd.merge(result_gevraagde_jaar, df_bevolking_gevraagde_jaar, on=['age_sex'], how='outer')
    endresult_gevraagde_jaar = endresult_gevraagde_jaar[endresult_gevraagde_jaar["geslacht"] != "T"]
    
    # Calculate expected number of deaths
    endresult_gevraagde_jaar["aantal_overleden_voorspelling"] = round(
        endresult_gevraagde_jaar["per100k"] * endresult_gevraagde_jaar["aantal"] / 100_000, 1
    )

    # Calculate the sum for winter and summer separately
    winter_deaths = endresult_gevraagde_jaar[endresult_gevraagde_jaar["season"] == "winter"][
        "aantal_overleden_voorspelling"].sum()
    summer_deaths = endresult_gevraagde_jaar[endresult_gevraagde_jaar["season"] == "summer"][
        "aantal_overleden_voorspelling"].sum()
    
    # Sum total deaths for the requested year
    verw_overleden = int(winter_deaths + summer_deaths)
    bevolkingsgrootte = df_bevolking_gevraagde_jaar["aantal"].sum() / 2  # divide by 2 due to 'T' values

    # Plot the results if conditions are met
    if (gevraagde_jaar == 2024) & (start == 2015):
        to_plot = True
    else:
        to_plot = False
    if to_plot:
        # Plot the results with Plotly
        predictions_gevraagde_jaar["age_sex_season"] = predictions_gevraagde_jaar['age_sex']+"_"+predictions_gevraagde_jaar["season"]
        fig = px.scatter(predictions_gevraagde_jaar, x='jaar', y='per100k', color='age_sex_season',
                         title=f'Linear Regression from {start} - Predictions for {gevraagde_jaar} by Age and Sex',
                         labels={'jaar': 'Year', 'per100k': 'per100k'},
                         trendline="ols")
        
        # Update trendline traces to have a lighter color (reduce opacity)
        fig.update_traces(selector=dict(mode='lines'), line=dict(width=2, color='rgba(0,0,0,0.2)'))

        # Show the plot
        st.plotly_chart(fig)
        st.write(endresult_gevraagde_jaar)
 
    # Return the total expected deaths and population size
    return verw_overleden, bevolkingsgrootte


def main():
    st.title("Verwachte sterfte voor 2024 berekenen")
    st.info("""
        We voorspellen het aantal overlijdens voor 2024 met behulp van een lineaire regressie op basis van de overlijdensgegevens tussen 2015 en 2019. Dit doen we voor verschillende leeftijds- en geslachtsgroepen. De aanpak is geïnspireerd door Bonne Klok, die een vergelijkbare analyse heeft gedeeld op Twitter.

       Dit getal gebruiken we om de correctiefactor te berekenen, waarmee we de baseline corrigeren voor verbeterde gezondheid en veranderingen in de leeftijdsopbouw.
       
       Je kunt de tweet van Bonne Klok hier bekijken: https://twitter.com/BonneKlok/status/1832333262586323385.""")
    # Let the user select one or both countries
    countries = ["NL"] # st.multiselect("land [NL | BE]", ["NL", "BE"], default=["NL"])
    # start = st.number_input("Startjaar voor lineaire regressie", 2000, 2020, 2015)
    
    #gevraagde_jaar = st.number_input("Verwachting bereken voor jaar", 2021,2030,2024)
    

    # start_jaren = [2000,2005, 2010,2015]
    gevraagde_jaren= [2020,2021,2022,2023,2024]
    
    start_jaren = [2015]
    #gevraagde_jaren= [2024]
    

    # Maak een lege DataFrame met de gevraagde jaren als index en startjaren als kolommen
    tabel = pd.DataFrame(index=gevraagde_jaren, columns=start_jaren)
    col1,col2 = st.columns(2)
    with col1:
       

        # Vul de DataFrame met verwachte overlijdenscijfers
        for start in start_jaren:
            for gevraagde_jaar in gevraagde_jaren:
                verw_overleden, bevolkingsgrootte = bereken_verwachte_sterfte_kalenderjaar(countries, start, gevraagde_jaar)
                tabel.loc[gevraagde_jaar, start] = verw_overleden

        st.write(tabel)
    with col2:
        # Vul de DataFrame met verwachte overlijdenscijfers
        for start in start_jaren:
            for gevraagde_jaar in gevraagde_jaren:
                verw_overleden, bevolkingsgrootte = bereken_verwachte_sterfte_seizoen(countries, start, gevraagde_jaar)
                tabel.loc[gevraagde_jaar, start] = verw_overleden

        st.write(tabel)

    st.info("""Door CBS gebruikt:
             2020: 153402  |
            2021: 154887 |
            2022: 155494 |
            2023: 156666 
           
            """)
    st.subheader("Databronnen")
    st.info("Bevolkingsgrootte NL: https://opendata.cbs.nl/#/CBS/nl/dataset/03759ned/table?dl=39E0B")
    st.info("Sterfte: https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")

    st.info("Code: https://github.com/rcsmit/COVIDcases/blob/main/verwachte_sterfte.py")

if __name__ == "__main__":

    main()
