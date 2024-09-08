import pandas as pd

import plotly.graph_objects as go
import eurostat
import platform
import streamlit as st
import plotly.express as px

import statsmodels.api as sm
from sklearn.metrics import r2_score

import pandas as pd

# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor

import numpy as np


try:
    st.set_page_config(layout="wide")
except:
    pass  # Silently ignore if the page configuration has already been set


def get_bevolking(gevraagde_jaar: int, land: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve population data for the specified year and country, group by age, gender, and age groups, 
    and return a summary of the population count.

    Args:
        gevraagde_jaar (int): The year for which population data is requested.
        land (str): The country for which data is requested. Options are "NL" (Netherlands) or "BE" (Belgium).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - grouped_data: Population data grouped by age, gender, and age group.
            - grouped_data_gevraagde_jaar: Population data for the requested year.
    """
    # TODO : Download from https://opendata.cbs.nl/statline/#/CBS/nl/dataset/7461bev/table?ts=1696835440118 with cbsodata

    if land == "NL":
        if platform.processor() != "":
            file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_NL.csv"
        else:
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
    elif land == "BE":
        # https://ec.europa.eu/eurostat/databrowser/view/demo_pjan__custom_12780094/default/table?lang=en
        if platform.processor() != "":
            file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_BE.csv"
        else:
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_BE.csv"
    else:
        st.error(f"Error in land {land}")
    data = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )

    data["leeftijd"] = data["leeftijd"].astype(int)

    # Define age bins and labels
    bins = list(range(0, 95, 5)) + [1000]  # [0, 5, 10, ..., 90, 1000]
    labels = [f"Y{i}-{i+4}" for i in range(0, 90, 5)] + ["90-999"]

    # Create a new column for age bins
    data["age_group"] = pd.cut(data["leeftijd"], bins=bins, labels=labels, right=False)

    # Group by year, gender, and age_group and sum the counts
    grouped_data = (
        data.groupby(["jaar", "geslacht", "age_group"], observed=False)["aantal"]
        .sum()
        .reset_index()
    )

    # Save the resulting data to a new CSV file
    # grouped_data.to_csv('grouped_population_by_age_2010_2024.csv', index=False, sep=';')

    # print("Grouping complete and saved to grouped_population_by_age_2010_2024.csv")
    grouped_data["age_sex"] = (
        grouped_data["age_group"].astype(str)
        + "_"
        + grouped_data["geslacht"].astype(str)
    )

    for s in ["M", "F", "T"]:
        grouped_data.replace(f"Y0-4_{s}", f"Y_LT5_{s}", inplace=True)
        grouped_data.replace(f"90-999_{s}", f"Y_GE90_{s}", inplace=True)

    grouped_data_gevraagde_jaar = grouped_data[grouped_data["jaar"] == gevraagde_jaar]

    return grouped_data, grouped_data_gevraagde_jaar


def adjust_year_week(row: pd.Series) -> pd.Series:
    """
    Adjust the year and week number based on seasonal boundaries. 

    If the week is greater than or equal to 40, it is considered part of the next year, 
    otherwise it is considered part of the current year.

    Args:
        row (pd.Series): A row from the DataFrame containing the "weeknr" and "jaar" columns.

    Returns:
        pd.Series: The adjusted year and week number as a series.
    """
    if row["weeknr"] >= 40:
        adjusted_year = row["jaar"] + 1
        adjusted_week = row["weeknr"] - 39  # Weeks start from 1 after week 39
    else:
        adjusted_year = row["jaar"]
        adjusted_week = row["weeknr"] + 13  # Weeks 1-39 shift to 40+

    return pd.Series([adjusted_year, adjusted_week])


def determine_season(adjusted_weeknr: int) -> str:
    """
    Determine the season (summer or winter) based on the adjusted week number.

    Args:
        adjusted_weeknr (int): The adjusted week number (after week 39, the year changes).

    Returns:
        str: The season ("summer" or "winter") based on the week number.
    """
    if adjusted_weeknr <= 26:
        return "winter"
    else:
        return "summer"



@st.cache_data()
def get_sterfte(gevraagde_jaar: int, land: str, split_season: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve mortality data for a specific country and year, merge with population data, and calculate 
    deaths per 100k inhabitants.

    Args:
        gevraagde_jaar (int): The year for which mortality data is requested.
        land (str): The country code ("NL" for Netherlands, "BE" for Belgium).
        split_season (bool): Whether to split the data by season (winter/summer).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - df__: Merged DataFrame containing mortality, population data, and deaths per 100k.
            - df_bevolking_gevraagde_jaar: Population data for the requested year.
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    # https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true

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

    df_ = df_[df_["geo"] == land]

    df_["age_sex"] = df_["age"] + "_" + df_["sex"]
    df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["weeknr"] = (df_["TIME_PERIOD"].str[6:]).astype(int)

    if split_season:
        # Apply the function to adjust year and week number
        df_[["adjusted_jaar", "adjusted_weeknr"]] = df_.apply(adjust_year_week, axis=1)

        # Apply the function to create the season column
        df_["season"] = df_["adjusted_weeknr"].apply(determine_season)
    else:
        df_["season"] = "all_year"

    df_bevolking, df_bevolking_gevraagde_jaar = get_bevolking(gevraagde_jaar, land)
    summed_per_year = (
        df_.groupby(["jaar", "age_sex", "season"])["OBS_VALUE"].sum().reset_index()
    )

    df__ = pd.merge(summed_per_year, df_bevolking, on=["jaar", "age_sex"], how="outer")

    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["OBS_VALUE"].notna()]
    df__ = df__[df__["jaar"] != 2024]
    df__["per100k"] = df__["OBS_VALUE"] / (df__["aantal"] / 100_000)

    return df__, df_bevolking_gevraagde_jaar

def perform_lineair_regression(group_data: pd.DataFrame, gevraagde_jaar: int, regresion_type: str = "ols") -> np.ndarray:
    """
    Perform linear regression on the group data and predict mortality rates for the requested year.

    Args:
        group_data (pd.DataFrame): The group data (mortality and population).
        gevraagde_jaar (int): The year for which mortality is predicted.
        regresion_type (str, optional): The type of regression to use ("ols", "huber", or "ransac"). Defaults to "huber".

    Returns:
        np.ndarray: Predicted mortality rate per 100k for the requested year.

    RANSAC: Best for extreme outliers, but slow.
    Huber: Faster and more balanced, ideal for mild to moderate outliers.
    https://www.linkedin.com/pulse/tale-two-detectives-huber-vs-ransac-sravya-kamavarapu-anioc/
    """
    # Define X (independent variable) and y (dependent variable)
    X = group_data["jaar"].values.reshape(-1, 1)
    y = group_data["per100k"]

    if regresion_type == "ols":
        # Create and fit the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict per100k for gevraagde_jaar
        predicted_value = model.predict(np.array([[gevraagde_jaar]]))

    elif regresion_type == "huber":
        huber = HuberRegressor()
        huber.fit(X, y)
        predicted_value = huber.predict(np.array([[gevraagde_jaar]]))
    elif regresion_type == "ransac":
        # Example using RANSAC with a linear regression model
        ransac = RANSACRegressor(LinearRegression())
        ransac.fit(X, y)
        predicted_value = ransac.predict(np.array([[gevraagde_jaar]]))
    else:
        st.error(f"Error in regression type {regresion_type}")
        st.stop()
    return predicted_value


def get_df_combined(
    countries: list[str], start: int, gevraagde_jaar: int, split_season: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine mortality and population data for multiple countries, filtered by start year, 
    and prepare it for further analysis.

    Args:
        countries (list[str]): List of country codes ("NL", "BE").
        start (int): The start year for the linear regression.
        gevraagde_jaar (int): The requested year for prediction.
        split_season (bool): Whether to split the data by season.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - df_combined: Combined mortality and population data for all countries.
            - df_bevolking_gevraagde_jaar: Population data for the requested year.
    """
    
    df_list = []
    for land in countries:
        df, df_bevolking_gevraagde_jaar = get_sterfte(
            gevraagde_jaar, land, split_season
        )
        df["land"] = land  # Add a column to distinguish the countries
        df_list.append(df)

    df_combined = pd.concat(df_list)
    df_combined = df_combined[
        (df_combined["jaar"] >= start) & (df_combined["jaar"] < 2020)
    ]

    # Assuming df_combined is your dataframe
    df_combined["jaar"] = pd.to_numeric(df_combined["jaar"])

    return df_combined, df_bevolking_gevraagde_jaar

def make_plot(predictions_gevraagde_jaar, start, gevraagde_jaar):
    """Make a plot with the deaths in the past and the predicted value

    Args:
        predictions_gevraagde_jaar (_type_): _description_
    """    
    predictions_gevraagde_jaar["age_sex_season"] = (
            predictions_gevraagde_jaar["age_sex"]
            + "_"
            + predictions_gevraagde_jaar["season"]
        )
    fig = px.scatter(
        predictions_gevraagde_jaar,
        x="jaar",
        y="per100k",
        color="age_sex_season",
        title=f"Linear Regression from {start} - Predictions for {gevraagde_jaar} by Age and Sex",
        labels={"jaar": "Year", "per100k": "per100k"},
        trendline="ols",
    )

    # Update trendline traces to have a lighter color (reduce opacity)
    fig.update_traces(
        selector=dict(mode="lines"), line=dict(width=2, color="rgba(0,0,0,0.2)")
    )

    # Show the plot
    st.plotly_chart(fig)
def bereken_verwachte_sterfte(
    countries: list[str], start: int, gevraagde_jaar: int, regresion_type: str, split_season: bool, to_plot: bool = False
     ) -> tuple[float, float]:
    """
    Calculate the expected mortality for the requested year using linear regression.

    Args:
        countries (list[str]): List of country codes ("NL", "BE").
        start (int): The start year for the linear regression.
        gevraagde_jaar (int): The year for which the expected mortality is predicted.
        split_season (bool): Whether to split the data by season.
        regresion_type(string): Regression type
        to_plot (bool, optional): Whether to generate a plot. Defaults to False.

    Returns:
        tuple[float, float]: 
            - verw_overleden: The total expected number of deaths.
            - bevolkingsgrootte: The total population size for the requested year.
    """
    # Get data for all selected countries and concatenate them

    df_combined, df_bevolking_gevraagde_jaar = get_df_combined(
        countries, start, gevraagde_jaar, split_season
    )

    # Initialize an empty list to store results
    results = []

    # Loop through each group of age_sex and season
    for (age_sex_group, season), group_data in df_combined.groupby(
        ["age_sex", "season"]
    ):
        predicted_value = perform_lineair_regression(group_data, gevraagde_jaar, regresion_type)

        # Append the result as a dictionary
        results.append(
            {
                "age_sex": age_sex_group,
                "season": season,
                "jaar": gevraagde_jaar,
                "per100k": predicted_value[0],
            }
        )

        # Add actual data points to the results to include in the graph
        for _, row in group_data.iterrows():
            results.append(
                {
                    "age_sex": age_sex_group,
                    "season": season,
                    "jaar": row["jaar"],
                    "per100k": row["per100k"],
                }
            )

    # Convert the results into a DataFrame
    predictions_gevraagde_jaar = pd.DataFrame(results)

    # Merge with population data and calculate predictions
    result_gevraagde_jaar = predictions_gevraagde_jaar[
        predictions_gevraagde_jaar["jaar"] == gevraagde_jaar
    ]
    endresult_gevraagde_jaar = pd.merge(
        result_gevraagde_jaar, df_bevolking_gevraagde_jaar, on=["age_sex"], how="outer"
    )
    endresult_gevraagde_jaar = endresult_gevraagde_jaar[
        endresult_gevraagde_jaar["geslacht"] != "T"
    ]

    # Calculate expected number of deaths
    endresult_gevraagde_jaar["aantal_overleden_voorspelling"] = round(
        endresult_gevraagde_jaar["per100k"]
        * endresult_gevraagde_jaar["aantal"]
        / 100_000,
        1,
    )
    if split_season:
        # Calculate the sum for winter and summer separately
        winter_deaths = endresult_gevraagde_jaar[
            endresult_gevraagde_jaar["season"] == "winter"
        ]["aantal_overleden_voorspelling"].sum()
        summer_deaths = endresult_gevraagde_jaar[
            endresult_gevraagde_jaar["season"] == "summer"
        ]["aantal_overleden_voorspelling"].sum()

        # Sum total deaths for the requested year
        verw_overleden = int(winter_deaths + summer_deaths)
    else:
        verw_overleden = endresult_gevraagde_jaar["aantal_overleden_voorspelling"].sum()
    bevolkingsgrootte = (
        df_bevolking_gevraagde_jaar["aantal"].sum() / 2
    )  # divide by 2 due to 'T' values

    # Plot the results if conditions are met
    if (gevraagde_jaar == 2024) & (start == 2015):
        to_plot = True
    else:
        to_plot = False
    if to_plot:
        # Plot the results with Plotly
        make_plot(predictions_gevraagde_jaar, start, gevraagde_jaar)
        st.write(endresult_gevraagde_jaar)

    # Return the total expected deaths and population size
    return verw_overleden, bevolkingsgrootte

def bereken_verwachte_sterfte_simpel(
    countries: list[str], start: int, gevraagde_jaar: int, regresion_type: str, split_season: bool, to_plot: bool = False
     ) -> tuple[float, float]:
    """
    Calculate the expected mortality for the requested year using the average of 2015-2019.

    Args:
        countries (list[str]): List of country codes ("NL", "BE").
        start (int): The start year for the linear regression.
        gevraagde_jaar (int): The year for which the expected mortality is predicted.
        split_season (bool): Whether to split the data by season.
        regresion_type(string): Regression type
        to_plot (bool, optional): Whether to generate a plot. Defaults to False.

    Returns:
        tuple[float, float]: 
            - verw_overleden: The total expected number of deaths.
            - bevolkingsgrootte: The total population size for the requested year.
    """
    # Get data for all selected countries and concatenate them

    df_combined, df_bevolking_gevraagde_jaar = get_df_combined(
        countries, start, gevraagde_jaar, split_season
    )
   
    # Calculate average per100k for each age_group
    average_per100k = df_combined.groupby('age_group')['per100k'].mean().reset_index()

    # Rename the column for clarity
    average_per100k.columns = ['age_group', 'average_per100k']
    combined = pd.merge(average_per100k,df_bevolking_gevraagde_jaar,on="age_group")
    combined["verw_overleden"] = combined["average_per100k"] * combined["aantal"] / 100000
    totaal_verw_overleden = int(combined["verw_overleden"].sum()/2)
    bevolkingsgrootte = combined["aantal"].sum()/2
    
    return totaal_verw_overleden,bevolkingsgrootte
    

def main():
    """Streamlit application to predict mortality rates based on historical data for different age and gender groups."""
    
    st.title("Verwachte sterfte voor 2024 berekenen")
    st.info(
        """
        We voorspellen het aantal overlijdens voor 2024 met behulp van een lineaire regressie op basis van de overlijdensgegevens tussen 2015 en 2019.
        Bij 'splitsing' wordt een jaar beschouwd als week 40 van het voorafgaande jaar tot en met week 39. De winter is week 40 tot en met week 13.
         
         
         Dit doen we voor verschillende leeftijds- en geslachtsgroepen. De aanpak is geïnspireerd door Bonne Klok, die een vergelijkbare analyse heeft gedeeld op Twitter.
        Dit getal gebruiken we om de correctiefactor te berekenen, waarmee we de baseline corrigeren voor verbeterde gezondheid en veranderingen in de leeftijdsopbouw.
       
       Je kunt de tweet van Bonne Klok hier bekijken: https://twitter.com/BonneKlok/status/1832333262586323385.
       
       NB: In de grafieken is de ols-regressielijn te zien"""
    )
    # Let the user select one or both countries
    countries = ["NL"]  # st.multiselect("land [NL | BE]", ["NL", "BE"], default=["NL"])
    regresion_type =  st.selectbox("Regression type [ols|huber|ransac]", ["ols","huber","ransac"],0)
    # start = st.number_input("Startjaar voor lineaire regressie", 2000, 2020, 2015)

    # gevraagde_jaar = st.number_input("Verwachting bereken voor jaar", 2021,2030,2024)

    # start_jaren = [2000,2005, 2010,2015]
    gevraagde_jaren = [2020, 2021, 2022, 2023, 2024]

    start_jaren = [2015]
    tabel = pd.DataFrame(index=gevraagde_jaren, columns=start_jaren)
    st.subheader("Easy methode")
    st.write("Gemiddelde overlijdens per groep per 100k, vermenigvuldigd met groepsgrootte van het doeljaar")
    for start in start_jaren:
            for gevraagde_jaar in gevraagde_jaren:
                verw_overleden, bevolkingsgrootte = bereken_verwachte_sterfte_simpel(
                    countries, start, gevraagde_jaar, regresion_type, False
                )
                #st.write(f"{gevraagde_jaar} - {int(verw_overleden)} - {int(bevolkingsgrootte)}")
                tabel.loc[gevraagde_jaar, start] = verw_overleden
    st.write(tabel)
    col1, col2 = st.columns(2)
    with col1:

        st.subheader("Zonder splitsing")
        # Maak een lege DataFrame met de gevraagde jaren als index en startjaren als kolommen
        
        # Vul de DataFrame met verwachte overlijdenscijfers
        for start in start_jaren:
            for gevraagde_jaar in gevraagde_jaren:
                verw_overleden, bevolkingsgrootte = bereken_verwachte_sterfte(
                    countries, start, gevraagde_jaar, regresion_type, False
                )
                tabel.loc[gevraagde_jaar, start] = verw_overleden

        st.write(tabel)
    with col2:
        st.subheader("Met splitsing")

       
        # Vul de DataFrame met verwachte overlijdenscijfers
        for start in start_jaren:
            for gevraagde_jaar in gevraagde_jaren:
                verw_overleden, bevolkingsgrootte = bereken_verwachte_sterfte(
                    countries, start, gevraagde_jaar, regresion_type, True
                )
                tabel.loc[gevraagde_jaar, start] = verw_overleden

        st.write(tabel)

    st.info(
        """Door CBS gebruikt:

            2020: 153402  |
            2021: 154887 |
            2022: 155494 |
            2023: 156666 
           

           Door Bonne Klok geschat 166100
            """
    )

    # https://twitter.com/BonneKlok/status/1750533281337196960
    st.subheader("Databronnen")
    st.info(
        "Bevolkingsgrootte NL: https://opendata.cbs.nl/#/CBS/nl/dataset/03759ned/table?dl=39E0B"
    )
    st.info(
        "Sterfte: https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en"
    )

    st.info("Code: https://github.com/rcsmit/COVIDcases/blob/main/verwachte_sterfte.py")


if __name__ == "__main__":

    main()
