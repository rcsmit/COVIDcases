import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cbsodata
from scipy.stats import linregress
from scipy.optimize import curve_fit

# https://x.com/dimgrr/status/1916440956556988573/photo/1

def get_dataframe(file, delimiter=";"):
    """Get data from a file and return as a pandas DataFrame.

    Args:
        file (str): url or path to the file.
        delimiter (str, optional): _description_. Defaults to ";".

    Returns:
        pd.DataFrame: dataframe
    """   
    
    data = pd.read_csv(
        file,
        delimiter=delimiter,
        low_memory=False,
         encoding='utf-8',
          on_bad_lines='skip'
    )
    return data


def get_chances_to_die():
    """
    Load and transform mortality data for males and females into a long format.

    This function reads mortality data for males and females from CSV files, reshapes the data 
    into a long format with columns `age`, `year`, and `chance_of_death`, and ensures the `year` 
    column is numeric for merging with other datasets.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - chance_to_die_male_long: DataFrame with columns `age`, `year`, and `chance_of_death_male`.
            - chance_to_die_female_long: DataFrame with columns `age`, `year`, and `chance_of_death_female`.
    """
    chance_to_die_male = get_dataframe(r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\prognosetafel2024_vrouwen.csv",",")
    chance_to_die_female = get_dataframe(r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\prognosetafel2024_mannen.csv",",")
    
    # Transform chance_to_die_male into a long format with columns: age, year, chance_of_death
    chance_to_die_male_long = chance_to_die_male.melt(id_vars=["age"], var_name="year", value_name="chance_of_death_male")

    # Ensure the year column is numeric for merging
    chance_to_die_male_long["year"] = chance_to_die_male_long["year"].astype(int)

    # Transform chance_to_die_female into a long format with columns: age, year, chance_of_death
    chance_to_die_female_long = chance_to_die_female.melt(id_vars=["age"], var_name="year", value_name="chance_of_death_female")

    # Ensure the year column is numeric for merging
    chance_to_die_female_long["year"] = chance_to_die_female_long["year"].astype(int)
    return chance_to_die_male_long,chance_to_die_female_long


def get_total_inhabitants_per_agegroup(startjaar, eindjaar, bins, labels):
    """
    Calculate the total number of inhabitants per age group for a given range of years.

    This function retrieves population data, filters it for a specific gender ("T"), 
    extends the population data to future years, and groups the data into specified 
    age groups. It then calculates the total number of inhabitants per age group 
    for each year in the specified range.

    Args:
        startjaar (int): The starting year for the calculation.
        eindjaar (int): The ending year for the calculation.
        bins (list): A list of bin edges to define the age groups.
        labels (list): A list of labels corresponding to the age groups.

    Returns:
        pd.DataFrame: A DataFrame containing the total number of inhabitants per 
        age group and year, with columns:
            - leeftijdsgroep: The age group.
            - jaar: The year.
            - aantal: The total number of inhabitants in the age group for the year.
    """
   
    # https://www.cbs.nl/nl-nl/cijfers/detail/37168
    # https://www.cbs.nl/nl-nl/visualisaties/dashboard-population/populationspiramide ???
    population = get_dataframe(r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv")   
    # Replace "M" with "Mannen" and "F" with "Vrouwen" in the "Geslacht" column
    population["Geslacht"] = population["geslacht"].replace({"M": "Mannen", "F": "Vrouwen"})
    population = population[population["geslacht"] == "T"]
    population = extend_population_to_future(population, startjaar, eindjaar)
    population["leeftijdsgroep"] = pd.cut(population["leeftijd"], bins=bins, labels=labels, right=True)
    total_inhabitants_per_agegroup = population.groupby(["leeftijdsgroep", "jaar"])["aantal"].sum().reset_index()
    return total_inhabitants_per_agegroup

def get_diagnoses(bins, labels):
    """
    Load and process colorectal cancer diagnoses data.

    This function reads a CSV file containing colorectal cancer diagnoses, splits the `leeftijdsgroep` 
    column into two separate columns (`leeftijd_min` and `leeftijd_max`), and groups the data into 
    specified age groups. It then calculates the total number of diagnoses per age group and year.

    Args:
        bins (list): A list of bin edges to define the age groups.
        labels (list): A list of labels corresponding to the age groups.

    Returns:
        pd.DataFrame: A DataFrame containing the total number of colorectal cancer diagnoses per 
        age group and year, with columns:
            - leeftijdsgroep: The age group.
            - jaar: The year.
            - aantal_darmkanker_diagnoses: The total number of diagnoses for the age group and year.
    """
    darmkanker = get_dataframe(r"C:\Users\rcxsm\Downloads\darmkanker.csv")
    
    # Split the 'leeftijdsgroep' column into two new columns: 'leeftijd_min' and 'leeftijd_max'
    darmkanker[["leeftijd_min", "leeftijd_max"]] = darmkanker["leeftijdsgroep"].str.split("-", expand=True)

    # Convert the new columns to integers (if applicable)
    darmkanker["leeftijd_min"] = pd.to_numeric(darmkanker["leeftijd_min"], errors="coerce")
    darmkanker["leeftijd_max"] = pd.to_numeric(darmkanker["leeftijd_max"], errors="coerce")
    darmkanker["leeftijdsgroep"] = pd.cut(darmkanker["leeftijd_max"], bins=bins, labels=labels, right=True)
    darmkanker = darmkanker.groupby(["leeftijdsgroep", "jaar"])["aantal_darmkanker_diagnoses"].sum().reset_index()
    # # Ensure the new category (e.g., "0") exists in the categories
    # darmkanker["leeftijdsgroep"] = darmkanker["leeftijdsgroep"].cat.add_categories([0])
    darmkanker["leeftijdsgroep"] = darmkanker["leeftijdsgroep"].astype(str)
    # # Now you can safely assign the value
    # darmkanker.loc[darmkanker["leeftijdsgroep"].isna(), "leeftijdsgroep"] = 0
    
    return darmkanker


def voorspel_diagnosekans(diagnoses, startjaar=2000, eindjaar=2019, voorspeljaren=[2020, 2021, 2022, 2023, 2024,], model_type="linear"):
    """Predict mortality rates using the specified regression model.

    Args:
        diagnoses (pd.DataFrame): DataFrame containing mortality data.
        startjaar (int, optional): Start year for regression. Defaults to 2000.
        eindjaar (int, optional): End year for regression. Defaults to 2019.
        voorspeljaren (list, optional): Years to predict. Defaults to [2020, 2021, 2022, 2023].
        model_type (str, optional): Regression model type ("linear" or "quadratic"). Defaults to "linear".

    Returns:
        pd.DataFrame: DataFrame with predicted mortality rates.
    """
    
    models = {
        "linear": {
            "func": lambda x, a, b: a * x + b,
            "p0": [1, 1],
            "equation": "a*x + b",
            "params": ["a", "b"]
        },
        "quadratic": {
            "func": lambda x, a, b, c: a * x**2 + b * x + c,
            "p0": [1, 1, 1],
            "equation": "a*x^2 + b*x + c",
            "params": ["a", "b", "c"]
        }
    }
    
    

    # Select the model
    model = models[model_type]
    func = model["func"]
    p0 = model["p0"]
    voorspeljaren = range(startjaar,2036)
    diagnoses = diagnoses[diagnoses["jaar"] <= 2023]
    voorspellingen = []
    for (leeftijd, geslacht), groep in diagnoses.groupby(["leeftijdsgroep", "Geslacht"]):
        # Filter data for regression
        data = groep[(groep["jaar"] >= startjaar) & (groep["jaar"] <= eindjaar)]
        if len(data) < len(p0):  # Ensure enough data points for the model
            continue  # Skip if there is insufficient data

        # Fit the selected regression model
        try:
            popt, _ = curve_fit(func, data["jaar"], data["werkelijke_diagnosekans"], p0=p0)
            params = dict(zip(model["params"], popt))

            # Predict mortality rates for the prediction years
            for jaar in voorspeljaren:
                voorspelde_kans = func(jaar, *popt)
                voorspellingen.append({
                    "leeftijdsgroep": leeftijd,
                    "Geslacht": geslacht,
                    "jaar": jaar,
                    "voorspelde_diagnosekans": voorspelde_kans,
                    **params
                })
            for jaar in range(startjaar,eindjaar+1):
                voorspelde_kans = func(jaar, *popt)
                voorspellingen.append({
                    "leeftijdsgroep": leeftijd,
                    "Geslacht": geslacht,
                    "jaar": jaar,
                    "voorspelde_diagnosekans": voorspelde_kans,
                    **params
                })
        except RuntimeError:
            # Skip if the curve fitting fails
            continue

    return pd.DataFrame(voorspellingen),params

def bereken_verschil(diagnoses, voorspellingen, population):
    """
    Calculate the difference between actual and predicted diagnoses.

    This function merges the actual diagnoses data with the predicted diagnoses data, calculates the predicted number of diagnoses based on the population and predicted diagnosis probability, and computes the difference between actual and predicted diagnoses. It also calculates the total difference across all records.

    Args:
        diagnoses (pd.DataFrame): DataFrame containing actual diagnoses data.
        voorspellingen (pd.DataFrame): DataFrame containing predicted diagnosis probabilities.
        population (pd.DataFrame): DataFrame containing population data.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Updated diagnoses DataFrame with calculated predicted diagnoses and differences.
            - float: The total difference between actual and predicted diagnoses.
    """
    # Voeg voorspellingen toe aan diagnoses
    diagnoses = diagnoses.merge(voorspellingen, on=["leeftijdsgroep", "Geslacht", "jaar"], how="left")

    # Bereken voorspelde diagnose
    diagnoses["voorspelde_diagnose"] = diagnoses["voorspelde_diagnosekans"] * population["aantal"]

    # Bereken verschil
    diagnoses["verschil"] = diagnoses["OverledenenLeeftijdBijOverlijden_1"] - diagnoses["voorspelde_diagnose"]

    # Bereken totaal verschil
    totaal_verschil = diagnoses["verschil"].sum()
    return diagnoses, totaal_verschil


def extend_population_to_future(population, startjaar, eindjaar):
    """
    Extend the population data to future years by applying survival probabilities.

    This function takes the current population data and extends it to future years by applying 
    survival probabilities for males and females. It calculates the remaining population for 
    each age group based on the average survival probability, increments the age by 1, and 
    associates the updated population count with the next year. The function also adds a new 
    row for newborns (age 0) with a population count of 170106 for each year.

    Assumptions: 
        - ratio male/female is 50:50
        - number of newborns is constant (eg. 170106 the average between 2020 and 2024) 
        - there is no migration
    Args:
        population (pd.DataFrame): DataFrame containing the current population data with columns:
            - leeftijd: Age of the population group.
            - geslacht: Gender of the population group.
            - jaar: Year of the population data.
            - aantal: Population count.
        startjaar (int): The starting year for the extension.
        eindjaar (int): The ending year for the extension.

    Returns:
        pd.DataFrame: Updated population DataFrame extended to the specified future years, 
        with columns:
            - leeftijd: Age of the population group.
            - geslacht: Gender of the population group.
            - jaar: Year of the population data.
            - aantal: Updated population count.
    """
    chance_to_die_male_long, chance_to_die_female_long = get_chances_to_die()
     
    # Maak een lege lijst om de nieuwe rijen op te slaan
    nieuwe_rijen = []

    for jaar in range(startjaar, eindjaar):
        # Filter de data voor het huidige jaar
        huidig_jaar_data = population[population["jaar"] == jaar].copy()
   
        # Merge chance_to_die_male_long with huidig_jaar_data on age and year
        huidig_jaar_data = huidig_jaar_data.merge(chance_to_die_male_long, left_on=["leeftijd", "jaar"], right_on=["age", "year"], how="inner")

        # Drop redundant columns if necessary
        huidig_jaar_data = huidig_jaar_data.drop(columns=["age", "year"])

        # Merge chance_to_die_female_long with huidig_jaar_data on age and year
        huidig_jaar_data = huidig_jaar_data.merge(chance_to_die_female_long, left_on=["leeftijd", "jaar"], right_on=["age", "year"], how="inner")

        # Drop redundant columns if necessary
        huidig_jaar_data = huidig_jaar_data.drop(columns=["age", "year"])
      
        huidig_jaar_data["aantal"] = huidig_jaar_data["aantal"] *  (1-((huidig_jaar_data["chance_of_death_male"]  + huidig_jaar_data["chance_of_death_female"]) / 2))
        huidig_jaar_data["aantal"]= huidig_jaar_data["aantal"].astype(int)
        huidig_jaar_data = huidig_jaar_data.drop(columns=["chance_of_death_male", "chance_of_death_female"])
        # Verhoog de leeftijd met 1 en koppel het aantal aan het volgende jaar
        nieuwe_jaar_data = huidig_jaar_data.copy()
        nieuwe_jaar_data["leeftijd"] += 1
        nieuwe_jaar_data["jaar"] += 1
        nieuwe_jaar_data = pd.concat([nieuwe_jaar_data, pd.DataFrame({"leeftijd": [0], "geslacht": ["T"], "jaar": [jaar+1], "aantal": [170106]})], ignore_index=True)
        
        # # Voeg de nieuwe rijen toe aan de lijst
        nieuwe_rijen.append(nieuwe_jaar_data)

        # Combineer de originele data met de nieuwe rijen
        population = pd.concat([population] + nieuwe_rijen, ignore_index=True)
        nieuwe_rijen = []  
    
    
    # Create a pivot table for testing reasons

    # pivot_table = population.pivot_table(
    #     index="leeftijd",  # Rows: Age
    #     columns="jaar",    # Columns: Years
    #     values="aantal",   # Cells: Population count
    #     aggfunc="sum"      # Aggregation function: Sum
    # )

    # # print(pivot_table)
    # st.write(pivot_table)
    return population


def show_total_diagnoses(startjaar, geslacht, totaal_tabel_geslacht):
    """
    Display total diagnoses and predictions for a specific gender and starting year.

    This function generates interactive plots using Plotly to visualize the actual and predicted 
    diagnoses for a specific gender over time. It includes markers for actual diagnoses and lines 
    for predicted diagnoses, with additional markers and lines for data starting from a specified 
    year.

    Args:
        startjaar (int): The starting year for highlighting predictions.
        geslacht (str): The gender for which the diagnoses are displayed.
        totaal_tabel_geslacht (pd.DataFrame): DataFrame containing the total diagnoses and predictions 
            for the specified gender, with columns:
            - jaar: The year.
            - aantal_darmkanker_diagnoses: Actual diagnoses.
            - voorspelde_diagnose: Predicted diagnoses.
    """
    wx = [["aantal_darmkanker_diagnoses", "voorspelde_diagnose", "Total diagnoses"]]
    
    for w in wx:
        # totaal_tabel_geslacht[w] = totaal_tabel_geslacht[w].astype(float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=totaal_tabel_geslacht["jaar"], y=totaal_tabel_geslacht[w[0]], mode='markers', name=f'{w[0]}', ))
        fig.add_trace(go.Scatter(
                x=totaal_tabel_geslacht[(totaal_tabel_geslacht["jaar"] >= startjaar) & (totaal_tabel_geslacht["jaar"] <= startjaar)]["jaar"],
                y=totaal_tabel_geslacht[(totaal_tabel_geslacht["jaar"] >= startjaar) & (totaal_tabel_geslacht["jaar"] <= startjaar)][w[0]],
                mode='markers',
                name=f'{w[0]} (startjaar+)',
                #marker=dict(color='red')
            ))
        fig.add_trace(go.Scatter(x=totaal_tabel_geslacht["jaar"], y=totaal_tabel_geslacht[w[1]], mode='lines', name=f'{w[1]}', marker=dict(color='green') ))       
        fig.add_trace(go.Scatter(
                x=totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] >= startjaar]["jaar"],
                y=totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] >= startjaar][w[1]],
                mode='lines+markers',
                name=f'{w[1]} (startjaar+)',
                marker=dict(color='green')
            ))
        # Add the line to the graph
        
        #fig.add_trace(go.Scatter(x=totaal_tabel_geslacht["jaar"], y=totaal_tabel_geslacht["y_line"], mode='lines', name=f'Voorspelde lijn', line=dict(dash='dash', color='red')))
        fig.update_layout(title=f"{w[2]} {geslacht}", xaxis_title="Jaar", yaxis_title="Waarde")
        st.plotly_chart(fig, use_container_width=True)

def show_plots(to_show, totaal_tabel_geslacht):
    """
    Generate and display interactive plots for diagnoses and predictions across age groups.

    This function creates interactive plots using Plotly to visualize actual and predicted diagnoses 
    for specified age groups over time. It includes markers for actual diagnoses and lines for predicted 
    diagnoses, with additional markers and lines for data starting from a specified year.

    Args:
        to_show (list): List of age groups to include in the plots.
        totaal_tabel_geslacht (pd.DataFrame): DataFrame containing diagnoses and predictions for the specified 
            gender and age groups, with columns:
            - jaar: The year.
            - leeftijdsgroep: The age group.
            - werkelijke_diagnosekans: Actual diagnosis probability.
            - voorspelde_diagnosekans: Predicted diagnosis probability.
            - aantal: Population count.
            - aantal_darmkanker_diagnoses: Actual diagnoses.
            - voorspelde_diagnose: Predicted diagnoses.

    Returns:
        str: The gender for which the plots were generated.
    """
    geslacht = "T"
    for w in [["werkelijke_diagnosekans", "voorspelde_diagnosekans", "diagnoses per 100k"],["aantal","aantal","Aantal inwoners"] ,["aantal_darmkanker_diagnoses", "voorspelde_diagnose", "Diagnoses | voorspelde diagnose = diagnoseskans * population)"]]:
        fig = go.Figure()   
        for age in  to_show:
            totaal_tabel_leeftijd = totaal_tabel_geslacht[totaal_tabel_geslacht["leeftijdsgroep"] == age]
 
            fig.add_trace(go.Scatter(
                    x=totaal_tabel_leeftijd[totaal_tabel_leeftijd["jaar"] <= 2024]["jaar"],
                    y=totaal_tabel_leeftijd[totaal_tabel_leeftijd["jaar"] <= 2024][w[0]],
                    mode='markers',
                    name=f' {age}',
                ))
            
            
            # fig.add_trace(go.Scatter(
            #         x=totaal_tabel_leeftijd[totaal_tabel_leeftijd["jaar"] >= startjaar]["jaar"],
            #         y=totaal_tabel_leeftijd[totaal_tabel_leeftijd["jaar"] >= startjaar][w[0]],
            #         mode='markers',
            #         name=f' {age}',
                    
            #     ))
            
            fig.add_trace(go.Scatter(x=totaal_tabel_leeftijd["jaar"], y=totaal_tabel_leeftijd[w[1]], mode='lines', name=f'{age}',  ))       
            fig.add_trace(go.Scatter(
                    x=totaal_tabel_leeftijd[totaal_tabel_leeftijd["jaar"] >= 2025]["jaar"],
                    y=totaal_tabel_leeftijd[totaal_tabel_leeftijd["jaar"] >= 2025][w[1]],
                    mode='lines+markers',
                    name=f'{age}',
                )
                )
            # Add the line to the graph
            
        #fig.add_trace(go.Scatter(x=totaal_tabel_geslacht["jaar"], y=totaal_tabel_geslacht["y_line"], mode='lines', name=f'Voorspelde lijn', line=dict(dash='dash', color='red')))
        fig.update_layout(title=f"{w[2]}", xaxis_title="Jaar", yaxis_title="Waarde")
        st.plotly_chart(fig, use_container_width=True)
    return geslacht

def interface(labels, selected):
    """
    Create an interface for user input to configure regression models and data visualization.

    This function provides an interactive interface using Streamlit to allow users to:
    1. Select a regression model type (linear or quadratic).
    2. Specify the start and end years for known data.
    3. Choose specific age groups to display in the plots.

    Args:
        labels (list): A list of age group labels available for selection.
        selected (list): A list of pre-selected age groups to display by default.

    Returns:
        tuple: A tuple containing:
            - model_type (str): The selected regression model type ("linear" or "quadratic").
            - startjaar_bekend (int): The starting year for known data.
            - eindjaar_bekend (int): The ending year for known data.
            - to_show (list): The list of age groups selected for display.
    """
    # Allow user to select model type
    col1,col2,col3,col4=st.columns(4)
    with col1:
        model_type = st.selectbox("Select regression model", ["linear", "quadratic"])
    with col2:
        startjaar_bekend = st.number_input("Start year known", min_value=1960, max_value=2019, value=1990)
    with col3:
        eindjaar_bekend = st.number_input("Start year known", min_value=1960, max_value=2024, value=2020)

    to_show=st.multiselect("To show", labels, selected)

    return model_type, startjaar_bekend, eindjaar_bekend, to_show



def make_totaal_tabel(total_inhabitants_per_agegroup, diagnoses):
    """
    Create a combined table of total inhabitants and diagnoses.

    This function merges the total inhabitants per age group with the diagnoses data, fills missing values with 0, 
    and calculates the actual diagnosis probability per 100,000 inhabitants. It also adds a column for gender.

    Args:
        total_inhabitants_per_agegroup (pd.DataFrame): DataFrame containing the total number of inhabitants per 
            age group and year, with columns:
            - leeftijdsgroep: The age group.
            - jaar: The year.
            - aantal: The total number of inhabitants in the age group for the year.
        diagnoses (pd.DataFrame): DataFrame containing diagnoses data, with columns:
            - leeftijdsgroep: The age group.
            - jaar: The year.
            - aantal_darmkanker_diagnoses: The number of diagnoses for the age group and year.

    Returns:
        pd.DataFrame: A combined DataFrame with the following columns:
            - leeftijdsgroep: The age group.
            - jaar: The year.
            - aantal: The total number of inhabitants in the age group for the year.
            - aantal_darmkanker_diagnoses: The number of diagnoses for the age group and year.
            - werkelijke_diagnosekans: The actual diagnosis probability per 100,000 inhabitants.
            - Geslacht: The gender (set to "T").
    """
    totaal_tabel = total_inhabitants_per_agegroup.merge(diagnoses, on=["leeftijdsgroep","jaar"], how="outer")
    totaal_tabel = totaal_tabel.fillna(0)
    totaal_tabel["werkelijke_diagnosekans"] = totaal_tabel["aantal_darmkanker_diagnoses"].astype(int)/totaal_tabel["aantal"] * 100000
    totaal_tabel["Geslacht"] = "T"
    return totaal_tabel

def main():
    st.header("Overdiagnose berekening")
    st.info("""
1. We delen het aantal diagnoses per leeftijd *l* door het aantal mensen van diezelfde leeftijd *l*.
2a. We berekenen het aantal inwoners na 2025 (aantal inwoners van y-1, maal de overlevingskans. we houden geen rekening met migratie en geboorte. De waardes voor populatie onder de 10 jaar kloppen dan niet (aantal diagnoses is tevens nihil))
2b. We voorspellen de diagnosekans vanaf 2020 met een lineaire of kwadratische regressie, op basis van data van het gekozen beginjaar tot en met 2019.
3. We vermenigvuldigen de voorspelde diagnosekans met het aantal inwoners van leeftijd *l* in het betreffende jaar.
4. We berekenen het verschil tussen de werkelijke diagnose en de voorspelde diagnose.
5. We tellen deze verschillen op per jaar en per geslacht.
""")
    # Voeg een nieuwe kolom toe voor leeftijdsgroepen
    # bins = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 120]
    # labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
    #         "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-120"]
    # selected =  [ "40-44", "45-49", "50-54", "55-59", "60-64"]

    bins = [0, 14, 29,  39, 49,  120]
    labels = ["0-14","15-29", "30-39", "40-49", "50-120"]
    selected =  [ "15-29", "30-39", "40-49"]

    # Filter de data voor de jaren 2025 tot 2036
    startjaar = 2024
    eindjaar = 2036

    total_inhabitants_per_agegroup = get_total_inhabitants_per_agegroup(startjaar, eindjaar, bins, labels)
    diagnoses = get_diagnoses(bins, labels)
   
    totaal_tabel = make_totaal_tabel(total_inhabitants_per_agegroup, diagnoses)
    
    model_type, startjaar_bekend, eindjaar_bekend, to_show = interface(labels, selected)
    eindresultaat= pd.DataFrame()
    # Bereken diagnosekans
    geslacht = "T"
    totaal_tabel_geslacht = totaal_tabel[totaal_tabel["Geslacht"] == geslacht]
    
    totaal_tabel_geslacht = totaal_tabel_geslacht[totaal_tabel_geslacht["jaar"] >= startjaar_bekend]
   
    # Predict mortality rates
    voorspeljaren = range(startjaar,eindjaar)
    voorspellingen,parameters = voorspel_diagnosekans(totaal_tabel_geslacht, startjaar=startjaar_bekend, eindjaar=eindjaar_bekend, voorspeljaren=voorspeljaren, model_type=model_type)
    totaal_tabel_geslacht = voorspellingen.merge(totaal_tabel_geslacht, on=["jaar", "leeftijdsgroep", "Geslacht"], how="outer")
    
    totaal_tabel_geslacht["voorspelde_diagnose"] = totaal_tabel_geslacht["voorspelde_diagnosekans"] * totaal_tabel_geslacht["aantal"]/100000
        
    eindresultaat = pd.concat([eindresultaat, totaal_tabel_geslacht], ignore_index=True)    
    
    geslacht = show_plots(to_show, totaal_tabel_geslacht)
    
    # totaal_tabel_geslacht = totaal_tabel_geslacht.groupby("jaar").sum().reset_index()
    
    # show_total_diagnoses(startjaar, geslacht, totaal_tabel_geslacht)



if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')

    print(f"--------------{datetime.datetime.now()}-------------------------")

    print ("hello world")
    main()