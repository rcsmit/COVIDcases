import pandas as pd
import requests
from io import StringIO
import streamlit as st
# from oversterfte_compleet import get_sterftedata

def process_csv(url, sex):
    
    df = pd.read_csv(url)
    # Melt the DataFrame, using the first column (age) as id_vars
    df = df.melt(id_vars=df.columns[0], var_name='jaar', value_name='prob_to_die')
    
    # Rename the age column
    df = df.rename(columns={df.columns[0]: 'leeftijd'})
    
    df['geslacht'] = sex
    return df

def main_old(mortality_data,population_data): 
    st.subheader("Annual Mortality Projection Method") 
    st.info("""
    We calculate the number of deaths for each year by multiplying the population 
    size of year y by the probability of death for that year. This process is 
    repeated annually, starting from the initial year until the end year, and 
    the total deaths are summed for each year
    """)  
    # Process both CSV files
    # male_data = process_csv(male_url, 'M')
    # female_data = process_csv(female_url, 'F')

    # # Combine the data
    # mortality_data = pd.concat([male_data, female_data], ignore_index=True)
    # mortality_data['jaar'] = mortality_data['jaar'].astype(int)
    # mortality_data['leeftijd'] = mortality_data['leeftijd'].astype(int)

    # population_data =  pd.read_csv(bevolking_url, delimiter=";",)
    # # Convert 'jaar' and 'leeftijd' to int
    # population_data['jaar'] = population_data['jaar'].astype(int)
    # population_data['leeftijd'] = population_data['leeftijd'].astype(int)
    # Reorder columns
    
    # Merge mortality and population data
    merged_data = pd.merge(mortality_data, population_data, on=['leeftijd', 'jaar', 'geslacht'], how='inner')
    
    merged_data["verw_overl"] = round(merged_data["prob_to_die"] * merged_data["aantal"])


    # Calculate the sum of expected deaths per year
    #expected_deaths_per_year = merged_data.groupby(['jaar','geslacht'])['verw_overl'].sum().reset_index()
    expected_deaths_per_year = merged_data.groupby(['jaar'])['verw_overl'].sum().reset_index()

    # Display the sum of expected deaths per year
    st.write("Sum of expected deaths per year Annual Mortality Projection Method:")
    st.write(expected_deaths_per_year)




def project_population(start_year, end_year,   population_data,  mortality_data):
   
    results = []
    current_population =  population_data[population_data['Year'] == start_year].copy()

    for year in range(start_year, end_year + 1):
        # Get mortality probabilities for the current year
    
        year_mortality = mortality_data[mortality_data['Year'] == year]
        
        # Merge current population with mortality data
        merged = pd.merge(current_population, year_mortality, on=['Year','Age', 'Sex'])
        
        # Calculate survivors and deaths
        merged['Survivors'] = merged['Population'] * (1 - merged['Probability'])
        merged['Deaths'] = merged['Population'] - merged['Survivors']
        
        # Record results
        results.append(merged[['Year', 'Age', 'Sex', 'Population', 'Deaths']])
        
        # Age the population and shift to next year
        next_population = merged.copy()
        next_population['Age'] += 1
        next_population['Population'] = next_population['Survivors']
        next_population['Year'] = year + 1
        # Add newborns (age 0) from the initial population data
        
        newborns =  population_data[ (population_data['Age'] == 0)& (population_data['Year'] == year+1)].copy()
        newborns['Year'] = year + 1
        next_population = pd.concat([next_population, newborns])
        
        current_population = next_population[['Year', 'Age', 'Sex', 'Population']]

    return pd.concat(results)
def main_new(mortality_data,population_data):
    # Run the projection from 2020 to 2030
    projected_population = project_population(2020, 2025, population_data, mortality_data)
    st.subheader("Cohort Survival Method")
    st.info("""
          We begin with the population size from 2020 and apply the corresponding 
          mortality probabilities to calculate the number of survivors moving into 2021. 
          Newborns are then added to the population. This process is repeated for each
          subsequent year until the final year. One limitation of this approach is that 
          it does not account for migration.
            """)
    # Display the first few rows of the projected population
    st.write("projected_population - Cohort Survival Method")

    st.write(projected_population)

    # Calculate total population per year
    total_population_per_year = projected_population.groupby('Year')['Population'].sum().round(0).reset_index()
    st.write("\nTotal population per year  - Cohort Survival Method:")
    st.write(total_population_per_year)

    # Calculate total deaths per year
    total_deaths_per_year = projected_population.groupby('Year')['Deaths'].sum().round(0).reset_index()
    st.write("\nTotal deaths per year - Cohort Survival Method:")
    st.write(total_deaths_per_year)

def main():
    st.header("AG Table Mortality Forecast")
    

    # URLs of the CSV files
    male_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/prognosetafel2020_mannen.csv"
    female_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/prognosetafel2020_vrouwen.csv"
    bevolking_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
    # Function to fetch and process CSV data



    male_data = process_csv(male_url, 'M')
    female_data = process_csv(female_url, 'F')

    # Combine the data
    mortality_data = pd.concat([male_data, female_data], ignore_index=True)
    mortality_data['jaar'] = mortality_data['jaar'].astype(int)
    mortality_data['leeftijd'] = mortality_data['leeftijd'].astype(int)
    mortality_data= mortality_data[mortality_data["jaar"] <2025]
    mortality_data = mortality_data.rename(columns={'leeftijd': 'Age', 'jaar':'Year','geslacht': 'Sex', 'prob_to_die': 'Probability'})

    # Fetch and process population data
    population_data =  pd.read_csv(bevolking_url, delimiter=";",)
    population_data = population_data.rename(columns={'leeftijd': 'Age','jaar':'Year', 'geslacht': 'Sex', 'aantal': 'Population'})

    main_old(mortality_data,population_data)
    main_new(mortality_data,population_data)
    st.info("Script: https://github.com/rcsmit/COVIDcases/blob/main/agtable_mortality.py")

if __name__ == "__main__":
    import datetime

    print(
        f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------"
    )


    main()
    # Optionally, save results to CSV files
    # projected_population.to_csv('projected_population_2020_2030.csv', index=False)
    # total_population_per_year.to_csv('total_population_per_year_2020_2030.csv', index=False)
    # total_deaths_per_year.to_csv('total_deaths_per_year_2020_2030.csv', index=False)