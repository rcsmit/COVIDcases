import pandas as pd
import requests
from io import StringIO
import streamlit as st
# from oversterfte_compleet import get_sterftedata

def process_csv(url, sex):
    
    df = pd.read_csv(url)
    # Melt the DataFrame, using the first column (age) as id_vars
    df = df.melt(id_vars=df.columns[0], var_name='jaar', value_name='prob_to_die')
    df['prob_to_die'] = df['prob_to_die']*1000 

    # Create a column for the next age (x+1)
    df['age_next'] = df.groupby('jaar')['age'].shift(-1)
    df['prob_next'] = df.groupby('jaar')['prob_to_die'].shift(-1)

    # Fill missing prob_next with the value of prob_to_die for the last age
    df['prob_next'] = df['prob_next'].fillna(df['prob_to_die'])

    # Calculate the average probability of dying for age x and x+1
    df['avg_prob_to_die'] = df[['prob_to_die', 'prob_next']].mean(axis=1)

    # Drop unnecessary columns if you only need the final dataframe
    df = df.drop(columns=['age_next', 'prob_next'])

    # Rename the age column
    df = df.rename(columns={df.columns[0]: 'leeftijd'})
    
    df['geslacht'] = sex
    return df

def main_old(mortality_data,population_data): 
    # reproducing https://www.linkedin.com/posts/annelaning_berekening-van-oversterfte-2024-tm-week-activity-7226089542238306304--Nee/?utm_source=share&utm_medium=member_android
    st.subheader("Annual Mortality Projection Method") 
    st.info("""
            For a given year, the probability of death for two consecutive age groups (age x and age x+1) 
            is averaged. This average probability is applied to the population size of age x. 
            The expected deaths for each age group are then added together to calculate 
            the total expected deaths for that year.
            """)  

    # Merge mortality and population data
    merged_data = pd.merge(mortality_data, population_data, on=['Age', 'Year', 'Sex'], how='inner')
    
    merged_data["verw_overl"] = round(merged_data["Probability"] * merged_data["Population"])
    merged_data["verw_overl_avg"] = round(merged_data["Probability_avg"] * merged_data["Population"])

    st.write(merged_data)
    # Calculate the sum of expected deaths per year
    #expected_deaths_per_year = merged_data.groupby(['jaar','geslacht'])['verw_overl'].sum().reset_index()
    
    expected_deaths_per_year = merged_data.groupby(['Year'])['verw_overl'].sum().reset_index()
    expected_deaths_per_year_geslacht = merged_data.groupby(['Year', 'Sex'])['verw_overl'].sum().reset_index()
    
    expected_deaths_per_year_avg = merged_data.groupby(['Year'])['verw_overl_avg'].sum().reset_index()
    expected_deaths_per_year_geslacht_avg = merged_data.groupby(['Year','Sex'])['verw_overl_avg'].sum().reset_index()
    # st.write(expected_deaths_per_year)
    # st.write(expected_deaths_per_year_geslacht)
    # st.write(expected_deaths_per_year_geslacht_avg)

    
    # Pivot the data to have 'Year' in rows, 'Sex' in columns, and the sum of 'verw_overl_avg'
    pivot_table = merged_data.pivot_table(index='Year', 
                                        columns='Sex', 
                                        values='verw_overl_avg', 
                                        aggfunc='sum', 
                                        margins=True, 
                                        margins_name='Total').reset_index()

    # Display the result
    st.write(pivot_table)
    st.write("Deze klopt")

    # Display the sum of expected deaths per year
    st.write("Annual Mortality Projection Method: Sum of Expected Deaths per Year Based on the Average Probability of Death for Ages x and x+1")
    


    # CBS data for the same years
    cbs_data = pd.DataFrame({
        'Year': [2020, 2021, 2022, 2023,2024],
        'CBS_expectation': [153402, 154887, 155494, 156666,1]
    })

    # Merge your data with CBS data on Year
    merged_data_with_cbs = pd.merge(expected_deaths_per_year_avg, cbs_data, on='Year')

    # Calculate the difference and percentage difference
    merged_data_with_cbs['difference'] = merged_data_with_cbs['verw_overl_avg'] - merged_data_with_cbs['CBS_expectation']
    merged_data_with_cbs['percent_difference'] = (merged_data_with_cbs['difference'] / merged_data_with_cbs['CBS_expectation']) * 100

    # Display the results
    st.write(merged_data_with_cbs[['Year', 'verw_overl_avg', 'CBS_expectation', 'difference', 'percent_difference']])

   
    # Step 1: Create age groups
    bins = [-1, 62, 77, float('inf')]  # Define the age bins
    labels = ['0-64', '65-79', '80+']  # Define the age group labels
    merged_data['age_group'] = pd.cut(merged_data['Age'], bins=bins, labels=labels, right=True).copy(deep=True)

    # Check the distribution in ages in the age groups
    # st.table(merged_data[['Age', 'age_group']].drop_duplicates())

    # Step 2: Group by Year, Sex, and Age Group, summing the expected deaths
    grouped_data = merged_data.groupby(['Year', 'Sex', 'age_group'], observed=False)['verw_overl_avg'].sum().reset_index()

    
    # Step 2: Pivot the data
    pivot_table = grouped_data.pivot_table(index='Year',
                                        columns=['age_group', 'Sex'],
                                        values='verw_overl_avg',
                                        aggfunc='sum',
                                        fill_value=0,  observed=False)

    # Reset index to turn MultiIndex into columns and add totals
    pivot_table = pivot_table.reset_index()

    # Optional: Add totals for each age group and sex
    pivot_table.loc['Total'] = pivot_table.drop(columns='Year').sum(numeric_only=True)
    pivot_table['Year'] = pivot_table['Year'].fillna('Total')  # Fill NaN in the 'Year' column with 'Total'

   
    st.write("Agegroups per year, per sex")
    # Display the result
    st.write(pivot_table)


    # Step 3: Pivot the data with Year as rows and Sex as columns for each age group, including totals
    pivot_table_2 = grouped_data.pivot_table(index=['age_group'], 
                                        columns='Sex', 
                                        values='verw_overl_avg', 
                                        aggfunc='sum', 
                                        margins=True, 
                                        margins_name='Total').reset_index()
    st.write("Agegroup per sex")
    # Display the result
    st.write(pivot_table_2)

    # Step 2: Group by age group and sum the expected deaths for all years and all sexes combined
    grouped_data = merged_data.groupby('age_group', observed=False)['verw_overl_avg'].sum().reset_index()

    # Step 3: Add totals for all age groups
    grouped_data.loc['Total'] = grouped_data.sum(numeric_only=True)

    st.write ("Totals per age group for all years and both sexes 2020-2023")
    # Display the result
    st.write(grouped_data)

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
    
    for a in ["0"]:
        # URLs of the CSV files
        male_url = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/prognosetafel202{a}_mannen.csv"
        female_url = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/prognosetafel202{a}_vrouwen.csv"
        bevolking_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
        # Function to fetch and process CSV data
        st.write(male_url)


        male_data = process_csv(male_url, 'M')
        female_data = process_csv(female_url, 'F')

        # Combine the data
        mortality_data = pd.concat([male_data, female_data], ignore_index=True)
        mortality_data['jaar'] = mortality_data['jaar'].astype(int)
        
        mortality_data['leeftijd'] = mortality_data['leeftijd'].astype(int)
        mortality_data= mortality_data[mortality_data["jaar"] <2024]
        mortality_data = mortality_data.rename(columns={'leeftijd': 'Age', 'jaar':'Year','geslacht': 'Sex', 'prob_to_die': 'Probability', 'avg_prob_to_die':'Probability_avg'})

        # Fetch and process population data
        population_data =  pd.read_csv(bevolking_url, delimiter=";",)
        population_data['aantal'] = population_data['aantal']/1000
        population_data = population_data.rename(columns={'leeftijd': 'Age','jaar':'Year', 'geslacht': 'Sex', 'aantal': 'Population'})

        main_old(mortality_data,population_data)

        #main_new(mortality_data,population_data)

   
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