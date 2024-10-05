
import pandas as pd

import plotly.graph_objects as go
import eurostat
import platform
import streamlit as st
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
try:
    st.set_page_config(layout="wide")
except:
    pass

def get_bevolking(country, opdeling):
    if country == "NL":
        if platform.processor() != "":
            file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_NL.csv"
        else: 
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
    elif country == "BE":
        # https://ec.europa.eu/eurostat/databrowser/view/demo_pjan__custom_12780094/default/table?lang=en
        if platform.processor() != "":
            file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_BE.csv"
        else: 
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_BE.csv"
    else:
        st.error(f"Error in country {country}")
    data = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
   
    data['leeftijd'] = data['leeftijd'].astype(int)
    
    # Define age bins and labels
    bins = list(range(0, 95, 5)) + [1000]  # [0, 5, 10, ..., 90, 1000]
    labels = [f'Y{i}-{i+4}' for i in range(0, 90, 5)] + ['Y90-120']


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
        grouped_data.replace(f'Y90-120_{s}',f'Y_GE90_{s}', inplace=True)
    

    # Calculate totals per year and gender (geslacht)
    totals = grouped_data.groupby(['jaar', 'geslacht'], observed=False)['aantal'].sum().reset_index()


    # Assign 'Total' as the age group for these sums
    totals['age_group'] = 'TOTAL'
    totals['age_sex'] = "TOTAL_" + totals['geslacht'].astype(str)

    # Concatenate the original grouped data with the totals
    final_data = pd.concat([grouped_data, totals], ignore_index=True)
  

    def add_custom_age_group(data, min_age, max_age):
        # Find the age group labels that fit within the specified min and max age
        valid_age_groups = [f'Y{i}-{i+4}' for i in range(min_age, max_age + 1, 5) if i < 90]
        
        # Include edge cases for Y_LT5 and Y_GE90 if they fall within the range
        if min_age <= 4:
            valid_age_groups.append('Y_LT5')
        if max_age >= 90:
            valid_age_groups.append('Y_GE90')

        # Filter the grouped data based on these age groups and sum
        custom_age_group = data[data['age_group'].isin(valid_age_groups)].groupby(['jaar', 'geslacht'], observed=False)['aantal'].sum().reset_index()

        # Assign the label for the new age group
        custom_age_group['age_group'] = f'Y{min_age}-{max_age}'
        custom_age_group['age_sex'] = f'Y{min_age}-{max_age}_' + custom_age_group['geslacht'].astype(str)

        return custom_age_group
    # Concatenate the original grouped data with the totals

    for i in opdeling:
        
        custom_age_group = add_custom_age_group(data, i[0], i[1])
        final_data = pd.concat([final_data, custom_age_group], ignore_index=True)

    return final_data



def get_sterfte(opdeling,country="NL"):
    """_summary_

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
          

    if country == "NL": 
        if platform.processor() != "":
            file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_NL.csv"
        else:
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_NL.csv"
    elif country == "BE":
        if platform.processor() != "":
            file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_BE.csv"
        else:
            file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_BE.csv"
    else:
        st.error(f"Error in country {country}")          
    df_ = pd.read_csv(
        file,
        delimiter=",",
            low_memory=False,
            )  
   
    df_=df_[df_["geo"] == country]

    df_["age_sex"] = df_["age"] + "_" +df_["sex"]

    
    # Function to extract age_low and age_high based on patterns
    def extract_age_ranges(age):
        if age == "TOTAL":
            return 999, 999
        elif age == "UNK":
            return 9999, 9999
        elif age == "Y_LT5":
            return 0, 4
        elif age == "Y_GE90":
            return 90, 120
        else:
            # Extract the numeric part from the pattern 'Y10-14'
            parts = age[1:].split('-')
            return int(parts[0]), int(parts[1])

    # Apply the function to create the new columns
    df_['age_low'], df_['age_high'] = zip(*df_['age'].apply(extract_age_ranges))

    df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["weeknr"] = (df_["TIME_PERIOD"].str[6:]).astype(int)


    def add_custom_age_group_deaths(df_, min_age, max_age):
        # Filter the data based on the dynamic age range
        df_filtered = df_[(df_['age_low'] >= min_age) & (df_['age_high'] <= max_age)]

        # Group by TIME_PERIOD (week), sex, and sum the deaths (OBS_VALUE)
        totals = df_filtered.groupby(['TIME_PERIOD', 'sex'], observed=False)['OBS_VALUE'].sum().reset_index()

        # Assign a new label for the age group (dynamic)
        totals['age'] = f'Y{min_age}-{max_age}'
        totals["age_sex"] = totals["age"] + "_" +totals["sex"]
        totals["jaar"] = (totals["TIME_PERIOD"].str[:4]).astype(int)
        return totals
    
    for i in opdeling:
        custom_age_group = add_custom_age_group_deaths(df_, i[0], i[1])
        df_ = pd.concat([df_, custom_age_group], ignore_index=True)

  
   
    df_bevolking = get_bevolking(country, opdeling)

    summed_per_year = df_.groupby(["jaar", 'age_sex'])['OBS_VALUE'].sum() # .reset_index()
  
    df__ = pd.merge(summed_per_year, df_bevolking, on=['jaar', 'age_sex'], how='outer')
    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["OBS_VALUE"].notna()]
    df__ = df__[df__["jaar"] != 2024]
    df__["per100k"] = round(df__["OBS_VALUE"]/df__["aantal"]*100000,1)
    
    return df__

def plot(df, category, value_field, countries):
    if value_field == "percentage":
        value_field_ ="per100k"
    else:
        value_field_=value_field
    # Filter the data
    df_before_2020 = df[df["jaar"] < 2020]
    df_2020_and_up = df[df["jaar"] >= 2020]
    
   
    trendline_info = ""  # Initialize a string to store trendline info
    for country in countries:
        if country == "BE":
            color_before_2020 = '#B22222'  # Dark Red
            color_2020_and_up = '#DC143C'  # Crimson
            trendline_color = '#FFA07A'    # Light Salmon
        elif country == "NL":
            color_before_2020 = '#00008B'  # Dark Blue
            color_2020_and_up = '#1E90FF'  # Dodger Blue
            trendline_color = '#87CEFA'    # Light Sky Blue
    
        df_country_before_2020 = df_before_2020[df_before_2020["country"] == country]
        df_country_2020_and_up = df_2020_and_up[df_2020_and_up["country"] == country]
        
        sd = df_country_before_2020[value_field_].std()
        
        
        # Calculate the trendline for each country before 2020
        X = df_country_before_2020["jaar"]
        y = df_country_before_2020[value_field_]
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        # Define the extended range of years
        extended_years = np.arange(df_country_before_2020["jaar"].min(), 2025)

        try:
            model = sm.OLS(y, X).fit()
            trendline = model.predict(X)
            
            # Add the trendline for the original years to the DataFrame
            if value_field == 'OBS_VALUE':
                df_country_before_2020['predicted_deaths'] = trendline
            else:
                df_country_before_2020['predicted_per100k'] = trendline
            
            # Create a DataFrame for the extended years
            extended_X = sm.add_constant(extended_years)
            
              # Predict the trendline and bounds for the extended years
            trendline_extended = model.predict(extended_X)
            upper_bound_extended = trendline_extended + 2 * sd
            lower_bound_extended = trendline_extended - 2 * sd
             # Calculate the upper and lower bounds of the shaded area
            upper_bound = trendline + 2 * sd
            lower_bound = trendline - 2 * sd

           
            # Calculate R² value
            r2 = r2_score(y, trendline)
            # trendline_info += f"{country}\nTrendline formula: y = {model.params[1]:.4f}x + {model.params[0]:.4f}\nR² value: {r2:.4f}\n\n"

            # Adjusted code with .iloc for position-based access
            trendline_info += f"{country}\nTrendline formula: y = {model.params.iloc[1]:.4f}x + {model.params.iloc[0]:.4f}\nR² value: {r2:.4f}\n\n"
        
            # # Print the formula and R² value
            # st.write(f"Trendline formula: y = {model.params[1]:.4f}x + {model.params[0]:.4f}")
            # st.write(f"R² value: {r2:.4f}")

                       
        except:
            pass
       
        if value_field == 'OBS_VALUE':
            df_extended = pd.merge(df_country_2020_and_up, pd.DataFrame({
                    'jaar': extended_years,
                    'predicted_deaths': trendline_extended
                    }), on='jaar')
        else:
            df_extended = pd.merge(df_country_2020_and_up, pd.DataFrame({
                    'jaar': extended_years,
                    'predicted_per100k': trendline_extended
                    }), on='jaar')

        # Concatenate the original and extended DataFrames
        df_diff = pd.concat([df_country_before_2020, df_extended], ignore_index=True)
        
        # Optionally, sort by year
        df_diff = df_diff.sort_values(by='jaar').reset_index(drop=True)
        if value_field_ == 'per100k':
            df_diff['predicted_deaths'] = df_diff['predicted_per100k']*df_diff['aantal']/100000

        df_diff['oversterfte'] = round(df_diff['OBS_VALUE'] - df_diff['predicted_deaths']) 
        df_diff['aantal']=round(df_diff['aantal'])
        df_diff['percentage'] = round(((df_diff['OBS_VALUE'] - df_diff['predicted_deaths'])/df_diff['predicted_deaths'])*100,1)
        df_diff = df_diff[['jaar', 'aantal', 'per100k', 'oversterfte', 'percentage']]
        
         # Create the scatter plot with Plotly Express for values before 2020
        fig = go.Figure()
        if value_field == 'percentage':
             # Plot before 2020
            # Filter the data for the current country
            df_country_before_2020 = df_diff[df_diff['jaar']<2020]
            df_country_2020_and_up = df_diff[df_diff['jaar']>=2020]

            # Plot bars before 2020
            fig.add_trace(go.Bar(
                x=df_country_before_2020["jaar"],
                y=df_country_before_2020[value_field],
                name=f'{country} - Before 2020',
                marker=dict(color=color_before_2020)
            ))

            # Plot bars for 2020 and up
            fig.add_trace(go.Bar(
                x=df_country_2020_and_up["jaar"],
                y=df_country_2020_and_up[value_field],
                name=f'{country} - 2020 and up',
                marker=dict(color='red')  # Set the color to red for years >= 2020
            ))
            # # Plot 2020 and up
            # fig.add_trace(go.Scatter(x=df_country_2020_and_up["jaar"], y=df_country_2020_and_up[value_field], 
            #                         mode='markers', name=f'{country} - 2020 and up', marker=dict(color=color_2020_and_up)))
           
        else:
            # Plot before 2020
            fig.add_trace(go.Scatter(x=df_country_before_2020["jaar"], y=df_country_before_2020[value_field], 
                                    mode='markers', name=f'{country} - Before 2020', marker=dict(color=color_before_2020)))

            # Plot 2020 and up
            fig.add_trace(go.Scatter(x=df_country_2020_and_up["jaar"], y=df_country_2020_and_up[value_field], 
                                    mode='markers', name=f'{country} - 2020 and up', marker=dict(color=color_2020_and_up)))
            

            # Add the shaded area to the plot
            fig.add_trace(go.Scatter(
                x=df_country_before_2020["jaar"].tolist() + df_country_before_2020["jaar"].tolist()[::-1],
                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.3)',  # Adjust the color and opacity as needed
                line=dict(color='rgba(255,255,255,0)'),  # Invisible line
                name=f'±2 SD {country} till 2019'
            ))
            # Add the trendline to the plot
            fig.add_trace(go.Scatter(x=df_country_before_2020["jaar"], y=trendline, 
                                    mode='lines', name=f'Trendline {country} till 2019', line=dict(color=trendline_color)))
                # Add the shaded area to the plot
            fig.add_trace(go.Scatter(
                x=np.concatenate([extended_years, extended_years[::-1]]),
                y=np.concatenate([upper_bound_extended, lower_bound_extended[::-1]]),
                fill='toself',  # Corrected fill mode
                fillcolor='rgba(128, 128, 128, 0.3)',  # Adjust the color and opacity as needed
                line=dict(color='rgba(255,255,255,0)'),  # Invisible line
                name=f'±2 SD {country} until 2024'
            ))

            # Add the trendline to the plot
            fig.add_trace(go.Scatter(
                x=extended_years,
                y=trendline_extended,
                mode='lines',
                name=f'Trendline {country} until 2024',
                line=dict(color=trendline_color)
            ))
        
        fig.update_layout(
                title=category,
                xaxis_title="Year",
                yaxis_title=value_field,
            )
        # Show the plot
    st.plotly_chart(fig)
    with st.expander(f"Trendline/oversterfte info - {category}"):
        st.write(trendline_info)
        #if value_field == 'OBS_VALUE':
        st.write(df_diff) 

def plot_wrapper(df, t2, value_field, countries):
    df_ = df[df["age_sex"] == t2]
    if len(df_) > 0:
        plot(df_, t2, value_field, countries)
    else:
        st.info(f"No data - {t2}")

   
def interface_opdeling():
    def ends_in_4_9_or_120(number):
    # Check if the number ends in 4 or 9, or is exactly 120
        return number % 10 in {4, 9} or number == 120

    def ends_in_5_0_or_120(number):
    # Check if the number ends in 4 or 9, or is exactly 120
        return number % 10 in {5, 0} or number == 120
    # Get data for all selected countries and concatenate them
    
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        b1,b2 = st.columns(2)
        with b1:
            l1 = st.number_input("low1", 0,120,0)
        with b2:
            h1 = st.number_input("high1", 0,120,14)
    with col2:
        c1,c2 = st.columns(2)
        with c1:
            l2 = st.number_input("low2", 0,120,15)
        with c2:
            h2 = st.number_input("high2", 0,120,64)
    with col3:
        d1,d2 = st.columns(2)
        with d1:
            l3 = st.number_input("low3", 0,120,65)
        with d2:
            h3 = st.number_input("high3", 0,120,79)
    with col4:
        e1,e2 = st.columns(2)
        with e1:
            l4 = st.number_input("low4", 0,120,80)
        with e2:
            h4 = st.number_input("high4", 0,120,120)

    
    fout = False
    for l in [l1,l2,l3,l4]:
        if not ends_in_5_0_or_120(l):
            st.error(f"low number **{l}** is not compatible")
            fout = True
    for h in [h1,h2,h3,h4]:
        if not ends_in_4_9_or_120(h):
            st.error(f"high number **{h}** is not compatible")
            fout = True
    if fout:
        st.info("Please correct values")
        st.stop()

    opdeling = [[l1,h1],[l2,h2],[l3,h3],[l4,h4]]
    return opdeling

def main():
    st.title("Deaths in age groups ")
    
    # Let the user select one or both countries
    countries = ["NL"] # st.multiselect("Country [NL | BE]", ["NL", "BE"], default=["NL", "BE"])
 
    opdeling = interface_opdeling()

    df_list = []
    for country in countries:
        df = get_sterfte(opdeling, country)
        df["country"] = country  # Add a column to distinguish the countries
        df_list.append(df)
    
    df_combined = pd.concat(df_list)
    
    # Plot the data for both countries
    to_do = unique_values = df_combined["age_sex"].unique()
    #labels = ['TOTAL']+["Y0-19"]+["Y20-64"]+["Y65-79"]+["Y80-120"] + ['Y_LT5'] + [f'Y{i}-{i+4}' for i in range(5, 90, 5)] + ['Y_GE90']
    labels = ['TOTAL'] + [f'Y{start}-{end}' for start, end in opdeling] + ['Y_LT5'] + [f'Y{i}-{i+4}' for i in range(5, 90, 5)] + ['Y_GE90']
  
    colx, coly = st.columns(2)
    with colx:
        value_field = st.selectbox("Value field (per 100.000 | absolute value| percentage (based on per 100k))", ["per100k", "OBS_VALUE","percentage"], 0)
    with coly:
        how = st.selectbox("How (all from one year | compare startyears)", ["all from one year", "compare startyears"], 1)
    
    if how == "all from one year":
        start = st.number_input("Startjaar", 2000, 2020, 2000)
        df_combined = df_combined[df_combined["jaar"] >= start]
        
        for t in labels:
            col1, col2, col3 = st.columns(3)
            with col1:
                t2 = f"{t}_T"
                plot_wrapper(df_combined, t2, value_field, countries)
            with col2:
                t2 = f"{t}_M"
                plot_wrapper(df_combined, t2, value_field, countries)
            with col3:
                t2 = f"{t}_F"
                plot_wrapper(df_combined, t2, value_field, countries)
    else:
        y = st.selectbox("Which category (T=all, M=Male, F=Female)", ["T", "M", "F"], 0)
        for x in labels:
            col1, col2, col3 = st.columns(3)
            t2 = f"{x}_{y}"
            with col1:
                df_ = df_combined[df_combined["jaar"] >= 2000]
                plot_wrapper(df_, t2, value_field, countries)
            with col2:
                df_ = df_combined[df_combined["jaar"] >= 2010]
                plot_wrapper(df_, t2, value_field, countries)
            with col3:
                df_ = df_combined[df_combined["jaar"] >= 2015]
                plot_wrapper(df_, t2, value_field, countries)


    st.subheader("Databronnen")
    st.info("Bevolkingsgrootte NL: https://opendata.cbs.nl/#/CBS/nl/dataset/03759ned/table?dl=39E0B")
    st.info("Bevolkingsgrootte BE:https://ec.europa.eu/eurostat/databrowser/view/demo_pjan__custom_12780094/default/table?lang=en")
    st.info("Sterfte: https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")
    st.info("See also https://www.mortality.watch/explorer/?c=NLD&t=cmr&e=1&df=2010&dt=2023&ag=all&ce=0&st=1&pi=0&p=1")


if __name__ == "__main__":
    print ("gooo")
    main()
