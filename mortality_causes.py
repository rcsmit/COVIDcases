from fit_to_data_streamlit import *
from mortality_yearly_per_capita import get_sterfte, get_bevolking, interface_opdeling
#from oversterfte_compleet import
import streamlit as st
from scipy.optimize import curve_fit
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
import cbsodata
import numpy as np
import plotly.express as px

#def get_cbs_data():


    
@st.cache_data(ttl=60 * 60 * 24)
def get_sterftedata():
    """Get and manipulate data of the deaths

    Args:
        seriename (str, optional): _description_. Defaults to "m_v_0_999".
    """

    def manipulate_data_df(data):
        """Filters out week 0 and 53 and makes a category column (eg. "M_V_0_999")"""

        # data = data[~data['week'].isin([0, 53])] #filter out week 2020-53
        data["weeknr"] = (
            data["jaar"].astype(str) + "_" + data["week"].astype(str).str.zfill(2)
        )

        data["sex"] = data["Geslacht"].replace(
            ["Totaal mannen en vrouwen"], "T"
        )
        data["sexe"] = data["Geslacht"].replace(["Mannen"], "M")
        data["sexe"] = data["Geslacht"].replace(["Vrouwen"], "F")
        data["age"] = data["LeeftijdOp31December"].replace(
            ["Totaal leeftijd"], "TOTAL"
        )
        data["age"] = data["LeeftijdOp31December"].replace(
            ["0 tot 65 jaar"], "Y0_64"
        )
        data["age"] = data["LeeftijdOp31December"].replace(
            ["65 tot 80 jaar"], "Y65_79"
        )
        data["age"] = data["LeeftijdOp31December"].replace(
            ["80 jaar of ouder"], "Y80_999"
        )
        
        return data

 

    data_ruw = pd.DataFrame(cbsodata.get_data("70895ned"))


    data_ruw[["jaar", "week"]] = data_ruw.Perioden.str.split(
        " week ",
        expand=True,
    )
    data_ruw = manipulate_data_df(data_ruw)
    data_ruw["jaar"] = data_ruw["jaar"].astype(int)
   

    print (data_ruw)
    data_bevolking = pd.DataFrame(cbsodata.get_data("03759ned"))
    print (data_bevolking)



@st.cache_data()
def get_data(opdeling) -> pd.DataFrame:
    """
    Fetch mortality data using `get_sterfte` function with age group breakdown.

    Returns:
        pd.DataFrame: A DataFrame containing mortality data for different age groups.
    """
    # put in a seperate function to enable caching
  
    df =  get_sterfte(opdeling, "NL")
    return df

@st.cache_data()
def get_doodsoorzaken_cbs():
    data = pd.DataFrame(cbsodata.get_data('7052_95'))
    # data = pd.DataFrame(cbsodata.get_data('7233'))
    
    return data

@st.cache_data()
def get_doodsoorzaken(opdeling) -> pd.DataFrame:
 
    data= get_doodsoorzaken_cbs()
    
    # Melting the dataframe with all columns except the first four
    df = data.melt(id_vars=['ID', 'Geslacht', 'Leeftijd', 'Perioden'], 
                        value_vars=data.columns.difference(['ID', 'Geslacht', 'Leeftijd', 'Perioden']), 
                        var_name='doodsoorzaak', 
                        value_name='OBS_VALUE')
    
        # Wijzigen van de waarden in de kolom 'Geslacht'
    df['Geslacht'] = df['Geslacht'].replace({
        'Mannen': 'M',
        'Vrouwen': 'F',
        'Totaal mannen en vrouwen': 'T'
    })

    # Hernoemen van de kolom 'Geslacht' naar 'Sexe'
    df = df.rename(columns={'Geslacht': 'Sexe'})

    import re

    # Vervangen van specifieke waarden
    df['Leeftijd'] = df['Leeftijd'].replace({
        'Totaal alle leeftijden': 'Total',
        '0 jaar': 'Y0-4',

        
        '90 tot 95 jaar':"Y90-120",
        '95 jaar of ouder':"Y90-120"
    })

    # Functie om leeftijdsintervallen te hernoemen
    def format_age_group(leeftijd):
        pattern = r'(\d+) tot (\d+) jaar'
        match = re.match(pattern, leeftijd)
        if match:
            low_age = int(match.group(1))
            high_age = int(match.group(2)) - 1
            return f"Y{low_age}-{high_age}"
        return leeftijd

    # Toepassen van de functie op de 'Leeftijd' kolom
    df['Leeftijd'] = df['Leeftijd'].apply(format_age_group)
    
    # Hernoemen van de kolom 'Leeftijd' naar 'age_group'
    df = df.rename(columns={'Leeftijd': 'age_group'})
    # Groeperen op 'ID', 'Sexe', 'age_group', 'Perioden', en 'doodsoorzaak' en 'OBS_VALUE' optellen
    df = df.groupby(['Sexe', 'age_group', 'Perioden', 'doodsoorzaak'], as_index=False)['OBS_VALUE'].sum()
    df = df.rename(columns={'Perioden': 'jaar'})
    df = df.rename(columns={'Sexe': 'geslacht'})
    df["jaar"]= df["jaar"].astype(int)
    df=df[df["jaar"]>1999]
    
    #opdeling = [[0,19],[20,64],[65,79],[80,120]]
    df_bevolking = get_bevolking("NL", opdeling)
    

    
      
    # Function to extract age_low and age_high based on patterns
    def extract_age_ranges(age):
        if age == "Total":
            return 999, 999
        elif age == "UNK":
            return 9999, 9999
        elif age == "Y_LT5":
            return 0, 4
        elif age == "Y_90-120":
            return 90, 120
        else:
            # Extract the numeric part from the pattern 'Y10-14'
            parts = age[1:].split('-')
            return int(parts[0]), int(parts[1])

    # Apply the function to create the new columns
    df['age_low'], df['age_high'] = zip(*df['age_group'].apply(extract_age_ranges))

  

    def add_custom_age_group_deaths(df_, min_age, max_age):
        # Filter the data based on the dynamic age range
        df_filtered = df[(df['age_low'] >= min_age) & (df['age_high'] <= max_age)]

        # Group by TIME_PERIOD (week), sex, and sum the deaths (OBS_VALUE)
        totals = df_filtered.groupby(['jaar', 'geslacht','doodsoorzaak'], observed=False)['OBS_VALUE'].sum().reset_index()

        # Assign a new label for the age group (dynamic)
        totals['age_group'] = f'Y{min_age}-{max_age}'
        totals["age_sex"] = totals["age_group"] + "_" +totals["geslacht"]
        #totals["jaar"] = (totals["TIME_PERIOD"].str[:4]).astype(int)
        return totals
    
    df_custom_age_groups = pd.DataFrame()


    for i in opdeling:
        custom_age_group = add_custom_age_group_deaths(df, i[0], i[1])
        df_custom_age_groups = pd.concat([df_custom_age_groups, custom_age_group], ignore_index=True)

    df = pd.concat([df_custom_age_groups, df], ignore_index=True)

   
    df_eind = pd.merge(df, df_bevolking, on=['geslacht', 'age_group', 'jaar'], how = "left")
    
    df_eind = df_eind[df_eind["aantal"].notna()]
    df_eind = df_eind[df_eind["OBS_VALUE"].notna()]
    df_eind = df_eind[df_eind["jaar"] != 2024]
    df_eind["per100k"] = round(df_eind["OBS_VALUE"]/df_eind["aantal"]*100000,1) 
    
    return df_eind 
    
def main() -> None:
    """
    Main function for the Streamlit application that analyzes mortality data using linear and 
    secondary fitting models.

    Args:
        None

    Returns:
        None
    """
    st.subheader("Doodsoorzaken door det ijd heen")
    
    # choice = st.sidebar.selectbox("Overlijdens of doodsoorzaken",["overlijdens", "doodsoorzaken"],0)
    #opdeling = [[0,49], [50,64], [65,79], [80,89], [90,120],[80,120], [0,120]]
    opdeling = [[0,120], [0,64],[65,79],[80,120]] + interface_opdeling() 
    df_doodsoorzaken = get_doodsoorzaken(opdeling)
    df_doodsoorzaken = df_doodsoorzaken[(df_doodsoorzaken['age_sex_x'] =="Y0-120_T")& (df_doodsoorzaken['doodsoorzaak'] != "TotaalOnderliggendeDoodsoorzaken_1")& (df_doodsoorzaken['doodsoorzaak'].str.contains('totaal', case=False, na=False))]
    #st.write(df_doodsoorzaken)

    print (df_doodsoorzaken.dtypes)
    criterium = "per100k"
    sankey_diagram_ranking(df_doodsoorzaken, criterium)
    # sankey_diagram (df_doodsoorzaken, criterium)
    # line_graph(df_doodsoorzaken, criterium)

def sankey_diagram_ranking(df, criterium):
    col1,col2=st.columns(2)
    with col1:
        min=st.number_input("Minimum",2000,2023,2020)
    with col2:
        max=st.number_input("Maximum (incl)",2000,2023,2023)


    df=df[(df["jaar"] >=min) & (df["jaar"] <= max) ].copy(deep=True)
   
    df['OBS_VALUE'] = df['OBS_VALUE'].replace(0, 1)
    # Group by 'doodsoorzaak' and 'jaar', summing the 'OBS_VALUE'
    grouped_df = df.groupby(['doodsoorzaak', 'jaar'])['OBS_VALUE'].sum().reset_index()

    # Rank causes within each year, keeping top 10 for each year
    grouped_df['rank'] = grouped_df.groupby('jaar')['OBS_VALUE'].rank(ascending=False, method='first')
    ranked_df = grouped_df[grouped_df['rank'] <= 100]
    
    # Sort by year and rank (to ensure nodes are created in ranked order)
    ranked_df = ranked_df.sort_values(by=['jaar', 'rank'])

    pivot_table = ranked_df.pivot_table(
        index='doodsoorzaak',  # Rows
        columns='jaar',        # Columns
        values='rank',    # Values in the table
        aggfunc='sum',         # Aggregate function (in case there are duplicates)
        fill_value=0           # Fill missing values with 0
    )

    st.write(pivot_table)
    # Create unique labels for each cause and year combination, ordered by rank
    nodes = []
    nodes2 =[]
    positions = []  # Store (x, y) positions

    years = sorted(ranked_df['jaar'].unique())  # Sort the years in ascending order
    years = years  # Drop the last year (removes the last element)
    no_years = len(years)
    # Parameters
    start = 0.01  # First position SANKEY CAN NOT HAVE 0 AS VALUE
    end = 0.99   # Last position
    n_items = len(years)   # Number of items
    gap = (end - start) / (n_items - 1)
    for x_pos, year in enumerate(years) :
        year_df = ranked_df[ranked_df['jaar'] == year].sort_values(by='rank')
        
        nodes3 = year_df["doodsoorzaak"].tolist()
        nodes2.append(nodes3)
     
        # Assign x and y positions for the nodes
        for y_pos, cause in enumerate(nodes3):
            nodes.append(f"{cause}")
            positions.append((start + (x_pos * gap), ((y_pos)/(len(year_df)))+0.01))
   

    # Prepare the links (flows) for the Sankey diagram
    sources = []
    values = []

    targets_new = []
    basis = 20

    # Loop through the nodes for each year in nodes2
    for i,n in enumerate(nodes2):
        if i < (len(nodes2)-1):
           
            for m in range(len(n)):  # Loop through each item in the current year's list (n)
                bron = nodes2[i][m]  # Current node in year n[m]
                if bron in nodes2[i+1]:  # Ensure the item exists in the next year's list
                    x = nodes2[i+1].index(bron) 
                    targets_new.append(basis + x)  # Store the target index for the link
                else:
                    st.write(f"{bron} not found")

            basis += len(n)  # Update basis to account for the length of the current year's node list
            
        else:
            targets_new.append(basis + i)
    # Create links between causes in consecutive years
    teller =0
    for year in ranked_df['jaar'].unique()[:-1]:
        current_year_df = ranked_df[ranked_df['jaar'] == year]
        next_year_df = ranked_df[ranked_df['jaar'] == year + 1]

        for _, row in current_year_df.iterrows():
            cause = row['doodsoorzaak']
            next_year_row = next_year_df[next_year_df['doodsoorzaak'] == cause] #hier zit de fout

            if not next_year_row.empty:
                # source = node_map[f"{cause} ({year})"]
                target = None # node_map[f"{cause} ({year + 1})"]
                
                # Add source, target, and value (OBS_VALUE)
                sources.append(teller)
                # targets.append(target)
                values.append(row['OBS_VALUE'])  # Mortality counts

                teller+=1
       
    # Separate x and y positions for nodes
    x_positions, y_positions = zip(*positions)
   
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            
            x=x_positions,  # Custom x positions for each node
            y=y_positions,  # Custom y positions for each node
            #color="blue"  # Set node colors for better visibility
        ),arrangement='fixed',
        link=dict(
            source=sources,  # Indices of source nodes
            target=targets_new,  # Indices of target nodes
            value=values,    # Flow values (OBS_VALUE)
            #color="lightgray"  # Set link colors for better visibility
        )
    ))

    # Update layout to have vertical "lines" for each year
    



    fig.update_layout(title_text="Ranking van Doodsoorzaken door de Tijd 2020-2023", font_size=10)
    st.plotly_chart(fig)
    
def sankey_diagram (df_doodsoorzaken, criterium):
    # Group by 'doodsoorzaak' and 'jaar', summing the criterium
    grouped_df = df_doodsoorzaken.groupby(['doodsoorzaak', 'jaar'])[criterium].sum().reset_index()

    # Find the 10 highest mortality causes
    top10_causes = grouped_df.groupby('doodsoorzaak')[criterium].sum().nlargest(100).index

    # Filter the dataframe to include only the top 10 causes
    filtered_df = grouped_df[grouped_df['doodsoorzaak'].isin(top10_causes)]

    # Create unique labels for causes and years
    causes = filtered_df['doodsoorzaak'].unique().tolist()
    years = filtered_df['jaar'].unique().tolist()

    # Create a list of all unique labels (nodes in the Sankey diagram)
    all_labels = causes + [str(year) for year in years]

    # Map causes and years to their index positions
    cause_map = {cause: i for i, cause in enumerate(causes)}
    year_map = {str(year): i + len(causes) for i, year in enumerate(years)}

    # Define the source (causes) and target (years) nodes
    sources = [cause_map[row['doodsoorzaak']] for _, row in filtered_df.iterrows()]
    targets = [year_map[str(row['jaar'])] for _, row in filtered_df.iterrows()]

    # Define the values (OBS_VALUE, representing the flow/mortality counts)
    values = filtered_df[criterium].tolist()

    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
        ),
        link=dict(
            source=sources,  # Indices of the cause nodes
            target=targets,  # Indices of the year nodes
            value=values,    # Mortality counts (OBS_VALUE)
        )
    ))

    # Update layout
    fig.update_layout(title_text=f"Sankey Diagram of Top 10 Mortality Causes Through Time - {criterium}", font_size=10)

    # Show the diagram
    st.plotly_chart(fig)# fig.show()

def line_graph(df_doodsoorzaken, criterium):
    # Group by 'doodsoorzaak' and 'jaar', summing the criterium
    grouped_df = df_doodsoorzaken.groupby(['doodsoorzaak', 'jaar'])[criterium].sum().reset_index()

    # Find the 10 highest mortality causes (top 10 causes overall)
    top10_causes = grouped_df.groupby('doodsoorzaak')[criterium].sum().nlargest(10).index

    # Filter the dataframe to keep only the top 10 causes
    filtered_df = grouped_df[grouped_df['doodsoorzaak'].isin(top10_causes)]

    # Create a line plot for the top 10 causes through time
    fig = px.line(filtered_df, x='jaar', y=criterium, color='doodsoorzaak',
                title=f'Top 10 Highest Mortality Causes Through Time - {criterium}',
                labels={criterium: 'Mortality Count', 'jaar': 'Year', 'doodsoorzaak': 'Cause of Death'})

    # Show the plot
    st.plotly_chart(fig)
if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')

    print(f"--------------{datetime.datetime.now()}-------------------------")

    
    main()
    #get_sterftedata()