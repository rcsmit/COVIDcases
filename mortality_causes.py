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
import random


# https://chatgpt.com/c/670e84f2-de08-8004-84dd-e071bdc3acff
# https://claude.ai/chat/6c14dc65-6703-4d7e-9d4a-b6b74176269e


    
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
            ["80 jaar of ouder"], "Y80_120"
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
    #st.write(df)
    df =df[df['Leeftijd'] != "Totaal alle leeftijden"]
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
    df['Leeftijd'] = df['Leeftijd'].replace("Y1-4", "Y0-4")
    
    # Hernoemen van de kolom 'Leeftijd' naar 'age_group'
    df = df.rename(columns={'Leeftijd': 'age_group'})
    # Groeperen op 'ID', 'Sexe', 'age_group', 'Perioden', en 'doodsoorzaak' en 'OBS_VALUE' optellen
    df = df.groupby(['Sexe', 'age_group', 'Perioden', 'doodsoorzaak'], as_index=False)['OBS_VALUE'].sum()
    df = df.rename(columns={'Perioden': 'jaar'})
    df = df.rename(columns={'Sexe': 'geslacht'})
    df["jaar"]= df["jaar"].astype(int)
    
    df=df[df["jaar"]>1999]
    df_bevolking = get_bevolking("NL", opdeling)
    
    # Function to extract age_low and age_high based on patterns
    def extract_age_ranges(age):
        if age == "Total":
            return 0, 120
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
    df["age_sex"] = df["age_group"] + "_" +df["geslacht"]
   
    def add_custom_age_group_deaths(df, min_age, max_age):
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
    # df_test = df[(df["doodsoorzaak"]=="GemiddeldeBevolking_96") & (df["jaar"] == 2000)]
    # st.write(df_test)
    
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
    st.subheader("Doodsoorzaken door de tijd heen")
    
    # choice = st.sidebar.selectbox("Overlijdens of doodsoorzaken",["overlijdens", "doodsoorzaken"],0)
    #opdeling = [[0,49], [50,64], [65,79], [80,89], [90,120],[80,120], [0,120]]
    col1,col2,col3,col4,col5,col6,col7=st.columns(7)
    
    with col1:
        l1=st.number_input("Custom age low",0,120,20)
    with col2:
        l2=st.number_input("Custom age high",0,120,39)
    
    opdeling = [[l1,l2],[0,120], [0,64],[65,79],[80,120]] 

    df_doodsoorzaken = get_doodsoorzaken(opdeling)
    df_doodsoorzaken['age_transformed'] = df_doodsoorzaken['age_sex_x'].str.split('_').str[0]

    # Get the unique values as a list
    unique_values = df_doodsoorzaken['age_transformed'].unique().tolist()
    with col3:
        age_chosen = st.selectbox("Choose agegroup", unique_values,1)
    with col4:
        sex_chosen = st.selectbox ("Choose sex",["T", "M", "F"],0)
    with col5:
        criterium =  st.selectbox("Chosen value", ["OBS_VALUE","per100k" ],0)
   
    with col6:
        min=st.number_input("Start year",2000,2023,2020)
    with col7:
        max=st.number_input("End year (incl)",2000,2023,2023)
    
    
    # Filter based on age, sex, year range, and doodsoorzaak containing 'totaal'
    df_doodsoorzaken = df_doodsoorzaken[
        (df_doodsoorzaken['age_sex_x'] == f"{age_chosen}_{sex_chosen}") &
        (df_doodsoorzaken["jaar"].between(min, max)) &
        (df_doodsoorzaken['doodsoorzaak'].str.contains('totaal', case=False, na=False))
    ].copy(deep=True)

    # Exclude specific 'doodsoorzaak' values using 'isin' and logical negation
    exclude_doodsoorzaken = [
        "TotaalOnderliggendeDoodsoorzaken_1",
        "TotaalKwaadaardigeNieuwvormingen_9",
        "TotaalZiektenVanDeKransvaten_44",
        "TotaalChronischeAandOndersteLucht_53",
        "TotaalChronischeLeveraandoeningen_59",
        "TotaalOngevallen_81",
        "TotaalVervoersongevallen_82"
    ]

    df_doodsoorzaken = df_doodsoorzaken[~df_doodsoorzaken['doodsoorzaak'].isin(exclude_doodsoorzaken)]
                                    
    if max<=min:
        st.error("End year can not be same or earlier than start year")
        st.stop()
    sankey_diagram_ranking(df_doodsoorzaken, criterium, min,max)
    

def sankey_diagram_ranking(df, criterium, min,max):
    """make a sankey diagram

    Args:
        df (df): df
        criterium (str): OBS_VALUE | per100k
        min (int): start year, for the graph
        max (int): end year (incl), for the graph
    """   
   
    df[criterium] = df[criterium].replace(0, 1)
    # Group by 'doodsoorzaak' and 'jaar', summing the  criterium
    grouped_df = df.groupby(['doodsoorzaak', 'jaar'])[criterium].sum().reset_index()

    # Rank causes within each year, keeping top 10 for each year
    grouped_df['rank'] = grouped_df.groupby('jaar')[criterium].rank(ascending=False, method='first')
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

    pivot_table_criterium = grouped_df.pivot_table(
        index='doodsoorzaak',  # Rows
        columns='jaar',        # Columns
        values=criterium,    # Values in the table
        aggfunc='sum',         # Aggregate function (in case there are duplicates)
        fill_value=0           # Fill missing values with 0
    )

    
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
    basis = len (nodes3)

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
                values.append(row[criterium])  # Mortality counts

                teller+=1
            else:
                print("Next row is empty")
       
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

     

    # Step 1: Flatten nodes2 to get all unique values
    all_nodes = [node for sublist in nodes2 for node in sublist]

    # Get unique values from the flattened list
    unique_nodes = list(set(all_nodes))

    # Step 1: Predefine a consistent color palette
    if 1==1:
        
        fixed_colors = [
            "#FF5733", "#33FF57", "#3357FF", "#F39C12", "#9B59B6",  # Red, Green, Blue, Orange, Purple
            "#1ABC9C", "#E74C3C", "#2ECC71", "#3498DB", "#F1C40F",  # Teal, Red, Green, Blue, Yellow
            "#8E44AD", "#E67E22", "#2980B9", "#27AE60", "#C0392B",  # Dark Purple, Orange, Dark Blue, Dark Green, Dark Red
            "#D35400", "#34495E", "#16A085", "#F39C12", "#7F8C8D",  # Orange, Dark Grey, Teal, Orange, Grey
            "#FFC300", "#FF9F00", "#A569BD", "#D1F2EB", "#7D3C98",  # Yellow, Dark Yellow, Light Purple, Light Teal, Dark Purple
            "#2C3E50", "#C5C6C7", "#FF5733", "#FF8D1C", "#D1DB00"   # Dark Blue, Light Grey, Red, Orange, Bright Yellow
        ]
        # Step 2: Assign colors to unique nodes, cycling through the predefined palette
        #unique_nodes = set(node for sublist in nodes2 for node in sublist)
        color_map = {node: fixed_colors[i % len(fixed_colors)] for i, node in enumerate(unique_nodes)}

        # Step 3: Apply the color map to the nodes
        node_colors = [color_map[node] for sublist in nodes2 for node in sublist]

       # Now you have consistent `node_colors` and `link_colors`


    if 1==2:
        # Step 2: Create a color map
        # Generate random hex colors for each unique node
        def generate_color():
            return "#{:06x}".format(random.randint(0, 0xFFFFFF))

        color_map = {node: generate_color() for node in unique_nodes}

        # Step 3: Apply the color map to the nodes
        # Each node in the sankey diagram will have a corresponding color
        node_colors = [color_map[node] for sublist in nodes2 for node in sublist]

    # Step 1: Convert hex color to RGB (0-255 range)
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Step 2: Lighten the color by 50%
    def lighten_color(rgb_color, factor=0.5):
        return tuple(int(c + (255 - c) * factor) for c in rgb_color)

    # Step 3: Convert RGB back to hex
    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(*rgb_color)

    # Step 4: Generate link_colors based on node_colors
    link_colors = []

    for color in node_colors:
        rgb = hex_to_rgb(color)  # Convert hex to RGB
        lighter_rgb = lighten_color(rgb, factor=0.5)  # Lighten the color by 50%
        link_colors.append(rgb_to_hex(lighter_rgb))  # Convert back to hex

    # Now, link_colors contains the lighter colors for each link

    #TO DO
    # color_for_nodes = ["red","green","blue","violet","maroon"]
    fig.update_traces(node_color = node_colors)
    fig.update_traces(link_color = link_colors)
    # Update layout to have vertical "lines" for each year
 

    fig.update_layout(title_text=f"Ranking van Doodsoorzaken door de Tijd {min}-{max}", font_size=10)
    st.plotly_chart(fig)

    st.write(pivot_table)
    st.write(pivot_table_criterium)
    st.info("Data source: https://opendata.cbs.nl/statline/#/CBS/nl/dataset/7052_95/table?fromstatweb")

def probleem():
 
    data= get_doodsoorzaken_cbs()
    st.write(data)   
    # Melting the dataframe with all columns except the first four
    df = data.melt(id_vars=['ID', 'Geslacht', 'Leeftijd', 'Perioden'], 
                        value_vars=data.columns.difference(['ID', 'Geslacht', 'Leeftijd', 'Perioden']), 
                        var_name='doodsoorzaak', 
                        value_name='OBS_VALUE')
    st.write(df)

if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')

    print(f"--------------{datetime.datetime.now()}-------------------------")

    main()
    #get_sterftedata()