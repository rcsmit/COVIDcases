import pandas as pd
import cbsodata


import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
# import platform
from oversterfte_helpers import *
from sterfte_rivm import *
import get_rioolwater
# from streamlit import caching

# 70895ned = https://opendata.cbs.nl/#/CBS/nl/dataset/70895ned/table?ts=1659307527578
# Overledenen; geslacht en leeftijd, per week

# Downloaden van tabeloverzicht
# toc = pd.DataFrame(cbsodata.get_table_list())
try:
    st.set_page_config(layout="wide")
except:
    pass

# Downloaden van gehele tabel (kan een halve minuut duren)


def interface():
    how = st.sidebar.selectbox("How", ["quantiles", "Lines", "over_onder_sterfte", "meer_minder_sterfte", "year_minus_avg", "p_score"], index = 0)
    yaxis_to_zero = st.sidebar.selectbox("Y as beginnen bij 0", [False, True], index = 0)
    if (how == "year_minus_avg") or (how == "p_score") or (how == "over_onder_sterfte") or (how == "meer_minder_sterfte") :
        rightax = st.sidebar.selectbox("Right-ax", ["boosters", "herhaalprik", "herfstprik", "rioolwater", "kobak", None], index = 1, key = "aa")
        mergetype = st.sidebar.selectbox("How to merge", ["inner", "outer"], index = 0, key = "bb")
        sec_y = st.sidebar.selectbox("Secondary Y axis", [True, False], index = 0, key = "cc")
    else:
        rightax = None
        mergetype = None
        sec_y = None
    return how,yaxis_to_zero,rightax,mergetype, sec_y


def calculate_year_data(df_merged, year, show_official, series_name):
    st.subheader(year)
    if year != "All":
        df_merged_jaar = df_merged[df_merged["jaar_x_x"] == year].copy()
    else:
        df_merged_jaar = df_merged.copy()

    columns = [
        [series_name, "aantal_overlijdens"],
        ["q50", "verw_cbs"],
        ["low05", "low_cbs"],
        ["high95", "high_cbs"],
        ["voorspeld", "verw_rivm"],
        ["lower_ci", "low_rivm"],
        ["upper_ci", "high_rivm"]
    ]

    for c in columns:
        df_merged_jaar = df_merged_jaar.rename(columns={c[0]: c[1]})

    show_difference(df_merged_jaar, "weeknr", show_official)

    for n in ['cbs', 'rivm']:
        df_merged_jaar[f"oversterfte_{n}_simpel"] = df_merged_jaar["aantal_overlijdens"] - df_merged_jaar[f"verw_{n}"]
        df_merged_jaar[f"oversterfte_{n}_simpel_cumm"] = df_merged_jaar[f"oversterfte_{n}_simpel"].cumsum()
        
        df_merged_jaar[f'oversterfte_{n}_complex'] = np.where(
            df_merged_jaar['aantal_overlijdens'] > df_merged_jaar[f'high_{n}'], 
            df_merged_jaar['aantal_overlijdens'] - df_merged_jaar[f'high_{n}'], 
            np.where(
                df_merged_jaar['aantal_overlijdens'] < df_merged_jaar[f'low_{n}'], 
                df_merged_jaar['aantal_overlijdens'] - df_merged_jaar[f'low_{n}'], 
                0
            )
        )
        
        df_merged_jaar[f'oversterfte_{n}_middel'] = np.where(
            df_merged_jaar['aantal_overlijdens'] > df_merged_jaar[f'high_{n}'], 
            df_merged_jaar['aantal_overlijdens'] - df_merged_jaar[f'verw_{n}'], 
            np.where(
                df_merged_jaar['aantal_overlijdens'] < df_merged_jaar[f'low_{n}'], 
                df_merged_jaar['aantal_overlijdens'] - df_merged_jaar[f'verw_{n}'], 
                0
            )
        )

        df_merged_jaar[f"oversterfte_{n}_complex_cumm"] = df_merged_jaar[f"oversterfte_{n}_complex"].cumsum()
        df_merged_jaar[f"oversterfte_{n}_middel_cumm"] = df_merged_jaar[f"oversterfte_{n}_middel"].cumsum()
    return df_merged_jaar
    

def display_cumulative_oversterfte(df_merged_jaar, year):
   
    cbs_middel = df_merged_jaar['oversterfte_cbs_middel_cumm'].iloc[-1]
    cbs_simpel = df_merged_jaar['oversterfte_cbs_simpel_cumm'].iloc[-1]
    cbs_complex = df_merged_jaar['oversterfte_cbs_complex_cumm'].iloc[-1]

    rivm_middel = df_merged_jaar['oversterfte_rivm_middel_cumm'].iloc[-1]
    rivm_simpel = df_merged_jaar['oversterfte_rivm_simpel_cumm'].iloc[-1]
    rivm_complex = df_merged_jaar['oversterfte_rivm_complex_cumm'].iloc[-1]
    try:
        simpel_str = f"Simpel: rivm: {int(rivm_simpel)} | cbs: {int(cbs_simpel)} | verschil {int(rivm_simpel-cbs_simpel)}"
        middel_str = f"Middel: rivm: {int(rivm_middel)} | cbs: {int(cbs_middel)} | verschil {int(rivm_middel-cbs_middel)}"
        complex_str = f"Complex: rivm: {int(rivm_complex)} | cbs: {int(cbs_complex)} | verschil {int(rivm_complex-cbs_complex)}"
        texts = [simpel_str, middel_str, complex_str]
    except:
        texts = [None,None,None]
    temp1 = [None, None, None]
    col1, col2, col3 = st.columns(3)
    temp1[0], temp1[1], temp1[2] = col1, col2, col3

    for i, p in enumerate(['simpel', 'middel', 'complex']):
        with temp1[i]:
            fig = go.Figure()
            for n in ['rivm', 'cbs']:
                fig.add_trace(go.Scatter(
                    x=df_merged_jaar['weeknr'],
                    y=df_merged_jaar[f'oversterfte_{n}_{p}_cumm'],
                    mode='lines',
                    name=f'cummulatieve oversterfte {n}'
                ))

            fig.update_layout(
                title=f'Cumm oversterfte ({p}) - {year}',
                xaxis_title='Tijd',
                yaxis_title='Aantal'
            )

            st.plotly_chart(fig)
            st.write(texts[i])

def display_results(df_merged_jaar, year):
    st.subheader(f"Results - {year}")

    df_grouped = df_merged_jaar.groupby(by="jaar_x_x").sum().reset_index()
    df_grouped = df_grouped[[
        "jaar_x_x",
        "oversterfte_rivm_simpel",
        "oversterfte_rivm_middel",
        "oversterfte_rivm_complex",
        "oversterfte_cbs_simpel",
        "oversterfte_cbs_middel",
        "oversterfte_cbs_complex",
    ]]

    for x in ['simpel', 'middel', 'complex']:
        df_grouped[f"verschil_{x}"] = df_grouped[f"oversterfte_rivm_{x}"] - df_grouped[f"oversterfte_cbs_{x}"]

    df_grouped_transposed = df_grouped.transpose().astype(int)

    if year == "All":
        st.write(df_grouped_transposed)
    else:
        new_data = {
            'rivm': {
                'simpel': df_grouped['oversterfte_rivm_simpel'].iloc[0],
                'middel': df_grouped['oversterfte_rivm_middel'].iloc[0],
                'complex': df_grouped['oversterfte_rivm_complex'].iloc[0],
            },
            'cbs': {
                'simpel': df_grouped['oversterfte_cbs_simpel'].iloc[0],
                'middel': df_grouped['oversterfte_cbs_middel'].iloc[0],
                'complex': df_grouped['oversterfte_cbs_complex'].iloc[0],
            },
            'verschil': {
                'simpel': df_grouped['verschil_simpel'].iloc[0],
                'middel': df_grouped['verschil_middel'].iloc[0],
                'complex': df_grouped['verschil_complex'].iloc[0],
            }
        }

        new_df_grouped = pd.DataFrame(new_data).transpose().astype(int)
        st.write(new_df_grouped)

def make_df_merged(df_sterfte, series_name):
    df_data = get_data_for_series(df_sterfte, series_name).copy(deep=True)
    df_corona, df_quantile = make_df_qantile(series_name, df_data)
    df_official = get_df_offical()
    df_rivm = sterfte_rivm(df_sterfte, series_name)
    



    df_merged = df_corona.merge(df_quantile, left_on='weeknr', right_on='week_').merge(df_rivm, on='weeknr', how="outer")
    df_merged = df_merged.merge(df_official, left_on='weeknr', right_on='weeknr_z', how="outer")
    df_merged = df_merged.drop(columns=['week_'])
  
    df_merged["shifted_jaar"] = df_merged["jaar_x_x"] #.shift(28)
    df_merged["shifted_week"] = df_merged["weeknr"]#.shift(28)
    return df_merged

def plot_steigstra(df_transformed, series_name):
  
    # Pivot table
    df_pivot = df_transformed.set_index('week')

    # Function to transform the DataFrame
    def create_spaghetti_data(df, year1, year2):
     
        part1 = df.loc[28:52, year1]
        part2 = df.loc[1:27, year2]
        combined = pd.concat([part1, part2]).reset_index(drop=True)
   
        return combined

    # Create the spaghetti data
    years = df_pivot.columns[:-3]
 
    years = list(map(int, years))
    spaghetti_data = {year: create_spaghetti_data(df_pivot, year, year + 1) for year in years if (year + 1) in years}
    df_spaghetti = pd.DataFrame(spaghetti_data)

    df_spaghetti = df_spaghetti.cumsum(axis=0)


    # Generate the sequence from 27 to 52 followed by 1 to 26
    sequence = list(range(27, 53)) + list(range(1, 27))
    # Add the sequence as a new column
    df_spaghetti['weeknr_real'] = sequence

    #df_spaghetti.set_index('New_Column', inplace=True)
   
   # Calculate average for the first 5 columns
    df_spaghetti['average'] = df_spaghetti.iloc[:, :4].mean(axis=1)

    # Calculate low and high (mean - 1.96*std and mean + 1.96*std) for the first 5 columns
    df_spaghetti['low'] = df_spaghetti['average'] - 1.96 * df_spaghetti.iloc[:, :4].std(axis=1)
    df_spaghetti['high'] = df_spaghetti['average'] + 1.96 * df_spaghetti.iloc[:, :4].std(axis=1)
  
    # Plotting with Plotly
    fig = go.Figure()
    fig.add_vline(x=27, line=dict(color='gray', width=1, dash='dash'))

    for year in df_spaghetti.columns[:-4]:
        fig.add_trace(go.Scatter(
            x=df_spaghetti.index,
            y=df_spaghetti[year],
            mode='lines',
            name=f'{year} - {year+1}'
        ))

    fig.add_trace( go.Scatter(
                name='low',
                x=df_spaghetti.index,
                y=df_spaghetti["low"],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)',
               ))
    fig.add_trace(go.Scatter(
                name='high',
                x=df_spaghetti.index,
                y=df_spaghetti["high"],
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 188, 0, 0.1)"), fill='tonexty',
                
                fillcolor='rgba(68, 68, 68, 0.2)',)),
        
    fig.update_layout(
        title=f'Steigstra Plot of {series_name} over Different Years week 27 to week 26',
        xaxis_title='Weeks (28 to 27)',
        yaxis_title='Values',
    )

    st.plotly_chart(fig)

def calculate_steigstra(df_merged, series_naam, cumm=False, m="cbs"):

   

    # Set 'week' to 52 if 'weeknr' is '2022_52'

    # Get list of current columns excluding "shifted_jaar"
    other_columns = [col for col in df_merged.columns if col != "shifted_jaar"]
    # Reorder columns to put "shifted_jaar" at the beginning
    new_columns = ["shifted_jaar"] + other_columns

    # Reindex DataFrame with new column order
    df_merged = df_merged.reindex(columns=new_columns)
   
   
    columns = [
        [series_naam, "aantal_overlijdens"],
        ["q50", "verw_cbs"],
        ["low05", "low_cbs"],
        ["high95", "high_cbs"],
        ["voorspeld", "verw_rivm"],
        ["lower_ci", "low_rivm"],
        ["upper_ci", "high_rivm"]
    ]

    for c in columns:
        df_merged = df_merged.rename(columns={c[0]: c[1]})
    df_compleet = pd.DataFrame()
    for year in range(2015, 2025):
        df_merged_jaar = df_merged[df_merged["jaar_x_x"] == year].copy()
        for n in ['cbs']:
            df_merged_jaar[f"oversterfte_{n}_simpel"] = df_merged_jaar["aantal_overlijdens"] - df_merged_jaar[f"verw_{n}"]
            df_merged_jaar[f"oversterfte_{n}_simpel_cumm"] = df_merged_jaar[f"oversterfte_{n}_simpel"].cumsum()
            df_compleet =  pd.concat([df_compleet,df_merged_jaar])
  

    df_compleet["shifted_jaar"] = df_compleet["jaar_x_x"] #.shift(24)
   
    if cumm:
        df = df_compleet.pivot(index=['week'], columns="shifted_jaar", values = f'oversterfte_{m}_simpel_cumm').reset_index()
    else:
        df = df_compleet.pivot(index=['week'], columns="shifted_jaar", values = f'oversterfte_{m}_simpel').reset_index()
        
    # Calculate average and standard deviation
    df['average'] = df.mean(axis=1)
   
    # Calculate low and high (mean - 1.96*std and mean + 1.96*std)
    df['low'] = df['average'] - 1.96 * df.std(axis=1)
    df['high'] = df['average'] + 1.96 *  df.std(axis=1) 
    
    return df

def comparison(df_sterfte):
    show_official = st.sidebar.selectbox("Show official values", [True, False], 1)
    st.subheader("Vergelijking")

    series_name = "m_v_0_999"
    
    df_merged = make_df_merged(df_sterfte, series_name)

    for year in ["All", 2020, 2021, 2022, 2023, 2024]:
        df_merged_jaar = calculate_year_data(df_merged, year, show_official, series_name)
        display_cumulative_oversterfte(df_merged_jaar, year)
        display_results(df_merged_jaar, year)
   
    df=calculate_steigstra(df_merged, series_name)
    plot_steigstra(df, series_name)


def main():
    # serienames = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", 
    #                 "m_0_999",  "m_0_64",  "m_65_79",  "m_80_999",
    #                 "v_0_999",  "v_0_64",  "v_65_79",  "v_80_999"]

    serienames_ = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", 
                    "m_0_999",  "m_0_64",  "m_65_79",  "m_80_999",
                    "v_0_999",  "v_0_64",  "v_65_79",  "v_80_999"]

    serienames = st.sidebar.multiselect("Leeftijden", serienames_, ["m_v_0_999"])
    st.header("Oversterfte - minder leeftijdscategorieen")
    st.subheader("CBS Methode")
    st.write("Dit script heeft minder leeftijdscategorieen, maar de sterftedata wordt opgehaald van het CBS. Daarnaast wordt het 95% betrouwbaarheids interval berekend vanuit de jaren 2015-2019")
    how, yaxis_to_zero, rightax, mergetype, sec_y = interface()
    df_sterfte, df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater, df_kobak = get_all_data()
    
    # Define a list of tuples with arguments for duplicate_row function
    duplicate_operations = [
        ("2020_02", "2020_01"),
        ("2021_51", "2021_52"),
        ("2022_51", "2022_52"),
        ("2019_02", "2019_01"),
        ("2015_02", "2015_01"),
        ("2016_51", "2016_52")
    ]

    # Iterate over the list and apply duplicate_row function to df_sterfte
    for operation in duplicate_operations:
        df_sterfte = duplicate_row(df_sterfte, operation[0], operation[1])
    plot(df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_sterfte, df_kobak, serienames, how, yaxis_to_zero, rightax, mergetype, sec_y)
    if how == "quantiles":
        st.subheader("RIVM methode")
        for s in serienames:
            df_compleet = sterfte_rivm(df_sterfte, s)
         
            plot_graph_rivm(df_compleet,s, False)
        
        comparison(df_sterfte)
    else:
        st.info("De vergrlijking met vaccinateies, rioolwater etc is vooralsnog alleen mogelijk met CBS methode ")
    footer()


if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()

   
