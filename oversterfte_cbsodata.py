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

def main():
    # serienames = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", 
    #                 "m_0_999",  "m_0_64",  "m_65_79",  "m_80_999",
    #                 "v_0_999",  "v_0_64",  "v_65_79",  "v_80_999"]

    serienames_ = ["m_v_0_999","m_v_0_64","m_v_65_79","m_v_80_999", 
                    "m_0_999",  "m_0_64",  "m_65_79",  "m_80_999",
                    "v_0_999",  "v_0_64",  "v_65_79",  "v_80_999"]

    serienames = st.sidebar.multiselect("Leeftijden", serienames_, ["m_v_0_999"])
    st.header("Overstefte - minder leeftijdscategorieen")
    st.write("Dit script heeft minder leeftijdscategorieen, maar de sterftedata wordt opgehaald van het CBS. Daarnaast wordt het 95% betrouwbaarheids interval berekend vanuit de jaren 2015-2019")
    how, yaxis_to_zero, rightax, mergetype, sec_y = interface()
    df_sterfte, df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater, df_kobak = get_all_data()
    plot(df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_sterfte, df_kobak, serienames, how, yaxis_to_zero, rightax, mergetype, sec_y)
   
    footer()
    comparison(df_sterfte)
    
def comparison(df_sterfte):
    show_official = st.sidebar.selectbox("Show official values", [True,False], 1)
    st.subheader("Comparison")
    #df_sterfte, df_boosters,df_herhaalprik,df_herfstprik,df_rioolwater, df_kobak = get_all_data()
    #plot(df_boosters, df_herhaalprik, df_herfstprik, df_rioolwater, df_sterfte, df_kobak, ["m_v_0_999"],"quantiles", False, None, None, None) 
    series_name = "m_v_0_999"
    df_data = get_data_for_series(df_sterfte, series_name).copy(deep=True)
    df_corona, df_quantile = make_df_qantile(series_name, df_data) 
    df_official = get_df_offical()
    df_rivm = sterfte_rivm(df_sterfte, series_name)
    # plot_graph_rivm(df_rivm, series_name, False)

    df_corona = duplicate_row(df_corona, "2021_51", "2021_52")
    df_corona = duplicate_row(df_corona, "2022_51", "2022_52")
    # Merge de dataframes, waarbij we de kolomnaam van df_quantile aanpassen tijdens de merge
    df_merged__ = df_corona.merge(df_quantile, left_on='weeknr', right_on='week_').merge(df_rivm, on='weeknr')
    df_merged = df_merged__.merge(df_official,left_on='weeknr', right_on='weeknr_z', how="outer")
    print (df_merged.dtypes)
    print (df_official.dtypes)
    # Verwijder de extra 'week_' kolom uit het eindresultaat
    df_merged = df_merged.drop(columns=['week_'])
 
    for y in ["All",2020,2021,2022,2023,2024]:   
        st.subheader(y) 
        
        if y !="All":
            df_merged_jaar = df_merged[df_merged["jaar_x_x"] == y] .copy()  
            #df_merged_jaar = df_merged[df_merged["boekjaar_x"] == y] .copy()  
            
        else:
            df_merged_jaar = df_merged.copy()
        columns = [[series_name, "aantal_overlijdens"],
                    ["avg", "verw_cbs"],
                    ["low05", "low_cbs"],
                    ["high95", "high_cbs"],
                    ["voorspeld", "verw_rivm"],
                    ["lower_ci", "low_rivm"],
                    ["upper_ci", "high_rivm"]]
        for c in columns:
            print (c)
            df_merged_jaar =  df_merged_jaar.rename(columns={c[0]:c[1]})
        
        show_difference( df_merged_jaar, "weeknr", show_official)

        for n in ['cbs', 'rivm']:
            df_merged_jaar[f"oversterfte_{n}_simpel"] =  df_merged_jaar["aantal_overlijdens"] -  df_merged_jaar[f"verw_{n}"]
            df_merged_jaar[f"oversterfte_{n}_simpel_cumm"] =  df_merged_jaar[f"oversterfte_{n}_simpel"].cumsum()
            # Bereken de nieuwe kolom 'oversterfte'
            df_merged_jaar[f'oversterfte_{n}_complex'] = np.where(
                df_merged_jaar['aantal_overlijdens'] >  df_merged_jaar[f'high_{n}'], 
                df_merged_jaar['aantal_overlijdens'] -  df_merged_jaar[f'high_{n}'], 
                np.where(
                    df_merged_jaar['aantal_overlijdens'] <  df_merged_jaar[f'low_{n}'], 
                    df_merged_jaar['aantal_overlijdens'] -  df_merged_jaar[f'low_{n}'], 
                    0
                )
            )
            
            # Bereken de nieuwe kolom 'oversterfte'
            df_merged_jaar[f'oversterfte_{n}_middel'] = np.where(
                df_merged_jaar['aantal_overlijdens'] >  df_merged_jaar[f'high_{n}'], 
                df_merged_jaar['aantal_overlijdens'] -  df_merged_jaar[f'verw_{n}'], 
               np.where(
                    df_merged_jaar['aantal_overlijdens'] <  df_merged_jaar[f'low_{n}'], 
                    df_merged_jaar['aantal_overlijdens'] -  df_merged_jaar[f'verw_{n}'], 
                    0
                )
            )

            df_merged_jaar[f"oversterfte_{n}_complex_cumm"] =  df_merged_jaar[f"oversterfte_{n}_complex"].cumsum()
            df_merged_jaar[f"oversterfte_{n}_middel_cumm"] =  df_merged_jaar[f"oversterfte_{n}_middel"].cumsum()

        cbs_middel =  df_merged_jaar['oversterfte_cbs_middel_cumm'].iloc[-1]
        cbs_simpel =  df_merged_jaar['oversterfte_cbs_simpel_cumm'].iloc[-1]
        cbs_complex =  df_merged_jaar['oversterfte_cbs_complex_cumm'].iloc[-1]

        rivm_middel =  df_merged_jaar['oversterfte_rivm_middel_cumm'].iloc[-1]
        rivm_simpel =  df_merged_jaar['oversterfte_rivm_simpel_cumm'].iloc[-1]
        rivm_complex =  df_merged_jaar['oversterfte_rivm_complex_cumm'].iloc[-1]
        
        simpel_str  = f"Simpel : rivm : {int(rivm_simpel)} | cbs : {int(cbs_simpel)} | verschil {int(rivm_simpel-cbs_simpel)}"
        middel_str = f"Middel : rivm : {int(rivm_middel)} | cbs : {int(cbs_middel)} | verschil {int(rivm_middel-cbs_middel)}"
        complex_str= f"Complex : rivm : {int(rivm_complex)} | cbs : {int(cbs_complex)} | verschil {int(rivm_complex-cbs_complex)}"
        texts = [simpel_str, middel_str, complex_str] 
        #st.write( df_merged_jaar)
        temp1=[None,None,None]
        col1,col2,col3=st.columns(3)
        temp1[0],temp1[1],temp1[2] = col1,col2,col3
        for i, p in enumerate(['simpel', 'middel', 'complex']):
            with temp1[i]:
                fig = go.Figure()
                for n in ['rivm', 'cbs']:
                    # Voeg de werkelijke data toe
                    fig.add_trace(go.Scatter(
                        x= df_merged_jaar['weeknr'],
                        y= df_merged_jaar[f'oversterfte_{n}_{p}_cumm'],
                        mode='lines',
                        name=f'cummulatieve oversterfte {n}'
                    ))
                
                # Titel en labels toevoegen
                fig.update_layout(
                    title=f'Cumm oversterfte ({p}) - {y}',
                    xaxis_title='Tijd',
                    yaxis_title='Aantal'
                )

                st.plotly_chart(fig)
                st.write(texts[i])

        st.subheader(f"Results - {y}")
    
        df_grouped= df_merged_jaar.groupby(by="jaar_x_x").sum().reset_index()
    
        df_grouped = df_grouped[[
                    "jaar_x_x",
                    "oversterfte_rivm_simpel",
                    "oversterfte_rivm_middel",
                    "oversterfte_rivm_complex",
                    
                    "oversterfte_cbs_simpel",
                    "oversterfte_cbs_middel",
                    "oversterfte_cbs_complex",
                ]]
        # for i in [0,1,2]:
        #     st.write(texts[i])

        for x in ['simpel', 'middel', 'complex']:
            df_grouped[f"verschil_{x}"] =  df_grouped[f"oversterfte_rivm_{x}"] - df_grouped[f"oversterfte_cbs_{x}"]
        df_grouped_transposed = df_grouped.transpose().astype(int)
       
        
        if y =="All":
            st.write (df_grouped_transposed)
        else:
            # Create a new DataFrame with more logical structure
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

            # Convert the dictionary to a DataFrame
            new_df_grouped = pd.DataFrame(new_data)

            # Transpose the DataFrame for the desired format
            new_df_grouped = new_df_grouped.transpose().astype(int)

            # Display the new DataFrame
            st.write(new_df_grouped)

if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()

   
