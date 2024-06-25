
import pandas as pd

import plotly.graph_objects as go
import eurostat
import platform
import streamlit as st
import plotly.express as px

import statsmodels.api as sm
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")

def get_bevolking():
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_NL.csv"
    else: 
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
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
    grouped_data = data.groupby(['jaar', 'geslacht', 'age_group'])['aantal'].sum().reset_index()

    # Save the resulting data to a new CSV file
    # grouped_data.to_csv('grouped_population_by_age_2010_2024.csv', index=False, sep=';')

    # print("Grouping complete and saved to grouped_population_by_age_2010_2024.csv")
    grouped_data["age_sex"] = grouped_data['age_group'].astype(str) +"_"+grouped_data['geslacht'].astype(str)
    
    
    for s in ["M", "F", "T"]:
        grouped_data.replace(f'Y0-4_{s}', f'Y_LT5_{s}', inplace=True)
        grouped_data.replace(f'90-999_{s}',f'Y_GE90_{s}', inplace=True)
    

    return grouped_data


def get_sterfte(country="NL"):
    """_summary_

    Returns:
        _type_: _description_
    """
    # Data from https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en
    # https://ec.europa.eu/eurostat/databrowser/bookmark/fbd80cd8-7b96-4ad9-98be-1358dd80f191?lang=en
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/dataflow/ESTAT/DEMO_R_MWK_05/1.0?references=descendants&detail=referencepartial&format=sdmx_2.1_generic&compressed=true
    do_local = False
    if do_local:
        #st.warning("STATIC DATA dd 23/06/2024")
            
        file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_new.csv"
        df_ = pd.read_csv(
            file,
            delimiter=",",
            low_memory=False,
            )  
     
    else:
        try:
            df_ = get_data_eurostat()

        except:
            st.warning("STATIC DATA dd 23/06/2024")
            if platform.processor() != "":
                file = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\sterfte_eurostats_new.csv"
            
            else:
                file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_new.csv"
                
            df_ = pd.read_csv(
                file,
                delimiter=",",
                low_memory=False,
                )

    
    df_=df_[df_["geo"] == country]
    
    df_["age_sex"] = df_["age"] + "_" +df_["sex"]
    df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["weeknr"] = (df_["TIME_PERIOD"].str[6:]).astype(int)
   
    df_bevolking = get_bevolking()
    
    # df__= df_.merge(df_bevolking, on="age_sex", how="outer")
    # df__["per100k"] = df__["OBS_VALUE"] / df__["aantal"]
    
    # df__.columns = df__.columns.str.replace('jaar_x', 'jaar', regex=False)
    #df__.to_csv(r"C:\Users\rcxsm\Documents\endresult.csv")

    summed_per_year = df_.groupby(["jaar", 'age_sex'])['OBS_VALUE'].sum() # .reset_index()
  
    df__ = pd.merge(summed_per_year, df_bevolking, on=['jaar', 'age_sex'], how='outer')
    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["OBS_VALUE"].notna()]
    df__ = df__[df__["jaar"] != 2024]

    df__["per100k"] = df__["OBS_VALUE"]/df__["aantal"]*100000
 


    return df__

def plot(df, category, value_field):
    
    st.subheader(category)
    
        
    # Filter the data
    df_before_2020 = df[df["jaar"] < 2020]
    df_2020_and_up = df[df["jaar"] >= 2020]

    # Create the scatter plot with Plotly Express for values before 2020
    fig = px.scatter(df_before_2020, x="jaar", y=value_field, title=f"Overlijdens per 100k - {category}")

    # Add another scatter plot for values 2020 and up
    fig.add_trace(go.Scatter(x=df_2020_and_up["jaar"], y=df_2020_and_up[value_field], mode='markers', name='2020 and up', marker=dict(color='red')))

    # Calculate the trendline using statsmodels for data before 2020
    X = df_before_2020["jaar"]
    y = df_before_2020[value_field]
    X = sm.add_constant(X)  # Adds a constant term to the predictor

    model = sm.OLS(y, X).fit()
    trendline = model.predict(X)

    # Calculate R² value
    r2 = r2_score(y, trendline)

    # Add the trendline to the plot
    fig.add_trace(go.Scatter(x=df_before_2020["jaar"], y=trendline, mode='lines', name='Trendline tot 2019',marker=dict(color='green')))

    # Show the plot
    # Show the plot
    st.plotly_chart(fig)

    # Print the formula and R² value
    st.write(f"Trendline formula: y = {model.params[1]:.4f}x + {model.params[0]:.4f}")
    st.write(f"R² value: {r2:.4f}")
def main():
    st.title("Overlijdens in leeftijdsgroepen")
    df=get_sterfte()
    to_do = unique_values = df["age_sex"].unique()
    labels = ['Y_LT5'] + [f'Y{i}-{i+4}' for i in range(5, 90, 5)]   + ['Y_GE90'] 
    start=st.number_input("Startjaar", 2000,2020,2000)
    value_field=st.selectbox("Value field", ["per100k", "OBS_VALUE"],0)
    df=df[df["jaar"]>=start]
    for t in labels:
        
        col1,col2,col3= st.columns(3)

        with col1:
            t2=f"{t}_T"
            plot_wrapper(df, t2, value_field)
        with col2:
            t2=f"{t}_M"
            plot_wrapper(df, t2, value_field)
        with col3:
            t2=f"{t}_F"
            plot_wrapper(df, t2, value_field)
    st.subheader("Databronnen")
    st.info("Bevolkingsgrootte: https://opendata.cbs.nl/#/CBS/nl/dataset/03759ned/table?dl=39E0B")
    st.info("Sterfte: https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")

def plot_wrapper(df, t2, value_field):
    df_ = df[df["age_sex"] == t2]
    if len(df_)>0:
        plot (df_, t2, value_field)
    else:
        st.info("No data")
    


main()