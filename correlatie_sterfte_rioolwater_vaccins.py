from mortality_yearly_per_capita import get_bevolking
import streamlit as st

import pandas as pd
import platform

import pandas as pd
import streamlit as st
import numpy as np

import plotly.express as px

from scipy.stats import linregress
import statsmodels.api as sm
from scipy import stats
@st.cache_data()
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
    df_["week"] = (df_["TIME_PERIOD"].str[6:]).astype(int)

    df_ = df_[df_["sex"] == "T"]
    def add_custom_age_group_deaths(df_, min_age, max_age):
        # Filter the data based on the dynamic age range
        df_filtered = df_[(df_['age_low'] >= min_age) & (df_['age_high'] <= max_age)]

        # Group by TIME_PERIOD (week), sex, and sum the deaths (OBS_VALUE)
        totals = df_filtered.groupby(['TIME_PERIOD', 'sex'], observed=False)['OBS_VALUE'].sum().reset_index()

        # Assign a new label for the age group (dynamic)
        totals['age'] = f'Y{min_age}-{max_age}'
        totals["age_sex"] = totals["age"] + "_" +totals["sex"]
        totals["jaar"] = (totals["TIME_PERIOD"].str[:4]).astype(int)
        totals["week"] = (totals["TIME_PERIOD"].str[6:]).astype(int)
        return totals
    
    for i in opdeling:
        custom_age_group = add_custom_age_group_deaths(df_, i[0], i[1])
        df_ = pd.concat([df_, custom_age_group], ignore_index=True)

  
   
    df_bevolking = get_bevolking("NL", opdeling)

    
    df__ = pd.merge(df_, df_bevolking, on=['jaar', 'age_sex'], how='outer')
    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["OBS_VALUE"].notna()]
    df__ = df__[df__["jaar"] != 2024]
    df__["per100k"] = round(df__["OBS_VALUE"]/df__["aantal"]*100000,1)
    
    return df__


#@st.cache_data()
def get_rioolwater():
    # https://www.rivm.nl/corona/actueel/weekcijfers
    
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rioolwater_2024okt.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rioolwater_2024okt.csv"
    df = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
  
    
    return df



@st.cache_data()
def get_vaccinaties():
    # https://www.ecdc.europa.eu/en/publications-data/data-covid-19-vaccination-eu-eea

    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\vaccinaties_NL_2023.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/vaccinaties_NL_2023.csv"
    df = pd.read_csv(
        file,
        delimiter=",",
        
        low_memory=False,
    )
  
    
    df['age_sex'] =df['age_sex']+'_T'

    df = df.groupby(['YearWeekISO', 'age_sex']).sum(numeric_only=True).reset_index()
    df['TotalDoses'] = df[['FirstDose', 'SecondDose', 'DoseAdditional1', 'DoseAdditional2', 
                       'DoseAdditional3', 'DoseAdditional4', 'DoseAdditional5', 'UnknownDose']].sum(axis=1)

    df["jaar"] = (df["YearWeekISO"].str[:4]).astype(int)
    df["week"] = (df["YearWeekISO"].str[6:]).astype(int)
    return df



def multiple_lineair_regression(df,x_values,y_value_):
    """Calculates multiple lineair regression. User can choose the Y value and the X values

    Args:
        df_ (df): df with info
    """    
    st.subheader("Multiple Lineair Regression")
    standard=  False#  st.sidebar.checkbox("Standardizing dataframe", True)
    intercept=  True# st.sidebar.checkbox("Intercept", False)
    only_complete = False # st.sidebar.checkbox("Only complete rows", False)
    if only_complete:
        df=df.dropna()
    else:
        df = df.dropna(subset=x_values)
        df = df.dropna(subset=y_value_)
    
   
    if standard:
        # https://stackoverflow.com/questions/50842397/how-to-get-standardised-beta-coefficients-for-multiple-linear-regression-using
        df = df.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)
        df = df[x_values].dropna().apply(stats.zscore)
        numeric_columns = df.select_dtypes(include=[np.number])
        
        # # Apply Z-score normalization to the selected columns
        df = numeric_columns.apply(stats.zscore)

                
        # Select numeric columns for Z-score normalization
        numeric_columns = df.select_dtypes(include=[np.number])
    
        # Exclude 'country' and 'population' from Z-score normalization
        #columns_to_exclude = ['population']
        #numeric_columns = numeric_columns.drop(columns=columns_to_exclude)
  
        # # Apply Z-score normalization to the selected columns
        z_scored_df = numeric_columns.apply(stats.zscore)
   
        # Add 'country' and 'population' columns back to the Z-scored DataFrame
        df_standardized =   pd.concat([df, z_scored_df], axis=1)
    else:
        df_standardized = df
    # st.write("**DATA**")
    # st.write(df_standardized)
    # st.write(f"Length : {len(df_standardized)}")
    x = df_standardized[x_values]
    y = df_standardized[y_value_]
  
    w = df_standardized["Population"]
    
    # with statsmodels
    if intercept:
        x= sm.add_constant(x) # adding a constant
    
    model = sm.OLS(y, x).fit()
    #predictions = model.predict(x) 
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    print_model = model.summary()
    st.write(print_model)
    # df_standardized = df_standardized.dropna(subset="Population")
    # st.write("**OUTPUT WEIGHTED LEAST SQUARES (weightfactor = population)**")
    # wls_model = sm.WLS(y,x, weights=w).fit()
    # print_wls_model = wls_model.summary()
    # st.write(print_wls_model)


def make_scatterplot(df, x, y,age_sex):
    """Makes a scatterplot

    Args:
        df_ (df): the dataframe
        x (str): column used for x axis
        y (str): column used for y axis
        show_log_x (bool): _description_
        show_log_y (bool): _description_
        trendline_per_continent (bool): Do we want a trendline per continent
    """    
    st.subheader("Scatterplot")
   

    slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])
    r_squared = r_value ** 2
    # Calculate correlation coefficient
    correlation_coefficient = np.corrcoef(df[x], df[y])[0, 1]
    

    title_ = f"{age_sex} - {x} vs {y} [n = {len(df)}]"
    r_sq_corr = f'R2 = {r_squared:.2f} / Corr coeff = {correlation_coefficient:.2f}'
    fig = px.scatter(df, x=x, y=y,  hover_data=['jaar','week'],   title=f'{title_} || {r_sq_corr}')
    fig.add_trace(px.line(x=df[x], y=slope * df[x] + intercept, line_shape='linear').data[0])

    # Show the plot
    st.plotly_chart(fig)
def line_plot_2_axis(df,x,y1,y2,age_sex):
        
    import plotly.graph_objects as go

    # Create a figure
    fig = go.Figure()

    # Add OBS_VALUE as the first line on the left y-axis
    fig.add_trace(
        go.Scatter(
            x=df[x], 
            y=df[y1], 
            mode='lines', 
            name=y1,
            line=dict(color='blue')
        )
    )

    # Add RNA_flow_per_100000 as the second line on the right y-axis
    fig.add_trace(
        go.Scatter(
            x=df[x], 
            y=df[y2], 
            mode='lines', 
            name=y2,
            line=dict(color='red'),
            yaxis='y2'
        )
    )

    # Update layout to include two y-axes
    fig.update_layout(
        title=f'{age_sex} - {x} vs {y1} and {y2}',
        xaxis_title=x,
        yaxis_title=y1,
        yaxis2=dict(
            title=y2,
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.5, y=1, orientation='h')
    )

    # Show the figure
    st.plotly_chart(fig)

def main():

    st.subheader("Relatie sterfte/rioolwater/vaccins")
    st.info("Inspired by https://www.linkedin.com/posts/annelaning_vaccinatie-corona-prevalentie-activity-7214244468269481986-KutC/")
    opdeling = [[0,120],[15,17],[18,24], [25,49],[50,59],[60,69],[70,79],[80,120]]
    df = get_sterfte(opdeling)
    rioolwater = get_rioolwater()
    
    df_vaccinaties =get_vaccinaties() 
    
    age_sex_list   = df["age_sex"].unique().tolist()  
    for age_sex in age_sex_list:
        df_to_use =df[df["age_sex"] == age_sex].copy(deep=True)
        
        
        df_result = pd.merge(df_to_use,rioolwater,on=["jaar", "week"], how="inner")
        df_result = pd.merge(df_result, df_vaccinaties, on=["jaar", "week","age_sex"], how="inner")
        df_result["RNA_flow_per_100000"] = df_result["RNA_flow_per_100000"]
        df_result["YearWeekISO"] = df_result["jaar"].astype(int).astype(str) + "_"+ df_result["week"].astype(int).astype(str)
        if len(df_result)>0:
            st.subheader(age_sex)
           
            x_values = ["RNA_flow_per_100000","TotalDoses"]  
            y_value_ = "OBS_VALUE"
            multiple_lineair_regression(df_result,x_values,y_value_)
            col1,col2=st.columns(2)
            with col1:
                line_plot_2_axis(df_result, "YearWeekISO","OBS_VALUE", "RNA_flow_per_100000",age_sex)
                make_scatterplot(df_result, "OBS_VALUE", "RNA_flow_per_100000",age_sex)
                
            with col2:
                line_plot_2_axis(df_result, "YearWeekISO","OBS_VALUE", "TotalDoses",age_sex)
                make_scatterplot(df_result, "OBS_VALUE", "TotalDoses",age_sex)
        else:
            pass
            #st.write("No records")
    st.subheader("Data sources")
    st.info("https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")   
    st.info("https://www.ecdc.europa.eu/en/publications-data/data-covid-19-vaccination-eu-eea")
    st.info("https://www.rivm.nl/corona/actueel/weekcijfers")
if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')
    print(f"--------------{datetime.datetime.now()}-------------------------")
    main()