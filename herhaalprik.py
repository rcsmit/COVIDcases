import platform
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# https://chatgpt.com/c/66f0a0c4-5ac4-8004-b47f-e121ccd1eaea


@st.cache_data()
def get_rioolwater():
    """Get the data
    In : -
    Out : df        : dataframe
         UPDATETIME : Date and time from the last update"""
    with st.spinner("GETTING ALL DATA ..."):
        url1 = "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
        df = pd.read_csv(url1, delimiter=";", low_memory=False)
        df["Date_measurement"] = pd.to_datetime(df["Date_measurement"], format="%Y-%m-%d")
        
        # Create 'year' and 'week' columns from the 'Date_measurement' column
        df['jaar'] = df['Date_measurement'].dt.year
        df['week'] = df['Date_measurement'].dt.isocalendar().week
        
        df=df[ (df["jaar"] == 2022) & (df["week"] >= 9)& (df["week"] <= 29)]

        # Group by 'year' and 'week', then sum 'RNA_flow_per_100000'
        df = df.groupby(['jaar', 'week'], as_index=False)['RNA_flow_per_100000'].sum()

        # OLS goes wrong with high numbers
        # https://github.com/statsmodels/statsmodels/issues/9258
        df['RNA_flow_per_100000'] = df['RNA_flow_per_100000'] / 10**17
        return df
def get_herhaalprik():
    """_summary_

    Returns:
        _type_: _description_
    """    
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\herhaalprik_per_week_per_leeftijdscat.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/herhaalprik_per_week_per_leeftijdscat.csv"
    df = pd.read_csv(
        file,
        delimiter=";",
        
        low_memory=False,
    )
    #df["weeknr"] = df["jaar"].astype(str) +"_" + df["weeknr"].astype(str).str.zfill(2)
    df = df.drop('jaar', axis=1)

    df.rename(columns={
        'jaar': 'jaar', 
        'weeknr': 'week', 
        'herhaalprik_m_v_0_999': 'Y0-120_T', 
        'herhaalprik_m_v_0_49': 'Y0-49_T', 
        'herhaalprik_m_v_50_64': 'Y50-64_T', 
        'herhaalprik_m_v_65_79': 'Y65-79_T', 
        'herhaalprik_m_v_80_89': 'Y80-89_T', 
        'herhaalprik_m_v_90_999': 'Y90-120_T'
    }, inplace=True)
    
    df["jaar"] = 2022
    df_long = df.melt(
        id_vars=['jaar', 'week'],  # These columns will remain as they are
        value_vars=[
            'Y0-120_T', 'Y0-49_T', 'Y50-64_T', 
            'Y65-79_T', 'Y80-89_T', 'Y90-120_T'
        ],  # Columns to unpivot
        var_name='age_sex',  # New column name for the age/sex groups
        value_name='aantal_prikken'  # New column name for the values
    )

    return df_long


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
        totals["week"] = (totals["TIME_PERIOD"].str[6:]).astype(int)
        return totals
    
    for i in opdeling:
        custom_age_group = add_custom_age_group_deaths(df_, i[0], i[1])
        df_ = pd.concat([df_, custom_age_group], ignore_index=True)

  

    df_=df_[(df_["sex"] == "T") & (df_["jaar"] == 2022) & (df_["week"] >= 9)& (df_["week"] <= 29)]

   
    return df_


# Define a function to create a line graph for each age_sex group
def plot_age_sex_line(df, what, age_sex_group):
    # Filter data for the specific age_sex group
    df_filtered = df[df['age_sex'] == age_sex_group]
    
    # Create a subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add trace for aantal_prikken (left y-axis)
    fig.add_trace(
        go.Scatter(x=df_filtered['week'], y=df_filtered[what], name=what, mode='lines'),
        secondary_y=False
    )

    # Add trace for OBS_VALUE (right y-axis)
    fig.add_trace(
        go.Scatter(x=df_filtered['week'], y=df_filtered['OBS_VALUE'], name="OBS_VALUE", mode='lines'),
        secondary_y=True
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Week")

    # Set y-axis titles
    fig.update_yaxes(title_text=what, secondary_y=False)
    fig.update_yaxes(title_text="OBS_VALUE", secondary_y=True)

    # Set plot title
    fig.update_layout(
        title_text=f"Line Graph for {age_sex_group}",
        showlegend=True
    )

    # Show plot
   
    st.plotly_chart(fig)

# Define a function to plot scatter, trendline, and R² for each age_sex
def plot_age_sex_scatter(df, what, age_sex_group):
    # Filter data for the specific age_sex group
    df_filtered = df[df['age_sex'] == age_sex_group]
    
    # Prepare data for OLS regression
    X = df_filtered[what]
    Y = df_filtered['OBS_VALUE']
    X_with_const = sm.add_constant(X)  # Adds a constant term to the model
    
    # Perform OLS regression
    model = sm.OLS(Y, X_with_const).fit()
    df_filtered['trendline'] = model.predict(X_with_const)  # Get trendline values
    r_squared = model.rsquared  # Calculate R²
     # Calculate the correlation between 'aantal_prikken' and 'OBS_VALUE'
    correlation = df_filtered[what].corr(df_filtered['OBS_VALUE'])
    
    # Create scatter plot with trendline
    fig = px.scatter(df_filtered, x=what, y='OBS_VALUE',
                     title=f"Scatter plot for {age_sex_group}<br>corr = {correlation}<br>R² = {r_squared:.4f}",
                     labels={what: what, 'OBS_VALUE': 'OBS_VALUE'},
                     trendline="ols", trendline_color_override='red')

    # Add the OLS trendline
    fig.add_traces(px.line(df_filtered, x=what, y='trendline').data)
    
    # Show plot
   
    st.plotly_chart(fig)

def main():

    st.subheader("Herhaalprikken vs sterfte")
    st.info("Reproductie van https://twitter.com/dimgrr/status/1620775536795746308 maar ook voor leeftijdsgroepen. Tevens sterfte vs RNA deeltjes in rioolwater weergegeven. Weeknummers zijn van 2022")
    opdeling = [[0,49], [50,64], [65,79], [80,89], [90,120], [0,120]]
    df_sterfte = get_sterfte(opdeling)
    
    df_herhaalprik = get_herhaalprik()
    df_rioolwater = get_rioolwater()
    
    df_data = pd.merge(df_sterfte, df_herhaalprik, on=["jaar","week", "age_sex"], how ="outer")
    df_data = pd.merge(df_data, df_rioolwater,  on=["jaar","week"], how = "left")
       
    #for what in ["aantal_prikken", "RNA_flow_per_100000"]:
        
        # Call the function for each unique age_sex group
    for age_sex_group in df_data['age_sex'].unique():

        col1,col2,col3,col4= st.columns(4)
        what = "aantal_prikken"
        with col1:
            st.write(what)
            plot_age_sex_scatter(df_data, what, age_sex_group)
        # Call the function for each unique age_sex group
        with col2:
            st.write(what)

            plot_age_sex_line(df_data, what, age_sex_group)

        what =  "RNA_flow_per_100000"
        with col3:
            st.write(what)

            plot_age_sex_scatter(df_data, what, age_sex_group)
        # Call the function for each unique age_sex group
        with col4:
            st.write(what)

            plot_age_sex_line(df_data, what, age_sex_group)

    st.info("De waarde voor rioolwater is gedeeld door 10^17 om de correlatie en R2 te kunnen berekenen")
if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()