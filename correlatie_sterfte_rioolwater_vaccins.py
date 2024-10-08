from mortality_yearly_per_capita import get_bevolking
import streamlit as st
from typing import List, Tuple
import pandas as pd
import platform

import datetime as dt
import pandas as pd
import streamlit as st
import numpy as np

import plotly.express as px

from scipy.stats import linregress
import statsmodels.api as sm
from scipy import stats

@st.cache_data()
def get_rioolwater_oud() -> pd.DataFrame:
    """
    Fetch and process historical wastewater data.

    Returns:
        pd.DataFrame: Processed wastewater data with year, week, and RNA flow per 100,000 people.
    """
    with st.spinner("GETTING ALL DATA ..."):
        url1 = "https://data.rivm.nl/covid-19/COVID-19_rioolwaterdata.csv"
        df = pd.read_csv(url1, delimiter=";", low_memory=False)
        df["Date_measurement"] = pd.to_datetime(df["Date_measurement"], format="%Y-%m-%d")

        # Create 'year' and 'week' columns from the 'Date_measurement' column
        df['jaar'] = df['Date_measurement'].dt.year
        df['week'] = df['Date_measurement'].dt.isocalendar().week

        # df=df[ (df["jaar"] == 2022) & (df["week"] >= 9)& (df["week"] <= 29)]

        # Group by 'year' and 'week', then sum 'RNA_flow_per_100000'
        df = df.groupby(['jaar', 'week'], as_index=False)['RNA_flow_per_100000'].sum()

        # OLS goes wrong with high numbers
        # https://github.com/statsmodels/statsmodels/issues/9258
        df['RNA_flow_per_100000'] = df['RNA_flow_per_100000'] / 10**17
        return df


def get_maandelijkse_overlijdens(oorzaak):

    if platform.processor() != "":
        file = f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\overlijdens_{oorzaak}.csv"
    else:
        file = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_{oorzaak}.csv"
 
    # Load the CSV file
    df = pd.read_csv(file)

    # Melt the dataframe
    df_melted = df.melt(id_vars=['maand'], var_name='year', value_name=f'OBS_VALUE_{oorzaak}')

    # Map Dutch month names to their numerical equivalent
    month_map = {
        "Januari": "01", "Februari": "02", "Maart": "03", "April": "04", 
        "Mei": "05", "Juni": "06", "Juli": "07", "Augustus": "08", 
        "September": "09", "Oktober": "10", "November": "11", "December": "12"
    }

    # Apply mapping and create YearMonth column in the format YYYY-MM
    df_melted['month'] = df_melted['maand'].map(month_map)
    df_melted['YearMonth'] = df_melted['year'] + '-' + df_melted['month']

    # Drop extra columns and keep only relevant ones
    df_melted_clean = df_melted[['YearMonth', f'OBS_VALUE_{oorzaak}']]
    return df_melted_clean
@st.cache_data()
def get_sterfte(opdeling: List[Tuple[int, int]], country: str = "NL") -> pd.DataFrame:
    """
    Fetch and process mortality data for a given country.

    Args:
        opdeling (List[Tuple[int, int]]): List of age ranges to process.
        country (str, optional): Country code. Defaults to "NL".

    Returns:
        pd.DataFrame: Processed mortality data.
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
    def extract_age_ranges(age: str) -> Tuple[int, int]:
        """
        Extract age ranges from age string.

        Args:
            age (str): Age string.

        Returns:
            Tuple[int, int]: Lower and upper age range.
        """
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

    def add_custom_age_group_deaths(df: pd.DataFrame, min_age: int, max_age: int) -> pd.DataFrame:
        """
        Add custom age group deaths to the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            min_age (int): Minimum age for the group.
            max_age (int): Maximum age for the group.

        Returns:
            pd.DataFrame: Dataframe with added custom age group.
        """
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

# Function to convert date to YearWeekISO
def date_to_yearweekiso(date):
    #date = dt.datetime.strptime(date_str, '%Y-%m-%d')
    # Convert to YearWeekISO format (ISO year and ISO week)
    return date.strftime('%G-W%V')

@st.cache_data()
def get_vaccinaties_owid():
    # https://ourworldindata.org/grapher/daily-covid-19-vaccination-doses?tab=chart&country=~NLD
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\vaccinations_OWOD_NL_daily.csv"
    else:
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/vaccinations_OWOD_NL_daily.csv"
    df = pd.read_csv(
        file,
        delimiter=",",

        low_memory=False,
    )

    df['age_sex'] ='TOTAL_T'


    # Convert 'datum' to datetime if it's not already
    df['datum'] = pd.to_datetime(df['datum'], errors='coerce')

    # Get the range of dates from the minimum date in the 'datum' column to today
    start_date = df['datum'].min()
    end_date = dt.datetime.now()

    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a DataFrame from the date range
    date_df = pd.DataFrame(date_range, columns=['datum'])

    # Merge the original dataframe with the new date dataframe to fill in missing dates
    df_filled = pd.merge(date_df, df, on='datum', how='left')
    df_filled["TotalDoses"] = df_filled["TotalDoses"].fillna(0)
    df_filled["age_sex"] = df_filled["age_sex"].fillna("TOTAL_T")
    
    # If you want to keep any existing data in df, you can use:
    df_filled = pd.concat([df, df_filled]).drop_duplicates(subset='datum').sort_values('datum').reset_index(drop=True)

    # Apply the conversion function to the 'datum' column
    df_filled['YearWeekISO'] = df_filled['datum'].apply(date_to_yearweekiso)

    df_filled["jaar"] = (df_filled["YearWeekISO"].str[:4]).astype(int)
    df_filled["week"] = (df_filled["YearWeekISO"].str[6:]).astype(int)
    df_filled = df_filled.groupby(['jaar','week','YearWeekISO']).sum(numeric_only=True).reset_index()
    
    return df_filled

#@st.cache_data()
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

def compare_vaccinations(df_vaccinaties):
    st.subheader("Compare vaccinations EDCD - OWID")
    df_vaccinaties_owid = get_vaccinaties_owid()

    df_vaccinaties_total = df_vaccinaties[df_vaccinaties['age_sex']=="TOTAL_T"]

    df_v = pd.merge(df_vaccinaties_total, df_vaccinaties_owid, on="YearWeekISO")
    df_v["week"] = df_v["week_x"]
    df_v["jaar"] = df_v["jaar_x"]

    line_plot_2_axis(df_v,"YearWeekISO","TotalDoses_x","TotalDoses_y","TOTAL_T")

    df_v_month = from_week_to_month(df_v,"sum")
    line_plot_2_axis(df_v_month,"YearMonth","TotalDoses_x","TotalDoses_y","TOTAL_T")

def compare_rioolwater(rioolwater):
    st.subheader("compare the rioolwater given by RIVM (x)  and calculated from the file with various meetpunten (y)")
    rioolwater_oud =  get_rioolwater_oud()


    # compare the rioolwater given by RIVM and calculated from the file with various meetpunten
   
    rw = pd.merge(rioolwater,rioolwater_oud, on=["jaar", "week"])

    rw["YearWeekISO"] = rw["jaar"].astype(str) +"-W" +rw["week"].astype(str)
    line_plot_2_axis(rw,"YearWeekISO","RNA_flow_per_100000_x","RNA_flow_per_100000_y","TOTAL")
    rw = from_week_to_month(rw,"mean")
    line_plot_2_axis(rw,"YearMonth","RNA_flow_per_100000_x","RNA_flow_per_100000_y","TOTAL")

def from_week_to_month(rw, how):
    rw["YearWeekISO"] = rw["jaar"].astype(int).astype(str) + "-W"+ rw["week"].astype(int).astype(str)

    # Apply the conversion function to the YearWeekISO column
    rw['YearMonth'] = rw['YearWeekISO'].apply(yearweek_to_yearmonth)
    if how == "sum":
        rw = rw.groupby(['YearMonth'], as_index=False).sum( numeric_only=True)
    else:
        rw = rw.groupby(['YearMonth'], as_index=False).mean( numeric_only=True)
    return rw

def multiple_linear_regression(df: pd.DataFrame, x_values: List[str], y_value_: str, age_sex: str):
    """
    Perform multiple linear regression and display results.

    Args:
        df (pd.DataFrame): Input dataframe.
        x_values (List[str]): List of independent variable column names.
        y_value (str): Dependent variable column name.
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
    
    
    # try:    
    #     # Voeg een sinus- en cosinusfunctie toe om seizoensinvloeden te modelleren
    #     df_standardized['sin_time'] = np.sin(2 * np.pi * df_standardized['week']/ 52)
    #     df_standardized['cos_time'] = np.cos(2 * np.pi * df_standardized['week'] / 52)
    # except:
    #     df_standardized['sin_time'] = np.sin(2 * np.pi * df_standardized['maand']/ 12)
    #     df_standardized['cos_time'] = np.cos(2 * np.pi * df_standardized['maand'] / 12)
    # with statsmodels
    
    #x_values = x_values +  ['sin_time', 'cos_time']

    x = df_standardized[x_values]
    y = df_standardized[y_value_]

    if intercept:
        x= sm.add_constant(x) # adding a constant

    model = sm.OLS(y, x).fit()
    #predictions = model.predict(x)
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    print_model = model.summary()
    st.write(print_model)
   
    data = {
        'Y value':y_value_,
        # 'Coefficients': model.params,
        # 'P-values': model.pvalues,
        # 'T-values': model.tvalues,
        # 'Residuals': model.resid,
        'P_const': model.pvalues["const"],
        'P_RNA':model.pvalues["RNA_flow_per_100000"],
        'P_vacc':model.pvalues["TotalDoses"],
        'R-squared': [model.rsquared], #* len(model.params),
        'Adjusted R-squared': [model.rsquared_adj], # * len(model.params),
        'F-statistic': [model.fvalue], # * len(model.params),
        'F-statistic P-value': [model.f_pvalue], # * len(model.params)
    }
        
    #st.write(data)
    # Stap 3: Coefficiënten, P-waarden, en T-waarden omzetten naar dictionaries
    # We gebruiken .split('\n') om elke variabele en waarde uit te splitsen en daarna de strings om te zetten naar floats.

    # # Coefficiënten
    # coefficients = dict([item.strip().split() for item in data["Coefficients"].split('\n') if item])

    # # P-waarden
    # p_values = dict([item.strip().split() for item in data["P-values"].split('\n') if item])

    # # T-waarden
    # t_values = dict([item.strip().split() for item in data["T-values"].split('\n') if item])

    # # Residuen
    # residuals = [float(r.split()[1]) for r in data["Residuals"].split('\n') if r]

    # Stap 4: Omzetten naar een DataFrame
    y_value_x =  f"{data["Y value"]}_{age_sex}"
    df = pd.DataFrame({
        "Y value":y_value_x,
        # "Coefficients": pd.Series(coefficients),
        # "P-values": pd.Series(p_values),
        # "T-values": pd.Series(t_values),
        # "Residuals": residuals,
        'P_const': data['P_const'],
        'P_RNA':data['P_RNA'],
        'P_vacc':data['P_vacc'],

        "R-squared": data["R-squared"],
        "Adjusted R-squared": data["Adjusted R-squared"],
        "F-statistic": data["F-statistic"],
        "F-statistic P-value": data["F-statistic P-value"]
    })

    return df


def make_scatterplot(df: pd.DataFrame, x: str, y: str, age_sex: str):
    """
    Create and display a scatterplot.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): X-axis column name.
        y (str): Y-axis column name.
        age_sex (str): Age and sex group.
    """
    #st.subheader("Scatterplot")

    slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])
    r_squared = r_value ** 2
    # Calculate correlation coefficient
    correlation_coefficient = np.corrcoef(df[x], df[y])[0, 1]

    title_ = f"{age_sex} - {x} vs {y} [n = {len(df)}]"
    r_sq_corr = f'R2 = {r_squared:.2f} / Corr coeff = {correlation_coefficient:.2f}'
    try:
        fig = px.scatter(df, x=x, y=y,  hover_data=['jaar','week'],   title=f'{title_} || {r_sq_corr}')
    except:
        fig = px.scatter(df, x=x, y=y,    title=f'{title_} || {r_sq_corr}')
    fig.add_trace(px.line(x=df[x], y=slope * df[x] + intercept, line_shape='linear').data[0])

    # Show the plot
    st.plotly_chart(fig)

def line_plot_2_axis(df: pd.DataFrame, x: str, y1: str, y2: str, age_sex: str):
    """
    Create and display a line plot with two y-axes.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): X-axis column name.
        y1 (str): First y-axis column name.
        y2 (str): Second y-axis column name.
        age_sex (str): Age and sex group.
    """
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
    # try:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=df[x],
    #             y=df["sin_time"],
    #             mode='lines',
    #             name="sin_time",
            
    #         )
    #     )

    #     fig.add_trace(
    #         go.Scatter(
    #             x=df[x],
    #             y=df["cos_time"],
    #             mode='lines',
    #             name="cos_time",
                
    #         )
    #     )

    # except:
    #     pass
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

def yearweek_to_yearmonth(yearweek: str) -> str:
    """
    Convert YearWeekISO to YearMonth format.

    Args:
        yearweek (str): Year and week in ISO format (e.g., "2023-W01").

    Returns:
        str: Year and month in format "YYYY-MM".
    """
    year, week = yearweek.split('-W')
    # Calculate the Monday of the given ISO week
    date = dt.datetime.strptime(f'{year} {week} 1', '%G %V %u')
    # Extract year and month from the date
    return date.strftime('%Y-%m')

def main():

    st.subheader("Relatie sterfte/rioolwater/vaccins")
    st.info("Inspired by https://www.linkedin.com/posts/annelaning_vaccinatie-corona-prevalentie-activity-7214244468269481986-KutC/")
    opdeling = [[0,120],[15,17],[18,24], [25,49],[50,59],[60,69],[70,79],[80,120]]
    df = get_sterfte(opdeling)
    rioolwater = get_rioolwater()
    df_vaccinaties =get_vaccinaties()
    df_vaccinaties_owid =get_vaccinaties_owid()
    df_vaccinaties_owid["age_sex"] = "TOTAL_T"
    with st.expander("Rioolwater"):
        compare_rioolwater(rioolwater)
    with st.expander("Vaccinations"):
        compare_vaccinations(df_vaccinaties)

    results = []

    
    age_sex_list   = df["age_sex"].unique().tolist()
    
    for age_sex in age_sex_list:
        df_to_use =df[df["age_sex"] == age_sex].copy(deep=True)

        df_result = pd.merge(df_to_use,rioolwater,on=["jaar", "week"], how="inner")
        
        df_result = pd.merge(df_result, df_vaccinaties, on=["jaar", "week","age_sex"], how="inner")
        
        df_result = df_result[df_result["age_sex"] == age_sex]
        df_result["TotalDoses"].fillna(0)
        
        #df_result["RNA_flow_per_100000"] = df_result["RNA_flow_per_100000"]
        #df_result['OBS_VALUE'] = df_result['OBS_VALUE'].shift(2)
        if age_sex == "TOTAL_T":
            for oorzaak in ["hart_vaat_ziektes","covid",  "ademhalingsorganen","accidentele_val","wegverkeersongevallen", "nieuwvormingen"]:
                with st.expander(oorzaak):
                    st.subheader(f"TOTAL overlijdens {oorzaak} vs rioolwater en vaccinaties")
                    df_iteration = analyse_maandelijkse_overlijdens(oorzaak,age_sex, df_result, "YearMonth")
                    # Zet de resultaten van deze iteratie in een DataFrame
                    
                    # Voeg deze DataFrame toe aan de lijst van resultaten
                    results.append(df_iteration)
        df_result["YearWeekISO"] = df_result["jaar"].astype(int).astype(str) + "-W"+ df_result["week"].astype(int).astype(str)
        
        monthly=False
        if monthly==True:
            df_result = from_week_to_month(df_result, "sum")
            time_period = "YearMonth"
        else:
            time_period = "YearWeekISO"

        #df_result['OBS_VALUE'] = df_result['OBS_VALUE'].rolling(window=5).mean()
        if len(df_result)>0:
            with st.expander(f"{age_sex} - Alle overlijdensoorzaken"):
                st.subheader(f"{age_sex} - Alle overlijdensoorzaken")
                df_iteration = perform_analyse(age_sex, df_result, time_period,"RNA_flow_per_100000","TotalDoses", "OBS_VALUE")
                # Zet de resultaten van deze iteratie in een DataFrame
                
                
                # Voeg deze DataFrame toe aan de lijst van resultaten
                results.append(df_iteration)
        else:
            pass
            #st.write("No records")
    
    # Als de loop klaar is, concateneer alle DataFrames in één DataFrame
    df_complete = pd.concat(results, ignore_index=True)

    # Bekijk de complete DataFrame
    st.write(df_complete)
    make_scatterplot(df_complete, "F-statistic P-value", "Adjusted R-squared","")
    #st.write("De OBS_VALUE is 2 weken opgeschoven naar rechts")
    st.subheader("Data sources")
    st.info("https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")
    st.info("https://www.ecdc.europa.eu/en/publications-data/data-covid-19-vaccination-eu-eea")
    st.info("https://www.rivm.nl/corona/actueel/weekcijfers")

def analyse_maandelijkse_overlijdens(oorzaak, age_sex, df_result, time_period):
    
    df_result_month = from_week_to_month(df_result,"sum")
    df_hartvaat = get_maandelijkse_overlijdens(oorzaak)
    
    df_month = pd.merge(df_result_month, df_hartvaat, on="YearMonth") 
    df_month["maand"] = (df_month["YearMonth"].str[5:]).astype(int)
 
    data = perform_analyse(age_sex, df_month, time_period, "RNA_flow_per_100000","TotalDoses",f"OBS_VALUE_{oorzaak}")
    return data
def perform_analyse(age_sex, df, time_period,x1,x2,y):
    # Voeg een sinus- en cosinusfunctie toe om seizoensinvloeden te modelleren
    try:           
        df['sin_time'] = np.sin(2 * np.pi * df['maand']/ 12)
        df['cos_time'] = np.cos(2 * np.pi * df['maand'] / 12)

    except:
       
        df['sin_time'] = np.sin(2 * np.pi * df['week']/ 52)
        df['cos_time'] = np.cos(2 * np.pi * df['week'] / 52)
        
    x_values = [x1,x2] # + ['sin_time', 'cos_time']
    y_value_ = y
    
   
    col1,col2=st.columns(2)
    with col1:
        line_plot_2_axis(df, time_period,y_value_, x1,age_sex)
        make_scatterplot(df, y_value_, x1,age_sex)
  
    with col2:
        line_plot_2_axis(df, time_period,y_value_, x2,age_sex)
        make_scatterplot(df, y_value_, x2,age_sex)
    data = multiple_linear_regression(df,x_values,y_value_, age_sex)
    return data

if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')
    print(f"--------------{datetime.datetime.now()}-------------------------")
    main()
    #expand()