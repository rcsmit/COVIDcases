from mortality_yearly_per_capita import get_bevolking
import streamlit as st
from typing import List, Tuple
import pandas as pd
import platform
import random
import datetime as dt
import pandas as pd
import streamlit as st
import numpy as np
from covid_dashboard_rcsmit import find_lag_time
import plotly.express as px

from scipy.stats import linregress
import statsmodels.api as sm
from scipy import stats
from oversterfte_compleet import  get_sterftedata, get_data_for_series_wrapper,make_df_quantile #, layout_annotations_fig
# for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from oversterfte_eurostats_maand import get_data_eurostat
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
#from scipy.signal import savgol_filter

# WAAROM TWEE KEER add_custom_age_group_deaths ??? TODO

@st.cache_data()
def get_ziekenhuis_ic() -> pd.DataFrame:
    """
    Fetch and process historical wastewater data.

    Returns:
        pd.DataFrame: Processed wastewater data with year, week, and RNA flow per 100,000 people.
    """
    with st.spinner("GETTING ALL DATA ..."):
     
        # URLs for the datasets
        url1 = "https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep_tm_03102021.csv"
        url2 = "https://data.rivm.nl/covid-19/COVID-19_ziekenhuis_ic_opnames_per_leeftijdsgroep.csv"

        # Load the two CSV files into dataframes
        df1 = pd.read_csv(url1, delimiter=';')
        df2 = pd.read_csv(url2, delimiter=';')

        # Combine the two dataframes
        df_combined = pd.concat([df1, df2], ignore_index=True)

        # Ensure the Date_of_statistics_week_start column is in datetime format
        df_combined['Date_of_statistics_week_start_nr'] =  pd.to_datetime(df_combined["Date_of_statistics_week_start"], format="%Y-%m-%d")
        
        # Create 'year' and 'week' columns from the 'Date_measurement' column
        df_combined['jaar'] = df_combined['Date_of_statistics_week_start_nr'].dt.year
        df_combined['week'] = df_combined['Date_of_statistics_week_start_nr'].dt.isocalendar().week
        df_combined["periodenr"] = df_combined["jaar"].astype(str) + "_" + df_combined["week"].astype(str).str.zfill(2)
   
        # df=df[ (df["jaar"] == 2022) & (df["week"] >= 9)& (df["week"] <= 29)]

        # Group by 'year' and 'week', then sum 'RNA_flow_per_100000'
        df_combined = df_combined.groupby(['jaar', 'week'], as_index=False).sum(["Hospital_admission", "IC_admission"])


        
        return df_combined

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
@st.cache_data()
def get_oversterfte(opdeling):
    # if platform.processor() != "":
    #     file = f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\basevalues_sterfte.csv"
    #     file = f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\basevalues_sterfte_Y0-120_T.csv"
    
    # else:
    #     #file = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/basevalues_sterfte.csv"
    #     file = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/basevalues_sterfte_Y0-120_T.csv"
    # # Load the CSV file
    # df_ = pd.read_csv(file)
    df__ = get_sterftedata(2015, "m_v_0_999")

    df_data= get_data_for_series_wrapper(df__,"m_v_0_999",2015)
    df_, df_corona, df_quantile = make_df_quantile("m_v_0_999", df_data, "week") 
    #df_to_export = df_data[["weeknr", "avg", "aantal_overlijdens"]].copy()
    df_["age_sex"] = "Y0-120_T"

    df_ = df_.assign(
        jaar_week=df_["periodenr"],
        base_value=df_["avg"],
        OBS_VALUE_=df_["m_v_0_999"]
    )


    df_ = df_[["jaar_week","base_value","OBS_VALUE_"]]
    df_["age_sex"]= "Y0-120_T"

    df_["jaar"] = (df_["jaar_week"].str[:4]).astype(int)
    df_["week"] = (df_["jaar_week"].str[5:]).astype(int)
    df_["YearWeekISO"] = df_["jaar"].astype(int).astype(str) + "-W"+ df_["week"].astype(int).astype(str)
    df_["TIME_PERIOD"] = df_["jaar"].astype(int).astype(str) + "-W"+ df_["week"].astype(int).astype(str)

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
    #df_['age_low'], df_['age_high'] = zip(*df_['age'].apply(extract_age_ranges))
    
    df_["jaar"] = df_["jaar"].astype(int)
    df_["week"] = df_["week"].astype(int)
    #df_ = df_[df_["sex"] == "T"]

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
        totals = df_filtered.groupby(['TIME_PERIOD', 'sex'], observed=False)['avg'].sum().reset_index()

        # Assign a new label for the age group (dynamic)
        totals['age'] = f'Y{min_age}-{max_age}'
        totals["age_sex"] = totals["age"] + "_" +totals["sex"]
        totals["jaar"] = (totals["TIME_PERIOD"].str[:4]).astype(int)
        totals["week"] = (totals["TIME_PERIOD"].str[6:]).astype(int)
        return totals


    # for i in opdeling:
    #     custom_age_group = add_custom_age_group_deaths(df_, i[0], i[1])
    #     df_ = pd.concat([df_, custom_age_group], ignore_index=True)
    #df_["age_sex"] = df_["age_group"] + "_" +df_["geslacht"]
  
    df_bevolking = get_bevolking("NL", opdeling)
    
    df__ = pd.merge(df_, df_bevolking, on=['jaar', 'age_sex'], how='outer')

    
    df__ = df__[df__["aantal"].notna()]
    df__ = df__[df__["base_value"].notna()]
    #df__ = df__[df__["jaar"] != 2024]
    df__["per100k"] = round(df__["OBS_VALUE_"]/df__["aantal"]*100000,1)


    df__["oversterfte"] = df__["OBS_VALUE_"] - df__["base_value"]
    df__["p_score"] = ( df__["OBS_VALUE_"]- df__["base_value"]) /   df__["base_value"]
   
    return df__

@st.cache_data()
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
    df_melted_clean = df_melted[['YearMonth', f'OBS_VALUE_{oorzaak}']].dropna()
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

    if 1==2:
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
    df_ = get_data_eurostat()
   

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
    #df__ = df__[df__["jaar"] != 2024]
    df__["per100k"] = round(df__["OBS_VALUE"]/df__["aantal"]*100000,1)

    return df__

@st.cache_data()
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

    df["periodenr"] = df["jaar"].astype(str) + "_" + df["week"].astype(str).str.zfill(2)
    

    return df


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

def multiple_linear_regression(df: pd.DataFrame, x_values: List[str], y_value_: str, age_sex: str, normalize:bool):
    """
    Perform multiple linear regression and display results.

    Args:
        df (pd.DataFrame): Input dataframe.
        x_values (List[str]): List of independent variable column names.
        y_value (str): Dependent variable column name.
    Returns:
        Datadict :  C_RNA':model.params["RNA_flow_per_100000"],
                    'C_vacc':model.params["TotalDoses"],
                    'P_const': model.pvalues["const"],
                    'P_RNA':model.pvalues["RNA_flow_per_100000"],
                    'P_vacc':model.pvalues["TotalDoses"],
                    'R-squared': [model.rsquared], #* len(model.params),
                    'Adjusted R-squared': [model.rsquared_adj], # * len(model.params),
                    'F-statistic': [model.fvalue], # * len(model.params),
                    'F-statistic P-value':
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

    x = df[x_values]
    y = df[y_value_]

    if normalize:

        # Normalize each feature in x to the range [0, 1]
        x_normalized = (x - x.min()) / (x.max() - x.min())

        # If intercept is required, add a constant term
        if intercept:
            x_normalized = sm.add_constant(x_normalized)  # adding a constant
    
        # Fit the OLS model using the normalized data
        model = sm.OLS(y, x_normalized).fit()
    else:
        if intercept:
            x= sm.add_constant(x) # adding a constant

        model = sm.OLS(y, x).fit()
    #predictions = model.predict(x)
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    #col1,col2=st.columns(2)
    # with col1:
    #     st.write("**Model**")
    #     print_model = model.summary()
    #     st.write(print_model)
    # with col2:
    robust_model = model.get_robustcov_results(cov_type='HC0')  # You can use 'HC0', 'HC1', 'HC2', or 'HC3'
    st.write("**robust model**")
    st.write(robust_model.summary())
    col1,col2,col3=st.columns([2,1,1])
   
    with col1:
        st.write("**Correlation matrix**")
        correlation_matrix = x.corr()
        st.write(correlation_matrix)
    with col2:
        # Calculate VIF for each variable

        # VIF = 1: No correlation between the variable and others.
        # 1 < VIF < 5: Moderate correlation, likely acceptable.
        # VIF > 5: High correlation, indicating possible multicollinearity.
        # VIF > 10: Strong multicollinearity, action is needed.
        st.write("**VIF**")
        vif = pd.DataFrame()
        vif["Variable"] = x.columns
        vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
        st.write(vif)
    with col3:
        U, s, Vt = np.linalg.svd(x)
        st.write("**spread of singular values**")
        st.write(s)  # Look at the spread of singular values
    data_dict = {
        'Y value':f"{y_value_}_{age_sex}",
        # 'Coefficients': model.params,
        # 'P-values': model.pvalues,
        # 'T-values': model.tvalues,
        # 'Residuals': model.resid,
        'C_RNA':model.params["RNA_flow_per_100000"],
        'C_vacc':model.params["TotalDoses"],
        'P_const': model.pvalues["const"],
        'P_RNA':model.pvalues["RNA_flow_per_100000"],
        'P_vacc':model.pvalues["TotalDoses"],
        'R-squared': model.rsquared, #* len(model.params),
        'Adjusted R-squared': model.rsquared_adj, # * len(model.params),
        'F-statistic': model.fvalue, # * len(model.params),
        'F-statistic P-value': model.f_pvalue, # * len(model.params)
    }
        

    
    
    return data_dict


def make_scatterplot(df: pd.DataFrame, x: str, y: str, age_sex: str):
    """
    Create and display a scatterplot.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): X-axis column name.
        y (str): Y-axis column name.
        age_sex (str): Age and sex group.
    """
    df[x]=df[x].astype(float)
    df[y]=df[y].astype(float)
    slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])
    r_squared = r_value ** 2
    # Calculate correlation coefficient
    correlation_coefficient = np.corrcoef(df[x], df[y])[0, 1]

    title_ = f"{age_sex} - {x} vs {y} [n = {len(df)}]"
    r_sq_corr = f'R2 = {r_squared:.2f} / Corr coeff = {correlation_coefficient:.2f}'
    try:
        fig = px.scatter(df, x=x, y=y,  hover_data=['jaar','week'],   title=f'{title_} ||<br> {r_sq_corr}')
    except:
        fig = px.scatter(df, x=x, y=y,    title=f'{title_} ||<br> {r_sq_corr}')
    fig.add_trace(px.line(x=df[x], y=slope * df[x] + intercept, line_shape='linear').data[0])

    # Show the plot
    key=str(int(random.random()*10000))
    st.plotly_chart(fig, key=key)

def line_plot_2_axis(df: pd.DataFrame, x: str, y1: str, y2: str, age_sex: str):
    """
    Create and display a line plot with two y-axes.

    Args:
        df (pd.DataFrame): Input dataframe.
        x (str): X-axis column name.
        y1 (str): First y-axis column name.
        y2 (str): Second y-axis column name.
        age_sex (str): Age and sex group. (for the title)
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
    if y1=="base_value":

        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y2],
                mode='lines',
                name=y2,
                line=dict(color='red'),
              
            )
        )
    else:
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
    try:
        if 2020 in df["jaar"].values:
            fig.add_vrect(
                x0="2020-W13",
                x1="2020-W18",
                annotation_text="Eerste golf",
                annotation_position="top left",
                fillcolor="pink",
                opacity=0.25,
                line_width=0,
                )
            fig.add_vrect(
                x0="2020-W39",
                x1="2021-W03",
                annotation_text="Tweede golf",
                annotation_position="top left",
                fillcolor="pink",
                opacity=0.25,
                line_width=0,
            )


             # hittegolven
            fig.add_vrect(
                x0="2020-W33",
                x1="2020-W34",
                annotation_text=" ",
                annotation_position="top left",
                fillcolor="yellow",
                opacity=0.35,
                line_width=0,
            )
            fig.add_vrect(
                x0="2020-W01",
                x1="2020-W52",
                fillcolor="grey",
                opacity=0.1,
                line_width=0,
            )
        if 2021 in df["jaar"].values:
            fig.add_vrect(
                x0="2021-W33",
                x1="2021-W52",
                annotation_text="Derde golf",
                annotation_position="top left",
                fillcolor="pink",
                opacity=0.25,
                line_width=0,
            )
            fig.add_vrect(
                x0="2021-W01",
                x1="2021-W02",
                fillcolor="grey",
                opacity=0.35,
                line_width=0,
            )
       
        if 2022 in df["jaar"].values:
        
            fig.add_vrect(
                x0="2022-W32",
                x1="2022-W33",
                annotation_text=" ",
                annotation_position="top left",
                fillcolor="yellow",
                opacity=0.35,
                line_width=0,
            )
            fig.add_vrect(
                x0="2022-W01",
                x1="2022-W52",
                fillcolor="grey",
                opacity=0.1,
                line_width=0,
            )
        if 2023 in df["jaar"].values:
        
            fig.add_vrect(
                x0="2023-W23",
                x1="2023-W24",
                annotation_text=" ",
                annotation_position="top left",
                fillcolor="yellow",
                opacity=0.35,
                line_width=0,
            )
            fig.add_vrect(
                x0="2023-W36",
                x1="2023-W37",
                annotation_text="Geel = Hitte golf",
                annotation_position="top left",
                fillcolor="yellow",
                opacity=0.35,
                line_width=0,
            )
            
        if 2024 in df["jaar"].values:
            # geen hittegolf in 2024
            fig.add_vrect(
                x0="2024-W01",
                x1="2024-W39",
                fillcolor="grey",
                opacity=0.1,
                line_width=0,
            )
           
    except:
        pass
    key=str(int(random.random()*10000))
    st.plotly_chart(fig, key=key)

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


def analyse_maandelijkse_overlijdens(oorzaak, age_sex, df_result, time_period, seizoen, maand, normalize):
    """_summary_

    Args:
        oorzaak (_type_): _description_
        age_sex (_type_): _description_
        df_result (_type_): _description_
        time_period (_type_): _description_
        seizoen (bool): _description_
        maand (bool): _description_

    Returns:
        _type_: _description_
    """    
    
    df_result_month = from_week_to_month(df_result,"sum")
    #df_result_month = df_result_month[df_result_month["jaar"] != 2024]
    df_hartvaat = get_maandelijkse_overlijdens(oorzaak)
    
    df_month = pd.merge(df_result_month, df_hartvaat, on="YearMonth", how="inner") 
    df_month["maand"] = (df_month["YearMonth"].str[5:]).astype(int)
    
    data_dict,_,_,_,_ = perform_analyse(age_sex, df_month, time_period, "RNA_flow_per_100000","TotalDoses",f"OBS_VALUE_{oorzaak}", seizoen, maand, normalize)
    return data_dict
def perform_analyse(age_sex, df, time_period,x1,x2,y, seizoen, maand, normalize):
    """_summary_

    Args:
        age_sex (_type_): _description_
        df (_type_): _description_
        time_period (_type_): _description_
        x1 (_type_): _description_
        x2 (_type_): _description_
        y (_type_): _description_
        seizoen (_type_): _description_
        maand (_type_): _description_

    Returns:
        tuple: data_dict
        
                    C_RNA':model.params["RNA_flow_per_100000"],
                    'C_vacc':model.params["TotalDoses"],
                    'P_const': model.pvalues["const"],
                    'P_RNA':model.pvalues["RNA_flow_per_100000"],
                    'P_vacc':model.pvalues["TotalDoses"],
                    'R-squared': [model.rsquared], #* len(model.params),
                    'Adjusted R-squared': [model.rsquared_adj], # * len(model.params),
                    'F-statistic': [model.fvalue], # * len(model.params),
                    'F-statistic P-value':
                max_lag,max_corr,max_lag_sma,max_corr_sma
    """    
    # Voeg een sinus- en cosinusfunctie toe om seizoensinvloeden te modelleren
    try:           
        df['sin_time'] = np.sin(2 * np.pi * df['maand']/ 12)
        df['cos_time'] = np.cos(2 * np.pi * df['maand'] / 12)
        m= True
    except:
       
        df['sin_time'] = np.sin(2 * np.pi * df['week']/ 52)
        df['cos_time'] = np.cos(2 * np.pi * df['week'] / 52)
        m=False
        
    x_values = [x1,x2] # + 
    if seizoen:
        x_values += ['sin_time', 'cos_time']
    if maand:
        if m:
            x_values += ['maand']
        else:
            x_values += ['week']
    y_value_ = y
    
    #col1,col2=st.columns(2)
    col1,col2,col3=st.columns(3)
    with col1:
        line_plot_2_axis(df, time_period,y_value_, x1,age_sex)
        make_scatterplot(df, y_value_, x1,age_sex)
  
    with col2:
        line_plot_2_axis(df, time_period,y_value_, x2,age_sex)
        make_scatterplot(df, y_value_, x2,age_sex)
    with col3:
        line_plot_2_axis(df, time_period,x1, x2,age_sex)
        make_scatterplot(df, x1, x2,age_sex)
    try:
        data_dict = multiple_linear_regression(df,x_values,y_value_, age_sex, normalize)
    except:
        data_dict = None
    max_lag,max_corr,max_lag_sma,max_corr_sma = find_lag_time(df, x1, y_value_, -14, 14)
    return data_dict,max_lag,max_corr,max_lag_sma,max_corr_sma



def main():
    st.subheader("Relatie sterfte/rioolwater/vaccins")
    st.info("Inspired by https://www.linkedin.com/posts/annelaning_vaccinatie-corona-prevalentie-activity-7214244468269481986-KutC/")
    
    opdeling = [[0, 120], [15, 17], [18, 24], [25, 49], [50, 59], [60, 69], [70, 79], [80, 120]]
    col1, col2, col3, col4, col5 = st.columns(5, vertical_alignment="center")
    
    with col1:
        fixed_periods = st.checkbox("Fixed periods", False)
    
    if not fixed_periods:
        with col2:
            start_week = st.number_input("Startweek", 1, 52, 1)
        with col3:
            start_jaar = st.number_input("Startjaar", 2020, 2024, 2020)
        with col4:
            eind_week = st.number_input("Eindweek", 1, 52, 52)
        with col5:
            eind_jaar = st.number_input("Eindjaar", 2020, 2024, 2024)
        
        pseudo_start_week = start_jaar * 52 + start_week
        pseudo_eind_week = eind_jaar * 52 + eind_week
        
        if pseudo_start_week >= pseudo_eind_week:
            st.error("Eind kan niet voor start")
            st.stop()
    
    df = get_sterfte(opdeling)
    df_rioolwater = get_rioolwater()
    df_vaccinaties = get_vaccinaties()
    df_oversterfte = get_oversterfte(opdeling)
    df_ziekenhuis_ic = get_ziekenhuis_ic()

    df_oversterfte["age_sex"] = df_oversterfte["age_sex"].replace("Y0-120_T", "TOTAL_T")
    
    df_merged = (
        pd.merge(df, df_rioolwater,  on=["jaar", "week"], how="left")
        .merge(df_ziekenhuis_ic, on=["jaar", "week"], how="left")
        .merge(df_vaccinaties, on=["jaar", "week", "age_sex"], how="left")
        .fillna(0).infer_objects(copy=False)
        .merge(df_oversterfte, on=["jaar", "week", "age_sex"], how="left")
    )
    
    df_merged["pseudoweek"] = df_merged["jaar"] * 52 + df_merged["week"]


    col1, col2, col3, col4,col5,col6 = st.columns(6, vertical_alignment="center")
    with col1:
        y_value = st.selectbox("Y value", ["OBS_VALUE", "oversterfte", "p_score", "Hospital_admission",  "IC_admission"], 0, help="Alleen bij leeftijdscategorieen")
    with col2:
        normalize = st.checkbox("Normaliseer X values", True, help="Normalizeren omdat de vaccindosissen een hoog getal kunnen zijn")
    with col3:
        seizoen = st.checkbox("Seizoensinvloeden meenemen", True)
    with col4:
        maand = st.checkbox("Maand-/week invloeden meenemene")
    with col5:
        shift_weeks = st.slider(f"Shift {y_value}", -52,52,0)
    with col6:
        window = st.slider(f"SMA window {y_value}", 1,52,1)
                
    if fixed_periods:
        periods = [
            [1, 2020, 26, 2021],
            [27, 2021, 26, 2022],
            [27, 2022, 26, 2023],
            [27, 2023, 52, 2024],
            [1,2022,52,2023],
            [1,2020,52,2024]
        ]
        results = []
        with st.expander("results"):
        #if 1==1:
            for start_wk, start_yr, end_wk, end_yr in periods:
                pseudo_start_week = start_yr * 52 + start_wk
                pseudo_eind_week = end_yr * 52 + end_wk
                
                st.subheader(f"{start_wk}-{start_yr} -- {end_wk}-{end_yr}")
                df_period = df_merged[(df_merged["pseudoweek"] >= pseudo_start_week) & (df_merged["pseudoweek"] <= pseudo_eind_week)]
                df_period = df_period[df_period["week"] != 53]
                
                age_sex = "TOTAL_T"
                df_filtered = df_period[df_period["age_sex"] == age_sex].copy(deep=True)
                df_filtered[y_value] = df_filtered[y_value].rolling(window=window, center=True).mean()
                df_filtered[y_value] = df_filtered[y_value].shift(shift_weeks)
                
                data_dict ,max_lag,max_corr,max_lag_sma,max_corr_sma = perform_analyse(age_sex, df_filtered, "jaar_week", "RNA_flow_per_100000", "TotalDoses", y_value, seizoen, maand, normalize)
              
           
                period = f"{start_wk}-{start_yr}-{end_wk}/{end_yr}"

                result = {
                    "period": period,
                    "Y value": data_dict['Y value'],
                    "coef_RNA": round(data_dict['C_RNA'],4),
                    "coef_vacc": round(data_dict['C_vacc'],4),
                    "p_RNA": round(data_dict['P_RNA'],4),
                    "p_vacc": round(data_dict['P_vacc'],4),
                    "Adj. R2": round(data_dict['Adjusted R-squared'],4),
                    "F-stat.": round(data_dict['F-statistic'],4),
                    "p_F-stat.": round(data_dict['F-statistic P-value'],4),
                    "max_lag_days": max_lag,
                    "max_corr": max_corr,
                    "max_lag_days_sma_(7)": max_lag_sma,
                    "max_corr_sma_(7)": max_corr_sma
                }
                
                # Append the result dictionary to the results list
                results.append(result)

        # Convert the results list to a dataframe
        df_results = pd.DataFrame(results)

        # Display the resulting dataframe

        st.subheader("Results")
        st.write(df_results)
        
    else:
        # not a loop of fixed periods. Just one period
        df_period = df_merged[(df_merged["pseudoweek"] >= pseudo_start_week) & (df_merged["pseudoweek"] <= pseudo_eind_week)]
        df_period = df_period[df_period["week"] != 53]
        age_sex = "TOTAL_T"
        df_filtered = df_period[df_period["age_sex"] == age_sex]
        df_filtered[y_value] = df_filtered[y_value].rolling(window=window, center=True).mean()
        df_filtered[y_value] = df_filtered[y_value].shift(shift_weeks)
                
        with st.expander("Rioolwater"):
            compare_rioolwater(df_rioolwater)
        
        with st.expander("OBS VALUE - oversterfte - Pvalue"):
            
            col1, col2, col3 = st.columns(3)
            with col1:
                line_plot_2_axis(df_filtered,  "TIME_PERIOD_x", "OBS_VALUE", "oversterfte", age_sex)
                make_scatterplot(df_filtered, "OBS_VALUE", "oversterfte", age_sex)
            with col2:
                line_plot_2_axis(df_filtered,  "TIME_PERIOD_x", "OBS_VALUE", "p_score", age_sex)
                make_scatterplot(df_filtered, "OBS_VALUE", "p_score", "")
            with col3:
                line_plot_2_axis(df_filtered,  "TIME_PERIOD_x", "base_value", "OBS_VALUE", age_sex)
                make_scatterplot(df_filtered, "base_value", "OBS_VALUE", age_sex)
        
        # Analyze based on age groups and causes
        age_sex_list = ["TOTAL_T"] if y_value in ["oversterfte", "p_score"] else df["age_sex"].unique().tolist()
        
        results = []
        df_complete=pd.DataFrame()
        for age_sex in age_sex_list:
            df_result = df_period[df_period["age_sex"] == age_sex].copy()
            df_result["TotalDoses"].fillna(0)
            
            if age_sex == "TOTAL_T":
                for oorzaak in ["hart_vaat_ziektes", "covid", "ademhalingsorganen", "accidentele_val", "wegverkeersongevallen", "nieuwvormingen"]:
                    if df_result["TotalDoses"].sum() != 0:
                        with st.expander(oorzaak):
                            st.subheader(f"TOTAL overlijdens {oorzaak} vs rioolwater en vaccinaties")
                            
                            df_iteration = analyse_maandelijkse_overlijdens(oorzaak, age_sex, df_result, "YearMonth", seizoen, maand, normalize)
                            results.append(df_iteration)
            
            time_period = "YearMonth" if maand else "TIME_PERIOD_x"
            #st.write(df_result)
            if df_result["TotalDoses"].sum() != 0:
                with st.expander(f"{age_sex} - Alle overlijdensoorzaken"):
                    st.subheader(f"{age_sex} - Alle overlijdensoorzaken")
                    st.write(df_result)
                    #df_result.to_csv(f"{age_sex}")
                    data_dict,_,_,_,_ = perform_analyse(age_sex, df_result, time_period, "RNA_flow_per_100000", "TotalDoses", y_value, seizoen, maand, normalize)
                    
                    xx=data_dict["Y value"]
                    y_value_x =  f"{xx}_{age_sex}"
                    # df_iteration = pd.DataFrame({
                    #     "Y value":y_value_x,
                    
                    #     #'P_const': data['P_const'],
                    #     'coef_RNA':data_dict['C_RNA'],
                    #     'coef_vacc':data_dict['C_vacc'],

                    #     'p_RNA':data_dict['P_RNA'],
                    #     'p_vacc':data_dict['P_vacc'],

                    #     #"R-squared": data["R-squared"],
                    #     "Adj. R2": data_dict["Adjusted R-squared"],
                    #     "F-stat.": data_dict["F-statistic"],
                    #     "p_F-stat.": data_dict["F-statistic P-value"]
                    # })

                    
                    # Append the DataFrame to the results list
                    results.append(data_dict)
      
        # Convert the results list to a dataframe
        df_results = pd.DataFrame(results)

        # Display the resulting dataframe

        st.subheader("Results")
        st.write(df_results)

        # st.write(df_complete)
        make_scatterplot(df_results, "F-statistic P-value", "Adjusted R-squared", "")
    
    st.subheader("Data sources")
    st.info("https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")
    st.info("https://www.ecdc.europa.eu/en/publications-data/data-covid-19-vaccination-eu-eea")
    st.info("https://www.rivm.nl/corona/actueel/weekcijfers")

if __name__ == "__main__":
    import os
    import datetime
    
    os.system('cls')
    print(f"-------xx-------{datetime.datetime.now()}-------------------------")
    main()
  