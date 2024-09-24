from fit_to_data_streamlit import *
from mortality_yearly_per_capita import get_sterfte
import streamlit as st
from scipy.optimize import curve_fit
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score


# Function to calculate the exponential with constants a and b
def exponential(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Calculate the exponential function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): Coefficient for exponential.
        b (float): Exponent.

    Returns:
        np.ndarray: The result of a * exp(b * x).
    """
    return a*np.exp(b*x)






@st.cache_data()
def get_data() -> pd.DataFrame:
    """
    Fetch mortality data using `get_sterfte` function with age group breakdown.

    Returns:
        pd.DataFrame: A DataFrame containing mortality data for different age groups.
    """
    # put in a seperate function to enable caching
    opdeling = [[0,49], [50,64], [65,79], [80,89], [90,120],[80,120], [0,120]]

    df =  get_sterfte(opdeling, "NL")
    
    return df


def main_(df: pd.DataFrame, value_field: str, age_group: str, sexe: str, START_YEAR: int, verbose: bool) -> tuple[float, float]:  
    """Main analysis function: performs exponential and linear curve fitting, projections, and plotting.

    Args:
        df (pd.DataFrame): Input DataFrame containing mortality data.
        value_field (str): Field to perform fitting on (e.g., 'OBS_VALUE', 'per100k').
        age_group (str): Age group for analysis.
        sexe (str): Gender category ('T', 'M', 'V').
        START_YEAR (int): Year from which to start the analysis.
        verbose (bool) : show graphs
    Returns:
        excess_mortality_lineair
        excess_mortality_exponential
    """
    df_before_2020, df_2020_and_up = prepare_data(df, age_group, sexe, START_YEAR)
    x_=df_before_2020["jaar"]
    y_ = df_before_2020[value_field]
    # Fit the dummy exponential data
    pars, cov = curve_fit(f=exponential, xdata=x_, ydata=y_, p0=[0, 0], bounds=(-np.inf, np.inf), maxfev=20000)

    trendline, extended_years, trendline_extended= fit_and_predict(df_before_2020, x_, y_)
   
    if value_field == 'OBS_VALUE':
        df_before_2020['predicted_deaths'] = trendline
    else:
        df_before_2020['predicted_per100k'] = trendline
        
    if value_field == 'OBS_VALUE':
        df_extended = pd.merge(df_2020_and_up, pd.DataFrame({
                    'jaar': extended_years,
                    'predicted_deaths': trendline_extended
                    }), on='jaar')
    else:
        df_extended = pd.merge(df_2020_and_up, pd.DataFrame({
                'jaar': extended_years,
                'predicted_per100k': trendline_extended
                }), on='jaar')
        #df_extended['predicted_deaths'] = df_extended['predicted_per100k']*df_extended['aantal']/100000
        
    # Concatenate the original and extended DataFrames
    df_diff = pd.concat([df_before_2020, df_extended], ignore_index=True)

    # Optionally, sort by year
    df_diff = df_diff.sort_values(by='jaar').reset_index(drop=True)
    
    if value_field == 'per100k':
        df_diff['predicted_deaths'] = df_diff['predicted_per100k']*df_diff['aantal']/100000

    df_diff = do_calculations_df_diff(pars, df_diff) 
    
    if verbose:
        plot_fitting_on_value_field(value_field, df_before_2020, df_2020_and_up, trendline, extended_years, trendline_extended, df_diff, age_group, sexe)

        if value_field =="per100k":
            st.subheader("**From per 100k transformation back to Absolute Numbers**")
            plot_group_size(df_diff,  age_group, sexe)
            plot_transformed_to_absolute(df_before_2020, df_2020_and_up, df_diff, age_group, sexe)
        
    excess_mortality_lineair, excess_mortality_exponential = show_excess_mortality(value_field, df_diff, verbose)
    return  excess_mortality_lineair, excess_mortality_exponential
def show_excess_mortality(value_field: str, df_diff: pd.DataFrame, verbose: bool) -> None:
    """
    Display the excess mortality figures based on the chosen fitting method (linear/exponential).

    Args:
        value_field (str): Field used in the analysis ('OBS_VALUE' or 'per100k').
        df_diff (pd.DataFrame): DataFrame containing observed and predicted mortality data.
        verbose (bool) : give output
    Returns:
        None
    """
    excess_mortality_lineair = round(df_diff[df_diff['jaar'].between(2020, 2023)]['oversterfte'].sum())
    if value_field =="per100k":
        excess_mortality_exponential = round(df_diff[df_diff['jaar'].between(2020, 2023)]['oversterfte_expon'].sum())
    else:
        excess_mortality_exponential = round(df_diff[df_diff['jaar'].between(2020, 2023)]['oversterfte_expon_totals'].sum())
    
    if verbose:
        st.write(f"{value_field} - Excess mortality lineair {excess_mortality_lineair}")
        st.write(f"{value_field} - Excess mortality exponential {excess_mortality_exponential}")
    return excess_mortality_lineair, excess_mortality_exponential
def do_calculations_df_diff(pars: np.ndarray, df_diff: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate excess mortality, fitted curves, and other metrics for the given DataFrame.

    Args:
        pars (np.ndarray): Parameters of the exponential fit.
        df_diff (pd.DataFrame): DataFrame with observed and predicted mortality data.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated fields.
    """

    df_diff['oversterfte'] = round(df_diff['OBS_VALUE'] - df_diff['predicted_deaths']) 
    df_diff['aantal']=round(df_diff['aantal'])
    df_diff['percentage'] = round(((df_diff['OBS_VALUE'] - df_diff['predicted_deaths'])/df_diff['predicted_deaths'])*100,1)
    df_diff['fitted_curve'] = exponential(df_diff["jaar"], *pars)
    df_diff['fitted_curve_transf_absolut'] = df_diff['fitted_curve'] *df_diff['aantal'] /100000
    df_diff['oversterfte_expon_totals'] = df_diff['OBS_VALUE'] -  df_diff['fitted_curve']
    df_diff['oversterfte_expon'] = round(df_diff['OBS_VALUE'] - df_diff['fitted_curve_transf_absolut'])
       # Concatenate the original and extended DataFrames    
    # Optionally, sort by year
    df_diff = df_diff.sort_values(by='jaar').reset_index(drop=True)
    return df_diff

def fit_and_predict(df_before_2020: pd.DataFrame, x_: pd.Series, y_: pd.Series) -> tuple:
    """
    Fit a linear trend and predict future values.

    Args:
        df_before_2020 (pd.DataFrame): DataFrame containing data before 2020.
        x_ (pd.Series): Series containing years.
        y_ (pd.Series): Series containing values (e.g., OBS_VALUE or per100k).

    Returns:
        tuple: trendline for historical data, extended years, trendline for extended period.
    """
    X = sm.add_constant(x_)  # Adds a constant term to the predictor
    model = sm.OLS(y_, X).fit()
    trendline = model.predict(X)
    extended_years = np.arange(df_before_2020["jaar"].min(), 2025)
    # Create a DataFrame for the extended years
    extended_X = sm.add_constant(extended_years)
    # Predict the trendline and bounds for the extended years
    trendline_extended = model.predict(extended_X)
    
    return trendline,extended_years,trendline_extended


def plot_fitting_on_value_field(value_field: str, df_before_2020: pd.DataFrame, df_2020_and_up: pd.DataFrame, trendline: np.ndarray, extended_years: np.ndarray, trendline_extended: np.ndarray, df_diff: pd.DataFrame,age_group: str, sexe: str, ) -> None:
    """
    Plot the fitting results, including data before and after 2020, trendlines, and exponential fits.

    Args:
        value_field (str): The field used for plotting (e.g., 'OBS_VALUE', 'per100k').
        df_before_2020 (pd.DataFrame): DataFrame containing data before 2020.
        df_2020_and_up (pd.DataFrame): DataFrame containing data from 2020 and onwards.
        trendline (np.ndarray): Linear trendline fitted on data before 2020.
        extended_years (np.ndarray): Array of extended years for future projections.
        trendline_extended (np.ndarray): Linear trendline extended to future years.
        df_diff (pd.DataFrame): DataFrame with all data points and predictions.
        r2(float): R2 score of the trendline
        age_group (str): Age group
        sexe (str): Gender category ('T', 'M', 'V').

    Returns:
        None
    """

    fig = go.Figure()
        # Plot bars before 2020
    fig.add_trace(go.Scatter(
        x=df_before_2020["jaar"],
        y=df_before_2020[value_field],
        name=f'before 2020',
        mode='markers',
        marker=dict(color="blue")
    ))

    # Plot bars for 2020 and up
    fig.add_trace(go.Scatter(
        x=df_2020_and_up["jaar"],
        y=df_2020_and_up[value_field],
        name=f'2020 and up',
        mode='markers',
        marker=dict(color='red')  # Set the color to red for years >= 2020
    ))

    
    fig.add_trace(go.Scatter(x=df_before_2020["jaar"], y=trendline, 
                                    mode='lines', name=f'Trendline OLS till 2019', line=dict(color="green")))
    
    
    fig.add_trace(go.Scatter(
                    x=extended_years,
                    y=trendline_extended,
                    mode='lines',
                    name=f'Trendline OLS until 2024',
                    line=dict(color="green")
                ))
    
    #add the fitted curve
    fig.add_trace(go.Scatter(x=df_diff["jaar"], y=df_diff["fitted_curve"], mode='lines', marker=dict(color='yellow'), name='Fitted Exponential Curve'))
    # Exclude the last four values
    df_filtered = df_diff[:-4]  # Slices the DataFrame to exclude the last 4 rows

    # Calculate R² score
    r2_a = round(r2_score(df_filtered[value_field], trendline),4)
    r2_b = round(r2_score(df_filtered[value_field], df_filtered["fitted_curve"]),4)

    
    fig.update_layout(
                title=f"{age_group} - {sexe} | {value_field} | r2 : green  {r2_a}- yellow {r2_b}",
                xaxis_title="Year",
                yaxis_title=value_field,
            )
    st.plotly_chart(fig)

def plot_transformed_to_absolute(df_before_2020: pd.DataFrame, df_2020_and_up: pd.DataFrame, df_diff: pd.DataFrame, age_group: str, sexe: str) -> None:
    """
    Plot the observed deaths and predicted deaths from both trendline and exponential curve.

    Args:
        df_before_2020 (pd.DataFrame): DataFrame containing data before 2020.
        df_2020_and_up (pd.DataFrame): DataFrame containing data from 2020 onwards.
        df_diff (pd.DataFrame): DataFrame with calculated fields, including predicted deaths and fitted curves.
        age_group (str): Age group 
        sexe (str): Gender category ('T', 'M', 'V').
    Returns:
        None
    """
    fig = go.Figure()
        # Plot bars before 2020
    fig.add_trace(go.Scatter(
            x=df_before_2020["jaar"],
            y=df_before_2020["OBS_VALUE"],
            name=f'before 2020',
            mode='markers',
            marker=dict(color="blue")
        ))

        # Plot bars for 2020 and up
    fig.add_trace(go.Scatter(
            x=df_2020_and_up["jaar"],
            y=df_2020_and_up["OBS_VALUE"],
            name=f'2020 and up',
            mode='markers',
            marker=dict(color='red')  # Set the color to red for years >= 2020
        ))
         # Plot bars for 2020 and up
    fig.add_trace(go.Scatter(
            x=df_diff["jaar"],
            y=df_diff["predicted_deaths"],
            name=f'trendline OLS',
            marker=dict(color='green')  
        ))
        #add the fitted curve
        #fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', marker=dict(color='yellow'), name='Fitted Curve'))
    df_diff["fitted_aantal"] = df_diff["fitted_curve"] * df_diff["aantal"]/100000
    fig.add_trace(go.Scatter(x=df_diff["jaar"], y=df_diff["fitted_aantal"], mode='lines', marker=dict(color='yellow'), name='Fitted Exponential  Curve'))
    

    # Exclude the last four values
    df_filtered = df_diff[:-4]  # Slices the DataFrame to exclude the last 4 rows

    # Calculate R² score
    r2 = round(r2_score(df_filtered["OBS_VALUE"], df_filtered["predicted_deaths"]),4)
    r2_b = round(r2_score(df_filtered["OBS_VALUE"], df_filtered["fitted_aantal"]),4)
    fig.update_layout(
                title=f"{age_group} - {sexe} | Deaths Transformed from relatieve back to absolute numbers | r2 green: {r2} | yellow: {r2_b}",
                xaxis_title="Year",
                yaxis_title="Deaths",
            )
    st.plotly_chart(fig)

def plot_group_size(df_diff: pd.DataFrame, age_group: str, sexe: str ) -> None:
    """
    Plot the group size (population count) over the years.

    Args:
        df_diff (pd.DataFrame): DataFrame containing population counts by year.
        age_group (str): Age group.
        sexe (str): Gender category ('T', 'M', 'V').
        
    Returns:
        None
    """
    fig = go.Figure()
        # Plot bars before 2020
    fig.add_trace(go.Bar(
            x=df_diff["jaar"],
            y=df_diff["aantal"],
            name=f'before 2020',
            marker=dict(color="blue")
        ))
    fig.update_layout(
                title=f"{age_group} - {sexe} | Number of people in the population",
                xaxis_title="Year",
                yaxis_title="Deaths",
            )
    st.plotly_chart(fig)

def prepare_data(df: pd.DataFrame, age_group: str, sexe: str, START_YEAR: int) -> tuple:
    """
    Filter the DataFrame based on age group, gender, and start year, splitting it into pre-2020 and post-2020 data.

    Args:
        df (pd.DataFrame): Original DataFrame with mortality data.
        age_group (str): Age group to filter by.
        sexe (str): Gender category ('T', 'M', 'V').
        START_YEAR (int): Year from which to start the analysis.

    Returns:
        tuple: DataFrames for data before 2020 and for 2020 onwards.
    """
    df=df[df["age_group"] == age_group]
    df=df[df["geslacht"] == sexe]
    
    df_before_2020 = df[(df["jaar"] >= START_YEAR) & (df["jaar"] < 2020)]
    df_2020_and_up = df[df["jaar"] >= 2020]
    return df_before_2020,df_2020_and_up

@st.cache_data()
def calculate_results(df: pd.DataFrame, age_groups_selected: list[str], start_years: list[int], sexe: str, verbose: bool) -> pd.DataFrame: 
    """
    Calculate excess mortality using both linear and exponential models for each age group, 
    value field, and start year combination. The function caches the result to optimize performance 
    for repeated calculations in Streamlit.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing mortality and population data.
    age_groups_selected : list
        A list of age groups for which to calculate the excess mortality.
    start_years : list
        A list of starting years for which the models should be calculated.
    sexe : str
        The sex category to filter the dataframe (e.g., 'M' for male, 'F' for female).
    verbose : bool
        If True, print detailed progress information during the calculation.

    Returns:
    --------
    df_results : pd.DataFrame
        A dataframe containing the results of excess mortality calculations for both 
        linear and exponential models. Each row includes the start year, model type 
        (linear or exponential), value field (e.g., 'OBS_VALUE', 'per100k'), age group, 
        and calculated excess mortality.

    Notes:
    ------
    The value fields 'OBS_VALUE' and 'per100k' are calculated for each model, age group and start year.
    """
    # Define the start years for subcolumns
    counter = 0
    total = 2* len(age_groups_selected)*len(start_years)

    # Initialize DataFrames to store the results
    results = []
   
    
    for value_field in ["OBS_VALUE", "per100k"]:

        
        # Initialize an empty list to hold result rows
        

        # Loop through each age group
        for age_group in age_groups_selected:

            
            # Loop through each start year and calculate the results
            for START_YEAR in start_years:
                print (f"{counter+1}/{total} | {value_field=} - {age_group=} { START_YEAR=}")
                excess_mortality_lineair, excess_mortality_exponential = main_(df, value_field, age_group, sexe, START_YEAR, verbose)
                
                # Append results for lineair model
                results.append({
                    "start_year": START_YEAR,
                    "model": "lineair",
                    "value_field": value_field,
                    "age_group": age_group,
                    "excess_mortality": excess_mortality_lineair
                })

                # Append results for exponential model
                results.append({
                    "start_year": START_YEAR,
                    "model": "exponential",
                    "value_field": value_field,
                    "age_group": age_group,
                    "excess_mortality": excess_mortality_exponential
                })
                counter +=1

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

        
   
    return df_results
    
def main() -> None:
    """
    Main function for the Streamlit application that analyzes mortality data using linear and 
    exponential fitting models.

    Args:
        None

    Returns:
        None
    """
    st.subheader("Mortality Analysis Using Exponential Curve Fitting and Trendline Projection")
    st.info("""
            This Streamlit application analyzes mortality data for specific age 
            and sex groups using historical trends. The script leverages both linear 
            and exponential curve fitting to predict future mortality rates and 
            over-mortality for the years 2020 and beyond. 
            
            * Fitting the Model on Absolute Figures: We first apply exponential curve 
            fitting to the absolute mortality figures (total number of deaths) for the 
            years before 2020. This helps us understand the overall trend.

            * Fitting the Model on Relative Numbers: Next, we fit the model on relative 
            numbers, such as deaths per 100,000 people. This allows for a normalized 
            comparison across different population sizes.

            * Transformation to Absolute Numbers: After fitting the model on the 
            relative numbers, we transform these results back into absolute figures 
            (total deaths) for easier interpretation and comparison with actual data.
            
            Inspired by https://twitter.com/rcsmit/status/1838204715424755786 """)
    df = get_data() 
    age_groups  = df["age_group"].unique().tolist()  #[:5] 
   

    what_to_do = st.sidebar.selectbox("What to do [selection|akk]", ["selection", "all"],0)
    sexe = st.sidebar.selectbox("Sexe [T|M|V]", ["T","M","V"],0)
    if what_to_do == "selection":
        age_groups_selected = [st.sidebar.selectbox("age group", age_groups)]
        start_years = [st.sidebar.number_input("Fitting from year",2000,2019,2010)]
        verbose=True
    else:
        start_years = [2000, 2010, 2015]
        verbose = False
        age_groups_selected = age_groups
        possible_columns = [
                            ['model', 'value_field', 'start_year'],
                            ['model', 'start_year', 'value_field'],
                            ['value_field', 'model', 'start_year'],
                            ['value_field', 'start_year', 'model'],
                            ['start_year', 'model', 'value_field'],
                            ['start_year', 'value_field', 'model']
                        ]
        columns = st.sidebar.selectbox("Column hierarchie", possible_columns,0)
    
    
    df_results = calculate_results(df,age_groups_selected, start_years, sexe, verbose)
     # Pivot the DataFrame to create a multi-level column structure
    df_pivot = df_results.pivot_table(
        index='age_group',
        columns=columns,
        values='excess_mortality'
    )

    # # Flatten the multi-level columns for easier display
    # df_pivot.columns = [f"{year} | {model} | {field}" for year, model, field in df_pivot.columns]

    
    for col in df_pivot.columns:
        df_pivot[col] = df_pivot[col].astype(str)
    # Display the results as a table in Streamlit
    st.subheader("Excess Mortality Comparison")
    st.dataframe(df_pivot)
       
if __name__ == "__main__":
    import datetime
    print (f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------")
    main()