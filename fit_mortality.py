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


def exponential(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Calculate the exponential function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): Amplitude of the exponential function.
        b (float): Growth rate.

    Returns:
        np.ndarray: The result of a * exp(b * x).
    """
    return a * np.exp(b * x)

def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Calculate the quadratic function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): Coefficient of x^2.
        b (float): Coefficient of x.
        c (float): Constant term.

    Returns:
        np.ndarray: The result of a * x^2 + b * x + c.
    """
    return a * x**2 + b * x + c 

def logistic(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Calculate the logistic grwoth function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): Coefficient of x^2.
        b (float): Coefficient of x.
        c (float): Constant term.
        d
    Returns:
        np.ndarray: The result of 
    """
    return a+ ((b-a)/(1+((x/c)**d)))

def gompertz(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Calculate the Gompertz function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): Upper asymptote.
        b (float): Growth displacement.
        c (float): Growth rate.

    Returns:
        np.ndarray: The result of a * exp(-b * exp(-c * x)).
    """
    return a * np.exp(-b * np.exp(-c * x))

def first_derivative_gompertz(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Calculate the first derivative of the Gompertz function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): Upper asymptote.
        b (float): Growth displacement.
        c (float): Growth rate.

    Returns:
        np.ndarray: The result of the first derivative of the Gompertz function.
    """
    return a * b * c * np.exp(b * (-1 * np.exp(-c * x)) - c * x)

def gaussian(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Calculate the Gaussian function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): Amplitude.
        b (float): Mean (center of the peak).
        c (float): Standard deviation (width of the curve).

    Returns:
        np.ndarray: The result of a * exp(-((x - b)^2) / c).
    """
    return a * np.exp(-((x - b) ** 2) / c)

def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Calculate the linear function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): y-intercept.
        b (float): Slope.

    Returns:
        np.ndarray: The result of a + (b * x).
    """
    return a + (b * x)

def exponential_2(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Calculate an alternative exponential function.

    Args:
        x (np.ndarray): Input array (independent variable).
        a (float): Initial value.
        b (float): Growth rate.

    Returns:
        np.ndarray: The result of a * ((1 + b)^x).
    """
    return a * ((1 + b)**x)

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
    #df=df[df["jaar"]>1999]
    
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
    df["age_sex"] = df["age_group"] + "_" +df["geslacht"]
  
  

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
    df_eind = df_eind[df_eind["jaar"] != 2025]
    df_eind["per100k"] = round(df_eind["OBS_VALUE"]/df_eind["aantal"]*100000,1) 
    
    return df_eind 
def main_(df: pd.DataFrame, value_field: str, age_group: str, sexe: str, START_YEAR: int, verbose: bool, secondary_choice_: list[str], show_confidence_intervals: bool, doordsoorzaak_keuze:str, what_to_plot:list[str], scaled:bool) -> tuple[float, float]:
#def main_(df: pd.DataFrame, value_field: str, age_group: str, sexe: str, START_YEAR: int, verbose: bool, secondary_choice:str) -> tuple[float, float]:  
    """Main analysis function: performs secondary (exponential or quadratic) and linear curve fitting, projections, and plotting.

    Args:
        df (pd.DataFrame): Input DataFrame containing mortality data.
        value_field (str): Field to perform fitting on (e.g., 'OBS_VALUE', 'per100k').
        age_group (str): Age group for analysis.
        sexe (str): Gender category ('T', 'M', 'V').
        START_YEAR (int): Year from which to start the analysis.
        verbose (bool) : show graphs
        secondary_choice: str
        show_confidence_intervals
        doordsoorzaak_keuze
        what_to_plot
        scaled
    Returns:
        excess_mortality_lineair
        excess_mortality_secondary
    """
    df_before_2020, df_2020_and_up = prepare_data(df, age_group, sexe, START_YEAR)
    x_=df_before_2020["jaar"]
    y_ = df_before_2020[value_field]
   
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
    
    df_diff = do_calculations_df_diff_lineair( df_diff) 
    result_str =[]
    # Fit the dummy secondary data
    for secondary_choice in secondary_choice_:
        try:
         
            # Dictionary to store function-specific information
            function_info = {
                "quadratic": {
                    "func": quadratic,
                    "p0": [1, 1, 1],
                    "equation": "a*x^2 + b*x + c",
                    "params": ["a", "b", "c"]
                },
                "exponential": {
                    "func": exponential,
                    "p0": [1, 1],
                    "equation": "a * exp(b*x)",
                    "params": ["a", "b"]
                },
                "gompertz": {
                    "func": gompertz,
                    "p0": [1, 1, 1],
                    "equation": "a * exp(-b * exp(-c * x))",
                    "params": ["a", "b", "c"]
                },
                "first_derivative_gompertz": {
                    "func": first_derivative_gompertz,
                    "p0": [1, 1, 1],
                    "equation": "a * b * c * exp(b * (-1 * exp(-c * x)) - c * x)",
                    "params": ["a", "b", "c"]
                },
                "gaussian": {
                    "func": gaussian,
                    "p0": lambda x: [1, np.mean(x), np.std(x)],
                    "equation": "a * exp(-((x - b)^2) / c)",
                    "params": ["a", "b", "c"]
                },
                "linear": {
                    "func": linear,
                    "p0": [1, 1],
                    "equation": "a + b*x",
                    "params": ["a", "b"]
                },
                "exponential_2": {
                    "func": exponential_2,
                    "p0": [1, 1],
                    "equation": "a * ((1 + b)^x)",
                    "params": ["a", "b"]
                },
                 "logistic": {
                    "func": logistic,
                    "p0": [1, 1,1,1],
                    "equation": "a+ ((b-a)/(1+((x/c)**d)))",
                    "params": ["a", "b","c","d"]
                }
            }

            if secondary_choice in function_info:
                info = function_info[secondary_choice]
                p0 = info["p0"](x_) if callable(info["p0"]) else info["p0"]
                
                pars, cov = curve_fit(f=info["func"], xdata=x_, ydata=y_, p0=p0, bounds=(-np.inf, np.inf), maxfev=20000)
                
                param_str = ", ".join(f"{param} = {value:.4f}" for param, value in zip(info["params"], pars))
                result_str.append(f"*{secondary_choice}* - {info['equation']} | {param_str}")
                
            else:
                st.warning(f"Error in secondary choice {secondary_choice}.")
                st.stop()

            df_diff = do_calculations_df_diff_secondary_choice(pars, cov, df_diff, secondary_choice) 
       
        except Exception as error:
            print (f"No fitting possible for {secondary_choice} - {error}")

    if verbose:
        show_result_str = False
        if ( ((value_field == 'OBS_VALUE') and ("OBS_VALUE" in what_to_plot)) or ((value_field == 'per100k') and ("per100k" in what_to_plot))):
            if not scaled:
                plot_fitting_on_value_field(value_field, df_before_2020, df_2020_and_up, trendline, extended_years, trendline_extended, df_diff, age_group, sexe, secondary_choice_, doordsoorzaak_keuze)
            else:
                plot_fitting_on_value_field_scaled(value_field, df_before_2020, df_2020_and_up, trendline, extended_years, trendline_extended, df_diff, age_group, sexe, secondary_choice_, doordsoorzaak_keuze)
            
            show_result_str = True
        if value_field =="per100k":
            if ("number_of_people" in what_to_plot) or "100k_to_population" in what_to_plot:
                st.subheader("**From per 100k transformation back to Absolute Numbers**")
            if "number_of_people" in what_to_plot:
                plot_group_size(df_diff,  age_group, sexe,doordsoorzaak_keuze)
                show_result_str = True
            if "100k_to_population" in what_to_plot:
                plot_transformed_to_absolute(df_before_2020, df_2020_and_up, df_diff, age_group, sexe, secondary_choice_,doordsoorzaak_keuze)
                show_result_str = True
        if show_result_str :
            for r in result_str:
                st.write(r)  
        else:
            verbose=False  
        excess_mortality_lineair, excess_mortality_secondary_ = show_excess_mortality(value_field, df_diff, verbose,secondary_choice_)
    
        
    return  excess_mortality_lineair, excess_mortality_secondary_
def show_excess_mortality(value_field: str, df_diff: pd.DataFrame, verbose: bool, secondary_choice_:list[str]) -> None:
    """
    Display the excess mortality figures based on the chosen fitting method (linear/secondary).

    Args:
        value_field (str): Field used in the analysis ('OBS_VALUE' or 'per100k').
        df_diff (pd.DataFrame): DataFrame containing observed and predicted mortality data.
        verbose (bool) : give output
        secondary_choice (str):
    Returns:
        None
    """
  
    excess_mortality_lineair = round(df_diff[df_diff['jaar'].between(2020, 2023)]['oversterfte'].sum())
    # if verbose:
    #     st.write(f"{value_field} - Excess mortality lineair {excess_mortality_lineair} | {round(excess_mortality_lineair/4)} per year")
    excess_mortality_secondary_ = []
    for secondary_choice in secondary_choice_:
        try:
            if value_field =="per100k":
                excess_mortality_secondary = round(df_diff[df_diff['jaar'].between(2020, 2023)][f'oversterfte_expon_{secondary_choice}'].sum())
            else:
                excess_mortality_secondary = round(df_diff[df_diff['jaar'].between(2020, 2023)][f'oversterfte_expon_totals_{secondary_choice}'].sum())
        except:
            excess_mortality_secondary = None
        if verbose:
            if excess_mortality_secondary is not None:
                excess_per_year = None if excess_mortality_secondary == None else round(excess_mortality_secondary/4)
                st.write(f"{value_field} - Excess mortality {secondary_choice} {excess_mortality_secondary} | {excess_per_year} per year")
    excess_mortality_secondary_.append(excess_mortality_secondary)
    return excess_mortality_lineair, excess_mortality_secondary_

def do_calculations_df_diff_lineair(df_diff: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate excess mortality, fitted curves, and other metrics for the given DataFrame.

    Args:
     
        df_diff (pd.DataFrame): DataFrame with observed and prediuadrcted mortality data.
      
    Returns:
        pd.DataFrame: Updated DataFrame with calculated fields.
    """

    df_diff['oversterfte'] = round(df_diff['OBS_VALUE'] - df_diff['predicted_deaths']) 
    df_diff['aantal']=round(df_diff['aantal'])
    df_diff['percentage'] = round(((df_diff['OBS_VALUE'] - df_diff['predicted_deaths'])/df_diff['predicted_deaths'])*100,1)
    return df_diff
def do_calculations_df_diff_secondary_choice(pars: np.ndarray,pcov:np.ndarray, df_diff: pd.DataFrame, secondary_choice:str) -> pd.DataFrame:
    """
    Calculate excess mortality, fitted curves, and other metrics for the given DataFrame.

    Args:
        pars (np.ndarray): Parameters of the secondary fit.
        pop
        df_diff (pd.DataFrame): DataFrame with observed and prediuadrcted mortality data.
        secondary_choice
    Returns:
        pd.DataFrame: Updated DataFrame with calculated fields.
    """

    # st.write(pcov)
    perr = 0# np.sqrt(np.diag(pcov))
    n_std = 0.0  # 95% confidence interval

    # Dictionary mapping function names to their corresponding functions
    function_map = {
        "exponential": exponential,
        "quadratic": quadratic,
        "gompertz": gompertz,
        "first_derivative_gompertz": first_derivative_gompertz,
        "gaussian": gaussian,
        "linear": linear,
        "exponential_2": exponential_2,
        "logistic":logistic,
    }

    if secondary_choice in function_map:
        func = function_map[secondary_choice]
        
        # Calculate fitted curve and confidence intervals
        df_diff[f'fitted_curve_{secondary_choice}'] = func(df_diff["jaar"], *pars)
        df_diff[f'y_fit_upper_{secondary_choice}'] = func(df_diff["jaar"], *(pars + n_std * perr))
        df_diff[f'y_fit_lower_{secondary_choice}'] = func(df_diff["jaar"], *(pars - n_std * perr))
    else:
        st.write(f"Error in secondary choice |{secondary_choice}|")
        st.stop()
    df_diff[f'fitted_curve_transf_absolut_{secondary_choice}'] = df_diff[f'fitted_curve_{secondary_choice}'] *df_diff['aantal'] /100000
    df_diff[f'oversterfte_expon_totals_{secondary_choice}'] = df_diff['OBS_VALUE'] -  df_diff[f'fitted_curve_{secondary_choice}']
    df_diff[f'oversterfte_expon_{secondary_choice}'] = round(df_diff['OBS_VALUE'] - df_diff[f'fitted_curve_transf_absolut_{secondary_choice}'])
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


def plot_fitting_on_value_field(value_field: str, df_before_2020: pd.DataFrame, df_2020_and_up: pd.DataFrame, trendline: np.ndarray, extended_years: np.ndarray, trendline_extended: np.ndarray, df_diff: pd.DataFrame,age_group: str, sexe: str, secondary_choice_:list[str], doordsoorzaak_keuze:str) -> None:
    """
    Plot the fitting results, including data before and after 2020, trendlines, and secondary fits.

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
        secondary_choice (str): type of fitting for the 2nd choice [exponential|quadratic]
         | {doordsoorzaak_keuze}
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

    
    # fig.add_trace(go.Scatter(x=df_before_2020["jaar"], y=trendline, 
    #                                 mode='lines', name=f'Trendline OLS till 2019', line=dict(color="green")))
    
    
    # fig.add_trace(go.Scatter(
    #                 x=extended_years,
    #                 y=trendline_extended,
    #                 mode='lines',
    #                 name=f'Trendline OLS until 2024',
    #                 line=dict(color="green")
    #             ))
    
    # fig.add_trace(go.Scatter(
    #     x=df_diff["jaar"],
    #     y=df_diff["y_fit_upper"],
    #     mode='lines',
    #     line=dict(width=0),
    #     showlegend=False
    # ))
    # fig.add_trace(go.Scatter(
    #     x=df_diff["jaar"],
    #     y=df_diff["y_fit_lower"],
    #     mode='lines',
    #     line=dict(width=0),
    #     fillcolor='rgba(255, 255, 0, 0.1)',
    #     fill='tonexty',
    #     name=f'95% CI ({secondary_choice.capitalize()})'
    # ))

    #add the fitted curve
    df_filtered = df_diff[:-4]  # Slices the DataFrame to exclude the last 4 rows


    title=f"{age_group} - {sexe} | {value_field} | {doordsoorzaak_keuze}"
    # try:           
    #     r2 = round(r2_score(df_filtered[value_field], trendline),4)
    #     print (r2)
    #     title += f"| r2 OLS: {r2} "
    # except:
    #     # pass 
    #     title += f"| r2 OLS: n/a "
    colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    for i,secondary_choice in enumerate(secondary_choice_):
        try:
            r2_b = round(r2_score(df_filtered[value_field], df_filtered[f"fitted_curve_{secondary_choice}"]),4)
            title += f"<br>r2 {secondary_choice}: {r2_b}"
            fig.add_trace(go.Scatter(x=df_diff["jaar"], y=df_diff[f"fitted_curve_{secondary_choice}"], mode='lines', line=dict(color=colors[i]), name=f'Fitted {secondary_choice} Curve'))
        except:
            pass 
            #st.write(f"{secondary_choice}: n/a ")
    # except:
    #     r2_a,r2_b=None,None

    fig.update_layout(
                title=title,
                xaxis_title="Year",
                yaxis_title=value_field,
            )
    st.plotly_chart(fig)

def plot_fitting_on_value_field_scaled(value_field: str, df_before_2020: pd.DataFrame, df_2020_and_up: pd.DataFrame, trendline: np.ndarray, extended_years: np.ndarray, trendline_extended: np.ndarray, df_diff: pd.DataFrame, age_group: str, sexe: str, secondary_choice_: list[str], doordsoorzaak_keuze: str) -> None:
    """
    Plot the fitting results, including data before and after 2020, trendlines, and secondary fits.

    Args:
        value_field (str): The field used for plotting (e.g., 'OBS_VALUE', 'per100k').
        df_before_2020 (pd.DataFrame): DataFrame containing data before 2020.
        df_2020_and_up (pd.DataFrame): DataFrame containing data from 2020 and onwards.
        trendline (np.ndarray): Linear trendline fitted on data before 2020.
        extended_years (np.ndarray): Array of extended years for future projections.
        trendline_extended (np.ndarray): Linear trendline extended to future years.
        df_diff (pd.DataFrame): DataFrame with all data points and predictions.
        age_group (str): Age group.
        sexe (str): Gender category ('T', 'M', 'V').
        secondary_choice_ (list[str]): Types of fitting for the secondary choices [exponential|quadratic].
        doordsoorzaak_keuze (str): Cause of death choice.
        
    Returns:
        None
    """
    
    
    

    # Add additional trendlines for secondary fits
    colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    value_curve = df_diff[value_field]
    for i, secondary_choice in enumerate(secondary_choice_):
        fig = go.Figure()
        try:
            # Calculate the fitted curve and scale it
            
            fitted_curve = df_diff[f"fitted_curve_{secondary_choice}"]
           
            values = ((value_curve-fitted_curve) / fitted_curve) 
            
            years = df_diff["jaar"]
             # Filter for positive and negative values
            positive_years = [years[i] for i in range(len(values)) if values[i] >= 0]
            positive_values = [value for value in values if value >= 0]
            
            negative_years = [years[i] for i in range(len(values)) if values[i] < 0]
            negative_values = [value for value in values if value < 0]

            # Plot positive bars
            fig.add_trace(go.Bar(
                x=positive_years,
                y=positive_values,
                name='Positive Values',
                marker=dict(color='blue'),
                showlegend=True
            ))

            # Plot negative bars (inverted for visual effect)
            fig.add_trace(go.Bar(
                x=negative_years,
                y=[value for value in negative_values],  # Negate the values for downward bars
                name='Negative Values',
                marker=dict(color='red'),
                showlegend=True
            ))
        except KeyError:
            pass  # Skip if fitted curve does not exist

        # Set the plot title and labels
        title = f"{age_group} - {sexe} | {value_field} | {doordsoorzaak_keuze} | {secondary_choice}"
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Relative Value (Base = Trendline)",
            yaxis_tickformat='.0%',  # Format y-axis as percentages
        )

        # Show the plot
        st.plotly_chart(fig)



def plot_transformed_to_absolute(df_before_2020: pd.DataFrame, df_2020_and_up: pd.DataFrame, df_diff: pd.DataFrame, age_group: str, sexe: str, secondary_choice_:list[str], doordsoorzaak_keuze:str) -> None:
    """
    Plot the observed deaths and predicted deaths from both trendline and secondary curve.

    Args:
        df_before_2020 (pd.DataFrame): DataFrame containing data before 2020.
        df_2020_and_up (pd.DataFrame): DataFrame containing data from 2020 onwards.
        df_diff (pd.DataFrame): DataFrame with calculated fields, including predicted deaths and fitted curves.
        age_group (str): Age group 
        sexe (str): Gender category ('T', 'M', 'V').
        secondary_choice (str): type of fitting for the 2nd choice [exponential|quadratic]
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
            mode='lines',
            marker=dict(color='green')  
        ))
       
    # Exclude the last four values
    df_filtered = df_diff[:-4]  # Slices the DataFrame to exclude the last 4 rows
    title=f"{age_group} - {sexe} | {doordsoorzaak_keuze} |Deaths Transformed from relatieve back to absolute numbers " 
    r2 = round(r2_score(df_filtered["OBS_VALUE"], df_filtered["predicted_deaths"]),4)
    title += f"| r2 OLS: {r2} "
    colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

    for i,secondary_choice in enumerate(secondary_choice_):
        try:
            df_diff[f"fitted_aantal_{secondary_choice}"] = df_diff[f"fitted_curve_{secondary_choice}"] * df_diff["aantal"]/100000
            fig.add_trace(go.Scatter(x=df_diff["jaar"], y=df_diff[f"fitted_aantal_{secondary_choice}"], mode='lines', line=dict(color=colors[i]), name=f'Fitted {secondary_choice} Curve'))

            r2_b = round(r2_score(df_diff["OBS_VALUE"], df_diff[f"fitted_aantal_{secondary_choice}"]),4)
            title += f"| r2 {secondary_choice}: {r2_b} "
        except Exception as error:
            print(f"{secondary_choice} : graphline failed|Deaths Transformed from relatieve back to absolute numbers {error} ")
    

    # Calculate R² score
    
    fig.update_layout(
                title= title,
                xaxis_title="Year",
                yaxis_title="Deaths",
            )
    st.plotly_chart(fig)

def plot_group_size(df_diff: pd.DataFrame, age_group: str, sexe: str, doordsoorzaak_keuze:str) -> None:
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
                title=f"{age_group} - {sexe} |Number of people in the population",
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

#@st.cache_data()
def calculate_results(df: pd.DataFrame, age_groups_selected_: list[str], start_years: list[int], sexe: str, verbose: bool, secondary_choice_: list[str], show_confidence_intervals: bool,doordsoorzaak_keuze:str, what_to_plot:list[str], scaled) -> pd.DataFrame:
#def calculate_results(df: pd.DataFrame, age_groups_selected: list[str], start_years: list[int], sexe: str, verbose: bool, secondary_choice:str) -> pd.DataFrame: 
    """
    Calculate excess mortality using both linear and secondary models for each age group, 
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
    secondary_choice: str, 
    show_confidence_intervals:
    doordsoorzaak_keuze
    what_to_plot
    Returns:
    --------
    df_results : pd.DataFrame
        A dataframe containing the results of excess mortality calculations for both 
        linear and secondary models. Each row includes the start year, model type 
        (linear or secondary), value field (e.g., 'OBS_VALUE', 'per100k'), age group, 
        and calculated excess mortality.

    Notes:
    ------
    The value fields 'OBS_VALUE' and 'per100k' are calculated for each model, age group and start year.
    """
    # Define the start years for subcolumns
    counter = 0
    total = 2* len(age_groups_selected_)*len(start_years)

    # Initialize DataFrames to store the results
    results = []
    
    
    for value_field in ["OBS_VALUE", "per100k"]:
        #for age_group in age_groups_selected_:
        if type(age_groups_selected_) == list:
            pass
        else:
            age_groups_selected_ = [age_groups_selected_]
        
        for age_group in age_groups_selected_:
            
            for START_YEAR in start_years:
                print(f"{counter+1}/{total} | {value_field=} - {age_group=} { START_YEAR=}")
                excess_mortality_lineair, excess_mortality_secondary_ = main_(df, value_field, age_group, sexe, START_YEAR, verbose, secondary_choice_, show_confidence_intervals,doordsoorzaak_keuze, what_to_plot, scaled)
                #excess_mortality_lineair, excess_mortality_secondary = main_(df, value_field, age_group, sexe, START_YEAR, verbose, secondary_choice)
                
                # Append results for lineair model
                results.append({
                    "start_year": START_YEAR,
                    "model": "lineair",
                    "value_field": value_field,
                    "age_group": age_group,
                    "excess_mortality": excess_mortality_lineair
                })
                for secondary_choice, excess_mortality_secondary in zip(secondary_choice_,excess_mortality_secondary_):
                    # Append results for secondary model
                    results.append({
                        "start_year": START_YEAR,
                        "model": secondary_choice,
                        "value_field": value_field,
                        "age_group": age_group,
                        "excess_mortality": excess_mortality_secondary
                    })
                counter +=1

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

        
   
    return df_results
    
def main() -> None:
    """
    Main function for the Streamlit application that analyzes mortality data using linear and 
    secondary fitting models.

    Args:
        None

    Returns:
        None
    """
    st.subheader("Mortality Analysis Using secondary Curve Fitting and Trendline Projection")
    st.info("""
            This Streamlit application analyzes mortality data for specific age 
            and sex groups using historical trends. You can also choose the cause of death.
            The script leverages both linear 
            and secondary curve fitting to predict future mortality rates and 
            over-mortality for the years 2020 and beyond. 
            
            * Fitting the Model on Absolute Figures: We first apply secondary curve 
            fitting to the absolute mortality figures (total number of deaths) for the 
            years before 2020. This helps us understand the overall trend.

            * Fitting the Model on Relative Numbers: Next, we fit the model on relative 
            numbers, such as deaths per 100,000 people. This allows for a normalized 
            comparison across different population sizes.

            * Transformation to Absolute Numbers: After fitting the model on the 
            relative numbers, we transform these results back into absolute figures 
            (total deaths) for easier interpretation and comparison with actual data.
            
            Inspired by https://twitter.com/HansV_16/status/1838178383835848708 """)
    st.info("https://rene-smit.com/low-excess-mortality-observed-using-quadratic-fitting-in-mortality-trend-analysis/")
    # choice = st.sidebar.selectbox("Overlijdens of doodsoorzaken",["overlijdens", "doodsoorzaken"],0)
    #opdeling = [[0,49], [50,64], [65,79], [80,89], [90,120],[80,120], [0,120]]
    opdeling = [[0,120], [0,64],[65,79],[80,120]] + interface_opdeling() 
    df_doodsoorzaken = get_doodsoorzaken(opdeling)
    
    doodsoorzaken   = ["ALLE DOODSOORZAKEN"] + df_doodsoorzaken["doodsoorzaak"].unique().tolist()  
    #st.write(doodsoorzaken)
    preset= st.sidebar.selectbox("Preset", ["None","accuut_hartinfarct"])
    
    if preset == "None":
        doodsoorz_def = 0
        sec_choice_def = ["linear", "quadratic"]
        age_groups_def = 1
        what_to_plot_def = ["OBS_VALUE","per100k","number_of_people","100k_to_population"]
    else:
        doodsoorz_def = 83
        sec_choice_def = ["linear"]
        age_groups_def = 0 
        what_to_plot_def = ["per100k"]
    doodsoorzaak_keuze =  st.sidebar.selectbox("Doodsoorzaak",doodsoorzaken,doodsoorz_def)
    if doodsoorzaak_keuze =="ALLE DOODSOORZAKEN":
        df = get_data(opdeling) 
        doodsoorzaak_keuze=""
    else:  
        df =  df_doodsoorzaken[df_doodsoorzaken["doodsoorzaak"] == doodsoorzaak_keuze]
   
    age_groups_ = df["age_group"].unique().tolist() #[:2]
    #Convert opdeling ranges to strings matching the "Yxx-yy" format
    priority_groups = [f"Y{start:02d}-{end:02d}" for start, end in opdeling]

    # Define a function to extract the starting age from the age group string
    def get_starting_age(age_group):
       
        if age_group == "TOTAL":
            age = 0
        else:
            try:
                # xx-
                age = int(age_group[1:3])
            except:
                # 0-
                age =int(age_group[1:2])
        return age  # Extract xx from Yxx-yy and convert to integer

    
    def custom_sort_key(age_group):
        if age_group in priority_groups:
            return (0, priority_groups.index(age_group))
        else:
            return (1, get_starting_age(age_group))

    # Sort the age groups based on the starting age
    age_groups_sorted = sorted(age_groups_, key=get_starting_age)
    age_groups  = ["ALLE LEEFTIJDEN IN EEN LOOP"] +age_groups_sorted
   
    #age_groups = ["Y70-74"]
    what_to_do = "selection" # st.sidebar.selectbox("What to do - startyears [selection|all]", ["selection", "all"],what_to_do_def)
    sexe = st.sidebar.selectbox("Sexe [T|M|F]", ["T","M","F"],0)
    possible_columns = [
                            ['model', 'value_field', 'start_year'],
                            ['model', 'start_year', 'value_field'],
                            ['value_field', 'model', 'start_year'],
                            ['value_field', 'start_year', 'model'],
                            ['start_year', 'model', 'value_field'],
                            ['start_year', 'value_field', 'model']
                        ]
    columns = st.sidebar.selectbox("Column hierarchie", possible_columns,0)
    if what_to_do == "selection":
        age_groups_selected = [st.sidebar.selectbox("age group", age_groups,age_groups_def)]
        start_years = [st.sidebar.number_input("Fitting from year",1950,2019,2010)]
        verbose=True
        secondary_choice_ = st.sidebar.multiselect(
            "Secondary choice",
            [ "linear","exponential", "quadratic", "gompertz", "first_derivative_gompertz", "gaussian", "exponential_2", "logistic"],
            sec_choice_def
            
        )
        if len(secondary_choice_)==0:
            st.warning("Choose at least one secondary choice")
            st.stop()
    else:
        start_years = [2000, 2010, 2015]
        verbose = False
        age_groups_selected = age_groups_sorted
        secondary_choice_ = [st.sidebar.selectbox("Secondary choice [exponential|quadratic]", ["exponential","quadratic"],1)]
    # Add this line to create the new selectbox
    what_to_plot =  st.sidebar.multiselect("What to plot",["OBS_VALUE","per100k","number_of_people","100k_to_population"],what_to_plot_def)
    show_confidence_intervals = st.sidebar.checkbox("Show confidence intervals", value=False)
    
    if ("OBS_VALUE" in what_to_plot) or ("per100k" in what_to_plot):
        scaled = st.sidebar.checkbox("Show relative values", [False, True],0)    
    else:
        scaled = False
    #df_results = calculate_results(df,age_groups_selected, start_years, sexe, verbose, secondary_choice)
    if age_groups_selected[0] == "ALLE LEEFTIJDEN IN EEN LOOP":
        
        for age_groups_selected_x in age_groups_:
            st.subheader(age_groups_selected_x)
            df_results = calculate_results(df, age_groups_selected_x, start_years, sexe, verbose, secondary_choice_, show_confidence_intervals,doodsoorzaak_keuze,what_to_plot, scaled)
   
    else:
        
        for age_groups_selected_x in age_groups_selected:
            st.subheader(age_groups_selected_x)
            age_groups_selected_y = [age_groups_selected_x]
            
            df_results = calculate_results(df, age_groups_selected_y, start_years, sexe, verbose, secondary_choice_, show_confidence_intervals,doodsoorzaak_keuze,what_to_plot, scaled)
            df_results.to_csv(f"data_{age_groups_selected_x}.csv")
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
    st.info("Doodsoorzaken: https://www.cbs.nl/nl-nl/cijfers/detail/7052_95")
if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')

    print(f"--------------{datetime.datetime.now()}-------------------------")

    
    main()
    #get_sterftedata()