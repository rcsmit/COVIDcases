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
from oversterfte_compleet import  get_sterftedata, get_data_for_series_wrapper,make_df_quantile, layout_annotations_fig
# for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from oversterfte_eurostats_maand import get_data_eurostat
# WAAROM TWEE KEER add_custom_age_group_deaths ??? TODO

@st.cache_data
def get_oversterfte(opdeling):
    # if platform.processor() != "":
    #     file = f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\basevalues_sterfte.csv"
    #     file = f"C:\\Users\\rcxsm\\Documents\\python_scripts\\covid19_seir_models\\COVIDcases\\input\\basevalues_sterfte_Y0-120_T.csv"
    
    # else:
    #     #file = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/basevalues_sterfte.csv"
    #     file = f"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/basevalues_sterfte_Y0-120_T.csv"
    # # Load the CSV file
    # df_ = pd.read_csv(file)
    df__ = get_sterftedata(2000, "m_v_0_999")

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
def get_sterfte(opdeling: List[Tuple[int, int]],min:int,max:int, country: str = "NL") -> pd.DataFrame:
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
    df_ = df_[(df_["jaar"] >= min) & (df_["jaar"] < max)].copy(
        deep=True
    )
   
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

def multiple_linear_regression(df: pd.DataFrame, x_values: List[str], y_value_: str, age_sex: str, normalize:bool, use_sin:bool, use_cos:bool):
    """
    Perform multiple linear regression and display results.

    Args:
        df (pd.DataFrame): Input dataframe.
        x_values (List[str]): List of independent variable column names.
        y_value (str): Dependent variable column name.
    """
    
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
    x_ = np.column_stack((df["sin_time"], df["cos_time"]))  # Shap
    if normalize:

        # Normalize each feature in x to the range [0, 1]
        x_normalized = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

        # If intercept is required, add a constant term
        if intercept:
            x_normalized = sm.add_constant(x_normalized)  # adding a constant
    
        # Fit the OLS model using the normalized data
        model = sm.OLS(y.astype(float), x_normalized.astype(float)).fit()
    else:
        if intercept:
            x= sm.add_constant(x) # adding a constant

        model = sm.OLS(y.astype(float), x.astype(float)).fit()
    predictions = None
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    # with col2:
    robust_model = model.get_robustcov_results(cov_type='HC0')  # You can use 'HC0', 'HC1', 'HC2', or 'HC3'
    with st.expander("OUTPUT"):
        st.write("**robust model**")
        st.write(robust_model.summary())
        col1,col2,col3=st.columns([2,1,1])     
        with col1:
            st.write("**Correlation matrix**")
            correlation_matrix = x.corr()
            st.write(correlation_matrix)
        with col2:
            try:
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
            except:
                st.info("No VIF")
        with col3:
            try:
                U, s, Vt = np.linalg.svd(x)
                st.write("**spread of singular values**")
                st.write(s)  # Look at the spread of singular values
            except:
                st.info("No spread")
    if 1==2:
        data = {
            'Y value':y_value_,
            # 'Coefficients': model.params,
            # 'P-values': model.pvalues,
            # 'T-values': model.tvalues,
            # 'Residuals': model.resid,
            'P_const': model.pvalues["const"],
            'P_sin':model.pvalues["sin_time"],
            'P_cos':model.pvalues["cos_time"],
            'R-squared': [model.rsquared], #* len(model.params),
            'Adjusted R-squared': [model.rsquared_adj], # * len(model.params),
            'F-statistic': [model.fvalue], # * len(model.params),
            'F-statistic P-value': [model.f_pvalue], # * len(model.params)
        }
            

        # Stap 4: Omzetten naar een DataFrame
        xx=data["Y value"]
        y_value_x =  f"{xx}_{age_sex}"
        df = pd.DataFrame({
            "Y value":y_value_x,
        
            #'P_const': data['P_const'],
            'P_sin':data['P_cos'],
            'P_cos':data['P_sin'],

            #"R-squared": data["R-squared"],
            "Adjusted R-squared": data["Adjusted R-squared"],
            #"F-statistic": data["F-statistic"],
            "F-statistic P-value": data["F-statistic P-value"]
        })
    a = model.params["const"],
    if use_sin:
        b = model.params["sin_time"]
    else:
        b=0
    if use_cos:
        c = model.params["cos_time"]
    else:
        c=0

    r2 = model.rsquared_adj
    f = model.fvalue
    f_p = model.f_pvalue
    return a,b,c, r2,f,f_p

def line_plot_2_axis(df: pd.DataFrame, x: str, y1: str,  age_sex: str,a:float,b:float,c:float, r2,f,f_p, use_cos, use_sin, min,max):
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

    # Coefficients from the OLS results
    const = a
    sin_coef = b
    cos_coef = c
    df["week_in_radians"] = (df["week"] / 52) * 2 * np.pi
    df["regression_line"] = const + sin_coef * np.sin(df["week_in_radians"]) + cos_coef * np.cos(df["week_in_radians"])

    # Create a figure
    fig = go.Figure()


    # Add scatter plot of the actual data
    #fig.add_trace(go.Scatter(x=df[x], y=predictions, mode='lines', name='Prediction'))

    # Add OBS_VALUE as the first line on the left y-axis
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y1],
            mode='lines',
            name=y1,
            line=dict(color='red')
        )
    )
    fig.add_trace(go.Scatter(x=df[x], y=df["regression_line"], mode='lines', name='Prediction', line=dict(color='blue')))
    
    title = f'Aantal overledenen per week {min}-{max}<br>'
    title += f"R2 = {round(r2,3)} | F = {round(f,1)} | F (p_value) = {round(f_p,3)}<br>"
    if use_cos:
        title += " | cos incl."
    if use_sin:
        title += " | sin incl."

    # Update layout to include two y-axes
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y1,
       
        legend=dict(x=0.5, y=1, orientation='h')
    )
    

    st.plotly_chart(fig)




def perform_analyse(age_sex, df, time_period,y, seizoen, maand, normalize, use_sin, use_cos):
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
        _type_: _description_
    """  
    
  
    # Voeg een sinus- en cosinusfunctie toe om seizoensinvloeden te modelleren
    
    df = df[df["week"] !=53]
    df['sin_time'] = np.sin(2 * np.pi * df['week']/ 52)
    df['cos_time'] = np.cos(2 * np.pi * df['week'] / 52)
    m=False
        
    x_values = [] # + 
    if seizoen:
        if use_cos:
            x_values += ['cos_time']
        if use_sin:
            x_values += ['sin_time']
    if maand:
        if m:
            x_values += ['maand']
        else:
            x_values += ['week']
    y_value_ = y
    
    a,b,c, r2,f,f_p = multiple_linear_regression(df,x_values,y_value_, age_sex, normalize,use_sin, use_cos)
    return a,b,c, r2,f,f_p

def get_and_prepare_data(opdeling, min, max):
    df = get_sterfte(opdeling,min,max)
    df_oversterfte = get_oversterfte(opdeling)
    df_oversterfte["age_sex"] = df_oversterfte["age_sex"].replace("Y0-120_T", "TOTAL_T")
    df_result3 = pd.merge(df, df_oversterfte, on=["jaar", "week","age_sex"], how="left")
    age_sex = "TOTAL_T"
    df_result5= df_result3[df_result3["age_sex"] == age_sex]
    df_result5["periodenr"] =df_result5["jaar"].astype(str) + "_" + df_result5["week"].astype(int).astype(str).str.zfill(2)
    return age_sex,df_result5
   

def main():
    st.subheader("Multiple Lineair Regression")
    st.info("""
This analysis examines seasonality in the number of deaths in the Netherlands, 
where deaths fluctuate based on the time of year. Using regression models, 
we analyze these patterns by applying cosine and sine functions to capture seasonal cycles.

Key results include:

**R²**: Measures how well the model explains the data. 
A value above 0.7 suggests the model explains a strong portion of the variation in deaths.

**F-statistic**: Indicates the overall effectiveness of the model. 
A high F-statistic (generally above 10) indicates the model is meaningful.

**F p-value**: Tests if the relationships are statistically significant. 
If below 0.05, the results are statistically significant, meaning the seasonal effects are unlikely to be due to chance.
This approach helps us understand how seasonal changes impact weekly death rates.
            """)
    opdeling = [[0,120],[15,17],[18,24], [25,49],[50,59],[60,69],[70,79],[80,120]]
   
    
    results = []
    y_value = "OBS_VALUE" #st.selectbox("Y value",  ["OBS_VALUE", "oversterfte", "p_score"],0,help = "Alleen bij leeftijdscategorieen" )
    normalize = False # st.checkbox("Normaliseer X values", True, help="Normalizeren omdat de vaccindosissen een hoog getal kunnen zijn")
    seizoen = True# st.checkbox("Seizoensinvloeden meenemen")
    maand = False #st.checkbox("Maand-/week invloeden meenemene")
    col1,col2 = st.columns([5,1])
    with col1:
        (min,max) = st.slider("years", 2000,2024,(2000, 2024))
    with col2:
        columns =  st.checkbox("columns")
    
    #     use_cos = st.checkbox("use cos", True)
    # with col3:
    #     use_sin = st.checkbox("use sin", True)
    # if (use_cos==False) and (use_sin==False):
    #     st.error("Use at least one of the two")
    #     st.stop()
    time_period = "YearWeekISO"
    
    age_sex, df_result5 = get_and_prepare_data(opdeling, min, max)

    def run_analysis_and_plot(df, age_sex, time_period, y_value, seizoen, maand, normalize, use_sin, use_cos, min_val, max_val):
        a, b, c, r2, f, f_p = perform_analyse(age_sex, df, time_period, y_value, seizoen, maand, normalize, use_sin, use_cos)
        line_plot_2_axis(df, "periodenr", "OBS_VALUE", age_sex, a, b, c, r2, f, f_p, use_cos, use_sin, min_val, max_val)

    options = [(True, True), (True, False), (False, True)]
    # If columns are available
    if columns:
        col1, col2, col3 = st.columns(3)
        
        for col, (use_sin, use_cos) in zip([col1, col2, col3], options):
            with col:
                run_analysis_and_plot(df_result5, age_sex, time_period, y_value, seizoen, maand, normalize, use_sin, use_cos, min, max)

    # If no columns are available
    else:
        for use_sin, use_cos in options:
            run_analysis_and_plot(df_result5, age_sex, time_period, y_value, seizoen, maand, normalize, use_sin, use_cos, min, max)
  
    
    st.subheader("Data sources")
    st.info("https://ec.europa.eu/eurostat/databrowser/product/view/demo_r_mwk_05?lang=en")
    st.info("""
Het combineren van zowel sin als cos in een regressiemodel levert vaak het 
beste resultaat omdat beide functies samen de volledige seizoenscyclus 
kunnen beschrijven. Ze vullen elkaar aan door elk een deel van de golfbeweging op te vangen:

Cosinus is hoog in januari (bij 1) en laag in juli (bij 7), wat een typische 
winter-zomer trend beschrijft. Dit past goed bij het patroon van hogere 
sterftecijfers in de wintermaanden, vooral door bijvoorbeeld griep of 
koude weersomstandigheden.

Sinus werkt iets anders: deze is nul bij januari en juli en heeft pieken 
in de lente en herfst (bij ongeveer week 13 en 39). Dit helpt het model 
de kleinere schommelingen in sterfte te beschrijven die mogelijk niet volledig 
door alleen cosinus worden opgevangen.

Samen zorgen ze ervoor dat het model zowel de amplitude (hoe groot de schommelingen zijn)
als de faseverschuiving (wanneer pieken en dalen plaatsvinden) van de 
seizoensgebonden sterfte nauwkeurig kan beschrijven. Dit geeft het model 
flexibiliteit om het volledige patroon door het jaar heen beter te volgen, 
wat leidt tot een betere R², F-statistic, en p-waarde.

Kortom, door zowel sin als cos te gebruiken, kan het model zowel de symmetrische 
als de asymmetrische variaties van sterfte over de seizoenen heen effectief modelleren. [ChatGPT]
            """)
if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')
    print(f"-------xx-------{datetime.datetime.now()}-------------------------")
    main()
    #expand()