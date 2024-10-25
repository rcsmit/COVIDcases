import streamlit as st
from typing import List, Tuple
import pandas as pd
import platform

import datetime as dt
import pandas as pd
import streamlit as st
import numpy as np

import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from scipy import stats
# for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from oversterfte_eurostats_maand import get_data_eurostat
from statsmodels.tsa.seasonal import STL
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error



def get_cbs_baseline():
    cbs_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/basevalues_sterfte_Y0-120_T.csv"
    df_cbs = pd.read_csv(
        cbs_url,
        delimiter=",",
        low_memory=False,
    )
    return df_cbs

@st.cache_data()
def get_sterfte(min,max, country: str = "NL") -> pd.DataFrame:
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

    
    df_ = get_data_eurostat()
    df_=df_[df_["geo"] == country]
    df_["jaar"] = (df_["TIME_PERIOD"].str[:4]).astype(int)
    df_["week"] = (df_["TIME_PERIOD"].str[6:]).astype(int)
    df_ = df_[(df_["jaar"] >= min) & (df_["jaar"] <= max)].copy(
        deep=True
    )
   
    df_ = df_[(df_["sex"] == "T") & (df_["age"] == "TOTAL")]

    #path = r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\bevolking_leeftijd_NL.csv"
    path = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
    df_bevolking = pd.read_csv(path, delimiter=';')
    
    # Step 2: Filter for `geslacht == 'T'`
    df_bevolking = df_bevolking[df_bevolking['geslacht'] == 'T']

    # Step 3: Group by `jaar` and sum the `aantal`
    df_bevolking_grouped = df_bevolking.groupby('jaar', as_index=False)['aantal'].sum()

    # Step 4: Merge with `df_`, keeping all records from `df_`
    df_merged = df_.merge(df_bevolking_grouped, on='jaar', how='left')
    df_merged["per100k"] = df_merged["OBS_VALUE"] / df_merged["aantal"] *100_000
    return df_merged

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
        a (float) : constant
        b (float) : sin coef
        c (float) : cos coef
         r2 (float) :r2, for graph title
         f (float) : for graph title
         f_p (float) : for graph title
         use_cos (bool) : for graph title
           use_sin (bool) : for graph title
           min : min jaar for graph title
           max : max jaar for graph title
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
    """Add a sinus and cosinus term and perform mulitple regression

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
    """_summary_

    Args:
        opdeling (_type_): _description_
        min (_type_): min jaar
        max (_type_): max jaar

    Returns:
        _type_: _description_
    """    
    df_result3 = get_sterfte(min,max)
    
    # df_oversterfte = get_oversterfte(opdeling)
    # df_oversterfte["age_sex"] = df_oversterfte["age_sex"].replace("Y0-120_T", "TOTAL_T")
    # df_result3 = pd.merge(df, df_oversterfte, on=["jaar", "week","age_sex"], how="left")
    age_sex = "TOTAL_T"
    df_result5= df_result3[df_result3["age_sex"] == age_sex]
    df_result5["periodenr"] =df_result5["jaar"].astype(str) + "_" + df_result5["week"].astype(int).astype(str).str.zfill(2)
    
    return age_sex,df_result5

def analyse_cos_sin(df_result5, age_sex, time_period, columns, y_value, seizoen, maand, normalize, use_sin, use_cos, min, max):
    def run_analysis_and_plot(df, age_sex, time_period, y_value, seizoen, maand, normalize, use_sin, use_cos, min_val, max_val):
        a, b, c, r2, f, f_p = perform_analyse(age_sex, df, time_period, y_value, seizoen, maand, normalize, use_sin, use_cos)
        line_plot_2_axis(df, "periodenr", "OBS_VALUE", age_sex, a, b, c, r2, f, f_p, use_cos, use_sin, min_val, max_val)

    #options = [(True, True), (True, False), (False, True)]
    options = [(True, True)]

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
  

def plot_stl_four_seperate(df):
    for what in ['OBS_VALUE', 'trend', 'seasonal', 'residual']:
        # Create Plotly visualization
        fig = go.Figure()
        mode='markers' if what=='residual' else 'lines'
        # Add original data
        fig.add_trace(go.Scatter(x=df.index, y=df[what], mode=mode, name='Original Data'))

        # Update layout
        fig.update_layout(title=f'STL Decomposition of Deaths Data - {what}',
                        xaxis_title='Date',
                        yaxis_title=what,
                        legend_title='Components')

        # Show the plot
        st.plotly_chart(fig)

    
def plot_stl_all_in_one(df,y_value):
    fig = go.Figure()
    # Add original data
    df['residual_CBS'] =  df['OBS_VALUE_'] - df['basevalue'] 
    
    if y_value=="OBS_VALUE":
        fig.add_trace(go.Scatter(x=df["periodenr"], y=df['OBS_VALUE'], mode='lines', name='Original Data Eurostats'))
        fig.add_trace(go.Scatter(x=df["periodenr"], y=df['OBS_VALUE_'], mode='lines', name='Original Data CBS'))
        fig.add_trace(go.Scatter(x=df["periodenr"], y=df['residual_CBS'], mode='lines', name='Residual CBS (oversterfte)'))
        fig.add_trace(go.Scatter(x=df["periodenr"], y=df['basevalue'], mode='lines', name='basevalue CBS'))

    else:
        fig.add_trace(go.Scatter(x=df["periodenr"], y=df['per100k'], mode='lines', name='Original Data Eurostats'))
    
    
    # Add trend component
    fig.add_trace(go.Scatter(x=df["periodenr"], y=df['trend'], mode='lines', name='Trend'))

    # Add seasonal component
    fig.add_trace(go.Scatter(x=df["periodenr"], y=df['seasonal'], mode='lines', name='Seasonal'))
    #df['basevalue']=df['basevalue'].rolling(window=7).mean()
    
    # Add residual component
    fig.add_trace(go.Scatter(x=df["periodenr"], y=df['residual'], mode='lines', name='Residual STL'))
    
    fig.add_trace(go.Scatter(x=df["periodenr"], y=df['trend_seasonal_residual'], mode='lines', name='t+s+r'))
    fig.add_trace(go.Scatter(x=df["periodenr"], y=df['trend_seasonal'], mode='lines', name='t+s'))

    fig.add_trace(go.Scatter(x=df["periodenr"], y=df['forecast'], mode='lines', name='Forecast ARIMA'))
 
    # Update layout
    fig.update_layout(title='All STL lines + data/baseline CBS',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend_title='Components')

    # Show the plot
    st.plotly_chart(fig)


def stl(df):
    st.subheader("STL")
    # Sample dataframe
    y_value = st.selectbox("Yvalue",["OBS_VALUE", "per100k"],0)
    # Assuming your dataframe is named 'df' and has columns 'jaar', 'week', and 'OBS_VALUE'
    # Here, we'll create a 'date' column by combining 'jaar' and 'week'
    df['jaar_week'] = df['jaar'].astype(str) + '-W' + df['week'].astype(str).str.zfill(2)
    df['date'] = pd.to_datetime(df['jaar_week'] + '-1', format='%Y-W%W-%w')

    # Sort by date and set as index for time series analysis
    df = df.sort_values('date').set_index('date')
   
    # Apply STL decomposition
    # seasonal_deg=0, trend_deg=0, low_pass_deg=0: Use piecewise constant smoothing for all components 
    # (no polynomial fitting). Robust=True: Downweight outliers to prevent them from distorting trend and seasonality.

    # https://chatgpt.com/c/671acc54-c38c-8004-96d3-4905d94495aa
    stl_51 = STL(df[y_value], seasonal=51, period=51,  seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True)  # You can tweak the seasonal period as needed
    stl_53 = STL(df[y_value], seasonal=53, period=53,  seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True)  # You can tweak the seasonal period as needed
    
    #result_51 = stl_51.fit()
    result = stl_53.fit()

    # Extract the components

    df['trend'] = result.trend 
    df['seasonal'] = result.seasonal 
    df['residual'] = result.resid 
    # df['trend'] = (result_51.trend + result_53.trend)/2
    # df['seasonal'] = (result_51.seasonal + result_53.seasonal)/2
    # df['residual'] = (result_51.resid + result_53.resid) / 2
    df['trend_seasonal'] = df['trend'] + df["seasonal"]
    df['trend_seasonal_residual'] = df['trend'] + df["seasonal"] + df["residual"]
    
    # plot_stl_four_seperate(df)
    

    # show_excess_mortality(df)

    df_merged = merge_with_cbs(df)
    df_prediction = predict(df_merged, y_value)
     
    plot_stl_all_in_one(df_prediction, y_value)

    
def predict(df,y_value='OBS_VALUE'):
        # Assuming df contains 'jaar', 'week', and 'OBS_VALUE' columns
    df_before_2020 = df[df['jaar'] < 2020].copy()
    df_after_2020 = df[df['jaar'] >= 2020].copy()


  
    # Set 'date' as the index for time series analysis
    df_before_2020['jaar_week'] = df_before_2020['jaar'].astype(str) + '-W' + df_before_2020['week'].astype(str).str.zfill(2)
    df_before_2020['date'] = pd.to_datetime(df_before_2020['jaar_week'] + '-1', format='%Y-W%W-%w')
    df_before_2020 = df_before_2020.sort_values('date').set_index('date')

    # STL decomposition
    stl = STL(df_before_2020[y_value], seasonal=51, period=52, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True)
    result = stl.fit()
    df_before_2020['seasonal'] = result.seasonal
    df_before_2020['trend'] = result.trend
    df_before_2020['residual'] = result.resid

    
    # Forecast the trend component
    trend_model = ARIMA(df_before_2020['trend'].dropna(), order=(1, 1, 1))
    trend_fit = trend_model.fit()
    trend_forecast = trend_fit.forecast(steps=len(df_after_2020))

    # Forecast the residual component
    residual_model = ARIMA(df_before_2020['residual'].dropna(), order=(1, 1, 1))
    residual_fit = residual_model.fit()
    residual_forecast = residual_fit.forecast(steps=len(df_after_2020))


    # Repeat seasonal pattern for the forecast period
    seasonal_cycle = np.tile(df_before_2020['seasonal'][-52:], int(np.ceil(len(df_after_2020) / 52)))[:len(df_after_2020)]

    # Combine forecast components to get the final forecast
    forecast = trend_forecast + seasonal_cycle + residual_forecast
    df_after_2020['forecast'] = forecast.values

    
    # Calculate Mean Squared Error
    mse = mean_squared_error(df_after_2020['OBS_VALUE'], df_after_2020['forecast'])
    st.write(f'Mean Squared Error: {mse}')

    # Visualize the results


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_before_2020["periodenr"], y=df_before_2020[y_value], mode='lines', name='Training Data'))
    fig.add_trace(go.Scatter(x=df_after_2020["periodenr"], y=df_after_2020[y_value], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df_after_2020["periodenr"], y=df_after_2020['forecast'], mode='lines', name='Forecast'))
    fig.update_layout(title='Forecast vs Actual', xaxis_title='Date', yaxis_title=y_value)
    st.plotly_chart(fig)
    df_combined = pd.concat([df_before_2020, df_after_2020], ignore_index=True)
    return df_combined
def merge_with_cbs(df):
    df_cbs = get_cbs_baseline()
    df['jaar_week'] = df['jaar'].astype(str) +  df['week'].astype(str).str.zfill(2)
    
    df_merged = pd.merge(df,df_cbs, left_on="periodenr", right_on="jaar_week", how="outer")
    return df_merged

def show_excess_mortality(df):
    for y in [2020, 2021, 2022, 2023, 2024]:
        df_y = df[df["jaar"] == y]
        overst= df_y["residual"].sum()
        # Display the residuals for the specified years
        st.write(f"{y} - oversterfte {int(overst)}")
def main():
    st.subheader("Multiple Lineair Regression and STL")
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
If below 0.05, the results are statistically significant, meaning the seasonal 
effects are unlikely to be due to chance.
This approach helps us understand how seasonal changes impact weekly death rates.

**Seasonal-Trend decomposition using LOESS**        
Seasonal-Trend decomposition using LOESS (STL) separates time series data
into seasonal, trend, and residual components. The seasonal part
captures recurring cycles (e.g., weekly or yearly), the trend shows the
overall direction, and the residual accounts for random fluctuations. STL
uses LOESS (Locally Estimated Scatterplot Smoothing), a flexible approach
for handling complex seasonal patterns by applying localized smoothing.

In this analysis, STL was applied to weekly mortality data, splitting it
into pre-2020 data for training and 2020 onward for validation. After
decomposition, I modeled the trend and residuals to forecast mortality,
recombining with the seasonal component for a full forecast. This allowed
for evaluating how well the model aligned with observed changes after 2020.

Also the CBS baseline and excess mortality is plotted in the graph           

            """)
    opdeling = [[0,120],[15,17],[18,24], [25,49],[50,59],[60,69],[70,79],[80,120]]
   
    
    results = []
    y_value = "OBS_VALUE" #st.selectbox("Y value",  ["OBS_VALUE", "oversterfte", "p_score"],0,help = "Alleen bij leeftijdscategorieen" )
    normalize = False # st.checkbox("Normaliseer X values", True, help="Normalizeren omdat de vaccindosissen een hoog getal kunnen zijn")
    seizoen = True# st.checkbox("Seizoensinvloeden meenemen")
    maand = False #st.checkbox("Maand-/week invloeden meenemene")
    use_sin,use_cos=True,True
    try:
        col1,col2 = st.columns([5,1], vertical_alignment="center")
    except:
        col1,col2 = st.columns([5,1])
    with col1:
        (min,max) = st.slider("years", 2000,2024,(2000, 2024))
    with col2:
        columns =  st.checkbox("columns")
    
   
    time_period = "YearWeekISO"
    
    age_sex, df_result5 = get_and_prepare_data(opdeling, min, max)
    analyse_cos_sin(df_result5, age_sex, time_period, columns, y_value, seizoen, maand, normalize, use_sin, use_cos, min, max)
    
    stl(df_result5)
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
    print(f"-------xx--sterfte_2000_2024.py-----{datetime.datetime.now()}-------------------------")
    main()
    #expand()