import pandas as pd
import streamlit as st
import numpy as np

import datetime
import plotly.express as px
import statsmodels.api as sm

import statsmodels.formula.api as smf

import plotly.graph_objs as go
import plotly.express as px
from oversterfte_compleet import get_sterftedata
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper

# https://chatgpt.com/c/66e2ce71-2938-8004-b28b-fbd1f01eac49
# https://chatgpt.com/c/66e2beaf-a1f8-8004-ae1d-0b28393d2a48

# https://claude.ai/chat/164c1d6e-faf6-4ab5-835f-0f58f7788537


def adjust_overledenen(df):
    """
    Adjust "Overledenen_1" and "aantal_dgn" based on the week number.
    
    - Specific cases:
      - Week 53 of 2020 is added to week 52 of 2020.
      - Week 0 of 2021 is added to week 1 of 2021.
      
    - General case:
      - If week = 0, add to week 52 of the previous year.
      - If week = 53, add to week 1 of the next year.
    """
    
   
    # Handle specific case: Week 53 of 2020 -> Week 52 of 2020
    if not df[(df["year"] == 2020) & (df["week"] == 53)].empty:
        # Ensure the row for week 52 exists before adding
        if not df[(df["year"] == 2020) & (df["week"] == 52)].empty:
            df.at[df[(df["year"] == 2020) & (df["week"] == 52)].index[0], "Overledenen_1"] += df.loc[
                (df["year"] == 2020) & (df["week"] == 53), "Overledenen_1"].values[0]
            df.at[df[(df["year"] == 2020) & (df["week"] == 52)].index[0], "aantal_dgn"] += df.loc[
                (df["year"] == 2020) & (df["week"] == 53), "aantal_dgn"].values[0]
    else:
        print("empty value for 2020-53")
    # Handle specific case: Week 0 of 2021 -> Week 1 of 2021
    if not df[(df["year"] == 2021) & (df["week"] == 0)].empty:
        # Ensure the row for week 1 exists before adding
        if not df[(df["year"] == 2021) & (df["week"] == 1)].empty:
            df.at[df[(df["year"] == 2021) & (df["week"] == 1)].index[0], "Overledenen_1"] += df.loc[
                (df["year"] == 2021) & (df["week"] == 0), "Overledenen_1"].values[0]
            df.at[df[(df["year"] == 2021) & (df["week"] == 1)].index[0], "aantal_dgn"] += df.loc[
                (df["year"] == 2021) & (df["week"] == 0), "aantal_dgn"].values[0]
    else:
        print("empty value for 2021-0")

    # Remove the rows for week 53 of 2020 and week 0 of 2021
    df = df[~((df["year"] == 2020) & (df["week"] == 53))]
    df = df[~((df["year"] == 2021) & (df["week"] == 0))]

    # Apply general rule for other years
    for index, row in df.iterrows():
        if row["week"] == 0 and not (row["year"] == 2021):
            previous_year = row["year"] - 1
            df.loc[
                (df["year"] == previous_year) & (df["week"] == 52), "Overledenen_1"
            ] += row["Overledenen_1"]
            df.loc[
                (df["year"] == previous_year) & (df["week"] == 52), "aantal_dgn"
            ] += row["aantal_dgn"]

        elif row["week"] == 53 and not (row["year"] == 2020):
            next_year = row["year"] + 1
            df.loc[
                (df["year"] == next_year) & (df["week"] == 1), "Overledenen_1"
            ] += row["Overledenen_1"]
            df.loc[
                (df["year"] == next_year) & (df["week"] == 1), "aantal_dgn"
            ] += row["aantal_dgn"]

    # Remove any other remaining rows with week 0 or 53
    df = df[~df["week"].isin([0, 53])]
    
    return df




def get_sterfte_data_fixed():
    """
    Fetch and preprocess death data from a remote CSV file. 
    Obsolete, the script uses get_data() from oversterfte_compleet.py
    
    Returns:
    --------
    pd.DataFrame
        The processed DataFrame containing death data.
    """

    #C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\overlijdens_per_week.csv
    file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/overlijdens_per_week.csv"

    df_ = pd.read_csv(
        file,
        delimiter=";",
        low_memory=False,
    )   
    df_ = df_.rename(columns={"weeknr": "week"})
    df_ = df_.rename(columns={"jaar": "year"})
    df_ = df_[df_['year'] >2014]
    return df_


def perform_poisson_analysis(df: pd.DataFrame, take_factor_into_account: bool) -> None:
    """
    Fit Poisson models to observed death data, predict expected deaths, and calculate excess deaths.
    This version includes a time trend in the model.

    Args:
        df (pd.DataFrame): DataFrame with columns 'jaar' (year), 'week', and 'observed_deaths'.
        take_factor_into_account (bool): Whether to consider demographic changes in calculations.

    Returns:
        None: Results are displayed using Streamlit's st.write() and st.plotly_chart().
    """
    BASELINE_DEATHS = 149832  # average deaths per year 2015-2019
    DEMOGRAPHIC_FACTORS = {
        2014: 1, 2015: 1, 2016: 1, 2017: 1, 2018: 1, 2019: 1,
        2020: 153402 / BASELINE_DEATHS,
        2021: 154887 / BASELINE_DEATHS,
        2022: 155494 / BASELINE_DEATHS,
        2023: 156666 / BASELINE_DEATHS,
        2024: 157846 / BASELINE_DEATHS,
    }

    df = prepare_data(df)
    model = fit_poisson_model(df[df['jaar'].between(2015, 2019)])

    fig_observed = go.Figure()
    fig_excess = go.Figure()
    expected_deaths_added = False
    st.subheader("Oversterfte:")
    for year in range(2015, 2025):
        data_year = prepare_year_data(df, year)
        #factor = DEMOGRAPHIC_FACTORS[year] if take_factor_into_account else 1
        factor = 1
        data_year['expected_deaths'] = predict_deaths(model, data_year) * factor
        data_year['excess_deaths'] = data_year['observed_deaths'] - data_year['expected_deaths']

        add_observed_trace(fig_observed, data_year, year)
        
        if year >= 2020:
            if take_factor_into_account or not expected_deaths_added:
                add_expected_trace(fig_observed, data_year, year, take_factor_into_account, expected_deaths_added)
                expected_deaths_added = True
            add_excess_trace(fig_excess, data_year, year)
            st.write(f"{year} - {int(data_year['excess_deaths'].sum())}")

    update_and_display_figures(fig_observed, fig_excess)

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the data by adding calculated columns for seasonality, non-linear week effects, and time trend.

    Args:
        df (pd.DataFrame): Input DataFrame with 'week' and 'jaar' columns.

    Returns:
        pd.DataFrame: DataFrame with additional calculated columns.
    """
    df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
    df['week_squared'] = df['week'] ** 2
    df['time'] = (df['jaar'] - df['jaar'].min()) * 52 + df['week']  # Time in weeks since start of data
    return df

def fit_poisson_model(train_data: pd.DataFrame) -> GLMResultsWrapper:
    """
    Fit a Poisson regression model to the training data, including a time trend.

    Args:
        train_data (pd.DataFrame): Training data for the model.

    Returns:
        GLMResultsWrapper: Fitted Poisson model.
    """
    formula = 'observed_deaths ~ week + week_squared + sin_week + cos_week + time'
    return smf.poisson(formula, data=train_data).fit(scale='X2')

def prepare_year_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Prepare data for a specific year, ensuring the time variable is correctly set.

    Args:
        df (pd.DataFrame): Full dataset.
        year (int): Year to prepare data for.

    Returns:
        pd.DataFrame: Prepared data for the specified year.
    """
    data_year = df[df['jaar'] == year].copy()
    min_year = df['jaar'].min()
    data_year['time'] = (year - min_year) * 52 + data_year['week']
    return data_year

def predict_deaths(model: GLMResultsWrapper, data: pd.DataFrame) -> np.ndarray:
    """
    Predict expected deaths using the fitted model.

    Args:
        model (GLMResultsWrapper): Fitted Poisson model.
        data (pd.DataFrame): Data to predict on.

    Returns:
        np.ndarray: Array of predicted death counts.
    """
    return model.predict(data)

def add_observed_trace(fig: go.Figure, data: pd.DataFrame, year: int) -> None:
    """
    Add a trace for observed deaths to the figure.

    Args:
        fig (go.Figure): Plotly figure to add the trace to.
        data (pd.DataFrame): Data for the specific year.
        year (int): Year of the data.
    """
    line_width = 2 if year >= 2020 else 1  # Thicker lines for 2020 and later
    fig.add_trace(go.Scatter(
        x=data['week'],
        y=data['observed_deaths'],
        mode='lines',
        name=f'Observed {year}',
        line=dict(width=line_width)
    ))

def add_expected_trace(fig: go.Figure, data: pd.DataFrame, year: int, take_factor_into_account: bool, expected_deaths_added: bool) -> None:
    """
    Add a trace for expected deaths to the figure.

    Args:
        fig (go.Figure): Plotly figure to add the trace to.
        data (pd.DataFrame): Data for the specific year.
        year (int): Year of the data.
        take_factor_into_account (bool): Whether to consider demographic factors.
        expected_deaths_added (bool): Whether an expected deaths trace has already been added.
    """
    if take_factor_into_account:
        fig.add_trace(go.Scatter(
            x=data['week'],
            y=data['expected_deaths'],
            mode='lines',
            name=f'Expected {year}',
            line=dict(width=2, dash='dash')  # Thicker dashed line for expected deaths
        ))
    elif not expected_deaths_added:
        fig.add_trace(go.Scatter(
            x=data['week'],
            y=data['expected_deaths'],
            mode='lines',
            name='Expected',
            line=dict(color="black", width=2, dash='dash')  # Thicker dashed line for expected deaths
        ))

def add_excess_trace(fig: go.Figure, data: pd.DataFrame, year: int) -> None:
    """
    Add a trace for excess deaths to the figure.

    Args:
        fig (go.Figure): Plotly figure to add the trace to.
        data (pd.DataFrame): Data for the specific year.
        year (int): Year of the data.
    """
    fig.add_trace(go.Scatter(
        x=data['week'],
        y=data['excess_deaths'],
        mode='lines',
        name=f'Excess Deaths {year}',
        line=dict(width=2)  # Thicker line for excess deaths
    ))
def update_and_display_figures(fig_observed: go.Figure, fig_excess: go.Figure) -> None:
    """
    Update the layout of the figures and display them using Streamlit.

    Args:
        fig_observed (go.Figure): Figure for observed vs expected deaths.
        fig_excess (go.Figure): Figure for excess deaths.
    """
    fig_observed.update_layout(
        title='Observed vs Expected Deaths (2015-2024)',
        xaxis_title='Week', yaxis_title='Number of Deaths',
        legend_title='Legend', template='plotly_white'
    )
    fig_excess.update_layout(
        title='Excess Deaths by Year (2020-2024)',
        xaxis_title='Week', yaxis_title='Excess Deaths',
        legend_title='Legend', template='plotly_white'
    )
    st.plotly_chart(fig_observed)
    st.plotly_chart(fig_excess)

def add_fields(df_data,series_name):
    df_data["year"] = df_data["jaar"]
    df_data = df_data.sort_values(by=["year", "week"])
    df_data["observed_deaths"]= df_data[f"{series_name}"]
    df_data["Overledenen_1"] = df_data[f"{series_name}"]
    df_data["jaar"] = df_data["year"]
    df_data["weeknr"] = (
        df_data["jaar"].astype(str) + "_" + df_data["week"].astype(str).str.zfill(2)
    )
    return df_data
def main():
    st.header("Oversterfte berekening mbv Poisson model")
    st.info("""
    
        This function fits a Poisson model to observed death data from 2015 to 2019, and then predicts 
        expected deaths for the years 2020-2024 based on the fitted models. It calculates excess deaths by comparing observed deaths 
        to the expected deaths from both models. 
        
        The model to captures not just seasonal and weekly patterns, 
        but also overall trends in the data over time. This means it can account for gradual changes 
        in death rates that occur from year to year, beyond just the seasonal fluctuations. 
        
        NB : This model assumes that any trend observed in the training data (2015-2019) 
        continues linearly into the future. This may not always be a valid assumption, 
        especially over long time periods, big demographic changes or during unusual events (like a pandemic).
    """)
    

    serienames_ = [
        "m_v_0_999",
        "m_v_0_64",
        "m_v_65_79",
        "m_v_80_999",
        "m_0_999",
        "m_0_64",
        "m_65_79",
        "m_80_999",
        "v_0_999",
        "v_0_64",
        "v_65_79",
        "v_80_999",
    ]

    series_name = "m_v_0_999" # st.sidebar.selectbox("Leeftijden", serienames_, 0)
    take_factor_in_account = True #  st.sidebar.selectbox("Take Demographic factors in account", [True,False], 1)
    # before the Poisson model could take into account changes over time, I used a factor to multiply the baseline with.
    # For not changing the code totally, I just set take_factor_in_account on True

   

    df_data = get_sterftedata()
    df_data = add_fields(df_data, series_name)
    df_data = adjust_overledenen(df_data)
    perform_poisson_analysis(df_data, take_factor_in_account)
    
    st.info("Script: https://github.com/rcsmit/COVIDcases/blob/main/calculate_baselines.py")
if __name__ == "__main__":
    import datetime

    print(
        f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------"
    )

    main()
