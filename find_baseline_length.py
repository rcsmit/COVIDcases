import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
import streamlit as st


from math import sqrt
import itertools
from isoweek import Week
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Replicating finding the baseline 
# https://www.mortality.watch/about
# For each jurisdiction, period of time and age, pre-pandemic data 
# is back tested to choose the baseline length n with the lowest root 
# mean squared error (RMSE) for a four to ten-year period of a four-year forecast. 
# A linear regression model is used (fable::TSLM + trend()) with a seasonal parameter 
# added for sub-year resolutions.


# https://claude.ai/chat/8c720e6f-eb4e-45a4-9d35-cdfac86df65f
# https://chatgpt.com/c/66e974bb-d224-8004-b1c1-6a453bebc86c
# def prepare_data(df, jurisdiction, age):
#     # Filter data for the specific jurisdiction and age
#     data = df[(df['geo'] == jurisdiction) & (df['age'] == age)]
    



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
        return totals
    
    for i in opdeling:
        custom_age_group = add_custom_age_group_deaths(df_, i[0], i[1])
        df_ = pd.concat([df_, custom_age_group], ignore_index=True)

  
   
    #df_bevolking = get_bevolking(country, opdeling)

    # summed_per_year = df_.groupby(["jaar", 'age_sex'])['OBS_VALUE'].sum() # .reset_index()
  
    # df__ = pd.merge(summed_per_year, df_bevolking, on=['jaar', 'age_sex'], how='outer')
    # df__ = df__[df__["aantal"].notna()]
    # df__ = df__[df__["OBS_VALUE"].notna()]
    # df__ = df__[df__["jaar"] != 2024]
    # df__["per100k"] = round(df__["OBS_VALUE"]/df__["aantal"]*100000,1)
    
    return df_



def prepare_data(df, jurisdiction, age):
    # Filter data for the specific jurisdiction and age
    data = df[(df['geo'] == jurisdiction) & (df['age'] == age)]
    
    # Convert TIME_PERIOD to datetime
    def iso_to_datetime(iso_week):
        year, week = map(int, iso_week.split('-W'))
        return Week(year, week).monday()  # Use Monday as the start of the week

    data.loc[:,'TIME_PERIOD'] = data['TIME_PERIOD'].apply(iso_to_datetime)
    data = data[data["jaar"]>2009]

    # Group by TIME_PERIOD and calculate the mean of OBS_VALUE
    data = data.groupby('TIME_PERIOD')['OBS_VALUE'].mean().reset_index()
    
    # Sort by time and select only OBS_VALUE
    data = data.sort_index()['OBS_VALUE']
    
    # Ensure the index is DatetimeIndex and set the frequency to weekly
    #data.index = pd.DatetimeIndex(data.index)
    #data = data.asfreq('W-MON')  # Set frequency to weekly, with Monday as the start of the week
    
    
    
    # # Handle any missing values that might have been introduced
    # data = data.interpolate()
    # st.write (data)
    return data

def calculate_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))



def find_best_baseline_sarimax(data, min_years=4, max_years=10):
    # very slow
    best_rmse = float('inf')
    best_n = 0
    
    for n in range(min_years * 52, (max_years * 52) + 1, 52):
        train = data[:-208]
        test = data[-208:]
        
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
        fit = model.fit(disp=False)
        
        predictions = fit.forecast(208)
        rmse = calculate_rmse(test, predictions)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_n = n
    
    return best_n // 52

def create_forecast_model_sarimax(data, baseline_years):
    # very slow
    train = data[-baseline_years*52:]
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
    return model.fit(disp=False)


def find_best_baseline_lineairregression(data, min_years=4, max_years=10):
    best_rmse = float('inf')
    best_n = 0

    for n in range(min_years * 52, (max_years * 52) + 1, 52):
        # Prepare training data
        X_train = np.arange(n).reshape(-1, 1)  # Time indices as independent variable
        y_train = data[-(n + 208):-208]        # The data for the last 'n' weeks before the test set

        # Prepare test data
        X_test = np.arange(n, n + 208).reshape(-1, 1)  # Time indices for the test set
        y_test = data[-208:]                           # Actual data for the last 208 weeks
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions for the test set
        predictions = model.predict(X_test)
        
        # Calculate RMSE
        rmse = calculate_rmse(y_test, predictions)
        
        # Update best baseline if current model is better
        if rmse < best_rmse:
            best_rmse = rmse
            best_n = n
    
    return best_n #// 52  # Return the best number of years


def find_best_baseline_exponentialsmoothing(data, min_years=4, max_years=10):
    best_rmse = float('inf')
    best_n = 0
    
    for n in range(min_years * 52, (max_years * 52) + 1, 52):
        train = data[-(n + 208):-208]  # Train on the last 'n' weeks before the test set
        test = data[-208:]  # Last 208 weeks for testing
        
        model = ExponentialSmoothing(train, seasonal_periods=52, trend='add', seasonal='add')
        fit = model.fit()
        
        predictions = fit.forecast(208)
        rmse = calculate_rmse(test, predictions)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_n = n
    
    return best_n #// 52

def create_forecast_model(data, baseline_years):
    train = data[-baseline_years*52:]
    model = ExponentialSmoothing(train, seasonal_periods=52, trend='add', seasonal='add')
    return model.fit()



def process_jurisdiction_age(args):
    df, jurisdiction, age, method = args
    data = prepare_data(df, jurisdiction, age)
    
    if len(data) < 8 * 52:
        return (jurisdiction, age, None)
    
    # Dynamically call the chosen method
    if method == "lineair":
        best_baseline = find_best_baseline_lineairregression(data)
    elif method == "exponentialsmoothing":
        best_baseline = find_best_baseline_exponentialsmoothing(data)
    elif method == "sarimax":
        best_baseline = find_best_baseline_sarimax(data)

    else:
        raise ValueError(f"Unknown method: {method}")
    
    #return (jurisdiction, age, best_baseline,  method)
    return (jurisdiction, age, method,{'baseline_years': best_baseline, 'method': method})

def main_(df):

    
    jurisdictions = ["NL"]  # df['geo'].unique()
    ages =df['age'].unique() #  ["TOTAL", "Y30-34"]  # 
    methods = ["lineair", "exponentialsmoothing"] #, "sarimax"]  # Add more methods as needed

    # Create a list of arguments for each combination of jurisdiction, age, and method
    args_list = [(df, jurisdiction, age, method) for jurisdiction, age, method in itertools.product(jurisdictions, ages, methods)]

    results = {}
    with ProcessPoolExecutor() as executor:
        future_to_args = {executor.submit(process_jurisdiction_age, args): args for args in args_list}
        try:
            # some give not a value for one of the variables
            for future in as_completed(future_to_args):
                jurisdiction, age, method, result= future.result()
                if result is not None:
                    # Store results using a tuple of (jurisdiction, age, method) as the key
                    results[(jurisdiction, age, method)] = result
                    # st.write(f"Processed {jurisdiction}, {age}, {method}, {result}")
        except Exception as e:
            # Catch any exception and return a detailed error message
            st.write(f"Error processing {jurisdiction}, {age}, {method}: {str(e)}")
    # Now you have results for each combination of jurisdiction, age, and method
    return results


# Assuming `results` is your dictionary from the model with (jurisdiction, age) as keys
def results_to_dataframe(results):
    
    # Initialize a list to hold rows for the dataframe
    rows = []
    
    # Loop through the results and extract data
    for (jurisdiction, age, method), result in results.items():
        # Append a row to the list of rows
        rows.append({
            "Jurisdiction": jurisdiction,
            "Age": age,
            "Method":result['method'],
            "Best Baseline (Years)": round(result['baseline_years'] / 52, 1)  # Convert weeks to years
        })
    
    # Convert list of rows to a DataFrame
        df_results = pd.DataFrame(rows)

      
    return df_results

def main():
    opdeling = [[0,14],[15,65],[65,79],[80,120]]
    df= get_sterfte(opdeling)
    st.info("""
            We try to find the best baseline. For each age(group), pre-pandemic data 
            is back tested to choose the baseline length n with the lowest root 
            mean squared error (RMSE) for a four to ten-year period of a four-year forecast. """)
    # Assuming your dataframe is called 'df'
    results = main_(df)
    
    df_results = results_to_dataframe(results)
    
    pivot_df = df_results.pivot(index='Age', columns='Method', values='Best Baseline (Years)')
    st.write (pivot_df)
    # # You can now access the results for each jurisdiction and age
    # for (jurisdiction, age), result in results.items():
    #     st.write(f"Jurisdiction: {jurisdiction}, age: {age} | Best baseline: {round(result['baseline_years']/52,1)} years")
       
    #     # st.write(f"Forecast for next 4 years:")
    #     # st.write(result['forecast'])
        
    st.info("""
    
            Inspired by mortality.watch. They use a linear regression model  (fable::TSLM + trend()) 
            with a seasonal parameter added for sub-year resolutions. In Python it equivalents with 
            SARIMAX, but this is very slow and thus ommited
            """)

if __name__ == "__main__":
   
    main()
