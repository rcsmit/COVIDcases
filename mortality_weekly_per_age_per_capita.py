import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
# 

# https://claude.ai/chat/467c298a-027f-49ac-ae9e-9bd43ef92d8e
# https://chatgpt.com/c/66f0053e-79bc-8004-9e84-b77b055c4de1
# https://claude.ai/chat/6e66dfdb-9b05-4223-b5e1-6ef25798c1c5

def plot_deaths_per_100k_per_year(df, age_group, sex):
    # Filter the DataFrame based on the selected age group and sex
    df_filtered = df[(df['age_group'] == age_group) & (df['geslacht'] == sex)]
   
    fig = px.line(
        df_filtered,
        x='week',             # X-axis: weeks
        y='deaths_per_100k',   # Y-axis: deaths per 100k
        color='year',          # Different line for each year
        title=f'Deaths per 100k for {age_group} ({sex}) by Week',
        labels={'week': 'Week', 'deaths_per_100k': 'Deaths per 100k'},
    )
    
    # Show the plot
    st.plotly_chart(fig)


def main():
    st.subheader("Weekly and yearly mortality/100k")
    st.info("reproducing https://x.com/dimgrr/status/1837603581962453167")
    # Load the data
    # Note: Replace these with the actual file paths on your system
    population_df, deaths_df = get_data()

    # Define age bins and labels
    bins = list(range(0, 95, 5)) + [1000]  # [0, 5, 10, ..., 90, 1000]
    labels = [f'Y{i}-{i+4}' for i in range(0, 90, 5)] + ['Y_GE90']

    # Process the population dataframe
    population_df['leeftijd'] = population_df['leeftijd'].astype(int)
    population_df['age_group'] = pd.cut(population_df['leeftijd'], bins=bins, labels=labels, right=False)
    print (population_df)
    population_grouped = population_df.groupby(['jaar', 'age_group', 'geslacht'], observed=False)['aantal'].sum().reset_index()

    # Replace age group labels
    # for s in ["M", "F", "T"]:
    population_grouped['age_group'] = population_grouped['age_group'].cat.add_categories('Y_LT5')

    population_grouped.loc[population_grouped['age_group'] == 'Y0-4', 'age_group'] = 'Y_LT5'

        # population_grouped['age_group'] = population_grouped.apply(lambda row: f"{row['age_group']}_{row['geslacht']}" if row['age_group'] not in [f'Y_LT5_{s}', f'Y_GE90_{s}'] else row['age_group'], axis=1)

    # Process the deaths dataframe
    # deaths_df['TIME_PERIOD'] = pd.to_datetime(deaths_df['TIME_PERIOD'])
    # deaths_df['week'] = deaths_df['TIME_PERIOD'].dt.isocalendar().week
    # deaths_df['year'] = deaths_df['TIME_PERIOD'].dt.year

    deaths_df["year"] = (deaths_df["TIME_PERIOD"].str[:4]).astype(int)
    deaths_df["week"] = (deaths_df["TIME_PERIOD"].str[6:]).astype(int)
    deaths_grouped_week = deaths_df.groupby(['age', 'sex', 'week', 'year'])['OBS_VALUE'].sum().reset_index()
    deaths_grouped_jaar = deaths_df.groupby(['age', 'sex', 'year'])['OBS_VALUE'].sum().reset_index()
    
    plot_wrapper(deaths_grouped_week, population_grouped)
    plot_wrapper(deaths_grouped_jaar, population_grouped)
    lin_regression(deaths_grouped_jaar, population_grouped)

def lin_regression(deaths_grouped, population_grouped):
    merged_df = pd.merge(deaths_grouped, population_grouped, 
                        left_on=['year', 'age', 'sex'], 
                        right_on=['jaar', 'age_group', 'geslacht'])
    # Calculate deaths per 100,000 people
    merged_df['deaths_per_100k'] = (merged_df['OBS_VALUE'] / merged_df['aantal']) * 100000
    
    merged_df = merged_df.sort_values(by=['year'], ascending=[True])
    df_with_predictions = predict_death_rates(merged_df)
    st.write(df_with_predictions)
    plot_death_comparison(df_with_predictions)
    plot_death_comparison_per_100k(df_with_predictions)
    st.write(sum_predicted_deaths(df_with_predictions))
def sum_predicted_deaths(df):
    """
    Sum predicted deaths by year and sex.
    
    Parameters:
    df: DataFrame with predicted_deaths and geslacht columns
    
    Returns:
    DataFrame with yearly sums by sex
    """
    # Get only the rows with predictions
    predicted_data = df[df['predicted_deaths'].notna()]
    
    # Group by year and sex, sum the predicted deaths
    yearly_sums = (predicted_data.groupby(['year', 'geslacht'])['predicted_deaths']
                  .sum()
                  .round()
                  .reset_index())
    
    # Pivot the table to show sex as columns
    summary_table = yearly_sums.pivot(
        index='year',
        columns='geslacht',
        values='predicted_deaths'
    ).reset_index()
    
    # Add total column
    summary_table['M+F'] = summary_table['M']+summary_table['F']
    
    # Round all values
    numeric_columns = summary_table.columns[summary_table.dtypes != 'object']
    summary_table[numeric_columns] = summary_table[numeric_columns].round(0).astype(int)
    
    return summary_table



def predict_death_rates(df):
    """
    Create death rate predictions for 2020-2024 based on 2000-2019 data,
    using actual population counts to calculate predicted deaths.
    
    Parameters:
    df: DataFrame with columns year, age_group, geslacht (sex), deaths_per_100k, aantal
    
    Returns:
    DataFrame with original data and predictions on same rows
    """
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Ensure year is int
    df_copy['year'] = df_copy['year'].astype(int)
    
    # Initialize prediction columns
    df_copy['predictions_per_100k'] = None
    df_copy['predicted_deaths'] = None
    
    # Get unique combinations of age group and sex
    combinations = df_copy.groupby(['age_group', 'geslacht'])
    
    for (age, sex), group in combinations:
        # Filter data for 2000-2019 for training
        historical_data = group[group['year'].between(2000, 2019)].copy()
        
        if len(historical_data) > 0:
            # Prepare the model
            X = historical_data[['year']].values
            y = historical_data['deaths_per_100k'].values
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate predictions for 2020-2024
            future_mask = (df_copy['age_group'] == age) & \
                         (df_copy['geslacht'] == sex) & \
                         (df_copy['year'].between(2020, 2024))
            
            if future_mask.any():
                future_years = df_copy.loc[future_mask, 'year'].values.reshape(-1, 1)
                future_predictions = model.predict(future_years)
                
                # Update predictions in the dataframe
                df_copy.loc[future_mask, 'predictions_per_100k'] = \
                    np.maximum(0, np.round(future_predictions, 1))
                
                # Calculate predicted deaths using actual aantal
                df_copy.loc[future_mask, 'predicted_deaths'] = \
                    np.round((df_copy.loc[future_mask, 'predictions_per_100k'] * 
                             df_copy.loc[future_mask, 'aantal']) / 100000)
    
    return df_copy
    
def plot_death_comparison(df):
    """
    Create a scatter plot comparing real and predicted deaths across all age groups.
    
    Parameters:
    df: DataFrame with columns year, age_group, deaths_per_100k, predictions_per_100k, aantal
    
    Returns:
    Plotly figure object
    """
    # Calculate real deaths for the entire period
    df=df[df["geslacht"]=="T"]
    df['real_deaths'] = (df['deaths_per_100k'] * df['aantal'] / 100000).round()
    
    # Create separate traces for real and predicted deaths
    fig = go.Figure()
    
    # Add real deaths (blue)
    real_deaths_data = df[df['deaths_per_100k'].notna()]
    for age in df['age_group'].unique():
        age_data = real_deaths_data[real_deaths_data['age_group'] == age]
        
        fig.add_trace(go.Scatter(
            x=age_data['year'],
            y=age_data['real_deaths'],
            name=f'{age} (Real)',
            mode='markers',
            marker=dict(color='blue', size=4),
            # legendgroup='real',
            hovertemplate=(
                '<b>Age Group:</b> %{text}<br>' +
                '<b>Year:</b> %{x}<br>' +
                '<b>Deaths:</b> %{y:,.0f}<br>' +
                '<extra></extra>'
            ),
            text=[age] * len(age_data)
        ))
    
    # Add predicted deaths (red)
    predicted_data = df[df['predictions_per_100k'].notna()]
    for age in df['age_group'].unique():
        age_data = predicted_data[predicted_data['age_group'] == age]
        
        fig.add_trace(go.Scatter(
            x=age_data['year'],
            y=age_data['predicted_deaths'],
            name=f'{age} (Predicted)',
            mode='markers',
            marker=dict(color='red', size=4),
            # legendgroup='predicted',
            hovertemplate=(
                '<b>Age Group:</b> %{text}<br>' +
                '<b>Year:</b> %{x}<br>' +
                '<b>Predicted Deaths:</b> %{y:,.0f}<br>' +
                '<extra></extra>'
            ),
            text=[age] * len(age_data)
        ))
    
    # Update layout
    fig.update_layout(
        title='Real vs Predicted Deaths by Age Group',
        xaxis_title='Year',
        yaxis_title='Number of Deaths',
        plot_bgcolor='white',
        hovermode='closest',
        legend_title='Age Groups',
        yaxis_type="log",
        showlegend=True
    )
    st.plotly_chart(fig)
 
def plot_death_comparison_per_100k(df):
    """
    Create a scatter plot comparing real and predicted deaths across all age groups.
    
    Parameters:
    df: DataFrame with columns year, age_group, deaths_per_100k, predictions_per_100k, aantal
    
    Returns:
    Plotly figure object
    """
    # Calculate real deaths for the entire period
    df=df[df["geslacht"]=="T"]
    
    # Create separate traces for real and predicted deaths
    fig = go.Figure()
    
    # Add real deaths (blue)
    real_deaths_data = df[df['deaths_per_100k'].notna()]
    for age in df['age_group'].unique():
        age_data = real_deaths_data[real_deaths_data['age_group'] == age]
        
        fig.add_trace(go.Scatter(
            x=age_data['year'],
            y=age_data['deaths_per_100k'],
            name=f'{age} (Real)',
            mode='markers',
            marker=dict(color='blue', size=4),
            # legendgroup='real',
            hovertemplate=(
                '<b>Age Group:</b> %{text}<br>' +
                '<b>Year:</b> %{x}<br>' +
                '<b>Deaths per 100k:</b> %{y:,.0f}<br>' +
                '<extra></extra>'
            ),
            text=[age] * len(age_data)
        ))
    
    # Add predicted deaths (red)
    predicted_data = df[df['predictions_per_100k'].notna()]
    for age in df['age_group'].unique():
        age_data = predicted_data[predicted_data['age_group'] == age]
        
        fig.add_trace(go.Scatter(
            x=age_data['year'],
            y=age_data['predictions_per_100k'],
            name=f'{age} (Predicted per 100k)',
            mode='markers',
            marker=dict(color='red', size=4),
            # legendgroup='predicted',
            hovertemplate=(
                '<b>Age Group:</b> %{text}<br>' +
                '<b>Year:</b> %{x}<br>' +
                '<b>Predicted Deaths:</b> %{y:,.0f}<br>' +
                '<extra></extra>'
            ),
            text=[age] * len(age_data)
        ))
    
    # Update layout
    fig.update_layout(
        title='Real vs Predicted Deaths per 100k by Age Group',
        xaxis_title='Year',
        yaxis_title='Number of Deaths per 100k',
        plot_bgcolor='white',
        hovermode='closest',
        legend_title='Age Groups',
      
        yaxis_type="log",
        showlegend=True
    )
    
    # # Update axes
    # fig.update_xaxis(
    #     gridcolor='lightgray',
    #     zeroline=True,
    #     zerolinecolor='lightgray'
    # )
    
    # fig.update_yaxis(
    #     gridcolor='lightgray',
    #     zeroline=True,
    #     zerolinecolor='lightgray'
    # )
    

    st.plotly_chart(fig)
    return fig
def plot_wrapper(deaths_grouped, population_grouped):
    # Merge deaths and population data
    merged_df = pd.merge(deaths_grouped, population_grouped, 
                        left_on=['year', 'age', 'sex'], 
                        right_on=['jaar', 'age_group', 'geslacht'])

    # Calculate deaths per 100,000 people
    merged_df['deaths_per_100k'] = (merged_df['OBS_VALUE'] / merged_df['aantal']) * 100000
    try:
        merged_df = merged_df.sort_values(by=['year', 'week'], ascending=[True, True])
    except:
        merged_df = merged_df.sort_values(by=['year'], ascending=[True])
    try:
        merged_df['TIME_PERIOD'] = merged_df['year'].astype(str)+' - '+merged_df['week'].astype(str)
    except:
        merged_df['TIME_PERIOD'] = merged_df['year'].astype(str)
    print (merged_df.dtypes)

    for sex in ["T", "M", "F"]:
        sex_mapping = {'M': 'Male', 'F': 'Female', 'T': 'Total'}
        sex_ = sex_mapping.get(sex, 'unknown')  # 'unknown' can be a default value for unrecognized sex codes

        # Create the plot
        make_plot(merged_df, sex, sex_)
    
    # Example usage:
    plot_deaths_per_100k_per_year(merged_df, 'Y_GE90', 'M')

def make_plot(merged_df, sex, sex_):
    fig = go.Figure()

        # Plot each age group for total population
    for age in merged_df[merged_df['sex'] == sex]['age'].unique():
        age_data = merged_df[(merged_df['age'] == age) & (merged_df['sex'] == sex)]
        fig.add_trace(go.Scatter(
                x=age_data['TIME_PERIOD'],
                #x=age_data['week'] + (age_data['year'] - age_data['year'].min()) * 52,
                y=age_data['deaths_per_100k'],
                mode='lines',
                name=age
            ))


        # Update layout
    fig.update_layout(
            title=f'Deaths per 100,000 People by Age Group per Week ({sex_} Population)',
            xaxis_title='Week (cumulative across years)',
            yaxis_title='Deaths per 100,000 People (log scale)',
            yaxis_type="log",
            legend_title='Age Group',
        )

        # Show the plot

    st.plotly_chart(fig)
    return 


@st.cache_data()
def get_data():
    population_df = pd.read_csv('https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv', sep=';')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_NL.csv', sep=',')
    return population_df,deaths_df


if __name__ == "__main__":
    #read_ogimet()
    main()

# If you want to save the plot as an HTML file, uncomment the following line:
# fig.write_html("deaths_per_100k_age_group.html")