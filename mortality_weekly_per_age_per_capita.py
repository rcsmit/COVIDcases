import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 

# https://claude.ai/chat/467c298a-027f-49ac-ae9e-9bd43ef92d8e
# https://chatgpt.com/c/66f0053e-79bc-8004-9e84-b77b055c4de1

def main():
    st.subheader("Weekly mortality/100k")
    st.info("reproducing https://x.com/dimgrr/status/1837603581962453167")
    # Load the data
    # Note: Replace these with the actual file paths on your system
    population_df = pd.read_csv('https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv', sep=';')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/sterfte_eurostats_NL.csv', sep=',')

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
    deaths_grouped = deaths_df.groupby(['age', 'sex', 'week', 'year'])['OBS_VALUE'].sum().reset_index()

    # Merge deaths and population data
    merged_df = pd.merge(deaths_grouped, population_grouped, 
                        left_on=['year', 'age', 'sex'], 
                        right_on=['jaar', 'age_group', 'geslacht'])

    # Calculate deaths per 100,000 people
    merged_df['deaths_per_100k'] = (merged_df['OBS_VALUE'] / merged_df['aantal']) * 100000
    merged_df = merged_df.sort_values(by=['year', 'week'], ascending=[True, True])
    merged_df['TIME_PERIOD'] = merged_df['year'].astype(str)+' - '+merged_df['week'].astype(str)
    print (merged_df)

    for sex in ["T", "M", "F"]:
        sex_mapping = {'M': 'Male', 'F': 'Female', 'T': 'Total'}
        sex_ = sex_mapping.get(sex, 'unknown')  # 'unknown' can be a default value for unrecognized sex codes

        # Create the plot
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


if __name__ == "__main__":
    #read_ogimet()
    main()

# If you want to save the plot as an HTML file, uncomment the following line:
# fig.write_html("deaths_per_100k_age_group.html")