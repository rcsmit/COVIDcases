from scipy.stats import binom

# https://claude.ai/chat/b094998c-3398-446e-997f-83e34832e41f
# https://chatgpt.com/c/66f3507c-3194-8004-a265-a915d19d7786
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
    
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def pdf(deaths, population, expected, vaccinated, added_risk):
    """_summary_

    Args:
        deaths (_type_): _description_
        population (_type_): _description_
        expected (_type_): _description_
        vaccinated (_type_): _description_
        added_risk (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Calculate vaccinated individuals in each risk category
    # newly_vaccinated = vaccinated_history[-1] - vaccinated_history[-2]
    # one_week_ago = vaccinated_history[-2] - vaccinated_history[-3]
    # two_weeks_ago = vaccinated_history[-3] - vaccinated_history[-4]
    # three_plus_weeks_ago = vaccinated_history[-4]
    vpdf = 0
    
    # Case when there are no vaccinated individuals
    if vaccinated == 0:
        vpdf = binom.pmf(deaths, population, expected / population)
        return vpdf
    
    # Case when everyone is vaccinated
    if vaccinated >= population:
        vpdf = binom.pmf(deaths, vaccinated, (expected / population) * (1 + added_risk))
        return vpdf
    
    # Case when there are both vaccinated and unvaccinated individuals
    for n in range(deaths + 1):
        # Unvaccinated population
        U = binom.pmf(deaths - n, population - vaccinated, expected / population)
        
        # Vaccinated population
        V = binom.pmf(n, vaccinated, (expected / population) * (1 + added_risk))
        
        # Sum up the probabilities
        vpdf += V * U
    
    return vpdf

@st.cache_data()
def get_data():
    vaccinations_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/vaccinations_NL_2021.csv"
    deaths_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/df_merged_m_v_65_79.csv"
    bevolking_url = "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/bevolking_leeftijd_NL.csv"
    df_vaccinations = pd.read_csv(
        vaccinations_url,
        delimiter=",",
        low_memory=False,
    )

    df_deaths =  pd.read_csv(
        deaths_url,
        delimiter=",",
        low_memory=False,
    )

    df_bevolking =  pd.read_csv(
        bevolking_url,
        delimiter=";",
        low_memory=False,
    )
    
    return df_vaccinations,df_deaths,df_bevolking
def make_graph_mpl(df):
    weeks = df["week"]
    likelihood_ratio = df["ll"]
    vaccination_doses = df["doses_65_79"]
    # Create a figure and subplots
    fig, ax1 = plt.subplots(figsize=(8, 10))

    # Plot likelihood ratio on the left side (red)
    ax1.barh(weeks, likelihood_ratio, color='red', alpha=0.6, label='Likelihood Ratio')
    ax1.set_xlabel('Likelihood Ratio')
    ax1.set_ylabel('Week')
    ax1.set_yticks(weeks)
    ax1.set_title('Age group 65-80, both Male and Female')

    # Invert the y-axis to match the style in the image (week 1 at the top)
    ax1.invert_yaxis()

    # Create a second y-axis for vaccination doses
    ax2 = ax1.twiny()

    # Plot vaccination doses on the right side (blue)
    ax2.barh(weeks, vaccination_doses, color='blue', alpha=0.6, label='Vaccination Doses')
    ax2.set_xlabel('Vaccination Doses')

    # Show plot
    
    st.pyplot(plt)

def make_graph(df):
    # Assuming you have your data in a pandas DataFrame called 'df'
    # If not, you'll need to load your data first
    # df = pd.read_csv('your_data.csv')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Likelihood Ratio bars
    fig.add_trace(
        go.Bar(
            x=df['week'],
            y=df['ll'],
            name="Likelihood Ratio",
            marker_color='red',
            opacity=0.6
        ),
        secondary_y=False,
    )

    # Add Vaccination Doses scatter
    fig.add_trace(
        go.Scatter(
            x=df['week'],
            y=df['doses_65_79'],
            name="Vaccination Doses",
            mode='markers',
            marker=dict(
                color='blue',
                size=10,
            )
        ),
        secondary_y=True,
    )

    # Add Likelihood Ratio of 1 line
    fig.add_hline(y=1, line_dash="dot", line_color="black", secondary_y=False)

    # Set x-axis title
    fig.update_xaxes(title_text="Week 2021")

    # Set y-axes titles
    fig.update_yaxes(title_text="Likelihood Ratio", secondary_y=False, type="log")
    fig.update_yaxes(title_text="Vaccination Doses", secondary_y=True)

    # Update layout
    fig.update_layout(
        title_text="Age group 65 - 80, both Male and Female",
        barmode='relative',
        height=600,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Show the figure
    st.plotly_chart(fig)


def main():
    df_vaccinations, df_deaths, df_bevolking = get_data()
    increased_prob = st.sidebar.number_input("Increased probability", 0.0,1.0,0.1)

    population = df_bevolking[
        (df_bevolking["geslacht"] == "T") & 
        (df_bevolking["jaar"] == 2021) & 
        (df_bevolking["leeftijd"] >= 65) & 
        (df_bevolking["leeftijd"] <= 79)
        ]["aantal"].sum()

    df = pd.merge(df_vaccinations, df_deaths, on=["jaar", "week"], how="inner")
    df = df[["jaar", "week", "doses_65_79","aantal_overlijdens","avg"]]
    results = []
    for i in range(1,42):
        vaccinated_list = []

        for offset in range(0, 4):
            week_value = df.loc[(df['jaar'] == 2021) & (df['week'] == (i - offset)), 'doses_65_79']
            
            if not week_value.empty:
                vaccinated_list.append(week_value.values[0])
            else:
                vaccinated_list.append(0)  # If no value exists, a
        #print (vaccinated_list)

        vaccinated = sum(vaccinated_list)   
        deaths = int(df.loc[(df['jaar'] == 2021) & (df['week'] == i), 'aantal_overlijdens'].values[0])
        expected = df.loc[(df['jaar'] == 2021) & (df['week'] == i), 'avg'].values[0]
        verhoogd = pdf(deaths, population,expected, vaccinated, increased_prob)
        not_verhooogd = pdf(deaths, population,expected, vaccinated, 0.0)
        ll = verhoogd/not_verhooogd
        print(f"{i} - {ll}")
        # Append the result to the list
        results.append({
            'jaar': 2021,  # Assuming years like 2014, 2015, etc.
            'week': i,
            'll': ll
        })


    # Convert the list of results to a DataFrame
    df_results = pd.DataFrame(results)

    # Assuming df_data is already defined, merge it with df_results
    df_merged = pd.merge(df, df_results, on=['jaar', 'week'], how='right')
    st.subheader("COVID-19 vaccinations and mortality - a Bayesian analysis")
    st.info("""Meester et al. Replicating COVID-19 vaccinations and mortality - a Bayesian analysis
            https://web.archive.org/web/20221202174753/https://www.researchgate.net/publication/357032975_COVID-19_vaccinations_and_mortality_-_a_Bayesian_analysis""")
    
    # Display the merged dataframe
    st.write(df_merged)
    make_graph(df_merged)
    make_graph_mpl(df_merged)



if __name__ == "__main__":
    import datetime

    print(
        f"-----------------------------------{datetime.datetime.now()}-----------------------------------------------------"
    )

    main()
