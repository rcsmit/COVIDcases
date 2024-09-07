import pandas as pd
import plotly.express as px
import streamlit as st
import platform

try:
    st.set_page_config(layout="wide")
except:
    pass

@st.cache_data()
def get_data():
    """get the data

    Returns:
        df: dataframe with data
    """
    if platform.processor() != "":
        file =  r"C:\Users\rcxsm\Documents\python_scripts\covid19_seir_models\COVIDcases\input\rivm_deltavax.csv"
    else: 
        file = r"https://raw.githubusercontent.com/rcsmit/COVIDcases/main/input/rivm_deltavax.csv"
   
    df = pd.read_csv(
            file,
            delimiter=";",
            
            low_memory=False,
        )

    # Split 'week_overlijden' into 'jaar_overlijden' and 'week_overlijden'
    df[['jaar_overlijden', 'week_overlijden']] = df['week_overlijden'].str.split('_', expand=True)

    # Convert the new columns to integers
    df['jaar_overlijden'] = df['jaar_overlijden'].astype(int)
    df['week_overlijden'] = df['week_overlijden'].astype(int)



    # Converteer week_overlijden en jaar_overlijden naar een datum, waarbij de week start op maandag
    df['datum_overlijden'] = pd.to_datetime(df['jaar_overlijden'].astype(str) + df['week_overlijden'].astype(str) + '1', format='%Y%W%w')

    # Maak kolommen voor elke vaccinatie datum (datum_vax1 t/m datum_vax6)
    for i in range(1, 7):
        df[f'datum_vax{i}'] = df['datum_overlijden'] - pd.to_timedelta(df[f'delta_vax{i}'], unit='d')
        df[f'jaar_vax{i}'] = df[f'datum_vax{i}'].dt.year
        df[f'maand_vax{i}'] = df[f'datum_vax{i}'].dt.month
    st.write (df)
    return df


def histogram_death_days_after_vax_x(df_deaths_after_vax_x,n, age_groups):

    cols = st.columns(len(age_groups))
    
    for idx, a in enumerate(age_groups):
        with cols[idx]:
             

            st.subheader(a)
            df_deaths_after_vax_x_age = df_deaths_after_vax_x[df_deaths_after_vax_x['leeftijdsgroep'] == a ]
            # Create the histogram with Plotly
            fig = px.histogram(
                df_deaths_after_vax_x_age,
                x=f'delta_vax{n}',
                nbins=int(df_deaths_after_vax_x_age[f'delta_vax{n}'].max() - df_deaths_after_vax_x_age[f'delta_vax{n}'].min()) + 1,
                title=f'Number of Deaths by Number of Days After Vaccination no.  {n} / [{a}]',
                labels={f'delta_vax{n}': f'Number of Days After Vaccination no.  {n} / [{a}]', 'count': 'Number of Deaths'}
            )

        

            # Show the plot
            st.plotly_chart(fig)


def death_after_n_moment_overlijden(df,n=2,jaar=2021,week=9):
    """ Histogram of how many days it took to die after the n-th vaccination
        reproducing https://twitter.com/mr_Smith_Econ/status/1831779516970119255
    Args:
        df (df): dataframe
        n (int, optional): Number of vaccination. Defaults to 2.
        jaar (int, optional): Year. Defaults to 2021.
        week (int, optional): week. Defaults to 13.
    """    
    df= df[(df[f'delta_vax{n}'] != -1) & (df[f'delta_vax{n+1}'] == -1)].copy(deep=True)
    df = df[(df['leeftijdsgroep'] == "80-89")|(df['leeftijdsgroep'] == ">= 90") ]
    df = df[(df['jaar_overlijden'] == jaar) & (df['week_overlijden'] == week)]
    fig = px.histogram(
        df,
        x=f'delta_vax{n}',
        nbins=int(df[f'delta_vax{n}'].max() - df[f'delta_vax{n}'].min()) + 1,
        title=f'Number of Deaths by Number of Days After Vaccination no.  {n} / [80+], died in week [{week}-{jaar}]',
        labels={f'delta_vax{n}': f'Number of Days After Vaccination no.  {n} / [80+]', 'count': 'Number of Deaths'}
    )

    # Show the plot
    st.plotly_chart(fig)

    #st.info("Er zijn 872.617 80+ in Nederland")


def death_after_n_moment_vaccinatie(df,n=2,jaar=2021,maand=2):
    """ Histogram of how many days it took to die after the n-th vaccination
        reproducing https://twitter.com/mr_Smith_Econ/status/1831779516970119255
    Args:
        df (df): dataframe
        n (int, optional): Number of vaccination. Defaults to 2.
        jaar (int, optional): Year. Defaults to 2021.
        week (int, optional): week. Defaults to 13.
    """    
    df= df[(df[f'delta_vax{n}'] != -1) & (df[f'delta_vax{n+1}'] == -1)].copy(deep=True)
    df = df[(df['leeftijdsgroep'] == "80-89")|(df['leeftijdsgroep'] == ">= 90") ]
    df = df[(df[f'jaar_vax{n}'] == jaar) & (df[f'maand_vax{n}'] == maand)]
    fig = px.histogram(
        df,
        x=f'delta_vax{n}',
        nbins=int(df[f'delta_vax{n}'].max() - df[f'delta_vax{n}'].min()) + 1,
        title=f'Number of Deaths by Number of Days After Vaccination no.  {n} / [80+], vaccinated in maand [{maand}-{jaar}]',
        labels={f'delta_vax{n}': f'Number of Days After Vaccination no.  {n} / [80+]', 'count': 'Number of Deaths'}
    )

    # Show the plot
    st.plotly_chart(fig)

    #st.info("Er zijn 872.617 80+ in Nederland")


def calculate_counts_and_percentages(df, conditions, age_groups):
    """Make table. Death after vax x in rows, agegroups in columns

    Args:
        df (df): data
        conditions (): list with the conditions
        age_groups (list): list with agegroups
    """
    result_counts = {age_group: [] for age_group in age_groups}
    result_percentages = {age_group: [] for age_group in age_groups}

    # Calculate counts
    for i, (col, val) in enumerate(conditions):
        prev_col = conditions[i-1][0] if i > 0 else None
        for age_group in age_groups:
            if prev_col:
                df_filtered = df[(df[prev_col] != -1) & (df[col] == val) & (df['leeftijdsgroep'] == age_group)]
            else:
                df_filtered = df[(df[col] == val) & (df['leeftijdsgroep'] == age_group)]
            count = len(df_filtered)
            result_counts[age_group].append(count)

    # Convert counts to DataFrame
    counts_df = pd.DataFrame(result_counts, index=[f'dead_after_{i+1}' for i in range(len(conditions))])

    # Calculate percentages
    percentages_df = counts_df.div(counts_df.sum(axis=0), axis=1).round(2) * 100

    # st.write results
    st.write("Counts DataFrame:")
    st.write(counts_df)
    st.write("\nPercentages DataFrame:")
    st.write(percentages_df)

def main():
    df = get_data()

    # Define the conditions and columns
    conditions = [
        ('delta_vax1', -1),
        ('delta_vax2', -1),
        ('delta_vax3', -1),
        ('delta_vax4', -1),
        ('delta_vax5', -1),
        ('delta_vax6', -1)
    ]

    # Define logical age groups
    age_groups = ['<50',  '50-59', '60-69', '70-79', '80-89', '>= 90']  # Adjust as needed

    calculate_counts_and_percentages(df, conditions, age_groups)
    death_after_n_moment_overlijden(df)
    death_after_n_moment_vaccinatie(df)

    # Calculate counts
    for i, (col, val) in enumerate(conditions):
        if i>0:
            prev_col = conditions[i-1][0]     
            df_filtered = df[(df[prev_col] != -1) & (df[col] == val)].copy(deep=True)
       
            st.subheader(f"After {i} vaccinations")
            histogram_death_days_after_vax_x(df_filtered,i, age_groups)


if __name__ == "__main__":
    main()
    