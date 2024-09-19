# based on https://twitter.com/MathStuart/status/1558485615930445826
# https://gist.github.com/stuartjonesstats/d791d433d35aeac09becbfe3b8a2deb1
# pythonized by https://twitter.com/koehlepe/status/1558593601013252096
# https://gist.github.com/koehlepe/4cc31e1299f7397f59ced04e1d4b305c

import numpy as np
from scipy.stats import weibull_min

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import platform
import plotly.express as px
def main():
    year_,lc_,disabled_=[],[],[]
    population_input = st.sidebar.number_input("Population", 0, 1_000_000_000, 10000)
    infected_input = st.sidebar.number_input("Infected per year (%)", 0, 100, 40) / 100
    disabled_input = st.sidebar.number_input("Risk of Long covid (%)", 0, 100, 20) / 100
    recovered_input = st.sidebar.number_input("Recovered per year (%)", 0, 100, 90) / 100
    number_of_years_input = st.sidebar.number_input("Number of years", 0, 100, 10) 
    
    for i in range(1, number_of_years_input + 1):  # Iterate over years
        disabled_bootstraps = []
        long_covid_bootstraps = []
        
        for h in range(100):  # Run 100 simulations per year
            disabled = np.array([])
            population = np.zeros(population_input)
            
            for j in range(i):  # Loop over each year of infection
                infected = np.random.choice(range(len(population)), size=int(infected_input * len(population)), replace=False)
                
                population[infected] -= 1  # Infected individuals become -1
                
                # Calculate probability using Weibull distribution
                runif = np.random.uniform(size=len(population))
                p = weibull_min.cdf(x=np.abs(population), c=2, scale=6)
                population = np.where((runif <= p) & (population < 0), population + 100, population)
            
            # Disable individuals who did not recover with probability 0.9 (like original)
            runif = np.random.uniform(size=len(population))
            del_mask = (population > 0) & (runif >= recovered_input)  # Fixed 90% recovery rate
            disabled = np.append(disabled, population[del_mask])
            population = population[~del_mask]  # Remove disabled individuals from the population
            
            long_covid_bootstraps.append(np.sum(disabled > 0) + np.sum(population > 0))  # Track long COVID cases
            disabled_bootstraps.append(len(disabled))  # Track disabled individuals
        
        # Display results for the current year, averaged over 100 simulations
        st.write(f"Year: {i} - LC%: {round(np.mean(long_covid_bootstraps) / 100,1)} - Disabled by LC%: {round(np.mean(disabled_bootstraps) / 100,1)}")
        year_.append(i)
        lc_.append(np.mean(long_covid_bootstraps) / population_input)
        disabled_.append(np.mean(disabled_bootstraps) / population_input)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace( go.Scatter(x=year_,
                            name='long covid',
                            y=lc_,
                            line=dict(width=2,color='rgba(205, 61,62, 1)'),
                            mode='lines'))
    fig.add_trace(  go.Scatter(
                    name='disabled',
                    x=year_,
                    y=disabled_,
                    mode='lines',
                    
                    line=dict(width=2,
                            color="rgba(94, 172, 219, 1)")
                    )  ,secondary_y=True) 

    st.plotly_chart(fig, use_container_width=True)
    st.write("based on https://twitter.com/MathStuart/status/1558485615930445826")
    st.write("pythonized by https://twitter.com/koehlepe/status/1558593601013252096")

    #koehlepe()
def koehlepe():
    # originele script

    for i in range(1,number_of_years_input+1):
        disabled_bootstraps = []
        long_covid_bootstraps = []
        for h in range(100):
            disabled = np.array([])
            population = np.zeros(population_input)
            for j in range(i):
                
                infected = np.random.choice(range(len(population)),size=int(infected_input*len(population)),replace=False)
                # infect_mask = np.zeros(len(population))
                # infect_mask[infected] = 1
                population[infected] = population[infected]-1
                runif = np.random.uniform(size=len(population))
                p = weibull_min.cdf(x=np.abs(population), c=2, scale=6)
                population = np.where((runif<=p)&(population<0),population+100,population)
                #population = np.where((runif<=p),population+100,population)
                # runif = np.random.uniform(size=len(population))
                # population = np.where((runif>=0.5)&(population>0),population-100,population)
                runif = np.random.uniform(size=len(population))
                del_mask = (population>0) & (runif>=recovered_input) #& (infect_mask==1)
                disabled = np.append(disabled,population[del_mask])
                population = population[~del_mask]
            long_covid_bootstraps.append(np.sum(disabled>0)+np.sum(population>0))
            disabled_bootstraps.append(len(disabled))
        st.write(f"Year: {i} - LC%: {np.mean(long_covid_bootstraps)/100.0} - Disabled by LC%: {np.mean(disabled_bootstraps)/100.0}")
        
if __name__ == "__main__":
    main()
