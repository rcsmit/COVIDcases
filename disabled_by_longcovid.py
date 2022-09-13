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
    population_input = st.sidebar.number_input("Population", 0, 1_000_000_000, 100)
    infected_input = st.sidebar.number_input("Infected per year (%)", 0, 100, 8) / 100
    disabled_input = st.sidebar.number_input("Risk of Long covid (%)", 0, 100, 20) / 100
    recovered_input = st.sidebar.number_input("Recovered per year (%)", 0, 100, 50) / 100
    number_of_years_input = st.sidebar.number_input("Number of years", 0, 100, 2) 
    number_of_runs_input = st.sidebar.number_input("Number of runs", 0, 100, 1) 
    year_, infected_,lc_, disabled_ = [],[],[],[]
    
    disabled_bootstraps = []
    long_covid_bootstraps = []
    for h in range(number_of_runs_input):
        for i in range(1,number_of_years_input):
            
            
            disabled = np.array([])
            population = np.zeros(population_input)
            
            #for j in range(i):
            infected = np.random.choice(range(len(population)),size=int(infected_input*len(population)),replace=False)  # i aantal willekeurige nummers, worden indexes
            # st.write(" infected")
            # st.write(infected)
            # infect_mask = np.zeros(len(population))
            # infect_mask[infected] = 1
            # st.write("population[infected]")  # i maal een 0 - 

            # st.write(population[infected])  # i maal een 0
            population[infected] = population[infected]-1   # i maal -1. de hokjes met de index van inffected woren -1
            # st.write(population[infected])
            # -1 is infected, 0 is niet infected
            runif = np.random.uniform(size=len(population)) # populatie aantal willekeurig nummer tussen 0 en 1
            # st.write(runif)
            #p = weibull_min.cdf(x=np.abs(population), c=2, scale=6)
            #population = np.where((runif<=p)&(population<0),population+100,population)
            population = np.where((runif<=(1-disabled_input)),population+100,population) # positive number means covid
            # +100 is long covid, +0 is geen long covid

                                    # geen longcovid +0    # long covid +100
            # ifnected -1           -1                       99
            # not infected 0         0                       100
            

            # st.write("np.where((runif<=disabled_input),population+100,population) ")
            # st.write(population)   #-1, 99,100  of 0  -1 als de cijfer is in infected
                # 
            # runif = np.random.uniform(size=len(population))
            # population = np.where((runif>=0.5)&(population>0),population-100,population)
            
            runif = np.random.uniform(size=len(population))  # populatie aantal willekeurig nummer tussen 0 en 1
            # st.write("runif")
            # st.write(runif)
            del_mask = (population>0) & (runif>=(recovered_input)) #& (infect_mask==1)  #p maal boolean
            # st.write("del_mask = (population>0) & (runif>=(recovered_input))")
            # st.write(del_mask)
            # st.write("population[del_mask]")
            # st.write(population[del_mask])
            disabled = np.append(disabled,population[del_mask])
            # st.write("disabled = np.append(disabled,population[del_mask])")
            # st.write(disabled)
            population = population[~del_mask]
            # st.writse(population)
            st.write(f"JAAR {i} - LEN POPULATION {len(population)}")
            long_covid_bootstraps.append(np.sum(disabled>0)+np.sum(population>0))
            disabled_bootstraps.append(len(disabled))
        year_.append(i)

        
        lc_.append (np.mean(long_covid_bootstraps)/population_input)
        disabled_.append(np.mean(disabled_bootstraps)/population_input)
        st.write(f"Year: {i} | LC%: {np.mean(long_covid_bootstraps)/population_input} | Disabled by LC%: {np.mean(disabled_bootstraps)/population_input}")

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

if __name__ == "__main__":
    main()
