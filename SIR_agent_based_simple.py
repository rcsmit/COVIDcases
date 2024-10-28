import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

class Agent:
    def __init__(self):
        self.state = 'S'  # Initial state: Susceptible
        self.days_in_hospital = 0  # Track days in hospital
        self.days_in_ic = 0        # Track days in IC

    def infect(self, prob_infection):
        if self.state == 'S' and np.random.rand() < prob_infection:
            self.state = 'I'  # Move to Infected state

    def update_state(self, prob_hospitalization, prob_ic, prob_recovery, prob_death, avg_stay_hospital, avg_stay_ic):
        if self.state == 'I':
            if np.random.rand() < prob_hospitalization:
                self.state = 'H'  # Hospitalized
                self.days_in_hospital = np.random.poisson(avg_stay_hospital)
            elif np.random.rand() < prob_recovery:
                self.state = 'R'  # Recovered from Infection

        elif self.state == 'H':
            if self.days_in_hospital > 0:
                self.days_in_hospital -= 1
            else:
                if np.random.rand() < prob_ic:
                    self.state = 'IC'  # Move to Intensive Care
                    self.days_in_ic = np.random.poisson(avg_stay_ic)
                elif np.random.rand() < prob_recovery:
                    self.state = 'R'  # Recovered from Hospitalization

        elif self.state == 'IC':
            if self.days_in_ic > 0:
                self.days_in_ic -= 1
            else:
                if np.random.rand() < prob_death:
                    self.state = 'D'  # Deceased
                elif np.random.rand() < prob_recovery:
                    self.state = 'R'  # Recovered from IC

def simulate(population_size, prob_infection, new_prob_infection_factor, day_new_prob_infection_factor, prob_hospitalization, prob_ic, prob_recovery, prob_death, avg_stay_hospital, avg_stay_ic, steps):
    
   
    # Initialize population of agents
    population = [Agent() for _ in range(population_size)]
    # Infect a few initial agents
    for agent in np.random.choice(population, int(0.01 * population_size), replace=False):
        agent.state = 'I'

    # Track state counts
    results = {'S': [], 'I': [], 'H': [], 'IC': [], 'R': [], 'D': [], 'Hospital_Occupation': [], 'IC_Occupation': []}
    placeholder=st.empty()
    for _ in range(steps):
        placeholder.info(f"{_+1}/{steps}")
        # Spread infection
        infected_agents = [agent for agent in population if agent.state == 'I']
        for agent in population:
            if agent.state == 'S':
                for infected in infected_agents:
                    if steps >= day_new_prob_infection_factor:
                        prob_infection = prob_infection * new_prob_infection_factor
                    agent.infect(prob_infection)

        # Update states of all agents
        for agent in population:
            agent.update_state(prob_hospitalization, prob_ic, prob_recovery, prob_death, avg_stay_hospital, avg_stay_ic)

        # Record the counts of each state
        counts = {state: sum(1 for agent in population if agent.state == state) for state in ['S', 'I', 'H', 'IC', 'R', 'D']}
        
        # Calculate hospital and IC occupation
        hospital_occupation = counts['H']
        ic_occupation = counts['IC']

        # Update results dictionary
        for key in results:
            if key in counts:
                results[key].append(counts[key])
        results['Hospital_Occupation'].append(hospital_occupation)
        results['IC_Occupation'].append(ic_occupation)
    placeholder.empty()
    return results
def main():
    # RIVM 2021
    names =  ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    ifr_ =   [2.04658E-06, 3.78694E-06, 1.76088E-05, 5.45016E-05, 0.000156108, 0.000558534, 0.002271095, 0.009964733, 0.048248607 ]
    h_  =   [0.0015, 0.0001, 0.0002, 0.0007, 0.0013, 0.0028, 0.0044, 0.0097, 0.0107] # I -> H
    i1_ =   [0.0000, 0.0271, 0.0422, 0.0482, 0.0719, 0.0886, 0.0170, 0.0860, 0.0154] # H-> IC
    i2_ =   [0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0531, 0.0080, 0.0367, 0.0356] # IC-> H
    d_  =   [0.0003, 0.0006, 0.0014, 0.0031, 0.0036, 0.0057, 0.0151, 0.0327, 0.0444] # H-> D
    dic_ =  [0.0071, 0.0071, 0.0071, 0.0071, 0.0071, 0.0090, 0.0463, 0.0225, 0.0234] # IC -> D
    dhic_ = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0010, 0.0040, 0.0120, 0.0290] # IC -> h -> D
    r_    = [0.1263, 0.1260, 0.1254, 0.1238, 0.1234, 0.1215, 0.1131, 0.0976, 0.0872] # recovery rate from hospital (before IC)
    ric_  = [0.0857, 0.0857, 0.0857, 0.0857, 0.0857, 0.0821, 0.0119, 0.0567, 0.0550] # recovery rate from hospital (after IC)


    def calculate_weighted(x_):
        N =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
        weighted_x = sum(n * x for n, x in zip(N, x_)) / sum(N)
        return weighted_x
    

    # # Simulation parameters
    # population_size = 50000
    # new_prob_infection_factor = 0.9
    # day_new_prob_infection_factor = 10
    
    # prob_infection = 0.061         # Probability of infection from a single contact (RIVM 2021 = 0.00061)
    # prob_recovery = 0.2          # General recovery probability - 5 dagen ziek
    # avg_stay_hospital = 10         # Average days in hospital (Klinkenberg, 2023)
    # avg_stay_ic = 14              # Average days in ICU (Klinkenberg(2023))
    # steps = 100                   # Number of steps in the simulation
    

    # Input parameters
    
    population_size = st.sidebar.number_input("Population Size", value=50000, min_value=1000)
    new_prob_infection_factor = st.sidebar.number_input("New Probability Infection Factor", value=0.9, min_value=0.0, max_value=10.0)
    day_new_prob_infection_factor = st.sidebar.number_input("Day New Probability Infection", value=10, min_value=1)
    prob_infection = st.sidebar.number_input("Probability of Infection from a Single Contact", value=0.061, min_value=0.0, max_value=1.0)
    prob_recovery = st.sidebar.number_input("General Recovery Probability", value=0.2, min_value=0.0, max_value=1.0)
    avg_stay_hospital = st.sidebar.number_input("Average Days in Hospital", value=10, min_value=1)
    avg_stay_ic = st.sidebar.number_input("Average Days in ICU", value=14, min_value=1)
    steps = st.sidebar.number_input("Number of Steps in the Simulation", value=100, min_value=10)
    if day_new_prob_infection_factor>=steps:
        st.error(f"Correction can not be after end of simulation - {steps}")
        st.stop()
    prob_hospitalization = calculate_weighted(h_)    # Probability of hospitalization for infected agents
    prob_ic = calculate_weighted(i1_)                # Probability of ICU admission from hospital
    prob_death = calculate_weighted(dic_)           # Probability of death from ICU
    
    # RIVM 2021 = The expected outcome of COVID-19 vaccination strategies, p.69
    # Klinkenberg, 2023 = Projecting COVID-19 intensive care admissions in the Netherlands for policy advice: 
    #                     February 2020 to January 2021, p.5
    # Run simulation
    results = simulate(population_size, prob_infection,  new_prob_infection_factor, day_new_prob_infection_factor,prob_hospitalization, prob_ic, prob_recovery, prob_death, avg_stay_hospital, avg_stay_ic, steps)

    # Plot results using Plotly
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("SIR Model with Hospitalization and IC", "Hospital and IC Occupation Over Time"))

    # First subplot: S, I, H, IC, R, D counts
    for state in ['I', 'H', 'IC', 'R', 'D']:
        fig.add_trace(go.Scatter(x=list(range(steps)), y=results[state], mode='lines', name=state), row=1, col=1)

    # Second subplot: Hospital and IC Occupation
    fig.add_trace(go.Scatter(x=list(range(steps)), y=results['Hospital_Occupation'], mode='lines', name='Hospital Occupation', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(steps)), y=results['IC_Occupation'], mode='lines', name='IC Occupation', line=dict(color='red')), row=2, col=1)

    # Update layout
    fig.update_layout(title_text="Agent-Based SIR Model with Hospital and IC Occupation")
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="Number of Agents", row=1, col=1)
    fig.update_yaxes(title_text="Occupation Count", row=2, col=1)

    st.plotly_chart(fig)

 
if __name__ == "__main__":
   
    print(
        f"-----------------------------------{datetime.now()}-----------------------------------------------------"
    )

    main()
    
