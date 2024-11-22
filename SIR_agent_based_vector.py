import numpy as np
import plotly.graph_objects as go
import streamlit as st
import time
def calculate_weighted(x_, what):
    N = [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
    weighted_x = sum(n * x for n, x in zip(N, x_)) / sum(N)
    st.sidebar.write(f"{what} - {weighted_x}")
    return weighted_x
def simulate_sir(population_size, initial_infected, prob_infection, prob_recovery, 
                 prob_hospitalization, prob_ic, prob_icd, prob_hd, prob_hr, prob_icr, avg_stay_hospital, avg_stay_ic, 
                 steps, base_contacts, contact_reduction_day, contact_reduction_factor,startijd):
    
    # Pre-allocate arrays for all states and counters
    states = np.zeros(population_size, dtype=np.int8)  # Using int8 instead of int to reduce memory
    states[:initial_infected] = 1
    
    days_in_hospital = np.zeros(population_size, dtype=np.int8)
    days_in_ic = np.zeros(population_size, dtype=np.int8)
    
    # Pre-allocate result arrays
    results = np.zeros((7, steps), dtype=np.int32)  # All counts in one array
    
    # Pre-compute contact rates
    contact_rates = np.full(steps, base_contacts, dtype=np.int32)
    contact_rates[contact_reduction_day:] = int(base_contacts * contact_reduction_factor)
    
    placeholder = st.empty()
    
    for step in range(steps):
        if (step + 1) % 10 == 0:
            placeholder.info(f"days: {step + 1}/{steps} {startijd - int(time.time())} sec.")
            
        # Count current states (vectorized operation)
        current_counts = np.bincount(states, minlength=6)
        results[0:6, step] = current_counts
        
        infected_indices = np.nonzero(states == 1)[0]
        n_infected = len(infected_indices)
        
        if n_infected > 0:
            current_contacts = contact_rates[step]
            
            # Generate all contacts at once
            contact_indices = np.random.choice(population_size, 
                                            size=n_infected * current_contacts, 
                                            replace=True)
            
            # Check which contacts are susceptible
            susceptible_mask = states[contact_indices] == 0
            
            # Generate infection probabilities only for susceptible contacts
            if susceptible_mask.any():
                infection_mask = (np.random.random(len(contact_indices)) < prob_infection) & susceptible_mask
                new_infections = np.unique(contact_indices[infection_mask])
                states[new_infections] = 1
                results[6, step] = len(new_infections)
        
        # Generate all random numbers at once
        rng = np.random.random((5, population_size))
        
        # Process state transitions (all vectorized operations)
        infected_mask = states == 1
        hospital_mask = states == 3
        ic_mask = states == 4
        
        # Recoveries from infection
        recoveries = infected_mask & (rng[0] < prob_recovery)
        states[recoveries] = 2
        
        # Hospitalizations
        to_hospital = infected_mask & (rng[1] < prob_hospitalization)
        states[to_hospital] = 3
        days_in_hospital[to_hospital] = avg_stay_hospital
        
        # Update hospital stays
        days_in_hospital[hospital_mask] -= 1
        
        # Deaths from hospital
        hospital_deaths = hospital_mask & (rng[2] < prob_hd)
        states[hospital_deaths] = 5
        
        # ICU transfers
        to_ic = hospital_mask & (rng[3] < prob_ic) & (days_in_hospital <= 3)
        states[to_ic] = 4
        days_in_ic[to_ic] = avg_stay_ic
        
        # Update ICU stays
        days_in_ic[ic_mask] -= 1
        
        # ICU outcomes
        ic_deaths = ic_mask & (rng[4] < prob_icd)
        ic_recoveries = ic_mask & ((days_in_ic <= 0) | (rng[4] < prob_icr))
        
        states[ic_deaths] = 5
        states[ic_recoveries] = 2
        
        # Hospital recoveries
        hospital_recoveries = hospital_mask & ((days_in_hospital <= 0) | (rng[0] < prob_hr))
        states[hospital_recoveries] = 2
    
    placeholder.empty()
    
    # Return all results
    return tuple(results)


def calculate_Reff(new_infections, d, T_g):
    """_summary_

    Args:
        new_infections (list): list with new infections
        d (int): day which you compare the value with 
        T_g (int or float): Generation time

    Returns:
        list: list with Reff
    """

    
    R_t = []
    for _ in range(d):
        R_t.append(None)
    for t in range(d, len(new_infections)):
        X_t = new_infections[t]  # New infections at time t
        X_t_minus_d = new_infections[t - d]  # New infections at time t - d

        if X_t_minus_d > 0:  # Avoid division by zero
            R_t_value = (X_t / X_t_minus_d) ** (T_g/d)
            R_t.append(R_t_value)
        else:
            R_t.append(np.nan)  # Set NaN if denominator is zero

    return R_t

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def create_plots(results_no_red, results_red):
    susceptible_no_red = results_no_red[0]
    infected_no_red = results_no_red[1]
    recovered_no_red = results_no_red[2]
    hospitalized_no_red = results_no_red[3]
    ic_no_red = results_no_red[4]
    death_no_red = results_no_red[5]
    new_infections_over_time_no_red = results_no_red[6]

    susceptible_red = results_red[0]
    infected_red = results_red[1]
    recovered_red = results_red[2]
    hospitalized_red = results_red[3]
    ic_red = results_red[4]
    death_red = results_red[5]
    new_infections_over_time_red = results_red[6]

    # Compute R_eff
    Tg=5
    d = 5 # only with 5 I get reliable results TODO
    Reff_no_red = calculate_Reff(moving_average(new_infections_over_time_no_red, window_size=1) ,d, Tg)
    Reff_red = calculate_Reff(moving_average(new_infections_over_time_red, window_size=1) ,d, Tg)

   
    # Create plots
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=infected_red, mode='lines', name='Infected reduction'))
    fig.add_trace(go.Scatter(y=infected_no_red, mode='lines', name='Infected no reduction'))
    
    fig.update_layout(title='SIR Model - Infections',
                     xaxis_title='Time Steps',
                     yaxis_title='Number of Individuals')
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=hospitalized_no_red, mode='lines', name='Hospitalized no reduction'))
    fig.add_trace(go.Scatter(y=ic_no_red, mode='lines', name='IC no reduction'))
    fig.add_trace(go.Scatter(y=death_no_red, mode='lines', name='Death no reduction'))
 
    fig.add_trace(go.Scatter(y=hospitalized_red, mode='lines', name='Hospitalized reduction'))
    fig.add_trace(go.Scatter(y=ic_red, mode='lines', name='IC reduction'))
    fig.add_trace(go.Scatter(y=death_red, mode='lines', name='Death reduction'))


    fig.update_layout(title='SIR Model - Hospital, ICU, and Deaths',
                     xaxis_title='Time Steps',
                     yaxis_title='Number of Individuals')
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=susceptible_no_red, mode='lines', name='Susceptible no reduction'))
    fig.add_trace(go.Scatter(y=recovered_no_red, mode='lines', name='Recovered no reduction'))
    fig.add_trace(go.Scatter(y=new_infections_over_time_no_red, mode='lines', name='New infections over time no reduction'))

    fig.add_trace(go.Scatter(y=susceptible_red, mode='lines', name='Susceptible reduction'))
    fig.add_trace(go.Scatter(y=recovered_red, mode='lines', name='Recovered reduction'))
    fig.add_trace(go.Scatter(y=new_infections_over_time_red, mode='lines', name='New infections over time reduction'))
    fig.update_layout(title='SIR Model - Overall Progression',
                     xaxis_title='Time Steps',
                     yaxis_title='Number of Individuals')
    st.plotly_chart(fig)


    # Create plots
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=Reff_red, mode='lines', name='R_eff reduction'))
    fig.add_trace(go.Scatter(y=Reff_no_red, mode='lines', name='R_eff no reduction'))
    fig.add_trace(go.Scatter(x=[0, len(Reff_no_red)], y=[1, 1], mode='lines', 
                          line=dict(color='red', dash='dash')))

    fig.update_layout(title='SIR Model - R_eff',
                     xaxis_title='Time Steps',
                     yaxis_title='R_eff')
    st.plotly_chart(fig)

def main():
    s1 = int(time.time())
    # RIVM 2021
    population_size, initial_infected, base_contacts, contact_reduction_day, \
        contact_reduction_factor, prob_infection, prob_recovery, prob_hospitalization, \
        prob_ic, prob_hd, prob_hr, prob_icr, avg_stay_hospital, avg_stay_ic, steps, prob_icd = get_parameters()

    
    # Run simulation
    results_red = simulate_sir(
        population_size, initial_infected, prob_infection, prob_recovery,
        prob_hospitalization, prob_ic, prob_icd, prob_hd, prob_hr, prob_icr,
        avg_stay_hospital, avg_stay_ic, steps, base_contacts,
        contact_reduction_day, contact_reduction_factor,s1
    )

    results_no_red = simulate_sir(
        population_size, initial_infected, prob_infection, prob_recovery,
        prob_hospitalization, prob_ic, prob_icd, prob_hd, prob_hr, prob_icr,
        avg_stay_hospital, avg_stay_ic, steps, base_contacts,
        contact_reduction_day,1,s1
    )
    create_plots(results_no_red, results_red)
    

    s2 = int(time.time())
    st.info(f"Needed time {s2-s1} seconds")

def get_parameters():
    names = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    ifr_ = [2.04658E-06, 3.78694E-06, 1.76088E-05, 5.45016E-05, 0.000156108, 0.000558534, 0.002271095, 0.009964733, 0.048248607]
    h_ = [0.0015, 0.0001, 0.0002, 0.0007, 0.0013, 0.0028, 0.0044, 0.0097, 0.0107]
    ic1_ = [0.0000, 0.0271, 0.0422, 0.0482, 0.0719, 0.0886, 0.0170, 0.0860, 0.0154]
    i2_ = [0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0531, 0.0080, 0.0367, 0.0356]
    hd_ = [0.0003, 0.0006, 0.0014, 0.0031, 0.0036, 0.0057, 0.0151, 0.0327, 0.0444]
    icd_ = [0.0071, 0.0071, 0.0071, 0.0071, 0.0071, 0.0090, 0.0463, 0.0225, 0.0234]
    dhic_ = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0010, 0.0040, 0.0120, 0.0290]
    hr_ = [0.1263, 0.1260, 0.1254, 0.1238, 0.1234, 0.1215, 0.1131, 0.0976, 0.0872]
    icr_ = [0.0857, 0.0857, 0.0857, 0.0857, 0.0857, 0.0821, 0.0119, 0.0567, 0.0550]

    # Parameters
    population_size = st.sidebar.number_input("Population Size", value=170000, min_value=1000)
    initial_infected = st.sidebar.number_input("Initial infected", 0, population_size, int(population_size/20))
    base_contacts = st.sidebar.number_input("Initial contact rate per infected individual", 0, 10000, 10)
    contact_reduction_day = st.sidebar.number_input("Day to start reducing contacts", 0, 100000, 10)
    contact_reduction_factor = 1-(st.sidebar.number_input("percentage reduction", 0, 100, 10)/100)

    prob_infection = st.sidebar.number_input("Prob infection", value=0.04000, min_value=0.0, max_value=1.0, format="%.8f")
    prob_recovery = st.sidebar.number_input("General Recovery Probability", value=0.2, min_value=0.0, max_value=1.0)
    avg_stay_hospital = st.sidebar.number_input("Average Days in Hospital", value=10, min_value=1)
    avg_stay_ic = st.sidebar.number_input("Average Days in ICU", value=14, min_value=1)
    steps = st.sidebar.number_input("Number of Steps in the Simulation", value=100, min_value=10)
    prob_hospitalization = calculate_weighted(h_, "ih")
    
    prob_ic = calculate_weighted(ic1_, "iic")
    prob_hd = calculate_weighted(hd_, "hd")
    prob_hr = calculate_weighted(hr_, "hr")
    prob_icr = calculate_weighted(icr_, "icr")
    prob_icd = calculate_weighted(icd_, "icd")

    ifr = calculate_weighted(ifr_, "ifr")
    return population_size,initial_infected,base_contacts,contact_reduction_day,contact_reduction_factor,prob_infection,prob_recovery,prob_hospitalization,prob_ic,prob_hd,prob_hr,prob_icr,avg_stay_hospital,avg_stay_ic,steps,prob_icd

if __name__ == "__main__":
    main()