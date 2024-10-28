import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# SIR MODEL
# Original: https://github.com/dimgeo/Data-Morgana/blob/main/dm.ijs 
# based on https://www.sciencedirect.com/science/article/pii/S1755436522000214#

# translated to Python by ClaudeAI https://claude.ai/chat/3f325f38-3c82-422b-aac8-8663026581cb

class EpidemiologicalModel:
    def __init__(self):
        # Model parameters
        self.delta = 1/3.69
        self.eta_m = 1/3.48
        self.eta_ds = 1/3.48
        self.eta_dc = 1/3.48
        self.eta_c = 1/42
        self.eta_s = 1/28
        self.mu = 1/79.10
        self.tau1 = 0.1
        self.tau2 = 0.2
        self.tau3 = 0.1
        
        # Progression factors
        self.fsa = 0.01
        self.fs = np.array([0.1, 0.5, 1, 1.2, 2.3, 4.5, 7.8, 27.6]) * self.fsa
        self.fca = 0.002
        self.fc = np.array([0.2, 0.3, 1, 1.8, 4.7, 10.6, 13.6, 8.7]) * self.fca
        self.aha = 0.0002
        self.alpha = np.array([0.1, 0.3, 1, 3.0, 10.0, 45.0, 120.0, 505.0]) * self.aha
        self.fm = 1 - self.fs + self.fc
        
        self.beta = 1.52
        self.sigma = np.array([0.53, 0.57, 0.58, 0.62, 0.70, 0.52, 0.50, 0.47])
        self.lambda_ = np.array([0.14213] * 8)
        self.ksi = 0
        self.DEm = 0.1
        self.DEsc = 0.1
        
        # Treatment efficacy parameters
        self.DEpostep = 0.3
        self.DEprep = 0.1
        self.DEp = 0.1

    def sirseci(self, state):
        """Treatment model for severe and critical cases"""
        S, E, Im, Rm, Is, Ic, Ds, Dc, Rsc, Is_t, Ic_t, Ds_t, Dc_t, Rsc_t = state
        
        # Untreated population
        S1 = S - (self.lambda_ + self.mu + self.ksi)*S + self.DEsc * (self.eta_ds*Is_t + self.eta_dc*Ic_t)
        E1 = E + self.lambda_ * S - (self.delta + self.mu + self.ksi) * E
        Im1 = Im + self.fm * self.delta * E - (self.eta_m + self.mu + self.ksi) * Im
        Rm1 = Rm + (self.eta_m*Im + self.eta_s*Ds + self.eta_c*Dc) - (self.mu + self.ksi)* Rm
        Is1 = Is + (self.fs * self.delta * E) - (self.eta_ds + self.mu + self.ksi + self.tau1)*Is
        Ic1 = Ic + (self.fc * self.delta * E) - (self.eta_dc + self.mu + self.ksi + self.tau1)*Ic
        Ds1 = Ds + self.eta_ds*Is - (self.eta_s + self.mu + self.ksi)*Ds
        Dc1 = Dc + self.eta_dc*Ic - (self.eta_c + self.mu + self.ksi)*Dc
        Rsc1 = Rsc + (self.eta_s * Ds + self.eta_c*Dc) - (self.mu + self.ksi)*Rsc
        
        # Treated population
        Is_t1 = Is_t + (self.tau1 * Is) - (self.eta_ds + self.ksi + self.mu)*Is_t
        Ic_t1 = Ic_t + (self.tau1 * Ic) - (self.eta_dc + self.ksi + self.mu)*Ic_t
        Ds_t1 = Ds_t + ((1-self.DEsc)*self.eta_ds*Is_t) - (self.eta_s+self.mu+self.ksi)*Ds_t
        Dc_t1 = Dc_t + ((1-self.DEsc)*self.eta_dc*Ic_t) - (self.eta_c+self.mu+self.ksi + self.alpha)*Dc_t
        Rsc_t1 = Rsc_t + (self.eta_s*Ds_t + ((self.DEm*self.alpha) + self.eta_c)*Dc_t) - (self.mu+self.ksi)*Rsc_t
        
        return np.array([S1, E1, Im1, Rm1, Is1, Ic1, Ds1, Dc1, Rsc1, Is_t1, Ic_t1, Ds_t1, Dc_t1, Rsc_t1])

    def sirpost(self, state):
        """Post-exposure treatment model"""
        S, E, Im, Is, Ic, Ds, Dc, R, E_t, Im_t, Is_t, Ic_t, Ds_t, R_t = state
        
        # Untreated population
        S1 = S - (self.lambda_ + self.mu + self.ksi)*S + (self.DEpostep * self.delta * E_t)
        E1 = E + (self.lambda_ * S) - (self.delta + self.mu + self.ksi + self.tau2) * E
        Im1 = Im + (self.fm * self.delta * E) - (self.eta_m + self.mu + self.ksi) * Im
        Is1 = Is + (self.fs * self.delta * E) - (self.eta_s + self.mu + self.ksi) * Is
        Ic1 = Ic + (self.fc * self.delta * E) - (self.eta_c + self.mu + self.ksi) * Ic
        Ds1 = Ds + (self.eta_ds * Is) - (self.eta_s + self.mu + self.ksi) * Ds
        Dc1 = Dc + (self.eta_dc * Ic) - (self.eta_c + self.mu + self.ksi + self.alpha) * Dc
        R1 = R + (self.eta_m * Im + self.eta_s * Ds + self.eta_c * Dc) - (self.mu + self.ksi)*R
        
        # Treated population
        E_t1 = E_t + (self.tau2*E) - (self.delta + self.mu + self.ksi)*E_t
        Im_t1 = Im_t + ((1 - self.DEsc + self.DEsc/self.fm) * (1 - self.DEpostep) * self.fm * self.delta * E_t) - (self.mu + self.ksi + self.eta_m/(1 - self.DEp))*Im_t
        Is_t1 = Is_t + ((1 - self.DEsc) * (1 - self.DEpostep) * self.fs * self.delta * E_t) - (self.eta_ds * self.mu + self.ksi) * Is_t
        Ic_t1 = Ic_t + ((1 - self.DEsc) * (1 - self.DEpostep) * self.fc * self.delta * E_t) - (self.eta_dc * self.mu + self.ksi) * Ic_t
        Ds_t1 = Ds_t + (self.eta_ds*Is_t) - (self.eta_s + self.mu + self.ksi)*Ds_t
        R_t1 = R_t + ((self.eta_m/(1 - self.DEp)) * Im_t) + (self.eta_s * Ds_t) - (self.mu + self.ksi)*R_t
        
        return np.array([S1, E1, Im1, Is1, Ic1, Ds1, Dc1, R1, E_t1, Im_t1, Is_t1, Ic_t1, Ds_t1, R_t1])

    def sipre(self, state):
        """Pre-exposure treatment model"""
        S, E, Im, Is, Ic, Ds, Dc, R, T, E_t, Im_t, Is_t, Ic_t, Ds_t, Dc_t, R_t = state
        
        # Untreated population
        S1 = S - (((self.lambda_ + self.mu + self.ksi) - self.tau3) * S)
        E1 = E + (self.lambda_ * S) - (self.delta + self.mu + self.ksi) * E
        Im1 = Im + (self.fm * self.delta * E) - (self.eta_m + self.mu + self.ksi) * Im
        Is1 = Is + (self.fs * self.delta * E) - (self.eta_ds + self.mu + self.ksi) * Is
        Ic1 = Ic + (self.fc * self.delta * E) - (self.eta_dc + self.mu + self.ksi) * Ic
        Ds1 = Ds + (self.eta_ds * Is) - (self.eta_s + self.mu + self.ksi) * Ds
        Dc1 = Dc + (self.eta_dc * Ic) - (self.eta_c + self.mu + self.ksi + self.alpha) * Dc
        R1 = R + (self.eta_m * Im) + (self.eta_s * Ds) + (self.eta_c * Dc) - (self.mu + self.ksi) * R
        
        # Treated population
        T1 = T + (self.tau3 * S) - ((1 - self.DEprep) * self.lambda_ + self.mu + self.ksi) * T
        E_t1 = E_t + ((1 - self.DEprep) * self.lambda_ * T) - (self.delta + self.mu + self.ksi) * E_t
        Im_t1 = Im_t + ((1 - self.DEsc + self.DEsc/self.fm) * self.fm * self.delta * E_t) - (self.mu + self.ksi + self.eta_m/(1 - self.DEp)) * Im_t
        Is_t1 = Is_t + ((1 - self.DEsc) * self.fs * self.delta * E_t) - (self.eta_ds + self.mu + self.ksi) * Is_t
        Ic_t1 = Ic_t + ((1 - self.DEsc) * self.fc * self.delta * E_t) - (self.eta_dc + self.mu + self.ksi) * Ic_t
        Ds_t1 = Ds_t + (self.eta_ds * Is_t) - (self.eta_s + self.mu + self.ksi) * Ds_t
        Dc_t1 = Dc_t + (self.eta_dc * Ic_t) - (self.eta_c + self.mu + self.ksi + self.alpha) * Dc_t
        R_t1 = R_t + (Im_t * self.eta_m/(1 - self.DEp)) + (self.eta_s * Ds_t) + (self.eta_c * Dc_t) - (self.mu + self.ksi) * R_t
        
        return np.array([S1, E1, Im1, Is1, Ic1, Ds1, Dc1, R1, T1, E_t1, Im_t1, Is_t1, Ic_t1, Ds_t1, Dc_t1, R_t1])

    def simulate(self, model_type, initial_state, steps):
        """Run simulation for specified number of steps"""
        results = np.zeros((steps, len(initial_state)))
        results[0] = initial_state
        
        model_func = {
            'sirseci': self.sirseci,
            'sirpost': self.sirpost,
            'sipre': self.sipre
        }[model_type]
        
        for i in range(1, steps):
            results[i] = model_func(results[i-1])
        
        return results

    def plot_results(self, results, model_type):
        """Plot simulation results using Plotly"""
        labels = {
            'sirseci': ['S', 'E', 'Im', 'Rm', 'Is', 'Ic', 'Ds', 'Dc', 'Rsc', 'Is_t', 'Ic_t', 'Ds_t', 'Dc_t', 'Rsc_t'],
            'sirpost': ['S', 'E', 'Im', 'Is', 'Ic', 'Ds', 'Dc', 'R', 'E_t', 'Im_t', 'Is_t', 'Ic_t', 'Ds_t', 'R_t'],
            'sipre': ['S', 'E', 'Im', 'Is', 'Ic', 'Ds', 'Dc', 'R', 'T', 'E_t', 'Im_t', 'Is_t', 'Ic_t', 'Ds_t', 'Dc_t', 'R_t']
        }[model_type]

        # Create subplots with 4x2 layout
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=labels[:8],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # Add traces for each compartment
        for i in range(min(8, results.shape[1])):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    y=results[:, i],
                    name=labels[i],
                    line=dict(width=2),
                    showlegend=True
                ),
                row=row,
                col=col
            )

            # Update y-axes to log scale
            fig.update_yaxes(type="log", row=row, col=col)

        # Update layout
        fig.update_layout(
            height=1000,
            width=1000,
            title_text=f"{model_type.upper()} Model Results",
            showlegend=True,
            template="plotly_white",
        )

        return fig

def run_example():
    """Example usage of the model"""
    # Create model instance
    model = EpidemiologicalModel()

    # Set up initial conditions
    initial_state = np.array([100000] + [10] * 13)  # For sirseci model
    
    # Run simulation
    results = model.simulate('sirseci', initial_state, steps=100)
    
    # Create and show plot
    fig = model.plot_results(results, 'sirseci')
    fig.show()

if __name__ == "__main__":
    run_example()