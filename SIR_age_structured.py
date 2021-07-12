import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def func(t, state, N_y, N_m, N_o, beta_y, gamma_y, beta_m, gamma_m, beta_o, gamma_o):
    # S, I, R values assigned from vector
    S_y, I_y, R_y, S_m, I_m, R_m, S_o, I_o, R_o = state

    # differential equations
    dS_ydt = -beta_y * S_y * I_y / N_y
    dI_ydt = beta_y * S_y * I_y / N_y - gamma_y * I_y
    dR_ydt = gamma_y * I_y

    dS_mdt = -beta_m * S_m * I_m / N_m
    dI_mdt = beta_m * S_m * I_m / N_m - gamma_m * I_m
    dR_mdt = gamma_m * I_m

    dS_odt = -beta_o * S_o * I_o / N_o
    dI_odt = beta_o * S_o * I_o / N_o - gamma_o * I_o
    dR_odt = gamma_o * I_o

    return dS_ydt, dI_ydt, dR_ydt, dS_mdt, dI_mdt, dR_mdt, dS_odt, dI_odt, dR_odt


# contact rate
beta_y = 0.5
beta_m = 0.3
beta_o = 0.15

# mean recovery rate
gamma_y = 0.1
gamma_m = 0.25
gamma_o = 0.5
N_y = 300
N_m = 300
N_o = 300

S0_y, I0_y, R0_y, S0_m, I0_m, R0_m, S0_o, I0_o, R0_o = 298, 2, 0, 290, 10, 0, 280, 20, 0

# initial conditions vector
y0 = [S0_y, I0_y, R0_y, S0_m, I0_m, R0_m, S0_o, I0_o, R0_o]
# 200 evenly spaced values (representing days)


p = (N_y, N_m, N_o, beta_y, beta_m, beta_y, gamma_y, gamma_m, gamma_y)

t_span = (0.0, 200.0)
t = np.arange(0.0, 200.0, 1.0)

result_odeint = odeint(func, y0, t, p, tfirst=True)
result_solve_ivp = solve_ivp(func, t_span, y0, args=p)

fig = plt.figure()
ax = fig.add_subplot(4, 2, 1)


ax.plot(t, result_odeint[:, 0], "black", lw=1.5, label="Susceptible")
ax.plot(t, result_odeint[:, 1], "orange", lw=1.5, label="Infected")
ax.plot(t, result_odeint[:, 2], "blue", lw=1.5, label="Recovered")

ax.set_title("odeint young")

ax = fig.add_subplot(4, 2, 2)

ax.plot(result_solve_ivp.y[0, :], "black", lw=1.5, label="Susceptible")
ax.plot(result_solve_ivp.y[1, :], "orange", lw=1.5, label="Infected")
ax.plot(result_solve_ivp.y[2, :], "blue", lw=1.5, label="Recovered")
ax.set_title("solve_ivp young")


ax = fig.add_subplot(4, 2, 3)


ax.plot(t, result_odeint[:, 3], "black", lw=1.5, label="Susceptible")
ax.plot(t, result_odeint[:, 4], "orange", lw=1.5, label="Infected")
ax.plot(t, result_odeint[:, 5], "blue", lw=1.5, label="Recovered")

ax.set_title("odeint mid")

ax = fig.add_subplot(4, 2, 4)

ax.plot(result_solve_ivp.y[3, :], "black", lw=1.5, label="Susceptible")
ax.plot(result_solve_ivp.y[4, :], "orange", lw=1.5, label="Infected")
ax.plot(result_solve_ivp.y[5, :], "blue", lw=1.5, label="Recovered")
ax.set_title("solve_ivp mid")


ax = fig.add_subplot(4, 2, 5)


ax.plot(t, result_odeint[:, 6], "black", lw=1.5, label="Susceptible")
ax.plot(t, result_odeint[:, 7], "orange", lw=1.5, label="Infected")
ax.plot(t, result_odeint[:, 8], "blue", lw=1.5, label="Recovered")

ax.set_title("odeint old")

ax = fig.add_subplot(4, 2, 6)

ax.plot(result_solve_ivp.y[6, :], "black", lw=1.5, label="Susceptible")
ax.plot(result_solve_ivp.y[7, :], "orange", lw=1.5, label="Infected")
ax.plot(result_solve_ivp.y[8, :], "blue", lw=1.5, label="Recovered")
ax.set_title("solve_ivp old")

S_tot_ode = result_odeint[:, 0] + result_odeint[:, 3] + result_odeint[:, 6]
I_tot_ode = result_odeint[:, 1] + result_odeint[:, 4] + result_odeint[:, 7]
R_tot_ode = result_odeint[:, 2] + result_odeint[:, 5] + result_odeint[:, 8]


ax = fig.add_subplot(4, 2, 7)


ax.plot(t, S_tot_ode, "black", lw=1.5, label="Susceptible")
ax.plot(t, I_tot_ode, "orange", lw=1.5, label="Infected")
ax.plot(t, R_tot_ode, "blue", lw=1.5, label="Recovered")

ax.set_title("odeint Totaal")

ax = fig.add_subplot(4, 2, 8)


S_tot_ivp = (
    result_solve_ivp.y[0, :] + result_solve_ivp.y[3, :] + result_solve_ivp.y[6, :]
)
I_tot_ivp = (
    result_solve_ivp.y[1, :] + result_solve_ivp.y[4, :] + result_solve_ivp.y[7, :]
)
R_tot_ivp = (
    result_solve_ivp.y[2, :] + result_solve_ivp.y[5, :] + result_solve_ivp.y[8, :]
)


ax.plot(S_tot_ivp, "black", lw=1.5, label="Susceptible")
ax.plot(I_tot_ivp, "orange", lw=1.5, label="Infected")
ax.plot(R_tot_ivp, "blue", lw=1.5, label="Recovered")
ax.set_title("solve_ivp Totaal")
plt.show()
