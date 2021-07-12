import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

def func(t, state, *argv):
    """The function with the formula

    Args:
        t (?): timepoints
        state (?): Numbers of S, I and R
        argv : groupsizes, beta's and gamma's
    Returns:
        [?]: the differences in each step (dSdt+ dIdt + dRdt)
    """
    lijst = list(state)
    arguments = [x for x in argv]

    S,I,R, N,beta,gamma, dSdt, dIdt, dRdt =[],[],[], [],[],[], [],[],[]
    for i in range (len(lijst)):
        if i < x:
            S.append(lijst[i])
        if i>=x and i < 2*x:
            I.append(lijst [i])
        if i>=2*x and i < 3*x:
            R.append(lijst[i])

    for i in range (len(arguments)):
        if i < x:
            N.append(arguments[i])
        if i>=x and  i < 2*x:
            beta.append(arguments [i])
        if i >= 2*x and i < 3*x:
            gamma.append(arguments[i])

    for i in range(x):
        dSdt.append( -beta[i] * S[i] * I[i] / N[i])
        dIdt.append( beta[i] * S[i] * I[i] / N[i] - gamma[i] * I[i])
        dRdt.append( gamma[i] * I[i])
    to_return = dSdt+ dIdt + dRdt
    return to_return

def draw_graph (result_odeint,result_solve_ivp, names, beta, gamma, t):
    """Draws graphs with subgraphs of each agegroup and total

    Args:
        result_odeint (?): result of the ODEint
        result_solve_ivp (?): result of the Solve_ivp
        names (list): names of the groups for the legenda
        beta (list): for the legenda
        gamma (list): for the legenda
        t (list): timevalues, for the x-axis
    """
    S_tot_ivp, I_tot_ivp, R_tot_ivp, S_tot_odeint, I_tot_odeint, R_tot_odeint = 0.0,0.0,0.0, 0.0,0.0,0.0
    fig = plt.figure()
    graph_index = 1 # TOFIX : make an automatic counter, depending on i
    for i in range (x):
        S_tot_ivp += result_solve_ivp.y[i, :]
        I_tot_ivp += result_solve_ivp.y[3+i, :]
        R_tot_ivp += result_solve_ivp.y[6+i, :]

        S_tot_odeint +=result_odeint[:, 0]
        I_tot_odeint += result_odeint[:, 3+i]
        R_tot_odeint += result_odeint[:, 6+i]

        ax = fig.add_subplot(x+1, 2,graph_index)
        ax.plot(result_solve_ivp.y[i, :], "black", lw=1.5, label="Susceptible")
        ax.plot(result_solve_ivp.y[3+i, :], "orange", lw=1.5, label="Infected")
        ax.plot(result_solve_ivp.y[6+i, :], "blue", lw=1.5, label="Recovered")

        graph_index +=1
        ax.set_title(f"solve_ivp { names[i]} | beta = {beta[i]} / gamma = {gamma[i]}")

        ax = fig.add_subplot(x+1, 2,graph_index)
        ax.plot(t, result_odeint[:, i], "black", lw=1.5, label="Susceptible")
        ax.plot(t, result_odeint[:, 3+i], "orange", lw=1.5, label="Infected")
        ax.plot(t, result_odeint[:, 6+i], "blue", lw=1.5, label="Recovered")
        ax.set_title(f"solve_odeint { names[i]} | beta = {beta[i]} / gamma = {gamma[i]}")
        graph_index +=1

    # TOTALS
    ax = fig.add_subplot(x+1, 2, x*2+1)
    ax.plot(S_tot_ivp, "black", lw=1.5, label="Susceptible")
    ax.plot(I_tot_ivp, "orange", lw=1.5, label="Infected")
    ax.plot(R_tot_ivp, "blue", lw=1.5, label="Recovered")
    ax.set_title("solve_ivp Totaal")

    ax = fig.add_subplot(x+1, 2, x*2+2)
    ax.plot(S_tot_odeint, "black", lw=1.5, label="Susceptible")
    ax.plot(I_tot_odeint, "orange", lw=1.5, label="Infected")
    ax.plot(R_tot_odeint, "blue", lw=1.5, label="Recovered")
    ax.set_title("solve_odeint Totaal")

    plt.legend()
    plt.show()

def main():
    global x

    names = ["young", "mid", "old"]
    beta = [1.1,0.5,0.15] # contact rate
    gamma = [1/4,1/10,1/20] # mean recovery rate (1/recovery days)
    x = len (names) # number of agegroups

    N = [300,300,300]
    S0 = [298,290,280]
    I0 = [2,10,20]
    R0 = [0,0,0]

    y0 = tuple(S0 + I0 + R0)
    p = tuple(N + beta + gamma)
    n = 101 # number of time points
    # time points
    t = np.linspace(0, 100, n)
    t_span = (0.0, 100.0)

    result_odeint = odeint(func, y0, t, p, tfirst=True)
    result_solve_ivp = solve_ivp(func, t_span, y0, args=p,  t_eval=t)

    draw_graph (result_odeint,result_solve_ivp, names, beta, gamma, t)


if __name__ == '__main__':
    main()
