import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/

# Total population, N.
N = 17500000
   
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 10000 ,0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
C0 = I0

days = 300

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
# β describes the effective contact rate of the disease: 
# an infected individual comes into contact with βN other 
# individuals per unit time (of which the fraction that are
# susceptible to contracting the disease is S/N).
# 1/gamma is recovery rate in days 

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article

gamma = 1/4
Rt0 = 1.13
# Generation time, time unit (day) and a list for the sliding R-number
Tg = 4
d = 1

# reproductionrate = beta / gamma
beta = Rt0*gamma/(S0/N)

print ("beta : "+  str(beta) + "/ gamma : " + str(gamma))

# A grid of time points (in days)
t = np.linspace(0, days, days)


slidingR=[]
slidingR.append(Rt0)

# The SIR model differential equations. Added dCdt to calculate number of cases
def deriv(y, t, N, beta, gamma):
    S, I, C, R = y
    dSdt = -beta * S * I / N  
    dIdt = beta * S * I / N - gamma * I
    dCdt = beta * S * I / N 
    dRdt = gamma * I
    
    return dSdt, dIdt, dRdt, dCdt

# Initial conditions vector
y0 = S0, I0, C0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, C, R  = ret.T

t = np.linspace(0, days, days)

newC=[]
newC.append(C[0])
Rt0_ = Rt0
Rgroeimodel = []
Rgroeimodel.append (Rt0)

newCgroeimodel = []
newCgroeimodel.append(C[0])
for time in range(1,days):
    newC.append(C[time]-C[time-1])
    f = (1-(I[time]/N))
    Rgroeimodel.append(Rgroeimodel[time-1]*f) 
    newCgroeimodel.append(C[time-1]*(Rgroeimodel[time]**(1/Tg)))

    slidingR_= (newC[time]/newC[time-1])**(Tg/d)   
    slidingR.append(slidingR_)


# Plot the data on three separate curves for S(t), I(t) and R(t)
fig2a = plt.figure(facecolor='w')
ax = fig2a.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, C, 'yellow', alpha=0.5, lw=2, label='Cases')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number (x1000)')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

# Show plot for the sliding R-number
fig2b = plt.figure(facecolor='w')
ax = fig2b.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, slidingR, 'b', alpha=0.5, lw=2, label='R number')
ax.set_xlabel('Time (days)')
ax.set_ylabel('R number')
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
#plt.show()

# Show plot for the sliding R-number
fig2c = plt.figure(facecolor='w')
ax = fig2c.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, slidingR, 'g', alpha=0.5, lw=2, label='R number')
ax.plot(t, newC, 'b', alpha=0.5, lw=2, label='newC')
#ax.plot(t, newCgroeimodel, 'r', alpha=0.5, lw=2, label='newCgroeimodel')

ax.set_xlabel('Time (days)')
ax.set_ylabel('Cases')
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
plt.show()

for x in range (0,10):
    print (str (x) + "- "+ str(slidingR[x]))

#print (slidingR)
# empty slidingR-list
print ("attack rate : " + str(100*C[days-1]/N)+ " %")
slidingR=[]
#print (newC)
