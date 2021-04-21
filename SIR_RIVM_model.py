# http://web.pdx.edu/~gjay/teaching/mth271_2020/html/09_SEIR_model.html

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

import seaborn; seaborn.set();



# FORMULAS AND VALUES TO MAKE A SIR MODEL FOR THE NETHERLANDS
# SOURCE : RIVM https://www.rivm.nl/sites/default/files/2021-03/Modellingresults%20COVID19%20vaccination%20version1.0%2020210324_0.pdf
# retreived 21st of April 2021
# Copied by René SMIT
# Fouten voorbehoude
#       0-9     10-19   20-29  30-39   40-49   50-59   60-69   70-79  80+
pop_ =      [1756000, 1980000, 2245000, 2176000, 2164000, 2548000, 2141000, 1615000, 839000]
fraction = [0.10055, 0.11338, 0.12855, 0.12460, 0.12391, 0.14590, 0.12260, 0.09248, 0.04804]

h_  =   [0.0015, 0.0001, 0.0002, 0.0007, 0.0013, 0.0028, 0.0044, 0.0097, 0.0107] # I -> H
i1_ =   [0.0000, 0.0271, 0.0422, 0.0482, 0.0719, 0.0886, 0.0170, 0.0860, 0.0154] # H-> IC
i2_ =   [0.0555, 0.0555, 0.0555, 0.0555, 0.0555, 0.0531, 0.0080, 0.0367, 0.0356] # IC-> H
d_  =   [0.0003, 0.0006, 0.0014, 0.0031, 0.0036, 0.0057, 0.0151, 0.0327, 0.0444] # H-> D
dic_ =  [0.0071, 0.0071, 0.0071, 0.0071, 0.0071, 0.0090, 0.0463, 0.0225, 0.0234] # IC -> D
dhic_ = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0010, 0.0040, 0.0120, 0.0290] # IC -> h -> D
r_    = [0.1263, 0.1260, 0.1254, 0.1238, 0.1234, 0.1215, 0.1131, 0.0976, 0.0872] # recovery rate from hospital (before IC)
ric_  = [0.0857, 0.0857, 0.0857, 0.0857, 0.0857, 0.0821, 0.0119, 0.0567, 0.0550] # recovery rate from hospital (after IC)
LE_   = [ 77.89,  67.93,  58.08,  48.48,  38.60,  29.22,  20.52,  12.76,  4.35 ] # Life expectancy
RSI_  = [ 1.000,  3.051,  5.751,  3.538,  3.705,  4.365,  5.688,  5.324,  7.211] # Relative suspceptibility/infectiousness

pop, h, i1, i2, d, dic, dhic, r, ric, LE, RSI = 0,0,0,0,0,0,0,0,0,0,0


for x in range (0, len(fraction)):

    h += h_[x] * fraction[x]
    i1+= i1_[x] * fraction[x]
    i2+= i2_[x] * fraction[x]
    d+= d_[x] * fraction[x]
    dic+= dic_[x] * fraction[x]
    dhic+= dhic_[x] * fraction[x]
    r+= r_[x] * fraction[x]
    ric+= ric_[x] * fraction[x]
    LE+= LE_[x] * fraction[x]
    pop += pop_[x]

global C, N
C, N = 0.1, pop


Shold1d__, Sv1ddt__, Shold2d__, Sv2d__ = 0,0 , 0, 0
E__, Ev1ddt__, Ev2d__ = 50_000,0,0
I__, Iv1d__, Iv2d__ = 50_000, 0,0
H__, Hv1d__, Hv2d__ = 1500, 0,0
IC__, ICv1ddt__, ICv2d__ = 600, 0, 0
HIC__, HICv1ddt__, HICv2d__ = 10, 0,0
D__ = 15_000
R__, Rv1d__, Rv2d__ = 0.148 * pop, 0,0
S__ = pop - E__ - I__ - H__ - IC__ - HIC__ - D__ - R__

R0 = 2.3 # basic reproduction number
Reff = 1.04 # effective reproduction number
beta = 0.00061 # transmission rate
sigma = 0.5 # inverse of latent period
gamma = 0.5 # inverse of infectious period

lamdaa = beta * ( C * (( I__ + Iv1d__ + Iv2d__) / N)) # force of infection
alfa = 3.5/17.5 # rate of vaccination with the first dose
alfa2 = 1/17.4 # UNKNOWN # rate of vaccination with the second dose
delta = 14 # TABLE2 # delay to protection of first dose
delta2 = 14 #TABLE2 # delay of protection of second dose
eta = 0.2 # 1- vacc_eff_first_dose # SEE TABLE 2
eta2 = 0.2 # 1- vacc_eff_sec_dose # # SEE TABLE 2





# name = [ delay to protection, vaccin efficacy] # aantal gezet 18 april
pfizer1 = [14, 92.6] # 2954000
pfizer2 = [7, 94.8]
moderna1 = [14, 89.6] # 323000
moderna2 = [14, 94.1]
astrazenica1 = [21, 58.3] # 1322000
astrazenica2 = [14, 62.1]
janssen = [14, 66.1] # 0

P_ascertainment = 0.18







def calculate(t, y, beta,
                    sigma,
                    gamma,
                    lamdaaa,
                    alfa,
                    alfa2,
                    delta,
                    delta2,
                    eta,
                    eta2,
                    h,
                    i1,
                    i2,
                    d,
                    dic,
                    dhic,
                    r,
                    ric):
    S, Shold1d, Sv1ddt, Shold2d, Sv2d,
    E, Ev1ddt, Ev2d,
    I, Idv1d, Iv2d,
    H, Hv1d, Hv2d,
    IC, ICv1ddt, ICv2d,
    HIC, HICv1ddt, HICv2d,
    D,
    R, Rv1d, Rv2d = y

    return np.array([
                    -lamdaa * S - alfa * S,
                    alfa * S - (1/delta)* Shold1d - lamdaa* Shold1d,
                    (1/delta) * Shold1d - eta * lamda * Sv1d - alfa2 * Sv1d,
                    alfa2 * Sv1d - (1/delta2)*Shold2d - eta * lamdaa * Shold2d,
                    (1/delta2) * Shold2d - eta2 * lamda * Sv2d,

                    lamdaa * (S + Shold1d) - sigma * E,
                    eta * lamda * (Sv1d+Shold2d) - sigma * Ev1d,
                    eta2 * lamdaa * Sv2d - sigma * Ev2d,


                    sigma * E - (gamma + h)*I,
                    sigma * Ev1d - (gamma + h)*Iv1d,
                    sigma * Ev2d- (gamma + h)*Iv2d,


                    h*I - (i1 +d +r) *H,
                    h*Iv1d- (i1+d+r)*Hv1d,
                    h*Iv2d- (i1+d+r)*Hv2,



                    i1*H - (i2+dic)*IC,
                    i1*Hv1d - (i2+dic)*ICv1d,
                    i1*Hv2d - (i2+dic)*ICv2d,

                    i2 * IC - (ric+dhic)*HIC,
                    i2 * ICv1d - (ric+dhic)*HICv1d,
                    i2 * ICv2d - (ric+dhic)*HICv2,

                    d* (H + v1d+ Hv2d) + dic * (IC  +ICv1d + ICv2d) + dhic * (HIC + HICv1d + HICv2d),

                    gamma * I + r* H + ric + HIC,
                    gamma * Iv1d + r* Hv1d + ric + HICv1d,
                    gamma * Iv1d + r* Hv1d + ric + HICv2d





    ])




sol = solve_ivp(calculate, [0, 60],
                        [S__, Shold1d__, Sv1ddt__, Shold2d__, Sv2d__,
                        E__, Ev1ddt__, Ev2d__,
                        I__, Iv1d__, Iv2d__,
                        H__, Hv1d__, Hv2d__,
                        IC__, ICv1ddt__, ICv2d__,
                        HIC__, HICv1ddt__, HICv2d__,
                        D__,
                        R__, Rv1d__, Rv2d__],



                rtol=1e-6, args=(beta,
                                sigma,
                                gamma,
                                lamdaa,
                                alfa,
                                alfa2,
                                delta,
                                delta2,
                                eta,
                                eta2,
                                h,
                                i1,
                                i2,
                                d,
                                dic,
                                dhic,
                                r,
                                ric))



fig = plt.figure(); ax = fig.gca()
curves = ax.plot(sol.t, sol.y.T)
#ax.legend(curves, ['Susceptible', 'Exposed', 'Infected', 'Recovered'])
plt.show()


# dSdt = -lamdaa * S - alfa * S,
# dShold1ddt =alfa * S - (1/delta)* Shold1d - lamdaa* Shold1d,
# dSv1ddt= (1/delta) * Shold1d - eta * lamda * Sv1d - alfa2 * Sv1d,
# dShold2ddt = alfa2 * Sv1d - (1/delta2)*Shold2d - eta * lamdaa * Shold2d,
# dSv2ddt = (1/delta2) * Shold2d - eta2 * lamda * Sv2d

# DEdt = lamdaa * (S + Shold1d) - sigma * E
# dEv1ddt = eta * lamda * (Sv1d+Shold2d) - sigma * Ev1d,
# dEv2ddt = eta2 * lamdaa * Sv2d - sigma * Ev2d


# dIdt = sigma * E - (gamma + h)*I
# dIdv1ddt = sigma * Ev1d - (gamma + h)*Iv1d,
# dIv2ddt = sigma * Ev2d- (gamma + h)*Iv2ddHdt = h*I - (i1 +d +r) *H,

# dHdt = h*I - (i1 +d +r) *H
# dHv1ddt = h*Iv1d- (i1+d+r)*Hv1d,
# dHv2ddt = h*Iv2d- (i1+d+r)*Hv2ddICdt = i1*H - (i2+dic)*IC,

# dICdt = i1*H - (i2+dic)*IC
# dICv1ddt = i1*Hv1d - (i2+dic)*ICv1d,
# dICv2ddt = i1*Hv2d - (i2+dic)*ICv2ddHICdt = i2 * IC - (ric+dhic)*HIC,
# dHICv1ddt = i2 * ICv1d - (ric+dhic)*HICv1d,
# dHICv2ddt = i2 * ICv2d - (ric+dhic)*HICv2

# ddDdt = d* (H + v1d+ Hv2d) + dic * (IC  +ICv1d + ICv2d) + dhic * (HIC + HICv1d + HICv2d)

# dRdt = gamma * I + r* H + ric + HIC,
# dRv1ddt = gamma * Iv1d + r* Hv1d + ric + HICv1d,
# dRv2ddt = gamma * Iv1d + r* Hv1d + ric + HICv2d

# # daily_infections = S + hold1d + (eta * (Sv1d + Shold2d)) + (eta2 * Sv2d)) * lamdaa,
# # daily_cases = sigma * (E + Ev1d + Ev2d) * P_ascertainment,
# # hospital_admissions = (I + Iv1d +Iv2d) * h,
# # ic_admissions = (H + Hv1d + Hv2d) * i1,
# # daily_deaths = (H + Hv1d + Hv2d) * d + (IC + ICv1d + ICv2d) * dIC + (HIC+ HICv1d + HICv2d) * hIC,
# # life_years_lost = deaths * life_expectancy

