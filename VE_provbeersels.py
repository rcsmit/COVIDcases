from R_functions import *

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def glm():
    # https://stackoverflow.com/questions/42277532/python-negative-binomial-regression-results-dont-match-those-from-r
    # https://www2.karlin.mff.cuni.cz/~pesta/NMFM404/NB.html  MET UITLEG OVER NEG BIN MODEL

    df = pd.read_stata("http://www.karlin.mff.cuni.cz/~pesta/prednasky/NMFM404/Data/nb_data.dta")
    #model = smf.glm(formula = "daysabs ~ math + prog", data=df, family=sm.families.NegativeBinomial()).fit()

    # in R summary(m1 <- glm.nb(daysabs ~ math + prog, data = dat))
    model = smf.glm(formula = "daysabs ~ math + C(prog)", data=df, family=sm.families.NegativeBinomial()).fit()

    print (model.summary())

#print (pbeta(7/17,0.700102+6,1+26))

def pfizer():
    # http://skranz.github.io//r/2020/11/11/CovidVaccineBayesian.html

    a0 = 0.700102; b0 = 1 # values from studyplan
    # a vaccinated|sick
    # b non vacc|sick
    # a = a0+8; b = b0+94-8
    a =47498/12365333
    b= 56063/3342667
    VE = 1-(a/b)

    theta_ci_low = ( qbeta ((0.025,a,b))
    theta_ci_high = ( qbeta(0.975 ,a,b))
    VE_ci_high = round(((1-2*theta_ci_low)/(1-theta_ci_low))*100,2)
    VE_ci_low =round(((1-2*theta_ci_high)/(1-theta_ci_high))*100,2)
    print (round(VE*100,2))
    print (VE_ci_low)
    print (VE_ci_high)

# https://ibecav.netlify.app/post/warpspeed-confidence-what-is-credible/
# https://stats.idre.ucla.edu/spss/dae/negative-binomial-regression/
# https://stats.stackexchange.com/questions/496774/which-statistical-model-is-being-used-in-the-pfizer-study-design-for-vaccine-eff
# https://statmodeling.stat.columbia.edu/2020/11/13/pfizer-beta-prior-vaccine-effect/
# https://statmodeling.stat.columbia.edu/2020/11/11/the-pfizer-biontech-vaccine-may-be-a-lot-more-effective-than-you-think/
# https://ibecav.netlify.app/post/warpspeed-confidence-what-is-credible/

pfizer()