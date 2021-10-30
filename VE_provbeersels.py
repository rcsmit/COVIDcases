# Calculating VE and CI's with various methods

from R_functions import *
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import math
from patsy import dmatrices
import scipy.stats as st

def traditional(e,f,g,h):
    """[summary]

    Args:
        e ([type]): sick vax
        f ([type]): sick unvax
        g ([type]): total vax
        h ([type]): total unvax

    Returns:
        0
    """


    # https://stats.stackexchange.com/questions/297837/how-are-p-value-and-odds-ratio-confidence-interval-in-fisher-test-are-related
    #  p<0.05 should be true only when the 95% CI does not include 1. All these results apply for other α levels as well.
    # https://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions
    # https://select-statistics.co.uk/calculators/confidence-interval-calculator-odds-ratio/

    # 90%	1.64	1.28
    # 95%	1.96	1.65
    # 99%	2.58	2.33
    Za2 = 2.58

    a = e # sick vax
    b = f # sick unfax
    c = g-e # healthy vax
    d =h-f # healthy unvax


    # https://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions
    #https://twitter.com/MrOoijer/status/1445074609506852878
    rel_risk = (a/(a+c))/(b/(b+d)) # relative risk !!
    yyy = 1/a  + 1/c


    print(f"Traditional method                    : {round((1- rel_risk)*100,2)} [{round((1-(np.exp(np.log(rel_risk) +Za2 * math.sqrt(yyy))))*100,2)}, {round((1-(np.exp(np.log(rel_risk) -Za2 * math.sqrt(yyy))))*100,2)}]  ")



    # #  https://select-statistics.co.uk/calculators/confidence-interval-calculator-odds-ratio/
    # # Explained in Martin Bland, An Introduction to Medical Statistics, appendix 13C
    # or_ = (a*d) / (b*c)
    # xxx = 1/a + 1/b + 1/c + 1/d
    # print(f"CI_OR_low = {np.exp(np.log(or_) -Za2 * math.sqrt(xxx))}")
    # print(f"or_fisher_2 = {or_}")
    # print(f"CI_OR_high = {np.exp(np.log(or_) +Za2 * math.sqrt(xxx))}")

def pfizer(a,b,c,d):
    """
    Calculates vaccine efficacy (VE) and 95% credible interval
    based on beta-binomial model
    Described on  http://skranz.github.io//r/2020/11/11/CovidVaccineBayesian.html
    """

    a0 = 0.700102; b0 = 1 # values from studyplan
    # a vaccinated|sick
    # b non vacc|sick
    # a = a0+8; b = b0+94-8
    #a =47498 #/12365333
    # b= 56063 * 12365333/3342667 # /3342667
    b= b*c/d
    VE = 1-(a/b)

    theta_ci_low =  qbeta (0.025,a,b)
    theta_ci_high = ( qbeta(0.975 ,a,b))
    VE_ci_high = round(((1-2*theta_ci_low)/(1-theta_ci_low))*100,2)
    VE_ci_low =round(((1-2*theta_ci_high)/(1-theta_ci_high))*100,2)
    print (f"SKRANZ pfizer beta binomial          : {round(VE*100,2)} [{VE_ci_low},{VE_ci_high}] ")

def boyangzhao(a,b,c,d):
    """ As described on https://boyangzhao.github.io/posts/vaccine_efficacy_bayesian
    Calculates vaccine efficacy (VE) and 95% credible interval based on beta-binomial model
    """

    def calc_theta(VE, r=1):
        # calculate case rate (theta) given VE and surveillance time ratio
        return r*(1-VE) / (1+r*(1-VE))

    def calc_VE(theta, r=1):
        # calculate VE given case rate (theta) and surveillance time ratio
        return 1 + theta/(r*(theta-1))

    def VE_95ci_betabinom(c_v, c_p, t_v, t_p):
        '''
        Calculates vaccine efficacy (VE) and 95% credible interval
        based on beta-binomial model

        params:
            c_v: number of cases in vaccinated group
            c_p: number of cases in placebo group
            t_v: surveillance time in vaccinated group
            t_p: surveillance time in placebo group
        '''

        a = 0.700102 + c_v
        b = 1 + c_p
        irr_v = c_v/t_v
        irr_p = c_p/t_p
        r = t_v/t_p

        # VE
        VE = 1 - irr_v/irr_p

        # confidence interval
        theta_ci_lower = st.beta.ppf(0.025, a, b)
        theta_ci_higher = st.beta.ppf(0.975, a, b)
        VE_ci_lower = calc_VE(theta_ci_higher, r)
        VE_ci_upper = calc_VE(theta_ci_lower, r)

        # P(VE>30%|data)
        p_ve30plus = st.beta.cdf(calc_theta(0.3, r), a, b)

        print(f'VE boyangzhao beta-binomial model    : {VE*100:.2f} [{VE_ci_lower*100:.2f} - {VE_ci_upper*100:.2f}]') #; ' +
        #    f'P(VE>30%|data): {p_ve30plus:0.30f}')

    # calculate VE and 95% CI
    VE_95ci_betabinom(a,b,c,d)

def who(a,b,c,d, how):
    # https://www.who.int/docs/default-source/coronaviruse/act-accelerator/covax/screening-method-r-script.zip?Status=Master&sfvrsn=d7d1f3e8_5
    # cited in file:///C:/Users/rcxsm/Downloads/WHO-2019-nCoV-vaccine-effectiveness-variants-2021.1-eng.pdf

    # Screening Method: Calculating a Crude Vaccine Effectiveness (VE) and 95% Confidence Limits
    # References: Orenstein et al. Field evaluation of vaccine efficacy.
    #             https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2536484/
    #             Farrington, C. Estimation of vaccine effectiveness using the screening method.
    #             https://doi.org/10.1093/ije/22.4.742

    # Data Entry NL okt without kids
    cases = a+b #int(103561/174) #replace XXX with total number of cases
    vaxcases = a #int(47498/174) #replace XXX with number of vaccinated cases
    pop_vax =  c
    pop_unvax = d
    perpopvax = c/(c+d) # 0.7872 #replace XXX with percent of population vaccinated (as a decimal)
    p_sick_unvax = b/d # (cases-vaxcases)/100_000

    # Read in input data containing four columns: cohort case vac ppv
    cohort="group"
    case=cases
    vac =vaxcases
    ppv =perpopvax
    pcv=vac/case
    logit_ppv=np.log(ppv/(1-ppv))
    # Restructure dataset to fit model
    df = make_df(a,b,c,d )
    df["logit_ppv"] = np.log(ppv/(1-ppv))

    # l=[]
    # for i in range(case):
    #     if i<=vac:
    #         l.append([cohort,case,vac,ppv,1,pcv,logit_ppv])

    #     else:
    #         l.append([cohort,case,vac,ppv,0,pcv,logit_ppv])
    # df = pd.DataFrame(l, columns = ['cohort','case','vac','ppv','y','pcv','logit_ppv'])


    #print (df)
    # Fit logistic regression
    #mylogit <- glm(y ~ 1+offset(logit_ppv), data=two, family="binomial")


    #mylogit  = smf.glm(formula = "y ~ logit_ppv", data=df, family=sm.families.Binomial()).fit()
#mylogit  = smf.glm(formula = "y ~ 1+logit_ppv", data=df,  family=sm.families.NegativeBinomial()).fit()

    # still strugling with the translation of [ 1+ offset (logit_ppv) ]
    if how=="neg_bin":
        mylogit  = smf.glm(formula = 'INFECTED ~  VACCINATED', data=df,  family=sm.families.NegativeBinomial()).fit()
    elif how == "bin":
        mylogit  = smf.glm(formula = 'INFECTED ~  VACCINATED', data=df,  family=sm.families.Binomial()).fit()
    else:
        print ("ERROR")

    #print (mylogit .summary())
    params = mylogit.params

    VE = VE_(params[1], p_sick_unvax)

    conf =   mylogit.conf_int()
    high, low  = conf[0][1],  conf[1][1]
    VE_low, VE_high = VE_(low, p_sick_unvax), VE_(high, p_sick_unvax)
    print (f"VE WHO smf {how}                     : {VE} % [{VE_low} , {VE_high}]")


def VE_(x, p_sick_unvax):
    """Calculate VE and CI's

    Args:
    x (float): The coeff. or CI
    Returns
    x (float): The VE or CI in %"""

    odds_ratio = np.exp(x)
    IRR = (odds_ratio / ((1-p_sick_unvax) + (p_sick_unvax*odds_ratio)))
    VE = round((1-IRR)*100,2)
    return VE
def make_df(a,b,c,d):
    """Make dataframe used for GLM
    """
    l=[]
    for i in range(c):
        if i<a:
            l.append([1,1])
        else:
            l.append([1,0])
    for i in range(d):
        if i<b:
            l.append([0,1])
        else:
            l.append([0,0])

    df = pd.DataFrame(l, columns = ['VACCINATED', 'INFECTED'])
    return df
def log_regression(a,b,c,d):
    """[summary]
    Calculate VE and CI's according
    https://timeseriesreasoning.com/contents/estimation-of-vaccine-efficacy-using-logistic-regression/
    * We'll use Patsy to carve out the X and y matrices
    * Build and train a Logit model (sm.Logit)

    Args:
        a ([type]): sick vax
        b ([type]): sick unvax
        c ([type]): total vax
        d ([type]): total unvax

    Returns:
        0"""

    df = make_df(a,b,c,d)
    p_sick_unvax = b/d
    #Form the regression equation
    expr = 'INFECTED ~  VACCINATED'

    #We'll use Patsy to carve out the X and y matrices
    y_train, X_train = dmatrices(expr, df, return_type='dataframe')

    #Build and train a Logit model
    logit_model = sm.Logit(endog=y_train, exog=X_train, disp=False)
    logit_results = logit_model.fit( disp=False)
    params = logit_results.params

    #Print the model summary
    #print(logit_results.summary2())


    VE = VE_(params[1], p_sick_unvax)

    # print(f"\nConfidence intervals")
    # print (logit_results.conf_int())  # confidence intervals


    conf =  logit_results.conf_int()
    high, low  = conf[0][1],  conf[1][1]
    VE_low, VE_high = VE_(low, p_sick_unvax), VE_(high, p_sick_unvax)
    print (f"VE logit model                       : {VE} % [{VE_low} , {VE_high}]")


    # print("\np values VACCINATED") # p values
    # print (logit_results.pvalues[1])


    #print(logit_results.cov_params)
    #
    # print ("\nCoef. Intercept")
    # intercept  =  params[0]
    # print (params[0])  # Coef. intercept

    # print ("\nCoef. VACCINATED")
    # coef_vacc = params[1]
    # print (params[1])  # Coef. VACCINATED !!

    # print ("\nOdds Ratio")
    # conf['Odds Ratio'] = params
    # conf.columns = ['5%', '95%', 'Odds Ratio']
    # print(np.exp(conf))

    # for attr in dir(logit_results):
    #     if not attr.startswith('_'):
    #         print(attr)

    # LLR p-value : moet laag zijn

    # Pseudo R-squ. : moet hoog zijn
    # P_VACCINATED moet laag zijn
    #ln(odds) = -1.5* VACCINATED -4.1

def r_script_who():
    """ Results of the WHO R-script (see comments)
    """
    print ("R Script VE                           : 77.71003 % [67.24307   84.90514] ")
    # https://www.who.int/docs/default-source/coronaviruse/act-accelerator/covax/screening-method-r-script.zip?Status=Master&sfvrsn=d7d1f3e8_5
    # cited in file:///C:/Users/rcxsm/Downloads/WHO-2019-nCoV-vaccine-effectiveness-variants-2021.1-eng.pdf
    # Data Entry NL okt without kids (corrected to make numbers per 100k)
    # cases<- int ((47498/123.65) + (56063/33.43))  #103561 #replace XXX with total number of cases
    # vaxcases<- int(47498/123.65) #47498 #replace XXX with number of vaccinated cases
    # perpopvax<- 12365333 / (12365333+ 3342667)  #0.7872 #replace XXX with percent of population vaccinated (as a decimal)
    # crucial line: mylogit <- glm(y ~ 1+offset(logit_ppv), data=two, family="binomial")



def remarks_links_info():
    """just some links
    """
    pass
    # https://ibecav.netlify.app/post/warpspeed-confidence-what-is-credible/
    # https://stats.idre.ucla.edu/spss/dae/negative-binomial-regression/
    # https://stats.stackexchange.com/questions/496774/which-statistical-model-is-being-used-in-the-pfizer-study-design-for-vaccine-eff
    # https://statmodeling.stat.columbia.edu/2020/11/13/pfizer-beta-prior-vaccine-effect/
    # https://statmodeling.stat.columbia.edu/2020/11/11/the-pfizer-biontech-vaccine-may-be-a-lot-more-effective-than-you-think/
    # https://ibecav.netlify.app/post/warpspeed-confidence-what-is-credible/
    # https://twitter.com/nataliexdean/status/1307067685310730241?s=20
    # https://ibecav.netlify.app/post/warspeed-5-priors-and-models-continued/
    # https://medium.com/swlh/the-fascinating-math-powering-the-covid-19-vaccine-trials-930a5e97c9c9

def main():
    # Netherlands october 2021 without 0-9-years sick_vaxxed, sick_unvaxed, people_vaxxed, people_unvaxxed
    # adjusted to 100k people in each group
    a,b,c,d = 47498/123.65,56063/33.43, 12_365_333/123.65, 3_342_667/33.43
    # original values
    # a,b,c,d = 47498,56063, 12_365_333, 3_342_667

    x =1 # number to divide to speed up the script. The smaller the numbers, the wider the CI's (and vv). Groups of 100k makes results like in the literature
    a,b,c,d = int(a/x), int (b/x), int(c/x), int(d/x)
    r_script_who()
    traditional(a,b,c,d )
    log_regression(a,b,c,d )
    boyangzhao(a,b,c,d )
    pfizer(a,b,c,d )
    who(a,b,c,d, "bin")
    who(a,b,c,d, "neg_bin")


main()

