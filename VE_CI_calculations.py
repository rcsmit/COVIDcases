# Calculating VE and CI's with various methods

from R_functions import *
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import math
from patsy import dmatrices
import scipy.stats as st
import streamlit as stl

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


    stl.write(f"Traditional method                    : {round((1- rel_risk)*100,2)} [{round((1-(np.exp(np.log(rel_risk) +Za2 * math.sqrt(yyy))))*100,2)}, {round((1-(np.exp(np.log(rel_risk) -Za2 * math.sqrt(yyy))))*100,2)}]  ")



    # #  https://select-statistics.co.uk/calculators/confidence-interval-calculator-odds-ratio/
    # # Explained in Martin Bland, An Introduction to Medical Statistics, appendix 13C
    # or_ = (a*d) / (b*c)
    # xxx = 1/a + 1/b + 1/c + 1/d
    # stl.write(f"CI_OR_low = {np.exp(np.log(or_) -Za2 * math.sqrt(xxx))}")
    # stl.write(f"or_fisher_2 = {or_}")
    # stl.write(f"CI_OR_high = {np.exp(np.log(or_) +Za2 * math.sqrt(xxx))}")

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
    stl.write(f"SKRANZ pfizer beta binomial          : {round(VE*100,2)} [{VE_ci_low},{VE_ci_high}] ")

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

        stl.write(f'VE boyangzhao beta-binomial model    : {VE*100:.2f} [{VE_ci_lower*100:.2f} - {VE_ci_upper*100:.2f}]') #; ' +
        #    f'P(VE>30%|data): {p_ve30plus:0.30f}')

    # calculate VE and 95% CI
    VE_95ci_betabinom(a,b,c,d)

def farrington(df, a,b,c,d, distribution, method):
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
    perpopvax =  12_365_333/ ( 12_365_333 + 3_342_667)# c/(c+d) # 0.7872 #replace XXX with percent of population vaccinated (as a decimal)
    p_sick_unvax = b/d # (cases-vaxcases)/100_000

    # Read in input data containing four columns: cohort case vac ppv
    cohort="group"
    case=cases
    vac =vaxcases
    ppv =perpopvax
    pcv=vac/case

    logit_ppv=np.log(ppv/(1-ppv))

    if method == "new":
        # Restructure dataset to fit model

        df["logit_ppv"] = logit_ppv

        # still strugling with the translation of [ 1+ offset (logit_ppv) ]
        if distribution =="neg_bin":
            mylogit  = smf.glm(formula = 'INFECTED ~  VACCINATED', data=df,  family=sm.families.NegativeBinomial()).fit()
        elif distribution == "bin":
            mylogit  = smf.glm(formula = 'INFECTED ~  VACCINATED', data=df,  family=sm.families.Binomial()).fit()
        elif distribution == "poisson":
            mylogit  = smf.glm(formula = 'INFECTED ~  VACCINATED', data=df,  family=sm.families.Poisson()).fit()
        elif distribution == "bin_offset":
            mylogit  = smf.glm(formula = 'INFECTED ~  VACCINATED', data=df, offset=df['logit_ppv'], family=sm.families.Binomial()).fit()
        else:
            stl.write("ERROR")
    elif method == "old":
        # translation of the R script
        l=[]
        for i in range(case):
            if i<=vac:
                l.append([cohort,case,vac,ppv,1,pcv,logit_ppv])
            else:
                l.append([cohort,case,vac,ppv,0,pcv,logit_ppv])
        df = pd.DataFrame(l, columns = ['cohort','case','vac','ppv','y','pcv','logit_ppv'])


        stl.write(df)
        #Fit logistic regression
        # mylogit <- glm(y ~ 1+offset(logit_ppv), data=two, family="binomial") -------- TE VERTALEN


        #mylogit  = smf.glm(formula = "y ~ logit_ppv", data=df, family=sm.families.Binomial()).fit()
        #mylogit  = smf.glm(formula = "y ~ 1+logit_ppv", data=df,  family=sm.families.Binomial()).fit()
        df["one"] = 1
        mylogit  = smf.glm(formula = 'case ~ vac ', data=df, offset=df['logit_ppv'], family=sm.families.Binomial()).fit()


    #stl.write(mylogit .summary())
    params = mylogit.params

    VE = VE_(params[1], p_sick_unvax)

    # rd = mylogit.resid_deviance[0]
    # stl.write( mylogit.summary())
    # nd = mylogit.null_deviance
    # goodness_of_fit = 1-(rd/nd) # https://stats.stackexchange.com/questions/46345/how-to-calculate-goodness-of-fit-in-glm-r

    # for attr in dir(mylogit):
    #     if not attr.startswith('_'):
    #         stl.write(attr)
    conf =   mylogit.conf_int()
    high, low  = conf[0][1],  conf[1][1]
    VE_low, VE_high = VE_(low, p_sick_unvax), VE_(high, p_sick_unvax)
    stl.write(f"VE smf {distribution}                     : {VE} % [{VE_low} , {VE_high}]") # | GoF = {goodness_of_fit}")


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
    """Make dataframe used for GLM.
    Called design matrix
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
def regression(df, a,b,c,d, distribution):
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


    p_sick_unvax = b/d
    #Form the regression equation
    expr = 'INFECTED ~  VACCINATED'

    #We'll use Patsy to carve out the X and y matrices
    y_train, X_train = dmatrices(expr, df, return_type='dataframe')

    #Build and train a Logit model
    if distribution=="logit":
        model = sm.Logit(endog=y_train, exog=X_train, disp=False)
    elif distribution =="poisson":
        model = sm.Poisson(endog=y_train, exog=X_train, disp=False)
    elif distribution == "neg_bin":
        model = sm.NegativeBinomial(endog=y_train, exog=X_train, disp=False)

    results = model.fit( disp=False)
    params = results.params

    #Print the model summary
    #stl.write(logit_results.summary2())


    VE = VE_(params[1], p_sick_unvax)

    # stl.write(f"\nConfidence intervals")
    # stl.write(logit_results.conf_int())  # confidence intervals


    conf =  results.conf_int()
    high, low  = conf[0][1],  conf[1][1]
    prsquared = results.prsquared
    VE_low, VE_high = VE_(low, p_sick_unvax), VE_(high, p_sick_unvax)
    stl.write(f"VE Regression {distribution}                       : {VE} % [{VE_low} , {VE_high}] | pseudo-R2 = {prsquared}")


    # stl.write("\np values VACCINATED") # p values
    # stl.write(logit_results.pvalues[1])


    #stl.write(logit_results.cov_params)
    #
    # stl.write("\nCoef. Intercept")
    # intercept  =  params[0]
    # stl.write(params[0])  # Coef. intercept

    # stl.write("\nCoef. VACCINATED")
    # coef_vacc = params[1]
    # stl.write(params[1])  # Coef. VACCINATED !!

    # stl.write("\nOdds Ratio")
    # conf['Odds Ratio'] = params
    # conf.columns = ['5%', '95%', 'Odds Ratio']
    # stl.write(np.exp(conf))

    # for attr in dir(logit_results):
    #     if not attr.startswith('_'):
    #         stl.write(attr)

    # LLR p-value : moet laag zijn

    # Pseudo R-squ. : moet hoog zijn
    # P_VACCINATED moet laag zijn
    #ln(odds) = -1.5* VACCINATED -4.1

def r_script_farrington():
    """ Results of the WHO R-script (see comments)
    """
    stl.write("R Script VE                           : 77.71003 % [67.24307   84.90514] ")
    # https://www.who.int/docs/default-source/coronaviruse/act-accelerator/covax/screening-method-r-script.zip?Status=Master&sfvrsn=d7d1f3e8_5
    # cited in file:///C:/Users/rcxsm/Downloads/WHO-2019-nCoV-vaccine-effectiveness-variants-2021.1-eng.pdf
    # Data Entry NL okt without kids (corrected to make numbers per 100k)
    # cases<- int ((47498/123.65) + (56063/33.43))  #103561 #replace XXX with total number of cases
    # vaxcases<- int(47498/123.65) #47498 #replace XXX with number of vaccinated cases
    # perpopvax<- 12365333 / (12365333+ 3342667)  #0.7872 #replace XXX with percent of population vaccinated (as a decimal)
    # crucial line: mylogit <- glm(y ~ 1+offset(logit_ppv), data=two, family="binomial")



def links_2():
    """just some links
    """
    stl.write("Sourcecode : https://github.com/rcsmit/COVIDcases/VE_CI_calculations.py")
    stl.subheader("Extra info")
    stl.write("https://ibecav.netlify.app/post/warpspeed-confidence-what-is-credible/")
    stl.write("https://stats.idre.ucla.edu/spss/dae/negative-binomial-regression/")
    stl.write("https://stats.stackexchange.com/questions/496774/which-statistical-model-is-being-used-in-the-pfizer-study-design-for-vaccine-eff")
    stl.write("https://statmodeling.stat.columbia.edu/2020/11/13/pfizer-beta-prior-vaccine-effect/")
    stl.write("https://statmodeling.stat.columbia.edu/2020/11/11/the-pfizer-biontech-vaccine-may-be-a-lot-more-effective-than-you-think/")
    stl.write("https://ibecav.netlify.app/post/warpspeed-confidence-what-is-credible/")
    stl.write("https://twitter.com/nataliexdean/status/1307067685310730241?s=20")
    stl.write("https://ibecav.netlify.app/post/warspeed-5-priors-and-models-continued/")
    stl.write("https://medium.com/swlh/the-fascinating-math-powering-the-covid-19-vaccine-trials-930a5e97c9c9")

def links_1():
    stl.subheader("Resources and inspiration")
    stl.write("R-script: https://github.com/rcsmit/COVIDcases/blob/main/not_active_on_streamlit/screeningmethod.R")
    stl.write("Traditional method : RR = (sick vax/pop vax)/(sic nonvax/pop nonvax) | VE = 1-RR | CI = (1-(exp(log(rel_risk) +/- 1.96 * sqrt(1/a+1/c))))*100")
    stl.write("Regression : https://timeseriesreasoning.com/contents/estimation-of-vaccine-efficacy-using-logistic-regression/")
    stl.write("boyangzhao : https://boyangzhao.github.io/posts/vaccine_efficacy_bayesian")
    stl.write("SKRANZ : http://skranz.github.io//r/2020/11/11/CovidVaccineBayesian.html")
    #stl.write("smf : -")

def interface():
    a_,b_,c_,d_, e_, f_ = 47498,56063, 12_365_333, 3_342_667,103561, 15708000 # the Netherlands, okt 2021, without 0-9 years
    per_100k = stl.sidebar.selectbox("Transform input in model per 100k", [True, False], index=0)
    input_total = stl.sidebar.selectbox("Input vax & total numbers", [True, False], index=1)


    x_ =100 # number to divide to speed up the script. The smaller the numbers, the wider the CI's (and vv). Groups of 100k makes results like in the literature
    if input_total == True:
        a__= stl.sidebar.number_input("Sick | vax", value=a_)
        e__= stl.sidebar.number_input("Sick | all", value=e_)
        b__ = e__  - a__

        c__= stl.sidebar.number_input("Population | vax", value=c_)
        f__ = stl.sidebar.number_input("Population | all", value=f_)
        d__ = f__ - c__ # population non vax

    else:
        a__= stl.sidebar.number_input("Sick | vax", value=a_)
        b__= stl.sidebar.number_input("Sick | non vax", value=b_)

        c__= stl.sidebar.number_input("Population | vax", value=c_)
        d__= stl.sidebar.number_input("Population | non vax", value=d_)
    x = stl.sidebar.number_input("Factor to divide by (x)", value=x_)
    stl.sidebar.write("Lower x when getting divide by zero errors or wrong results.  ")

    if per_100k == True:
        a =int ((a__ /(c__/100_000))/x)

        b= int((b__/ (d__/100_000))/x)
        c = int((c__ / (c__/100_000))/x)
        d = int((d__/ (d__/100_000) )/x)
    else:
        a,b,c,d = int(a__/x), int (b__/x), int(c__/x), int(d__/x)

    return a,b,c,d
def main():
    # Netherlands october 2021 without 0-9-years sick_vaxxed, sick_unvaxed, people_vaxxed, people_unvaxxed
    # adjusted to 100k people in each group

    #a__,b__,c__,d__ = 47498/123.65333,56063/ 33.42667, 12_365_333/123.65333, 3_342_667/33.42667
    # original values
    stl.subheader ("VE and CI calculator (screening method)")
    a,b,c,d = interface()
    df = make_df(a,b,c,d)
    #stl.write(f"{a=}, {b=} {c=} {d=}")
    r_script_farrington()
    traditional(a,b,c,d )
    boyangzhao(a,b,c,d )
    pfizer(a,b,c,d )
    regression(df,a,b,c,d, "logit" )
    regression(df,a,b,c,d, "poisson" )
    #regression(a,b,c,d, "neg_bin" )

    #farrington(a,b,c,d, "bin", "new")
    farrington(df,a,b,c,d, "neg_bin", "new")
    # farrington(df,a,b,c,d, "poisson", "new")
    # farrington(df,a,b,c,d, "bin_offset", "new")
    #farrington(None,a,b,c,d,None, "old")

    links_1()
    links_2()

if __name__ == "__main__":
    main()