# CALCULATE Specificity and sensitivity rapidtests
# René Smit, 20 april 2022, MIT LICENSE

# from fnmatch import fnmatchcase
from tabulate import tabulate
# import matplotlib.pyplot as plt
import streamlit as st
import math
from statsmodels.stats.proportion import proportion_confint
import scipy.stats as scist

def simple_asymptotic(x,n, Za2):
    # Simple Asymptotic = Wald
    # The simple asymptotic formula is based on the normal approximation to the binomial distribution. 
    # The approximation is close only for very large sample sizes.
 
    p = x/n
    SE = math.sqrt((p*(1-p))/n )   #https://www2.ccrb.cuhk.edu.hk/stat/confidence%20interval/Diagnostic%20Statistic.htm#Formula
    c1 = round( p - Za2*SE,3)
    c2 = round ( p + Za2*SE,3)

    return c1, c2
  
def calculate_confint(x, n, alpha=0.05):
    # https://www.statology.org/binomial-confidence-interval-python/
    # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval

    # normal : asymptotic normal approximation
    # agresti_coull : Agresti-Coull interval
    # beta : Clopper-Pearson interval based on Beta distribution
    # wilson : Wilson Score interval
    # jeffreys : Jeffreys Bayesian Interval
    # binom_test : Numerical inversion of binom_test

    # VEEL ONDERZOEKEN LATEN BETA ZIEN
    # Flowflex gebruikt agresti-coull
    #thaise test onbekend

    methods = ["normal", "agresti_coull" , "beta", "wilson", "jeffreys"] # , "binom_test"]
    for m in methods:
        try:
            a,b = proportion_confint(count=x, nobs=n, method=m)
            st.write (f"{m} - [{round(a*100,1)},{round(b*100,1)}]")
        except:
            st.write(f"Problem with {m}")


def binomial_ci(x, n, alpha=0.05):
    # sens  c1,c2 = binomial_ci(tp,(tp+fn), alpha=0.05) #exact (Clopper-Pearson)
    # spec c1y,c2y = binomial_ci(tn,(fp+tn), alpha=0.05) # exact (Clopper-Pearson)
 
    # The following gives exact (Clopper-Pearson) interval for binomial distribution in a simple way.
    # https://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
    # x is number of successes, n is number of trials
    from scipy import stats
    if x==0:
        c1 = 0
    else:
        c1 = round(stats.beta.interval(1-alpha, x,n-x+1)[0],1)
    if x==n:
        c2=1
    else:
        c2 = round(stats.beta.interval(1-alpha, x+1,n-x)[1],1)

    
    return c1, c2


# https://www.frontiersin.org/articles/10.3389/fpubh.2013.00039/full
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4958484/
# https://andrewpwheeler.com/2020/11/30/confidence-intervals-around-proportions/



def calculate_se_sp( tp, fp, fn, tn, alpha ):
    
    """Calculte the Sensitivity and Specificity with it CI's

    Returns:
        se, sp, acc: sensitivity and specificity 
    """


    se = tp/(tp+fn)*100
    sp = tn/(fp+tn)*100
    acc = (tp + tn) /(tp+fn+fp+tn)*100
    

    #  SE sensitivity = square root [sensitivity – (1-sensitivity)]/n sensitivity) https://www.ncbi.nlm.nih.gov/books/NBK305683/
    
    SE_sp = math.sqrt((sp/100*(1-sp/100))/(fp+tn) )  #/(tp+fn)  #https://www2.ccrb.cuhk.edu.hk/stat/confidence%20interval/Diagnostic%20Statistic.htm#Formula
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("Sensitivity")
        se = calulate_value_and_ci ("Sensitivity", tp,(tp+fn), alpha)
    with col2:
        st.subheader(f"Specificity")
        sp = calulate_value_and_ci ("Specificity", tn,(fp+tn), alpha)
    with col3:
        st.subheader(f" Accuracy")
        acc = calulate_value_and_ci ("Accuracy",(tp + tn), (tp+fn+fp+tn),alpha)

    return se, sp, acc

def calulate_value_and_ci(what, m,n, alpha):
    Za2 = scist.norm.ppf(1-(alpha/2))
    c1,c2 = simple_asymptotic(m,n, Za2)
    bin_c1,bin_c2 = binomial_ci(m,n, alpha=alpha) #exact (Clopper-Pearson)
    # https://www2.ccrb.cuhk.edu.hk/stat/confidence%20interval/Diagnostic%20Statistic.htm
    # https://www.medcalc.org/calc/diagnostic_test.php

    # https%3A%2F%2Fncss-wpengine.netdna-ssl.com%2Fwp-content%2Fthemes%2Fncss%2Fpdf%2FProcedures%2FPASS%2FConfidence_Intervals_for_One-Sample_Sensitivity_and_Specificity.pdf
    value = m/(n)*100
    st.write(f"{what} = {value} %")
   
   
    st.write(f"Simple Asymptotic (Wald):")
    st.write (f"[{round(c1*100,1)} - {round(c2*100,1)}]")
    st.write(f"exact (Clopper-Pearson) = beta):") #geeft andere waarde
    st.write(f"[{round(bin_c1*100,1)} - {round(bin_c2*100,1)}]")
    calculate_confint(m,n, alpha=alpha)
    st.write("")

        # https://www2.ccrb.cuhk.edu.hk/stat/confidence%20interval/Diagnostic%20Statistic.htm
        # The Specificity is 0.91 and the 95% C.I. is (0.89746, 0.92254).
        # The Positive Predictive Value (PPV) is 0.1 and the 95% C.I. is (0.05842, 0.14158).
        # The Negative Predictive Value (NPV) is 0.99454 and the 95% C.I. is (0.99116, 0.99791).
        # The Pre-Test Probability is 0.01478.
        # The Likelihood Ratio Positive (LR+) is 7.40741 and the 95% C.I. is (5.54896, 9.88828).
        # The Positive Post-Test Probability is 0.1.
        # The Likelihood Ratio Negative (LR-) is 0.3663 and the 95% C.I. is (0.22079, 0.60771).
        # The Negative Post-Test Probability is 0.00546.

    return value

def calculate(se, sp, prevalentie, number_of_tested_people,  output):
    """calculates the values. Made as a function to be able to make graphs
    with various prevalences

    https://www.ntvg.nl/artikelen/het-gebruik-van-de-coronazelftest-perspectief

    Args:
        test (list): list with the name, sensitivity and specificity of the test
        prevalentie (float): prevalentie of the virus
        number_of_tested_people (int)): how many people to test
        output (boolean): Shows the output as text or not

    Returns:
        fdr,for_, pos, fpr - see code
    """
   
    sensitivity = se/100    
 
    specificity = sp/100

    total_healthy = (100 - prevalentie) * number_of_tested_people/100
    total_sick = (prevalentie/100) * number_of_tested_people

    true_negative = round((total_healthy * specificity),0)
    false_negative = round((total_sick * (1 - sensitivity)),0)

    true_positive = round((total_sick * sensitivity),0)
    false_positive= round((total_healthy * (1 - specificity)),0)
    # print (true_positive)
    # print (f"{specificity=}")
    # print (f"{total_healthy=}")
    # print (f"{false_positive=}")

    #true_positive_bayes = round (100* (sensitivity * prevalentie) / ((sensitivity * prevalentie) + ((1-specificity)* (1-prevalentie)  )),2)
    try:
        fdr = round(100*false_positive/(false_positive+ true_positive),4)
        #print (fdr)
    except:
        fdr = 0
    acc = round(100*((true_positive+true_negative)/number_of_tested_people),4)
    for_ = round(100*false_negative/(false_negative + true_negative),4)
    try:
        fpr = round(100*false_positive/(false_positive+ true_negative),4)
    except:
        fpr = 0
    try:
        fnr =  round(100*(false_negative/(true_positive+ false_negative)),4)
    except:
        fnr = 0
    pos = round(100*((true_positive + false_positive)/number_of_tested_people),4)
    try:
        ppv = round((true_positive/(true_positive+false_positive)*100),3)
    except:
        ppv = 0
    npv = round((true_negative/(false_negative+true_negative)*100),3)
    tpr =round(100*true_positive/(true_positive+false_negative),3) # equal to Se
    try:
        tnr = round(100*true_negative/(false_positive+true_negative),3) # equal to Sp
    except:
        tnr = 0
    a, b,c,d = ("PPV - "+ str(ppv)), ("FDR - " + str(fdr)),  ("FOR - " + str(for_)), ("NPV - " + str(npv))
    e,f,g,h = ("TPR/Se - " + str(tpr)), ("FPR - " + str(fpr)), ("FNR - " + str(fnr)), ("TNR, Sp - " + str(tnr))
    data = [

        [
            "Result: 'sick' (+) ",
            true_positive,
            false_positive,

            false_positive+ true_positive,
        ],
          [
            "Result: 'healthy' (-)",
            false_negative,
            true_negative,

            true_negative + false_negative,
        ],
        [
            "Total",
            (false_negative + true_positive),
            (true_negative + false_positive),

            (true_positive + false_positive + false_negative + true_negative),
        ],
    ]

    data2 = [

        [
            "Result: 'sick' (+) ",
            a ,
            b,

            ppv+fdr
        ],
        [
            "Result: 'healthy' (-)",
            c,
            d,
            for_+npv
        ],

    ]
    data3= [

        [
            "Result: 'sick' (+) ",
            e ,
            f

        ],
        [
            "Result: 'healthy' (-)",
            g,
            h
        ],
         [
            "Total",
           round(tpr+fnr),
           round(fpr+ tnr)
        ],

    ]

    if output:
        st.subheader("Extra calculations")
        st.text (f"Number of tested people : {number_of_tested_people}")

        st.text (f"Prevalentie testpopulatie = {round(prevalentie,2)}%  (= {round(number_of_tested_people*prevalentie/100)} 'sick' persons)")


        st.text(f"specificity : {sp} - sensitivity : {se}\n")

        st.text(tabulate(data, headers=["#", "'Person is\nSick' (+)\nSensitivity (TPR)", "'Person is\nHealthy' (-)\nSpecificity (TNR)", "Total"]))

        st.text(f"True positive : {round(true_positive)}")
        st.text(f"True negative : {round(true_negative)}")

        st.text(f"False positive : {round(false_positive)}")
        st.text(f"False negative : {round(false_negative)}")


        st.text(
            f"Positive predictive value (PPV)              : {ppv} %  \n(Chance of being really 'sick' when tested 'sick')"
        )
        st.text(
            f"Negative predictive value (NPV)              : {npv} %  \n(Chance of being really 'healthy' when you are tested 'healthy')"
        )
        st.text(
            f"False Positive rate (FPR / α / Type 1 error) : {fpr} %  \n(Chance of being tested 'sick' while being 'healthy' - probability of false alarm)" )
        st.text(
            f"False Negative rate (FNR/ β / type II error  : {fnr} %  \n(Chance of being tested 'healty' while being 'sick' - miss rate)" )

        st.text(
            f"False Discovery rate (FDR)                   : {fdr} %  \n(Chance of being not 'sick' while you are tested 'sick')"  )
        st.text(
            f"False Omission Rate (FOR)                    : {for_} %  \n(Chance of being 'sick' when you are tested  'healthy')" )

        # st.text(
        #      f"True positivity rate (Bayes): {true_positive_bayes} % ") # TOFIX


        # if true_positive_bayes!= (100-fdr):
        #     st.text (f"DIFFERENCE !")
        #     st.text (100-fdr-true_positive_bayes)
        #
        st.text(
            f"Accuracy                                     : {acc} % ")
        st.text(
            f"Chance to be tested positive (true & false)  : {pos} %\n\n"
        )
        #
        st.text("\n\n")

        #
        st.text(tabulate(data2, headers=["%", "'Person is\nSick' (+)", "'Person is\nHealthy' (-)", "Total"]))
        st.text("\n\n")
        st.text(tabulate(data3, headers=["%", "'Person is\nSick' (+)", "'Person is\nHealthy' (-)"]))
        st.text("--------------------------------------------------------------------------")

    return fdr,for_, pos, fpr


def interface_test_results():
    col1, col2,col3,col4 = st.columns(4)
    with col1: 
        st.write("")
    with col2:
        st.write("PCR +")
    with col3:
        st.write("PCR -")
    with col3:
        st.write("Total")

    col1, col2,col3,col4 = st.columns(4)
    with col1: 
        st.write("")
        st.write("")
        st.write("")
        st.write("Rapid +")
    with col2:
        tp =  (st.number_input('True positive',None,None, 98))
    with col3:
        fp =  (st.number_input('False positive',None,None, 0))
    with col4:   
        st.write("")
        st.write("")

        st.write("")
        st.write(tp+fp)
    col1, col2,col3,col4 = st.columns(4)
    with col1: 
        st.write("")
        st.write("")
        st.write("")
        st.write("Rapid -")
    with col2:
        fn =  (st.number_input('False negative',None,None, 2))
    with col3:
        tn =  (st.number_input('True Negative',None,None, 210))
    with col4:  
        st.write("")
        st.write("")
        st.write("") 
        st.write(fn+tn)


    col1, col2,col3,col4 = st.columns(4)
    with col1: 
        st.write("")
    with col2:
        st.write(tp+fn)
    with col3:
        st.write(fp+tn)
    with col4:   
        st.write(tp+fn+fp+tn )
    return tp,fp,fn,tn

def interface_left_bar():
    prevalentie = st.sidebar.number_input('prevalence testpopulation in %',0.0,100.0, 1.0, format="%.4f")
    number_of_tested_people =  (st.sidebar.number_input('Number of tested people',None,None, 100_000))
    alpha =  (st.sidebar.number_input('Alpha',None,None, 0.05))
        
    st.sidebar.write("Attention: too small numbers give erorrs (Division by zero)")
    return prevalentie,number_of_tested_people, alpha



def main():
    
    st.title(" Calculating Sensitivity and Specificity of Rapid tests")
    what_to_do = st.sidebar.selectbox("What to do [Sens./Spec. | Diseas prob.]", ["se_sp", "bayes"],0)
    if what_to_do == "se_sp":
        main_se_sp()
    else:
        main_bayes()


def main_bayes():
    # https://chatgpt.com/c/66f06f0e-acd0-8004-9990-8cd0ec0d268f
    


    # Function to calculate the probability using Bayes' Theorem
    def calculate_probability(prevalence, false_positive_rate, sensitivity):
        # Convert percentages to decimal for calculations
        prevalence /= 100
        false_positive_rate /= 100
        sensitivity /= 100
        
        # Calculate P(Pos)
        P_pos = (sensitivity * prevalence) + (false_positive_rate * (1 - prevalence))
        
        # Calculate P(Disease | Pos)
        P_disease_given_pos = (sensitivity * prevalence) / P_pos
        
        # Convert result back to percentage
        return P_disease_given_pos * 100

    # Streamlit app
    st.subheader("Disease Probability Calculator")
    # Calculate probability when the button is pressed
     # Inputs from user
    prevalence = st.sidebar.number_input("Prevalence of Disease (%)", min_value=0.0, max_value=100.0, value=0.1, step=0.01)
    false_positive_rate = st.sidebar.number_input("False Positive Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    sensitivity = st.sidebar.number_input("Sensitivity of the Test (%)", min_value=0.0, max_value=100.0, value=100.0, step=0.1)


    probability = calculate_probability(prevalence, false_positive_rate, sensitivity)
    st.info(f"The probability that a person actually has the disease given a positive test result is **{probability:.2f}%**.")
    
    # Explanation of the problem
    st.markdown(r"""    ### Problem Explanation:""")
    st.markdown(r"""    Suppose you are conducting a test for a disease with a certain **prevalence** in the population. The test has a **false positive rate** (the probability that a healthy person tests positive) and a **sensitivity** (the probability that a sick person tests positive). Given a positive test result, we want to know the probability that the person actually has the disease.""")
    st.markdown(r"""""")
    st.markdown(r"""    We use **Bayes' Theorem** to compute this:""")
    st.markdown(r"""""")
    
    st.latex(r"""    P(\text{{Disease}} | \text{{Positive}}) = \frac{P(\text{{Positive}} | \text{{Disease}}) \times P(\text{{Disease}})}{P(\text{{Positive}})}""")
    st.markdown(r"""""")
   
    st.write("  Where:")  
    st.write("- P(Positive | Disease) is the sensitivity of the test.")
    st.write("- P(Disease) is the prevalence of the disease.")
    st.write("- P(Positive | No Disease) is the false positive rate.")
    st.write("- P(Positive) is the overall probability of a positive test, which combines both true positives and false positives.")
    st.markdown(r"""    """)
  
    st.latex(r"""    P(\text{{Positive}}) = P(\text{{Positive}} | \text{{Disease}}) \times P(\text{{Disease}}) + P(\text{{Positive}} | \text{{No Disease}}) \times P(\text{{No Disease}})""")
   
    st.markdown(r"""""")
    st.markdown(r"""    We will calculate the probability that a person who tests positive actually has the disease using these inputs.""")
    st.markdown(r"""    """)

           
    # Additional explanation on the result
    st.markdown("""
    #### Additional Explanation:
    The result tells you the likelihood that the person has the disease if they test positive. Even with a high sensitivity, if the prevalence of the disease is very low and the false positive rate is not negligible, the actual probability that a positive result indicates disease can still be quite low. This is because, in a large population of healthy people, even a small false positive rate can lead to many false positives.
    """)

def main_se_sp():

    tp, fp, fn, tn = interface_test_results()
    prevalentie, number_of_tested_people, alpha = interface_left_bar()

    se,sp, acc =  calculate_se_sp(tp, fp, fn, tn, alpha)
 
    fdr, for_, pos, fpr = calculate( se,sp, prevalentie, number_of_tested_people, True)


if __name__ == "__main__":
    main()
