# Berekening vaccin efficientie
from re import A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from scipy.stats import fisher_exact
def calculate_ci(a,b,c,d):
    Za2 = 1.96

    # Taylor series
    # https://apps.who.int/iris/bitstream/handle/10665/264550/PMC2491112.pdf
    ARV = a / (a+b)
    ARU = c/ (c+d)
    RR_taylor = ARV/ARU
    zzz = ((1-ARV)/a)+((1-ARU)/c)

    ci_low_taylor = round( np.exp(np.log(RR_taylor) -Za2 * math.sqrt(zzz)),4)
    ci_high_taylor = round( np.exp(np.log(RR_taylor) +Za2 * math.sqrt(zzz)),4)

    or_ = (a*d) / (b*c)
    xxx = 1/a + 1/b + 1/c + 1/d
    ci_low = round( np.exp(np.log(or_) -Za2 * math.sqrt(xxx)),2)
    ci_high = round( np.exp(np.log(or_) +Za2 * math.sqrt(xxx)),2)

    # https://stats.stackexchange.com/questions/506017/can-the-fisher-exact-test-be-used-to-get-a-confidence-interval-for-the-efficienc?rq=1
    # https://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions#comment38531_21305

    n0 = a+c+1
    n1 = b+d+1
    p0 = (a+0.5) / n0
    p1 = (b+0.5) / n1
    k = 1/(n0*p0*(1-p0))
    l = 1/(n1*p1*(1-p1))
    #print (f'{xxx}  {k+l}')
    ci_low_2 = round( np.exp(np.log(or_) - Za2 * math.sqrt(k+l)),4)
    ci_high_2 = round( np.exp(np.log(or_) + Za2 * math.sqrt(k+l)),4)
    #gives same result as method 1

    return ci_low, ci_high, ci_low_taylor, ci_high_taylor

def main():
    """ A single result
    """
    a,b,c,d =  21720 - 8, 21728 - 162, 8, 162
    a,b,c,d = 9, 1, 1, 3
    a,b,c,d = 20,830,80,70
    a,b,c,d = 8,(41135/2)-8,162,(41135/2)-162 #pfizer first trial https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=1390&context=numeracy
    odds, p =  fisher_exact([[a,b],[c,d]],  alternative='less')
    ve = round( (  (1-(a/(a+b))/(c/(c+d)))*100),3)
    ci_low, ci_high, ci_low2, ci_high2 = calculate_ci(a,b,c,d)
    if p<0.05 :
        werkt = "vaccin werkt"
    else :
        werkt = "vaccin werkt niet"
    print (f"a = {a} | odds = {round(odds,1)} [{ci_low} - {ci_high}] / [{ci_low2} - {ci_high2}] | {ve=} %  [{(1-ci_high2)*100} - {(1-ci_low2)*100}] | p = {round(p,4)}   {werkt}  {a=}  {b=}  {c=} {d=}"  )


def main_():
    """Result for all the combinations, giving row and column totals
    """
    a_b = 20
    c_d = 28
    a_c = 23
    b_d = 25

    if a_b < a_c:
        until = a_b
    else:
        until = a_b

    for i in range(1,until):
        a = i
        b = a_b - a
        c = a_c -a
        d = c_d - c
        ci_low, ci_high, ci_low2, ci_high2 = calculate_ci(a,b,c,d)
        ve = round( (  (1-(a/(a+b))/(c/(c+d)))*100),3)

        odds, p =  fisher_exact([[a,b],[c,d]],  alternative='less')
        if p<0.05 :
            werkt = "vaccin werkt"
        else :
            werkt = "vaccin werkt niet"

        # print ("    Sick healty")
        # print (f"V : {str(a).zfill(2)} - {str(b).zfill(2)} = {a_b}")
        # print (f"N : {str(c).zfill(2)} - {str(d).zfill(2)} = {c_d}")
        # print (f"    {str(a_c).zfill(2)} - {str(b_d).zfill(2)} = {c_d}")


        print (f"a = {a} | odds = {round(odds,1)} [{ci_low} - {ci_high}] / [{ci_low2} - {ci_high2}] | {ve=} % | p = {round(p,4)}   {werkt}  {a=}  {b=}  {c=} {d=}"  )


if __name__ == "__main__":
    #caching.clear_cache()
    #st.set_page_config(layout="wide")
    main()