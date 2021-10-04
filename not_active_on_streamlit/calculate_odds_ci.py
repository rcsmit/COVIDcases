import numpy as np
def calculate_ci():
    import math
    # https://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions
    # https://select-statistics.co.uk/calculators/confidence-interval-calculator-odds-ratio/
    # https://www.ncbi.nlm.nih.gov/books/NBK431098/

    # 90%	1.64	1.28
    # 95%	1.96	1.65
    # 99%	2.58	2.33
    Za2 = 1.96

    # a = 17 #df.iloc[i]["positive_above_20_days_after_2nd_dose"]
    # b = 83#df.iloc[i]["healthy_vax"]
    # c = 1#df.iloc[i]["Sum_positive_without_vaccination"]
    # d = 99 #df.iloc[i]["healthy_nonvax"]
    a,b,c,d = 100, 5, 70, 12
    a,b,c,d = 26,44,247,1002
    a,b,c,d = 647,2,622,27
    a,b,c,d = 21712 ,   8, 21566,  162
    or_ = (a*d) / (b*c)

    theta = ()

    theta = (a/(a+c))/(b/(b+d)) # relative risk !!
    SE_theta = ((1/a) - (1/(a+c))  + (1/b)  - (1/(b+d)))**2
    print(f"CI_low  thetah { np.exp(np.log(theta) -  Za2 * SE_theta)}")
    print(f"tHETa relative risk { theta}")
    print(f"CI_low thetha { np.exp(np.log(theta) + Za2 * SE_theta)}")
    print()
    xxx = 1/a + 1/b + 1/c + 1/d
    # print (np.log(or_) )
    # print (xxx)
    # print( (Za2 * math.sqrt(xxx)))
    print(f"CI_low {np.exp(np.log(or_) - (Za2 * math.sqrt(xxx)))}")
    print(f"or_fisher_2 = {or_}")
    print(f"CI_high  {np.exp(np.log(or_) + (Za2 * math.sqrt(xxx)))}")
    #st.write(df)
    print()


    # https://stats.stackexchange.com/questions/506017/can-the-fisher-exact-test-be-used-to-get-a-confidence-interval-for-the-efficienc?rq=1
    # NIET OVEREENKOMEND MET Fisher in R
    p0 = a/c
    p1 = b/d
    n0 = a
    n1 = b
    rrr = n0*p0*(1-p0)
    sss = n1*p1*(1-p1)
    SE_log_odds = math.sqrt((1/rrr) + (1/sss))
    ttt = ((1/rrr) + (1/sss))
    print(SE_log_odds)
    print(np.exp (  SE_log_odds))
    # print(f"CI_low {np.exp(np.log(or_)-  (Za2 *  SE_log_odds))}")
    # print(f"or = {or_}")
    # print(f"CI_high  {np.exp(np.log(or_) +  (Za2*  SE_log_odds))}")
    print(f"CI_low {np.exp(np.log(or_) - (Za2 * math.sqrt(ttt)))}")
    print(f"or_fisher_2 = {or_}")
    print(f"CI_high  {np.exp(np.log(or_) + (Za2 * math.sqrt(ttt)))}")

def calculate_fisher_from_R():

    # https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/fisher.test
    # https://stats.stackexchange.com/questions/495142/why-is-my-fishers-test-significant-but-odds-ratio-overlaps-1
    ### data

    # mat = matrix(c(3200,6,885,6),2, byrow = T)

    # ### parameters describing the data
    # x = c(3194:3206)   ### possible values for cell 1,1
    # m = 3200+6    ### sum of row 1
    # n = 885+6     ### sum of row 2
    # k = 3200+885  ### sum of column 1


    # ### fisher test
    # test = fisher.test(mat)
    # test

    # ### manual computation of p-values
    # f = dhyper(x,m,n,k)
    # plot(x,f)
    # pvalue = sum(f[x >= 3200])

    # ### compare p-values (gives the same)
    # pvalue
    # test$p.value


    # ### non-central hypergemoetric distribution
    # ### copied from fisher.test function in R
    # ### greatly simplified for easier overview
    # logdc = dhyper(x, m, n, k, log = TRUE)

    # ### PDF
    # dnhyper = function(ncp) {
    # d = logdc + log(ncp) * x
    # d = exp(d - max(d))
    # d / sum(d)
    # }


    # ### CDF
    # pnhyper = function(q, ncp = 1, uppertail = F) {
    # if (uppertail) {
    #     sum(dnhyper(ncp)[x >= q])
    # }
    # else  {
    #     sum(dnhyper(ncp)[x <= q])
    # }
    # }
    # pnhyper = Vectorize(pnhyper)

    # ### alpha level
    # alpha = (1-0.95)/2

    # ### compute upper and lower boundaries
    # x1 = uniroot(function(t) pnhyper(3200, t) - alpha,
    #             c(0.5, 20))$root
    # x2 = uniroot(function(t) pnhyper(3200, t, uppertail = T) - alpha,
    #             c(0.5, 20))$root
    pass
calculate_ci()
