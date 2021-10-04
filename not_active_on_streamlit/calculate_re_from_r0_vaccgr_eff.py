def calculate_Re(R0, vacc_gr, eff):
    return R0 * (1 - ((vacc_gr/100)*(eff/100)))


for breakpoint in range (1,2):
    for R0 in range (2,20,1):
        for vacc_gr in range (100,101,1):
            for eff in range (0,101,1) :
                Re = calculate_Re(R0, vacc_gr, eff)
                if round(Re,1) <= breakpoint: #and round(Re,1)>=0.9:
                    print (f"{R0}  {vacc_gr}  {eff} {round(Re,1)}")
                    break
            if round(Re,1) <= breakpoint:
                break
    print()

#https://www.eurosurveillance.org/content/10.2807/1560-7917.ES.2021.26.20.2100428

