import numpy as np
import math
import streamlit as stl
import plotly.express as px


def traditional(e,f,g,h,output):
    """Calcutes and VE with the CI on the traditional method
    rel_risk = (a/(a+c))/(b/(b+d)) # relative risk !!
    yyy = 1/a  + 1/c

    CI = ((1-(np.exp(np.log(rel_risk) +/- Za2 * math.sqrt(yyy))))*100

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
    Za2 = 1.96

    a = e # sick vax
    b = f # sick unfax
    c = g-e # healthy vax
    d =h-f # healthy unvax
    # https://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions
    #https://twitter.com/MrOoijer/status/1445074609506852878
    rel_risk = (a/(a+c))/(b/(b+d)) # relative risk !!
    yyy = 1/a  + 1/c

    if output == True:

        stl.write(f"VE Traditional method                    : {round((1- rel_risk)*100,2)}% [{round((1-(np.exp(np.log(rel_risk) +Za2 * math.sqrt(yyy))))*100,2)}-{round((1-(np.exp(np.log(rel_risk) -Za2 * math.sqrt(yyy))))*100,2)}]  ")


def interface():

    what = stl.sidebar.selectbox("Default values", ["hospital sept 2021", "ic sept 2021",  "cases mid sept - 31_10_21"], index=0)
    descr = stl.sidebar.text_input("Title in output", value=what)

    if what == "ic sept 2021":
        a_,b_,population_,vac_rate_old_, vac_rate_new_,days_ = 44,237,17400000,82,100,30
    elif what == "hospital sept 2021":
        a_,b_,population_,vac_rate_old_, vac_rate_new_,days_ = 340,1029,17400000,82,100,30
    elif what == "cases mid sept - 31_10_21":
        a_,b_,population_,vac_rate_old_, vac_rate_new_,days_ = 70318,82805,17400000,71,100,42
    number_days =stl.sidebar.number_input("Number of days", 0,100,value=days_)
    a= stl.sidebar.number_input("Sick | vax", value=a_)
    b= stl.sidebar.number_input("Sick | non vax", value=b_)
    population= stl.sidebar.number_input("Population total", value=population_)
    vac_rate_old= stl.sidebar.number_input("Vacc. rate old | all", 0,100,value=vac_rate_old_)
    vac_rate_new = stl.sidebar.number_input("Vacc. rate new | all",0,100, value=vac_rate_new_)
    on_y_axis=stl.sidebar.selectbox("On Y axis", [ "number_cases_new","number_cases_new_per_day", "difference_absolute",  "difference_percentage"], index=0)
    return a,b,population,vac_rate_old, vac_rate_new, on_y_axis, descr, number_days
def calculate(a,b,population,vac_rate_old, vac_rate_new, on_y_axis, output, number_days):
    number_vax = population * vac_rate_old/100
    number_non_vax = (population * (100-vac_rate_old))/100
    pvc = a/number_vax
    puc = b/number_non_vax

    rr = (a/number_vax)/(b/number_non_vax)
    traditional (a,b,number_vax, number_non_vax, output)
    #pfizer (a,b,number_vax, number_non_vax, output)
    if output == True:
        #stl.write (f"Proportion vaccinated {pvc} | Proportion non-vaccinated {puc}" )
        #stl.write (f"Cases 100% vax {int(pvc*population)} | Cases 0% vax {int(puc* population)}" )
        stl.write(f"Factor unvax vs vax : {round((puc/pvc),1)} x ")

    sick_vax_new = population *  vac_rate_new * pvc / 100
    sick_unvax_new = population * (100-vac_rate_new)*puc / 100
    sick_total_old = a+b
    sick_total_new = round(sick_vax_new + sick_unvax_new)
    sick_difference =  sick_total_new - sick_total_old
    sick_difference_percentage = round((( sick_total_new - sick_total_old) / sick_total_old)*100,1)
    if output == True:
        stl.write(f"Number of cases old {sick_total_old} | Per day {round(sick_total_old/number_days,1)}")
        stl.write(f"Numer of cases new {sick_total_new}  | Per day {round(sick_total_new/number_days,1)}")
        if sick_difference != 0 :
            stl.write(f"Difference in cases : {sick_difference} | Per day {round(sick_difference/number_days,1)} | ({sick_difference_percentage} %)")
        else:
            stl.write(f"No difference in cases : {sick_difference} ")

    if on_y_axis == "difference_absolute":
        y = sick_difference
    elif on_y_axis == "difference_percentage":
        y = sick_difference_percentage
    elif on_y_axis == "number_cases_new":
        y = sick_total_new
    elif on_y_axis == "number_cases_new_per_day":
        y = sick_total_new / number_days
    return y

def main():
    a,b,population,vac_rate_old, vac_rate_new, on_y_axis, descr, number_days = interface()

    stl.subheader ("VE scenario calculator (screening method)")
    stl.write("Very estimative. Doesn't take in account confounders, changes in R-number, variants etc.")

    stl.subheader (descr)
    calculate(a,b,population,vac_rate_old, vac_rate_new, on_y_axis, True, number_days)

    x_, y_ = [],[]

    #for x in range(int(vac_rate_old), 101):
    for x in range(0, 101):
        y = calculate(a,b,population,vac_rate_old, x, on_y_axis, False, number_days)
        x_.append(x)
        y_.append(y)

    fig = px.line(x=x_, y=y_, labels={
                     "x": "Vaccination percentage", "y": on_y_axis} ,   title = f"{descr} - {on_y_axis}")
    fig.add_vline(x=vac_rate_old)
    stl.plotly_chart(fig)

if __name__ == "__main__":
    main()