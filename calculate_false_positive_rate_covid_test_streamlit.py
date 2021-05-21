# CALCULATE FALSE DISCOVERY RATE AT VARIOUS LEVELS OF CONTAGIOUS PEOPLE ('besmettelijken')
# René Smit, 14 april 2021, MIT LICENSE

from tabulate import tabulate
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock



def calculate(test, prevalentie, number_of_tested_people, population, output):

    name = test[0]
    sensitivity = test[1]
    specificity = test[2]

    total_healthy = (1 - prevalentie) * number_of_tested_people
    total_sick = prevalentie * number_of_tested_people

    true_negative = round((total_healthy * specificity),0)
    false_negative = round((total_sick * (1 - sensitivity)),0)

    true_positive = round((total_sick * sensitivity),0)
    false_positive= round((total_healthy * (1 - specificity)),0)


    true_positive_bayes = round (100* (sensitivity * prevalentie) / ((sensitivity * prevalentie) + ((1-specificity)* (1-prevalentie)  )),2)

    fdr = round(100*false_positive/(false_positive+ true_positive),4)
    acc = round(100*((true_positive+true_negative)/number_of_tested_people),4)
    for_ = round(100*false_negative/(false_negative + true_negative),4)
    try:
        fpr = round(100*false_positive/(false_positive+ true_negative),4)
    except:
        fpr = 0
    fnr =  round(100*(false_negative/(true_positive+ false_negative)),4)
    pos = round(100*((true_positive + false_positive)/number_of_tested_people),4)
    ppv = round((true_positive/(true_positive+false_positive)*100),3)
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

        st.text (f"Prevalentie = {round(prevalentie*100,2)} % ({prevalentie*population}/{population})\nNumber of tested people : {number_of_tested_people}")


        st.text(f"Name test: {name} - specificity : {test[2]} - sensitivity : {test[1]}\n")

        st.text(tabulate(data, headers=["#", "'Person is\nSick' (+)\nSensitivity (TPR)", "'Person is\nHealthy' (-)\nSpecificity (TNR)", "Total"]))




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

def main():
    # [ name, sensitivity (% true positive), specificity (% true negative) ]

    #testen_ = [["Dimgrr", 1.0, 0.7], ["ROCHE SELFTEST", 0.8, 0.97],["BIOSYNEX SELFTEST", 0.972, 1.000],
    #  ["PCR TEST", 0.95, 0.998], ["PCR TEST WHO", 0.95, 0.97]]
    #testen_ =  [["PCR TEST", 0.95, 0.998]]
    testen_ = [["ROCHE SELFTEST", 0.8, 0.97]] #, ["PCR TEST", 0.95, 0.998]]
    #testen_ = [["538 TEST", 0.8, 0.999], ["PCR TEST", 0.95, 0.998]]
    #testen_ = [["538 TEST", 0.8, 0.7], ["PCR TEST", 0.95, 0.96]]
    #testen_ = [["Reumatest", 0.7, 0.8]]
    #testen_ = [["Wikipedia", 0.67, 0.91]] #https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Worked_example


    contagious  = (st.sidebar.number_input('Contagious',None,None, 174_835))
    population = (st.sidebar.number_input('Total population', None,None, 17_483_471))
    number_of_tested_people =  (st.sidebar.number_input('Number of tested people',None,None, 100_000))
    st.sidebar.write("Attention: too small numbers give erorrs (Division by zero)")

    scenario = st.sidebar.radio(
        "Select a test",
        ("PCR", "PCR RIVM best","PCR RIVM worst", "Sneltest Roche", "PCR WHO", "BIOSYNEX SELFTEST")
                )

    if scenario == "PCR":
        naam = "PCR"
        se = 0.998
        sp = 0.95
    elif scenario == "PCR WHO":
        naam = "PCR WHO"
        se = 0.95
        sp = 0.97
    elif scenario == "PCR RIVM best":
        naam = "PCR RIVM best"
        se = 0.98
        sp = 0.995
    elif scenario == "PCR RIVM worst":
        naam = "PCR RIVM worst"
        se = 0.67
        sp = 0.96
        
    elif scenario == "Sneltest Roche":
        naam = "Sneltest Roche"
        se = 0.8
        sp = 0.97
    elif scenario == "BIOSYNEX SELFTEST":
        naam = "BIOSYNEX SELFTEST"
        se = 0.972
        sp = 1.000
    name = (st.sidebar.text_input('Name', naam))
    specificity = (st.sidebar.number_input('Specificity',None,None, sp, format="%.4f"))
    sensitivity = (st.sidebar.number_input('Sensitivity',None,None, se, format="%.4f"))
    testen = [name, sensitivity, specificity]
    b = contagious
    prevalentie = b / population
    fdr, for_, pos, fpr = calculate(testen, prevalentie, number_of_tested_people, population, True)

    titel = (f"{testen[0]} - sensitivity {sensitivity} - specificity {specificity}")
    besm = []
    false_discovery_rate = []
    false_negative_rate = []
    chance_to_be_tested_positive = []
    false_positive_rate = []

    #population = 17_500_000 # The Netherlands
    #population = 9_000_000   # Israel
    #population = 100_000
    #population = 25_000_000 # AUSTRALIA
    # (un)comment next one line to have a loop of "contagious people"
    for b in range (int(population*0.001), population, int(population*0.001)):
        prevalentie = b / population
        fdr, for_, pos, fpr = calculate(testen, prevalentie, number_of_tested_people, population, False)
        besm.append(b)
        false_discovery_rate.append(fdr)
        false_positive_rate.append(fpr)
        false_negative_rate.append(for_)
        chance_to_be_tested_positive.append(pos)

    graph = True
    if graph :
        with _lock:
            fig1y = plt.figure()
            ax = fig1y.add_subplot(111)


            ax3 = ax.twinx()
            plt.title(titel)

            ax.set_xlabel(f'aantal besmettelijken (population = {population})')

            # (un)comment next lines (not) to   SHOW FALSE POS AND FALSE NEG RATE
            ax.plot(besm,false_discovery_rate,  'b')
            ax.set_ylabel('blue: false discovery rate (%)')


            ax3.plot(besm,false_negative_rate,'purple' )
            ax3.set_ylabel('purple: False omission rate (%)')

            # ax3.plot(besm,false_positive_rate,'g',  marker='o',)
            # ax3.set_ylabel('green: False positive rate (%)')



            # (un)comment next lines (not) to  SHOW CHANCE TO BE TESTED POSITIVE and FALSE POS RATE
            #ax.plot(besm,chance_to_be_tested_positive,'g',  marker='o',)
            #ax3.plot(besm,false_discovery_rate,  'r',marker='o',)
            #ax.set_ylabel('green: chance to be tested positive (%)')
            #ax3.set_ylabel('red: false discovery rate (%)')

            # plt.show()
            st.pyplot(fig1y)


        with _lock:
            fig1z = plt.figure()
            ax = fig1z.add_subplot(111)


            ax3 = ax.twinx()
            plt.title(titel)

            ax.set_xlabel(f'aantal besmettelijken (population = {population})')
            ax.set_xlim(0, int(population*0.02))
            # (un)comment next lines (not) to   SHOW FALSE POS AND FALSE NEG RATE
            ax.plot(besm,false_discovery_rate,  'b')
            ax.set_ylabel('blue: false discovery rate (%)')


            ax3.plot(besm,false_negative_rate,'purple')
            ax3.set_ylabel('purple: False omission rate (%)')

            # ax3.plot(besm,false_positive_rate,'g',  marker='o',)
            # ax3.set_ylabel('green: False positive rate (%)')



            # (un)comment next lines (not) to  SHOW CHANCE TO BE TESTED POSITIVE and FALSE POS RATE
            #ax.plot(besm,chance_to_be_tested_positive,'g',  marker='o',)
            #ax3.plot(besm,false_discovery_rate,  'r',marker='o',)
            #ax.set_ylabel('green: chance to be tested positive (%)')
            #ax3.set_ylabel('red: false discovery rate (%)')

            # plt.show()
            st.pyplot(fig1z)
            
    st.header ("Read this too")
    toelichting = ("<a href= 'https://virologydownunder.com/the-false-positive-pcr-problem-is-not-a-problem/' target='_blank'>The “false-positive PCR” problem is not a problem</a>)
    st.markdown(toelichting, unsafe_allow_html=True)           
                   

if __name__ == "__main__":
    main()
