# CALCULATE FALSE DISCOVERY RATE AT VARIOUS LEVELS OF CONTAGIOUS PEOPLE ('besmettelijken')
# René Smit, 14 april 2021, MIT LICENSE

from tabulate import tabulate
import matplotlib.pyplot as plt

def calculate(test, prevalentie, number_of_tested_people, population):

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
    data = [
        [
            "Result: 'healthy' (-)",
            true_negative,
            false_negative,
            true_negative + false_negative,
        ],
        [
            "Result: 'sick' (+) ",
            false_positive,
            true_positive,
            false_positive+ true_positive,
        ],
        [
            "Total",
            (true_negative + false_positive),
            (false_negative + true_positive),
            (true_positive + false_positive + false_negative + true_negative),
        ],
    ]
    fdr = round(100*false_positive/(false_positive+ true_positive),4)
    acc = round(100*((true_positive+true_negative)/number_of_tested_people),4)
    for_ = round(100*false_negative/(false_negative + true_negative),4)
    fpr = round(100*false_positive/(false_positive+ true_negative),4)
    fnr =  round(100*(false_negative/(true_positive+ false_negative)),4)
    pos = round(100*((true_positive + false_positive)/number_of_tested_people),4)
    output = True
    if output:
        print("--------------------------------------------------------------------------")
        print()
        print (f"Prevalentie = {round(prevalentie*100,2)} % ({prevalentie*population}/{population})\nNumber of tested people : {number_of_tested_people}")
        print()

        print(f"Name test: {name} - specificity : {test[2]} - sensitivity : {test[1]}\n")

        print(tabulate(data, headers=["", "'Healthy' (-)\nSpecificity (TNR)", "'Sick' (+)\nSensitivity (TPR)", "Total"]))

        print()
        print(
            f"Positive predictive value (PPV)              : {round((true_positive/(true_positive+false_positive)*100),3)} % - (Chance of being really 'sick' when tested 'sick')"
        )
        print(
            f"Negative predictive value (NPV)              : {round((true_negative/(false_negative+true_negative)*100),3)} % - (Chance of being really 'healthy' when you are tested 'healthy')"
        )
        print()

        print(
            f"False Positive rate (FPR / α / Type 1 error) : {fpr} % - (chance of being tested 'sick' while being 'healthy' - probability of false alarm)"
        )
        print(
            f"False Negative rate (FNR/ β / type II error  : {fnr} % - (chance of being tested 'healty' while being 'sick' - miss rate)"
        )
        print()
        print(
            f"False Discovery rate (FDR)                   : {fdr} % - (Chance of being not 'sick' while you are tested 'sick')"
        )
        print(
            f"False Omission Rate (FOR)                    : {for_} % - (Chance of being 'sick' when you are tested  'healthy')"
        )
        print()
        # print(
        #      f"True positivity rate (Bayes): {true_positive_bayes} % ") # TOFIX


        # if true_positive_bayes!= (100-fdr):
        #     print (f"DIFFERENCE !")
        #     print (100-fdr-true_positive_bayes)
        # print()
        print(
            f"Accuracy                                     : {acc} % ")
        print(
            f"Chance to be tested positive (true & false)  : {pos} %"
        )
        print()
    return fdr,for_, pos, fpr

def main():
    # [ name, sensitivity (% true positive), specificity (% true negative) ]

    #testen_ = [["Dimgrr", 1.0, 0.7], ["ROCHE SELFTEST", 0.8, 0.97],["BIOSYNEX SELFTEST", 0.972, 1.000],
    #  ["PCR TEST", 0.95, 0.998], ["PCR TEST WHO", 0.95, 0.97]]
    testen_ =  [["PCR TEST", 0.95, 0.998]]
    #testen_ = [["ROCHE SELFTEST", 0.8, 0.97]] #, ["PCR TEST", 0.95, 0.998]]
    #testen_ = [["538 TEST", 0.8, 0.999], ["PCR TEST", 0.95, 0.998]]
    #testen_ = [["538 TEST", 0.8, 0.7], ["PCR TEST", 0.95, 0.96]]
    #testen_ = [["Reumatest", 0.7, 0.8]]
    #testen_ = [["Wikipedia", 0.67, 0.91]] #https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Worked_example
    for testen in testen_:
        titel = (f"{testen[0]} - sensitivity {testen[1]} - specificity {testen[2]}")
        besm = []
        false_discovery_rate = []
        false_negative_rate = []
        chance_to_be_tested_positive = []
        false_positive_rate = []

        population = 17_500_000 # The Netherlands
        #population = 9_000_000   # Israel
        #population = 100_000
        # (un)comment next one line to have a loop of "contagious people"
        #for b in range (5_000, 17_500_000, 100_000):

        # (un)comment next two lines to have 1 value of 'contagious people'
        #b = 200 # aantal besmettelijken
        b = 175_0
        if b != None: # line added to keep the code indented

            number_of_tested_people = 20000  # don't make too small to prevent rouding errors

            prevalentie = b / population
            fdr, for_, pos, fpr = calculate(testen, prevalentie, number_of_tested_people, population)
            besm.append(b)
            false_discovery_rate.append(fdr)
            false_positive_rate.append(fpr)
            false_negative_rate.append(for_)
            chance_to_be_tested_positive.append(pos)

        graph = True
        if graph :
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax3 = ax.twinx()
            plt.title(titel)

            ax.set_xlabel('aantal besmettelijken in NL (#)')

            # (un)comment next lines (not) to   SHOW FALSE POS AND FALSE NEG RATE
            ax.plot(besm,false_discovery_rate,  'r',marker='o',)
            ax.set_ylabel('red: false discovery rate (%)')


            #ax3.plot(besm,false_negative_rate,'g',  marker='o',)
            #ax3.set_ylabel('green: False non-discovery rate (%)')

            ax3.plot(besm,false_positive_rate,'g',  marker='o',)
            ax3.set_ylabel('green: False positive rate (%)')



            # (un)comment next lines (not) to  SHOW CHANCE TO BE TESTED POSITIVE and FALSE POS RATE
            #ax.plot(besm,chance_to_be_tested_positive,'g',  marker='o',)
            #ax3.plot(besm,false_discovery_rate,  'r',marker='o',)
            #ax.set_ylabel('green: chance to be tested positive (%)')
            #ax3.set_ylabel('red: false discovery rate (%)')

            plt.show()

if __name__ == "__main__":
    main()


"""
https://www.healthnewsreview.org/toolkit/tips-for-understanding-studies/understanding-medical-tests-sensitivity-specificity-and-positive-predictive-value/
Sensitivity measures how often a test correctly generates a positive result
for people who have the condition that’s being tested for
(also known as the “true positive” rate). A test that’s highly sensitive
will flag almost everyone who has the disease and not generate many
false-negative results. (Example: a test with 90% sensitivity will
correctly return a positive result for 90% of people who have the disease,
but will return a negative result — a false-negative — for 10% of the people who
have the disease and should have tested positive.)

Specificity measures a test’s ability to correctly generate a
negative result for people who don’t have the condition that’s
being tested for (also known as the “true negative” rate).
A high-specificity test will correctly rule out almost everyone
who doesn’t have the disease and won’t generate many
false-positive results. (Example: a test with 90% specificity will
correctly return a negative result for 90% of people who don’t have the disease, but will return a positive result — a false-positive — for 10% of the people who don’t have the disease and should have tested negative.)

SEE ALSO :
# https://lci.rivm.nl/covid-19/bijlage/aanvullend
"""
