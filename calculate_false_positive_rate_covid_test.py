# CALCULATE FALSE POSITIVITY RATE AT VARIOUS LEVELS OF INFECTED PEOPLE ('besmettelijken')
# René Smit, 14 april 2021, MIT LICENSE

from tabulate import tabulate
import matplotlib.pyplot as plt

def calculate(test, prevalentie, number_of_tested_people):

    name = test[0]
    sensitivity = test[1]
    specificity = test[2]

    total_healthy = (1 - prevalentie) * number_of_tested_people
    total_sick = prevalentie * number_of_tested_people

    true_negative = round((total_healthy * specificity),0)
    false_negative = round((total_sick * (1 - sensitivity)),0)

    true_positive = round((total_sick * sensitivity),0)
    false_positive = round((total_healthy * (1 - specificity)),0)
    data = [
        [
            "Result: healthy (-)",
            true_negative,
            false_negative,
            true_negative + false_negative,
        ],
        [
            "Result: sick (+) ",
            false_positive,
            true_positive,
            false_positive + true_positive,
        ],
        [
            "Total",
            (true_negative + false_positive),
            (false_negative + true_positive),
            (true_positive + false_positive) + (false_negative + true_negative),
        ],
    ]
    fpr = round(100*false_positive/(false_positive + true_positive),1)
    fnr = round(100*false_negative/(false_negative + true_negative),3)
    pos = 100*(true_positive + false_positive)/number_of_tested_people
    output = False
    if output == True:
        print("--------------------------------------------------------------------------")
        print()
        print (f"Prevalentie = {round(prevalentie*100,2)} % ({prevalentie*17500000}/17.5 miljoen)\nNumber of tested people : {number_of_tested_people}")
        print()

        print(f"Name test: {name} - sensitivity : {test[1]} - specificity : {test[2]}")

        print(tabulate(data, headers=["", "Healthy (-)\nSpecificity", "Sick (+)\nSensitivity", "Total"]))

        print()
        print(
            f"Positive predictive value (PPV): {100-fpr} % - (Tested sick while you are sick)"
        )
        print(
            f"Negative predictive value (NPV): {100-fnr} % - (Tested sick while you are sick)"
        )
        print()
        print(
            f"False positivity rate: {fpr} % - (Tested sick while you are healthy)"
        )
        print(
            f"False negativity rate: {fnr} % - (Tested healthy while you are sick)"
        )
        print()
        print(
            f"Chance to be tested positive (true & false): {pos} %"
        )
        print()
    return fpr,fnr, pos

def main():
    # [ name, sensitivity (% true positive), specificity (% true negative) ]

    #testen_ = [["Dimgrr", 1.0, 0.7], ["ROCHE SELFTEST", 0.8, 0.97],["BIOSYNEX SELFTEST", 0.972, 1.000],
    #  ["PCR TEST", 0.95, 0.998], ["PCR TEST WHO", 0.95, 0.97]]

    testen_ = [["ROCHE SELFTEST", 0.8, 0.97], ["PCR TEST", 0.95, 0.998]]

    for testen in testen_:
        titel = (f"{testen[0]} - sensitivity {testen[1]} - specificity {testen[2]}")
        besm = []
        false_positive_rate = []
        false_negative_rate = []
        chance_to_be_tested_positive = []

        for b in range (5_000, 200_000, 20_000):
        # b = 50_000 # aantal besmettelijken
        # if b != None:
            number_of_tested_people = 10000.0  # don't make too small to prevent rouding errors

            prevalentie = b / 17_500_000
            fpr, fnr, pos = calculate(testen, prevalentie, number_of_tested_people)
            besm.append(b)
            false_positive_rate.append(fpr)
            false_negative_rate.append(fnr)
            chance_to_be_tested_positive.append(pos)

        graph = True
        if graph == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax3 = ax.twinx()
            plt.title(titel)

            # SHOW FALSE POS AND FALSE NEG RATE
            # ax.plot(besm,false_positive_rate,  'r',marker='o',)
            # ax3.plot(besm,false_negative_rate,'g',  marker='o',)
            # ax.set_ylabel('red: false positive rate (%)')
            # ax3.set_ylabel('green: false negative rate (%)')

            # SHOW CHANCE TO BE TESTED POSITIVE and FALSE POS RATE
            ax.plot(besm,chance_to_be_tested_positive,'g',  marker='o',)
            ax3.plot(besm,false_positive_rate,  'r',marker='o',)
            ax.set_ylabel('green: chance to be tested positive (%)')
            ax3.set_ylabel('red: false positive rate (%)')

            

            ax.set_xlabel('aantal besmettelijken in NL (#)')
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