from tabulate import tabulate
import matplotlib.pyplot as plt


def calculate(test, besmettelijken, number_of_tested_people):
    prevalentie = besmettelijken / 17_500_000

    print (f"Prevalentie = {round(prevalentie*100,2)} % - Number of tested people : {number_of_tested_people}")
    print(" ")
    name = test[0]
    sensitivity = test[1]

    specificity = test[2]

    total_healthy = (1 - prevalentie) * number_of_tested_people
    total_sick = prevalentie * number_of_tested_people

    true_negative = int(total_healthy * specificity)
    false_negative = int(total_sick * (1 - sensitivity))

    true_positive = int(total_sick * sensitivity)
    false_positive = int(total_healthy * (1 - specificity))



    from tabulate import tabulate
    print(f"Name test: {name} - sensitivity : {test[1]} - specificity : {test[2]}")
    data = [
        [
            "Result: healthy",
            true_negative,
            false_negative,
            true_negative + false_negative,
        ],
        [
            "Result: sick  ",
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
    print(tabulate(data, headers=["", "Healthy", "Sick", "Total"]))

    print(
        f"False positivity rate: {round(100*false_positive/(false_positive + true_positive),1)} % - (Tested sick while you are healthy)"
    )
    print(
        f"False negativity rate: {round(100*false_negative/(false_negative + true_negative),3)} % - (Tested healthy while you are sick)"
    )

    print(" ")

    fpr = round(100*false_positive/(false_positive + true_positive),1)
    fnr = round(100*false_negative/(false_negative + true_negative),3)
    return fpr,fnr

def main():

    # [ name, sensitivity, specificity ]
    testen_ = [["ROCHE SELFTEST", 0.8, 0.97], ["PCR TEST", 0.95, 0.998]]
    #testen = ["ROCHE SELFTEST", 0.8, 0.97]
    #testen =  ["PCR TEST", 0.95, 0.998]
    for testen in testen_:
        titel = (f"{testen[0]} - sensitivity {testen[1]} - specificity {testen[2]}")
        besmettelijken = 100_000.0
        number_of_tested_people = 100_000.0
        besm = []
        false_positive_rate = []
        false_negative_rate = []
        for b in range (10_000, 200_000, 10_000):

            fpr, fnr = calculate(testen, b, number_of_tested_people)
            besm.append(b)
            false_positive_rate.append(fpr)
            false_negative_rate.append(fnr)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax3 = ax.twinx()
        plt.title(titel)
        ax.plot(besm,false_positive_rate,  'r',marker='o',)
        ax3.plot(besm,false_negative_rate,'g',  marker='o',)
        plt.xlabel('aantal besmettelijken in NL')
        ax.set_ylabel('false positive rate (%)')
        ax3.set_ylabel('false negative rate (%)')

        plt.show()

if __name__ == "__main__":
    main()
