# CALCULATE FALSE DISCOVERY RATE AT VARIOUS LEVELS OF CONTAGIOUS PEOPLE ('besmettelijken')
# René Smit, 14 april 2021, MIT LICENSE

from tabulate import tabulate
import matplotlib.pyplot as plt
import streamlit as st

# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock



def calculate(test, prevalentie, number_of_tested_people,  output):
    """calculates the values. Made as a function to be able to make graphs
    with various prevalences

    Args:
        test (list): list with the name, sensitivity and specificity of the test
        prevalentie (float): prevalentie of the virus
        number_of_tested_people (int)): how many people to test
        output (boolean): Shows the output as text or not

    Returns:
        fdr,for_, pos, fpr - see code
    """
    name = test[0]
    sensitivity = test[1]
    specificity = test[2]

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

    true_positive_bayes = round (100* (sensitivity * prevalentie) / ((sensitivity * prevalentie) + ((1-specificity)* (1-prevalentie)  )),2)

    fdr = round(100*false_positive/(false_positive+ true_positive),4)
    #print (fdr)
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
        st.text (f"Number of tested people : {number_of_tested_people}")

        st.text (f"Prevalentie testpopulatie = {round(prevalentie*100,2)}%  (= {round(number_of_tested_people*prevalentie)} 'sick' persons)")


        st.text(f"Name test: {name} - specificity : {test[2]} - sensitivity : {test[1]}\n")

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

def main():
    prevalentie, number_of_tested_people, name, specificity, sensitivity = interface()
    testen = [name, sensitivity, specificity]
    #b = contagious
    #prevalentie = b / population
    fdr, for_, pos, fpr = calculate(testen, prevalentie, number_of_tested_people, True)

    population = 17_500_000 # The Netherlands
    #population = 9_000_000   # Israel
    #population = 100_000
    #population = 25_000_000 # AUSTRALIA
    # (un)comment next one line to have a loop of "contagious people"

    graph =  st.sidebar.selectbox("Show Graph", [True, False], index=0)
    if graph:
        lijst_l = ["false_discovery_rate", "chance_to_be_tested_positive" ]
        lijst_r = ["false_omission_rate", "false_positive_rate", "false_discovery_rate"]
        what_to_show_l  = st.sidebar.multiselect(
                "What to show  (multiple possible)", lijst_l, ["false_discovery_rate" ]
            )
        what_to_show_r  = st.sidebar.multiselect(
                "What to show  (multiple possible)", lijst_r, ["false_omission_rate"]
            )

        chance_to_be_tested_positive = []
        false_positive_rate = []
        besm,prev = [],[]
        false_discovery_rate = []
        false_negative_rate = []
        titel = (f"{testen[0]} - sensitivity {sensitivity} - specificity {specificity}")

        for prevalentie in range (1,100,1):
            #prevalentie = b / population
            fdr, for_, pos, fpr = calculate(testen, prevalentie, number_of_tested_people,  False)
            prev.append(prevalentie)
            besm.append(prevalentie*population/100)
            false_discovery_rate.append(fdr)
            false_positive_rate.append(fpr)
            false_negative_rate.append(for_)
            chance_to_be_tested_positive.append(pos)


        # with _lock:
        fig1y = plt.figure()
        ax = fig1y.add_subplot(111)


        ax3 = ax.twinx()
        plt.title(titel)

        ax.set_xlabel(f'prevalentie testpopulation(%)')
        if "false_discovery_rate" in what_to_show_l:
            # print (prev)
            # print (false_discovery_rate)
            # (un)comment next lines (not) to   SHOW FALSE POS AND FALSE NEG RATE
            ax.plot(prev,false_discovery_rate,  'blue')
            ax.set_ylabel('blue: false discovery rate (%)')
        if "chance_to_be_tested_positive" in what_to_show_l:
        # (un)comment next lines (not) to  SHOW CHANCE TO BE TESTED POSITIVE and FALSE POS RATE
            ax.plot(prev,chance_to_be_tested_positive,'g',  )

            ax.set_ylabel('green: chance to be tested positive (%)')
        if "false_omission_rate" in what_to_show_r:

            ax3.plot(prev,false_negative_rate,'purple' )
            ax3.set_ylabel('purple: False omission rate (%)')
        if "false_positive_rate" in what_to_show_r:

            ax3.plot(prev,false_positive_rate,'g',  )
            ax3.set_ylabel('green: False positive rate (%)')



        if "false_discovery_rate" in what_to_show_r:
            ax3.plot(prev,false_discovery_rate,  'r',)
            ax3.set_ylabel('red: false discovery rate (%)')

        # plt.show()
        st.pyplot(fig1y)


        # with _lock:
        #     fig1z = plt.figure()
        #     ax = fig1z.add_subplot(111)


        #     ax3 = ax.twinx()
        #     plt.title(titel)

        #     ax.set_xlabel(f'aantal besmettelijken (population = {population})')
        #     ax.set_xlim(0, int(population*0.02))
        #     # (un)comment next lines (not) to   SHOW FALSE POS AND FALSE NEG RATE
        #     ax.plot(besm,false_discovery_rate,  'b')
        #     ax.set_ylabel('blue: false discovery rate (%)')


        #     ax3.plot(besm,false_negative_rate,'purple')
        #     ax3.set_ylabel('purple: False omission rate (%)')

        #     # ax3.plot(besm,false_positive_rate,'g',  )
        #     # ax3.set_ylabel('green: False positive rate (%)')



        #     # (un)comment next lines (not) to  SHOW CHANCE TO BE TESTED POSITIVE and FALSE POS RATE
        #     #ax.plot(besm,chance_to_be_tested_positive,'g',  )
        #     #ax3.plot(besm,false_discovery_rate,  'r',)
        #     #ax.set_ylabel('green: chance to be tested positive (%)')
        #     #ax3.set_ylabel('red: false discovery rate (%)')

        #     # plt.show()
        #     st.pyplot(fig1z)

    st.header ("Read this too")
    toelichting = ("Attention: prevalention is an output what normally can't be used as an input. Besides take note of the difference of the prevalence in the population in general and the testpopulation. There is also a difference between the different rates of the test itself and the testprocess as whole.<br><br>"

                  "<blockquote class='twitter-tweet' data-conversation='none'><p lang='en' dir='ltr'>88% false positive rate has some implications in Australia, where:<br>17 m tests<br>29886 positives<br>910 deaths.<br>12 % of 29886 = 3587.<br>-&gt; CFR = 25%<br>Reductio ad absurdum.<br>Suspect specificity is significantly higher than 99.8%</p>&mdash; Clayton Clent (@ClaytonClent) <a href='https://twitter.com/ClaytonClent/status/1390448827362996224?ref_src=twsrc%5Etfw'>May 6, 2021</a></blockquote> <script async src='https://platform.twitter.com/widgets.js' charset='utf-8'></script><br>"
                    "<a href= 'https://virologydownunder.com/the-false-positive-pcr-problem-is-not-a-problem/' target='_blank'>The “false-positive PCR” problem is not a problem</a><br>"

                  "<a href= 'https://larremorelab.github.io/covid19testgroup' target='_blank'>Various calculations around tests</a><br>"

                   "<a href= 'https://twitter.com/mus_nico/status/1397349689243144192' target='_blank'>Diverse tweets van Nico de Mus</a><br>"

                  )
    st.markdown(toelichting, unsafe_allow_html=True)

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        'Sourcecode : <a href="https://github.com/rcsmit/COVIDcases/blob/main/calculate_false_positive_rate_covid_test_streamlit.py" target="_blank">github.com/rcsmit</a><br>'
        'How-to tutorial : <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>')

    st.markdown(tekst, unsafe_allow_html=True)

def interface():
    prevalentie = st.sidebar.number_input('prevalence testpopulation in %',0.0,100.0, 1.0, format="%.4f")
    number_of_tested_people =  (st.sidebar.number_input('Number of tested people',None,None, 100_000))
    st.sidebar.write("Attention: too small numbers give erorrs (Division by zero)")

    scenario = st.sidebar.radio(
        "Select a test",
        ("PCR", "PCR RIVM best","PCR RIVM worst", "Sneltest Roche", "Sneltest OMT", "Sneltest Deepblue", "Sneltest Seven Eleven", "PCR WHO", "PCR Nico", "BIOSYNEX SELFTEST","Antigen rapid Roche", "Antigen MP Biomedicals")
                )

    if scenario == "BIOSYNEX SELFTEST":
        naam = "BIOSYNEX SELFTEST"
        se = 0.972
        sp = 1.000
    elif scenario == "PCR Nico":
        naam = "PCR Nico"
        se = 0.98
        sp = 0.99996 # https://twitter.com/mus_nico/status/1395724466043461633

    elif scenario == "Sneltest Deepblue Lazada":
        naam = "Sneltest Deepblue"
        se =  0.964
        sp =  0.998
    elif scenario == "Sneltest Seven Eleven":
        naam =  "Sneltest Seven Eleven"
        se =  0.9318
        sp =  0.9932
    elif scenario == "PCR RIVM best":
        naam = "PCR RIVM best"
        se = 0.98
        sp = 0.995
    elif scenario == "PCR RIVM worst":
        naam = "PCR RIVM worst"
        se = 0.67
        sp = 0.96

    elif scenario == "PCR WHO":
        naam = "PCR WHO"
        se = 0.95
        sp = 0.97
    elif scenario == "PCR":
        naam = "PCR"
        se = 0.998
        sp = 0.95
    elif scenario == "Sneltest OMT":
        naam = "Sneltest OMT"
        se = 0.8333
        sp = 0.998
    elif scenario == "Sneltest Roche":
        naam = "Sneltest Roche"
        se = 0.8333
        sp = 0.991
    elif scenario == "Antigen rapid Roche":
        naam = "Antigen rapid Roche"
        se = 0.89
        sp = 0.991
    elif scenario == "Antigen MP Biomedicals":
        naam = "Antigen MP Biomedicals"
        se = 0.965
        sp = 0.991 #https://twitter.com/dimgrr/status/1413157690721902599/photo/1


    # # https://www.ftm.nl/artikelen/testsamenleving-extreem-duur-veroorzaakt-uitsluiting-gezonde-mensen
    # https://www.rivm.nl/sites/default/files/2020-12/Toelichting%20betrouwbaarheid%20PCR.pdf
    name = (st.sidebar.text_input('Name', naam))
    specificity = (st.sidebar.number_input('Specificity',None,None, sp, format="%.4f"))
    sensitivity = (st.sidebar.number_input('Sensitivity',None,None, se, format="%.4f"))
    return prevalentie,number_of_tested_people,name,specificity,sensitivity


if __name__ == "__main__":
    main()
