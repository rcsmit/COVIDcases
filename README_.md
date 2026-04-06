# COVIDcases

Various files with models and graphs concerning COVID-19. 

Accessible at https://share.streamlit.io/rcsmit/covidcases/main/covid_menu_streamlit.py

Sorry, sourcecode, input and outputfiles are mixed in one directory. If I'd do it again, I would split it up and make an utils.py file also. 

## Interesting files/scripts

* **1) covid_dashboard_rcsmit.py** - aggregates a lot of information and statistics from the Netherlands. Shows correlations and graphs. 
![image](https://user-images.githubusercontent.com/1609141/112730553-8b1cf680-8f32-11eb-83f6-1569f5114678.png)

 [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rcsmit/covidcases/main/covid_dashboard_rcsmit.py)

* **2) Plot_hosp_ic_streamlit.py** - Plot the number of hospital and ICU admissions per age in time in the Netherlands
<img width="877" alt="plothospIC" src="https://user-images.githubusercontent.com/1609141/118802804-e02a1880-b8a2-11eb-8772-cc495bf7bca8.png">
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rcsmit/covidcases/main/grafiek_pos_testen_per_leeftijdscategorie_streamlit.py)


**3) calculate_false_positive_rate_covid_test.py** - HOE BETROUWBAAR ZIJN DE TESTEN ?
![image](https://user-images.githubusercontent.com/1609141/115085095-2b4eb580-9f0a-11eb-8c1f-02642e846114.png)
![image](https://user-images.githubusercontent.com/1609141/115085050-14a85e80-9f0a-11eb-9732-87a78ffa73d3.png)

* **4) number_of_cases_interactive.py** - Plotting the number of COVID cases with different values. Contains a SIR-graph and a classical SIR-model. Including immunity

![image](https://user-images.githubusercontent.com/1609141/112731094-945b9280-8f35-11eb-8c3d-a99e5f48487d.png)

 [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py)


**5) calculate_ifr_prevalence.py** - calculate percentage of population who had covid and the IFR from the prevalence
![image](https://user-images.githubusercontent.com/1609141/115160069-8f05e980-a096-11eb-87f4-106738c6feed.png)


**6)fit_to_data_streamlit.py** - FIT THE DATA 
![image](https://user-images.githubusercontent.com/1609141/115085210-651fbc00-9f0a-11eb-99e6-6aa4504fd325.png)

* **7) SEIR_hobbeland.py** -  Make an interactive version of the SEIR model, inspired by Hobbeland - https://twitter.com/MinaCoen/status/1362910764739231745
![image](https://user-images.githubusercontent.com/1609141/112730583-adaf0f80-8f32-11eb-9517-0b2fd6443c42.png)

* **8) Show contact matrix**
* **9) r getal per provincie**
* **10) Cases from suspectibles**
* **11) Fit to data OWID**
* **12) Calculate R per country owid**
* **13) Covid dashboard OWID/Google or Waze**
* **14) Dag verschillen per leeftijd**
* **15) Calculate spec)/abs) humidity from rel) hum**
* **16) R getal per leeftijdscategorie**
* **17) Show rioolwaardes**
* **18) SIR model met leeftijdsgroepen**

* **19) grafiek_pos_testen_per_leeftijdscategorie_streamlit)py** - draw graphs of positieve cases per age in time**
 ![image](https://user-images)githubusercontent)com/1609141/112730260-e0f09f00-8f30-11eb-9bff-a835c2f965f7)png)**
 [![Open in Streamlit](https://static)streamlit)io/badges/streamlit_badge_black_white)svg)](https://share)streamlit)io/rcsmit/covidcases/main/grafiek_pos_testen_per_leeftijdscategorie_streamlit)py)**
 
* **20) per provincie per leeftijd**
* **21) kans om covid op te lopen**
![image](https://user-images.githubusercontent.com/1609141/140588996-b5ab2727-3fd8-40e4-ba7a-efe32f394b32.png)

* **22) Data per gemeente**
* **23) VE Israel**
* **24) Hosp/death NL**
* **25) VE Nederland**
* **26) Scatterplots QoG OWID** - Combine a lot of information of Qog and Our World in Data
* ![image](https://user-images.githubusercontent.com/1609141/140589065-ad81c492-5371-4cf6-91ba-e31552c337a5.png)

* **27) VE & CI calculations**
* **28) VE scenario calculator**
* **29) VE vs inv. odds**


## Help files
* **grafiek_pos_testen_per_leeftijdscategorie_PREPARE)py**
* **prepare_casuslandelijk.py** 
* 
## Required files
* **mobilityR.csv** - Contains google- and apple mobility data for *masterfile_covid.py*
* **requirements.txt** - required to use *number_of_cases_interactive.py* at share.streamlit.io 
 
## DATA FILES

* **knmi2.csv** - weather info
* **mobility.csv** - mobility info
* **pos_test_leeftijdscat_wekelijks.csv** - table 14 from 'wekelijks rapport RIVM' with tested/tested positive with different ageclasses -from 2020/8/1
* **postestennaarleeftijd2.csv** - table 14 from 'wekelijks rapport RIVM' with tested/tested positive with different ageclasses - from 2021/1/1
* **input_latest_age_pos_test_kids_seperated.csv** - table 14 from 'wekelijks rapport RIVM' with tested/tested positive with different ageclasses - from 2021/1/1 >20 in 10 years cohorts and kids still separated
* **input_latest_age_pos_tests.csv** - table 14 from 'wekelijks rapport RIVM' with tested/tested positive with different ageclasses - from 2021/1/1 >20 in 10 years cohorts and kids in one chohort
* **restrictions.csv** - restrictions in NL
* **SWEDEN_our_world_in_data.csv** - info about sweden
* **weektabel.csv** - all data from the dashboard aggregated per week - 2021

## Installing
Just download or copy paste. Do not forget the required files
## Built With
* Python
* [Streamlit](http://www.streamlit.io/) - Streamlit

## Tutorial
See explanation how to make interactive graphs with Streamlit here 
https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments
* Thanks to [Han-Kwang Nienhuys](https://twitter.com/hk_nien) for his help on several topics and some code snippets (MIT-licensed)
* Thanks to [Josette Schoenmakers](https://twitter.com/JosetteSchoenma) and [Maarten van den Berg](https://twitter.com/mr_Smith_Econ) for their inspiration, help and feedback

## Contact
* [Twitter @rcsmit](https://twitter.com/rcsmit)

