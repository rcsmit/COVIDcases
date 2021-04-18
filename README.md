# COVIDcases

Various files with models and graphs concerning COVID-19. 

Sorry, sourcecode, input and outputfiles are mixed in one directory. If I'd do it again, I would split it up and make an utils.py file also. 

## Interesting files/scripts
* **number_of_cases_interactive.py** - Plotting the number of COVID cases with different values. Contains a SIR-graph and a classical SIR-model. Including immunity

![image](https://user-images.githubusercontent.com/1609141/112731094-945b9280-8f35-11eb-8c3d-a99e5f48487d.png)

 [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py)

* **covid_dashboard_rcsmit.py** - aggregates a lot of information and statistics from the Netherlands. Shows correlations and graphs. 
![image](https://user-images.githubusercontent.com/1609141/112730553-8b1cf680-8f32-11eb-83f6-1569f5114678.png)

 [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rcsmit/covidcases/main/covid_dashboard_rcsmit.py.py)

* **grafiek_pos_testen_per_leeftijdscategorie_PREPARE.py**
* **grafiek_pos_testen_per_leeftijdscategorie_streamlit.py** - draw graphs of positieve cases per age in time
 ![image](https://user-images.githubusercontent.com/1609141/112730260-e0f09f00-8f30-11eb-9bff-a835c2f965f7.png)

 [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/rcsmit/covidcases/main/grafiek_pos_testen_per_leeftijdscategorie_streamlit.py.py)
* **SEIR_hobbeland.py** -  Make an interactive version of the SEIR model, inspired by Hobbeland - https://twitter.com/MinaCoen/status/1362910764739231745
![image](https://user-images.githubusercontent.com/1609141/112730583-adaf0f80-8f32-11eb-9517-0b2fd6443c42.png)

* **prepare_casuslandelijk.py** 
* **stackplot.py** - draw a stackplot of portion of agegroups in time for positive tests, hospitalizations and deceased
![image](https://user-images.githubusercontent.com/1609141/112730524-527d1d00-8f32-11eb-9747-9f41de65a80b.png)
![image](https://user-images.githubusercontent.com/1609141/112730428-cb2fa980-8f31-11eb-8349-1839c8ddd84c.png)

* **postestennaarleeftyd.py** - ZIJN KINDEREN DE REDEN DAT HET PERCENTAGE POSITIEF DAALT ?
![image](https://user-images.githubusercontent.com/1609141/112730409-ab988100-8f31-11eb-9c8c-742fe94f2f98.png)

* **getest_leeftijd_weekcijfers.py** -  IS ER EEN VERBAND MET HET PERCENTAGE POSITIEF PER LEEFTIJDSGROEP EN DE ZIEKENHUISOPNAMES?
![image](https://user-images.githubusercontent.com/1609141/112730368-7e4bd300-8f31-11eb-8b72-a6d39b579ea9.png)

**calculate_false_positive_rate_covid_test.py** - HOE BETROUWBAAR ZIJN DE TESTEN ?
![image](https://user-images.githubusercontent.com/1609141/115085095-2b4eb580-9f0a-11eb-8c1f-02642e846114.png)
![image](https://user-images.githubusercontent.com/1609141/115085050-14a85e80-9f0a-11eb-9732-87a78ffa73d3.png)

**fit_to_data_streamlit.py** - FIT THE DATA 
![image](https://user-images.githubusercontent.com/1609141/115085210-651fbc00-9f0a-11eb-99e6-6aa4504fd325.png)

**calculate_ifr_prevalence.py** - calculate percentage of population who had covid and the IFR from the prevalence
![image](https://user-images.githubusercontent.com/1609141/115160069-8f05e980-a096-11eb-87f4-106738c6feed.png)

## Required files
* **mobilityR.csv** - Contains google- and apple mobility data for *masterfile_covid.py*
* **requirements.txt** - required to use *number_of_cases_interactive.py* at share.streamlit.io 

## OLD / NOT MAINTAINED
* **mobility_vs_R_vs_hospitals.py** - first version of masterfile_covid
* **number_of_cases_decreasing_R.py** - old version of number_of_cases_interactive.py
* **plot.py** - plotting the number of cases, going from R0 to a list of R-values
* **sliding_r-number.py** - integrated in number_of_cases_interactive.py
* **number_of_cases_interactive_IC.py** - try to calculate the number of people at the ICU. Estimate too high due, probably due to lower ICU-rates when there are more cases
* **masterfile_covid.py** - continued in covid_dashboard_rcsmit.py

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

