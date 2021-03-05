# COVIDcases

Various files with models and graphs concerning COVID-19

## Interesting files/scripts
* **number_of_cases_interactive.py** - Plotting the number of COVID cases with different values. Contains a SIR-graph and a classical SIR-model. Including immunity
*  [Open in Streamlit](https://share.streamlit.io/rcsmit/covidcases/main/number_of_cases_interactive.py)

* **masterfile_covid.py** - aggregates a lot of information and statistics from the Netherlands. Shows correlations and graphs. To download the newest data, 
change *download = True*. Maybe change some directory names

* **SEIR_hobbeland.py** -  Make an interactive version of the SEIR model, inspired by Hobbeland - https://twitter.com/MinaCoen/status/1362910764739231745

## Required files
* **mobilityR.csv** - Contains google- and apple mobility data for *masterfile_covid.py*
* **requirements.txt** - required to use *number_of_cases_interactive.py* at share.streamlit.io 

## OLD
* **mobility_vs_R_vs_hospitals.py** - first version of masterfile_covid
* **number_of_cases_decreasing_R.py** - old version of number_of_cases_interactive.py
* **plot.py** - plotting the number of cases, going from R0 to a list of R-values
* **sliding_r-number.py** - integrated in number_of_cases_interactive.py

### Installing
Just download or copy paste. Do not forget the required files
## Built With

* [Streamlit](http://www.streamlit.io/) - Streamlit
* Python


## Tutorial

See explanation how to make interactive graphs with Streamlit here 
https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d

## License
[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgments
* Thanks to [Han-Kwang Nienhuys](https://twitter.com/hk_nien) for his help on several topics and some code snippets (MIT-licensed)

## Coontact
* [Twitter @rcsmit](https://twitter.com/rcsmit)

