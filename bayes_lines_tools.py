import pandas as pd
import numpy as np
import streamlit as st

def init():
    st.sidebar.subheader("Datapoints")
    rid =  st.sidebar.text_input("Report Indentifier","Lansingerland GGD", key="aa")
    tst =  st.sidebar.number_input("Number of tests", 0,None, 27000 )  
    pos =  st.sidebar.number_input("Positives reported", 0,None, 242 )
    match_range =  st.sidebar.number_input("Match range", 0.0,100.0, 99.99)
    st.sidebar.subheader("Sensitivity")
    se_min =st.sidebar.number_input("min",0,100,30,  key="d")
    se_max =st.sidebar.number_input("max",0,100,100, key="e")
    se_step =st.sidebar.number_input("step",0.0,100.0,.5,  key="f")
    st.sidebar.subheader("Specificity")
    sp_min =st.sidebar.number_input("min",0,100,70, key="a")
    sp_max =st.sidebar.number_input("max",0,100,100, key="b")
    sp_step =st.sidebar.number_input("step",0.0,100.0,.5, key="c")
    st.sidebar.subheader("Prevalence")
    prev_min =st.sidebar.number_input("min",0.0,100.0,.5, key="g")
    prev_max =st.sidebar.number_input("max",0.0,100.0,24.5, key="h")
    prev_step =st.sidebar.number_input("step",0.0,100.0,.5, key="i")
    return  rid, tst, pos, match_range, se_min,se_max,se_step, sp_min, sp_max, sp_step, prev_min, prev_max, prev_step



def calculate(df,  rid, tests, positives, match_range, sens_, spec_, prev_):
    # from percentage to fractions
    sensitivity, specificity,prevalence = sens_/100, spec_/100,  prev_/100
              
    has_disease = (tests * prevalence)
    hasnot_disease = (tests * (1 - prevalence)) 

    true_positives = round(tests * prevalence * sensitivity,0) 
    true_negatives = round(tests * (1 - prevalence) * specificity,0) 

    # formulas taken from the excel file, those in sql statement don't work
    false_positives = hasnot_disease - true_negatives # positives - true_positives           
    false_negatives = has_disease - true_positives # (tests - positives) - true_negatives  
                        
    tested_pos =  (false_positives + true_positives)
   
    if  ((tested_pos  / positives)> match_range/100) and ((tested_pos  / positives) < 1/match_range*100) :
        matching = True
        df_ =  pd.DataFrame([ {
                                    
                                    "rid": rid,
                                    "sens": sensitivity*100,
                                    "spec": specificity*100,
                                    "prev": prevalence*100,
                                    "TP": true_positives,
                                    "TN": true_negatives,
                                    "FP": false_positives,
                                    "TN": true_negatives,
                                    "FN": false_negatives,
                                    "pos": positives,
                                    "has_disease": has_disease,
                                    "hasnot_disease": hasnot_disease,
                                    "tested_pos": tested_pos,
                                    "matching": matching
                                    }]
                            )           

        df = pd.concat([df, df_],axis = 0)   
    return df

def main():
    df = pd.DataFrame()
    rid, tests, positives, match_range, se_min,se_max,se_step, sp_min, sp_max, sp_step, prev_min, prev_max, prev_step = init()
    # ranges are multiplied by 10 because range can't handle floatas
    for sens_ in np.arange (se_min,se_max,se_step):
        for spec_ in np.arange (sp_min, sp_max, sp_step):
            # Don't know why this is in the SQL statement
            # if sensitivity+specificity>1:
            #         prevalence = ((positives / tests + specificity - 1) / (sensitivity + specificity - 1) )
  
            for prev_ in np.arange ( prev_min, prev_max, prev_step):
                df = calculate(df, rid, tests, positives, match_range, sens_, spec_, prev_)

    # show the result
    st.header("Bayes Lines Tool (BLT)")
    st.write ("Reproduction of SQL code of https://bayeslines.org/ in Python")

    if len(df)>0:
        st.write("These Confusion Matrices were found")
        st.write(df)

        
    else:
        st.warning("No results for this data. Hint: Change ranges")
    st.write("Paper: https://zenodo.org/record/4600597#.YroY0nbP3rc")
    st.write("Aukema, Wouter, Kämmerer, Ulrike, Borger, Pieter, Goddek, Simon, Malhotra, Bobby Rajesh, McKernan, Kevin, & Klement, Rainer Johannes. (2021). Bayes Lines Tool (BLT) - A SQL-script for analyzing diagnostic test results with an application to SARS-CoV-2-testing. https://doi.org/10.5281/zenodo.4459271")
    
    st.write("Source python script: https://github.com/rcsmit/COVIDcases/blob/main/bayes_lines_tools.py")
    st.write("Reaction on this paper: https://www.pepijnvanerp.nl/2021/01/bayes-lines-tool-flawed/") 

main()