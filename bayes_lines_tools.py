import pandas as pd
import numpy as np
import streamlit as st

def init():
    reg = st.sidebar.text_input("Regio", "Lansingerland")
    rid =  st.sidebar.text_input("Regio ID","Lansingerland GGD")
    tst =  st.sidebar.number_input("Number of tests", 0,None, 27000 )  
    pos =  st.sidebar.number_input("Number of tests", 0,None, 242 )
    match_range =  st.sidebar.number_input("Match range", 0.0,100.0, 99.99)
    return reg, rid, tst, pos, match_range



def calculate(df, reg, rid, tests, positives, match_range, sens_, spec_, prev_):
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
                                    "reg": reg,
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
    df = None
    reg, rid, tests, positives, match_range = init()
    # ranges are multiplied by 10 because range can't handle floatas
    for sens_ in np.arange (30,100,.5):
        for spec_ in np.arange (70, 100,.5):
            # Don't know why this is in the SQL statement
            # if sensitivity+specificity>1:
            #         prevalence = ((positives / tests + specificity - 1) / (sensitivity + specificity - 1) )
  
            for prev_ in np.arange (.5,24.5,.5):
                df = calculate(df, reg, rid, tests, positives, match_range, sens_, spec_, prev_)

    # show the result
    st.write ("Reproduction of SQL code of https://bayeslines.org/ in Python")
    st.write(df)

main()