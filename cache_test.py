import streamlit as st
import time
from helpers import *
from streamlit.script_runner import RerunException


@st.cache(suppress_st_warning=True)  # 👈 Changed this
def expensive_computation(a, b):
    # 👇 Added this
    st.write("Cache miss: expensive_computation(", a, ",", b, ") ran")
    time.sleep(2)  # This makes the function take 2s to run
    return a * b

a = 2
b = 21
res = expensive_computation(a, b)
#select_period_input_cache()
st.write("Result:", res)

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")

my_slot0 = st.sidebar.empty()
my_slot1 = st.sidebar.empty()
my_slot0.info("Clear cache")
if my_slot1.button("Clear"):
    my_slot0.error("Do you really, really, wanna do this?")
    if my_slot1.button("Yes I'm ready to rumble"):
        caching.clear_cache()
        st.balloons()
        my_slot0.error("Cache is cleared, please reload to scrape new values")
        time.sleep(10)
        if my_slot1.button("reload"):
            raise RerunException
