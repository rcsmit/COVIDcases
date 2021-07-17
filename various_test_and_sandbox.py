import streamlit as st
import pandas as pd

def cell_background_test(val,max):
    """Creates the CSS code for a cell with a certain value to create a heatmap effect
    Args:
        val ([int]): the value of the cell

    Returns:
        [string]: the css code for the cell
    """
    opacity = 0
    try:
        v = abs(val)
        color = '193, 57, 43'
        value_table = [ [0,0],
                        [0.25,0.25],
                        [0.50,0.50],
                        [0.75,0.75],
                        [1,1]]
        for vt in value_table:
            if v >= round(vt[0]*max) :
                opacity = vt[1]
    except:
        # give cells with eg. text or dates a white background
        color = '255,255,255'
        opacity = 1
    return f'background: rgba({color}, {opacity})'

def  make_legenda(max_value):
        stapfracties =   [0, 0.0625 , 0.125,  0.25,  0.50, 0.75,  1]
        stapjes =[]
        for i in range(len(stapfracties)):
            stapjes.append((stapfracties[i]*max_value))
        d = {'legenda': stapjes}

        df_legenda = pd.DataFrame(data=d)
        st.write (df_legenda.style.format(None, na_rep="-").applymap(lambda x:  cell_background_test(x,max_value)).set_precision(2))

def main():
    st.header("This is just a sandbox")
    make_legenda(150)

if __name__ == "__main__":
    main()


# Gives error

# _translate() missing 2 required positional arguments: 'sparse_index' and 'sparse_cols'

# Traceback (most recent call last): File "/app/covidcases/covid_menu_streamlit.py", line 63, in main module.main() 
# File "/app/covidcases/show_contactmatrix.py", line 66, 
# in main st.write (df_first_pivot.style.format(None, na_rep="-").applymap(lambda x: cell_background_helper(x,"lineair", 10, None)).set_precision(2))

# File "/home/appuser/venv/lib/python3.7/site-packages/streamlit/elements/write.py", line 181, in write self.dg.dataframe(arg) 

# File "/home/appuser/venv/lib/python3.7/site-packages/streamlit/elements/data_frame.py", line 85, in dataframe marshall_data_frame(data, data_frame_proto)

# File "/home/appuser/venv/lib/python3.7/site-packages/streamlit/elements/data_frame.py", line 150, in marshall_data_frame _marshall_styles(proto_df.style, df, styler) 

# File "/home/appuser/venv/lib/python3.7/site-packages/streamlit/elements/data_frame.py", line 169, in _marshall_styles translated_style = styler._translate() 

# TypeError: _translate() missing 2 required positional arguments: 'sparse_index' and 'sparse_cols'