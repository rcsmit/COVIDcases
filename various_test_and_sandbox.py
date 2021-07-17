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

def main_2():
    # Python program to generate a heatmap
    # which displays the value in each cell
    # corresponding to the given dataframe

    # import required libraries
    import pandas as pd

    # defining index for the dataframe
    idx = ['1', '2', '3', '4']

    # defining columns for the dataframe
    cols = list('ABCD')

    # entering values in the index and columns
    # and converting them into a panda dataframe
    df = pd.DataFrame([[10, 20, 30, 40], [50, 30, 8, 15],
                    [25, 14, 41, 8], [7, 14, 21, 28]],
                    columns = cols, index = idx)

    # displaying dataframe as an heatmap
    # with diverging colourmap as virdis
    df.style.background_gradient(cmap ='viridis')\
            .set_properties(**{'font-size': '20px'})
def  make_legenda(max_value):
        stapfracties =   [0, 0.0625 , 0.125,  0.25,  0.50, 0.75,  1]
        stapjes =[]
        for i in range(len(stapfracties)):
            stapjes.append((stapfracties[i]*max_value))
        d = {'legenda': stapjes}

        df_legenda = pd.DataFrame(data=d)
        #st.write (df_legenda.style.format(None, na_rep="-").applymap(lambda x:  cell_background_test(x,max_value)).set_precision(2))
        #st.write (df_legenda.style.format(None, na_rep="-").background_gradient(axis=None)) #.set_precision(2))
        st.write(df_legenda.style.format(None, na_rep="-").background_gradient(axis=None)) #.set_precision(2))

def main():
    #st.header("This is just a sandbox")
    # Check the version
    #st.write(pd.__version__)
    # Check the version of the dependencies
    #pd.show_versions()
    make_legenda(150)
    #st.stop()

if __name__ == "__main__":
    main()


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.applymap.html


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