def cell_background(val):
    """Creates the CSS code for a cell with a certain value to create a heatmap effect
    Args:
        val ([int]): the value of the cell

    Returns:
        [string]: the css code for the cell
    """
    try:
        v = abs(val)
        opacity = 1 if v >100 else v/100
        # color = 'green' if val >0 else 'red'
        if val > 0 :
             color = '193, 57, 43'
        elif val < 0:
            color = '1, 152, 117'
        else:
            color = '255,255,173'
    except:
        # give cells with eg. text or dates a white background
        color = '255,255,255'
        opacity = 1
    return f'background: rgba({color}, {opacity})'

def select_period(df, show_from, show_until):
    """ _ _ _ """
    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"

    mask = (df["Date_statistics"].dt.date >= show_from) & (df["Date_statistics"].dt.date <= show_until)
    df = df.loc[mask]
    df = df.reset_index()
    return df


def save_df(df, name):
    """  save dataframe on harddisk """
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\covid19_seir_models\\output\\"
    )

    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")


def drop_columns(df, what_to_drop):
    """  drop columns. what_to_drop : list """
    if what_to_drop != None:
        print("dropping " + str(what_to_drop))
        for d in what_to_drop:
            df = df.drop(columns=[d], axis=1)
    return df