import importlib
import traceback
import streamlit as st

st.set_page_config(page_title="COVID Scripts of René Smit", layout="wide")

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

from covid_catalogue import options, CATEGORIES, CHOICE_TO_CAT_I, LETTER_TO_CAT_I, give_options_categories


def dynamic_import(module: str):
    """Import a module stored in a variable."""
    return importlib.import_module(module)


def show_info() -> None:
    """Render footer info in the sidebar."""
    tekst = (
        "<style> .infobox { background-color: lightblue; padding: 5px; }</style>"
        "<hr><div class='infobox'>Made by Rene Smit. "
        "(<a href='http://www.twitter.com/rcsmit' target='_blank'>@rcsmit</a>)<br>"
        'Sourcecode: <a href="https://github.com/rcsmit/COVIDcases" target="_blank">github.com/rcsmit/COVIDcases</a><br>'
        'How-to tutorial: <a href="https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d" target="_blank">rcsmit.medium.com</a><br>'
    )
    st.sidebar.markdown(tekst, unsafe_allow_html=True)


def main() -> None:
    """Main entry point: render sidebar navigation and dispatch to selected module."""
    # -----------------------------------------------------------------------
    # 1.  Read query params
    #     Supported:
    #       ?choice=42          → open script 42 (and its category)
    #       ?cat=C              → open category C, show its first script
    #       ?choice=42&cat=C    → choice wins; cat is kept in sync
    # -----------------------------------------------------------------------
    raw_choice = st.query_params.get("choice", None)
    raw_cat    = st.query_params.get("cat",    None)

    active_choice = 0
    if raw_choice is not None:
        try:
            c = int(raw_choice)
            if 0 <= c < len(options):
                active_choice = c
        except ValueError:
            pass

    if raw_choice is not None:
        active_cat_i = CHOICE_TO_CAT_I.get(active_choice, 0)
    elif raw_cat is not None:
        active_cat_i = LETTER_TO_CAT_I.get(raw_cat.upper(), 0)
        active_choice = CATEGORIES[active_cat_i][2][0]
    else:
        active_cat_i = 0

    # -----------------------------------------------------------------------
    # 2.  Sidebar — accordion categories with script buttons
    # -----------------------------------------------------------------------
    selected_choice = active_choice
    selected_cat_i  = active_cat_i

    with st.sidebar:
        st.markdown("### 🦠 COVID Scripts")
        st.caption("Pick a category, then a script. Options/parameters are below the menu.")
        st.markdown("---")

        for cat_i, (letter, cat_name, cat_indices, color) in enumerate(CATEGORIES):
            is_open = (cat_i == active_cat_i)

            with st.expander(f"**[{letter}]** {cat_name}", expanded=is_open):
                for idx in cat_indices:
                    label     = options[idx][0]
                    desc      = options[idx][2]
                    is_active = (idx == active_choice)

                    short     = label.split("] ", 1)[-1]
                    number    = label.split(" ", 1)[0]
                    btn_label = ("▶ " if is_active else "") + short

                    if st.button(
                        btn_label,
                        key=f"btn_{cat_i}_{idx}",
                        use_container_width=True,
                        help=f"{desc} {number}",
                        type="primary" if is_active else "secondary",
                    ):
                        selected_choice = idx
                        selected_cat_i  = cat_i

        st.markdown("---")


    # -----------------------------------------------------------------------
    # 3.  Keep both query params in sync so every URL is a valid deeplink
    # -----------------------------------------------------------------------
    current_letter = CATEGORIES[selected_cat_i][0]
    st.query_params["choice"] = str(selected_choice)
    st.query_params["cat"]    = current_letter

    # -----------------------------------------------------------------------
    # 4.  Dynamically import and run the selected module
    # -----------------------------------------------------------------------
    m = options[selected_choice][1].replace(" ", "_")

    try:
        module = dynamic_import(m)
    except Exception as e:
        st.error(f"Module '{m}' not found or error in the script")
        st.warning(str(e))
        st.warning(traceback.format_exc())
        st.stop()

    try:
        module.main()
        st.info(f"SCRIPT: https://github.com/rcsmit/COVIDcases/blob/main/{m}.py")
    except Exception as e:
        st.error(f"Function 'main()' in module '{m}' not found or error in the script")
        st.warning(str(e))
        st.warning(traceback.format_exc())
        st.stop()


if __name__ == "__main__":
    main()
    show_info()
