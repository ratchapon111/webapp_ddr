import streamlit as st
from time import sleep
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.source_util import get_pages
import streamlit_authenticator as stauth


def get_current_page_name():
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Couldn't get script context")

    pages = get_pages("")

    return pages[ctx.page_script_hash]["page_name"]


def make_sidebar():
    with st.sidebar:
        st.title("Diabetic Retinopathy Detection App")
        st.write("")
        st.write("")

        if st.session_state.get("logged_in", True):
            st.page_link("pages/1_Prediction.py", label="Prediction")
            st.page_link("pages/2_Record.py", label="Record")

            st.write("")
            st.write("")

            if st.button("log out"):
                logout()

        elif get_current_page_name() != "HELLO":
            # If anyone tries to access a secret page without being logged in,
            # redirect them to the login page
            st.switch_page("HELLO.py")


def logout():
    st.session_state.logged_in = False
    sleep(0.5)
    st.switch_page("HELLO.py")
