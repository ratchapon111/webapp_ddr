
from navigation import make_sidebar
import yaml
from time import sleep
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError)



make_sidebar()


# Loading config file
with open(r'G:\project\webapp_ddr\config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# authenticator = stauth.Authenticate(
# r'G:\project\webapp_ddr\config.yaml')

# Creating a login widget
try:
    authenticator.login()

except LoginError as e:
    st.error(e)

# Authenticating user
if st.session_state['authentication_status']:
    authenticator.logout()
    sleep(0.5)

elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')


