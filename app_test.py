import os

# import base64
import logging
from threading import Thread
from pyngrok import ngrok
import streamlit as st


# ----- Pyngrok Integration -----
def run_streamlit():
    os.system("streamlit run app.py --server.port 8501")


# Start Streamlit in a separate thread
thread = Thread(target=run_streamlit)
thread.start()

# Initialize ngrok tunnel
ngrok.set_auth_token(
    "2qTfDPgn368YBJYKTMLlcevGB2E_ZNNJ8jJNBkSSRwRxEEQa"
)  # Replace with your ngrok auth token
public_url = ngrok.connect(8501, "http")
# st.sidebar.success(f"🌐 Public URL: {public_url}")
print(f"🌐 Public URL: {public_url}")
# logging.info(f"Ngrok tunnel created at {public_url}")

# Ensure the thread remains active
thread.join()
