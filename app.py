import streamlit as st
import base64
# from pyngrok import ngrok
# run_tts
from chatbot import ask, run_stt, run_tts
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
import logging
import time
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Float feature initialization
float_init()
# #-------------------Add Pyngrok---------------------
# # Function to initialize Ngrok tunnel
# def initialize_ngrok():
#     # Check if the tunnel is already created to prevent multiple tunnels on reruns
#     if "ngrok_tunnel" not in st.session_state:
#         # Optional: Set your Ngrok auth token if you have one
#         # Replace 'YOUR_NGROK_AUTH_TOKEN' with your actual token
#         ngrok_auth_token = "2qTfDPgn368YBJYKTMLlcevGB2E_ZNNJ8jJNBkSSRwRxEEQa"  # Replace with your token
#         if ngrok_auth_token != "2qTfDPgn368YBJYKTMLlcevGB2E_ZNNJ8jJNBkSSRwRxEEQa":
#             ngrok.set_auth_token(ngrok_auth_token)
        
#         # Start Ngrok tunnel on the default Streamlit port 8501
#         tunnel = ngrok.connect(8501, "http")
#         st.session_state.ngrok_url = tunnel.public_url
#         logging.info(f"Ngrok tunnel created at {tunnel.public_url}")

# # Initialize Ngrok
# initialize_ngrok()

# # Display the Ngrok URL in the sidebar for easy access
# if "ngrok_url" in st.session_state:
#     st.sidebar.success(f"🌐 **Public URL:** {st.session_state.ngrok_url}")
    
#-----------------------------------Done add Pyngrok----------------


# Create footer container and add content
footer_container = st.container()

# Streamlit UI
st.title("VNU-IS Chatbot")


# c = st.container()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(placeholder="Mời bạn nhập câu hỏi...")

# -----Set footer--------------------
with footer_container:
    # prompt = st.chat_input(placeholder="Mời bạn nhập câu hỏi...")
    audio_bytes = audio_recorder(
        text="",
        icon_size="1x",
        neutral_color="#a3a8b8",
        recording_color="#de1212",
    )
    if audio_bytes is not None:
        prompt = run_stt(audio_bytes)

# prompt = st.chat_input(placeholder="Mời bạn nhập câu hỏi...")

if prompt is not None and prompt != "":
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(ask(query=prompt))
        logging.info(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})

# Handle Play Audio button click
if len(st.session_state.messages) >= 1 :
    if st.button("📢 Play Audio"):
        logging.info("Button clicked: Play Audio")

        # Retrieve the last "assistant" response from the session state
        last_assistant_message = None
        for message in reversed(st.session_state.messages):
            if message["role"] == "assistant":
                last_assistant_message = message["content"]
                break

        if last_assistant_message:
            try:
                audio_base64 = run_tts(text=last_assistant_message, lang="vi")
                logging.info("TTS generation completed")

                if audio_base64:
                    # Decode the audio from base64
                    audio_bytes = base64.b64decode(audio_base64)
                    logging.info(f"Audio bytes length: {len(audio_bytes)}")
                    
                    # # Ensure audio is not empty
                    # if len(audio_bytes) > 0:
                    #     # Save the audio bytes to a temporary file
                    #     temp_audio_file = "generated_audio.wav"  # Adjust file name and extension if needed
                    #     with open(temp_audio_file, "wb") as f:
                    #         f.write(audio_bytes)

                        # Play the audio using st.audio and specify the correct format
                    st.audio(audio_bytes, format="audio/wav", autoplay = True)
                    # else:
                    #     st.error("Failed to generate valid audio data.")
                else:
                    st.error("Failed to generate audio.")
            except Exception as e:
                logging.error(f"Error during TTS or audio playback: {e}")
                st.error(f"Error processing audio: {e}")
        else:
            st.error("No response available to generate audio.")
footer_container.float(
    "position: fixed;bottom: 100px;right: -95px;background-color: white;padding-top: 0px"
)
