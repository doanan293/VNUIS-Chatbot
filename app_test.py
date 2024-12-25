import os
# import base64
import logging
from threading import Thread
from pyngrok import ngrok
import streamlit as st
# from chatbot import ask, run_stt, run_tts
# from audio_recorder_streamlit import audio_recorder
# from streamlit_float import *

# # Logging configuration
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # Float feature initialization
# float_init()

# # Set title for Streamlit app
# st.title("VNU-IS Chatbot")

# # Initialize session state for messages
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display previous chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Input prompt for user message
# prompt = st.chat_input(placeholder="Mời bạn nhập câu hỏi...")

# # Create footer container for audio recording
# footer_container = st.container()

# # ----- Footer and Audio Recording -----
# with footer_container:
#     audio_bytes = audio_recorder(
#         text="",
#         icon_size="1x",
#         neutral_color="#a3a8b8",
#         recording_color="#de1212",
#     )
#     if audio_bytes is not None:
#         prompt = run_stt(audio_bytes)

# if prompt is not None and prompt != "":
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("assistant"):
#         response = st.write_stream(ask(query=prompt))
#         logging.info(response)

#     st.session_state.messages.append({"role": "assistant", "content": response})

# # Handle Play Audio button
# if len(st.session_state.messages) >= 1:
#     if st.button("📢 Play Audio"):
#         last_assistant_message = next(
#             (msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "assistant"), None
#         )
#         if last_assistant_message:
#             try:
#                 audio_base64 = run_tts(text=last_assistant_message, lang="vi")
#                 if audio_base64:
#                     audio_bytes = base64.b64decode(audio_base64)
#                     st.audio(audio_bytes, format="audio/wav", autoplay=True)
#                 else:
#                     st.error("Failed to generate audio.")
#             except Exception as e:
#                 logging.error(f"Error during TTS or audio playback: {e}")
#                 st.error(f"Error processing audio: {e}")
#         else:
#             st.error("No response available to generate audio.")

# footer_container.float(
#     "position: fixed;bottom: 100px;right: -95px;background-color: white;padding-top: 0px"
# )

# ----- Pyngrok Integration -----
def run_streamlit():
    os.system("streamlit run app.py --server.port 8501")

# Start Streamlit in a separate thread
thread = Thread(target=run_streamlit)
thread.start()

# Initialize ngrok tunnel
ngrok.set_auth_token('2qTfDPgn368YBJYKTMLlcevGB2E_ZNNJ8jJNBkSSRwRxEEQa')  # Replace with your ngrok auth token
public_url = ngrok.connect(8501, "http")
# st.sidebar.success(f"🌐 Public URL: {public_url}")
print(f"🌐 Public URL: {public_url}")
logging.info(f"Ngrok tunnel created at {public_url}")

# Ensure the thread remains active
thread.join()
