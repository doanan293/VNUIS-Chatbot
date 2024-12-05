import streamlit as st

from chatbot_final import ask, run_stt, run_tts
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Float feature initialization
float_init()

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

prompt = st.chat_input(placeholder="Mời bạn nhập câu hỏi...")

if prompt is not None and prompt != "":
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(ask(query=prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})

# Float the footer container and provide CSS to target it with
# footer_container.float("bottom: 0;background-color: white;")
footer_container.float(
    "position: fixed;bottom: 100px;right: -95px;background-color: white;padding-top: 0px"
)
