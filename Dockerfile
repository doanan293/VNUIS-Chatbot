# FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
# Set the working directory
WORKDIR /app


# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set the LD_LIBRARY_PATH for ODBC libraries
ENV LD_LIBRARY_PATH="/opt/venv/lib:$LD_LIBRARY_PATH"

COPY requirements.txt /app
# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install flash-attn --no-build-isolation
COPY . /app

CMD python3 Embedding_Docs/embedding_docs_pc.py && \
    streamlit run chatbot_midterm.py

# CMD python3 Load_Data_DB/load_data_from_db.py && \
#     python3 Load_Data_DB/clean_data.py && \
#     python3 Gen_Data/gendata.py && \
#     python3 Embedding_Docs/embedding_docs.py && \
#     python3 app.py

#    python3 Processing_Data/prepocessing_data.py && \
#Tạo docker image: sudo docker build -t chatbot -f Dockerfile .
#Chạy docker container với gpu: sudo docker run --gpus all -p 8000:8000 chatbot
#Chạy docker container không gpu: sudo docker run -p 8000:8000 chatbot