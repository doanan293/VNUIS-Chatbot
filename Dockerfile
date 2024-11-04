# FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
# Set the working directory
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        unixodbc \
        unixodbc-dev \
        freetds-dev \
        freetds-bin \
        tdsodbc \
        ffmpeg \
        git \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# RUN apt-get update \
#  && apt-get install unixodbc -y \
#  && apt-get install unixodbc-dev -y \
#  && apt-get install freetds-dev -y \
#  && apt-get install freetds-bin -y \
#  && apt-get install tdsodbc -y \
#  && apt-get install ffmpeg -y \
#  && apt-get install git -y \
#  && apt-get install --reinstall build-essential -y

# populate "ocbcinst.ini" as this is where ODBC driver config sits
# Populate "odbcinst.ini" with FreeTDS driver configuration
RUN echo "[FreeTDS]" >> /etc/odbcinst.ini && \
    echo "Description = FreeTDS Driver" >> /etc/odbcinst.ini && \
    echo "Driver = /usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so" >> /etc/odbcinst.ini && \
    echo "Setup = /usr/lib/x86_64-linux-gnu/odbc/libtdsS.so" >> /etc/odbcinst.ini

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set the LD_LIBRARY_PATH for ODBC libraries
ENV LD_LIBRARY_PATH="/opt/venv/lib:$LD_LIBRARY_PATH"

COPY requirements.txt /app
# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir -r requirements.txt --use-deprecated=legacy-resolver

COPY Model /app/Model
COPY Data /app/Data
COPY Embedding_Docs /app/Embedding_Docs
COPY Load_Data_DB /app/Load_Data_DB
COPY Processing_Data /app/Processing_Data
COPY Gen_Data /app/Gen_Data
COPY app.py /app
COPY chatbot.py /app
COPY Xu_ly_text.py /app/
COPY .env /app


CMD python3 Gen_Data/load_db_gendata.py && \
    python3 Embedding_Docs/embedding_docs.py && \
    python3 app.py

# CMD python3 Load_Data_DB/load_data_from_db.py && \
#     python3 Load_Data_DB/clean_data.py && \
#     python3 Gen_Data/gendata.py && \
#     python3 Embedding_Docs/embedding_docs.py && \
#     python3 app.py

#    python3 Processing_Data/prepocessing_data.py && \
#Tạo docker image: sudo docker build -t chatbot -f Dockerfile .
#Chạy docker container với gpu: sudo docker run --gpus all -p 8000:8000 chatbot
#Chạy docker container không gpu: sudo docker run -p 8000:8000 chatbot