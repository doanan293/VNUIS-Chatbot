# from pydantic import BaseModel
import base64
import multiprocessing as mp
import os

# from chatbot import retrieve_relevant_resources
import tempfile
import time
from contextlib import asynccontextmanager

import healthcheck_python as hp
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from chatbot import retrieve_relevant_resources, run_stt, run_tts
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("Start app.py")
# Set the PY_HEALTH_MULTIPROC_DIR to a temporary directory
os.environ["PY_HEALTH_MULTIPROC_DIR"] = tempfile.mkdtemp()
load_dotenv()


# class QueryRequest(BaseModel):
#     query: str


# class AnswerRequest(BaseModel):
#     answer: str


# Define the lifespan event handler for managing startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    hp.start_http_server(port=8080)
    global p1, p2
    p1 = P1(timeout=10)
    p2 = P1()
    p1.start()
    p2.start()

    try:
        yield
    finally:
        p1.close()
        p2.close()
        p1.join()
        p2.join()


# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# ------sunteco does not use this-------
# # Set up CORS middleware
origins = os.getenv("PROJECTCB1_CORS", "").split(";")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------
# Create an instance of HTTPBearer for Bearer token authentication
security = HTTPBearer()


# Function to verify the Bearer token
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != os.getenv(
        "PROJECTCB1_VALID_API_KEY"
    ):  # Replace with your token verification logic
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Define the FastAPI endpoint with authorization
@app.post("/api/v1/cb1/ask")
def ask_question(
    query: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token),
):
    # query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    answer, score = retrieve_relevant_resources(query=query)

    if score >= 0.5:
        return JSONResponse(content={"answer": answer})
    else:
        return JSONResponse(
            content={
                "answer": "Xin lỗi bạn, câu hỏi của bạn nằm ngoài phạm vi dịch vụ bạn đang sử dụng."
            }
        )


@app.post("/api/v1/cb1/get-text-from-audio")
async def speech_to_text(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    transcription = run_stt(audio_bytes)
    return JSONResponse(content={"transcription": transcription})


# request: QueryRequest
@app.post("/api/v1/cb1/get-audio-from-text")
async def get_audio_from_text(answer: str):
    # answer = request.answer
    if not answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty")

    # Xử lý text-to-speech
    wav_data = run_tts(text=answer)
    audio_base64 = base64.b64encode(wav_data).decode("utf-8")
    return JSONResponse(content={"audio": audio_base64})


# Define the P1 process class
class P1(mp.Process):
    def __init__(self, timeout=0):
        super().__init__()
        self._stop_bit = mp.Event()
        self.timeout = timeout

    def close(self) -> None:
        self._stop_bit.set()

    def healthcheck(self):
        return True, "Healthcheck is OK"

    def do_something(self):
        time.sleep(5)

    def run(self):
        hp.init_check(timeout=self.timeout)
        hp.add_check(self.healthcheck)

        hp.live()
        while not self._stop_bit.is_set():
            hp.healthy()
            self.do_something()


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
