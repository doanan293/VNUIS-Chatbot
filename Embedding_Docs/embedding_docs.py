import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# from huggingface_hub import login
import torch
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Start embedding docs")
load_dotenv()
# # Get the Hugging Face access token from environment variables
# hf_token = os.getenv("PROJECTCB1_HUGGINGFACE_ACCESS_TOKEN")

# Log in to Hugging Face using the access token
# if hf_token:
#     login(token=hf_token)
# else:
#     print(
#         "Access token not found. Please set the HUGGINGFACE_ACCESS_TOKEN in your .env file."
#     )

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
embedding_model = SentenceTransformer(
    model_name_or_path=os.getenv("PROJECTCB1_EMBEDDING_MODEL"), device=device
)
file_path = os.getenv("PROJECTCB1_EMBEDDING_DATA_PATH")
df = pd.read_csv(file_path)

question_list = df["Question"].to_list()

if device == "cuda":
    batch_size = 128
else:
    batch_size = int(os.getenv("PROJECTCB1_EMBEDDING_DOCS_BATCH_SIZE"))

logging.info("Start encoding")
start_time = time.time()
embeddings = embedding_model.encode(
    question_list,
    batch_size=batch_size,
    show_progress_bar=False,
    convert_to_tensor=False,
)
end_time = time.time()
load_time = end_time - start_time
print(load_time)

df["embedding"] = list(embeddings)
logging.info("Start saving to file")
embeddings_df_save_path = os.getenv("PROJECTCB1_EMBEDDING_DATA_PATH")
df.to_csv(embeddings_df_save_path, index=False)
logging.info("Saved_to file")
