import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import json

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
logging.info(f"Using {device}")
# device = "cpu"
model_path = os.getenv("PROJECTCB1_EMBEDDING_MODEL")
embedding_model = SentenceTransformer(
    model_name_or_path=model_path,
    device=device,
    # model_kwargs={"torch_dtype": "bfloat16"},
    trust_remote_code=True,
)
file_path = os.getenv("PROJECTCB1_DATA_DB")
# Đọc chỉ riêng cột "Final_Answer" và ép kiểu thành chuỗi
# df = pd.read_csv(file_path, usecols=["Question"], dtype={"Question": str})
df = pd.read_csv(file_path, usecols=["Relevant docs"], dtype={"Relevant docs": str})
# df = pd.read_csv(file_path)
df.drop_duplicates(keep="first", ignore_index=True, inplace=True)
# question_list = df["Question"].to_list()
question_list = df["Relevant docs"].to_list()

if device == "cuda":
    batch_size = 32
else:
    batch_size = 4

logging.info("Start encoding")
start_time = time.time()
embeddings = embedding_model.encode(
    question_list,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_tensor=False,
)
end_time = time.time()
load_time = end_time - start_time
print(load_time)

df["embedding"] = [json.dumps(embedding.tolist()) for embedding in embeddings]
logging.info("Start saving to file")
embeddings_df_save_path = os.getenv("PROJECTCB1_DATA_DB")
df.to_csv(embeddings_df_save_path, index=False)
logging.info("Saved_to file")


"""import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
# from huggingface_hub import login
import torch
from tqdm.auto import tqdm
import time

load_dotenv()
# Get the Hugging Face access token from environment variables
hf_token = os.getenv("PROJECTCB1_HUGGINGFACE_ACCESS_TOKEN")

# Log in to Hugging Face using the access token
if hf_token:
    login(token=hf_token)
else:
    print(
        "Access token not found. Please set the HUGGINGFACE_ACCESS_TOKEN in your .env file."
    )
# # Log in to Hugging Face using the access token
# if hf_token:
#     login(token=hf_token)
# else:
#     print("Access token not found. Please set the HUGGINGFACE_ACCESS_TOKEN in your .env file.")

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
embedding_model = SentenceTransformer(
    model_name_or_path=os.getenv("PROJECTCB1_EMBEDDING_MODEL"), device=device
)
file_path = os.getenv("PROJECTCB1_DATA_FINAL")
df = pd.read_csv(file_path)

question_list = df["Question"].to_list()
start_time = time.time()
embeddings = embedding_model.encode(
    question_list, batch_size=128, show_progress_bar=True, convert_to_tensor=False
)
end_time = time.time()
load_time = end_time - start_time
print(load_time)

df["embedding"] = embeddings
embeddings_df_save_path = os.getenv("PROJECTCB1_EMBEDDING_DATA_PATH")
df.to_csv(embeddings_df_save_path, index=False)"""

# df = df.to_dict(orient="records")
# start_time = time.time()
# for i in tqdm(df):
#     i["embedding"] = embedding_model.encode(i["Question"])
# end_time = time.time()
# load_time = end_time - start_time
# print(load_time)


# Define the batch size
# batch_size = 1  # You can adjust this depending on your memory constraints

# # Initialize an empty list to store embeddings
# embeddings = []

# # Process the embeddings in batches
# for i in range(0, len(df), batch_size):
#     batch_sentences = df["Question"].iloc[i : i + batch_size].tolist()
#     batch_embeddings = embedding_model.encode(batch_sentences)
#     embeddings.extend(batch_embeddings)

# start_time = time.time()
# question_embedding = embedding_model.encode(
#     question_list, batch_size=64, show_progress_bar=True
# )
# # Load the CSV file into a DataFrame
# end_time = time.time()
# load_time = end_time - start_time
# print(load_time)

# start_time = time.time()
# question_embedding = embedding_model.encode(
#     question_list, batch_size=64, convert_to_tensor=True, show_progress_bar=True
# )
# # Load the CSV file into a DataFrame
# end_time = time.time()
# load_time = end_time - start_time
# print(load_time)

# Define the batch size
# batch_size = 32  # You can adjust this depending on your memory constraints

# # Initialize an empty list to store embeddings
# embeddings = []


# Process the embeddings in batches
# for i in range(0, len(df), batch_size):
#     batch_sentences = df["Question"].iloc[i : i + batch_size].tolist()
#     batch_embeddings = embedding_model.encode(batch_sentences)
#     embeddings.extend(batch_embeddings)


# # Add the embeddings to the DataFrame
# df["embedding"] = embeddings

# # Save embeddings to file
# embeddings_df_save_path = os.getenv("PROJECTCB1_EMBEDDING_DATA_PATH")
# df.to_csv(embeddings_df_save_path, index=False, encoding="utf-8-sig")
