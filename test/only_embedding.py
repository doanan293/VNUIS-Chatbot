"""from sentence_transformers import CrossEncoder, SentenceTransformer, util
import torch
import numpy as np
import pandas as pd
import textwrap
import time
import sys
import locale
import os
from dotenv import load_dotenv

load_dotenv()
# Load models and data once
embedding_path = os.getenv("EMBEDDING_MODEL")
embedding_model = SentenceTransformer(model_name_or_path=embedding_path, device="cuda")

pr_model_path = os.getenv("POST_RETRIEVAL_MODEL")
pr_model = CrossEncoder(model_name=pr_model_path, device="cuda")
# Load embeddings and data once
text_chunks_and_embedding_df = pd.read_csv(os.getenv("EMBEDDING_DATA_PATH"))
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" "))
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(
    "cuda")

# Define helper function to print wrapped text
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# Function to retrieve relevant resources
def retrieve_relevant_resources(query: str, n_resources_to_return: int = 1):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    context_items = [pages_and_chunks[i] for i in indices]

    results =  [item["Final_Answer"] for item in context_items]
    result = results[0]
    # print(ques_items)
    score = scores.item()

    return result, score

if __name__ == "__main__":
    sys.stdin.reconfigure(encoding='utf-8-sig')
    while True:
        query = input("Câu hỏi: ")
        if query.lower() in ['exit', 'quit']:
            break

        start_time = time.time()  # Start timing

        answer, score = retrieve_relevant_resources(query=query)
        # retrieve_relevant_resources(query=query)
        end_time = time.time()  # End timing

        response_time = end_time - start_time  # Calculate response time

        print("Trả lời:")
        if score >= 40:
            print_wrapped(answer)
        else:
            print("Xin lỗi bạn, câu hỏi của bạn không nằm trong kiến thức mà tôi có. Tôi sẽ cập nhật thêm kiến thức này ở phiên bản sau.")
        print(score)
        print(f"Thời gian phản hồi: {response_time:.2f} giây")

"""

from sentence_transformers import CrossEncoder, SentenceTransformer, util
import torch
import numpy as np
import pandas as pd
import textwrap
import time
import sys
import locale
import os
from dotenv import load_dotenv

load_dotenv()
# Load models and data once
embedding_path = os.getenv("PROJECTCB1_EMBEDDING_MODEL")
embedding_model = SentenceTransformer(model_name_or_path=embedding_path, device="cuda")

# pr_model_path = os.getenv("POST_RETRIEVAL_MODEL")
# pr_model = CrossEncoder(model_name=pr_model_path, device="cuda")
# Load embeddings and data once
text_chunks_and_embedding_df = pd.read_csv(os.getenv("PROJECTCB1_EMBEDDING_DATA_PATH"))
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df[
    "embedding"
].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
embeddings = torch.tensor(
    np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32
).to("cuda")


# Define helper function to print wrapped text
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)


# Function to retrieve relevant resources
def retrieve_relevant_resources(query: str, n_resources_to_return: int = 5):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    # Use cosine similarity instead of dot score
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=cosine_scores, k=n_resources_to_return)
    context_items = [pages_and_chunks[i] for i in indices]

    results = [item["Final_Answer"] for item in context_items]
    result = results[0]
    score = scores.item()
    # print(ques_items)

    return result, score


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8-sig")
    while True:
        query = input("Câu hỏi: ")
        if query.lower() in ["exit", "quit"]:
            break

        start_time = time.time()  # Start timing

        answer, score = retrieve_relevant_resources(query=query)
        # retrieve_relevant_resources(query=query)
        end_time = time.time()  # End timing

        response_time = end_time - start_time  # Calculate response time

        print("Trả lời:")
        if score >= 0:
            print_wrapped(answer)
        else:
            print(
                "Xin lỗi bạn, câu hỏi của bạn không nằm trong kiến thức mà tôi có. Tôi sẽ cập nhật thêm kiến thức này ở phiên bản sau."
            )
        print(score)
        print(f"Thời gian phản hồi: {response_time:.2f} giây")
