# import torch
# import numpy as np
# import pandas as pd
# from sentence_transformers import CrossEncoder, SentenceTransformer, util
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import TextStreamer
# import torch
# import streamlit as st

# device = "cuda" if torch.cuda.is_available() else "cpu"


# eb_model_path = "./Model/vnuis_embedding_bge"
# embeddings_path = "./Data/Embedding.csv"


# # Hàm nội bộ
# def load_embedding_model(embedding_model_path):
#     embedding_model = SentenceTransformer(
#         model_name_or_path=embedding_model_path, device=device, trust_remote_code=True
#     )
#     return embedding_model


# def load_embeddings(embeddings_path):
#     text_chunks_and_embedding_df = pd.read_csv(embeddings_path)

#     # Convert the embedding column from a JSON string to a list of floats
#     text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df[
#         "embedding"
#     ].apply(json.loads)

#     # Convert to PyTorch tensor
#     embeddings = torch.tensor(
#         np.array(text_chunks_and_embedding_df["embedding"].tolist()),
#         dtype=torch.float32,
#     ).to(device)

#     pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
#     return embeddings, pages_and_chunks


# # Khai báo các mô hình

# print("Loading models... ")
# # Load model embedding

# embedding_model = load_embedding_model(eb_model_path)


# embeddings, pages_and_chunks = load_embeddings(embeddings_path)  # Load embeddings

# device = "cuda"  # the device to load the model onto
# model_path = "SeaLLMs/SeaLLMs-v3-1.5B-Chat"
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map=device,
#     # load_in_4bit = True,
#     attn_implementation="flash_attention_2",
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# print("Finish load model")


# def retrieve_relevant_resources(
#     query: str, n_resources_to_return: int = 3, threshold: int = 0.1
# ):
#     query_embedding = embedding_model.encode(query, convert_to_tensor=True)
#     # Use cosine similarity instead of dot score
#     cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

#     # Get all scores and corresponding indices, then filter based on score > 0.5
#     scores, indices = torch.topk(input=cosine_scores, k=n_resources_to_return)
#     filtered_scores_indices = [
#         (score.item(), index.item())
#         for score, index in zip(scores, indices)
#         if score.item() > threshold
#     ]

#     # Extract the scores and indices after filtering
#     filtered_indices = [index for _, index in filtered_scores_indices]

#     # Take top 'n_resources_to_return' from the filtered list
#     # top_scores = filtered_scores[:n_resources_to_return]
#     top_indices = filtered_indices[:n_resources_to_return]

#     context_items = [pages_and_chunks[i] for i in top_indices]
#     results = [item["Relevant docs"] for item in context_items]
#     # ques = [item["Question"] for item in context_items]
#     # pr_results = reranking_model.rank(query, results, return_documents=True, top_k=5)
#     return results


# # Function to format the prompt
# def prompt_formatter_root(query: str, results: list) -> str:
#     context = '- "' + '"\n\n- "'.join(results) + '"'

#     base_prompt = """Hãy cho bản thân không gian để suy nghĩ bằng cách trích xuất các đoạn văn có liên quan từ ngữ cảnh dưới đây trước khi trả lời câu hỏi của người dùng.
# Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi của người dùng:

# {context}

# Đừng trả về cách suy nghĩ của bạn ví dụ như "dựa vào như cảnh, trong ngữ cảnh cung cấp,...". Trả về trực tiếp câu trả lời
# Câu hỏi của người dùng: {query}
# Trả lời:"""
#     prompt = base_prompt.format(context=context, query=query)
#     return prompt


# # def ask(query:str) -> str:
# # messages = [
# # {"role": "system", "content": "Bạn là một trợ lí Tiếng Việt hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác."},
# # ]


# def ask(query: str) -> str:
#     messages = [
#         {
#             "role": "system",
#             "content": "Bạn là Chatbot của Trường Quốc Tế - Đại học Quốc Gia Hà Nội. Hãy trả lời câu hỏi của người dùng một cách chính xác.",
#         },
#     ]
#     results = retrieve_relevant_resources(query, n_resources_to_return=3, threshold=0.3)
#     prompt = prompt_formatter_root(query, results)
#     messages.append({"role": "user", "content": prompt})
#     text = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)
#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         do_sample=True,
#         temperature=0.1,
#         top_k=20,
#         top_p=0.95,
#         max_new_tokens=512,
#         repetition_penalty=1.05,
#         streamer=streamer,
#     )
#     generated_ids = [
#         output_ids[len(input_ids) :]
#         for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

#     return response


# # -------------------------------------------------

# st.title("VNU-IS Chatbot")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         # {"role": "assistant", "content": "Mình là VNU-IS Chatbot, bạn cần hỗ trợ gì ạ?"}
#     ]

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# # Accept user input
# if prompt := st.chat_input("Mời bạn nhập câu hỏi..."):
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Generate response from the model
#     # response =

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         response = st.markdown(ask(query=prompt))
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import streamlit as st

# Define device only once
device = "cuda" if torch.cuda.is_available() else "cpu"

# File paths
eb_model_path = "./Model/vnuis_embedding_bge"
embeddings_path = "./Data/Embedding.csv"
model_path = "Qwen/Qwen2.5-3B-Instruct"


# Load models and embeddings
@st.cache_resource
def load_embedding_model(embedding_model_path, device):
    try:
        embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model_path,
            device=device,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")
    return embedding_model


@st.cache_data
def load_embeddings(embeddings_path, device):
    try:
        text_chunks_and_embedding_df = pd.read_csv(embeddings_path)

        # Convert the embedding column from JSON strings to lists of floats
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df[
            "embedding"
        ].apply(json.loads)

        # Convert embeddings to PyTorch tensors
        embeddings = torch.tensor(
            np.array(text_chunks_and_embedding_df["embedding"].tolist()),
            dtype=torch.float32,
        ).to(device)

        pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings: {e}")

    return embeddings, pages_and_chunks


@st.cache_resource
def load_chat_model(model_path, device):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load language model: {e}")

    return model, tokenizer


# Load models and embeddings
print("Loading models... ")
embedding_model = load_embedding_model(eb_model_path, device)
embeddings, pages_and_chunks = load_embeddings(embeddings_path, device)
model, tokenizer = load_chat_model(model_path, device)

print("Finished loading models.")


# Retrieval function
def retrieve_relevant_resources(
    query: str, n_resources_to_return: int = 3, threshold: float = 0.1
):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]

    # Get top scores with a threshold
    scores, indices = torch.topk(input=cosine_scores, k=n_resources_to_return)
    filtered_scores_indices = [
        (score.item(), index.item())
        for score, index in zip(scores, indices)
        if score.item() > threshold
    ]

    filtered_indices = [index for _, index in filtered_scores_indices]
    top_indices = filtered_indices[:n_resources_to_return]
    context_items = [pages_and_chunks[i] for i in top_indices]
    results = [item["Relevant docs"] for item in context_items]

    return results


# Prompt formatter
def prompt_formatter_root(query: str, results: list) -> str:
    context = '- "' + '"\n\n- "'.join(results) + '"'
    base_prompt = """Hãy cho bản thân không gian để suy nghĩ bằng cách trích xuất các đoạn văn có liên quan từ ngữ cảnh dưới đây trước khi trả lời câu hỏi của người dùng.
Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi của người dùng:

{context}

Câu hỏi của người dùng: {query}
Trả về trực tiếp câu trả lời, không cần câu dẫn dắt như "theo thông tin cung cấp, dựa vào ngữ cảnh,...".
Trả lời:"""
    return base_prompt.format(context=context, query=query)


# Generate response using model
def ask(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Tôi là Chatbot của Trường Quốc Tế - Đại học Quốc Gia Hà Nội. Hãy trả lời câu hỏi của người dùng một cách chính xác.",
        },
    ]
    results = retrieve_relevant_resources(query, n_resources_to_return=2, threshold=0.1)
    prompt = prompt_formatter_root(query, results)
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generated_ids = model.generate(
        model_inputs.input_ids,
        do_sample=True,
        temperature=0.1,
        top_k=40,
        top_p=0.95,
        max_new_tokens=512,
        repetition_penalty=1.05,
        streamer=streamer,
    )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# Streamlit UI
st.title("VNU-IS Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mời bạn nhập câu hỏi..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = ask(query=prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})