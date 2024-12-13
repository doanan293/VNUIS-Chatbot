import torch
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from underthesea import sent_tokenize
from vinorm import TTSnorm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import io
import soundfile as sf  # Thư viện xử lý âm thanh
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pydub import AudioSegment
from io import BytesIO
import librosa
from routes.Xu_ly_text import Xu_ly_text, Xu_ly_text_de_doc
import re
from dotenv import load_dotenv
import os
import timeit

# -----------------------
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# from huggingface_hub import login
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


# Configure Chrome options
def setup_chrome():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--ignore-certificate-errors")  # Bỏ qua lỗi chứng chỉ SSL
    options.add_argument("--allow-insecure-localhost")  # Cho phép kết nối không an toàn
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    # service=Service("/usr/bin/chromedriver"),,
    # Initialize the WebDriver
    driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=options)
    return driver


driver = setup_chrome()

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------Define function to load Model and Data---------------------------
# Hàm nội bộ
def load_embedding_model(embedding_model_path, device):
    try:
        embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model_path,
            model_kwargs={"torch_dtype": "bfloat16"},
            device=device,
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")
    return embedding_model


# def load_reranking_model(pr_model_path):
#     pr_model = CrossEncoder(model_name=pr_model_path, device=device)
#     return pr_model


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
            dtype=torch.bfloat16,
        ).to(device)

        pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings: {e}")

    return embeddings, pages_and_chunks


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_tts(xtts_checkpoint, xtts_config, xtts_vocab):
    clear_gpu_cache()
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)

    use_deepspeed = torch.cuda.is_available()

    XTTS_MODEL.load_checkpoint(
        config,
        checkpoint_path=xtts_checkpoint,
        vocab_path=xtts_vocab,
        use_deepspeed=use_deepspeed,
    )
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()
    return XTTS_MODEL


def load_model_stt(stt_model_path: str):
    processor = Wav2Vec2Processor.from_pretrained(stt_model_path)
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_path).to(device)
    return processor, stt_model


def load_chat_model(model_path, device):
    try:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="flash_attention_2",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # model.eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=device,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load language model: {e}")

    return model, tokenizer


# -----------------------Load model and data--------------------------------------------------

print("Loading models... ")
# Load model embedding
eb_model_path = os.getenv("PROJECTCB1_EMBEDDING_MODEL")
embedding_model = load_embedding_model(
    embedding_model_path=eb_model_path, device=device
)

# Load data
embeddings_path = os.getenv("PROJECTCB1_EMBEDDING_DATA_PATH")
embeddings, pages_and_chunks = load_embeddings(
    embeddings_path=embeddings_path, device=device
)

# Load model TTS capleaf/viXTTS
tts_model_path = os.getenv("PROJECTCB1_TTS_MODEL")
vixtts_model = load_model_tts(
    xtts_checkpoint=f"{tts_model_path}/model.pth",
    xtts_config=f"{tts_model_path}/config.json",
    xtts_vocab=f"{tts_model_path}/vocab.json",
)

# Load reference audio for tts
reference_audio = os.getenv("PROJECTCB1_REFERENCE_AUDIO")  # Mẫu giọng nói

# Load model STT nguyenvulebinh/wav2vec2-base-vietnamese-250h
# stt_model_path = os.getenv("PROJECTCB1_STT_MODEL")
# processor, stt_model = load_model_stt(stt_model_path=stt_model_path)

# Load LLM
llm_path = os.getenv("PROJECTCB1_LLM_MODEL")
model, tokenizer = load_chat_model(llm_path, device=device)
# ------------------------------------------------------------------------------------
# Load reranking
# rr_model_path = "embedding_model/PhoRanker"
# reranking_model = load_reranking_model(rr_model_path)

# Dowload TTS capleaf/viXTTS
# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="capleaf/viXTTS", repo_type="model", local_dir="Model/TTS_model"
# )


print("Models Loaded!")


# ------------------------------------------Text processing-------------------------
def normalize_vietnamese_text(text):
    text = Xu_ly_text_de_doc(text)
    text = (
        TTSnorm(text, unknown=False, lower=False, rule=True)
        .replace("  ", " ")
        .replace(":", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("+", " ")
        .replace("..", ".")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
    )

    return text.lower()


def split_sentences(text, max_length=245):
    text = (
        text.replace("\n", ". ").replace(";", ".").replace("?", ".").replace("!", ".")
    )

    sentences = re.findall(r"[^,.]+[,.]", text)
    grouped_sentences = []
    current_group = ""

    for sentence in sentences:
        # Nếu thêm câu vào mà không vượt quá giới hạn max_length
        if len(current_group) + len(sentence) + 1 < max_length:
            if current_group:
                current_group += " " + sentence  # Ghép câu mới vào câu trước đó
            else:
                current_group = sentence  # Câu đầu tiên của nhóm
        elif len(sentence) > max_length:  # Xử lý
            if current_group:
                grouped_sentences.append(current_group)
                current_group = ""
            k = 0
            tamthoi = []
            for i in sentence.split(" "):
                tamthoi += [i]
                if len(tamthoi) >= 40:
                    grouped_sentences += [" ".join(tamthoi)]
                    tamthoi = []
            if tamthoi:
                grouped_sentences += [" ".join(tamthoi)]
        else:
            grouped_sentences.append(current_group)  # Thêm nhóm vào list
            current_group = sentence  # Khởi tạo nhóm mới với câu hiện tại

    if current_group:
        grouped_sentences.append(current_group)  # Thêm nhóm cuối cùng vào list

    return grouped_sentences


# processor.save_pretrained(stt_model_path)
# model.save_pretrained(stt_model_path)
# Hàm sử dụng cho API


# Retrieval function
def retrieve_relevant_resources(
    query: str, n_resources_to_return: int = 3, threshold: float = 0.1
):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    # cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    # Get top scores with a threshold
    # scores, indices = torch.topk(input=cosine_scores, k=n_resources_to_return)
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    print(scores)
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


# Không sử dung các câu dẫn dắt, hãy trả về trực tiếp câu trả lời.
# Đảm bảo câu trả lời giải thích rõ nhất có thể.
# Prompt formatter
def prompt_formatter_root(query: str, results: list) -> str:
    context = '- "' + '"\n\n- "'.join(results) + '"'
    base_prompt = """Hãy cho bản thân không gian để suy nghĩ bằng cách trích xuất các đoạn văn có liên quan từ ngữ cảnh dưới đây trước khi trả lời câu hỏi của người dùng.
Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi của người dùng:

{context}

Câu hỏi của người dùng: "{query}"
Không sử dung các câu dẫn dắt, hãy trả về trực tiếp câu trả lời. Đảm bảo câu trả lời giải thích đầy đủ và rõ ràng nhất có thể. 
Trả lời:"""
    return base_prompt.format(context=context, query=query)


# ----------------------------------Output function-----------------------------------------------
def run_stt(audio_bytes):
    # Đọc tệp âm thanh từ byte
    audio = AudioSegment.from_file(BytesIO(audio_bytes))

    # Chuyển đổi âm thanh thành mảng numpy
    samples = np.array(audio.get_array_of_samples())

    # Đảm bảo là mono (1 kênh)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
        samples = samples.mean(
            axis=1
        )  # Lấy trung bình giá trị của tất cả các kênh để chuyển sang mono

    # Chuẩn hóa lại tần số mẫu về 16000 Hz
    samples_16k = librosa.resample(
        samples.astype(np.float32), orig_sr=audio.frame_rate, target_sr=16000
    )

    # Tokenize dữ liệu đầu vào
    input_values = processor(
        samples_16k, return_tensors="pt", padding="longest", sampling_rate=16000
    ).input_values

    # Chuyển sang GPU và chuyển đổi sang float
    input_values = input_values.to(device).float()

    # Lấy kết quả dự đoán từ mô hình
    logits = stt_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Giải mã kết quả dự đoán thành văn bản
    transcription = processor.batch_decode(predicted_ids)[0]
    text = Xu_ly_text(transcription)
    return text


def run_tts(text, lang="vi"):
    if vixtts_model is None or not reference_audio:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = vixtts_model.get_conditioning_latents(
        audio_path=reference_audio,
        gpt_cond_len=vixtts_model.config.gpt_cond_len,
        max_ref_length=vixtts_model.config.max_ref_len,
        sound_norm_refs=vixtts_model.config.sound_norm_refs,
    )

    # Chuẩn hóa
    tts_text = normalize_vietnamese_text(text)
    tts_texts = split_sentences(tts_text)
    print(tts_texts)
    wav_chunks = []
    for text in tts_texts:
        if text.strip() == "":
            continue

        wav_chunk = vixtts_model.inference(
            text=text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.01,  # 0.3
            length_penalty=1.0,  # 1.0
            repetition_penalty=50.0,  # 10.0
            top_k=5,  # 30
            top_p=0.95,  # 0.85
        )

        keep_len = -1
        wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len])
        wav_chunks.append(wav_chunk["wav"])

    out_wav = (
        torch.cat(wav_chunks, dim=0).squeeze(0).cpu().numpy()
    )  # Chuyển sang numpy array

    # Chuyển đổi Tensor thành định dạng WAV
    buffer = io.BytesIO()

    # Ghi âm thanh vào buffer, đảm bảo dữ liệu đầu vào là numpy array và định dạng đúng
    try:
        sf.write(buffer, out_wav, 24000, format="WAV")
        buffer.seek(0)
        wav_data = buffer.read()
    except Exception as e:
        print(f"Error writing WAV file: {e}")
        return None, None, None

    return wav_data


#  Phân biệt rõ ràng, dự án và tiểu dự án là hoàn toàn khác nhau


# ---------------------------Web searching----------------------------
def fetch_links(query: str, max_links: int = 10):
    """Fetch links from Google search results."""
    url = f"https://www.google.com/search?q={query}"
    driver.get(url)
    links = []

    try:
        # Wait for search results to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, '//a[@jsname="UWckNb"]'))
        )
        link_elements = driver.find_elements(By.XPATH, '//a[@jsname="UWckNb"]')

        for link_element in link_elements[:max_links]:
            href = link_element.get_attribute("href")
            if href:
                links.append(href)
                print(f"Link found: {href}")
    except Exception as e:
        print(f"Error fetching links: {e}")

    return links


def fetch_page_content(url: str):
    """Fetch page content for a given URL."""
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        content = driver.find_element(By.TAG_NAME, "body").text
        print(f"Content fetched from {url}")
        return content
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None


def web_searching(query: str):
    """Perform web search and retrieve page content."""
    results_web_searching = []
    links = fetch_links(query)
    for link in links:
        if any(bad_domain in link for bad_domain in ["youtube.com"]):
            print(f"Skipping: {link}")
            continue

        content = fetch_page_content(link)
        if content:
            results_web_searching.append(content)
        else:
            continue
        if len(results_web_searching) == 2:
            break
    # Return first valid content
    text_web_searching = "\n\n".join(
        [f'- "{content}"' for content in results_web_searching]
    )
    text_web_searching = text_web_searching[:20000]
    return text_web_searching


# def rewrite_result_web_searching(query: str) -> str:
#     # messages = [
#     #     {
#     #         "role": "system",
#     #         "content": """Bạn là một trợ lý tiếng Việt hữu ích. Hãy tổng hợp và viết lại các đoạn thông tin được cung cấp một cách chính xác.""",
#     #     },
#     # ]
#     messages = [
#         {
#             "role": "system",
#             "content": """Bạn là một trợ lý tiếng Việt hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác.""",
#         },
#     ]
#     web_search_result = web_searching(query=query)

#     #     prompt = f"""Các đoạn thông tin dưới đây có thể có những nội dung không liên quan và cần lược bỏ bớt.
#     # Vui lòng tổng hợp và viết lại các đoạn thông tin sau sao cho ngắn gọn, rõ ràng, dễ hiểu hơn và giữ nguyên ý chính theo chủ đề "{query}"
#     # Các đoạn thông tin như sau:

#     # {web_search_result}

#     # Không sử dung các câu dẫn dắt, hãy trả về trực tiếp nội dung được viết lại ngắn gọn, xúc tích tập trung vào chủ đề.
#     # Trả lời:"""

#     #     prompt = f"""Hãy cho bản thân không gian để suy nghĩ bằng cách trích xuất các đoạn văn có liên quan từ ngữ cảnh dưới đây trước khi trả lời câu hỏi của người dùng.
#     # Câu hỏi của người dùng: {query}
#     # Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi của người dùng:

#     # {web_search_result}

#     #
#     # Trả lời:"""
#     #
#     prompt = f"""Hãy cho bản thân không gian để suy nghĩ bằng cách trích xuất các đoạn văn có liên quan từ ngữ cảnh dưới đây trước khi trả lời câu hỏi của người dùng.
# Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi của người dùng:

# {web_search_result}

# Câu hỏi của người dùng: "{query}"
# Không sử dung các câu dẫn dắt, hãy trả về trực tiếp câu trả lời. Đảm bảo câu trả lời giải thích đầy đủ, rõ ràng nhất có thể.
# Trả lời:"""

#     messages.append({"role": "user", "content": prompt})

#     text = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)
#     streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

#     generated_ids = model.generate(
#         model_inputs.input_ids,
#         do_sample=True,
#         temperature=0.01,
#         top_k=40,
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


# ---------------------------------Ask---------------------------
# Generate response using model
# def ask(query: str) -> str:
#     start = timeit.default_timer()
#     messages = [
#         {
#             "role": "system",
#             "content": """Bạn là một trợ lí tiếng Việt hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác.""",
#         },
#     ]
#     results = retrieve_relevant_resources(
#         query, n_resources_to_return=5, threshold=0.45
#     )

#     if len(results) == 0:
#         response = rewrite_result_web_searching(query=query)
#         return response

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
#         temperature=0.01,
#         top_k=40,
#         top_p=0.95,
#         max_new_tokens=1024,
#         repetition_penalty=1.05,
#         streamer=streamer,
#     )

#     generated_ids = [
#         output_ids[len(input_ids) :]
#         for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     stop = timeit.default_timer()
#     print("Time: ", stop - start)
#     return response


# ------------------------streaming-------------------------------
# def rewrite_result_web_searching(query: str) -> str:
#     messages = [
#         {
#             "role": "system",
#             "content": """Bạn là một trợ lý tiếng Việt hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác.""",
#         },
#     ]
#     web_search_result = web_searching(query=query)

#     prompt = f"""Hãy cho bản thân không gian để suy nghĩ bằng cách trích xuất các đoạn văn có liên quan từ ngữ cảnh dưới đây trước khi trả lời câu hỏi của người dùng.
# Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi của người dùng:

# {web_search_result}

# Câu hỏi của người dùng: "{query}"
# Không sử dung các câu dẫn dắt, hãy trả về trực tiếp câu trả lời. Đảm bảo câu trả lời giải thích đầy đủ, rõ ràng nhất có thể.
# Trả lời:"""

#     messages.append({"role": "user", "content": prompt})

#     text = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = tokenizer(text, return_tensors="pt").to(device)

#     streamer = TextIteratorStreamer(
#         tokenizer, skip_prompt=True, skip_special_tokens=True
#     )

#     # model_inputs = tokenizer([text], return_tensors="pt").to(device)

#     inputs = tokenizer(text, return_tensors="pt").to("cuda")

#     thread = Thread(
#         target=model.generate,
#         kwargs={
#             "input_ids": inputs["input_ids"],
#             "streamer": streamer,
#             "max_new_tokens": 1024,
#         },
#     )
#     thread.start()  # now start the thread

#     # for this example we'll both print out the new text and save it to a file
#     # -----------------------------
#     for new_text in streamer:
#         yield new_text + ""
#     thread.join()


# ------------------
from transformers import TextIteratorStreamer
from threading import Thread


def ask(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": """Bạn là một trợ lí tiếng Việt hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác.""",
        },
    ]
    results = retrieve_relevant_resources(query, n_resources_to_return=5, threshold=0.3)
    if len(results) == 0:
        web_search_result = web_searching(query=query)

        prompt = f"""Hãy cho bản thân không gian để suy nghĩ bằng cách trích xuất các đoạn văn có liên quan từ ngữ cảnh dưới đây trước khi trả lời câu hỏi của người dùng.
Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi của người dùng:

{web_search_result}

Câu hỏi của người dùng: "{query}"
Không sử dung các câu dẫn dắt, hãy trả về trực tiếp câu trả lời. Đảm bảo câu trả lời giải thích đầy đủ, rõ ràng nhất có thể. 
Trả lời:"""
    else:
        prompt = prompt_formatter_root(query, results)
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # model_inputs = tokenizer([text], return_tensors="pt").to(device)

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    thread = Thread(
        target=model.generate,
        kwargs={
            "input_ids": inputs["input_ids"],
            "streamer": streamer,
            "max_new_tokens": 1024,
        },
    )
    thread.start()  # now start the thread

    # for this example we'll both print out the new text and save it to a file
    # -----------------------------
    for new_text in streamer:
        yield new_text + ""
    # We can now process this text however we want,
    # for this example we'll print the text to stdout and
    # at the same time save the output to a text file.
    # In reality however we can do any processing we want in this loop
    # meaning we can fit this to just about any usecase we want!
    #   print(new_text, end="")

    thread.join()  # join our thread
