import io
import json
import logging
import os
import re
from io import BytesIO
from threading import Thread

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from dotenv import load_dotenv
from pydub import AudioSegment

# from huggingface_hub import login
# Opitimized
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
from TTS.tts.models.xtts import Xtts  # type: ignore

from routes.Xu_ly_text import Xu_ly_text_de_doc
from vi_cleaner.vi_cleaner import ViCleaner  # type: ignore

# Configure logging instead of using print statements
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def chrome():
    # Configure Chrome options
    options = Options()
    options.add_argument("--headless=new")  # Use the new headless mode
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-translate")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--allow-insecure-localhost")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )

    # Disable image loading to speed up page loads
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.stylesheets": 2,
        "profile.default_content_setting_values.cookies": 2,
        "profile.default_content_setting_values.javascript": 1,  # Keep JavaScript enabled if needed
        "profile.default_content_setting_values.plugins": 2,
        "profile.default_content_setting_values.popups": 2,
        "profile.default_content_setting_values.geolocation": 2,
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_setting_values.auto_select_certificate": 2,
        "profile.default_content_setting_values.fullscreen": 2,
        "profile.default_content_setting_values.mouselock": 2,
        "profile.default_content_setting_values.mixed_script": 2,
        "profile.default_content_setting_values.media_stream": 2,
    }
    options.add_experimental_option("prefs", prefs)

    # Set page load strategy to 'eager' to speed up page loads
    options.page_load_strategy = "eager"

    # Specify the path to the ChromeDriver
    service = Service("/usr/local/bin/chromedriver")

    # Initialize the WebDriver with the specified service and options
    driver = webdriver.Chrome(service=service, options=options)
    return driver


# service=Service("/usr/bin/chromedriver"),
# Initialize the WebDriver


load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")


# --------------------------Define function to load Model and Data---------------------------
# Hàm nội bộ
def load_embedding_model(embedding_model_path, device):
    try:
        embedding_model = SentenceTransformer(
            model_name_or_path=embedding_model_path,
            device=device,
            model_kwargs={"torch_dtype": "bfloat16"},
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
        
    XTTS_MODEL.eval()
    return XTTS_MODEL


def load_model_stt(stt_model_path: str):
    stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        stt_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    stt_model.eval()
    # stt_model.to(device)

    processor = AutoProcessor.from_pretrained(stt_model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=stt_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    return pipe, stt_model


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
embeddings_path = os.getenv("PROJECTCB1_DATA_FINAL")
embeddings, pages_and_chunks = load_embeddings(
    embeddings_path=embeddings_path, device=device
)

# # Load model TTS capleaf/viXTTS
# tts_model_path = os.getenv("PROJECTCB1_TTS_MODEL")
# tts_model = load_model_tts(
#     xtts_checkpoint=f"{tts_model_path}/model.pth",
#     xtts_config=f"{tts_model_path}/config.json",
#     xtts_vocab=f"{tts_model_path}/vocab.json",
# )

# logging.info("Done TTS")
# # Load reference audio for tts
# reference_audio = os.getenv("PROJECTCB1_REFERENCE_AUDIO")  # Mẫu giọng nói

# Load model STT openai/whisper-large-v3-turbo
stt_model_path = os.getenv("PROJECTCB1_STT_MODEL")
pipe, stt_model = load_model_stt(stt_model_path=stt_model_path)

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
    cleaner = ViCleaner(text)
    text = cleaner.clean()
    text = Xu_ly_text_de_doc(text)
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


# # Retrieval function


rerank_model_path = os.getenv("PROJECTCB1_RERANK_MODEL")
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    rerank_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
rerank_model.eval()


# Retrieval with rerank
def retrieve_relevant_resources(
    query: str,
    number_result_embedding: int = 20,
    number_result_reranking: int = 5,
    threshold: int = -4,
):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    # cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    # Get top scores with a threshold
    # scores, indices = torch.topk(input=cosine_scores, k=n_resources_to_return)
    scores, indices = torch.topk(input=dot_scores, k=number_result_embedding)
    print(scores)

    context_items = [pages_and_chunks[i] for i in indices]
    results = [item["Relevant docs"] for item in context_items]

    pairs = [[query, result] for result in results]

    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=1024
        )
        inputs = {
            key: value.to("cuda") for key, value in inputs.items()
        }  # Move all inputs to the same device as the model

        # Compute scores
        rerank_scores = rerank_model(**inputs, return_dict=True).logits.view(
            -1,
        )

        top_scores, top_indices = torch.topk(rerank_scores, k=number_result_reranking)
        # Help me add script to only take the score > -3
        filtered_indices = top_indices[top_scores > threshold]
        rerank_result = [results[i] for i in filtered_indices]

    # return results, scores, top_scores, rerank_result
    return rerank_result


# Không sử dung các câu dẫn dắt, hãy trả về trực tiếp câu trả lời.
# Đảm bảo câu trả lời giải thích rõ nhất có thể.
# Prompt formatter
def prompt_formatter_root(query: str, results: list) -> str:
    context = '- "' + '"\n\n- "'.join(results) + '"'
    base_prompt = """Hãy cho bản thân không gian để suy nghĩ bằng cách trích xuất các đoạn văn có liên quan từ ngữ cảnh dưới đây trước khi trả lời câu hỏi của người dùng.
Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi của người dùng:

{context}

Câu hỏi của người dùng: "{query}"
Không sử dung các câu dẫn dắt, hãy trả về trực tiếp câu trả lời. Đảm bảo câu trả lời giải thích rõ nhất có thể. 
Trả lời:"""
    return base_prompt.format(context=context, query=query)


# ----------------------------------Output function-----------------------------------------------
def convert_to_wav(audio_bytes):
    try:
        # Sử dụng pydub để đọc tệp âm thanh từ BytesIO
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        # Tạo một tệp WAV trong bộ nhớ
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)  # Đặt lại con trỏ tệp về vị trí ban đầu
        return wav_io
    except Exception as e:
        raise ValueError(f"Error converting audio to WAV: {e}")


# Hàm xử lý âm thanh và nhận dạng văn bản
def run_stt(audio_bytes):
    # Chuyển đổi âm thanh thành WAV
    wav_io = convert_to_wav(audio_bytes)

    # Sử dụng librosa để đọc tệp WAV và lấy dữ liệu âm thanh
    audio, sr = librosa.load(wav_io, sr=16000)

    result = pipe(
        audio, return_timestamps=True, generate_kwargs={"language": "vietnamese"}
    )
    # text = Xu_ly_text(result["text"])
    return result["text"]


def run_tts(text, lang="vi"):
    if tts_model is None or not reference_audio:
        return "You need to run the previous step to load the model !!", None, None

    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
        audio_path=reference_audio,
        # gpt_cond_len=tts_model.config.gpt_cond_len,
        # max_ref_length=tts_model.config.max_ref_len,
        # sound_norm_refs=tts_model.config.sound_norm_refs,
        gpt_cond_len=0,
        max_ref_length=1,
        sound_norm_refs=True,
    )

    # Chuẩn hóa
    tts_text = normalize_vietnamese_text(text)
    tts_texts = split_sentences(tts_text)

    print(tts_texts)
    wav_chunks = []
    for text in tts_texts:
        if text.strip() == "":
            continue

        wav_chunk = tts_model.inference(
            text=text,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            do_sample=False,
            length_penalty=1.0,  # 1.0
            repetition_penalty=10.0,  # 10.0
        )

        keep_len = -1
        wav_chunk["wav"] = torch.tensor(wav_chunk["wav"][:keep_len]).to(device)
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
        return None

    return wav_data


# ---------------------------Web searching----------------------------


def web_searching(query: str, max_links: int = 10, max_contents: int = 2):
    """
    Perform a web search and retrieve page content using a single Chrome driver instance.

    Args:
        query (str): The search query.
        max_links (int): Maximum number of links to fetch from search results.
        max_contents (int): Maximum number of page contents to retrieve.

    Returns:
        str: Combined content from the fetched web pages.
    """
    results_web_searching = []
    driver = None

    try:
        # Initialize the Chrome driver once
        driver = chrome()
        # Remove implicit wait to avoid conflicts with explicit waits
        wait = WebDriverWait(driver, 10)  # Explicit wait

        # Step 1: Fetch links from Google search results
        search_url = f"https://www.google.com/search?q={query}"
        driver.get(search_url)
        logging.info(f"Navigated to search URL: {search_url}")

        # Wait for search results to load using CSS selector
        wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a[jsname="UWckNb"]'))
        )
        link_elements = driver.find_elements(By.CSS_SELECTOR, 'a[jsname="UWckNb"]')
        links = []

        for link_element in link_elements[:max_links]:
            href = link_element.get_attribute("href")
            if href:
                links.append(href)
                logging.info(f"Link found: {href}")

        # Step 2: Iterate through the fetched links and get page content
        bad_domains = ["youtube.com"]
        for link in links:
            if any(bad_domain in link for bad_domain in bad_domains):
                logging.info(f"Skipping unwanted domain: {link}")
                continue

            try:
                driver.get(link)
                logging.info(f"Navigated to URL: {link}")

                # Wait for the body of the page to load
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                content = driver.find_element(By.TAG_NAME, "body").text
                if content:
                    results_web_searching.append(content)
                    logging.info(f"Content fetched from {link}")
            except Exception as e:
                logging.warning(f"Error fetching content from {link}: {e}")
                continue

            if len(results_web_searching) >= max_contents:
                logging.info(f"Reached maximum content limit: {max_contents}")
                break

        # Step 3: Combine the fetched contents
        text_web_searching = "\n\n".join(
            [f'- "{content}"' for content in results_web_searching]
        )
        text_web_searching = text_web_searching[:20000]  # Limit to 20,000 characters
        logging.info("Web searching completed successfully.")

    except Exception as general_e:
        logging.error(f"An unexpected error occurred: {general_e}")

    finally:
        if driver:
            driver.quit()  # Ensure the driver is properly closed
            logging.info("Chrome driver has been closed.")

    return text_web_searching


# ---------------------------------Ask---------------------------


def ask(query: str) -> str:
    # time.sleep(60)
    messages = [
        {
            "role": "system",
            "content": """Bạn là một trợ lí tiếng Việt hữu ích. Hãy trả lời câu hỏi của người dùng một cách chính xác.""",
        },
    ]
    results = retrieve_relevant_resources(
        query, number_result_embedding=20, number_result_reranking=3, threshold=-4
    )
    if len(results) == 0:
        web_search_result = web_searching(query=query, max_contents=1)

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
            "do_sample": True,
            "max_new_tokens": 1024,
            "temperature": 0.01,
            # "top_k": 40,
            # "top_p": 0.95,
            # "repetition_penalty": 1.05,
        },
    )
    thread.start()  # now start the thread

    # for this example we'll both print out the new text and save it to a file
    # -----------------------------
    try:
        # Yield each piece of generated text as it's available
        for new_text in streamer:
            yield new_text + ""
        # for new_text in streamer:
        #     new_text = new_text.replace("\n", "\\n")
        #     yield "data: " + new_text + "\n\n"
    finally:
        # Ensure the thread is properly joined even if the generator is not fully consumed
        logging.info("Done ask query")
        thread.join()
