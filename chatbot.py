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
from Xu_ly_text import Xu_ly_text, Xu_ly_text_de_doc
import re
from dotenv import load_dotenv
import os
# from huggingface_hub import login

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

# # Get the Hugging Face access token from environment variables
# hf_token = os.getenv("PROJECTCB1_HUGGINGFACE_ACCESS_TOKEN")

# # Log in to Hugging Face using the access token
# if hf_token:
#     login(token=hf_token)
# else:
#     print(
#         "Access token not found. Please set the HUGGINGFACE_ACCESS_TOKEN in your .env file."
#     )

eb_model_path = os.getenv("PROJECTCB1_EMBEDDING_MODEL")
embeddings_path = os.getenv("PROJECTCB1_EMBEDDING_DATA_PATH")


# Hàm nội bộ
def load_embedding_model(embedding_model_path):
    embedding_model = SentenceTransformer(
        model_name_or_path=embedding_model_path, device=device
    )
    return embedding_model


def load_reranking_model(pr_model_path):
    pr_model = CrossEncoder(model_name=pr_model_path, device=device)
    return pr_model


def load_embeddings(embeddings_path):
    text_chunks_and_embedding_df = pd.read_csv(embeddings_path)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df[
        "embedding"
    ].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    embeddings = torch.tensor(
        np.array(text_chunks_and_embedding_df["embedding"].tolist()),
        dtype=torch.float32,
    ).to(device)
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


# Khai báo các mô hình

print("Loading models... ")
# Load model embedding

embedding_model = load_embedding_model(eb_model_path)

# Load reranking
# rr_model_path = "embedding_model/PhoRanker"
# reranking_model = load_reranking_model(rr_model_path)

# Dowload TTS capleaf/viXTTS
# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="capleaf/viXTTS", repo_type="model", local_dir="Model/TTS_model"
# )

tts_model_path = os.getenv("PROJECTCB1_TTS_MODEL")
# Load model TTS capleaf/viXTTS
vixtts_model = load_model_tts(
    xtts_checkpoint=f"{tts_model_path}/model.pth",
    xtts_config=f"{tts_model_path}/config.json",
    xtts_vocab=f"{tts_model_path}/vocab.json",
)

embeddings, pages_and_chunks = load_embeddings(embeddings_path)  # Load embeddings
reference_audio = os.getenv("PROJECTCB1_REFERENCE_AUDIO")  # Mẫu giọng nói

# Load model STT nguyenvulebinh/wav2vec2-base-vietnamese-250h

stt_model_path = os.getenv("PROJECTCB1_STT_MODEL")
processor = Wav2Vec2Processor.from_pretrained(stt_model_path)
model = Wav2Vec2ForCTC.from_pretrained(stt_model_path).to(device)
print("Models Loaded!")

# processor.save_pretrained(stt_model_path)
# model.save_pretrained(stt_model_path)
# Hàm sử dụng cho API


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
    logits = model(input_values).logits
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


def retrieve_relevant_resources(query: str, n_resources_to_return: int = 1):
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


# def retrieve_relevant_resources(query: str, n_resources_to_return: int = 1):
#     query_embedding = embedding_model.encode(query, convert_to_tensor=True)
#     dot_scores = util.dot_score(query_embedding, embeddings)[0]
#     scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
#     context_items = [pages_and_chunks[i] for i in indices]
#     score = scores[0].item()
#     answer, similar_question = context_items[0]['Final_Answer'], context_items[0]['Question']

#     return answer, score, similar_question

# def retrieve_relevant_resources(query: str, n_resources_to_return: int = 1):
#     query_embedding = embedding_model.encode(query, convert_to_tensor=True)
#     dot_scores = util.dot_score(query_embedding, embeddings)[0]
#     scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
#     context_items = [pages_and_chunks[i] for i in indices]
#
#     ques_items = [item["Question"] for item in context_items]
#     pr_results = pr_model.rank(query, ques_items, return_documents=True, top_k=1)
#     text_results = [item["text"] for item in pr_results]
#
#     score_results = [item["score"] for item in pr_results]
#     text = text_results[0]
#     score = score_results[0]
#     result = None
#     for item in context_items:
#         if item["Question"] == text:
#             result = item["Answer"]
#             break
#     return result, score
