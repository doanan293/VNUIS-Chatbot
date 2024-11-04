from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
# Load mô hình từ đường dẫn
embedding_path = os.getenv("EMBEDDING_MODEL")
embedding_model = SentenceTransformer(model_name_or_path=embedding_path, device="cuda")

# Tạo một embedding mẫu từ một câu bất kỳ
sentence = "Đây là một câu"
embedding = embedding_model.encode(sentence)

# Tính norm của vector embedding
norm = np.linalg.norm(embedding)
print(f"Norm của vector embedding: {norm}")

# Kiểm tra xem norm có gần bằng 1 hay không
if np.isclose(norm, 1.0, atol=1e-3):
    print("Embedding đã được chuẩn hóa.")
else:
    print("Embedding chưa được chuẩn hóa.")
