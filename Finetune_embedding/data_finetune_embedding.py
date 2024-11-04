import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info("GenData start")

load_dotenv()

# qna = pd.read_csv(os.getenv("PROJECTCB1_RESULT_PATH"))

noidungcautraloi = pd.read_csv(os.getenv("PROJECTCB1_NOIDUNGCAUTRALOI_PATH"))
vanbandungchung = pd.read_csv(os.getenv("PROJECTCB1_VANBANDUNGCHUNG_PATH"))

merged_df1 = pd.merge(
    noidungcautraloi,
    vanbandungchung,
    left_on="idsohieuvanban",
    right_on="oid",
    how="left",
)
merged_df1 = merged_df1.drop(["oid"], axis=1)

danhmuc = pd.read_csv(os.getenv("PROJECTCB1_DANHMUC_PATH"))
merged_df2 = pd.merge(
    merged_df1, danhmuc, left_on="idloaivanban", right_on="Oid", how="left"
)
merged_df2 = merged_df2.drop(["Oid"], axis=1)

# merged_df3 = merged_df3.drop(columns=['oid'])
merged_df3 = pd.merge(
    merged_df2, danhmuc, left_on="idcoquanbanhanhvanban", right_on="Oid", how="left"
)
merged_df3 = merged_df3.drop(["idcoquanbanhanhvanban", "idloaivanban", "Oid"], axis=1)
merged_df3 = merged_df3.rename(
    columns={"MSName_x": "loaivanban", "MSName_y": "coquanbanhanhvanban"}
)
merged_df3 = merged_df3.drop(["CategoryID_x", "CategoryID_y"], axis=1)

merged_df4 = pd.merge(
    merged_df3, vanbandungchung, left_on="idsohieuvanbanmoi", right_on="oid", how="left"
)
merged_df4 = merged_df4.drop(["oid"], axis=1)

merged_df5 = pd.merge(
    merged_df4, danhmuc, left_on="idloaivanban", right_on="Oid", how="left"
)
merged_df5 = merged_df5.drop(["Oid"], axis=1)
merged_df5 = merged_df5.rename(columns={"MSName": "loaivanbanmoi"})

merged_df6 = pd.merge(
    merged_df5, danhmuc, left_on="idcoquanbanhanhvanban", right_on="Oid", how="left"
)
merged_df6 = merged_df6.rename(columns={"MSName": "coquanbanhanhvanbanmoi"})
merged_df6 = merged_df6.drop(["Oid", "CategoryID_y", "CategoryID_x"], axis=1)

noidungvandechitiet = pd.read_csv(os.getenv("PROJECTCB1_NOIDUNGVANDECHITIET_PATH"))

merged_df7 = pd.merge(
    merged_df6,
    noidungvandechitiet,
    left_on="idnoidungvandechitiet",
    right_on="oid",
    how="left",
)
merged_df7 = merged_df7.drop(["oid"], axis=1)

noidungvande = pd.read_csv(os.getenv("PROJECTCB1_NOIDUNGVANDE_PATH"))
# Now perform the merge
merged_df8 = pd.merge(
    merged_df7, noidungvande, left_on="idnoidungvande", right_on="oid", how="left"
)
merged_df8 = merged_df8.drop(columns=["oid"])

merged_df9 = pd.merge(
    merged_df8, danhmuc, left_on="idlinhvucquanly", right_on="Oid", how="left"
)

merged_df10 = merged_df9.loc[
    :,
    [
        "dieu",
        "khoan",
        "tieumuc",
        "diem",
        "tiet",
        "noidung",
        "dieumoi",
        "khoanmoi",
        "tieumucmoi",
        "diemmoi",
        "tietmoi",
        "noidungmoi",
        "sokyhieu_x",
        "tenvanban_x",
        "ngaybanhanh_x",
        "ngayhieuluc_x",
        "ngayhethieuluc_x",
        "loaivanban",
        "coquanbanhanhvanban",
        "sokyhieu_y",
        "tenvanban_y",
        "ngaybanhanh_y",
        "ngayhieuluc_y",
        "ngayhethieuluc_y",
        "loaivanbanmoi",
        "coquanbanhanhvanbanmoi",
        "noidungcauhoi",
        "masocauhoi",
        "masonoidung",
        "tennoidungvande",
        "MSName",
        "idnoidungvandechitiet",
    ],
]
merged_df10 = merged_df10.rename(columns={"MSName": "linhvucquanly"})
merged_df10 = merged_df10.rename(
    columns={
        "sokyhieu_x": "sokyhieu",
        "tenvanban_x": "tenvanban",
        "ngaybanhanh_x": "ngaybanhanh",
        "ngayhieuluc_x": "ngayhieuluc",
        "ngayhethieuluc_x": "ngayhethieuluc",
        "sokyhieu_y": "sokyhieumoi",
        "tenvanban_y": "tenvanbanmoi",
        "ngaybanhanh_y": "ngaybanhanhvanbanmoi",
        "ngayhieuluc_y": "ngayhieulucvanbanmoi",
        "ngayhethieuluc_y": "ngayhethieulucvanbanmoi",
    }
)

import pandas as pd
import re

# Biểu thức chính quy cho số La Mã hợp lệ
roman_numeral_pattern = "^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"


def is_roman_numeral(s):
    # Kiểm tra xem chuỗi có khớp với mẫu số La Mã không
    return bool(re.match(roman_numeral_pattern, s.upper()))


def create_answer(row):
    dieu_phan = (
        str(row.get("dieu", ""))
        .replace("Điều", "")
        .replace("Phần", "")
        .replace("Mục", "")
        .strip()
        if not pd.isna(row.get("dieu"))
        else ""
    )

    khoan_muc = (
        str(row.get("khoan", ""))
        .lower()
        .replace("khoản", "")
        .replace("mục", "")
        .replace(".", "")
        .strip()
        if not pd.isna(row.get("khoan"))
        else ""
    )

    noidung = str(row.get("noidung", "")) if not pd.isna(row.get("noidung")) else ""

    tieumuc = (
        str(row.get("tieumuc", "")).lower().replace("tiểu mục", "").strip()
        if not pd.isna(row.get("tieumuc"))
        else ""
    )
    tieumuc = f", Tiểu mục {tieumuc}" if tieumuc else ""

    diem = (
        str(row.get("diem", "")).lower().replace("điểm", "").replace(")", "").strip()
        if not pd.isna(row.get("diem"))
        else ""
    )
    diem = f", điểm {diem}" if diem else ""

    tiet = (
        str(row.get("tiet", "")).lower().replace("tiết", "").strip()
        if not pd.isna(row.get("tiet"))
        else ""
    )
    tiet = f", tiết {tiet}" if tiet else ""

    loaivanban = (
        f" {row.get('loaivanban', '')}" if not pd.isna(row.get("loaivanban")) else ""
    )
    sokyhieu = f" {row.get('sokyhieu', '')}" if not pd.isna(row.get("sokyhieu")) else ""

    # Kiểm tra nếu dieu_phan là số La Mã
    if is_roman_numeral(dieu_phan):
        if khoan_muc:
            cancu = (
                f" (Theo Phần {dieu_phan}, Mục {khoan_muc}{tieumuc}{tiet}{loaivanban}{sokyhieu})"
                if dieu_phan
                else ""
            )
        else:
            cancu = (
                f" (Theo Phần {dieu_phan}{khoan_muc}{tieumuc}{tiet}{loaivanban}{sokyhieu})"
                if dieu_phan
                else ""
            )
    else:
        if khoan_muc:
            cancu = (
                f" (Theo Điều {dieu_phan}, khoản {khoan_muc}{diem}{tiet}{loaivanban}{sokyhieu})"
                if dieu_phan
                else ""
            )
        else:
            cancu = (
                f" (Theo Điều {dieu_phan}{khoan_muc}{diem}{tiet}{loaivanban}{sokyhieu})"
                if dieu_phan
                else ""
            )

    # Tạo final_answer
    answer = f"{noidung}{cancu}".strip()  # Sử dụng strip() để loại bỏ khoảng trắng thừa
    return answer


def create_new_answer(row):
    dieu_phan_moi = (
        str(row.get("dieumoi", ""))
        .replace("Điều", "")
        .replace("Phần", "")
        .replace("Mục", "")
        .strip()
        if not pd.isna(row.get("dieumoi"))
        else ""
    )

    khoan_muc_moi = (
        str(row.get("khoanmoi", ""))
        .lower()
        .replace("khoản", "")
        .replace("mục", "")
        .replace(".", "")
        .strip()
        if not pd.isna(row.get("khoanmoi"))
        else ""
    )

    noidungmoi = (
        str(row.get("noidungmoi", "")) if not pd.isna(row.get("noidungmoi")) else ""
    )

    tieumucmoi = (
        str(row.get("tieumucmoi", "")).lower().replace("tiểu mục", "").strip()
        if not pd.isna(row.get("tieumucmoi"))
        else ""
    )
    tieumucmoi = f", Tiểu mục {tieumucmoi}" if tieumucmoi else ""

    diemmoi = (
        str(row.get("diemmoi", "")).lower().replace("điểm", "").replace(")", "").strip()
        if not pd.isna(row.get("diemmoi"))
        else ""
    )
    diemmoi = f", điểm {diemmoi}" if diemmoi else ""

    tietmoi = (
        str(row.get("tietmoi", "")).lower().replace("tiết", "").strip()
        if not pd.isna(row.get("tietmoi"))
        else ""
    )
    tietmoi = f", tiết {tietmoi}" if tietmoi else ""

    loaivanbanmoi = (
        f" {row.get('loaivanbanmoi', '')}"
        if not pd.isna(row.get("loaivanbanmoi"))
        else ""
    )
    sokyhieumoi = (
        f" {row.get('sokyhieumoi', '')}" if not pd.isna(row.get("sokyhieumoi")) else ""
    )

    # Kiểm tra nếu dieu_phan là số La Mã
    if is_roman_numeral(dieu_phan_moi):
        if khoan_muc_moi:
            cancu = (
                f" (Theo Phần {dieu_phan_moi}, Mục {khoan_muc_moi}{tieumucmoi}{tietmoi}{loaivanbanmoi}{sokyhieumoi})"
                if dieu_phan_moi
                else ""
            )
        else:
            cancu = (
                f" (Theo Phần {dieu_phan_moi}{khoan_muc_moi}{tieumucmoi}{tietmoi}{loaivanbanmoi}{sokyhieumoi})"
                if dieu_phan_moi
                else ""
            )
    else:
        if khoan_muc_moi:
            cancu = (
                f" (Theo Điều {dieu_phan_moi}, khoản {khoan_muc_moi}{diemmoi}{tietmoi}{loaivanbanmoi}{sokyhieumoi})"
                if dieu_phan_moi
                else ""
            )
        else:
            cancu = (
                f" (Theo Điều {dieu_phan_moi}{khoan_muc_moi}{diemmoi}{tietmoi}{loaivanbanmoi}{sokyhieumoi})"
                if dieu_phan_moi
                else ""
            )

    # Tạo final_answer
    answer = (
        f"{noidungmoi}{cancu}".strip()
    )  # Sử dụng strip() để loại bỏ khoảng trắng thừa
    return answer


# Áp dụng hàm trên từng dòng của DataFrame
merged_df10["Answer"] = merged_df10.apply(create_answer, axis=1)
merged_df10["New_Answer"] = merged_df10.apply(create_new_answer, axis=1)

merged_df11 = merged_df10[
    [
        "Answer",
        "noidungcauhoi",
        "idnoidungvandechitiet",
        "New_Answer",
        "masocauhoi",
        "masonoidung",
        "linhvucquanly",
    ]
]

# Tạo cột 'Final_Answer' dựa trên điều kiện kiểm tra chuỗi 'nan'
# merged_df11["Final_Answer"] = np.where(
#     merged_df11["New_Answer"].isna() | (merged_df11["New_Answer"] == ""),
#     merged_df11["Answer"],
#     merged_df11["New_Answer"]
# )
merged_df11["Final_Answer"] = merged_df11["Answer"]


result = merged_df11.groupby("idnoidungvandechitiet", as_index=False).agg(
    {
        "Final_Answer": lambda x: "\n".join(x),  # Filter out empty strings
        "noidungcauhoi": "first",
        "masocauhoi": "first",
        "masonoidung": "first",
        "linhvucquanly": "first",
        # Keep the first value in Column3
    }
)

result = result.rename(columns={"noidungcauhoi": "Question"})
result = result.dropna()

result = result[["idnoidungvandechitiet", "Question", "linhvucquanly", "Final_Answer"]]

cauhoituongduong = pd.read_csv(os.getenv("PROJECTCB1_CHTD_PATH"))

cauhoituongduong = cauhoituongduong[
    ["IdNoiDungVanDeChiTiet", "NoiDungCauHoiTuongDuong"]
]

cauhoituongduong = cauhoituongduong.rename(
    columns={
        "IdNoiDungVanDeChiTiet": "idnoidungvandechitiet",
        "NoiDungCauHoiTuongDuong": "Question",
    }
)

cauhoituongduong["linhvucquanly"] = cauhoituongduong["idnoidungvandechitiet"].map(
    result.set_index("idnoidungvandechitiet")["linhvucquanly"]
)

result = pd.concat([result, cauhoituongduong], axis=0)


result = result.drop_duplicates(subset=["Question"], keep="first")
# Replace empty strings with NaN and then drop rows with NaN values
result = result.replace("", pd.NA).dropna(axis=0)
result = result.reset_index(drop=True)

corpus = result[["Final_Answer", "idnoidungvandechitiet", "Question"]]
corpus["Final_Answer"] = corpus["Question"] + " " + corpus["Final_Answer"]
corpus = corpus[["Final_Answer", "idnoidungvandechitiet"]]
corpus.drop_duplicates(subset=["idnoidungvandechitiet"], keep="first", inplace=True)
corpus = corpus.reset_index(drop=True)
corpus.to_csv("./Data/Generated_Data/corpus.csv", index=False)


result["Question"] = result["Question"].astype(str)
result["Question_same1"] = result["Question"]

result["linhvucquanly_same2"] = result["linhvucquanly"].astype(str)
result["Question_same2"] = result["Question"] + " " + result["linhvucquanly_same2"]

result["linhvucquanly_same3"] = result["linhvucquanly"].str.replace(
    "Chương trình mục tiêu quốc gia", "ctmtqg"
)
result["Question_same3"] = result["Question"] + " " + result["linhvucquanly_same3"]


result["linhvucquanly_same4"] = (
    result["linhvucquanly"]
    .str.replace(
        "Chương trình mục tiêu quốc gia dân tộc miền núi giai đoạn 2021 - 2030",
        "ctmtqg",
    )
    .replace(
        "Chương trình mục tiêu quốc gia dân tộc miền núi giai đoạn I: 2021 - 2025",
        "ctmtqg",
    )
)
result["Question_same4"] = result["Question"] + " " + result["linhvucquanly_same4"]

result["linhvucquanly_same5"] = (
    result["linhvucquanly"]
    .str.replace(
        "Chương trình mục tiêu quốc gia dân tộc miền núi giai đoạn 2021 - 2030",
        "Chương trình mục tiêu quốc gia",
    )
    .replace(
        "Chương trình mục tiêu quốc gia dân tộc miền núi giai đoạn I: 2021 - 2025",
        "Chương trình mục tiêu quốc gia",
    )
)
result["Question_same5"] = result["Question"] + " " + result["linhvucquanly_same5"]

result = pd.melt(
    result,
    id_vars=["idnoidungvandechitiet"],
    value_vars=[
        "Question_same1",
        "Question_same2",
        "Question_same3",
        "Question_same4",
        "Question_same5",
    ],
    var_name="Original_Question_Column",
    value_name="Question_Text",
)

# Dropping the helper column
df_final = result.drop(columns=["Original_Question_Column"])

df_final = df_final.rename(columns={"Question_Text": "Question"})


import string
import unicodedata


# Function to remove punctuation from strings, handling non-string types
def remove_punctuation(text):
    if not isinstance(text, str):
        text = str(text)  # Convert non-string entries to strings
    return text.translate(str.maketrans("", "", string.punctuation))


# Alternative approach to remove punctuation and accents
def remove_punctuation_and_accents_alternative(text):
    if not isinstance(text, str):
        text = str(text)  # Convert non-string entries to strings
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize and remove accents
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text


# Applying the functions to the 'Question' column
df_final["Question_same"] = df_final["Question"].apply(remove_punctuation)
# df_final["Question_same2"] = df_final["Question"].apply(
#     remove_punctuation_and_accents_alternative
# )

df_final = pd.melt(
    df_final,
    id_vars=["idnoidungvandechitiet"],
    value_vars=["Question", "Question_same"],
    var_name="Original_Question_Column",
    value_name="Question_Text",
)

df_final = df_final.drop(columns=["Original_Question_Column"])
df_final = df_final.rename(columns={"Question_Text": "Question"})


df_final.drop_duplicates(subset=["Question"], keep="first", inplace=True)
df_final.reset_index(drop=True, inplace=True)
df_final["id_queries"] = df_final.index


queries = df_final[["id_queries", "Question"]]
queries.to_csv("./Data/Generated_Data/queries.csv", index=False)

eval = df_final[["id_queries", "idnoidungvandechitiet"]]
eval.to_csv("./Data/Generated_Data/eval.csv", index=False)

import pandas as pd
import numpy as np
import json

df = pd.read_csv("./Data/Generated_Data/synthetic.csv")

import pandas as pd

# Adding the 'oid' column
df["idnoidungvandechitiet"] = pd.factorize(df["Final_Answer"])[0]


df["idnoidungvandechitiet"] = "synanswer" + df["idnoidungvandechitiet"].astype(str)
# df.drop(columns=["oid"],inplace=True)

df["id_queries"] = df.index
df["id_queries"] = "synquestion" + df["id_queries"].astype(str)
corpus_syn = df[["Final_Answer", "idnoidungvandechitiet"]]
corpus_syn.drop_duplicates(
    subset=["Final_Answer"], keep="first", inplace=True, ignore_index=True
)

queries_syn = df[["Question", "id_queries"]]
eval_syn = df[["idnoidungvandechitiet", "id_queries"]]

corpus = pd.concat([corpus, corpus_syn], axis=0)
eval = pd.concat([eval, eval_syn], axis=0)
queries = pd.concat([queries, queries_syn], axis=0)


queries.to_csv("./Data/Generated_Data/queries.csv", index=False)
eval.to_csv("./Data/Generated_Data/eval.csv", index=False)
corpus.to_csv("./Data/Generated_Data/corpus.csv", index=False)
