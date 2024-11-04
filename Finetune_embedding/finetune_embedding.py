import numpy as np
import json
import torch
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
import pandas as pd
from datasets import Dataset
from huggingface_hub import login
import os
from dotenv import load_dotenv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer
from dotenv import load_dotenv

load_dotenv()
corpus = pd.read_csv("./Data/finetune_embedding/corpus.csv")
queries = pd.read_csv("./Data/finetune_embedding/queries.csv")
relevant_docs_data = pd.read_csv("./Data/finetune_embedding/eval.csv")


corpus = Dataset.from_pandas(corpus)
queries = Dataset.from_pandas(queries)
relevant_docs_data = Dataset.from_pandas(relevant_docs_data)
# Convert the datasets to dictionaries
corpus = dict(
    zip(corpus["idnoidungvandechitiet"], corpus["Final_Answer"])
)  # Our corpus (cid => document)
queries = dict(
    zip(queries["id_queries"], queries["Question"])
)  # Our queries (qid => question)
# Convert integer keys to string keys
queries = {str(key): value for key, value in queries.items()}
# Now your 'queries' dictionary will have string keys

relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for qid, corpus_ids in zip(
    relevant_docs_data["id_queries"], relevant_docs_data["idnoidungvandechitiet"]
):
    qid = str(qid)
    corpus_ids = str(corpus_ids)
    if qid not in relevant_docs:
        relevant_docs[qid] = set()
    relevant_docs[qid].add(corpus_ids)

model = SentenceTransformer(
    model_name_or_path="bkai-foundation-models/vietnamese-bi-encoder",
    trust_remote_code=True,
)
matryoshka_dimensions = [768, 512, 256, 128, 64]  # Important: large to small
matryoshka_evaluators = []
# Iterate over the different dimensions
for dim in matryoshka_dimensions:
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"dim_{dim}",
        truncate_dim=dim,  # Truncate the embeddings to a certain dimension
        score_functions={"cosine": cos_sim},
    )
    matryoshka_evaluators.append(ir_evaluator)

# Create a sequential evaluator
evaluator = SequentialEvaluator(matryoshka_evaluators)

# Evaluate the model and ensure it's all on the correct device
results = evaluator(model)

for k, v in results.items():
    print(k, v)


def prepare_training_dataset(queries, corpus, relevant_docs):
    anchors = []
    positives = []
    for query_id, docs in relevant_docs.items():
        for doc_id in docs:
            anchors.append(queries[query_id])
            positives.append(corpus[doc_id])
    df = {"anchor": anchors, "positive": positives}

    return Dataset.from_dict(df)


pairs = prepare_training_dataset(queries, corpus, relevant_docs)

matryoshka_dimensions = [768, 512, 256, 128, 64]  # Important: large to small
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)

# Get today's date in the desired format (e.g., YYYYMMDD)
today_date = datetime.today().strftime("%Y%m%d")

# Define the model name with today's date
model_name = f"vcs_embedding_bkai_{today_date}"

args = SentenceTransformerTrainingArguments(
    output_dir=f"./Model/{model_name}",  # output directory and hugging face model ID
    num_train_epochs=int(os.getenv("PROJECTCB1_EPOCHS")),  # number of epochs
    per_device_train_batch_size=32,  # train batch size
    gradient_accumulation_steps=4,  # for a global batch size of 512
    per_device_eval_batch_size=32,  # evaluation batch size
    # gradient_checkpointing=True,
    warmup_ratio=0.1,  # warmup ratio
    learning_rate=2e-5,  # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",  # use constant learning rate scheduler
    optim="adamw_torch_fused",  # use fused adamw optimizer
    # tf32=True,                                  # use tf32 precision
    bf16=True,  # use bf16 precision
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="steps",  # evaluate after each epoch
    save_strategy="no",  # save after each epoch
    logging_steps=int(os.getenv("PROJECTCB1_EVALUATION_STEP")),  # log every 10 steps
    save_total_limit=1,  # save only the last 3 models
    load_best_model_at_end=False,  # load the best model when training ends
    metric_for_best_model="eval_dim_768_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,  # training arguments
    train_dataset=pairs,
    loss=train_loss,
    evaluator=evaluator,
)
# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save the best model
trainer.save_model()


fine_tuned_model = SentenceTransformer(
    args.output_dir, device="cuda" if torch.cuda.is_available() else "cpu"
)
# Evaluate the model
results = evaluator(fine_tuned_model)

for k, v in results.items():
    print(k, v)
