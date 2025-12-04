# encoding: utf-8
"""
使用 sentence-transformers 对 Drug->Reaction 任务做双塔微调（小规模试验）。
"""

import os
import json
from typing import List

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

TRAIN_FILE = os.path.join(DATA_DIR, "train_pairs_drug2reaction_small.jsonl")  # 先用小样本
DEV_FILE = os.path.join(DATA_DIR, "drug2reaction_eval.jsonl")                 # 直接用我们已有的评估集做 dev

# 只用本地模型，不再联网
MODEL_NAME = os.getenv("BIENCODER_BASE_MODEL", r"D:\models\bge-m3")

OUTPUT_DIR = "models/drug2reaction_biencoder_trial"

BATCH_SIZE = 32
EPOCHS = 1  # 小试一轮即可，后面再放大
LR = 2e-5

def load_train_examples(path: str) -> List[InputExample]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("query", "").strip()
            d = obj.get("doc", "").strip()
            if not q or not d:
                continue
            examples.append(InputExample(texts=[q, d]))
    print(f"[info] 训练样本数: {len(examples)}")
    return examples


def build_ir_evaluator(path: str) -> InformationRetrievalEvaluator:
    """
    用 drug2reaction_eval.jsonl 构造一个 IR evaluator：
    - queries: 问题 id -> 问题文本
    - corpus:  reaction id -> doc 文本
    - relevant_docs: 问题 id -> {reaction id: 1}
    这里为了简单起见，把 reaction 文本直接当作 doc id（字符串）。
    """
    queries = {}
    corpus = {}
    relevant_docs = {}

    with open(path, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("task_type") != "Drug2Reaction":
                continue

            qid = f"q{idx}"
            q_text = obj.get("question_zh", "").strip()
            answers = [a.strip() for a in obj.get("answers", []) if isinstance(a, str) and a.strip()]
            if not q_text or not answers:
                continue

            queries[qid] = q_text
            relevant_docs[qid] = {}

            for reac in answers:
                doc_id = reac  # 简单用 reaction 文本本身作为 id
                doc_text = f"Reaction: {reac}"
                corpus[doc_id] = doc_text
                relevant_docs[qid][doc_id] = 1

            idx += 1

    print(f"[info] evaluator: {len(queries)} queries, {len(corpus)} docs")
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="drug2reaction_dev",
        show_progress_bar=True,
        write_csv=True
        # 老版本不支持 k_values 参数，使用默认配置（通常包含 @10）
    )


def main():
    # 1) 加载基座模型
    print(f"[info] loading base model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 2) 加载训练数据
    train_examples = load_train_examples(TRAIN_FILE)
    train_dataloader = DataLoader(train_examples, batch_size=BATCH_SIZE, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 3) 构建 evaluator（用 eval jsonl 直接做 dev）
    evaluator = build_ir_evaluator(DEV_FILE)

    # 4) 训练配置（旧版本用 model.fit）
    warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)  # 简单设个 10% warmup

    print(f"[info] start training: epochs={EPOCHS}, lr={LR}, warmup_steps={warmup_steps}")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        evaluator=evaluator,
        evaluation_steps=100,               # 可按需要调整
        warmup_steps=warmup_steps,
        output_path=OUTPUT_DIR,
        optimizer_params={"lr": LR},
        show_progress_bar=True,
    )

    print(f"[done] 训练完成，模型已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
