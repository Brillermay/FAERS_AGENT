# encoding: utf-8
"""
根据 drug_reaction_stats.csv 自动生成评估用：
  T1: Drug -> Reaction 的中文问题 + 标准答案集合

输出：JSONL，每行一个样本：
{
  "task_type": "Drug2Reaction",
  "drugname": "ACETAMINOPHEN",
  "question_zh": "...",
  "answers": ["Nausea", "Headache", ...]
}
"""

import os
import json
from typing import Dict, List, Any

import pandas as pd
import requests  # 使用 Ollama 本地模型

# ==== 配置部分 ====

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

STATS_CSV = os.path.join(DATA_DIR, "drug_reaction_stats.csv")
OUT_JSONL = os.path.join(DATA_DIR, "drug2reaction_eval.jsonl")

# 每个药物选多少个 reaction 作为“标准答案”
TOP_ANSWERS_PER_DRUG = 10

# 选择多少个药物来生成评估问题（可以先小规模，比如 100，再逐渐放大）
MAX_DRUGS = 200

# 每个药物生成几条不同问句
QUESTIONS_PER_DRUG = 3

# ==== Ollama 配置 ====
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EVAL_QUESTION_MODEL = os.getenv("EVAL_QUESTION_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

SYSTEM_PROMPT = (
    "你是一名临床医生，正在使用 FAERS 药物不良反应数据库，"
    "负责把英文药物名转换成自然的中文临床提问。"
)


def _ollama_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    调用本地 Ollama /api/chat，返回生成文本（不流式）
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    # Ollama 返回结构：{"message": {"role": "...", "content": "..."}}
    return (data.get("message") or {}).get("content", "") or ""


def load_drug_answer_sets(csv_path: str) -> Dict[str, List[str]]:
    """
    从 drug_reaction_stats.csv 读入：
    返回 {drugname: [reac1, reac2, ...]}，按 freq 排序截断为 TOP_ANSWERS_PER_DRUG。
    """
    df = pd.read_csv(csv_path)
    # 简单保证排序
    df = df.sort_values(by=["drugname", "freq"], ascending=[True, False])

    groups: Dict[str, List[str]] = {}
    for drug, sub in df.groupby("drugname"):
        # 取前 TOP_ANSWERS_PER_DRUG 个 reac
        answers = sub["reac"].head(TOP_ANSWERS_PER_DRUG).tolist()
        if answers:
            groups[drug] = answers
    return groups


def gen_questions_for_drug(drugname: str, answers: List[str]) -> List[str]:
    """
    使用本地 Ollama 模型，为某个药物生成多条中文问题。
    answers 目前只是为了以后可能做提示增强，这里暂时不用或只用于描述任务。
    """
    # 构造 user 提示词
    user_prompt = (
        f"药物英文名：{drugname}。\n"
        "请用简体中文生成几条临床上可能会问的问题，这些问题的意图是："
        "询问该药物的常见不良反应，不要在问题中直接列出反应名称，只在问题中出现药物名。\n"
        "要求：\n"
        "1. 用自然的临床提问语气；\n"
        "2. 每个问题单独一行输出；\n"
        "3. 不要加序号或解释，只输出问题文本本身；\n"
        "4. 不要提到 FAERS 或数据库这些词。"
    )

    try:
        text = _ollama_chat(EVAL_QUESTION_MODEL, SYSTEM_PROMPT, user_prompt).strip()
    except Exception as e:
        print(f"[warn] 生成问题失败 drug={drugname}: {e}")
        return []

    # 按行拆分，过滤空行
    qs = [line.strip() for line in text.split("\n") if line.strip()]
    # 截断到 QUESTIONS_PER_DRUG
    return qs[:QUESTIONS_PER_DRUG]


def main():
    # 1) 读入 drug -> answers 映射
    drug_answers = load_drug_answer_sets(STATS_CSV)
    print(f"[info] 共有 {len(drug_answers)} 个药物有统计结果")

    # 2) 选择前 MAX_DRUGS 个药物（可以额外按字母排序，避免随机）
    all_drugs = sorted(drug_answers.keys())
    selected_drugs = all_drugs[:MAX_DRUGS]
    print(f"[info] 准备为前 {len(selected_drugs)} 个药物生成评估问题，模型={EVAL_QUESTION_MODEL}")

    # 3) 逐个药物生成问题并写 JSONL
    count_samples = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f_out:
        for idx, drug in enumerate(selected_drugs, start=1):
            answers = drug_answers[drug]
            questions = gen_questions_for_drug(drug, answers)
            if not questions:
                continue

            for q in questions:
                sample = {
                    "task_type": "Drug2Reaction",
                    "drugname": drug,
                    "question_zh": q,
                    "answers": answers,  # reaction 英文标签列表
                }
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count_samples += 1

            print(f"[info] [{idx}/{len(selected_drugs)}] {drug} -> {len(questions)} 问题")

    print(f"[done] 共生成评估样本 {count_samples} 条，已写入 {OUT_JSONL}")


if __name__ == "__main__":
    main()
