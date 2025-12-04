# encoding: utf-8
import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

EVAL_FILE = os.path.join(DATA_DIR, "drug2reaction_eval.jsonl")
OUT_FULL = os.path.join(DATA_DIR, "train_pairs_drug2reaction.jsonl")
OUT_SMALL = os.path.join(DATA_DIR, "train_pairs_drug2reaction_small.jsonl")
random.seed(42)
SMALL_MAX_PAIRS = 5000

def main():
    pairs = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("task_type") != "Drug2Reaction":
                continue
            q = (obj.get("question_zh") or "").strip()
            answers = [a.strip() for a in obj.get("answers", []) if isinstance(a, str) and a.strip()]
            if not q or not answers:
                continue
            for reac in answers:
                doc_text = f"Reaction: {reac}"
                pairs.append({"query": q, "doc": doc_text})

    print(f"[info] 总训练对数: {len(pairs)}")
    with open(OUT_FULL, "w", encoding="utf-8") as f_out:
        for p in pairs:
            f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[info] 全量训练对已写入: {OUT_FULL}")

    if len(pairs) > SMALL_MAX_PAIRS:
        small = random.sample(pairs, SMALL_MAX_PAIRS)
    else:
        small = pairs
    with open(OUT_SMALL, "w", encoding="utf-8") as f_out:
        for p in small:
            f_out.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[info] 小样本训练对已写入: {OUT_SMALL} (size={len(small)})")

if __name__ == "__main__":
    main()
