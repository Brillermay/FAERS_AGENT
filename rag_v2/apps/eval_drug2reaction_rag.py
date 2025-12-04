# encoding: utf-8
"""
在 drug2reaction_eval.jsonl 上评估当前 RAG (Drug -> Reaction 检索) 的基线性能。
"""

import os
import json
from typing import List, Dict, Any, Tuple
import contextlib
from io import StringIO

from rag_v2.core.input_layer import parse_input
from rag_v2.core.initial_search_layer import initial_search
from rag_v2.core.relation_aggregate_layer import aggregate_relations
from rag_v2.core.ranking_layer import rank_expansions

# 使用相对于项目根目录的路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EVAL_FILE = os.path.join(ROOT_DIR, "data", "drug2reaction_eval.jsonl")
TOPK_PRED = 10   # 预测时取前多少个 Reaction
RESULT_JSON = os.path.join(ROOT_DIR, "results", "eval_drug2reaction_rag_results.json")


def _run_pipeline_silent(question: str) -> Dict[str, Any]:
    """运行 1~4 层流水线，但静音内部 print 输出。"""
    with contextlib.redirect_stdout(StringIO()):
        parsed = parse_input(question)
        searched = initial_search(parsed)
        aggregated = aggregate_relations(searched)
        ranked = rank_expansions(aggregated)
    return ranked


def _evaluate_one(sample: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    对单条样本计算 Hit@k, Recall@k, MRR@k
    返回: (hit, recall, mrr)
    """
    q = sample["question_zh"]
    gold = [a.strip() for a in sample["answers"] if a and isinstance(a, str)]
    if not gold:
        return 0.0, 0.0, 0.0
    gold_set = set(x.lower() for x in gold)

    # 1~4 层流水线（静音）
    ranked = _run_pipeline_silent(q)

    block = ranked["ranked"].get("Reaction") or {}
    rows = block.get("rows", [])[:TOPK_PRED]
    preds = [(r.get("text") or "").strip() for r in rows if r.get("text")]
    pred_norm = [p.lower() for p in preds]

    # Hit@k: 是否命中至少一个
    hit = 1.0 if any(p in gold_set for p in pred_norm) else 0.0

    # Recall@k: 命中数 / gold 总数
    inter = gold_set.intersection(pred_norm)
    recall = len(inter) / len(gold_set)

    # MRR@k: 第一个命中位置的倒数
    rank_pos = None
    for idx, p in enumerate(pred_norm):
        if p in gold_set:
            rank_pos = idx + 1
            break
    mrr = 1.0 / rank_pos if rank_pos is not None else 0.0

    return hit, recall, mrr


def main():
    total = 0
    sum_hit = 0.0
    sum_recall = 0.0
    sum_mrr = 0.0

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("task_type") != "Drug2Reaction":
                continue

            h, r, m = _evaluate_one(sample)
            sum_hit += h
            sum_recall += r
            sum_mrr += m
            total += 1

            # 进度输出可以少一点，比如每 100 条一次
            if total % 100 == 0:
                print(f"[progress] {total} samples processed")

    if total == 0:
        print("[warn] 无有效样本")
        return

    avg_hit = sum_hit / total
    avg_recall = sum_recall / total
    avg_mrr = sum_mrr / total

    results = {
        "samples": total,
        f"Hit@{TOPK_PRED}": avg_hit,
        f"Recall@{TOPK_PRED}": avg_recall,
        f"MRR@{TOPK_PRED}": avg_mrr,
    }

    print("\n===== Drug2Reaction RAG 基线评估 =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("=======================================")

    # 写入结果文件，便于留档和后续对比
    with open(RESULT_JSON, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)
    print(f"[saved] 评估结果已写入: {RESULT_JSON}")


if __name__ == "__main__":
    main()
