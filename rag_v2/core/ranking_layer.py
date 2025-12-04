# 第 4 层：向量重排层（改进版）
# 改动点（采纳三点）：
# 1. 若所有 freq 相同 => 退化为纯语义排序(score=sim)，scoring_mode=uniform_sim
# 2. 自适应频次归一：若 max_freq/min_freq > 阈值(默认5) 用线性 freq/max_freq；否则用 log1p 方案
# 3. 过滤 embedding 缺失的行（不参与排序），并在 stats 中记录 missing_embedding_count
import json
import math
import os
from typing import Dict, Any, List
import numpy as np

from rag_v2.core.input_layer import parse_input
from rag_v2.core.initial_search_layer import initial_search
from rag_v2.core.relation_aggregate_layer import aggregate_relations

ALPHA = float(os.getenv("RANK_ALPHA", "0.6"))  # 语义权重
TOPK_FALLBACK = int(os.getenv("RANK_TOPK_LIMIT", "50"))
DISPARITY_THRESHOLD = float(os.getenv("RANK_FREQ_DISPARITY_THRESHOLD", "5"))  # 频次差异阈值

def _dot(a: List[float], b: List[float]) -> float:
    # 原来只检查空，现在加上维度检查，避免 1024 vs 768 这种情况直接报错
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # 维度不一致的旧向量，直接忽略（相似度视为 0）
        return 0.0
    return float(np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))

def _rank_rows(rows: List[Dict[str, Any]], query_vec: List[float], alpha: float) -> Dict[str, Any]:
    """
    返回:
      {
        'ranked': [...],
        'stats': {
            'original_count': int,
            'used_count': int,
            'missing_embedding_count': int,
            'scoring_mode': str,
            'freq_mode': str,
            'alpha': float
        }
      }
    """
    if not rows:
        return {"ranked": [], "stats": {
            "original_count": 0,
            "used_count": 0,
            "missing_embedding_count": 0,
            "scoring_mode": "none",
            "freq_mode": "none",
            "alpha": alpha
        }}

    # 过滤缺失 embedding
    usable = [r for r in rows if isinstance(r.get("embedding"), list)]
    missing_cnt = len(rows) - len(usable)

    if not usable:
        # 全部缺失 => 直接空
        return {"ranked": [], "stats": {
            "original_count": len(rows),
            "used_count": 0,
            "missing_embedding_count": missing_cnt,
            "scoring_mode": "all_missing",
            "freq_mode": "none",
            "alpha": alpha
        }}

    freqs = [(r.get("freq", 0) or 0) for r in usable]
    max_freq = max(freqs)
    min_freq = min(freqs)

    ranked = []
    uniform = (max_freq == min_freq)
    if uniform:
        scoring_mode = "uniform_sim"
        freq_mode = "uniform"
    else:
        disparity = max_freq / max(min_freq, 1)
        if disparity > DISPARITY_THRESHOLD:
            freq_mode = "linear"
            scoring_mode = "alpha_mix"
        else:
            freq_mode = "log"
            scoring_mode = "alpha_mix"

    log_denom = math.log1p(max_freq) if not uniform and freq_mode == "log" else 1.0

    for r in usable:
        freq = r.get("freq", 0) or 0
        emb = r.get("embedding")
        sim = _dot(query_vec, emb) if query_vec and isinstance(emb, list) else 0.0

        if uniform:
            freq_norm = 1.0
            score = sim  # 退化为纯语义
        else:
            if freq_mode == "linear":
                freq_norm = freq / max_freq if max_freq > 0 else 0.0
            else:
                freq_norm = math.log1p(freq) / (log_denom + 1e-9)
            score = alpha * sim + (1 - alpha) * freq_norm

        ranked.append({
            "id": r.get("id"),
            "label": r.get("label"),
            "text": r.get("text"),
            "freq": freq,
            "sim": sim,
            "freq_norm": freq_norm,
            "score": score,
            "embedding": emb,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    stats = {
        "original_count": len(rows),
        "used_count": len(usable),
        "missing_embedding_count": missing_cnt,
        "scoring_mode": scoring_mode,
        "freq_mode": freq_mode,
        "alpha": alpha,
        "max_freq": max_freq,
        "min_freq": min_freq,
        "disparity": (max_freq / max(min_freq, 1)) if not uniform else 1.0
    }
    return {"ranked": ranked, "stats": stats}

def rank_expansions(aggregate_output: Dict[str, Any]) -> Dict[str, Any]:
    query_vec = aggregate_output.get("query_vector")
    topk = aggregate_output.get("topk") or TOPK_FALLBACK
    expansions = aggregate_output.get("expansions", {})

    ranked_intents: Dict[str, Any] = {}
    for intent, block in expansions.items():
        rows = block.get("rows", [])
        rank_result = _rank_rows(rows, query_vec, ALPHA)
        ranked_rows = rank_result["ranked"]
        stats = rank_result["stats"]
        stats["query_vector_dim"] = len(query_vec) if isinstance(query_vec, list) else None

        ranked_intents[intent] = {
            "source": block.get("source"),
            "columns": ["id", "label", "text", "freq", "sim", "freq_norm", "score", "embedding"],
            "rows": ranked_rows[:topk],
            "stats": stats
        }

    output = {
        "question": aggregate_output.get("question"),
        "intents": aggregate_output.get("intents"),
        "topk": topk,
        "query_vector": query_vec,
        "ranked": ranked_intents,
        "meta": {
            "layer": "ranking",
            "previous_meta": aggregate_output.get("meta"),
            "scoring": f"adaptive; uniform->sim; else score=alpha*sim+(1-alpha)*freq_norm; freq_mode=linear(if disparity>{DISPARITY_THRESHOLD})/log",
            "alpha": ALPHA,
            "freq_disparity_threshold": DISPARITY_THRESHOLD
        }
    }

    # 调试输出（隐藏向量内容）
    display_output = {
        k: (f"dim={len(v)}" if k == "query_vector" and isinstance(v, list) else v)
        for k, v in output.items()
    }
    masked_ranked = {}
    for intent, block in output["ranked"].items():
        blk = dict(block)
        masked_rows = []
        for r in blk["rows"]:
            rr = dict(r)
            emb = rr.get("embedding")
            if isinstance(emb, list):
                rr["embedding"] = f"dim={len(emb)}"
            elif emb is None:
                rr["embedding"] = None
            masked_rows.append(rr)
        blk["rows"] = masked_rows
        masked_ranked[intent] = blk
    display_output["ranked"] = masked_ranked

    print("======== 向量重排层输出 ========")
    print(json.dumps(display_output, ensure_ascii=False, indent=2))
    print("================================")
    return output

if __name__ == "__main__":
    tests = [
        "阿司匹林的常见不良反应有哪些？",
        "NAUSEA 常见于哪些药物？",
        "阿司匹林一般用于哪些适应症？",
        "关于氯吡格雷，有哪些患者结局？",
        "请列出常见药物。",
    ]
    for q in tests:
        print("\n问题：", q)
        parsed = parse_input(q)
        searched = initial_search(parsed)
        aggregated = aggregate_relations(searched)
        rank_expansions(aggregated)