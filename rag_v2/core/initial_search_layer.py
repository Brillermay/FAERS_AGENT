# 第 2 层：根据输入层 JSON 在 Neo4j 中做初步查询，返回候选节点及其属性/向量
import json
from collections import defaultdict
from typing import Dict, List, Any

from py2neo import Graph
from py2neo.errors import ClientError

from rag_v2.core.input_layer import parse_input  # 便于单独调试，可直接复用第 1 层

# ---------- Neo4j 连接 ----------
graph = Graph("bolt://localhost:7687", auth=("neo4j", "rainshineking274"))

# 每类节点的主属性 & 全文索引名
LABEL_INFO: Dict[str, Dict[str, str]] = {
    "Drug": {"prop": "drugname", "fulltext": "drug_ft"},
    "Reaction": {"prop": "reac", "fulltext": "reaction_ft"},
    "Indication": {"prop": "indi", "fulltext": "indication_ft"},
    "Outcome": {"prop": "outccode", "fulltext": "outcome_ft"},
}

def _fulltext_query(label: str, term: str, limit: int) -> List[Dict[str, Any]]:
    info = LABEL_INFO[label]
    prop = info["prop"]
    index_name = info["fulltext"]

    query = f"""
    CALL db.index.fulltext.queryNodes($index, $term) YIELD node, score
    RETURN id(node) AS id, node.{prop} AS value, node.embedding AS embedding, score
    ORDER BY score DESC
    LIMIT $limit
    """
    return graph.run(query, index=index_name, term=term, limit=limit).data()

def _contains_query(label: str, term: str, limit: int) -> List[Dict[str, Any]]:
    info = LABEL_INFO[label]
    prop = info["prop"]
    query = f"""
    MATCH (n:{label})
    WHERE toLower(n.{prop}) CONTAINS toLower($term)
    RETURN id(n) AS id, n.{prop} AS value, n.embedding AS embedding
    ORDER BY toLower(n.{prop}) ASC
    LIMIT $limit
    """
    rows = graph.run(query, term=term, limit=limit).data()
    for r in rows:
        r["score"] = None
    return rows

def _fallback_all(label: str, limit: int) -> List[Dict[str, Any]]:
    """无 seed 时兜底返回若干节点，确保 intents 中的类型也有候选。"""
    info = LABEL_INFO[label]
    prop = info["prop"]
    query = f"""
    MATCH (n:{label})
    RETURN id(n) AS id, n.{prop} AS value, n.embedding AS embedding
    ORDER BY toLower(n.{prop}) ASC
    LIMIT $limit
    """
    rows = graph.run(query, limit=limit).data()
    for r in rows:
        r["score"] = None
    return rows

def _search_single_seed(label: str, term: str, limit: int) -> List[Dict[str, Any]]:
    if not term:
        return []
    try:
        rows = _fulltext_query(label, term, limit)
        if rows:
            return rows
    except ClientError:
        # 全文索引不存在，回退到 CONTAINS
        pass
    except Exception:
        pass
    return _contains_query(label, term, limit)

def initial_search(parsed: Dict[str, Any], per_seed_limit: int = 40, fallback_limit: int = 20) -> Dict[str, Any]:
    """
    输入：第 1 层 parse_input 的结果
    输出：每类节点的候选列表，包含 id / value / embedding / score
    """
    seeds = parsed.get("seeds", [])
    intents = parsed.get("intents", [])
    topk = parsed.get("topk", 10)

    candidates: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    for seed in seeds:
        label = seed["type"]
        if label not in LABEL_INFO:
            continue
        term = seed.get("normalized") or seed.get("text")
        rows = _search_single_seed(label, term, per_seed_limit)
        for r in rows:
            nid = int(r["id"])
            existing = candidates[label].get(nid)
            if not existing or (r.get("score") or 0) > (existing.get("score") or 0):
                candidates[label][nid] = {
                    "id": nid,
                    "label": label,
                    "value": r.get("value"),
                    "score": r.get("score"),
                    "embedding": r.get("embedding"),
                    "seed": term,
                }

    # 对 intents 中要求的类型，如果当前还没有候选，则取全量兜底
    for label in intents:
        if label not in LABEL_INFO:
            continue
        if not candidates[label]:
            rows = _fallback_all(label, fallback_limit)
            for r in rows:
                nid = int(r["id"])
                candidates[label][nid] = {
                    "id": nid,
                    "label": label,
                    "value": r.get("value"),
                    "score": r.get("score"),
                    "embedding": r.get("embedding"),
                    "seed": None,
                }

    # 将 dict 转为 list，并按 score/value 排序
    result_items: Dict[str, List[Dict[str, Any]]] = {}
    for label, table in candidates.items():
        items = list(table.values())
        items.sort(key=lambda x: (x["score"] is not None, x["score"], x["value"]), reverse=True)
        result_items[label] = items[: max(topk, 10)]  # 适当多保留一些给下一层

    output = {
        "question": parsed["question"],
        "seeds": parsed["seeds"],
        "intents": intents,
        "topk": topk,
        "query_vector": parsed.get("query_vector"),  # 直接透传（第1层已生成）
        "candidates": result_items,
        "meta": {
            "input_meta": parsed.get("meta"),
            "per_seed_limit": per_seed_limit,
            "fallback_limit": fallback_limit,
        },
    }

    # ========== 屏蔽向量具体数值，仅显示维度 ==========
    display_output = {
        k: (f"dim={len(v)}" if k == "query_vector" and isinstance(v, list) else v)
        for k, v in output.items()
    }
    # 遍历 candidates，替换 embedding 为维度说明
    masked_candidates = {}
    for lbl, rows in output["candidates"].items():
        masked_rows = []
        for r in rows:
            rr = dict(r)
            emb = rr.get("embedding")
            if isinstance(emb, list):
                rr["embedding"] = f"dim={len(emb)}"
            elif emb is None:
                rr["embedding"] = None
            masked_rows.append(rr)
        masked_candidates[lbl] = masked_rows
    display_output["candidates"] = masked_candidates
    # ===============================================

    print("======== 初步查询层输出 ========")
    print(json.dumps(display_output, ensure_ascii=False, indent=2))
    print("================================")

    return output

# --- 简单联调（第 1 层 -> 第 2 层） ---
if __name__ == "__main__":
    questions = [
        "阿司匹林的常见不良反应有哪些？",
        "NAUSEA 常见于哪些药物？",
        "阿司匹林一般用于哪些适应症？",
        "关于氯吡格雷，有哪些患者结局？",
        "请列出常见药物。",
    ]
    for q in questions:
        print("\n问题：", q)
        parsed = parse_input(q)
        initial_search(parsed)