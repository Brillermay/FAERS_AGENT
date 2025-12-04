# encoding: utf-8
"""
导出评估用的 Drug→Reaction 统计信息

功能：
1. 从 Neo4j 统计每个药物与各不良反应的共现频次（按 distinct DrugSet）。
2. 可选：只保留病例数较多的“常见药物”。
3. 导出为 CSV 和/或 JSON，供后续评估集与合成问题使用。
"""

import os
import csv
import json
from typing import List, Dict, Any

from py2neo import Graph

# 与你现有代码保持一致的默认配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rainshineking274")

# 导出参数
MIN_CASES_PER_DRUG = 50     # 只保留病例数 >= 50 的药物，可按需要调整
TOP_REACTIONS_PER_DRUG = 50 # 每个药物最多保留前 50 个反应，可按需要调整

# 统一将数据输出到 rag_v2/data 目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_CSV = os.path.join(DATA_DIR, "drug_reaction_stats.csv")
OUT_JSON = os.path.join(DATA_DIR, "drug_reaction_stats.json")


def get_graph() -> Graph:
    return Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def fetch_common_drugs(graph: Graph, min_cases: int) -> List[str]:
    """
    返回病例数 >= min_cases 的常见药物名列表
    """
    cypher = """
    MATCH (ds:DrugSet)-[:CONTAINS_DRUG]->(d:Drug)
    WITH d.drugname AS drugname, count(DISTINCT ds) AS case_cnt
    WHERE case_cnt >= $min_cases
    RETURN drugname, case_cnt
    ORDER BY case_cnt DESC
    """
    rows = graph.run(cypher, min_cases=min_cases).data()
    print(f"[info] 常见药物数: {len(rows)} (case_cnt >= {min_cases})")
    return [r["drugname"] for r in rows]


def fetch_drug_reaction_stats(
    graph: Graph,
    drug_filter: List[str] = None,
    top_per_drug: int = TOP_REACTIONS_PER_DRUG,
) -> List[Dict[str, Any]]:
    """
    返回列表：每个元素是 {drugname, reac, freq}
    可以传入 drug_filter 只统计指定药物。
    """
    if drug_filter:
        cypher = """
        MATCH (ds:DrugSet)-[:CONTAINS_DRUG]->(d:Drug)
        WHERE d.drugname IN $druglist
        MATCH (ds)-[:CAUSES_REACTION]->(r:Reaction)
        WITH d.drugname AS drugname, r.reac AS reac, count(DISTINCT ds) AS freq
        RETURN drugname, reac, freq
        ORDER BY drugname, freq DESC
        """
        params = {"druglist": drug_filter}
    else:
        cypher = """
        MATCH (ds:DrugSet)-[:CONTAINS_DRUG]->(d:Drug)
        MATCH (ds)-[:CAUSES_REACTION]->(r:Reaction)
        WITH d.drugname AS drugname, r.reac AS reac, count(DISTINCT ds) AS freq
        RETURN drugname, reac, freq
        ORDER BY drugname, freq DESC
        """
        params = {}

    rows = graph.run(cypher, **params).data()

    # 按药物分组后截断到 top_per_drug
    result: List[Dict[str, Any]] = []
    current_drug = None
    count_for_drug = 0

    for r in rows:
        drug = r["drugname"]
        if drug != current_drug:
            current_drug = drug
            count_for_drug = 0

        if count_for_drug < top_per_drug:
            result.append(
                {
                    "drugname": drug,
                    "reac": r["reac"],
                    "freq": int(r["freq"]),
                }
            )
            count_for_drug += 1

    print(f"[info] 总记录数: {len(result)} (已按每个药物 top {top_per_drug} 截断)")
    return result


def save_as_csv(records: List[Dict[str, Any]], path: str):
    if not records:
        print("[warn] 无记录可写入 CSV")
        return
    fields = ["drugname", "reac", "freq"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"[info] CSV 已写入: {path}")


def save_as_json(records: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[info] JSON 已写入: {path}")


def main():
    graph = get_graph()
    print(f"[info] 连接 Neo4j: {NEO4J_URI} as {NEO4J_USER}")

    # 1) 先选“常见药物”作为评估候选（可根据需要关掉这一步）
    common_drugs = fetch_common_drugs(graph, MIN_CASES_PER_DRUG)

    # 2) 导出 Drug→Reaction 统计（只针对常见药物）
    stats = fetch_drug_reaction_stats(
        graph,
        drug_filter=common_drugs,
        top_per_drug=TOP_REACTIONS_PER_DRUG,
    )

    # 3) 保存为 CSV 和 JSON
    save_as_csv(stats, OUT_CSV)
    save_as_json(stats, OUT_JSON)


if __name__ == "__main__":
    main()