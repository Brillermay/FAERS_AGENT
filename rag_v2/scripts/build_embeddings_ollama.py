import os
import time
from typing import List, Dict, Tuple
import requests
import numpy as np
from py2neo import Graph
from tqdm import tqdm

# 使用统一的 embedding 工具模块（已移动到 core 子包）
from rag_v2.core.embedding_utils import embed_batch, get_current_model_name

# ----------- 环境配置 -----------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rainshineking274")

BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "128"))
SKIP_EXISTING = os.getenv("EMBED_SKIP_EXISTING", "true").lower() == "true"
LABELS_ENV = os.getenv("EMBED_LABELS", "Drug,Reaction,Indication,Outcome")
PROGRESS_MIN_INTERVAL = float(os.getenv("EMBED_PROGRESS_REFRESH", "0.5"))

LABELS = [x.strip() for x in LABELS_ENV.split(",") if x.strip()]

LABEL_PROP: Dict[str, str] = {
    "Drug": "drugname",
    "Reaction": "reac",
    "Indication": "indi",
    "Outcome": "outccode",
}

graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ----------- Neo4j 批处理 -----------
def count_pending(label: str) -> int:
    prop = LABEL_PROP[label]
    # 修复：当 skip=false 时，选择所有节点；当 skip=true 时，只选择没有 embedding 的节点
    q = f"""
    MATCH (n:{label})
    WHERE n.{prop} IS NOT NULL
      AND (NOT $skip OR n.embedding IS NULL)
    RETURN count(n) AS c
    """
    return int(graph.run(q, skip=SKIP_EXISTING).evaluate() or 0)

def fetch_batch(label: str, last_id: int, limit: int) -> List[Tuple[int, str]]:
    prop = LABEL_PROP[label]
    # 修复：当 skip=false 时，选择所有节点；当 skip=true 时，只选择没有 embedding 的节点
    q = f"""
    MATCH (n:{label})
    WHERE id(n) > $lastId
      AND n.{prop} IS NOT NULL
      AND (NOT $skip OR n.embedding IS NULL)
    RETURN id(n) AS id, n.{prop} AS text
    ORDER BY id(n) ASC
    LIMIT $limit
    """
    rows = graph.run(q, lastId=last_id, skip=SKIP_EXISTING, limit=limit).data()
    return [(int(r["id"]), r["text"]) for r in rows]

def write_embeddings(pairs: List[Tuple[int, List[float]]], model_name: str = None):
    if not pairs:
        return
    if model_name is None:
        model_name = get_current_model_name()
    q = """
    UNWIND $pairs AS p
    MATCH (n) WHERE id(n)=p.id
    SET n.embedding = p.emb,
        n.emd_version = $model
    """
    graph.run(q, pairs=[{"id": nid, "emb": emb} for nid, emb in pairs], model=model_name)

def process_label(label: str):
    if label not in LABEL_PROP:
        print(f"[skip] 未知标签: {label}")
        return
    total_pending = count_pending(label)
    if total_pending == 0:
        print(f"[{label}] 无需处理（无待向量化节点）")
        return

    print(f"\n=== 处理标签: {label} | 模型: {get_current_model_name()} | skip_existing={SKIP_EXISTING} | 待处理={total_pending} ===")
    last_id = -1
    processed = 0

    bar = tqdm(
        total=total_pending,
        desc=f"{label}",
        unit="node",
        dynamic_ncols=True,
        mininterval=PROGRESS_MIN_INTERVAL,
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    while True:
        batch = fetch_batch(label, last_id, BATCH_SIZE)
        if not batch:
            break
        ids = [nid for nid, _ in batch]
        texts = [txt or "" for _, txt in batch]

        # 使用统一的批量向量化接口
        vecs = embed_batch(texts, batch_size=BATCH_SIZE)
        write_embeddings(list(zip(ids, vecs)))

        processed += len(ids)
        last_id = ids[-1]
        bar.update(len(ids))

    bar.close()
    print(f"[done] {label} 完成，写入 {processed}")

def main():
    print(f"Neo4j: {NEO4J_URI} as {NEO4J_USER}")
    print(f"Embedding 模型: {get_current_model_name()}")
    print(f"Labels: {', '.join(LABELS)} | batch={BATCH_SIZE} | skip_existing={SKIP_EXISTING}")
    for label in LABELS:
        process_label(label)
    print("\n全部完成。")

if __name__ == "__main__":
    main()