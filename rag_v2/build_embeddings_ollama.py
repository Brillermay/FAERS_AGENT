import os
import time
from typing import List, Dict, Tuple
import requests
import numpy as np
from py2neo import Graph
from tqdm import tqdm

# ----------- 环境配置 -----------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rainshineking274")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

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

# ----------- Ollama 客户端 -----------
def _l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        return arr.astype(np.float32).tolist()
    return (arr / n).astype(np.float32).tolist()

def embed_one(text: str) -> List[float]:
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": OLLAMA_MODEL, "prompt": text or ""}
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError(f"Invalid embedding response: {data}")
    return _l2_normalize(vec)

def embed_batch(texts: List[str], retries: int = 3, sleep: float = 1.0) -> List[List[float]]:
    out: List[List[float]] = []
    for t in texts:
        last_err = None
        for _ in range(retries):
            try:
                out.append(embed_one(t))
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep)
        if last_err:
            raise last_err
    return out

# ----------- Neo4j 批处理 -----------
def count_pending(label: str) -> int:
    prop = LABEL_PROP[label]
    q = f"""
    MATCH (n:{label})
    WHERE n.{prop} IS NOT NULL
      AND ($skip = false OR n.embedding IS NULL)
    RETURN count(n) AS c
    """
    return int(graph.run(q, skip=SKIP_EXISTING).evaluate() or 0)

def fetch_batch(label: str, last_id: int, limit: int) -> List[Tuple[int, str]]:
    prop = LABEL_PROP[label]
    q = f"""
    MATCH (n:{label})
    WHERE id(n) > $lastId
      AND n.{prop} IS NOT NULL
      AND ($skip = false OR n.embedding IS NULL)
    RETURN id(n) AS id, n.{prop} AS text
    ORDER BY id(n) ASC
    LIMIT $limit
    """
    rows = graph.run(q, lastId=last_id, skip=SKIP_EXISTING, limit=limit).data()
    return [(int(r["id"]), r["text"]) for r in rows]

def write_embeddings(pairs: List[Tuple[int, List[float]]]):
    if not pairs:
        return
    q = """
    UNWIND $pairs AS p
    MATCH (n) WHERE id(n)=p.id
    SET n.embedding = p.emb,
        n.emd_version = $model
    """
    graph.run(q, pairs=[{"id": nid, "emb": emb} for nid, emb in pairs], model=OLLAMA_MODEL)

def process_label(label: str):
    if label not in LABEL_PROP:
        print(f"[skip] 未知标签: {label}")
        return
    total_pending = count_pending(label)
    if total_pending == 0:
        print(f"[{label}] 无需处理（无待向量化节点）")
        return

    print(f"\n=== 处理标签: {label} | 模型: {OLLAMA_MODEL} | skip_existing={SKIP_EXISTING} | 待处理={total_pending} ===")
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

        vecs = embed_batch(texts)
        write_embeddings(list(zip(ids, vecs)))

        processed += len(ids)
        last_id = ids[-1]
        bar.update(len(ids))

    bar.close()
    print(f"[done] {label} 完成，写入 {processed}")

def main():
    print(f"Neo4j: {NEO4J_URI} as {NEO4J_USER}")
    print(f"Ollama: {OLLAMA_BASE_URL} | model={OLLAMA_MODEL}")
    print(f"Labels: {', '.join(LABELS)} | batch={BATCH_SIZE} | skip_existing={SKIP_EXISTING}")
    for label in LABELS:
        process_label(label)
    print("\n全部完成。")

if __name__ == "__main__":
    main()