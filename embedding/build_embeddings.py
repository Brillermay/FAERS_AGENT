# 本地向量化脚本：使用 multilingual-e5-base 将各类节点的“必要属性”转为向量并写回 Neo4j
import os
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from py2neo import Graph
import torch
from sentence_transformers import SentenceTransformer

# ========== 配置 ==========
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rainshineking274")

MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")  # 768 维
DEVICE = os.getenv("EMBED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
SKIP_EXISTING = os.getenv("EMBED_SKIP_EXISTING", "true").lower() == "true"

# 需要向量化的节点及其“必要属性”（来自 build_kg_utils.py）
LABEL_FIELDS: Dict[str, List[str]] = {
    "Drug": ["drugname", "prod_ai", "dose_freq", "dose_amt", "dose_unit"],
    "Reaction": ["reac"],
    "Indication": ["indi"],
    "Outcome": ["outccode"],
    # Patient 信息多为数值或枚举，这里保留关键属性，按需开启
    "Patient": ["age", "age_grp", "sex", "wt", "occr_country", "event_dt"],
    # DrugSet 仅 primaryid/caseid，无语义文本，默认不做节点向量
}

VECTOR_INDEX_SPECS = {
    # label: (property, dim, similarity)
    "Drug": ("embedding", 768, "cosine"),
    "Reaction": ("embedding", 768, "cosine"),
    "Indication": ("embedding", 768, "cosine"),
    "Outcome": ("embedding", 768, "cosine"),
    "Patient": ("embedding", 768, "cosine"),
}

# ========== 模型准备 ==========
_model = SentenceTransformer(MODEL_NAME, device=DEVICE)

def _embed_passages(texts: List[str]) -> List[List[float]]:
    """E5 规范：库侧用 'passage: ' 前缀，返回 float32 且已 L2 归一。"""
    prefixed = ["passage: " + (t or "") for t in texts]
    vecs = _model.encode(
        prefixed,
        batch_size=min(1024, max(16, BATCH_SIZE)),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32).tolist()

# ========== Neo4j 工具 ==========
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def ensure_vector_indexes():
    """创建各标签的向量索引（存在则跳过）。"""
    for label, (prop, dim, sim) in VECTOR_INDEX_SPECS.items():
        try:
            idx_name = f"{label.lower()}_{prop}_idx"
            cypher = (
                f"CREATE VECTOR INDEX {idx_name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop}) "
                f"OPTIONS {{indexConfig: {{'vector.dimensions': {dim}, 'vector.similarity_function': '{sim}'}}}}"
            )
            graph.run(cypher)
        except Exception:
            # 忽略没有企业版/权限等异常
            pass

def _build_text(row: Dict[str, Any], fields: List[str], label: str) -> str:
    """将必要属性拼接为简短描述文本，用于向量化。"""
    parts: List[str] = []
    for f in fields:
        v = row.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.upper() == "NULL":
            continue
        parts.append(f"{f}:{s}")
    # 对不同标签做轻量提示（有助于语义区分）
    hint = f"type:{label}"
    return " | ".join([hint] + parts) if parts else hint

def _load_nodes(label: str, fields: List[str], skip_existing: bool) -> List[Dict[str, Any]]:
    """拉取需要向量化的节点的必要字段。"""
    selects = ", ".join([f"n.{f} AS {f}" for f in fields])
    where = "WHERE n.embedding IS NULL" if skip_existing else ""
    query = f"""
    MATCH (n:{label}) {where}
    RETURN id(n) AS id{", " if selects else ""}{selects}
    """
    return graph.run(query).data()

def _update_embeddings(pairs: List[Dict[str, Any]]):
    """批量写回 embedding。"""
    query = """
    UNWIND $pairs AS pair
    MATCH (n) WHERE id(n) = pair.id
    SET n.embedding = pair.embedding
    """
    graph.run(query, pairs=pairs)

def build_label_embeddings(label: str, fields: List[str], skip_existing: bool = True, batch_size: int = BATCH_SIZE):
    rows = _load_nodes(label, fields, skip_existing)
    if not rows:
        print(f"[{label}] 无需更新（skip_existing={skip_existing}）")
        return

    print(f"[{label}] 待向量化节点数: {len(rows)} | 字段: {fields}")
    texts: List[str] = []
    ids: List[int] = []
    for r in rows:
        ids.append(int(r["id"]))
        texts.append(_build_text(r, fields, label))

    # 分批嵌入与写回
    for i in tqdm(range(0, len(ids), batch_size), desc=f"embed {label}", unit="batch"):
        sub_ids = ids[i : i + batch_size]
        sub_texts = texts[i : i + batch_size]
        vecs = _embed_passages(sub_texts)
        pairs = [{"id": nid, "embedding": vec} for nid, vec in zip(sub_ids, vecs)]
        _update_embeddings(pairs)

    print(f"[{label}] 向量写回完成")

def main():
    print(f"使用模型: {MODEL_NAME} | 设备: {DEVICE} | 批大小: {BATCH_SIZE} | 跳过已存在: {SKIP_EXISTING}")
    ensure_vector_indexes()

    # 按标签依次生成节点向量
    for label, fields in LABEL_FIELDS.items():
        # DrugSet 默认跳过；如需为 DrugSet 生成可在 LABEL_FIELDS 中添加自定义文本来源
        build_label_embeddings(label, fields, skip_existing=SKIP_EXISTING, batch_size=BATCH_SIZE)

    print("全部节点向量化完成。")

if __name__ == "__main__":
    main()