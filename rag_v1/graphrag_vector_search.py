# 基于输入向量的简单 GraphRAG：本地相似度检索 + OpenAI 生成回答
import os
import sys
import numpy as np
from typing import List, Tuple, Dict
from py2neo import Graph
from openai import OpenAI

# 将 embedding 目录加入模块搜索路径，便于导入 text_embedder
CURRENT_DIR = os.path.dirname(__file__)
EMBED_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "embedding"))
if EMBED_DIR not in sys.path:
    sys.path.insert(0, EMBED_DIR)

from text_embedder import embed_query  # noqa: E402

# OpenAI 聊天配置（与无向量版本一致）
client = OpenAI(
    api_key="sk-7cuAMuvifXU3TogzrjZahugbyKSnHUZPmZ68SfzpxRFuJWLn",
    base_url="https://api.chatanywhere.tech/v1",
)

# Neo4j 连接
graph = Graph("bolt://localhost:7687", auth=("neo4j", "rainshineking274"))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def _resize_vector(vec, target_dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.size == target_dim:
        pass
    elif arr.size > target_dim:
        arr = arr[:target_dim]
    else:
        arr = np.pad(arr, (0, target_dim - arr.size))
    n = np.linalg.norm(arr)
    return arr / (n + 1e-12)

def _load_label_vectors(label: str, id_field: str, show_fields: List[str]) -> List[Dict]:
    """
    仅加载必要字段与 embedding，避免返回整节点。
    去重 id_field 与 show_fields，防止重复列名。
    """
    fields_ordered = []
    for f in [id_field] + (show_fields or []):
        if f and f not in fields_ordered:
            fields_ordered.append(f)

    select_fields = ", ".join([f"n.{f} AS {f}" for f in fields_ordered]) if fields_ordered else ""
    select_clause = f"{select_fields}, n.embedding AS emb" if select_fields else "n.embedding AS emb"

    query = f"""
    MATCH (n:{label})
    WHERE n.embedding IS NOT NULL
    RETURN {select_clause}
    """
    return graph.run(query).data()

def _rank_by_similarity(query_vec: List[float], rows: List[Dict], id_field: str, show_fields: List[str], topk: int) -> List[Tuple[float, Dict]]:
    if not rows:
        return []
    target_dim = len(rows[0]["emb"])
    q = _resize_vector(query_vec, target_dim)
    ranked = []
    for r in rows:
        emb = np.asarray(r["emb"], dtype=float)
        if emb.size != target_dim:
            emb = _resize_vector(emb, target_dim)
        sim = _cosine(q, emb)
        ranked.append((sim, r))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[:topk]

def _fmt_rows(rows: List[Tuple[float, Dict]], id_field: str, show_fields: List[str]) -> str:
    if not rows:
        return "（无检索结果）"
    lines = []
    for sim, r in rows:
        display = " | ".join([f"{f}:{r.get(f, '')}" for f in show_fields if f in r])
        lines.append(f"- {display} | 相似度:{sim:.4f}")
    return "\n".join(lines)

# 结构化补充检索（与无向量版本复用）
def top_reactions_for_drug(drug: str, k: int = 10):
    q = """
    MATCH (d:Drug)-[:CAUSES_REACTION]->(r:Reaction)
    WHERE toLower(d.drugname) = toLower($drug)
    RETURN r.reac AS reaction, count(*) AS freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, drug=drug, k=k).data()

def top_drugs_for_reaction(reac: str, k: int = 10):
    q = """
    MATCH (d:Drug)-[:CAUSES_REACTION]->(r:Reaction)
    WHERE toLower(r.reac) = toLower($reac)
    RETURN d.drugname AS drug, count(*) AS freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, reac=reac, k=k).data()

def indications_for_drug(drug: str, k: int = 10):
    q = """
    MATCH (d:Drug)-[:HAS_INDICATION]->(i:Indication)
    WHERE toLower(d.drugname) = toLower($drug)
    RETURN i.indi AS indication, count(*) AS freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, drug=drug, k=k).data()

def llm_answer(question: str, context: str) -> str:
    sys_prompt = (
        "你是医学药物安全助手。仅依据提供的图谱检索结果作答，"
        "若信息不足请明确说明。回答简洁、条理清晰。"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"问题：{question}\n\n检索结果：\n{context}"},
    ]
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def retrieve_with_embeddings(query_text: str, k_each: int = 5) -> str:
    """
    将输入转 embedding，分别在 Drug/Reaction/Indication 上做相似度检索，合并为上下文。
    """
    qvec = embed_query(query_text)

    drug_rows = _load_label_vectors("Drug", "drugname", ["drugname"])
    reac_rows = _load_label_vectors("Reaction", "reac", ["reac"])
    indi_rows = _load_label_vectors("Indication", "indi", ["indi"])

    drugs_top = _rank_by_similarity(qvec, drug_rows, "drugname", ["drugname"], k_each)
    reac_top = _rank_by_similarity(qvec, reac_rows, "reac", ["reac"], k_each)
    indi_top = _rank_by_similarity(qvec, indi_rows, "indi", ["indi"], k_each)

    ctx = []
    ctx.append("相似药物（Drug）:")
    ctx.append(_fmt_rows(drugs_top, "drugname", ["drugname"]))
    ctx.append("\n相似不良反应（Reaction）:")
    ctx.append(_fmt_rows(reac_top, "reac", ["reac"]))
    ctx.append("\n相似适应症（Indication）:")
    ctx.append(_fmt_rows(indi_top, "indi", ["indi"]))

    # 简单规则：若最相似的是某个药物，追加其常见反应与适应症
    if drugs_top:
        top_drug = drugs_top[0][1].get("drugname", "")
        if top_drug:
            dr = top_reactions_for_drug(top_drug, k=10)
            di = indications_for_drug(top_drug, k=10)
            if dr:
                ctx.append(f"\n与药物 {top_drug} 相关的高频不良反应：")
                ctx.extend([f"- {r['reaction']}（{r['freq']}）" for r in dr])
            if di:
                ctx.append(f"\n与药物 {top_drug} 相关的高频适应症：")
                ctx.extend([f"- {r['indication']}（{r['freq']}）" for r in di])

    return "\n".join(ctx)

def demo():
    questions = [
        "阿司匹林的常见不良反应有哪些？",
        "NAUSEA 这个不良反应最常见于哪些药物？",
        "有哪些药物与心血管相关？",
    ]
    for q in questions:
        print("=" * 60)
        print("问题：", q)
        ctx = retrieve_with_embeddings(q, k_each=5)
        print("检索上下文：\n", ctx)
        ans = llm_answer(q, ctx)
        print("\n回答：\n", ans)

if __name__ == "__main__":
    demo()