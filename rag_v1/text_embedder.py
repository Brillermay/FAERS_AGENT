import os
import numpy as np
from typing import List
from openai import OpenAI

# 建议将 key 设置到环境变量 OPENAI_API_KEY；未设置时使用此默认值
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-7cuAMuvifXU3TogzrjZahugbyKSnHUZPmZ68SfzpxRFuJWLn")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # 1536 维

_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def _l2_normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=float)
    n = np.linalg.norm(arr)
    return (arr / (n + 1e-12)).tolist()

def embed_query(text: str) -> List[float]:
    """
    将单条文本转为 L2 归一化后的 embedding。
    """
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = resp.data[0].embedding
    return _l2_normalize(vec)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    批量文本转 embedding（逐条调用，便于小规模测试）。
    返回均为 L2 归一化后的向量。
    """
    out = []
    for t in texts:
        resp = _client.embeddings.create(model=EMBED_MODEL, input=t)
        out.append(_l2_normalize(resp.data[0].embedding))
    return out

if __name__ == "__main__":
    demo_vec = embed_query("阿司匹林的常见不良反应有哪些？")
    print(f"维度: {len(demo_vec)}, 前5维: {demo_vec[:5]}")