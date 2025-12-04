# encoding: utf-8
"""
统一的 Embedding 工具模块，支持切换不同的向量模型。
支持：
- 基线模型：Ollama nomic-embed-text（通过 API）
- 新模型：本地双塔模型（SentenceTransformer）
"""

import os
from typing import List, Optional
import numpy as np

# ===== 配置 =====
# 默认使用新训练的双塔模型，可通过环境变量切换
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "biencoder")  # 改为默认 "biencoder"
BIENCODER_MODEL_PATH = os.getenv("BIENCODER_MODEL_PATH", r"D:\models\drug2reaction_biencoder_trial")

# Ollama 配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# 全局模型实例（延迟加载）
_biencoder_model = None


def _l2_normalize(vec: List[float]) -> List[float]:
    """L2 归一化向量"""
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        return arr.astype(np.float32).tolist()
    return (arr / n).astype(np.float32).tolist()


def _get_biencoder_model():
    """延迟加载双塔模型（避免导入时加载）"""
    global _biencoder_model
    if _biencoder_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _biencoder_model = SentenceTransformer(BIENCODER_MODEL_PATH)
            print(f"[info] 已加载双塔模型: {BIENCODER_MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"双塔模型加载失败: {e}\n路径: {BIENCODER_MODEL_PATH}")
    return _biencoder_model


def embed_ollama(text: str) -> List[float]:
    """使用 Ollama 基线模型生成向量"""
    import requests
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": OLLAMA_MODEL, "prompt": text or ""}
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError(f"Ollama 返回格式异常: {data}")
    return _l2_normalize(vec)


def embed_biencoder(text: str) -> List[float]:
    """使用新训练的双塔模型生成向量"""
    model = _get_biencoder_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist() if hasattr(vec, 'tolist') else list(vec)


def embed_text(text: str) -> List[float]:
    """
    统一的文本向量化接口，根据环境变量自动选择模型
    
    Args:
        text: 待向量化的文本
        
    Returns:
        归一化后的向量（List[float]）
    """
    if not text or not text.strip():
        raise ValueError("输入文本不能为空")
    
    text = text.strip()
    
    if EMBEDDING_MODEL_TYPE.lower() == "biencoder":
        return embed_biencoder(text)
    else:  # 默认使用 ollama
        return embed_ollama(text)


def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    批量向量化（对新模型更高效）
    
    Args:
        texts: 文本列表
        batch_size: 批大小（仅对 biencoder 有效）
        
    Returns:
        向量列表
    """
    if not texts:
        return []
    
    if EMBEDDING_MODEL_TYPE.lower() == "biencoder":
        model = _get_biencoder_model()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        return [v.tolist() if hasattr(v, 'tolist') else list(v) for v in vectors]
    else:
        # Ollama 需要逐个调用
        import requests
        return [embed_ollama(t) for t in texts]


def get_current_model_name() -> str:
    """获取当前使用的模型名称"""
    if EMBEDDING_MODEL_TYPE.lower() == "biencoder":
        return f"biencoder({BIENCODER_MODEL_PATH})"
    else:
        return f"ollama({OLLAMA_MODEL})"


# 导出主要接口
__all__ = ["embed_text", "embed_batch", "get_current_model_name"]
