# encoding: utf-8
"""
对比评估：基线模型（Ollama nomic-embed-text）vs 新训练的双塔模型（bge-m3 微调）
在 drug2reaction_eval.jsonl 上进行检索质量评估，输出详细对比报告。

保留测试痕迹，便于论文回顾。
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ===== 配置 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")

EVAL_FILE = os.path.join(DATA_DIR, "drug2reaction_eval.jsonl")
TOPK_PRED = 10

# 基线模型（Ollama）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# 新训练的双塔模型路径
BIENCODER_MODEL_PATH = os.getenv(
    "BIENCODER_MODEL_PATH",
    r"D:\models\drug2reaction_biencoder_trial"  # 本地路径
)

# 输出结果文件
BASELINE_RESULT_FILE = os.path.join(OUTPUT_DIR, "baseline_ollama_results.json")
NEW_MODEL_RESULT_FILE = os.path.join(OUTPUT_DIR, "biencoder_results.json")
COMPARISON_REPORT_FILE = os.path.join(OUTPUT_DIR, "comparison_report.json")

# Neo4j 连接（用于获取节点文本）
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "rainshineking274")

try:
    from py2neo import Graph
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
except Exception:
    graph = None
    print("[warn] Neo4j 连接失败，将使用评估集中的反应文本")


# ===== 工具函数 =====

def _l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        return arr.astype(np.float32).tolist()
    return (arr / n).astype(np.float32).tolist()


def embed_ollama(text: str) -> List[float]:
    """使用 Ollama 基线模型生成向量"""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {"model": OLLAMA_MODEL, "prompt": text or ""}
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    vec = data.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError(f"Ollama 返回格式异常: {data}")
    return _l2_normalize(vec)


def embed_biencoder(text: str, model: SentenceTransformer) -> List[float]:
    """使用新训练的双塔模型生成向量"""
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist() if hasattr(vec, 'tolist') else list(vec)


def cosine_sim(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def load_all_reactions_from_neo4j() -> Dict[str, str]:
    """从 Neo4j 加载所有 Reaction 节点的文本（用于构建文档库）"""
    if graph is None:
        return {}
    try:
        query = """
        MATCH (r:Reaction)
        WHERE r.reac IS NOT NULL
        RETURN r.reac AS reac
        """
        rows = graph.run(query).data()
        return {row["reac"]: f"Reaction: {row['reac']}" for row in rows}
    except Exception as e:
        print(f"[warn] 从 Neo4j 加载 Reaction 失败: {e}")
        return {}


def load_corpus_from_eval_file() -> Dict[str, str]:
    """从评估文件构建文档库（反应文本）"""
    corpus = {}
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("task_type") != "Drug2Reaction":
                continue
            answers = obj.get("answers", [])
            for reac in answers:
                if isinstance(reac, str) and reac.strip():
                    reac_norm = reac.strip()
                    corpus[reac_norm] = f"Reaction: {reac_norm}"
    return corpus


def build_reaction_corpus() -> Dict[str, str]:
    """构建反应文档库（优先用 Neo4j，否则用评估集）"""
    corpus = load_all_reactions_from_neo4j()
    if not corpus:
        print("[info] 使用评估集构建文档库")
        corpus = load_corpus_from_eval_file()
    else:
        print(f"[info] 从 Neo4j 加载了 {len(corpus)} 个 Reaction 节点")
        # 合并评估集中的反应（确保覆盖）
        eval_corpus = load_corpus_from_eval_file()
        corpus.update(eval_corpus)
    return corpus


def evaluate_one_model(
    model_name: str,
    embed_fn,
    corpus_embeddings: Dict[str, List[float]],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    评估单个模型在评估集上的性能
    返回: {
        "model_name": str,
        "metrics": {...},
        "timestamp": str,
        "details": [...]
    }
    """
    total = 0
    sum_hit = 0.0
    sum_recall = 0.0
    sum_mrr = 0.0
    details = []

    print(f"\n[评估开始] {model_name}")
    start_time = time.time()

    for idx, sample in enumerate(samples, 1):
        q = sample.get("question_zh", "").strip()
        gold = [a.strip() for a in sample.get("answers", []) if isinstance(a, str) and a.strip()]

        if not q or not gold:
            continue

        gold_set = set(x.lower() for x in gold)

        try:
            # 对问题生成向量
            query_vec = embed_fn(q)

            # 计算与所有文档的相似度
            scores = []
            for reac_text, doc_vec in corpus_embeddings.items():
                sim = cosine_sim(query_vec, doc_vec)
                scores.append((reac_text, sim))

            # 按相似度排序，取 Top-K
            scores.sort(key=lambda x: x[1], reverse=True)
            top_preds = [reac for reac, _ in scores[:TOPK_PRED]]

            # 标准化预测结果
            pred_norm = [p.lower().replace("reaction: ", "").strip() for p in top_preds]

            # 计算指标
            hit = 1.0 if any(p in gold_set for p in pred_norm) else 0.0
            inter = gold_set.intersection(set(pred_norm))
            recall = len(inter) / len(gold_set) if gold_set else 0.0

            # MRR: 第一个命中的位置倒数
            rank_pos = None
            for pos, p in enumerate(pred_norm, start=1):
                if p in gold_set:
                    rank_pos = pos
                    break
            mrr = 1.0 / rank_pos if rank_pos is not None else 0.0

            sum_hit += hit
            sum_recall += recall
            sum_mrr += mrr
            total += 1

            # 记录详细信息（可选，用于分析）
            details.append({
                "question": q,
                "drugname": sample.get("drugname", ""),
                "gold_count": len(gold_set),
                "hit": hit,
                "recall": recall,
                "mrr": mrr,
                "top1_pred": pred_norm[0] if pred_norm else None,
                "top1_sim": scores[0][1] if scores else 0.0,
            })

        except Exception as e:
            print(f"[error] 样本 {idx} 处理失败: {e}")
            continue

        if idx % 50 == 0:
            print(f"[进度] {idx}/{len(samples)} samples processed")

    elapsed = time.time() - start_time

    if total == 0:
        print(f"[warn] {model_name} 无有效样本")
        return {
            "model_name": model_name,
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
            "total_samples": 0,
            "elapsed_seconds": elapsed,
        }

    avg_hit = sum_hit / total
    avg_recall = sum_recall / total
    avg_mrr = sum_mrr / total

    metrics = {
        "samples": total,
        f"Hit@{TOPK_PRED}": avg_hit,
        f"Recall@{TOPK_PRED}": avg_recall,
        f"MRR@{TOPK_PRED}": avg_mrr,
    }

    result = {
        "model_name": model_name,
        "model_path": OLLAMA_MODEL if "ollama" in model_name.lower() else BIENCODER_MODEL_PATH,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "total_samples": total,
        "elapsed_seconds": elapsed,
        "details": details[:100],  # 只保存前100条详情，避免文件过大
    }

    print(f"\n[评估完成] {model_name}")
    print(f"  样本数: {total}")
    print(f"  Hit@{TOPK_PRED}: {avg_hit:.4f}")
    print(f"  Recall@{TOPK_PRED}: {avg_recall:.4f}")
    print(f"  MRR@{TOPK_PRED}: {avg_mrr:.4f}")
    print(f"  耗时: {elapsed:.2f} 秒")

    return result


def load_eval_samples() -> List[Dict[str, Any]]:
    """加载评估样本"""
    samples = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("task_type") == "Drug2Reaction":
                samples.append(obj)
    return samples


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("双塔模型对比评估")
    print("=" * 60)
    print(f"评估文件: {EVAL_FILE}")
    print(f"基线模型: Ollama {OLLAMA_MODEL}")
    print(f"新模型路径: {BIENCODER_MODEL_PATH}")
    print("=" * 60)

    # 1. 加载评估样本
    print("\n[步骤 1] 加载评估样本...")
    samples = load_eval_samples()
    print(f"[info] 共加载 {len(samples)} 条评估样本")

    # 2. 构建文档库
    print("\n[步骤 2] 构建 Reaction 文档库...")
    corpus = build_reaction_corpus()
    print(f"[info] 文档库大小: {len(corpus)} 个 Reaction")

    # 3. 加载新训练的模型
    print("\n[步骤 3] 加载新训练的双塔模型...")
    try:
        biencoder_model = SentenceTransformer(BIENCODER_MODEL_PATH)
        print(f"[info] 模型加载成功: {BIENCODER_MODEL_PATH}")
    except Exception as e:
        print(f"[error] 模型加载失败: {e}")
        print(f"[error] 请检查模型路径是否正确: {BIENCODER_MODEL_PATH}")
        return

    # 4. 为文档库生成 embedding（两个模型各一份）
    print("\n[步骤 4] 为文档库生成向量（基线模型）...")
    corpus_embeddings_baseline = {}
    for idx, (reac, text) in enumerate(corpus.items(), 1):
        try:
            vec = embed_ollama(text)
            corpus_embeddings_baseline[reac] = vec
        except Exception as e:
            print(f"[warn] 文档 '{reac}' 向量化失败: {e}")
        if idx % 50 == 0:
            print(f"  [进度] {idx}/{len(corpus)} documents embedded")
    print(f"[info] 基线模型文档向量完成: {len(corpus_embeddings_baseline)} 个")

    print("\n[步骤 5] 为文档库生成向量（新模型）...")
    corpus_embeddings_biencoder = {}
    doc_texts = list(corpus.values())
    try:
        # 批量编码更高效
        doc_vectors = biencoder_model.encode(doc_texts, normalize_embeddings=True, show_progress_bar=True)
        for (reac, _), vec in zip(corpus.items(), doc_vectors):
            corpus_embeddings_biencoder[reac] = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
        print(f"[info] 新模型文档向量完成: {len(corpus_embeddings_biencoder)} 个")
    except Exception as e:
        print(f"[error] 批量编码失败: {e}，改用逐个编码...")
        for idx, (reac, text) in enumerate(corpus.items(), 1):
            try:
                vec = embed_biencoder(text, biencoder_model)
                corpus_embeddings_biencoder[reac] = vec
            except Exception as e2:
                print(f"[warn] 文档 '{reac}' 向量化失败: {e2}")
            if idx % 50 == 0:
                print(f"  [进度] {idx}/{len(corpus)} documents embedded")

    # 5. 评估基线模型
    print("\n" + "=" * 60)
    baseline_result = evaluate_one_model(
        model_name="Baseline (Ollama nomic-embed-text)",
        embed_fn=embed_ollama,
        corpus_embeddings=corpus_embeddings_baseline,
        samples=samples,
    )
    with open(BASELINE_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(baseline_result, f, ensure_ascii=False, indent=2)
    print(f"[saved] 基线结果已保存: {BASELINE_RESULT_FILE}")

    # 6. 评估新模型
    print("\n" + "=" * 60)
    biencoder_result = evaluate_one_model(
        model_name="Fine-tuned Biencoder (bge-m3)",
        embed_fn=lambda q: embed_biencoder(q, biencoder_model),
        corpus_embeddings=corpus_embeddings_biencoder,
        samples=samples,
    )
    with open(NEW_MODEL_RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(biencoder_result, f, ensure_ascii=False, indent=2)
    print(f"[saved] 新模型结果已保存: {NEW_MODEL_RESULT_FILE}")

    # 7. 生成对比报告
    print("\n" + "=" * 60)
    print("对比报告")
    print("=" * 60)

    baseline_metrics = baseline_result.get("metrics", {})
    biencoder_metrics = biencoder_result.get("metrics", {})

    comparison = {
        "evaluation_config": {
            "eval_file": EVAL_FILE,
            "topk": TOPK_PRED,
            "timestamp": datetime.now().isoformat(),
        },
        "baseline": {
            "model_name": baseline_result["model_name"],
            "metrics": baseline_metrics,
        },
        "fine_tuned_model": {
            "model_name": biencoder_result["model_name"],
            "model_path": BIENCODER_MODEL_PATH,
            "metrics": biencoder_metrics,
        },
        "improvement": {
            "hit_improvement": biencoder_metrics.get(f"Hit@{TOPK_PRED}", 0.0) - baseline_metrics.get(f"Hit@{TOPK_PRED}", 0.0),
            "recall_improvement": biencoder_metrics.get(f"Recall@{TOPK_PRED}", 0.0) - baseline_metrics.get(f"Recall@{TOPK_PRED}", 0.0),
            "mrr_improvement": biencoder_metrics.get(f"MRR@{TOPK_PRED}", 0.0) - baseline_metrics.get(f"MRR@{TOPK_PRED}", 0.0),
            "hit_improvement_pct": (
                (biencoder_metrics.get(f"Hit@{TOPK_PRED}", 0.0) - baseline_metrics.get(f"Hit@{TOPK_PRED}", 0.0))
                / baseline_metrics.get(f"Hit@{TOPK_PRED}", 0.001) * 100
            ),
            "recall_improvement_pct": (
                (biencoder_metrics.get(f"Recall@{TOPK_PRED}", 0.0) - baseline_metrics.get(f"Recall@{TOPK_PRED}", 0.0))
                / baseline_metrics.get(f"Recall@{TOPK_PRED}", 0.001) * 100
            ),
            "mrr_improvement_pct": (
                (biencoder_metrics.get(f"MRR@{TOPK_PRED}", 0.0) - baseline_metrics.get(f"MRR@{TOPK_PRED}", 0.0))
                / baseline_metrics.get(f"MRR@{TOPK_PRED}", 0.001) * 100
            ),
        },
    }

    # 打印对比表格
    print("\n指标对比:")
    print(f"{'指标':<20} {'基线模型':<25} {'新模型':<25} {'提升':<15}")
    print("-" * 85)
    print(
        f"{'Hit@10':<20} "
        f"{baseline_metrics.get(f'Hit@{TOPK_PRED}', 0.0):<25.4f} "
        f"{biencoder_metrics.get(f'Hit@{TOPK_PRED}', 0.0):<25.4f} "
        f"{comparison['improvement']['hit_improvement']:+.4f} ({comparison['improvement']['hit_improvement_pct']:+.1f}%)"
    )
    print(
        f"{'Recall@10':<20} "
        f"{baseline_metrics.get(f'Recall@{TOPK_PRED}', 0.0):<25.4f} "
        f"{biencoder_metrics.get(f'Recall@{TOPK_PRED}', 0.0):<25.4f} "
        f"{comparison['improvement']['recall_improvement']:+.4f} ({comparison['improvement']['recall_improvement_pct']:+.1f}%)"
    )
    print(
        f"{'MRR@10':<20} "
        f"{baseline_metrics.get(f'MRR@{TOPK_PRED}', 0.0):<25.4f} "
        f"{biencoder_metrics.get(f'MRR@{TOPK_PRED}', 0.0):<25.4f} "
        f"{comparison['improvement']['mrr_improvement']:+.4f} ({comparison['improvement']['mrr_improvement_pct']:+.1f}%)"
    )
    print("-" * 85)

    # 保存对比报告
    with open(COMPARISON_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] 对比报告已保存: {COMPARISON_REPORT_FILE}")

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
