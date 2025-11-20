"""
第 5 层：答案生成层

功能：
1. 接收第 4 层（rank_expansions）的输出。
2. 对各意图结果按文本去重聚合（统计出现次数）。
3. Outcome 缩写映射为中文。
4. 调用本地 Ollama 模型（qwen2.5:7b-instruct）生成凝练中文回答；失败则回退枚举。
5. 控制台仅打印最终回答 + 成功标记。

依赖：
- 需本地已 pull 模型：qwen2.5:7b-instruct
  命令：ollama pull qwen2.5:7b-instruct
- 接口：POST http://127.0.0.1:11434/api/chat

测试：
python .\rag_v2\output_layer.py
"""

import os
import json
import contextlib
import requests
from io import StringIO
from typing import Dict, Any, List, Tuple

from rag_v2.input_layer import parse_input
from rag_v2.initial_search_layer import initial_search
from rag_v2.relation_aggregate_layer import aggregate_relations
from rag_v2.ranking_layer import rank_expansions

# ============ 配置 ============
USE_LLM = True
LLM_MODEL = "qwen2.5:7b-instruct"  # 本地 Ollama 模型名
MAX_ITEMS = 10
DEBUG_LLM = True  # True: 打印调用状态/错误；False: 静默

# Outcome 缩写映射
OUTCOME_ALIAS = {
    "DE": "死亡",
    "LT": "危及生命",
    "HO": "住院或延长住院",
    "DS": "致残/功能受限",
    "CA": "先天异常",
    "RI": "需干预以避免永久性伤害",
    "OT": "其他重要医学事件",
}

# ============ 工具函数 ============

def _fmt_intent(intent: str) -> str:
    mapping = {
        "Drug": "药物",
        "Reaction": "不良反应",
        "Indication": "适应症",
        "Outcome": "患者结局",
    }
    return mapping.get(intent, intent)

def _normalize_text(intent: str, text: str) -> Tuple[str, str]:
    """
    返回 (英文标准文本, 中文提示)。仅 Outcome 缩写映射。
    """
    t = (text or "").strip()
    if intent == "Outcome":
        t_up = t.upper()
        if t_up in OUTCOME_ALIAS:
            return t_up, OUTCOME_ALIAS[t_up]
    return t, ""

def _dedup_and_map(items: List[Dict[str, Any]], intent: str, limit: int) -> List[Dict[str, Any]]:
    """
    按 text 聚合去重，统计出现次数，保留最高 score 的 freq。
    输出字段：text_en, text_cn_hint, count, best_score, freq
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in items:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        key_en, cn_hint = _normalize_text(intent, txt)
        score = float(r.get("score") or 0.0)
        freq = r.get("freq")
        if key_en not in buckets:
            buckets[key_en] = {
                "text_en": key_en,
                "text_cn_hint": cn_hint,
                "count": 1,
                "best_score": score,
                "freq": freq,
            }
        else:
            buckets[key_en]["count"] += 1
            if score > buckets[key_en]["best_score"]:
                buckets[key_en]["best_score"] = score
                buckets[key_en]["freq"] = freq
            if cn_hint and not buckets[key_en]["text_cn_hint"]:
                buckets[key_en]["text_cn_hint"] = cn_hint
    merged = list(buckets.values())
    merged.sort(key=lambda x: (x["count"], x["best_score"]), reverse=True)
    return merged[:limit]

def _local_compose_answer(question: str,
                          per_intent: Dict[str, List[Dict[str, Any]]],
                          scoring_note: str) -> str:
    """
    无 LLM 或失败时的回退枚举输出（保留原英文，Outcome 显示中文）。
    """
    parts: List[str] = []
    for intent, rows in per_intent.items():
        title = _fmt_intent(intent)
        lines = [f"{title}{scoring_note}："]
        for i, r in enumerate(rows, 1):
            en = r["text_en"]
            cn = r.get("text_cn_hint")
            cnt = r.get("count", 1)
            label = f"{cn}（{en}）" if cn else en
            if cnt > 1:
                lines.append(f"{i}. {label}（出现{cnt}次）")
            else:
                lines.append(f"{i}. {label}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)

def _ollama_chat(model: str,
                 messages: List[Dict[str, str]],
                 timeout: int = 120) -> Tuple[str, bool, str]:
    """
    直接调用 Ollama /api/chat 接口。
    返回：(内容, 成功标记, 错误信息)
    """
    try:
        resp = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout
        )
        if DEBUG_LLM:
            print(f"[DEBUG_LLM] status={resp.status_code}")
        if resp.status_code != 200:
            err = f"status={resp.status_code} body={resp.text[:300]}"
            if DEBUG_LLM:
                print(f"[DEBUG_LLM] error={err}")
            return "", False, err
        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        if not content.strip():
            err = f"empty content raw={json.dumps(data)[:300]}"
            if DEBUG_LLM:
                print(f"[DEBUG_LLM] error={err}")
            return "", False, err
        return content.strip(), True, ""
    except Exception as e:
        err = repr(e)
        if DEBUG_LLM:
            print(f"[DEBUG_LLM] exception={err}")
        return "", False, err

def _llm_refine(question: str,
                per_intent: Dict[str, List[Dict[str, Any]]],
                scoring_mode_by_intent: Dict[str, str]) -> Tuple[str, bool, str]:
    """
    构造提示词 + 调用本地 LLM，返回 (回答文本, 成功标记, 错误描述)
    """
    lists_text: List[str] = []
    for intent, rows in per_intent.items():
        zh_title = _fmt_intent(intent)
        scoring_note = "仅语义相关排序" if scoring_mode_by_intent.get(intent) == "uniform_sim" else "综合语义与频次"
        lists_text.append(f"{zh_title}（{scoring_note}）：")
        for i, r in enumerate(rows, 1):
            en = r["text_en"]
            cn_hint = r.get("text_cn_hint")
            cnt = r.get("count", 1)
            if cn_hint:
                item = f"{i}. {cn_hint}（{en}）"
            else:
                item = f"{i}. {en}"
            if cnt > 1:
                item += f"（出现{cnt}次）"
            lists_text.append(item)
    context = "\n".join(lists_text)

    prompt = (
        "请用简体中文回答下面的问题，并基于给定要点进行归纳：\n"
        "要求：\n"
        "1. 不编造或扩展未提供的信息；\n"
        "2. 合并相近概念，不机械逐条照搬；\n"
        "3. 药物名可保留英文，若常见可加中文；\n"
        "4. 使用分点或分段，简洁专业；\n"
        "5. 不要原样复制“原始要点”列表；\n"
        "6. 如果要点很杂，可按类别归组。\n\n"
        f"用户问题：{question}\n\n原始要点：\n{context}\n\n请输出优化后的中文回答："
    )

    messages = [
        {"role": "system", "content": "你是医学药物安全知识图谱助手，回答需准确、凝练。"},
        {"role": "user", "content": prompt}
    ]
    return _ollama_chat(LLM_MODEL, messages)

# ============ 核心：答案生成 ============

def generate_answer(ranking_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入：ranking_layer 输出的 dict
    输出：包含最终 answer、meta
    """
    question = ranking_output.get("question", "")
    ranked = ranking_output.get("ranked", {})
    intents = ranking_output.get("intents") or []
    topk = ranking_output.get("topk", 10)

    # 准备供 LLM 使用的数据
    per_intent_for_llm: Dict[str, List[Dict[str, Any]]] = {}
    scoring_mode_by_intent: Dict[str, str] = {}

    for intent in intents:
        block = ranked.get(intent) or {}
        rows = block.get("rows", [])
        stats = block.get("stats", {}) or {}
        scoring_mode = stats.get("scoring_mode", "unknown")
        scoring_mode_by_intent[intent] = scoring_mode
        dedup_rows = _dedup_and_map(rows[:topk], intent, MAX_ITEMS)
        per_intent_for_llm[intent] = dedup_rows

    # 调 LLM 或回退
    if USE_LLM:
        llm_text, ok, err = _llm_refine(question, per_intent_for_llm, scoring_mode_by_intent)
        if not ok:
            first_mode = next(iter(scoring_mode_by_intent.values()), "uniform_sim")
            note = "（仅语义相关排序，LLM失败回退枚举）" if first_mode == "uniform_sim" else "（综合语义与频次，LLM失败回退枚举）"
            llm_text = _local_compose_answer(question, per_intent_for_llm, note)
    else:
        ok = False
        err = "LLM disabled"
        first_mode = next(iter(scoring_mode_by_intent.values()), "uniform_sim")
        note = "（仅语义相关排序，本地枚举）" if first_mode == "uniform_sim" else "（综合语义与频次，本地枚举）"
        llm_text = _local_compose_answer(question, per_intent_for_llm, note)

    output = {
        "question": question,
        "answer": llm_text,
        "meta": {
            "layer": "answer",
            "llm_model": LLM_MODEL,
            "llm_used": USE_LLM,
            "llm_success": ok,
            "llm_error": None if ok else err,
        },
    }

    # 控制台输出
    print("====== 最终回答 ======")
    print(llm_text)
    print(f"[llm_success={ok} model={LLM_MODEL} error={'' if ok else err}]")
    print("======================")
    return output

# ============ 端到端流水线（第 1~5 层） ============

def answer_pipeline(question: str, suppress_previous: bool = True) -> Dict[str, Any]:
    """
    执行从解析到答案生成。
    suppress_previous=True 时不打印前 1~4 层细节。
    """
    if suppress_previous:
        with contextlib.redirect_stdout(StringIO()):
            parsed = parse_input(question)
            searched = initial_search(parsed)
            aggregated = aggregate_relations(searched)
            ranking_output = rank_expansions(aggregated)
    else:
        parsed = parse_input(question)
        searched = initial_search(parsed)
        aggregated = aggregate_relations(searched)
        ranking_output = rank_expansions(aggregated)

    return generate_answer(ranking_output)

# ============ 主程序测试 ============

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
        answer_pipeline(q)