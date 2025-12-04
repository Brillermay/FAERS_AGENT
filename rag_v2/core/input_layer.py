# 第 1 层：输入解析 + 预留向量结构（不访问 Neo4j）
import json
import re
from typing import Dict, List, Any
from openai import OpenAI

# 使用统一的 embedding 工具模块（已移动到 core 子包）
from rag_v2.core.embedding_utils import embed_text, get_current_model_name

# ---------- OpenAI 配置 ----------
client = OpenAI(
    api_key="sk-7cuAMuvifXU3TogzrjZahugbyKSnHUZPmZ68SfzpxRFuJWLn",
    base_url="https://api.chatanywhere.tech/v1",
)

# ---------- LLM 提示：抽取节点候选 + 预期返回类型 ----------
SYSTEM_EXTRACT = (
    "你是药物安全知识图谱的解析器。图谱只有四类节点：Drug、Reaction、Indication、Outcome。\n"
    "请根据用户问题严格输出 JSON：\n"
    "{\n"
    "  \"seeds\": [\n"
    "    {\"type\": \"Drug|Reaction|Indication|Outcome\", \"text\": \"...\"}\n"
    "  ],\n"
    "  \"intents\": [\"Drug|Reaction|Indication|Outcome\"],\n"
    "  \"topk\": 10\n"
    "}\n"
    "说明：\n"
    "- seeds 可以包含多条，允许不同类型同时出现，只要问题中提到了即可。\n"
    "- intents 表示用户想查询/返回的节点类型；可包含多个。\n"
    "- topk 默认为 10，除非用户要求其它数量。\n"
    "- 只输出一个 JSON 对象，不要额外文字。"
)

SYSTEM_TRANSLATE = (
    "你是医学术语翻译助手。把输入的药物/反应/适应症/结果名称翻译成英文常用名，"
    "仅输出译文本身，不要解释。若已是英文则原样返回。"
)

def _extract_json(text: str) -> Dict[str, Any]:
    try:
        match = re.search(r"\{.*\}", text, re.S)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}

def _translate_to_en(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    if all(ord(ch) < 128 for ch in t):
        return t
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_TRANSLATE},
                {"role": "user", "content": t},
            ],
            temperature=0.0,
        )
        translated = resp.choices[0].message.content.strip()
        return translated or t
    except Exception:
        return t

def parse_input(question: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_EXTRACT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception:
        raw = "{}"

    data = _extract_json(raw)

    seeds_raw = data.get("seeds") or []
    intents_raw = data.get("intents") or []
    topk = int(data.get("topk") or 10)

    valid_types = {"Drug", "Reaction", "Indication", "Outcome"}
    seeds: List[Dict[str, str]] = []
    for item in seeds_raw:
        t = item.get("type")
        txt = item.get("text", "")
        if t in valid_types and isinstance(txt, str) and txt.strip():
            seeds.append(
                {
                    "type": t,
                    "text": txt.strip(),
                    "normalized": _translate_to_en(txt.strip()),
                }
            )

    intents = [t for t in intents_raw if t in valid_types]
    if not intents:
        intents = ["Drug", "Reaction", "Indication", "Outcome"]

    try:
        query_vector = embed_text(question)  # 使用统一接口
    except Exception as e:
        print(f"[warn] 问题向量化失败: {e}")
        query_vector = None

    result: Dict[str, Any] = {
        "question": question,
        "seeds": seeds,
        "intents": intents,
        "topk": topk,
        "query_vector": query_vector,
        "meta": {
            "raw_llm_output": raw,
            "parsed_ok": bool(seeds or intents),
            "embedding_model": get_current_model_name(),
            "query_vector_dim": len(query_vector) if query_vector else None,
        },
    }

    # 打印时隐藏向量，仅输出维度
    display_result = dict(result)
    if query_vector is not None:
        display_result["query_vector"] = f"dim={len(query_vector)}"
    else:
        display_result["query_vector"] = None

    print("======== 输入层解析结果 ========")
    print(json.dumps(display_result, ensure_ascii=False, indent=2))
    print("================================")
    return result

if __name__ == "__main__":
    tests = [
        "阿司匹林的常见不良反应有哪些？",
        "NAUSEA 常见于哪些药物？",
        "阿司匹林一般用于哪些适应症？",
        "关于氯吡格雷，有哪些患者结局？",
        "请列出常见的药物。",
    ]
    for q in tests:
        print("\n问题：", q)
        parse_input(q)