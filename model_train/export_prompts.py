# encoding:utf8
import os
import re
import json
import math
import random
import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
from py2neo import Graph

# 配置
OUT_DIR = "outputs"
SAMPLE_N = 10000
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")


# ---------- 简化复用的辅助函数（来源于 export_prompts.py） ----------
def find_8digit_date(s):
    if not s:
        return None
    m = re.search(r'(\d{8})', str(s))
    return m.group(1) if m else None


def parse_event_dt(raw):
    d = find_8digit_date(raw)
    if not d:
        return None
    try:
        return datetime.datetime.strptime(d, "%Y%m%d").date().isoformat()
    except Exception:
        return None


def to_float_safe(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "unknown", "uk", "na", "n/a", "null"):
        return None
    s = s.replace("≤", "<=").replace("—", "-").replace("–", "-")
    s = re.sub(r'[^\d\.\-<>/]', ' ', s)
    m = re.search(r'(-?\d+(\.\d+)?)', s)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None


UNIT_MAP = {
    "g": 1000.0,
    "mg": 1.0,
    "mcg": 0.001,
    "μg": 0.001,
    "ug": 0.001,
    "kg": 1000.0,
    "lb": 0.45359237 * 1000.0,
}


def normalize_dose(amount_raw, unit_raw):
    amt = to_float_safe(amount_raw)
    unit = (unit_raw or "").strip().lower()
    if not amt or not unit:
        return {"amt": None, "unit": unit or "unknown", "amt_mg": None}
    if unit in UNIT_MAP:
        factor = UNIT_MAP[unit]
        amt_mg = amt * factor
        return {"amt": amt, "unit": unit, "amt_mg": amt_mg}
    return {"amt": amt, "unit": unit, "amt_mg": None}


REACT_CLEAN_RE = re.compile(r'[^\w\s]', flags=re.UNICODE)


def normalize_reaction(s: str) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    s = REACT_CLEAN_RE.sub(' ', s)
    s = s.replace('_', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def safe_text(x):
    if x is None:
        return "unknown"
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "na", "none"):
        return "unknown"
    return s


def meaningful_indi(s: str) -> bool:
    if not s:
        return False
    t = s.strip().lower()
    bad_patterns = [
        "unknown", "product used for unknown indication", "therapeutic use unknown",
        "not applicable", "indication unknown"
    ]
    for p in bad_patterns:
        if p in t:
            return False
    return True


# ---------- 从 Neo4j 拉取 case-level 聚合数据 ----------
def get_graph():
    return Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def fetch_cases_from_kg(limit=None):
    """
    从 KG 拉取每个 DrugSet 的聚合记录：
    返回列表，每项为 dict:
      { primaryid, caseid, age, age_grp, sex, wt, event_dt, occr_country,
        drugs: [ {drugname, prod_ai, dose_amt, dose_unit, dose_freq}, ... ],
        indis: [..], reactions: [..] }
    """
    g = get_graph()
    q = """
    MATCH (ds:DrugSet)
    OPTIONAL MATCH (p:Patient {primaryid: ds.primaryid, caseid: ds.caseid})
    OPTIONAL MATCH (ds)-[:CONTAINS_DRUG]->(d:Drug)
    OPTIONAL MATCH (ds)-[:TREATS_FOR]->(i:Indication)
    OPTIONAL MATCH (ds)-[:CAUSES_REACTION]->(r:Reaction)
    RETURN ds.primaryid AS primaryid, ds.caseid AS caseid,
           p.age AS age, p.age_grp AS age_grp, p.sex AS sex, p.wt AS wt, p.event_dt AS event_dt, p.occr_country AS occr_country,
           [dd IN collect(DISTINCT d) | {drugname: dd.drugname, prod_ai: dd.prod_ai, dose_amt: dd.dose_amt, dose_unit: dd.dose_unit, dose_freq: dd.dose_freq}] AS drugs,
           [x IN collect(DISTINCT i.indi) WHERE x IS NOT NULL] AS indis,
           [x IN collect(DISTINCT r.reac) WHERE x IS NOT NULL] AS reactions
    """
    if isinstance(limit, int) and limit > 0:
        q += " LIMIT $limit"
        rows = list(g.run(q, limit=limit))
    else:
        rows = list(g.run(q))
    cases = []
    for row in tqdm(rows, desc="从 KG 构建 case 列表"):
        case = {
            "primaryid": row["primaryid"],
            "caseid": row["caseid"],
            "age": row.get("age"),
            "age_grp": row.get("age_grp"),
            "sex": row.get("sex"),
            "wt": row.get("wt"),
            "event_dt": row.get("event_dt"),
            "occr_country": row.get("occr_country"),
            "drugs": row.get("drugs") or [],
            "indis": row.get("indis") or [],
            "reacs_raw": row.get("reactions") or [],
            "reacs_norm": [normalize_reaction(r) for r in (row.get("reactions") or []) if r and str(r).strip()]
        }
        cases.append(case)
    return cases


# ---------- 基于 KG 构建 prompt 与 labels（沿用之前逻辑） ----------
def validate_and_clean_numeric(val, field_name, min_val=None, max_val=None):
    """验证并清理数值字段，返回清理后的值或 None"""
    if val is None:
        return None
    
    # 尝试转换为数值
    num_val = to_float_safe(val)
    if num_val is None:
        return None
    
    # 检查合理性范围
    if field_name == "age":
        if num_val < 0 or num_val > 120:  # 异常年龄
            return None
    elif field_name == "wt":
        if num_val < 0.5 or num_val > 500:  # 异常体重（kg）
            return None
    elif field_name.startswith("dose"):
        if num_val < 0:  # 负剂量
            return None
    
    # 自定义范围检查
    if min_val is not None and num_val < min_val:
        return None
    if max_val is not None and num_val > max_val:
        return None
        
    return num_val


def process_temporal_features(event_dt_raw):
    """处理时间特征，返回详细和粗粒度两种版本"""
    if not event_dt_raw:
        return {"season": "unknown", "season_month": "unknown"}
    
    try:
        # 解析日期（处理多种格式）
        dt = None
        event_str = str(event_dt_raw).strip()
        
        if len(event_str) == 8 and event_str.isdigit():
            dt = datetime.datetime.strptime(event_str, "%Y%m%d")
        elif '-' in event_str:
            dt = datetime.datetime.fromisoformat(event_str.split('T')[0])
        
        if dt is None:
            return {"season": "unknown", "season_month": "unknown"}
        
        # 季节映射
        season_map = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring", 
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"
        }
        
        season = season_map.get(dt.month, "unknown")
        season_month = f"{season}_{dt.month}" if season != "unknown" else "unknown"
        
        return {"season": season, "season_month": season_month}
        
    except Exception:
        return {"season": "unknown", "season_month": "unknown"}


# 药物频率标准化映射表（完整版）
DOSE_FREQ_MAP_DETAILED = {
    # 每日给药
    "QD": "once_daily",           # 每日一次
    "QOD": "every_other_day",     # 隔日一次
    "BID": "twice_daily",         # 每日两次
    "TID": "three_times_daily",   # 每日三次
    "QID": "four_times_daily",    # 每日四次
    
    # 按小时给药
    "Q2H": "every_2_hours",       # 每2小时
    "Q3H": "every_3_hours",       # 每3小时
    "Q4H": "every_4_hours",       # 每4小时
    "Q6H": "every_6_hours",       # 每6小时
    "Q8H": "every_8_hours",       # 每8小时
    "Q12H": "every_12_hours",     # 每12小时
    
    # 按周给药
    "/WK": "weekly",              # 每周
    "QW": "weekly",               # 每周（同义）
    "QOW": "every_other_week",    # 隔周
    "Q3W": "every_3_weeks",       # 每3周
    "BIW": "twice_weekly",        # 每周两次
    "TIW": "three_times_weekly",  # 每周三次
    
    # 按月/年/周期给药
    "/MONTH": "monthly",          # 每月
    "QM": "monthly",              # 每月（同义）
    "/YR": "yearly",              # 每年
    "/CYCLE": "per_cycle",        # 每周期
    "TRIMESTER": "per_trimester", # 每孕期
    
    # 按时间点给药
    "/HR": "per_hour",            # 每小时
    "/MIN": "per_minute",         # 每分钟
    "HS": "at_bedtime",           # 睡前
    "QAM": "every_morning",       # 每晨
    
    # 按需给药
    "PRN": "as_needed",           # 按需
    
    # 数量相关
    "1X": "once",                 # 一次
    
    # 未定义/特殊
    "UD": "undefined_schedule",   # 未定义给药方案
    "999": "unknown",             # 特殊编码，表示未知
    
    # 常见变体处理
    "DAILY": "once_daily",
    "WEEKLY": "weekly",
    "MONTHLY": "monthly",
    "YEARLY": "yearly",
    
    # 空值/未知处理
    "": "unknown",
    "UNK": "unknown",
    "UNKNOWN": "unknown",
}

# 粗粒度频率映射（新增）
DOSE_FREQ_MAP_COARSE = {
    # 每日相关 -> daily
    "QD": "daily", "QOD": "daily", "BID": "daily", "TID": "daily", "QID": "daily",
    "DAILY": "daily",
    
    # 小时相关 -> hourly  
    "Q2H": "hourly", "Q3H": "hourly", "Q4H": "hourly", "Q6H": "hourly", 
    "Q8H": "hourly", "Q12H": "hourly", "/HR": "hourly", "/MIN": "hourly",
    
    # 周相关 -> weekly
    "/WK": "weekly", "QW": "weekly", "QOW": "weekly", "Q3W": "weekly",
    "BIW": "weekly", "TIW": "weekly", "WEEKLY": "weekly",
    
    # 月/年相关 -> monthly
    "/MONTH": "monthly", "QM": "monthly", "/YR": "monthly", 
    "TRIMESTER": "monthly", "MONTHLY": "monthly", "YEARLY": "monthly",
    "/CYCLE": "monthly",
    
    # 特殊/按需 -> unknown
    "PRN": "unknown", "HS": "unknown", "QAM": "unknown", "1X": "unknown",
    "UD": "unknown", "999": "unknown", "": "unknown", "UNK": "unknown", "UNKNOWN": "unknown",
}


def normalize_dose_frequency(freq_raw, granularity="detailed"):
    """
    标准化药物给药频率
    Args:
        freq_raw: 原始频率字符串
        granularity: "coarse" 或 "detailed"
    Returns:
        标准化后的频率字符串
    """
    if not freq_raw:
        return "unknown"
    
    freq_str = str(freq_raw).strip().upper()
    
    # 选择映射表
    freq_map = DOSE_FREQ_MAP_COARSE if granularity == "coarse" else DOSE_FREQ_MAP_DETAILED
    
    # 直接映射
    if freq_str in freq_map:
        return freq_map[freq_str]
    
    # 处理一些常见的变体
    freq_clean = re.sub(r'[^\w]', '', freq_str)
    if freq_clean in freq_map:
        return freq_map[freq_clean]
    
    # 处理数字+单位的情况
    if granularity == "coarse":
        # 粗粒度：按模式分类
        if re.search(r'(?:DAY|DAILY)', freq_clean):
            return "daily"
        elif re.search(r'(?:HOUR|HR)', freq_clean):
            return "hourly"
        elif re.search(r'(?:WEEK|WK)', freq_clean):
            return "weekly"
        elif re.search(r'(?:MONTH|YEAR|YR)', freq_clean):
            return "monthly"
        else:
            return "unknown"
    else:
        # 细粒度：详细处理（复用之前的逻辑）
        if re.match(r'^\d+X?/?(?:DAY|DAILY)$', freq_clean):
            num_match = re.search(r'^(\d+)', freq_clean)
            if num_match:
                num = int(num_match.group(1))
                if num == 1:
                    return "once_daily"
                elif num == 2:
                    return "twice_daily"
                elif num == 3:
                    return "three_times_daily"
                elif num == 4:
                    return "four_times_daily"
                else:
                    return f"{num}_times_daily"
        
        # 处理 "EVERY X HOURS" 格式
        hour_match = re.search(r'(?:EVERY|Q)(\d+)H(?:OUR)?S?', freq_clean)
        if hour_match:
            hours = hour_match.group(1)
            return f"every_{hours}_hours"
        
        # 如果都匹配不上，保留原始但做基本清理
        return freq_raw.lower().replace(' ', '_') if len(freq_raw) <= 20 else "unknown"


def build_natural_prompt_parts(case, temporal_granularity="coarse"):
    """构建自然语言格式的患者信息"""
    # 年龄处理
    age = case.get("age")
    age_val = validate_and_clean_numeric(age, "age")
    
    if age_val is None and case.get("age_grp"):
        m = re.search(r'(\d{1,3})\D+(\d{1,3})', case.get("age_grp", ""))
        if m:
            age_val = (float(m.group(1)) + float(m.group(2))) / 2.0
            age_val = validate_and_clean_numeric(age_val, "age")
    
    age_text = f"{int(age_val)}-year-old" if age_val is not None else "unknown age"
    
    # 性别处理
    sex = safe_text(case.get("sex"))
    if sex.upper() in ['M', 'MALE', '1']:
        sex_text = "male"
    elif sex.upper() in ['F', 'FEMALE', '2']:
        sex_text = "female"
    else:
        sex_text = "unknown gender"
    
    # 国家处理
    country = safe_text(case.get("occr_country"))
    country_text = f"from {country}" if country != "unknown" else ""
    
    # 体重处理
    wt_val = validate_and_clean_numeric(case.get("wt"), "wt")
    wt_text = f"weight {wt_val}kg" if wt_val is not None else ""
    
    # 时间处理
    temporal_features = process_temporal_features(case.get("event_dt"))
    
    if temporal_granularity == "coarse":
        season = temporal_features.get("season", "unknown")
        time_text = f"recorded in {season}" if season != "unknown" else ""
    else:
        season_month = temporal_features.get("season_month", "unknown")
        if "_" in season_month and season_month != "unknown":
            season, month = season_month.split("_")
            month_names = {
                "1": "January", "2": "February", "3": "March", "4": "April",
                "5": "May", "6": "June", "7": "July", "8": "August", 
                "9": "September", "10": "October", "11": "November", "12": "December"
            }
            month_text = month_names.get(month, f"month {month}")
            time_text = f"recorded in {month_text} ({season})"
        else:
            time_text = ""
    
    # 组合有效部分
    parts = []
    if age_text != "unknown age":
        parts.append(age_text)
    if sex_text != "unknown gender":
        parts.append(sex_text)
    if country_text:
        parts.append(country_text)
    if wt_text:
        parts.append(wt_text)
    if time_text:
        parts.append(time_text)
    
    return ", ".join(parts) if parts else "unknown patient information"


def build_prompt_and_labels(case, drugs, indis, reactions_normed, top_k_labels=None, max_drugs=8, 
                          temporal_granularity="coarse", freq_granularity="coarse", format_version="v2"):
    """
    构建 prompt 和 labels
    Args:
        format_version: "v1" (结构化) 或 "v2" (增强关系) 或 "v3" (混合自然语言)
    """
    
    # 适应症处理
    meaningful = [x for x in (indis or []) if meaningful_indi(x)]
    
    # 药物处理
    drug_texts = []
    drugs = drugs or []
    seen = set()
    deduped = []
    for d in drugs:
        key = ((d.get("prod_ai") or d.get("drugname") or "")).strip().lower()
        if key in seen or not key:
            continue
        seen.add(key)
        deduped.append(d)
    
    # **三种格式的处理**
    if format_version == "v3":
        # ========= v3: 混合自然语言格式 =========
        # 患者信息 - 自然语言
        pat_natural = build_natural_prompt_parts(case, temporal_granularity)
        pat_block = f"[PAT] {pat_natural} [/PAT]"
        
        # 适应症 - 保持标签格式
        if meaningful:
            indi_text = "[INDI] " + "; ".join(sorted(set(meaningful))) + " [/INDI]"
        else:
            indi_text = "[INDI] unknown [/INDI]"
        
        # 药物 - 自然语言格式但保持结构
        drug_natural_texts = []
        for d in deduped[:max_drugs]:
            ai = safe_text(d.get("prod_ai") or d.get("drugname"))
            drugname = safe_text(d.get("drugname"))
            
            # 剂量处理
            dose_amt_val = validate_and_clean_numeric(d.get("dose_amt"), "dose_amt")
            dose_info = normalize_dose(dose_amt_val, d.get("dose_unit"))
            
            if dose_info['amt'] is not None:
                dose_part = f"{dose_info['amt']}{dose_info['unit']}"
            else:
                dose_part = "unknown dose"
            
            # 频率处理
            freq_normalized = normalize_dose_frequency(d.get("dose_freq"), granularity=freq_granularity)
            
            # 自然语言药物描述
            if ai != "unknown" and ai != drugname:
                drug_desc = f"{drugname} {dose_part} {freq_normalized} (active ingredient: {ai})"
            else:
                drug_desc = f"{drugname} {dose_part} {freq_normalized}"
            
            drug_natural_texts.append(drug_desc)
        
        if drug_natural_texts and meaningful:
            drugs_block = "[DRUGS_FOR_INDI]\n" + "\n".join(drug_natural_texts) + "\n[/DRUGS_FOR_INDI]"
        elif drug_natural_texts:
            drugs_block = "[DRUGS]\n" + "\n".join(drug_natural_texts) + "\n[/DRUGS]"
        else:
            drugs_block = "[DRUGS] none [/DRUGS]"
        
        prompt = "\n".join([pat_block, indi_text, drugs_block])
        
    elif format_version == "v2":
        # ========= v2: 增强关系格式（原有逻辑） =========
        # 患者信息 - 结构化
        age = case.get("age")
        age_val = validate_and_clean_numeric(age, "age")
        
        if age_val is None and case.get("age_grp"):
            m = re.search(r'(\d{1,3})\D+(\d{1,3})', case.get("age_grp", ""))
            if m:
                age_val = (float(m.group(1)) + float(m.group(2))) / 2.0
                age_val = validate_and_clean_numeric(age_val, "age")
        
        age_text = str(int(age_val)) if age_val is not None else "unknown"
        
        sex = safe_text(case.get("sex"))
        if sex.upper() in ['M', 'MALE', '1']:
            sex = 'M'
        elif sex.upper() in ['F', 'FEMALE', '2']:
            sex = 'F'
        else:
            sex = 'unknown'
        
        wt_val = validate_and_clean_numeric(case.get("wt"), "wt")
        wt_text = f"{wt_val}kg" if wt_val is not None else "unknown"
        
        country = safe_text(case.get("occr_country"))
        
        temporal_features = process_temporal_features(case.get("event_dt"))
        if temporal_granularity == "coarse":
            temporal_field = f"season: {temporal_features['season']}"
        else:
            temporal_field = f"season_month: {temporal_features['season_month']}"
        
        pat_block = f"[PAT] age: {age_text} sex: {sex} wt: {wt_text} country: {country} {temporal_field} [/PAT]"
        
        # 药物 - 结构化格式
        for d in deduped[:max_drugs]:
            ai = safe_text(d.get("prod_ai") or d.get("drugname"))
            drugname = safe_text(d.get("drugname"))
            dose_amt_val = validate_and_clean_numeric(d.get("dose_amt"), "dose_amt")
            dose_info = normalize_dose(dose_amt_val, d.get("dose_unit"))
            
            if dose_info['amt'] is not None:
                dose_repr = f"{dose_info['amt']} {dose_info['unit']}"
            else:
                dose_repr = "unknown"
            
            freq_normalized = normalize_dose_frequency(d.get("dose_freq"), granularity=freq_granularity)
            drug_texts.append(f"[DRUG] {drugname} | ai: {ai} | dose: {dose_repr} | freq: {freq_normalized} [/DRUG]")

        # 增强关系格式
        if meaningful:
            indi_text = "[INDI] " + "; ".join(sorted(set(meaningful))) + " [/INDI]"
            if drug_texts:
                drugs_block = "[DRUGS_FOR_INDI]\n" + "\n".join(drug_texts) + "\n[/DRUGS_FOR_INDI]"
            else:
                drugs_block = "[DRUGS_FOR_INDI]\n[DRUG] none [/DRUG]\n[/DRUGS_FOR_INDI]"
        else:
            indi_text = "[INDI] unknown [/INDI]"
            if drug_texts:
                drugs_block = "\n".join(drug_texts)
            else:
                drugs_block = "[DRUG] none [/DRUG]"
        
        prompt = "\n".join([pat_block, indi_text, drugs_block])
        
    else:
        # ========= v1: 原始结构化格式 =========
        # 患者信息 - 结构化
        age = case.get("age")
        age_val = validate_and_clean_numeric(age, "age")
        
        if age_val is None and case.get("age_grp"):
            m = re.search(r'(\d{1,3})\D+(\d{1,3})', case.get("age_grp", ""))
            if m:
                age_val = (float(m.group(1)) + float(m.group(2))) / 2.0
                age_val = validate_and_clean_numeric(age_val, "age")
        
        age_text = str(int(age_val)) if age_val is not None else "unknown"
        
        sex = safe_text(case.get("sex"))
        if sex.upper() in ['M', 'MALE', '1']:
            sex = 'M'
        elif sex.upper() in ['F', 'FEMALE', '2']:
            sex = 'F'
        else:
            sex = 'unknown'
        
        wt_val = validate_and_clean_numeric(case.get("wt"), "wt")
        wt_text = f"{wt_val}kg" if wt_val is not None else "unknown"
        
        country = safe_text(case.get("occr_country"))
        
        temporal_features = process_temporal_features(case.get("event_dt"))
        if temporal_granularity == "coarse":
            temporal_field = f"season: {temporal_features['season']}"
        else:
            temporal_field = f"season_month: {temporal_features['season_month']}"
        
        pat_block = f"[PAT] age: {age_text} sex: {sex} wt: {wt_text} country: {country} {temporal_field} [/PAT]"
        
        # 适应症 - 平铺
        if meaningful:
            indi_text = "[INDI] " + "; ".join(sorted(set(meaningful))) + " [/INDI]"
        else:
            indi_text = "[INDI] unknown [/INDI]"
        
        # 药物 - 平铺
        for d in deduped[:max_drugs]:
            ai = safe_text(d.get("prod_ai") or d.get("drugname"))
            drugname = safe_text(d.get("drugname"))
            dose_amt_val = validate_and_clean_numeric(d.get("dose_amt"), "dose_amt")
            dose_info = normalize_dose(dose_amt_val, d.get("dose_unit"))
            
            if dose_info['amt'] is not None:
                dose_repr = f"{dose_info['amt']} {dose_info['unit']}"
            else:
                dose_repr = "unknown"
            
            freq_normalized = normalize_dose_frequency(d.get("dose_freq"), granularity=freq_granularity)
            drug_texts.append(f"[DRUG] {drugname} | ai: {ai} | dose: {dose_repr} | freq: {freq_normalized} [/DRUG]")
        
        if not drug_texts:
            drugs_block = "[DRUG] none [/DRUG]"
        else:
            drugs_block = "\n".join(drug_texts)
        
        prompt = "\n".join([pat_block, indi_text, drugs_block])
    
    # 标签处理
    labels = sorted(set([r for r in reactions_normed if r]))
    if top_k_labels:
        labels = [r for r in labels if r in top_k_labels]
    
    return prompt, labels


def export_from_kg(out_dir=OUT_DIR, sample_n=SAMPLE_N, kg_limit=None, format_version="v2", generate_vocab=False):
    """
    Args:
        format_version: "v1" (结构化) 或 "v2" (增强关系) 或 "v3" (混合自然语言)
        generate_vocab: 是否生成词汇表文件
    """
    os.makedirs(out_dir, exist_ok=True)
    print("从 Neo4j KG 拉取 case 数据(聚合 DrugSet)...")
    cases = fetch_cases_from_kg(limit=kg_limit)
    print(f"拉取到 case 数: {len(cases)}")

    # 数据质量过滤
    print("过滤数据质量问题...")
    valid_cases = []
    for c in cases:
        if not c.get('primaryid') or not c.get('caseid'):
            continue
        valid_cases.append(c)
    
    print(f"过滤后有效 case 数: {len(valid_cases)}")
    cases = valid_cases

    # build reaction vocab
    all_reacs = Counter()
    for c in cases:
        all_reacs.update(c['reacs_norm'])
    
    # **新增: 过滤标签数量在1-20之间的样本**
    print("过滤标签数量在1-20之间的样本...")
    cases_filtered = []
    for c in cases:
        label_count = len(c['reacs_norm'])
        if 1 <= label_count <= 15:
            cases_filtered.append(c)
    
    print(f"标签数量过滤后 case 数: {len(cases_filtered)} (原: {len(cases)})")
    cases = cases_filtered
    
    # sample prioritize cases with reaction
    cases_with_reac = [c for c in cases if c['reacs_norm']]
    other_cases = [c for c in cases if not c['reacs_norm']]
    random.seed(42)
    sample = []
    n_with = min(len(cases_with_reac), int(sample_n * 0.9))
    if n_with > 0:
        sample.extend(random.sample(cases_with_reac, n_with))
    remaining = sample_n - len(sample)
    if remaining > 0:
        pool = [c for c in cases if c not in sample]
        if len(pool) >= remaining:
            sample.extend(random.sample(pool, remaining))
        else:
            sample.extend(pool)

    print(f"最终采样数: {len(sample)}(含 reaction 的样本: {sum(1 for c in sample if c['reacs_norm'])})")
    print(f"使用格式版本: {format_version}")
    
    # **新增: 打印标签分布统计**
    label_counts = [len(c['reacs_norm']) for c in sample]
    print(f"标签数量分布 - 最小: {min(label_counts)}, 最大: {max(label_counts)}, 平均: {sum(label_counts)/len(label_counts):.2f}")

    # **4种组合的配置**
    configs = [
        {"temporal": "coarse", "freq": "coarse", "suffix": "coarse_coarse"},
        {"temporal": "coarse", "freq": "detailed", "suffix": "coarse_detailed"},
        {"temporal": "detailed", "freq": "coarse", "suffix": "detailed_coarse"},
        {"temporal": "detailed", "freq": "detailed", "suffix": "detailed_detailed"},
    ]
    
    topk_set = set([r for r, _ in all_reacs.most_common()])
    results = {}
    
    for config in configs:
        print(f"生成 {config['suffix']} 版本...")
        rows = []
        
        for c in tqdm(sample, desc=f"生成 prompt ({config['suffix']})"):
            prompt, labels = build_prompt_and_labels(
                c, c['drugs'], c['indis'], c['reacs_norm'], 
                top_k_labels=topk_set,
                temporal_granularity=config['temporal'],
                freq_granularity=config['freq'],
                format_version=format_version
            )
            
            rows.append({
                "primaryid": c['primaryid'],
                "caseid": c['caseid'],
                "prompt": prompt,
                "labels_list": labels,
                "labels_count": len(labels),
            })
        
        # 保存文件(只保留CSV格式)
        df = pd.DataFrame(rows)
        filename_base = f"prompts_sample_{len(df)}_{config['suffix']}_{format_version}"
        out_csv = os.path.join(out_dir, f"{filename_base}.csv")
        
        df.to_csv(out_csv, index=False)
        
        results[config['suffix']] = df
        print(f"已保存: {out_csv}")
    
    # **可选的词汇表生成**
    if generate_vocab:
        vocab_csv = os.path.join(out_dir, f"reaction_vocab_freqs_{format_version}.csv")
        pd.DataFrame(all_reacs.most_common(), columns=["reaction_norm", "freq"]).to_csv(vocab_csv, index=False)
        print(f"已保存词汇表: {vocab_csv}")
    
    return results, list(all_reacs.keys())


if __name__ == "__main__":
    # **格式选择**
    print("选择输出格式:")
    print("1. v1 - 结构化平铺格式")
    print("2. v2 - 增强关系格式 (DRUGS_FOR_INDI)")
    print("3. v3 - 混合自然语言格式")
    choice = input("请输入 1, 2 或 3 (默认 v2): ").strip()
    format_choice = {"1": "v1", "3": "v3"}.get(choice, "v2")
    
    print(f"选择的格式: {format_choice}")
    
    # 生成数据
    results, vocab = export_from_kg(
        out_dir=OUT_DIR, 
        sample_n=SAMPLE_N, 
        kg_limit=None,
        format_version=format_choice,
        generate_vocab=False
    )
    
    print("导出完成,各版本示例:")
    for suffix, df in results.items():
        print(f"\n=== {suffix} ===")
        print(df.head(1)['prompt'].iloc[0])
        print()