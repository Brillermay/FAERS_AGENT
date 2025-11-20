# 第 3 层：关系扩展与聚合（基于初步候选），直接返回节点向量 embedding
# 更新：各 Cypher 频次统计改为 count(distinct ds) 或 count(distinct p)，以产生非均匀 freq
import json
from typing import Dict, Any, List, Set
from py2neo import Graph
from rag_v2.initial_search_layer import initial_search
from rag_v2.input_layer import parse_input

graph = Graph("bolt://localhost:7687", auth=("neo4j", "rainshineking274"))

LABEL_PROP = {
    "Drug": "drugname",
    "Reaction": "reac",
    "Indication": "indi",
    "Outcome": "outccode",
}

def _candidate_ids(candidates: Dict[str, List[Dict[str, Any]]], label: str) -> List[int]:
    rows = candidates.get(label) or []
    return [int(r["id"]) for r in rows]

def _all_ids_exact(label: str, value: str) -> List[int]:
    if not value:
        return []
    prop = LABEL_PROP[label]
    q = f"""
    MATCH (n:{label})
    WHERE toLower(n.{prop}) = toLower($val)
    RETURN id(n) AS id
    """
    return [int(r["id"]) for r in graph.run(q, val=value).data()]

def _distinct_drugset_count_by_drug_ids(drug_ids: List[int]) -> int:
    if not drug_ids:
        return 0
    q = """
    UNWIND $ids AS did
    MATCH (ds:DrugSet)-[:CONTAINS_DRUG]->(d:Drug) WHERE id(d)=did
    RETURN count(distinct ds) AS c
    """
    rec = graph.run(q, ids=drug_ids).evaluate()
    return int(rec or 0)

def _distinct_drugset_count_by_other_ids(label: str, ids: List[int]) -> int:
    if not ids:
        return 0
    if label == "Reaction":
        q = """
        UNWIND $ids AS rid
        MATCH (ds:DrugSet)-[:CAUSES_REACTION]->(r:Reaction) WHERE id(r)=rid
        RETURN count(distinct ds) AS c
        """
    elif label == "Indication":
        q = """
        UNWIND $ids AS iid
        MATCH (ds:DrugSet)-[:TREATS_FOR]->(i:Indication) WHERE id(i)=iid
        RETURN count(distinct ds) AS c
        """
    elif label == "Outcome":
        q = """
        UNWIND $ids AS oid
        MATCH (p:Patient)-[:HAS_OUTCOME]->(o:Outcome) WHERE id(o)=oid
        MATCH (p)-[:USED_IN_CASE]->(ds:DrugSet)
        RETURN count(distinct ds) AS c
        """
    else:
        return 0
    rec = graph.run(q, ids=ids).evaluate()
    return int(rec or 0)

# ---------- 聚合查询（freq 改为 distinct 计数） ----------
def _reactions_from_drugs(drug_ids: List[int], k: int):
    if not drug_ids:
        return []
    q = """
    UNWIND $drugIds AS did
    MATCH (ds:DrugSet)-[:CONTAINS_DRUG]->(d:Drug) WHERE id(d)=did
    WITH distinct ds
    MATCH (ds)-[:CAUSES_REACTION]->(r:Reaction)
    WITH r, count(distinct ds) AS freq
    RETURN id(r) AS id, 'Reaction' AS label, r.reac AS text, r.embedding AS embedding, freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, drugIds=drug_ids, k=k).data()

def _drugs_from_reactions(reaction_ids: List[int], k: int):
    if not reaction_ids:
        return []
    q = """
    UNWIND $reactionIds AS rid
    MATCH (ds:DrugSet)-[:CAUSES_REACTION]->(r:Reaction) WHERE id(r)=rid
    WITH distinct ds
    MATCH (ds)-[:CONTAINS_DRUG]->(d:Drug)
    WITH d, count(distinct ds) AS freq
    RETURN id(d) AS id, 'Drug' AS label, d.drugname AS text, d.embedding AS embedding, freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, reactionIds=reaction_ids, k=k).data()

def _indications_from_drugs(drug_ids: List[int], k: int):
    if not drug_ids:
        return []
    q = """
    UNWIND $drugIds AS did
    MATCH (ds:DrugSet)-[:CONTAINS_DRUG]->(d:Drug) WHERE id(d)=did
    WITH distinct ds
    MATCH (ds)-[:TREATS_FOR]->(i:Indication)
    WITH i, count(distinct ds) AS freq
    RETURN id(i) AS id, 'Indication' AS label, i.indi AS text, i.embedding AS embedding, freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, drugIds=drug_ids, k=k).data()

def _drugs_from_indications(indication_ids: List[int], k: int):
    if not indication_ids:
        return []
    q = """
    UNWIND $indicationIds AS iid
    MATCH (ds:DrugSet)-[:TREATS_FOR]->(i:Indication) WHERE id(i)=iid
    WITH distinct ds
    MATCH (ds)-[:CONTAINS_DRUG]->(d:Drug)
    WITH d, count(distinct ds) AS freq
    RETURN id(d) AS id, 'Drug' AS label, d.drugname AS text, d.embedding AS embedding, freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, indicationIds=indication_ids, k=k).data()

def _outcomes_from_drugs(drug_ids: List[int], k: int):
    if not drug_ids:
        return []
    q = """
    UNWIND $drugIds AS did
    MATCH (p:Patient)-[:USED_IN_CASE]->(ds:DrugSet)-[:CONTAINS_DRUG]->(d:Drug) WHERE id(d)=did
    WITH distinct p
    MATCH (p)-[:HAS_OUTCOME]->(o:Outcome)
    WITH o, count(distinct p) AS freq
    RETURN id(o) AS id, 'Outcome' AS label, o.outccode AS text, o.embedding AS embedding, freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, drugIds=drug_ids, k=k).data()

def _drugs_from_outcomes(outcome_ids: List[int], k: int):
    if not outcome_ids:
        return []
    q = """
    UNWIND $outcomeIds AS oid
    MATCH (p:Patient)-[:HAS_OUTCOME]->(o:Outcome) WHERE id(o)=oid
    WITH distinct p
    MATCH (p)-[:USED_IN_CASE]->(ds:DrugSet)
    MATCH (ds)-[:CONTAINS_DRUG]->(d:Drug)
    WITH d, count(distinct p) AS freq
    RETURN id(d) AS id, 'Drug' AS label, d.drugname AS text, d.embedding AS embedding, freq
    ORDER BY freq DESC
    LIMIT $k
    """
    return graph.run(q, outcomeIds=outcome_ids, k=k).data()

def _global_top(label: str, k: int):
    if label == "Reaction":
        q = """
        MATCH (ds:DrugSet)-[:CAUSES_REACTION]->(r:Reaction)
        WITH r, count(distinct ds) AS freq
        RETURN id(r) AS id, 'Reaction' AS label, r.reac AS text, r.embedding AS embedding, freq
        ORDER BY freq DESC
        LIMIT $k
        """
    elif label == "Drug":
        q = """
        MATCH (ds:DrugSet)-[:CONTAINS_DRUG]->(d:Drug)
        WITH d, count(distinct ds) AS freq
        RETURN id(d) AS id, 'Drug' AS label, d.drugname AS text, d.embedding AS embedding, freq
        ORDER BY freq DESC
        LIMIT $k
        """
    elif label == "Indication":
        q = """
        MATCH (ds:DrugSet)-[:TREATS_FOR]->(i:Indication)
        WITH i, count(distinct ds) AS freq
        RETURN id(i) AS id, 'Indication' AS label, i.indi AS text, i.embedding AS embedding, freq
        ORDER BY freq DESC
        LIMIT $k
        """
    elif label == "Outcome":
        q = """
        MATCH (p:Patient)-[:HAS_OUTCOME]->(o:Outcome)
        WITH o, count(distinct p) AS freq
        RETURN id(o) AS id, 'Outcome' AS label, o.outccode AS text, o.embedding AS embedding, freq
        ORDER BY freq DESC
        LIMIT $k
        """
    else:
        return []
    return graph.run(q, k=k).data()

def aggregate_relations(search_output: Dict[str, Any]) -> Dict[str, Any]:
    # ...existing code...
    intents = search_output.get("intents", [])
    candidates = search_output.get("candidates", {})
    topk = search_output.get("topk", 10)
    seeds = search_output.get("seeds", [])

    seed_values_by_type: Dict[str, Set[str]] = {"Drug": set(), "Reaction": set(), "Indication": set(), "Outcome": set()}
    for s in seeds:
        typ = s.get("type")
        norm = s.get("normalized")
        if typ in seed_values_by_type and norm:
            if typ == "Drug":
                norm = norm.upper()
            seed_values_by_type[typ].add(norm)

    for label, rows in candidates.items():
        if label not in seed_values_by_type:
            continue
        for r in rows:
            val = r.get("value")
            if val:
                if label == "Drug":
                    val = val.upper()
                seed_values_by_type[label].add(val)

    all_drug_ids: Set[int] = set()
    for name in seed_values_by_type["Drug"]:
        all_drug_ids.update(_all_ids_exact("Drug", name))
    all_reaction_ids: Set[int] = set()
    for name in seed_values_by_type["Reaction"]:
        all_reaction_ids.update(_all_ids_exact("Reaction", name))
    all_indication_ids: Set[int] = set()
    for name in seed_values_by_type["Indication"]:
        all_indication_ids.update(_all_ids_exact("Indication", name))
    all_outcome_ids: Set[int] = set()
    for name in seed_values_by_type["Outcome"]:
        all_outcome_ids.update(_all_ids_exact("Outcome", name))

    if not all_drug_ids and candidates.get("Drug"):
        all_drug_ids.update(_candidate_ids(candidates, "Drug"))
    if not all_reaction_ids and candidates.get("Reaction"):
        all_reaction_ids.update(_candidate_ids(candidates, "Reaction"))
    if not all_indication_ids and candidates.get("Indication"):
        all_indication_ids.update(_candidate_ids(candidates, "Indication"))
    if not all_outcome_ids and candidates.get("Outcome"):
        all_outcome_ids.update(_candidate_ids(candidates, "Outcome"))

    drug_ids = list(all_drug_ids)
    reaction_ids = list(all_reaction_ids)
    indication_ids = list(all_indication_ids)
    outcome_ids = list(all_outcome_ids)

    expansions: Dict[str, Dict[str, Any]] = {}
    for intent in intents:
        if intent == "Reaction":
            if drug_ids:
                rows = _reactions_from_drugs(drug_ids, topk)
                source = "Drug(seeds_full)"
                stats_count = _distinct_drugset_count_by_drug_ids(drug_ids)
                seed_names = list(seed_values_by_type["Drug"])
            else:
                rows = _global_top("Reaction", topk)
                source = "Global"
                stats_count = 0
                seed_names = []
            missing = sum(1 for r in rows if r.get("embedding") is None)
            expansions[intent] = {
                "source": source,
                "rows": rows,
                "columns": ["id", "label", "text", "embedding", "freq"],
                "stats": {
                    "distinct_drugset_count": stats_count,
                    "seed_drugs": seed_names,
                    "embedding_missing": missing,
                },
            }
        elif intent == "Drug":
            if reaction_ids:
                rows = _drugs_from_reactions(reaction_ids, topk)
                source = "Reaction(seeds_full)"
                stats_count = _distinct_drugset_count_by_other_ids("Reaction", reaction_ids)
                seed_names = list(seed_values_by_type["Reaction"])
            elif indication_ids:
                rows = _drugs_from_indications(indication_ids, topk)
                source = "Indication(seeds_full)"
                stats_count = _distinct_drugset_count_by_other_ids("Indication", indication_ids)
                seed_names = list(seed_values_by_type["Indication"])
            elif outcome_ids:
                rows = _drugs_from_outcomes(outcome_ids, topk)
                source = "Outcome(seeds_full)"
                stats_count = _distinct_drugset_count_by_other_ids("Outcome", outcome_ids)
                seed_names = list(seed_values_by_type["Outcome"])
            else:
                rows = _global_top("Drug", topk)
                source = "Global"
                stats_count = 0
                seed_names = []
            missing = sum(1 for r in rows if r.get("embedding") is None)
            expansions[intent] = {
                "source": source,
                "rows": rows,
                "columns": ["id", "label", "text", "embedding", "freq"],
                "stats": {
                    "distinct_drugset_count": stats_count,
                    "seed_terms": seed_names,
                    "embedding_missing": missing,
                },
            }
        elif intent == "Indication":
            if drug_ids:
                rows = _indications_from_drugs(drug_ids, topk)
                source = "Drug(seeds_full)"
                stats_count = _distinct_drugset_count_by_drug_ids(drug_ids)
                seed_names = list(seed_values_by_type["Drug"])
            else:
                rows = _global_top("Indication", topk)
                source = "Global"
                stats_count = 0
                seed_names = []
            missing = sum(1 for r in rows if r.get("embedding") is None)
            expansions[intent] = {
                "source": source,
                "rows": rows,
                "columns": ["id", "label", "text", "embedding", "freq"],
                "stats": {
                    "distinct_drugset_count": stats_count,
                    "seed_drugs": seed_names,
                    "embedding_missing": missing,
                },
            }
        elif intent == "Outcome":
            if drug_ids:
                rows = _outcomes_from_drugs(drug_ids, topk)
                source = "Drug(seeds_full)"
                stats_count = _distinct_drugset_count_by_drug_ids(drug_ids)
                seed_names = list(seed_values_by_type["Drug"])
            else:
                rows = _global_top("Outcome", topk)
                source = "Global"
                stats_count = 0
                seed_names = []
            missing = sum(1 for r in rows if r.get("embedding") is None)
            expansions[intent] = {
                "source": source,
                "rows": rows,
                "columns": ["id", "label", "text", "embedding", "freq"],
                "stats": {
                    "distinct_drugset_count": stats_count,
                    "seed_drugs": seed_names,
                    "embedding_missing": missing,
                },
            }
        else:
            expansions[intent] = {
                "source": "Unknown",
                "rows": [],
                "columns": [],
                "stats": {},
            }

    output = {
        "question": search_output.get("question"),
        "seeds": seeds,
        "intents": intents,
        "topk": topk,
        "query_vector": search_output.get("query_vector"),
        "seed_values_full": {k: list(v) for k, v in seed_values_by_type.items()},
        "id_sets": {
            "Drug_ids": len(drug_ids),
            "Reaction_ids": len(reaction_ids),
            "Indication_ids": len(indication_ids),
            "Outcome_ids": len(outcome_ids),
        },
        "expansions": expansions,
        "meta": {
            "layer": "relation_aggregate",
            "previous_meta": search_output.get("meta"),
            "normalization": "Drug seeds upper-cased",
            "embedding_included": True,
        },
    }

    display_output = dict(output)
    qv = display_output.get("query_vector")
    if isinstance(qv, list):
        display_output["query_vector"] = f"dim={len(qv)}"
    elif qv is None:
        display_output["query_vector"] = None

    masked_expansions = {}
    for intent, block in output["expansions"].items():
        blk = dict(block)
        masked_rows = []
        for r in blk.get("rows", []):
            rr = dict(r)
            emb = rr.get("embedding")
            if isinstance(emb, list):
                rr["embedding"] = f"dim={len(emb)}"
            elif emb is None:
                rr["embedding"] = None
            masked_rows.append(rr)
        blk["rows"] = masked_rows
        masked_expansions[intent] = blk
    display_output["expansions"] = masked_expansions

    print("======== 关系聚合层输出 ========")
    print(json.dumps(display_output, ensure_ascii=False, indent=2))
    print("================================")

    return output

if __name__ == "__main__":
    tests = [
        "阿司匹林的常见不良反应有哪些？",
        "NAUSEA 常见于哪些药物？",
        "阿司匹林一般用于哪些适应症？",
        "关于氯吡格雷，有哪些患者结局？",
        "请列出常见药物。",
        "有哪些药物与心肌梗死相关？",
        "头痛常见于哪些药物？",
    ]
    for q in tests:
        print("\n问题：", q)
        parsed = parse_input(q)
        searched = initial_search(parsed)
        aggregate_relations(searched)