# 简易 GraphRAG 演示：用 Cypher 检索 + OpenAI 生成回答
from openai import OpenAI
from py2neo import Graph

# 配置 OpenAI（请把 YOUR_API_KEY 替换为你的实际 Key）
client = OpenAI(
    api_key="sk-7cuAMuvifXU3TogzrjZahugbyKSnHUZPmZ68SfzpxRFuJWLn",
    base_url="https://api.chatanywhere.tech/v1",
)

# 连接 Neo4j（沿用你项目的本地连接与密码）
graph = Graph("bolt://localhost:7687", auth=("neo4j", "rainshineking274"))

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

def outcomes_for_drug(drug: str, k: int = 10):
    q = """
    MATCH (d:Drug)-[:CAUSES_REACTION]->(r:Reaction)-[:HAS_OUTCOME]->(o:Outcome)
    WHERE toLower(d.drugname) = toLower($drug)
    RETURN o.outccode AS outcome, count(*) AS freq
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

def fmt_table(rows, key_col, val_col):
    if not rows:
        return "（无检索结果）"
    lines = [f"- {row[key_col]}（{row[val_col]}）" for row in rows]
    return "\n".join(lines)

def demo():
    # 预设三个可运行的问题
    demos = [
        ("青霉素的常见不良反应有哪些？", lambda: fmt_table(top_reactions_for_drug("aspirin"), "reaction", "freq")),
        ("不良反应“NAUSEA”最常关联的药物是哪些？", lambda: fmt_table(top_drugs_for_reaction("NAUSEA"), "drug", "freq")),
        ("阿司匹林的常见适应症有哪些？", lambda: fmt_table(indications_for_drug("aspirin"), "indication", "freq")),
    ]
    for q, retr in demos:
        ctx = retr()
        print("=" * 60)
        print("问题：", q)
        print("检索结果：\n", ctx)
        ans = llm_answer(q, ctx)
        print("\n回答：\n", ans)

if __name__ == "__main__":
    demo()