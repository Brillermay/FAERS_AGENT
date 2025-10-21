# encoding:utf8
import os
import time
import numpy as np
from py2neo import Graph
from langchain_community.embeddings import DashScopeEmbeddings
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from requests.exceptions import SSLError, ConnectionError, ReadTimeout, ChunkedEncodingError

# 默认参数，可通过环境变量调整
DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "5"))
DEFAULT_DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-9e05a08baf0142088b0de2f6dabbb730")

class EmbeddingGeneratorLangChain:
    def __init__(self, uri=None, user=None, password=None, api_key=None, batch_size=None):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        # 密码直接写入（按要求）
        password = password or "rainshineking274"
        self.graph = Graph(uri, auth=(user, password))
        api_key = api_key or DEFAULT_DASHSCOPE_KEY
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=api_key
        )
        self.batch_size = batch_size or DEFAULT_BATCH_SIZE

    @retry(stop=stop_after_attempt(MAX_RETRIES),
           wait=wait_exponential(multiplier=1, min=1, max=30),
           retry=retry_if_exception_type((SSLError, ConnectionError, ReadTimeout, ChunkedEncodingError)))
    def _embed_with_retry(self, texts):
        """
        批量接口带重试（tenacity）。
        """
        return self.embeddings.embed_documents(texts)

    def _chunks(self, seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i+size]

    def generate_node_embeddings(self, node_label, text_fields, primaryid_field="primaryid", skip_existing=True):
        """
        批量生成节点 embeddings，并批量写回 Neo4j。
        会跳过已存在 embedding 的节点，支持失败回退逐条请求。
        """
        query = f"MATCH (n:{node_label}) RETURN n.{primaryid_field} AS primaryid, n"
        rows = self.graph.run(query).data()

        items = []
        for row in rows:
            node = row["n"]
            pid = row["primaryid"]
            # 跳过已有 embedding
            if skip_existing and node.get("embedding"):
                continue
            text_parts = [str(node.get(field, "")) for field in text_fields]
            text = " ".join([p for p in text_parts if p]).strip()
            if text:
                items.append((pid, text))

        if not items:
            print(f"{node_label} 无需生成 embedding")
            return

        batches = list(self._chunks(items, self.batch_size))
        for batch in tqdm(batches, desc=f"生成 {node_label} embeddings", unit="batch"):
            pids = [x[0] for x in batch]
            texts = [x[1] for x in batch]
            embeddings = None
            try:
                embeddings = self._embed_with_retry(texts)
            except Exception as e:
                print(f"[WARN] 批量 embed 失败，尝试逐条回退: {e}")
                embeddings = []
                for t in texts:
                    try:
                        emb = self.embeddings.embed_query(t)
                    except Exception as e2:
                        print(f"[WARN] 单条 embed 失败，跳过: {e2}")
                        emb = None
                    embeddings.append(emb)

            # 批量更新 Neo4j
            pairs = []
            for pid, emb in zip(pids, embeddings):
                if emb is None:
                    continue
                pairs.append({"primaryid": pid, "embedding": emb})

            if pairs:
                update_query = f"""
                UNWIND $pairs AS pair
                MATCH (n:{node_label} {{ {primaryid_field}: pair.primaryid }})
                SET n.embedding = pair.embedding
                """
                self.graph.run(update_query, pairs=pairs)

        print(f"{node_label} embeddings 生成并存储完成")

    def generate_relationship_embeddings(self, rel_type, node_labels, text_fields_a, text_fields_b, skip_existing=True):
        """
        批量生成关系 embeddings，并批量写回（按关系 id）。
        """
        query = f"MATCH (a:{node_labels[0]})-[r:{rel_type}]->(b:{node_labels[1]}) RETURN id(r) AS rid, a, b, r"
        rows = self.graph.run(query).data()

        items = []
        for row in rows:
            rid = row["rid"]
            r = row["r"]
            if skip_existing and r.get("embedding"):
                continue
            a = row["a"]
            b = row["b"]
            text_a = " ".join([str(a.get(f, "")) for f in text_fields_a])
            text_b = " ".join([str(b.get(f, "")) for f in text_fields_b])
            text = f"{rel_type} {text_a} {text_b}".strip()
            if text:
                items.append((int(rid), text))

        if not items:
            print(f"{rel_type} 无需生成 embedding")
            return

        batches = list(self._chunks(items, self.batch_size))
        for batch in tqdm(batches, desc=f"生成 {rel_type} embeddings", unit="batch"):
            rids = [x[0] for x in batch]
            texts = [x[1] for x in batch]
            embeddings = None
            try:
                embeddings = self._embed_with_retry(texts)
            except Exception as e:
                print(f"[WARN] 批量关系 embed 失败，逐条回退: {e}")
                embeddings = []
                for t in texts:
                    try:
                        emb = self.embeddings.embed_query(t)
                    except Exception as e2:
                        print(f"[WARN] 单条关系 embed 失败，跳过: {e2}")
                        emb = None
                    embeddings.append(emb)

            pairs = []
            for rid, emb in zip(rids, embeddings):
                if emb is None:
                    continue
                pairs.append({"rel_id": rid, "embedding": emb})

            if pairs:
                update_query = """
                UNWIND $pairs AS pair
                MATCH ()-[r]-() WHERE id(r) = pair.rel_id
                SET r.embedding = pair.embedding
                """
                self.graph.run(update_query, pairs=pairs)

        print(f"{rel_type} embeddings 生成并存储完成")

    def generate_subgraph_embedding(self, center_node_id, node_label="Patient", depth=1):
        """
        动态生成子图 embeddings（平均邻居节点 embedding）。
        """
        query = f"""
        MATCH (center:{node_label} {{primaryid: $center_node_id}})
        MATCH path = (center)-[*0..{depth}]-(n)
        RETURN DISTINCT n.embedding AS embedding
        """
        result = self.graph.run(query, center_node_id=center_node_id).data()
        embeddings = [row["embedding"] for row in result if row.get("embedding")]
        if not embeddings:
            return None
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        return avg_embedding

    def store_all_subgraph_embeddings(self, node_label="Patient", depth=1, skip_existing=True):
        """
        可选：为所有中心节点计算并存储子图 embedding（谨慎运行，可能量大）。
        """
        query = f"MATCH (n:{node_label}) RETURN n.primaryid AS primaryid, n"
        rows = self.graph.run(query).data()
        tasks = []
        for row in rows:
            pid = row["primaryid"]
            node = row["n"]
            if skip_existing and node.get("subgraph_embedding"):
                continue
            tasks.append(pid)

        for pid in tqdm(tasks, desc=f"计算并存储 {node_label} 子图向量"):
            emb = self.generate_subgraph_embedding(pid, node_label=node_label, depth=depth)
            if emb is not None:
                update_q = f"MATCH (n:{node_label} {{primaryid: $pid}}) SET n.subgraph_embedding = $emb"
                self.graph.run(update_q, pid=pid, emb=emb)

    def generate_all_embeddings(self):
        # 按 build_kg_utils 中的字段尽量覆盖
        self.generate_node_embeddings("Patient", ["age", "age_grp", "event_dt", "sex", "wt", "reporter_country", "occr_country"])
        self.generate_node_embeddings("Drug", ["drugname", "prod_ai", "dose_amt", "dose_unit", "dose_freq", "role_cod", "route", "dechal", "rechal"])
        self.generate_node_embeddings("Reaction", ["reac"])
        self.generate_node_embeddings("Outcome", ["outccode"])
        self.generate_node_embeddings("Indication", ["indi"])

        # 关系 embeddings（名称需与构建时一致）
        self.generate_relationship_embeddings("USED_IN_CASE", ["Patient", "Drug"], ["age", "sex"], ["drugname"])
        self.generate_relationship_embeddings("HAS_INDICATION", ["Drug", "Indication"], ["drugname"], ["indi"])
        self.generate_relationship_embeddings("CAUSES_REACTION", ["Drug", "Reaction"], ["drugname"], ["reac"])
        self.generate_relationship_embeddings("HAS_OUTCOME", ["Reaction", "Outcome"], ["reac"], ["outccode"])

if __name__ == "__main__":
    gen = EmbeddingGeneratorLangChain()
    gen.generate_all_embeddings()
    # 如需把子图 embedding 批量计算并存储（注意可能非常耗时/费用），取消注释：
    # gen.store_all_subgraph_embeddings(node_label="Patient", depth=1)
