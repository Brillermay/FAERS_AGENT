# encoding:utf8
import os
from py2neo import Graph
from langchain_community.embeddings import DashScopeEmbeddings
from tqdm import tqdm
import numpy as np  # 用于子图向量计算

class EmbeddingGeneratorLangChain:
    def __init__(self, uri=None, user=None, password=None, api_key=None):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "rainshineking274")
        self.graph = Graph(uri, auth=(user, password))
        
        # 使用 DashScope API Key
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-9e05a08baf0142088b0de2f6dabbb730")
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key=api_key
        )

    def generate_node_embeddings(self, node_label, text_fields, primaryid_field="primaryid"):
        """
        为指定节点类型生成 embeddings。
        :param node_label: 节点标签，如 "Drug"
        :param text_fields: 用于生成文本的字段列表，如 ["drugname", "prod_ai"]
        :param primaryid_field: 主键字段，默认 "primaryid"
        """
        # 查询所有节点
        query = f"MATCH (n:{node_label}) RETURN n.{primaryid_field} AS primaryid, n"
        nodes = self.graph.run(query).data()
        
        for node_data in tqdm(nodes, desc=f"生成 {node_label} embeddings"):
            node = node_data["n"]
            primaryid = node_data["primaryid"]
            
            # 拼接文本（覆盖所有字段）
            text_parts = [str(node.get(field, "")) for field in text_fields]
            text = " ".join(text_parts).strip()
            if not text:
                continue  # 跳过空文本
            
            # 生成 embedding
            embedding = self.embeddings.embed_query(text)
            
            # 更新节点
            update_query = f"""
            MATCH (n:{node_label} {{{primaryid_field}: $primaryid}})
            SET n.embedding = $embedding
            """
            self.graph.run(update_query, primaryid=primaryid, embedding=embedding)
        
        print(f"{node_label} embeddings 生成并存储完成")

    def generate_relationship_embeddings(self, rel_type, node_labels, text_fields_a, text_fields_b):
        """
        为指定关系类型生成 embeddings。
        :param rel_type: 关系类型，如 "USED_IN_CASE"
        :param node_labels: 连接的节点标签列表，如 ["Patient", "Drug"]
        :param text_fields_a: 起始节点字段，如 ["age", "sex"]
        :param text_fields_b: 结束节点字段，如 ["drugname"]
        """
        query = f"MATCH (a:{node_labels[0]})-[r:{rel_type}]->(b:{node_labels[1]}) RETURN r, a, b"
        rels = self.graph.run(query).data()
        
        for rel_data in tqdm(rels, desc=f"生成 {rel_type} embeddings"):
            r = rel_data["r"]
            a = rel_data["a"]
            b = rel_data["b"]
            
            # 拼接文本（关系类型 + 节点字段）
            text_a = " ".join([str(a.get(field, "")) for field in text_fields_a])
            text_b = " ".join([str(b.get(field, "")) for field in text_fields_b])
            text = f"{rel_type} {text_a} {text_b}".strip()
            if not text:
                continue
            
            embedding = self.embeddings.embed_query(text)
            
            # 更新关系属性
            update_query = f"MATCH ()-[r:{rel_type}]-() WHERE id(r) = $rel_id SET r.embedding = $embedding"
            self.graph.run(update_query, rel_id=r.identity, embedding=embedding)
        
        print(f"{rel_type} embeddings 生成并存储完成")

    def generate_subgraph_embedding(self, center_node_id, node_label="Patient", depth=1):
        """
        动态生成子图 embeddings（基于中心节点）。
        :param center_node_id: 中心节点 ID，如 primaryid
        :param node_label: 中心节点标签
        :param depth: 子图深度
        :return: 子图向量（平均节点 embeddings）
        """
        # 查询子图节点
        query = f"""
        MATCH (center:{node_label} {{primaryid: $center_node_id}})
        MATCH path = (center)-[*0..{depth}]-(n)
        RETURN DISTINCT n.embedding AS embedding
        """
        result = self.graph.run(query, center_node_id=center_node_id).data()
        
        embeddings = [emb["embedding"] for emb in result if emb["embedding"]]
        if not embeddings:
            return None
        
        # 计算平均向量
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        return avg_embedding

    def generate_all_embeddings(self):
        # 节点 embeddings（覆盖所有字段）
        self.generate_node_embeddings("Patient", ["age", "age_grp", "event_dt", "sex", "wt", "reporter_country", "occr_country"])
        self.generate_node_embeddings("Drug", ["drugname", "prod_ai", "dose_amt", "dose_unit", "dose_freq"])
        self.generate_node_embeddings("Reaction", ["reac"])
        self.generate_node_embeddings("Outcome", ["outccode"])
        self.generate_node_embeddings("Indication", ["indi"])
        
        # 关系 embeddings（基于 build_kg_utils.py 的关系）
        self.generate_relationship_embeddings("USED_IN_CASE", ["Patient", "Drug"], ["age", "sex"], ["drugname"])
        self.generate_relationship_embeddings("HAS_INDICATION", ["Drug", "Indication"], ["drugname"], ["indi"])
        self.generate_relationship_embeddings("CAUSES_REACTION", ["Drug", "Reaction"], ["drugname"], ["reac"])
        self.generate_relationship_embeddings("HAS_OUTCOME", ["Reaction", "Outcome"], ["reac"], ["outccode"])

if __name__ == "__main__":
    generator = EmbeddingGeneratorLangChain()
    generator.generate_all_embeddings()
    # 示例：动态生成子图 embedding
    # subgraph_emb = generator.generate_subgraph_embedding("some_primaryid")
    # print(subgraph_emb)