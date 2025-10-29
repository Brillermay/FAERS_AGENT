from py2neo import Graph
import sys
import time

def clear_neo4j_database(uri, user, password, batch_size=10000):
    """
    分批次清空Neo4j数据库（避免内存溢出）
    :param batch_size: 每批删除的节点数量
    """
    try:
        graph = Graph(uri, auth=(user, password))
        print("成功连接到Neo4j数据库")

        total_nodes_deleted = 0
        total_rels_deleted = 0

        while True:
            # 1. 先删除关系（减少节点关联，降低内存占用）
            rel_result = graph.run(f"MATCH ()-[r]->() WITH r LIMIT {batch_size} DELETE r RETURN count(r) AS cnt")
            rel_cnt = rel_result.evaluate()
            total_rels_deleted += rel_cnt

            # 2. 再删除节点
            node_result = graph.run(f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(n) AS cnt")
            node_cnt = node_result.evaluate()
            total_nodes_deleted += node_cnt

            print(f"已删除 {total_nodes_deleted} 个节点，{total_rels_deleted} 个关系（本批：{node_cnt}节点，{rel_cnt}关系）")

            # 当两批都没有删除内容时，说明已清空
            if node_cnt == 0 and rel_cnt == 0:
                break

            # 每批删除后休眠0.5秒，给内存释放时间
            time.sleep(0.5)

        print(f"数据库清空完成！共删除 {total_nodes_deleted} 个节点，{total_rels_deleted} 个关系")

    except Exception as e:
        print(f"操作失败：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "rainshineking274"

    confirm = input("警告：此操作将删除Neo4j中的所有数据，且无法恢复！确认继续？(输入 'yes' 执行)：")
    if confirm.lower() == "yes":
        # 可根据数据量调整batch_size（数据量大则调小，比如5000）
        clear_neo4j_database(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, batch_size=10000)
    else:
        print("操作已取消")