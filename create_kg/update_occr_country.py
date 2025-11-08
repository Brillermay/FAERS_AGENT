# encoding:utf8
import os
from py2neo import Graph
from tqdm import tqdm

#服务器上不需要运行这个代码

class CountryUpdater:
    def __init__(self, uri=None, user=None, password=None):
        # 优先使用传入参数，其次使用环境变量，最后使用默认 Bolt 地址
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "12345678")

        try:
            self.graph = Graph(uri, auth=(user, password))
        except Exception as e:
            raise RuntimeError(f"连接 Neo4j 失败: {e}")

    def update_occr_country_from_file(self, data_path):
        """使用 Cypher 批量更新从DEMO文件重新读取的occr_country"""
        
        print("开始更新患者节点的occr_country字段...")
        
        # 首先读取正确的occr_country数据
        country_data = {}
        
        with open(data_path + 'DEMO23Q1.txt', 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), ncols=80, desc="读取正确的occr_country数据"):
                fields = line.strip().split('$')
                
                # 确保有足够的字段且关键字段不为空
                if len(fields) > 24 and fields[0].strip() and fields[1].strip():
                    primaryid = fields[0]
                    caseid = fields[1]
                    occr_country = fields[24] if len(fields) > 24 else ""
                    
                    # 使用 (primaryid, caseid) 作为键
                    key = (primaryid, caseid)
                    country_data[key] = occr_country
        
        print(f"读取到 {len(country_data)} 条occr_country数据")
        
        # 使用 Cypher 批量更新Patient节点
        total_updated = 0
        batch_size = 1000
        keys = list(country_data.keys())
        
        for i in tqdm(range(0, len(keys), batch_size), desc="Cypher批量更新Patient节点"):
            batch_keys = keys[i:i + batch_size]
            
            # 准备批量更新的参数
            updates = []
            for key in batch_keys:
                primaryid, caseid = key
                updates.append({
                    'primaryid': primaryid,
                    'caseid': caseid,
                    'occr_country': country_data[key]
                })
            
            # 使用 Cypher 批量更新
            cypher = """
            UNWIND $updates as update
            MATCH (p:Patient {primaryid: update.primaryid, caseid: update.caseid})
            SET p.occr_country = update.occr_country
            RETURN count(p) as updated_count
            """
            
            try:
                # 修复：正确处理 Cursor 对象
                cursor = self.graph.run(cypher, updates=updates)
                result = cursor.evaluate()  # 使用 evaluate() 获取单个值
                if result is not None:
                    batch_updated = result
                    total_updated += batch_updated
                else:
                    # 如果 evaluate() 返回 None，尝试用 data() 方法
                    cursor = self.graph.run(cypher, updates=updates)
                    results = cursor.data()
                    if results and len(results) > 0:
                        batch_updated = results[0].get('updated_count', 0)
                        total_updated += batch_updated
                    
            except Exception as e:
                print(f"批量更新第 {i//batch_size + 1} 批时出错: {e}")
        
        print(f"更新完成！")
        print(f"成功更新: {total_updated} 个Patient节点")
        
        return total_updated

    def update_occr_country_with_transaction(self, data_path):
        """使用事务的方式进行 Cypher 批量更新（更安全）"""
        
        print("开始使用事务更新患者节点的occr_country字段...")
        
        # 读取数据
        country_data = {}
        with open(data_path + 'DEMO23Q1.txt', 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), ncols=80, desc="读取正确的occr_country数据"):
                fields = line.strip().split('$')
                if len(fields) > 24 and fields[0].strip() and fields[1].strip():
                    primaryid = fields[0]
                    caseid = fields[1]
                    occr_country = fields[24] if len(fields) > 24 else ""
                    country_data[(primaryid, caseid)] = occr_country
        
        print(f"读取到 {len(country_data)} 条occr_country数据")
        
        # 使用事务批量更新
        total_updated = 0
        batch_size = 500  # 事务中使用较小的批次
        keys = list(country_data.keys())
        
        for i in tqdm(range(0, len(keys), batch_size), desc="事务批量更新"):
            batch_keys = keys[i:i + batch_size]
            
            try:
                # 开始事务
                tx = self.graph.begin()
                
                for key in batch_keys:
                    primaryid, caseid = key
                    new_occr_country = country_data[key]
                    
                    # 单个更新的 Cypher 语句
                    cypher = """
                    MATCH (p:Patient {primaryid: $primaryid, caseid: $caseid})
                    SET p.occr_country = $occr_country
                    RETURN count(p) as updated
                    """
                    
                    # 修复：正确处理事务中的 Cursor
                    cursor = tx.run(cypher, 
                                  primaryid=primaryid, 
                                  caseid=caseid, 
                                  occr_country=new_occr_country)
                    
                    # 获取结果
                    results = cursor.data()
                    if results and len(results) > 0:
                        updated_count = results[0].get('updated', 0)
                        total_updated += updated_count
                
                # 提交事务
                tx.commit()
                
            except Exception as e:
                print(f"事务更新第 {i//batch_size + 1} 批时出错: {e}")
                if 'tx' in locals():
                    tx.rollback()
        
        print(f"事务更新完成！成功更新: {total_updated} 个Patient节点")
        return total_updated

    def verify_update(self, sample_size=10):
        """验证更新是否成功，随机检查几个节点"""
        print(f"\n验证更新结果（随机检查 {sample_size} 个节点）...")
        
        # 随机获取一些Patient节点
        cypher = f"MATCH (p:Patient) RETURN p.primaryid, p.caseid, p.occr_country ORDER BY rand() LIMIT {sample_size}"
        results = self.graph.run(cypher).data()
        
        print("样本节点的occr_country值：")
        for result in results:
            print(f"  primaryid: {result['p.primaryid']}, caseid: {result['p.caseid']}, occr_country: '{result['p.occr_country']}'")

    def get_update_statistics(self):
        """获取更新统计信息"""
        print("\n=== 更新统计信息 ===")
        
        # 统计总的Patient节点数
        total_patients = self.graph.run("MATCH (p:Patient) RETURN count(p) as total").evaluate()
        print(f"总Patient节点数: {total_patients}")
        
        # 统计有occr_country的节点数
        with_country = self.graph.run("MATCH (p:Patient) WHERE p.occr_country IS NOT NULL AND p.occr_country <> '' RETURN count(p) as count").evaluate()
        print(f"有occr_country的节点数: {with_country}")
        
        # 统计没有occr_country的节点数
        without_country = self.graph.run("MATCH (p:Patient) WHERE p.occr_country IS NULL OR p.occr_country = '' RETURN count(p) as count").evaluate()
        print(f"没有occr_country的节点数: {without_country}")
        
        # 显示不同国家的分布（前10）
        print("\n国家分布（前10）：")
        country_dist = self.graph.run("""
            MATCH (p:Patient) 
            WHERE p.occr_country IS NOT NULL AND p.occr_country <> ''
            RETURN p.occr_country as country, count(p) as count 
            ORDER BY count DESC 
            LIMIT 10
        """).data()
        
        for item in country_dist:
            print(f"  {item['country']}: {item['count']} 个节点")


def main():
    # 数据文件路径
    data_path = "data_source/faers_ascii_2023Q1/ASCII/"
    
    # 创建更新器
    updater = CountryUpdater()
    
    try:
        # 选择更新方式
        print("选择更新方式：")
        print("1. 使用 UNWIND 批量更新（推荐，速度快）")
        print("2. 使用事务批量更新（安全，速度中等）")
        
        choice = input("请选择 (1 或 2，默认为 1): ").strip() or "1"
        
        if choice == "1":
            # 使用 UNWIND 批量更新
            updated_count = updater.update_occr_country_from_file(data_path)
        else:
            # 使用事务批量更新
            updated_count = updater.update_occr_country_with_transaction(data_path)
        
        # 验证更新结果
        updater.verify_update()
        
        # 显示统计信息
        updater.get_update_statistics()
        
        print(f"\n=== 更新总结 ===")
        print(f"总共更新: {updated_count} 个Patient节点")
        
    except Exception as e:
        print(f"更新过程中发生错误: {e}")


if __name__ == '__main__':
    main()