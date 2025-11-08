# encoding:utf8
import os
import re
import json
import codecs
import threading
from py2neo import Graph, Node, Relationship
import pandas as pd
import numpy as np
from tqdm import tqdm


class MedicalExtractor(object):
    def __init__(self, uri=None, user=None, password=None):
        super(MedicalExtractor, self).__init__()
        # 优先使用传入参数，其次使用环境变量，最后使用默认 Bolt 地址
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "12345678")

        try:
            self.graph = Graph(uri, auth=(user, password))
        except Exception as e:
            raise RuntimeError(f"连接 Neo4j 失败: {e}")

        # 共5类节点
        self.patients = []  # 患者
        self.drugs = []  # 药物
        self.indications = []  # 适应症
        self.reactions = []  # 不良反应
        self.outcomes = []  # 患者结果

        # 构建节点实体关系
        self.rels_usedincase = []  # 患者－药物关系
        self.rels_hasindication = []  # 药物－适应症关系
        self.rels_causesreaction = []  # 药物－不良反应关系
        self.rels_hasoutcome = []  # 不良反应－患者结果关系

        # 添加一个集合来存储无效的 primaryid
        self.invalid_primaryids = set()
        # 添加一个集合来存储有效的 primaryid (用于优化查询)
        self.valid_primaryids = set()

    def extract_triples(self, data_path):
        print("从txt文件中转换抽取三元组")

        # 修改读取 demo.txt 文件的部分
        with open(data_path + 'DEMO23Q2.txt', 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), ncols=80, desc="读取患者数据"):
                fields = line.strip().split('$')
                # 检查所有必需字段是否为空
                required_indices = [0, 1, 4, 13, 16, 18, 24]
                if any(not fields[i].strip() for i in required_indices):
                    self.invalid_primaryids.add(fields[0])
                    continue

                patient_data = {
                    'primaryid': fields[0],
                    'caseid': fields[1],
                    'event_dt': fields[4],
                    'age': fields[13],
                    'age_grp': fields[15],
                    'sex': fields[16],
                    'wt': fields[18],
                    'occr_country': fields[24],
                }
                self.valid_primaryids.add(fields[0])
                # 将患者数据存入patients列表
                self.patients.append(patient_data)

        # 修改读取 drug.txt 文件的部分
        with open(data_path + 'DRUG23Q2.txt', 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), ncols=80, desc="读取药物数据"):
                fields = line.strip().split('$')
                # 检查 primaryid 是否已经被标记为无效
                if fields[0] in self.invalid_primaryids:
                    continue
                
                # 检查必需的药物字段是否为空
                required_drug_indices = [0, 1, 4, 5]  # primaryid, caseid, drugname, prod_ai, route
                if any(not fields[i].strip() for i in required_drug_indices):
                    self.invalid_primaryids.add(fields[0])
                    # 如果这个 primaryid 之前是有效的，现在需要移除
                    if fields[0] in self.valid_primaryids:
                        self.valid_primaryids.remove(fields[0])
                    continue

                drug_data = {
                    'primaryid': fields[0],
                    'caseid': fields[1],
                    'drugname': fields[4],
                    'prod_ai': fields[5],
                    'dose_amt': fields[16],
                    'dose_unit': fields[17],
                    'dose_freq': fields[19],
                }
                # 将药物数据存入drugs列表
                self.drugs.append(drug_data)

        # 修改读取 reac.txt 文件的部分
        with open(data_path + 'REAC23Q2.txt', 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), ncols=80, desc="读取不良反应数据"):
                fields = line.strip().split('$')
                if fields[0] in self.invalid_primaryids:
                    continue

                reac_data = {
                    'primaryid': fields[0],
                    'caseid': fields[1],
                    'reac': fields[2],
                }
                # 将不良反应数据存入reactions列表
                self.reactions.append(reac_data)

        # 修改读取 outc.txt 文件的部分
        with open(data_path + 'OUTC23Q2.txt', 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), ncols=80, desc="读取患者结果数据"):
                fields = line.strip().split('$')
                if fields[0] in self.invalid_primaryids:
                    continue

                outc_data = {
                    'primaryid': fields[0],
                    'caseid': fields[1],
                    'outccode': fields[2],
                }
                # 将患者结果数据存入outcomes列表
                self.outcomes.append(outc_data)

        # 修改读取 indi.txt 文件的部分
        with open(data_path + 'INDI23Q2.txt', 'r', encoding='utf8') as f:
            for line in tqdm(f.readlines(), ncols=80, desc="读取适应症数据"):
                fields = line.strip().split('$')
                if fields[0] in self.invalid_primaryids:
                    continue

                indi_data = {
                    'primaryid': fields[0],
                    'caseid': fields[1],
                    'indi': fields[3],
                }
                # 将适应症数据存入indications列表
                self.indications.append(indi_data)

        print(f"发现并过滤掉 {len(self.invalid_primaryids)} 条无效患者记录")
        # 构建图谱的节点和关系
        self.create_graph_nodes_and_relationships()

    def create_graph_nodes_and_relationships(self):
        # 创建患者节点
        for patient in tqdm(self.patients, ncols=80, desc="创建患者节点"):
            if patient['primaryid'] not in self.valid_primaryids:
                continue
            patient_node = Node("Patient", primaryid=patient['primaryid'], caseid=patient['caseid'])
            # 添加新的患者特征
            patient_node['age'] = patient['age']
            patient_node['age_grp'] = patient['age_grp']
            patient_node['event_dt'] = patient['event_dt']
            patient_node['sex'] = patient['sex']
            patient_node['wt'] = patient['wt']
            patient_node['occr_country'] = patient['occr_country']
            self.graph.create(patient_node)

        # 创建 DrugSet 节点（为每个唯一的 primaryid + caseid 组合创建一个）
        drugset_created = set()
        for patient in tqdm(self.patients, ncols=80, desc="创建药物集合节点"):
            if patient['primaryid'] not in self.valid_primaryids:
                continue
            
            key = (patient['primaryid'], patient['caseid'])
            if key not in drugset_created:
                drugset_node = Node("DrugSet", primaryid=patient['primaryid'], caseid=patient['caseid'])
                self.graph.create(drugset_node)
                drugset_created.add(key)
                
                # 创建 Patient -> DrugSet 关系
                patient_node = self.graph.nodes.match("Patient", 
                                                    primaryid=patient['primaryid'],
                                                    caseid=patient['caseid']).first()
                if patient_node:
                    used_in_case_rel = Relationship(patient_node, "USED_IN_CASE", drugset_node)
                    self.graph.create(used_in_case_rel)

        # 创建药物节点和 DrugSet -> Drug 关系（去重处理）
        drug_created = set()
        for drug in tqdm(self.drugs, ncols=80, desc="创建药物节点与关系"):
            if drug['primaryid'] not in self.valid_primaryids:
                continue
            
            # 检查是否已经存在相同的药物（primaryid + caseid + drugname）
            drug_key = (drug['primaryid'], drug['caseid'], drug['drugname'])
            if drug_key in drug_created:
                continue
            
            drug_node = Node("Drug", primaryid=drug['primaryid'], caseid=drug['caseid'])
            drug_node['drugname'] = drug['drugname']
            drug_node['prod_ai'] = drug['prod_ai']
            drug_node['dose_freq'] = drug['dose_freq']
            drug_node['dose_amt'] = drug['dose_amt']
            drug_node['dose_unit'] = drug['dose_unit']
            self.graph.create(drug_node)
            drug_created.add(drug_key)

            # 创建 DrugSet -> Drug 关系
            drugset_node = self.graph.nodes.match("DrugSet", 
                                                primaryid=drug['primaryid'],
                                                caseid=drug['caseid']).first()
            if drugset_node:
                contains_drug_rel = Relationship(drugset_node, "CONTAINS_DRUG", drug_node)
                self.graph.create(contains_drug_rel)

        # 创建不良反应节点和 DrugSet -> Reaction 关系（去重处理）
        reaction_created = set()
        for reaction in tqdm(self.reactions, ncols=80, desc="创建不良反应节点与关系"):
            if reaction['primaryid'] not in self.valid_primaryids:
                continue
            
            # 检查是否已经存在相同的不良反应（primaryid + caseid + reac）
            reaction_key = (reaction['primaryid'], reaction['caseid'], reaction['reac'])
            if reaction_key in reaction_created:
                continue
            
            reaction_node = Node("Reaction", primaryid=reaction['primaryid'], caseid=reaction['caseid'])
            reaction_node['reac'] = reaction['reac']
            self.graph.create(reaction_node)
            reaction_created.add(reaction_key)

            # 创建 DrugSet -> Reaction 关系
            drugset_node = self.graph.nodes.match("DrugSet", 
                                                primaryid=reaction['primaryid'],
                                                caseid=reaction['caseid']).first()
            if drugset_node:
                causes_reaction_rel = Relationship(drugset_node, "CAUSES_REACTION", reaction_node)
                self.graph.create(causes_reaction_rel)

        # 创建患者结果节点和 Patient -> Outcome 关系（去重处理）
        outcome_created = set()
        for outcome in tqdm(self.outcomes, ncols=80, desc="创建患者结果节点与关系"):
            if outcome['primaryid'] not in self.valid_primaryids:
                continue
            
            # 检查是否已经存在相同的患者结果（primaryid + caseid + outccode）
            outcome_key = (outcome['primaryid'], outcome['caseid'], outcome['outccode'])
            if outcome_key in outcome_created:
                continue
            
            outcome_node = Node("Outcome", primaryid=outcome['primaryid'], caseid=outcome['caseid'])
            outcome_node['outccode'] = outcome['outccode']
            self.graph.create(outcome_node)
            outcome_created.add(outcome_key)

            # 创建 Patient -> Outcome 关系
            patient_node = self.graph.nodes.match("Patient", 
                                                primaryid=outcome['primaryid'],
                                                caseid=outcome['caseid']).first()
            if patient_node:
                has_outcome_rel = Relationship(patient_node, "HAS_OUTCOME", outcome_node)
                self.graph.create(has_outcome_rel)

        # 创建适应症节点和 DrugSet -> Indication 关系（去重处理）
        indication_created = set()
        for indication in tqdm(self.indications, ncols=80, desc="创建适应症节点与关系"):
            if indication['primaryid'] not in self.valid_primaryids:
                continue
            
            # 检查是否已经存在相同的适应症（primaryid + caseid + indi）
            indication_key = (indication['primaryid'], indication['caseid'], indication['indi'])
            if indication_key in indication_created:
                continue
            
            indication_node = Node("Indication", primaryid=indication['primaryid'], caseid=indication['caseid'])
            indication_node['indi'] = indication['indi']
            self.graph.create(indication_node)
            indication_created.add(indication_key)

            # 创建 DrugSet -> Indication 关系
            drugset_node = self.graph.nodes.match("DrugSet", 
                                                primaryid=indication['primaryid'],
                                                caseid=indication['caseid']).first()
            if drugset_node:
                treats_for_rel = Relationship(drugset_node, "TREATS_FOR", indication_node)
                self.graph.create(treats_for_rel)

        print(f"图谱节点和关系构建完成，共过滤掉 {len(self.invalid_primaryids)} 条无效记录")


if __name__ == '__main__':
    path = "data_source/faers_ascii_2023Q2/ASCII/"
    extractor = MedicalExtractor()
    extractor.extract_triples(path)

