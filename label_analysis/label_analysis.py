import os
import re
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from py2neo import Graph

class ReactionAnalyzer:
    def __init__(self, uri=None, user=None, password=None):
        # 与 build_kg_utils.py 保持一致的默认配置
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "12345678")
        self.graph = Graph(uri, auth=(user, password))

    # --- 数据拉取 ---------------------------------------------------
    def fetch_reaction_counts(self):
        q = """
        MATCH (ds:DrugSet)-[:CAUSES_REACTION]->(r:Reaction)
        RETURN r.reac AS reaction, COUNT(*) AS freq
        ORDER BY freq DESC
        """
        return pd.DataFrame(self.graph.run(q).data())

    def fetch_case_reaction_counts(self):
        q = """
        MATCH (ds:DrugSet)
        OPTIONAL MATCH (ds)-[:CAUSES_REACTION]->(r:Reaction)
        WITH ds.primaryid AS primaryid, ds.caseid AS caseid, COLLECT(DISTINCT r.reac) AS reacs
        RETURN primaryid, caseid, SIZE([x IN reacs WHERE x IS NOT NULL]) AS num_reactions
        """
        return pd.DataFrame(self.graph.run(q).data())

    def fetch_unique_reactions(self):
        q = "MATCH (r:Reaction) RETURN DISTINCT r.reac AS reaction"
        return [row['reaction'] for row in self.graph.run(q)]

    # --- 分析 -----------------------------------------------------
    def analyze_reaction_distribution(self, rc_df, top_k_show=20):
        total = int(rc_df['freq'].sum())
        rc_df['cum_freq'] = rc_df['freq'].cumsum()
        rc_df['cum_pct'] = rc_df['cum_freq'] / total * 100

        print("=== Reaction 标签分布 ===")
        print(f"总不良反应记录数: {total}")
        print(f"唯一不良反应类型数: {rc_df.shape[0]}")
        print("\nTop {} 不良反应:".format(top_k_show))
        print(rc_df.head(top_k_show).to_string(index=False))

        # 长尾：找出覆盖80%需要的label个数
        top80_n = int(rc_df[rc_df['cum_pct'] <= 80].shape[0]) + 1
        print(f"\n前 {top80_n} 种不良反应覆盖约80%记录 (近似)")

        singletons = (rc_df['freq'] == 1).sum()
        print(f"只出现1次的不良反应数量: {singletons} ({singletons/rc_df.shape[0]*100:.2f}%)")

        return rc_df

    def analyze_multi_label_stats(self, cases_df):
        if cases_df.empty:
            print("case 数据为空")
            return None

        arr = cases_df['num_reactions'].to_numpy(dtype=int)
        print("\n=== 多标签统计 (按 DrugSet / case) ===")
        print(f"case 总数: {arr.size}")
        print(f"平均每个 case 的 reaction 数量: {arr.mean():.2f}")
        print(f"中位数: {np.median(arr):.1f}")
        print(f"标准差: {arr.std():.2f}")
        print(f"最小: {arr.min()}, 最大: {arr.max()}")

        distr = Counter(arr)
        print("\n前10 种 reaction 数量分布 (count -> cases):")
        for k in sorted(distr.keys())[:10]:
            print(f"  {k} -> {distr[k]} cases ({distr[k]/arr.size*100:.2f}%)")
        return arr, distr

    def analyze_label_standardization(self, unique_reacs, sample_show=50):
        print("\n=== 标签标准化检查 ===")
        patterns = {
            'has_numbers': 0,
            'all_caps': 0,
            'mixed_case': 0,
            'has_special_chars': 0,
            'very_long': 0,
            'very_short': 0,
            'has_spaces': 0,
        }
        for s in unique_reacs:
            s = s or ""
            if re.search(r'\d', s): patterns['has_numbers'] += 1
            if s.isupper(): patterns['all_caps'] += 1
            elif not s.islower(): patterns['mixed_case'] += 1
            if re.search(r'[^\w\s]', s): patterns['has_special_chars'] += 1
            if len(s) > 80: patterns['very_long'] += 1
            if len(s) < 3: patterns['very_short'] += 1
            if ' ' in s: patterns['has_spaces'] += 1

        total_unique = len(unique_reacs)
        print(f"唯一 reaction 数量: {total_unique}")
        for k, v in patterns.items():
            print(f"  {k}: {v} ({v/total_unique*100:.2f}%)")

        # 简单的归一化示例与相似样例展示
        def normalize_label(x):
            x = (x or "").strip().lower()
            x = re.sub(r'[^\w\s]', ' ', x)
            x = re.sub(r'\s+', ' ', x).strip()
            return x

        norm_map = {}
        for orig in unique_reacs:
            n = normalize_label(orig)
            norm_map.setdefault(n, []).append(orig)

        print("\n示例：需要合并的归一化组（最多显示前 {} 组）:".format(sample_show))
        shown = 0
        for k, vals in norm_map.items():
            if len(vals) > 1:
                print(f"  '{k}' => {vals[:5]}")
                shown += 1
                if shown >= sample_show:
                    break

        return patterns, norm_map

    def analyze_physiological_reactions(self, exclude=None, top_k=20, save_fig=False, out_path="physio_reaction_analysis.png"):
        """
        专门分析“生理反应类”标签分布。
        规则：默认把以下三类视为非生理反应并排除：
          - 'Drug ineffective'
          - 'Off label use'
          - 'Product dose omission issue'
        生理反应 = 所有不在 exclude 列表中的 reaction 文本。

        返回值：dict 包含 rc_df (reaction 频次表) 和 case_stats (按 DrugSet 的生理 reaction 数量数组与分布)
        """
        if exclude is None:
            exclude = ["Drug ineffective", "Off label use", "Product dose omission issue"]

        # 拉取排除特定标签后的 reaction 频次
        q = """
        MATCH (ds:DrugSet)-[:CAUSES_REACTION]->(r:Reaction)
        WHERE NOT r.reac IN $excl
        RETURN r.reac AS reaction, COUNT(*) AS freq
        ORDER BY freq DESC
        """
        rc_df = pd.DataFrame(self.graph.run(q, excl=exclude).data())
        if rc_df.empty:
            print("没有生理反应相关的数据（排除列表可能过宽）")
            return {'rc_df': rc_df, 'case_stats': None}

        # 基本分布统计
        total = int(rc_df['freq'].sum())
        rc_df['cum_freq'] = rc_df['freq'].cumsum()
        rc_df['cum_pct'] = rc_df['cum_freq'] / total * 100

        print("\n=== 生理反应 标签分布（排除: {}） ===".format(", ".join(exclude)))
        print(f"总记录数(排除后): {total}")
        print(f"唯一标签数(排除后): {rc_df.shape[0]}")
        print("\nTop {} 生理反应:".format(top_k))
        print(rc_df.head(top_k).to_string(index=False))

        # 长尾和单例统计
        top80_n = int(rc_df[rc_df['cum_pct'] <= 80].shape[0]) + 1
        singletons = (rc_df['freq'] == 1).sum()
        print(f"\n前 {top80_n} 种生理反应覆盖约80%记录 (近似)")
        print(f"只出现1次的生理反应数量: {singletons} ({singletons/rc_df.shape[0]*100:.2f}%)")

        # 按 DrugSet 统计每个 case 的生理 reaction 数量
        q2 = """
        MATCH (ds:DrugSet)
        OPTIONAL MATCH (ds)-[:CAUSES_REACTION]->(r:Reaction)
        WHERE NOT r.reac IN $excl
        WITH ds.primaryid AS primaryid, ds.caseid AS caseid, COLLECT(DISTINCT r.reac) AS reacs
        RETURN primaryid, caseid, SIZE([x IN reacs WHERE x IS NOT NULL]) AS num_physio_reactions
        """
        cases_df = pd.DataFrame(self.graph.run(q2, excl=exclude).data())
        if not cases_df.empty:
            arr = cases_df['num_physio_reactions'].to_numpy(dtype=int)
            print("\n=== 生理反应 多标签统计（按 DrugSet / case） ===")
            print(f"case 总数: {arr.size}")
            print(f"平均每个 case 的生理 reaction 数量: {arr.mean():.2f}")
            print(f"中位数: {np.median(arr):.1f}, 标准差: {arr.std():.2f}, 最小: {arr.min()}, 最大: {arr.max()}")

            # 前10 分布
            distr = Counter(arr)
            print("\n前10 种 生理 reaction 数量分布 (count -> cases):")
            for k in sorted(distr.keys())[:10]:
                print(f"  {k} -> {distr[k]} cases ({distr[k]/arr.size*100:.2f}%)")
        else:
            arr = None
            print("没有按 case 的生理 reaction 统计结果（可能全部被排除）")

        # 可视化（可选）
        if save_fig and (not rc_df.empty):
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Top 20 生理反应
            top20 = rc_df.head(20)
            axes[0].barh(top20['reaction'].astype(str), top20['freq'])
            axes[0].invert_yaxis()
            axes[0].set_title("Top 20 生理反应（排除特殊标签）")
            axes[0].set_xlabel("出现次数")

            # case 生理 reaction 数量分布（若可用）
            if arr is not None:
                axes[1].hist(arr, bins=range(0, int(arr.max())+2), edgecolor='black', alpha=0.7)
                axes[1].set_title("每个 case 的生理 reaction 数量分布")
                axes[1].set_xlabel("生理 reaction 数量")
                axes[1].set_ylabel("case 数")
            else:
                axes[1].text(0.5, 0.5, "无 case 数据", ha='center', va='center')

            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"生理反应可视化已保存: {out_path}")

        return {'rc_df': rc_df, 'case_stats': {'arr': arr, 'distr': (Counter(arr) if arr is not None else None)}}

    # --- 可视化 ---------------------------------------------------
    def generate_visualizations(self, rc_df, case_arr, out_path="reaction_analysis.png"):
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top 20
        top20 = rc_df.head(20)
        axes[0, 0].barh(top20['reaction'].astype(str), top20['freq'])
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_title("Top 20 不良反应")
        axes[0, 0].set_xlabel("出现次数")

        # 长尾（log）
        counts = rc_df['freq'].tolist()[:200]
        axes[0, 1].plot(range(1, len(counts) + 1), counts)
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title("不良反应长尾分布 (前200)")
        axes[0, 1].set_xlabel("排名")

        # case reaction count hist
        axes[1, 0].hist(case_arr, bins=range(0, int(case_arr.max())+2), edgecolor='black', alpha=0.7)
        axes[1, 0].set_title("每个 case 的 reaction 数量分布")
        axes[1, 0].set_xlabel("reaction 数量")
        axes[1, 0].set_ylabel("case 数")

        # 累积分布图
        vals, counts2 = np.unique(case_arr, return_counts=True)
        cumsum = np.cumsum(counts2) / counts2.sum() * 100
        axes[1, 1].plot(vals, cumsum, marker='o')
        axes[1, 1].set_title("Case 中 reaction 数量累积百分比")
        axes[1, 1].set_xlabel("reaction 数量")
        axes[1, 1].set_ylabel("累积百分比 (%)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"可视化已保存: {out_path}")

    # --- 一体化运行 ------------------------------------------------
    def run_complete_analysis(self, save_fig=True):
        rc_df = self.fetch_reaction_counts()
        cases_df = self.fetch_case_reaction_counts()
        unique_reacs = self.fetch_unique_reactions()

        rc_df = self.analyze_reaction_distribution(rc_df)
        case_arr, distr = self.analyze_multi_label_stats(cases_df)
        patterns, norm_map = self.analyze_label_standardization(unique_reacs)

        if save_fig and (rc_df is not None) and (case_arr is not None):
            self.generate_visualizations(rc_df, case_arr)

        return {
            'reaction_counts_df': rc_df,
            'case_counts_df': cases_df,
            'unique_reactions': unique_reacs,
            'multi_label_array': case_arr,
            'standardization_patterns': patterns,
            'norm_map_sample': {k: v[:5] for k, v in list(norm_map.items())[:200]}
        }

if __name__ == '__main__':
    analyzer = ReactionAnalyzer()
    analyzer.analyze_physiological_reactions(save_fig=True)
    results = analyzer.run_complete_analysis(save_fig=True)
    print("\n分析完成。")