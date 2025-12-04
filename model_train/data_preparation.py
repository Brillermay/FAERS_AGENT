import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import json

class ReactionDataPreparator:
    def __init__(
        self, 
        csv_path, 
        max_samples=10000,
        high_freq_min=75,
        med_freq_min=10,
        low_freq_min=5,
        high_ratio=0.6,
        med_ratio=0.9,
        low_ratio=0.3,
        exclude_non_physiological=False,  # 🔥 添加这个参数
        test_size=0.2,  # 🔥 添加这个参数
        random_state=42  # 🔥 添加这个参数
    ):
        """
        Args:
            csv_path: CSV文件路径
            exclude_non_physiological: 是否排除非生理反应
            test_size: 验证集比例
            random_state: 随机种子
        """
        self.csv_path = csv_path
        self.max_samples = max_samples
        self.high_freq_min = high_freq_min
        self.med_freq_min = med_freq_min
        self.low_freq_min = low_freq_min
        self.high_ratio = high_ratio
        self.med_ratio = med_ratio
        self.low_ratio = low_ratio
        self.exclude_non_physiological = exclude_non_physiological  # 🔥 添加
        self.test_size = test_size  # 🔥 添加
        self.random_state = random_state  # 🔥 添加
        
        # 非生理反应排除列表
        self.non_physiological_reactions = {
            "off label use",
            "drug ineffective", 
            "product dose omission issue",
            "drug interaction",
            "product dose omission",
            "therapeutic use unknown",
            "product used for unknown indication"
        }
        
    def load_and_clean_data(self):
        """加载并清理数据"""
        print("📁 加载数据...")
        df = pd.read_csv(self.csv_path)
        print(f"原始数据: {len(df)} 条记录")
        
        # 解析labels_list
        def parse_labels(labels_str):
            try:
                if isinstance(labels_str, str):
                    labels = eval(labels_str)
                else:
                    labels = labels_str if labels_str else []
                return [l.strip().lower() for l in labels if l and str(l).strip()]
            except:
                return []
        
        df['labels_parsed'] = df['labels_list'].apply(parse_labels)
        
        # 过滤掉没有标签的样本
        df = df[df['labels_parsed'].apply(len) > 0].reset_index(drop=True)
        print(f"有标签的记录: {len(df)} 条")
        
        if self.exclude_non_physiological:
            print("🧹 过滤非生理反应...")
            def filter_physiological(labels):
                return [l for l in labels if l not in self.non_physiological_reactions]
            
            df['labels_filtered'] = df['labels_parsed'].apply(filter_physiological)
            df = df[df['labels_filtered'].apply(len) > 0].reset_index(drop=True)
            print(f"过滤后记录: {len(df)} 条")
            df['labels_final'] = df['labels_filtered']
        else:
            df['labels_final'] = df['labels_parsed']
        
        return df
    
    def get_reaction_stats(self, df):
        """获取反应统计信息"""
        all_reactions = []
        for labels in df['labels_final']:
            all_reactions.extend(labels)
        
        reaction_counts = Counter(all_reactions)
        total_reactions = len(reaction_counts)
        total_occurrences = sum(reaction_counts.values())
        
        print(f"📊 反应统计:")
        print(f"  - 唯一反应数: {total_reactions}")
        print(f"  - 总出现次数: {total_occurrences}")
        print(f"  - 平均每例反应数: {total_occurrences/len(df):.2f}")
        print(f"  - 只出现1次的反应: {sum(1 for c in reaction_counts.values() if c == 1)} ({sum(1 for c in reaction_counts.values() if c == 1)/total_reactions*100:.1f}%)")
        
        return reaction_counts
    
    def strategy_a_layered_sampling(self, reaction_counts, head_k=200, tail_k=300):
        """方案A: 分层采样"""
        print(f"🎯 方案A: Head-{head_k} + Tail-{tail_k} 采样")
        
        # 按频次排序
        sorted_reactions = reaction_counts.most_common()
        
        # Head: 最常见的reactions
        head_reactions = [r for r, c in sorted_reactions[:head_k]]
        
        # Tail: 从长尾中随机采样（排除head中已有的）
        tail_candidates = [r for r, c in sorted_reactions[head_k:] if c >= 2]  # 至少出现2次
        
        if len(tail_candidates) >= tail_k:
            np.random.seed(self.random_state)
            tail_reactions = list(np.random.choice(tail_candidates, tail_k, replace=False))
        else:
            tail_reactions = tail_candidates
            print(f"⚠️  长尾候选不足，实际选择: {len(tail_reactions)}")
        
        selected_reactions = head_reactions + tail_reactions
        
        print(f"✅ 方案A选择的反应数: {len(selected_reactions)}")
        print(f"  - Head: {len(head_reactions)} (频次覆盖: {sum(reaction_counts[r] for r in head_reactions)/sum(reaction_counts.values())*100:.1f}%)")
        print(f"  - Tail: {len(tail_reactions)}")
        
        return selected_reactions
    
    def strategy_b_frequency_stratified(self, reaction_counts, 
                                      high_freq_min=50, med_freq_min=10, low_freq_min=2,
                                      high_ratio=0.4, med_ratio=0.3, low_ratio=0.3, 
                                      total_target=500):
        """方案B: 频次分层"""
        print(f"🎯 方案B: 频次分层采样 (目标: {total_target})")
        
        # 按频次分层
        high_freq = [(r, c) for r, c in reaction_counts.items() if c >= high_freq_min]
        med_freq = [(r, c) for r, c in reaction_counts.items() if med_freq_min <= c < high_freq_min]  
        low_freq = [(r, c) for r, c in reaction_counts.items() if low_freq_min <= c < med_freq_min]
        
        print(f"  - 高频 (≥{high_freq_min}): {len(high_freq)}")
        print(f"  - 中频 ({med_freq_min}-{high_freq_min-1}): {len(med_freq)}")
        print(f"  - 低频 ({low_freq_min}-{med_freq_min-1}): {len(low_freq)}")
        
        # 按比例采样
        high_target = int(total_target * high_ratio)
        med_target = int(total_target * med_ratio)
        low_target = total_target - high_target - med_target
        
        np.random.seed(self.random_state)
        
        # 采样 - 高频按频次排序取前N个
        high_freq_sorted = sorted(high_freq, key=lambda x: -x[1])
        high_selected = [r for r, c in high_freq_sorted[:high_target]]
        
        # 中频随机采样
        if len(med_freq) >= med_target:
            med_reactions = [r for r, c in med_freq]
            selected_indices = np.random.choice(len(med_reactions), med_target, replace=False)
            med_selected = [med_reactions[i] for i in selected_indices]
        else:
            med_selected = [r for r, c in med_freq]
            
        # 低频随机采样
        if len(low_freq) >= low_target:
            low_reactions = [r for r, c in low_freq]
            selected_indices = np.random.choice(len(low_reactions), low_target, replace=False)
            low_selected = [low_reactions[i] for i in selected_indices]
        else:
            low_selected = [r for r, c in low_freq]
        
        selected_reactions = high_selected + med_selected + low_selected
        
        print(f"✅ 方案B选择的反应数: {len(selected_reactions)}")
        print(f"  - 高频: {len(high_selected)}")
        print(f"  - 中频: {len(med_selected)}")
        print(f"  - 低频: {len(low_selected)}")
        
        return selected_reactions
    
    def prepare_training_data(self, df, selected_reactions, 
                            prioritize_rare=True, max_samples=3000):
        """准备训练数据（内存优化版：返回 label indices 而非稠密矩阵）"""
        print(f"🔧 准备训练数据（稀疏标签）...")
        
        # 过滤样本：只保留包含选中reactions的samples
        def has_selected_reaction(labels):
            return any(l in selected_reactions for l in labels)
        
        df_filtered = df[df['labels_final'].apply(has_selected_reaction)].copy()
        print(f"包含目标反应的样本: {len(df_filtered)}")
        
        # 重新过滤标签
        def filter_labels(labels):
            return [l for l in labels if l in selected_reactions]
        
        df_filtered['labels_final_filtered'] = df_filtered['labels_final'].apply(filter_labels)
        df_filtered = df_filtered[df_filtered['labels_final_filtered'].apply(len) > 0]
        
        # 优先保留包含稀有反应的样本
        if prioritize_rare and len(df_filtered) > max_samples:
            print(f"🎲 优先采样包含稀有反应的样本...")
            reaction_counts_selected = Counter()
            for labels in df_filtered['labels_final_filtered']:
                reaction_counts_selected.update(labels)
            
            # 计算每个样本的稀有度分数（包含的reaction频次倒数之和）
            def rarity_score(labels):
                return sum(1.0 / reaction_counts_selected[l] for l in labels)
            
            df_filtered['rarity_score'] = df_filtered['labels_final_filtered'].apply(rarity_score)
            
            # 按稀有度分层采样
            df_sorted = df_filtered.sort_values('rarity_score', ascending=False)
            
            # 取top 70%高稀有度 + 30%随机
            high_rare_n = int(max_samples * 0.7)
            random_n = max_samples - high_rare_n
            
            high_rare_samples = df_sorted.head(high_rare_n)
            remaining_samples = df_sorted.iloc[high_rare_n:]
            
            if len(remaining_samples) >= random_n:
                random_samples = remaining_samples.sample(n=random_n, random_state=self.random_state)
            else:
                random_samples = remaining_samples
            
            df_final = pd.concat([high_rare_samples, random_samples]).drop(columns=['rarity_score'])
        else:
            df_final = df_filtered.sample(n=min(len(df_filtered), max_samples), 
                                        random_state=self.random_state)
        
        print(f"最终训练样本数: {len(df_final)}")
        
        # Fit MultiLabelBinarizer but 不 transform 为稠密矩阵
        mlb = MultiLabelBinarizer()
        mlb.fit(df_final['labels_final_filtered'])
        label_to_idx = {label: idx for idx, label in enumerate(mlb.classes_)}
        
        # 为每个样本构造 label indices（稀疏存储）
        df_final['label_indices'] = df_final['labels_final_filtered'].apply(
            lambda ls: [label_to_idx[l] for l in ls]
        )
        
        print(f"标签空间维度: {len(mlb.classes_)}")
        # 计算平均每样本标签数
        avg_labels_per_sample = df_final['label_indices'].apply(len).mean()
        print(f"平均每样本标签数: {avg_labels_per_sample:.2f}")
        
        # 划分训练/验证集 (按primaryid分割，避免泄露)
        unique_ids = df_final['primaryid'].unique()
        train_ids, val_ids = train_test_split(unique_ids, 
                                            test_size=self.test_size, 
                                            random_state=self.random_state)
        
        train_mask = df_final['primaryid'].isin(train_ids)
        val_mask = df_final['primaryid'].isin(val_ids)
        
        train_data = {
            'prompts': df_final[train_mask]['prompt'].tolist(),
            'labels': df_final[train_mask]['label_indices'].tolist(),
            'primaryids': df_final[train_mask]['primaryid'].tolist()
        }
        
        val_data = {
            'prompts': df_final[val_mask]['prompt'].tolist(),
            'labels': df_final[val_mask]['label_indices'].tolist(),
            'primaryids': df_final[val_mask]['primaryid'].tolist()
        }
        
        print(f"训练集: {len(train_data['prompts'])} 样本")
        print(f"验证集: {len(val_data['prompts'])} 样本")
        
        return train_data, val_data, mlb, selected_reactions

    def select_reactions(self):
        """按新策略选择反应"""
        import random
        random.seed(42)
        
        # 按频率分组
        high_freq = [r for r, c in self.reaction_counts.items() if c >= self.high_freq_min]
        med_freq = [r for r, c in self.reaction_counts.items() if self.med_freq_min <= c < self.high_freq_min]
        low_freq = [r for r, c in self.reaction_counts.items() if self.low_freq_min <= c < self.med_freq_min]
        
        print(f"\n📊 频率分布:")
        print(f"  高频 (>={self.high_freq_min}): {len(high_freq)}")
        print(f"  中频 ({self.med_freq_min}-{self.high_freq_min-1}): {len(med_freq)}")
        print(f"  低频 ({self.low_freq_min}-{self.med_freq_min-1}): {len(low_freq)}")
        
        # 按比例采样
        num_high = int(len(high_freq) * self.high_ratio)
        num_med = int(len(med_freq) * self.med_ratio)
        num_low = int(len(low_freq) * self.low_ratio)
        
        selected_high = random.sample(high_freq, num_high) if num_high > 0 else []
        selected_med = random.sample(med_freq, num_med) if num_med > 0 else []
        selected_low = random.sample(low_freq, num_low) if num_low > 0 else []
        
        self.selected_reactions = selected_high + selected_med + selected_low
        
        print(f"\n✅ 选择结果:")
        print(f"  高频: {len(selected_high)} ({self.high_ratio*100:.0f}%)")
        print(f"  中频: {len(selected_med)} ({self.med_ratio*100:.0f}%)")
        print(f"  低频: {len(selected_low)} ({self.low_ratio*100:.0f}%)")
        print(f"  总计: {len(self.selected_reactions)}")


def prepare_data_strategy_a(csv_path, max_samples=3000, head_k=200, tail_k=300, **kwargs):
    """方案A数据准备的便捷函数"""
    # 修复：移除max_samples从__init__参数中，在prepare_training_data中使用
    preparator = ReactionDataPreparator(csv_path, **kwargs)
    df = preparator.load_and_clean_data()
    reaction_counts = preparator.get_reaction_stats(df)
    selected_reactions = preparator.strategy_a_layered_sampling(reaction_counts, head_k=head_k, tail_k=tail_k)
    return preparator.prepare_training_data(df, selected_reactions, max_samples=max_samples)


def prepare_data_strategy_b(
    csv_path,
    max_samples=10000,
    high_freq_min=75,
    med_freq_min=10,
    low_freq_min=5,
    high_ratio=0.6,
    med_ratio=0.9,
    low_ratio=0.3,
    exclude_non_physiological=False,  # 🔥 添加
    test_size=0.2,  # 🔥 添加
    random_state=42  # 🔥 添加
):
    """策略B：按比例从各频段采样"""
    
    preparator = ReactionDataPreparator(
        csv_path=csv_path,
        max_samples=max_samples,
        high_freq_min=high_freq_min,
        med_freq_min=med_freq_min,
        low_freq_min=low_freq_min,
        high_ratio=high_ratio,
        med_ratio=med_ratio,
        low_ratio=low_ratio,
        exclude_non_physiological=exclude_non_physiological,  # 🔥 传递
        test_size=test_size,  # 🔥 传递
        random_state=random_state  # 🔥 传递
    )
    
    df = preparator.load_and_clean_data()
    reaction_counts = preparator.get_reaction_stats(df)
    
    # 按频率分组
    high_freq_labels = [label for label, count in reaction_counts.items() if count >= high_freq_min]
    med_freq_labels = [label for label, count in reaction_counts.items() if med_freq_min <= count < high_freq_min]
    low_freq_labels = [label for label, count in reaction_counts.items() if low_freq_min <= count < med_freq_min]
    
    print(f"\n📊 标签频率分布:")
    print(f"  高频标签 (>={high_freq_min}): {len(high_freq_labels)} 个")
    print(f"  中频标签 ({med_freq_min}-{high_freq_min-1}): {len(med_freq_labels)} 个")
    print(f"  低频标签 ({low_freq_min}-{med_freq_min-1}): {len(low_freq_labels)} 个")
    
    # 按比例采样
    num_high = int(len(high_freq_labels) * high_ratio)
    num_med = int(len(med_freq_labels) * med_ratio)
    num_low = int(len(low_freq_labels) * low_ratio)
    
    import random
    random.seed(random_state)
    
    selected_high = random.sample(high_freq_labels, num_high) if num_high > 0 and len(high_freq_labels) >= num_high else high_freq_labels
    selected_med = random.sample(med_freq_labels, num_med) if num_med > 0 and len(med_freq_labels) >= num_med else med_freq_labels
    selected_low = random.sample(low_freq_labels, num_low) if num_low > 0 and len(low_freq_labels) >= num_low else low_freq_labels
    
    selected_reactions = selected_high + selected_med + selected_low
    
    print(f"\n✅ 选择的标签:")
    print(f"  高频: {len(selected_high)} / {len(high_freq_labels)} ({high_ratio*100:.0f}%)")
    print(f"  中频: {len(selected_med)} / {len(med_freq_labels)} ({med_ratio*100:.0f}%)")
    print(f"  低频: {len(selected_low)} / {len(low_freq_labels)} ({low_ratio*100:.0f}%)")
    print(f"  总计: {len(selected_reactions)} 个标签")
    
    return preparator.prepare_training_data(df, selected_reactions, max_samples=max_samples)


if __name__ == "__main__":
    # 测试
    csv_path = "../outputs/prompts_sample_10000_coarse_coarse_v2.csv"  # 修正路径
    
    print("=== 方案A测试 ===")
    train_a, val_a, mlb_a, reactions_a = prepare_data_strategy_a(csv_path)
    
    print("\n=== 方案B测试 ===") 
    train_b, val_b, mlb_b, reactions_b = prepare_data_strategy_b(csv_path)