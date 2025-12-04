import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, 
    Trainer, TrainingArguments,
)
from torch.utils.data import Dataset
import pickle
import json
from tqdm import tqdm
import random
from data_preparation import prepare_data_strategy_b


# ============= 🔥 核心1: 交叉注意力模型架构 =============
class CrossAttentionReactionModel(nn.Module):
    """
    交叉注意力架构:
    1. 编码病例 (患者+适应症+药物)
    2. 编码所有反应名称 (预计算)
    3. 交叉注意力: 病例关注反应,学习关联
    4. 分类头输出所有反应的概率
    """
    def __init__(self, model_name, num_labels, reaction_names, dropout_rate=0.1):
        super().__init__()
        
        # BERT编码器
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 通常是768
        
        self.num_labels = num_labels
        self.reaction_names = reaction_names
        
        # 🔥 核心改进1: 多头交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 🔥 核心改进2: 反应感知的分类头
        # 输入: 病例表示 + 注意力输出 (拼接)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_labels)  # 直接输出700个反应的logits
        )
        
        # 🔥 预计算并存储所有反应的嵌入
        self.register_buffer('reaction_embeddings', torch.zeros(num_labels, hidden_size))
        self._initialized = False
        
    def initialize_reaction_embeddings(self, tokenizer, device, batch_size=32):
        """
        预先编码所有700个反应名称
        只需在训练开始时调用一次
        """
        if self._initialized:
            return
        
        print("🔧 预计算反应嵌入...")
        self.eval()
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.reaction_names), batch_size), desc="编码反应"):
                batch_reactions = self.reaction_names[i:i+batch_size]
                
                encoding = tokenizer(
                    batch_reactions,
                    truncation=True,
                    padding=True,
                    max_length=64,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                all_embeddings.append(embeddings.cpu())
        
        self.reaction_embeddings = torch.cat(all_embeddings, dim=0).to(device)
        self._initialized = True
        print(f"✅ 反应嵌入矩阵: {self.reaction_embeddings.shape}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        
        Args:
            input_ids: (B, seq_len) 病例文本
            attention_mask: (B, seq_len)
            labels: (B, num_labels) 多标签目标 [0或1]
        
        Returns:
            dict: {'loss': loss, 'logits': logits}
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Step 1: 编码病例
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, seq_len, hidden_size)
        case_cls = sequence_output[:, 0, :]  # (B, hidden_size)
        
        # Step 2: 交叉注意力 - 病例如何关注每个反应
        # query: 病例的[CLS]表示
        # key/value: 所有反应的嵌入
        
        query = case_cls.unsqueeze(1)  # (B, 1, hidden_size)
        
        # 扩展反应嵌入到batch维度
        reaction_emb = self.reaction_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  
        # (B, num_labels, hidden_size)
        
        # 🔥 交叉注意力计算
        attended_output, attention_weights = self.cross_attention(
            query=query,           # (B, 1, hidden_size) - 病例询问
            key=reaction_emb,      # (B, num_labels, hidden_size) - 反应作为key
            value=reaction_emb     # (B, num_labels, hidden_size) - 反应作为value
        )
        # attended_output: (B, 1, hidden_size) - 加权融合的反应信息
        attended_output = attended_output.squeeze(1)  # (B, hidden_size)
        
        # Step 3: 拼接原始病例表示和注意力输出
        combined = torch.cat([case_cls, attended_output], dim=1)  # (B, hidden_size * 2)
        
        # Step 4: 分类得到所有反应的logits
        logits = self.classifier(combined)  # (B, num_labels)
        
        # Step 5: 计算损失
        loss = None
        if labels is not None:
            loss_fct = FocalLoss(alpha=0.25, gamma=2.0)
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}


# ============= 🔥 核心2: Focal Loss实现 =============
class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-label Classification
    
    FL(p_t) = -α(1-p_t)^γ * log(p_t)
    
    - α: 平衡正负样本权重
    - γ: 调节难易样本权重 (γ越大,简单样本权重越低)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (B, num_labels) - 模型输出
            labels: (B, num_labels) - 二值标签 [0或1]
        """
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 🔥 核心公式: p_t = p if y==1 else 1-p
        p_t = probs * labels + (1 - probs) * (1 - labels)
        
        # 🔥 Focal weight: (1 - p_t)^gamma
        # 当p_t接近1(简单样本),权重接近0
        # 当p_t接近0(难样本),权重接近1
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        
        # 🔥 最终损失 = alpha * focal_weight * bce_loss
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()


# ============= 🔥 核心3: 简化的多标签数据集 =============
class MultiLabelDataset(Dataset):
    """
    不需要负采样,直接返回多标签向量
    """
    def __init__(self, prompts, label_indices, num_labels):
        """
        Args:
            prompts: 病例文本列表
            label_indices: 每个样本的标签索引列表
            num_labels: 总标签数 (700)
        """
        self.prompts = prompts
        self.label_indices = label_indices
        self.num_labels = num_labels
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        labels = self.label_indices[idx]
        
        # 转换为多标签向量
        label_vec = np.zeros(self.num_labels, dtype=np.float32)
        
        if labels is not None and len(labels) > 0:
            # 兼容不同格式
            labels = list(labels) if not isinstance(labels, list) else labels
            valid_labels = [i for i in labels if 0 <= i < self.num_labels]
            
            if valid_labels:
                label_vec[valid_labels] = 1.0
        
        return {
            'prompt': prompt,
            'labels': label_vec
        }


# ============= 🔥 核心4: 简化的Collator =============
def multilabel_collate_fn(batch, tokenizer, max_length=512):
    """
    将batch转换为模型输入
    """
    prompts = [b['prompt'] for b in batch]
    labels = torch.tensor([b['labels'] for b in batch], dtype=torch.float32)
    
    # Tokenize病例
    encoding = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': labels
    }


# ============= 🔥 核心5: 标准Trainer =============
class MultiLabelTrainer(Trainer):
    """使用标准的Trainer,不需要特殊处理"""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


# ============= 🔥 核心6: 评估指标 =============
def compute_multilabel_metrics(eval_pred, all_reaction_names):
    """
    计算多标签分类指标
    
    Args:
        eval_pred: EvalPrediction对象
        all_reaction_names: 所有反应名称列表
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Sigmoid转换为概率
    probs = 1 / (1 + np.exp(-logits))
    
    # 计算不同K值的指标
    k_values = [1, 3, 5, 10, 20]
    metrics = {}
    
    for k in k_values:
        # 获取Top-K预测
        top_k_indices = np.argsort(-probs, axis=1)[:, :k]
        
        recalls = []
        precisions = []
        f1s = []
        
        for i in range(len(labels)):
            true_labels = set(np.where(labels[i] == 1)[0])
            pred_labels = set(top_k_indices[i])
            
            if len(true_labels) == 0:
                continue
            
            # Recall@K
            recall = len(true_labels & pred_labels) / len(true_labels)
            recalls.append(recall)
            
            # Precision@K
            precision = len(true_labels & pred_labels) / k if k > 0 else 0
            precisions.append(precision)
            
            # F1@K
            if recall + precision > 0:
                f1 = 2 * recall * precision / (recall + precision)
                f1s.append(f1)
        
        metrics[f'recall_at_{k}'] = np.mean(recalls) if recalls else 0.0
        metrics[f'precision_at_{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'f1_at_{k}'] = np.mean(f1s) if f1s else 0.0
    
    # 计算MRR
    mrrs = []
    for i in range(len(labels)):
        true_labels = set(np.where(labels[i] == 1)[0])
        if len(true_labels) == 0:
            continue
        
        sorted_indices = np.argsort(-probs[i])
        for rank, idx in enumerate(sorted_indices, 1):
            if idx in true_labels:
                mrrs.append(1.0 / rank)
                break
    
    metrics['mrr'] = np.mean(mrrs) if mrrs else 0.0
    
    return metrics


# ============= 配置 =============
CROSSATT_CONFIG = {
    'csv_path': './outputs/prompts_sample_10000_detailed_detailed_v2.csv',
    'max_samples': 10000,
    'high_freq_min': 75,
    'med_freq_min': 10,
    'low_freq_min': 5,
    'high_ratio': 0.6,
    'med_ratio': 0.9,
    'low_ratio': 0.3,
    
    'model_name': '/home/motao/project/models/Bio_ClinicalBERT_local',
    'max_length': 512,
    'dropout_rate': 0.15,
    
    'epochs': 8,
    'batch_size': 8,  # 可以更大,因为不需要负采样
    'learning_rate': 3e-5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 2,
    
    'output_dir': '/home/motao/project/strategy_crossatt_output',
}


# ============= 主训练类 =============
class CrossAttentionTrainer:
    def __init__(self, config):
        self.config = config
        os.makedirs(config['output_dir'], exist_ok=True)
    
    def prepare_data(self):
        print("🔧 准备数据...")
        
        train_data, val_data, mlb, selected_reactions = prepare_data_strategy_b(
            self.config['csv_path'],
            max_samples=self.config['max_samples'],
            high_freq_min=self.config['high_freq_min'],
            med_freq_min=self.config['med_freq_min'],
            low_freq_min=self.config['low_freq_min'],
            high_ratio=self.config['high_ratio'],
            med_ratio=self.config['med_ratio'],
            low_ratio=self.config['low_ratio']
        )
        
        self.train_data = train_data
        self.val_data = val_data
        self.mlb = mlb
        self.all_reaction_names = list(mlb.classes_)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ 数据准备完成 - 反应数: {len(self.all_reaction_names)}")
    
    def create_datasets(self):
        print("🔧 创建数据集...")
        
        if isinstance(self.train_data, dict):
            train_prompts = self.train_data.get('prompts', self.train_data.get('texts', []))
            train_labels = self.train_data.get('labels', self.train_data.get('label_indices', []))
            
            val_prompts = self.val_data.get('prompts', self.val_data.get('texts', []))
            val_labels = self.val_data.get('labels', self.val_data.get('label_indices', []))
        else:
            raise ValueError(f"不支持的数据格式")
        
        self.train_dataset = MultiLabelDataset(
            train_prompts, train_labels, len(self.all_reaction_names)
        )
        
        self.val_dataset = MultiLabelDataset(
            val_prompts, val_labels, len(self.all_reaction_names)
        )
        
        print(f"✅ 训练集: {len(self.train_dataset)}, 验证集: {len(self.val_dataset)}")
    
    def train(self):
        print("🚀 开始交叉注意力训练...")
        
        self.prepare_data()
        self.create_datasets()
        
        # 初始化模型
        model = CrossAttentionReactionModel(
            model_name=self.config['model_name'],
            num_labels=len(self.all_reaction_names),
            reaction_names=self.all_reaction_names,
            dropout_rate=self.config['dropout_rate']
        )
        
        # 🔥 预计算反应嵌入
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.initialize_reaction_embeddings(self.tokenizer, device)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'] * 2,
            learning_rate=self.config['learning_rate'],
            warmup_ratio=self.config['warmup_ratio'],
            weight_decay=self.config['weight_decay'],
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=2,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="recall_at_5",
            greater_is_better=True,
            report_to=None,
            fp16=True,
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            remove_unused_columns=False,
        )
        
        def data_collator(batch):
            return multilabel_collate_fn(batch, self.tokenizer, self.config['max_length'])
        
        # 🔥 使用标准Trainer
        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_multilabel_metrics(p, self.all_reaction_names),
        )
        
        print("训练开始...")
        trainer.train()
        
        print("\n💾 保存最终模型...")
        trainer.save_model(os.path.join(self.config['output_dir'], "final_model"))
        
        print("\n🎯 最终评估...")
        final_metrics = trainer.evaluate()
        
        # 打印详细结果
        print("\n" + "="*50)
        print("📊 最终评估结果")
        print("="*50)
        
        print("\n🎯 Recall@K:")
        for k in [1, 3, 5, 10, 15]:
            print(f"  Recall@{k:2d}: {final_metrics.get(f'eval_recall_at_{k}', 0):.4f}")
        
        print("\n📌 Precision@K:")
        for k in [1, 3, 5, 10, 15]:
            print(f"  Precision@{k:2d}: {final_metrics.get(f'eval_precision_at_{k}', 0):.4f}")
        
        print("\n🌟 F1@K:")
        for k in [1, 3, 5, 10, 15]:
            print(f"  F1@{k:2d}: {final_metrics.get(f'eval_f1_at_{k}', 0):.4f}")
        
        print(f"\n⭐ MRR: {final_metrics.get('eval_mrr', 0):.4f}")
        print("="*50 + "\n")
        
        # 保存结果
        with open(os.path.join(self.config['output_dir'], "eval_results.json"), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        with open(os.path.join(self.config['output_dir'], "metadata.pkl"), 'wb') as f:
            pickle.dump({
                'mlb': self.mlb,
                'all_reaction_names': self.all_reaction_names,
                'config': self.config,
                'eval_metrics': final_metrics,
            }, f)
        
        print("✅ 训练完成!")
        return final_metrics


def main():
    print("=" * 50)
    print("🎯 交叉注意力架构 (端到端多标签分类)")
    print("=" * 50)
    
    trainer = CrossAttentionTrainer(CROSSATT_CONFIG)
    trainer.train()


if __name__ == "__main__":
    main()