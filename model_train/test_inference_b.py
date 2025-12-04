import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import pickle
import numpy as np
from transformers import AutoTokenizer
from train_strategy_b import RetrievalReactionModel
from typing import List, Tuple


class RetrievalReactionPredictor:
    """检索式模型推理器"""
    
    def __init__(self, model_dir='/home/motao/project/strategy_retrieval_output/final_model'):
        """
        加载训练好的检索式模型
        
        Args:
            model_dir: 模型保存目录（包含 final_model 文件夹）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        print(f"📂 从 {model_dir} 加载模型...")
        
        # 加载元数据
        parent_dir = os.path.dirname(model_dir)  # 上级目录
        metadata_path = os.path.join(parent_dir, 'metadata.pkl')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"未找到元数据文件: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.mlb = metadata['mlb']
        self.all_reaction_names = metadata['all_reaction_names']
        self.config = metadata['config']
        
        print(f"📊 反应类别数: {len(self.all_reaction_names)}")
        print(f"🏷️  前5个反应: {self.all_reaction_names[:5]}")
        
        # 加载 tokenizer
        model_name = self.config['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"📦 Tokenizer 加载完成: {model_name}")
        
        # 初始化模型结构
        self.model = RetrievalReactionModel(
            model_name=model_name,
            num_labels=len(self.all_reaction_names),
            reaction_names=self.all_reaction_names,
            dropout_rate=self.config.get('dropout_rate', 0.1)
        )
        
        # 加载模型权重
        weight_files = [
            'model.safetensors',
            'pytorch_model.bin',
            'model.bin',
        ]
        
        loaded = False
        for weight_file in weight_files:
            weight_path = os.path.join(model_dir, weight_file)
            if os.path.exists(weight_path):
                print(f"📥 加载权重: {weight_file}")
                
                if weight_file.endswith('.safetensors'):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(weight_path)
                    except ImportError:
                        print("⚠️  safetensors 未安装，使用 torch 加载...")
                        state_dict = torch.load(weight_path, map_location=self.device)
                else:
                    state_dict = torch.load(weight_path, map_location=self.device)
                
                self.model.load_state_dict(state_dict, strict=False)
                loaded = True
                break
        
        if not loaded:
            actual_files = os.listdir(model_dir) if os.path.exists(model_dir) else []
            raise FileNotFoundError(f"未找到权重文件，目录内容: {actual_files}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 预计算所有反应的向量（用于快速推理）
        print("🔧 预计算反应向量...")
        self._precompute_reaction_embeddings()
        
        print("✅ 模型加载完成！\n")
    
    def _precompute_reaction_embeddings(self, batch_size=32):
        """
        预计算所有700个反应的向量表示
        推理时只需编码病例，然后做向量点积
        """
        all_vecs = []
        
        with torch.no_grad():
            for i in range(0, len(self.all_reaction_names), batch_size):
                batch_reactions = self.all_reaction_names[i:i+batch_size]
                
                # Tokenize
                encoding = self.tokenizer(
                    batch_reactions,
                    truncation=True,
                    padding=True,
                    max_length=64,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # 编码
                vecs = self.model.encode_reaction(input_ids, attention_mask)
                all_vecs.append(vecs.cpu())
        
        # 拼接所有向量: (num_labels, 256)
        self.reaction_embeddings = torch.cat(all_vecs, dim=0)
        print(f"✅ 反应向量矩阵: {self.reaction_embeddings.shape}")
    
    def predict(self, prompt: str, top_k: int = 10, return_scores: bool = True) -> List[Tuple[str, float]]:
        """
        对单个病例进行预测
        
        Args:
            prompt: 病例描述文本
            top_k: 返回前K个预测
            return_scores: 是否返回相似度分数
            
        Returns:
            [(反应名称, 相似度分数), ...] 或 [反应名称, ...]
        """
        # Tokenize 病例
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # 编码病例: (1, 256)
            case_vec = self.model.encode_case(input_ids, attention_mask)
            
            # 计算与所有反应的相似度: (1, num_labels)
            scores = torch.mm(case_vec, self.reaction_embeddings.T.to(self.device))
            scores = scores.cpu().numpy()[0]  # (num_labels,)
        
        # 排序获取 Top-K
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for idx in top_indices:
            reaction_name = self.all_reaction_names[idx]
            score = float(scores[idx])
            
            if return_scores:
                results.append((reaction_name, score))
            else:
                results.append(reaction_name)
        
        return results
    
    def predict_batch(self, prompts: List[str], top_k: int = 10, 
                     batch_size: int = 16, return_scores: bool = True) -> List[List[Tuple[str, float]]]:
        """
        批量预测（更高效）
        
        Args:
            prompts: 病例列表
            top_k: 每个病例返回前K个预测
            batch_size: 批处理大小
            return_scores: 是否返回分数
            
        Returns:
            每个病例的预测结果列表
        """
        all_results = []
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                
                # Tokenize
                encoding = self.tokenizer(
                    batch_prompts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # 编码病例: (B, 256)
                case_vecs = self.model.encode_case(input_ids, attention_mask)
                
                # 计算相似度: (B, num_labels)
                scores = torch.mm(case_vecs, self.reaction_embeddings.T.to(self.device))
                scores = scores.cpu().numpy()
                
                # 对每个样本提取 Top-K
                for sample_scores in scores:
                    top_indices = np.argsort(-sample_scores)[:top_k]
                    
                    sample_results = []
                    for idx in top_indices:
                        reaction_name = self.all_reaction_names[idx]
                        score = float(sample_scores[idx])
                        
                        if return_scores:
                            sample_results.append((reaction_name, score))
                        else:
                            sample_results.append(reaction_name)
                    
                    all_results.append(sample_results)
        
        return all_results
    
    def explain_prediction(self, prompt: str, reaction_name: str) -> dict:
        """
        解释为什么预测某个反应（返回相似度分数和排名）
        
        Args:
            prompt: 病例描述
            reaction_name: 要解释的反应名称
            
        Returns:
            {'score': 相似度分数, 'rank': 排名, 'percentile': 百分位}
        """
        if reaction_name not in self.all_reaction_names:
            raise ValueError(f"反应 '{reaction_name}' 不在模型词表中")
        
        # 获取所有相似度
        predictions = self.predict(prompt, top_k=len(self.all_reaction_names), return_scores=True)
        
        # 查找目标反应
        for rank, (pred_reaction, score) in enumerate(predictions, 1):
            if pred_reaction == reaction_name:
                percentile = (1 - rank / len(self.all_reaction_names)) * 100
                return {
                    'score': score,
                    'rank': rank,
                    'percentile': percentile,
                    'total_reactions': len(self.all_reaction_names)
                }
        
        return {'error': f'未找到反应 {reaction_name}'}


def main():
    """测试推理"""
    print("=" * 60)
    print("🎯 检索式模型推理测试")
    print("=" * 60 + "\n")
    
    # 初始化预测器
    predictor = RetrievalReactionPredictor()
    
    # 测试病例
    test_prompt = """[PAT] age: 80 sex: M wt: 95.0kg country: GB season: spring [/PAT]
[INDI] Ill-defined disorder [/INDI]
[DRUGS_FOR_INDI]
[DRUG] ARICEPT | ai: DONEPEZIL HYDROCHLORIDE | dose: 5.0 mg | freq: daily [/DRUG]
[DRUG] AMLODIPINE | ai: AMLODIPINE BESYLATE | dose: unknown | freq: daily [/DRUG]
[DRUG] ATORVASTATIN | ai: ATORVASTATIN | dose: unknown | freq: daily [/DRUG]
[DRUG] BISOPROLOL | ai: BISOPROLOL | dose: unknown | freq: daily [/DRUG]
[/DRUGS_FOR_INDI]"""
    
    # 测试1: 单个预测
    print("📋 测试病例:")
    print("-" * 60)
    print(test_prompt[:200] + "...")
    print("-" * 60 + "\n")
    
    print("🔍 预测中...\n")
    predictions = predictor.predict(test_prompt, top_k=10)
    
    print("📊 Top-10 预测结果:")
    print("=" * 60)
    for i, (reaction, score) in enumerate(predictions, 1):
        # 相似度得分范围通常在 [-1, 1]，归一化到 [0, 100]
        confidence = (score + 1) / 2 * 100
        print(f"{i:2d}. {reaction:45s} | 相似度: {score:6.4f} ({confidence:5.2f}%)")
    print("=" * 60 + "\n")
    
    # 测试2: 批量预测
    print("📦 批量预测测试...")
    batch_prompts = [test_prompt] * 3  # 复制3份测试
    batch_results = predictor.predict_batch(batch_prompts, top_k=5, batch_size=2)
    
    print(f"✅ 批量预测完成，处理 {len(batch_results)} 个样本")
    print(f"   样本1的Top-5: {[r[0] for r in batch_results[0]]}\n")
    
    # 测试3: 解释预测
    if predictions:
        target_reaction = predictions[0][0]  # 第一个预测
        print(f"🔍 解释预测: 为什么预测 '{target_reaction}'?")
        explanation = predictor.explain_prediction(test_prompt, target_reaction)
        print(f"   - 相似度分数: {explanation['score']:.4f}")
        print(f"   - 排名: {explanation['rank']}/{explanation['total_reactions']}")
        print(f"   - 百分位: {explanation['percentile']:.2f}%")
        print()
    
    # 性能统计
    print("📈 模型信息:")
    print(f"   - 总反应数: {len(predictor.all_reaction_names)}")
    print(f"   - 向量维度: {predictor.reaction_embeddings.shape[1]}")
    print(f"   - 设备: {predictor.device}")
    print()


if __name__ == "__main__":
    main()