import os
from transformers import AutoTokenizer, AutoModel

# 设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 创建模型保存目录
model_dir = "./Bio_ClinicalBERT_local"
os.makedirs(model_dir, exist_ok=True)

print("🔄 开始下载 Bio_ClinicalBERT 模型...")

try:
    # 下载tokenizer和模型
    print("📥 正在下载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    print("📥 正在下载 model...")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    # 保存到本地
    print("💾 正在保存 tokenizer...")
    tokenizer.save_pretrained(model_dir)
    
    print("💾 正在保存 model...")
    model.save_pretrained(model_dir)
    
    print(f"✅ 模型已成功下载到: {model_dir}")
    print("📁 下载的文件包括:")
    for file in os.listdir(model_dir):
        print(f"  - {file}")
        
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("\n💡 建议尝试手动下载或使用其他替代模型")