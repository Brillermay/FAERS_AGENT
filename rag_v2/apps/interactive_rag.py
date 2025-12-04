# encoding: utf-8
"""
交互式 RAG 问答系统
用户可以输入问题，系统会返回完整 RAG 流程的答案。
"""

import os
import sys
from rag_v2.core.output_layer import answer_pipeline

# 确保使用新训练的双塔模型
os.environ["EMBEDDING_MODEL_TYPE"] = "biencoder"
os.environ["BIENCODER_MODEL_PATH"] = r"D:\models\drug2reaction_biencoder_trial"

def main():
    print("=" * 60)
    print("FAERS RAG 问答系统")
    print("=" * 60)
    print("提示：")
    print("  - 输入问题后按回车查看回答")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'test' 运行预设测试问题")
    print("=" * 60)
    print()
    
    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("\n再见！")
                break
            
            if question.lower() == 'test':
                print("\n运行预设测试问题...\n")
                test_questions = [
                    "阿司匹林的常见不良反应有哪些？",
                    "NAUSEA 常见于哪些药物？",
                    "阿司匹林一般用于哪些适应症？",
                    "关于氯吡格雷，有哪些患者结局？",
                    "请列出常见药物。",
                ]
                for q in test_questions:
                    print("\n" + "=" * 60)
                    print(f"问题：{q}")
                    print("=" * 60)
                    answer_pipeline(q, suppress_previous=True)
                    print("\n" + "-" * 60)
                continue
            
            # 运行完整的 RAG 流水线
            print("\n" + "=" * 60)
            print(f"问题：{question}")
            print("=" * 60)
            result = answer_pipeline(question, suppress_previous=True)
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n[错误] 处理问题时出现异常: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
