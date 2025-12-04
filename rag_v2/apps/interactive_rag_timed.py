# encoding: utf-8
"""
带时间统计的交互式 RAG 问答系统
用户可以输入问题，系统会返回完整 RAG 流程的答案，并显示每层的时间统计。
"""

import os
import time
import contextlib
from io import StringIO

from rag_v2.core.input_layer import parse_input
from rag_v2.core.initial_search_layer import initial_search
from rag_v2.core.relation_aggregate_layer import aggregate_relations
from rag_v2.core.ranking_layer import rank_expansions
from rag_v2.core.output_layer import generate_answer

# 确保使用新训练的双塔模型
os.environ["EMBEDDING_MODEL_TYPE"] = "biencoder"
os.environ["BIENCODER_MODEL_PATH"] = r"D:\models\drug2reaction_biencoder_trial"


def answer_pipeline_timed(question: str, suppress_previous: bool = True, show_timings: bool = True) -> dict:
    """
    带时间统计的完整 RAG 流水线
    suppress_previous: 是否隐藏前4层的详细输出
    show_timings: 是否显示时间统计
    """
    timings = {}
    
    # 第 1 层：输入解析
    start = time.time()
    if suppress_previous:
        with contextlib.redirect_stdout(StringIO()):
            parsed = parse_input(question)
    else:
        parsed = parse_input(question)
    timings['layer1_parse'] = time.time() - start
    
    # 第 2 层：初步搜索
    start = time.time()
    if suppress_previous:
        with contextlib.redirect_stdout(StringIO()):
            searched = initial_search(parsed)
    else:
        searched = initial_search(parsed)
    timings['layer2_search'] = time.time() - start
    
    # 第 3 层：关系聚合
    start = time.time()
    if suppress_previous:
        with contextlib.redirect_stdout(StringIO()):
            aggregated = aggregate_relations(searched)
    else:
        aggregated = aggregate_relations(searched)
    timings['layer3_aggregate'] = time.time() - start
    
    # 第 4 层：向量排序
    start = time.time()
    if suppress_previous:
        with contextlib.redirect_stdout(StringIO()):
            ranked = rank_expansions(aggregated)
    else:
        ranked = rank_expansions(aggregated)
    timings['layer4_rank'] = time.time() - start
    
    # 第 5 层：答案生成
    start = time.time()
    result = generate_answer(ranked)
    timings['layer5_generate'] = time.time() - start
    
    # 计算总时间
    timings['total'] = sum([
        timings['layer1_parse'],
        timings['layer2_search'],
        timings['layer3_aggregate'],
        timings['layer4_rank'],
        timings['layer5_generate']
    ])
    
    # 显示时间统计
    if show_timings:
        print("\n" + "=" * 60)
        print("响应时间统计（秒）")
        print("=" * 60)
        print(f"  第 1 层（输入解析）:        {timings['layer1_parse']:.2f}s  ({timings['layer1_parse']/timings['total']*100:.1f}%)")
        print(f"  第 2 层（初步搜索）:        {timings['layer2_search']:.2f}s  ({timings['layer2_search']/timings['total']*100:.1f}%)")
        print(f"  第 3 层（关系聚合）:        {timings['layer3_aggregate']:.2f}s  ({timings['layer3_aggregate']/timings['total']*100:.1f}%)")
        print(f"  第 4 层（向量排序）:        {timings['layer4_rank']:.2f}s  ({timings['layer4_rank']/timings['total']*100:.1f}%)")
        print(f"  第 5 层（答案生成）:        {timings['layer5_generate']:.2f}s  ({timings['layer5_generate']/timings['total']*100:.1f}%)")
        print("-" * 60)
        print(f"  总计:                      {timings['total']:.2f}s")
        print("=" * 60)
    
    # 将时间统计添加到结果中
    result['timings'] = timings
    
    return result


def main():
    print("=" * 60)
    print("FAERS RAG 问答系统（带时间统计）")
    print("=" * 60)
    print("提示：")
    print("  - 输入问题后按回车查看回答和时间统计")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'test' 运行预设测试问题")
    print("  - 输入 'verbose' 切换详细输出模式（显示所有层的信息）")
    print("=" * 60)
    print()
    
    verbose_mode = False
    
    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("\n再见！")
                break
            
            if question.lower() == 'verbose':
                verbose_mode = not verbose_mode
                print(f"\n[模式切换] {'详细输出模式' if verbose_mode else '简洁输出模式'}")
                continue
            
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
                    answer_pipeline_timed(q, suppress_previous=not verbose_mode, show_timings=True)
                    print("\n" + "-" * 60)
                continue
            
            # 运行完整的 RAG 流水线
            print("\n" + "=" * 60)
            print(f"问题：{question}")
            print("=" * 60)
            result = answer_pipeline_timed(question, suppress_previous=not verbose_mode, show_timings=True)
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
