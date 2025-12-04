import os
import ast
import json
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 更可靠的中文字体选择/注册逻辑
def _choose_and_register_cjk_font(candidates=None):
    if candidates is None:
        candidates = [
            'PingFang', 'PingFang SC', 'PingFang TC',
            'Noto Sans CJK', 'NotoSansCJK', 'Noto Sans CJK SC',
            'SimHei', 'SimSun', 'Heiti', 'STHeiti',
            'Microsoft YaHei', 'Arial Unicode MS'
        ]
    sys_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf') + \
                font_manager.findSystemFonts(fontpaths=None, fontext='otf') + \
                font_manager.findSystemFonts(fontpaths=None, fontext='ttc')
    for path in sys_fonts:
        try:
            prop = font_manager.FontProperties(fname=path)
            name = prop.get_name()
            if any(cand.lower() in name.lower() for cand in candidates):
                # 注册并设置优先字体
                try:
                    font_manager.fontManager.addfont(path)
                except Exception:
                    pass
                matplotlib.rcParams['font.sans-serif'] = [name]
                matplotlib.rcParams['font.family'] = 'sans-serif'
                matplotlib.rcParams['axes.unicode_minus'] = False
                return name, path
        except Exception:
            continue
    # 回退
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None, None

chosen_name, chosen_path = _choose_and_register_cjk_font()
if chosen_name:
    print(f"使用字体: {chosen_name} ({chosen_path})")
else:
    print("未找到系统 CJK 字体，回退到 DejaVu Sans。建议安装 Noto Sans CJK 或 SimHei 等中文字体。")

def load_label_column(csv_path, label_cols=None):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # 寻找可能含标签的列（优先显式列名）
    candidates = []
    if label_cols:
        candidates = [c for c in label_cols if c in df.columns]
    if not candidates:
        for c in df.columns:
            low = c.lower()
            if any(p in low for p in ('label', 'reaction', 'adverse', 'ae')):
                candidates.append(c)
        # 回退策略：寻找包含类似 "['xxx']" 的列
        if not candidates:
            for c in df.columns:
                sample = df[c].iloc[:50].astype(str)
                if sample.str.startswith('[').any() and sample.str.contains("'").any():
                    candidates.append(c)
                    break
    if not candidates:
        raise RuntimeError("无法自动识别标签列，请通过 --label-col 指定")
    col = candidates[0]
    raws = df[col].astype(str).tolist()
    parsed = []
    for s in raws:
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                parsed.append([str(x) for x in v])
            elif isinstance(v, str) and s.strip() != "":
                parsed.append([v])
            else:
                parsed.append([])
        except Exception:
            # 尝试以分隔符分割
            if s.strip() == "":
                parsed.append([])
            else:
                parsed.append([x.strip() for x in s.split(';') if x.strip() != ""])
    return parsed, col

def compute_counts(label_lists, min_count=1):
    cnt = Counter()
    for labs in label_lists:
        for l in labs:
            cnt[l] += 1
    # 过滤小于 min_count
    for k in list(cnt.keys()):
        if cnt[k] < min_count:
            del cnt[k]
    return cnt

def log_transform_counts(counts):
    labels = list(counts.keys())
    vals = np.array([counts[l] for l in labels], dtype=float)
    logv = np.log10(vals + 1.0)
    return labels, vals, logv

def cluster_log_counts(logv, k=3, random_state=42):
    X = logv.reshape(-1,1)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X)
    centers = km.cluster_centers_.flatten()
    labels = km.labels_
    # 将簇按中心值降序排序，并返回簇等级（0=high,...）
    order = np.argsort(centers)[::-1]
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    ranked_labels = np.array([rank[l] for l in labels])
    sorted_centers = centers[order]
    return ranked_labels, sorted_centers, km

def detect_elbow(inertias):
    # 简单二阶差分寻找拐点（返回索引+2 对应 k）
    s = np.array(inertias)
    d1 = np.diff(s)
    d2 = np.diff(d1)
    if len(d2) == 0:
        return 2
    elbow_idx = np.argmax(np.abs(d2)) + 2
    return elbow_idx

def run(args):
    labels_lists, detected_col = load_label_column(args.csv, label_cols=(args.label_col.split(',') if args.label_col else None))
    counts = compute_counts(labels_lists, min_count=args.min_count)
    labels, vals, logv = log_transform_counts(counts)
    print(f"读取 {len(labels_lists)} 条记录，发现 {len(counts)} 个标签 (min_count={args.min_count})")
    # 选择 k
    chosen_k = args.k
    if chosen_k is None:
        inertias = []
        Ks = list(range(2, min(8, max(3, int(len(labels)/10)))))
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=0, n_init=8).fit(logv.reshape(-1,1))
            inertias.append(km.inertia_)
        chosen_k = detect_elbow(inertias)
        print(f"自动检测 elbow -> 建议 k = {chosen_k} (参考 inertias for Ks={Ks})")
    else:
        print(f"使用指定 k = {chosen_k}")
    # 聚类
    clustered, centers, km = cluster_log_counts(logv, k=chosen_k, random_state=args.random_state)
    # 构造输出 mapping
    mapping = {}
    for lab, cluster_id, raw_count in zip(labels, clustered, vals):
        mapping[lab] = {
            'count': int(raw_count),
            'log10': float(np.log10(raw_count+1)),
            'cluster': int(cluster_id)  # 0 = 高, 1 = 次高 ...
        }
    # 统计每簇摘要
    summary = {}
    for cid in range(chosen_k):
        members = [l for l,v in mapping.items() if v['cluster']==cid]
        counts_c = [mapping[m]['count'] for m in members]
        summary[cid] = {
            'n_labels': len(members),
            'min_count': int(min(counts_c)) if counts_c else 0,
            'max_count': int(max(counts_c)) if counts_c else 0,
            'mean_count': float(np.mean(counts_c)) if counts_c else 0.0,
            'center_log10': float(centers[cid]) if cid < len(centers) else None
        }
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, args.output_name)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'mapping': mapping, 'summary': summary, 'k': chosen_k, 'detected_col': detected_col}, f, ensure_ascii=False, indent=2)
    print(f"聚类结果已保存: {out_json}")
    # 画图（可选）
    plt.figure(figsize=(6,4))
    order = np.argsort(vals)[::-1]
    plt.plot(np.log10(vals[order]+1), marker='.', linestyle='-')
    plt.title('标签按频次降序的 log10(count+1)')
    plt.xlabel('rank')
    plt.ylabel('log10(count+1)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'freq_sorted_log10.png'), dpi=150)
    plt.close()
    # 散点 + 簇色
    plt.figure(figsize=(6,4))
    X = np.arange(len(vals))
    plt.scatter(X, np.log10(vals+1), c=clustered, cmap='tab10', s=8)
    plt.title('log10(count+1) 聚类结果')
    plt.ylabel('log10(count+1)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'clusters.png'), dpi=150)
    plt.close()
    # 简短打印 summary
    print("簇摘要:")
    for cid, s in summary.items():
        print(f"  cluster {cid}: n_labels={s['n_labels']}, count_range=({s['min_count']},{s['max_count']}), center_log10={s['center_log10']:.3f}")

if __name__ == "__main__":
    # 程序内配置（直接在这里修改参数）
    program_config = {
        'csv': './outputs/prompts_sample_30000_coarse_coarse_v2.csv',
        'label_col': 'labels_list',               # 若需指定列名，填入例如 'labels_list'
        'min_count':5,
        'k': 4,                     # 指定簇数或 None 使用自动 elbow 检测
        'output_dir': './outputs/label_clustering',
        'output_name': 'label_cluster_result.json',
        'random_state': 42
    }

    # 简单构造一个类似 argparse.Namespace 的对象传给 run()
    class SimpleArgs:
        pass

    args = SimpleArgs()
    for key, val in program_config.items():
        setattr(args, key, val)

    run(args)