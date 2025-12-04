# encoding: utf-8
"""
单独下载 BAAI/bge-m3 模型到指定目录（适用于本地有梯子时一次性下载）。
"""

import os
from huggingface_hub import snapshot_download

# 想把模型放到哪里：例如 D:\models\bge-m3
TARGET_DIR = r"D:\models\bge-m3"

# Hugging Face 仓库名
REPO_ID = "BAAI/bge-m3"

def main():
    # 如果目标目录已经存在，就不重复下载
    if os.path.isdir(TARGET_DIR) and os.listdir(TARGET_DIR):
        print(f"[info] 目标目录已存在且非空，跳过下载: {TARGET_DIR}")
        return

    # 使用 snapshot_download 下载仓库快照到一个临时缓存目录
    # local_dir: 直接指定最终目录；local_dir_use_symlinks=False 避免 Windows 符号链接问题
    print(f"[info] 开始下载 {REPO_ID} 到 {TARGET_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        revision="main",   # 默认分支
    )
    print(f"[done] 模型已下载到: {TARGET_DIR}")

if __name__ == "__main__":
    main()
