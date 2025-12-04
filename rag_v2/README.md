## rag_v2 子系统说明（FAERS RAG）

本目录是 FAERS Agent 项目中的 **知识图谱 RAG 子系统**，已经按“核心逻辑 / 应用入口 / 脚本 / 训练 / 数据 / 结果 / 实验”进行了分层，便于后续维护和扩展。

后续如果有新增文件，建议参考本说明更新结构和用途。

---

## 目录总览

- `core/`：RAG 在线推理的核心逻辑（5 层流水线 + 向量工具）
- `apps/`：给人类或上层 Agent 调用的 **入口脚本**（交互问答、RAG 评估）
- `scripts/`：数据准备 / 向量构建 / 环境相关的 **工具脚本**
- `training/`：双塔模型训练与检索质量评估相关代码及输出
- `data/`：RAG 与训练所依赖的 **静态数据文件**（CSV / JSONL 等）
- `results/`：RAG 行为评估等 **结果文件**
- `models/`：本地存放的 embedding / 双塔模型权重
- `config/`：配置文件（预留 dev/prod 等 profile）
- `experiments/`：只在开发阶段使用的 **实验脚本**，不要求长期兼容

---

## 1. `core/` —— RAG 核心逻辑

- `input_layer.py`  
  第 1 层：输入解析  
  - 调用 LLM 对用户问题进行解析（抽取 Drug/Reaction/Indication/Outcome seeds、intents、topk 等）。  
  - 调用 `embedding_utils` 生成 query 向量。  

- `initial_search_layer.py`  
  第 2 层：初步检索  
  - 基于第 1 层解析结果，在 Neo4j 中用全文索引 / CONTAINS 查询候选节点。  

- `relation_aggregate_layer.py`  
  第 3 层：关系扩展与聚合  
  - 从 DrugSet / Patient 等关系出发，统计各 Drug/Reaction/Indication/Outcome 的共现频次（distinct 计数）。  

- `ranking_layer.py`  
  第 4 层：向量重排  
  - 结合 query 向量与节点 embedding 做语义 + 频次加权排序。  
  - 处理 embedding 维度不一致时的兜底逻辑。  

- `output_layer.py`  
  第 5 层：答案生成  
  - 聚合各意图的候选，构造“原始要点”列表。  
  - 调用本地 Ollama（`qwen2.5:7b-instruct`）生成中文回答，并进行输出清洗。  
  - 提供如 `generate_answer` / `answer_pipeline` 等高层接口。  

- `embedding_utils.py`  
  统一的向量生成工具：  
  - 通过环境变量选择使用 **Ollama embedding** 或 **本地双塔模型**。  
  - 提供 `embed_text` / `embed_batch` / `get_current_model_name` 等函数。  

> 注意：**上层 Agent 将来只需要依赖 `core` 暴露的函数**，其余目录都可以视为实现细节或离线流程。

---

## 2. `apps/` —— 入口脚本（交互 & 评估）

- `interactive_rag_timed.py`  
  带时间统计的交互式问答脚本，是目前推荐的主入口。  
  - 用法（在项目根目录）：  
    ```bash
    python -m rag_v2.apps.interactive_rag_timed
    ```  
  - 功能：  
    - 逐层调用 `core` 的 5 层流水线，返回完整回答。  
    - 统计并打印 1~5 层以及总耗时。  

- `interactive_rag.py`  
  简化版交互问答，只跑完整流水线，不显示时间统计。主要保留作对比 / 备用。  

- `eval_drug2reaction_rag.py`  
  在 `data/drug2reaction_eval.jsonl` 上评估当前 RAG 的 Drug→Reaction 检索性能（Hit@K / Recall@K / MRR@K）。  
  - 输出结果写入：`results/eval_drug2reaction_rag_results.json`。  
  - 用法：  
    ```bash
    python -m rag_v2.apps.eval_drug2reaction_rag
    ```

---

## 3. `scripts/` —— 数据与环境脚本

- `build_embeddings_ollama.py`  
  为 Neo4j 中各类节点（Drug / Reaction / Indication / Outcome）构建或重算 `embedding` 属性。  
  - 实际已通过 `core.embedding_utils` 支持使用双塔模型。  
  - 可通过环境变量控制：  
    - `EMBEDDING_MODEL_TYPE`（`ollama` / `biencoder`）  
    - `EMBED_LABELS`（要处理的标签列表）  
    - `EMBED_SKIP_EXISTING`（是否跳过已有 embedding）等。  

- `export_drug_reaction_stats.py`  
  从 Neo4j 统计 Drug→Reaction 的共现频次，并导出：  
  - `data/drug_reaction_stats.csv`  
  - `data/drug_reaction_stats.json`  
  供评估集构造与分析使用。  

- `gen_eval_drug2reaction.py`  
  基于 `drug_reaction_stats.csv`，调用 Ollama 生成自然语言中文问题，构造评估集：  
  - 输入：`data/drug_reaction_stats.csv`  
  - 输出：`data/drug2reaction_eval.jsonl`  

- `download_bge_m3.py`  
  从 Hugging Face 下载 `BAAI/bge-m3` 到本地指定目录，用于离线训练（一次性工具脚本）。

---

## 4. `training/` —— 双塔训练与模型评估

- `train_biencoder_drug2reaction.py`  
  使用 `sentence-transformers` 对 Drug→Reaction 任务进行双塔微调。  
  - 训练数据：`data/train_pairs_drug2reaction_small.jsonl`  
  - Dev 集：`data/drug2reaction_eval.jsonl`  
  - 输出模型目录：`models/drug2reaction_biencoder_trial/`  

- `prepare_train_drug2reaction.py`  
  从 `data/drug2reaction_eval.jsonl` 生成训练对 (`query`, `doc`)，写入：  
  - `data/train_pairs_drug2reaction.jsonl`（全量）  
  - `data/train_pairs_drug2reaction_small.jsonl`（小样本）  

- `eval_biencoder_comparison.py`  
  对比评估：  
  - Baseline：Ollama 向量模型  
  - New：本地双塔模型  
  在 Drug→Reaction 任务上的检索质量差异，输出到：  
  - `training/evaluation_results/` 目录（包含 baseline / biencoder / comparison 报告）。  

- `checkpoints/`  
  训练过程中的中间结果（保留以便复现）。  

> 这一整块主要用于 **离线实验与模型选型**，不会被线上 Agent 直接调用。

---

## 5. `data/` —— 评估与训练数据

- `drug_reaction_stats.csv` / `drug_reaction_stats.json`  
  从 Neo4j 导出的 Drug→Reaction 聚合统计结果。  

- `drug2reaction_eval.jsonl`  
  Drug→Reaction 评估集，每条包含：  
  - 药物名、中文问题、标准答案（多个反应）。  

- `train_pairs_drug2reaction.jsonl` / `train_pairs_drug2reaction_small.jsonl`  
  用于 SentenceTransformers 双塔训练的 `(query, doc)` 对。  

> 建议：所有新产生的持久化数据文件，都放在 `data/` 下，并在本 README 中补充说明来源和用途。

---

## 6. `results/` —— 行为层评估结果

- `eval_drug2reaction_rag_results.json`  
  `apps/eval_drug2reaction_rag.py` 的输出结果，用于记录当前 RAG 配置在 Drug→Reaction 任务上的整体检索表现。

将来如有更多行为层评估脚本，建议也把结果统一写到 `results/` 下。

---

## 7. `models/` —— 模型权重

- `drug2reaction_biencoder_trial/`  
  由 `training/train_biencoder_drug2reaction.py` 训练得到的 SentenceTransformer 模型目录，可直接被 `core/embedding_utils.py` 加载使用。

> 如果后续有更多模型版本，可以在 `models/` 下增加子目录，并在配置中指定路径。

---

## 8. `config/` —— 配置（预留）

- `config.yaml`  
  目前仅为占位（例如包含 `profile: dev`），后续可以扩展为：  
  - dev / prod 的 Neo4j 连接  
  - embedding 模型路径  
  - Ollama 地址与模型名  
  等统一管理。

建议后续新增 `config/dev.yaml`、`config/prod.yaml` 等 profile 文件，并在核心代码中集中读取。

---

## 9. `experiments/` —— 开发/实验区

此目录专门用于放置：  

- 某一阶段只在开发时使用的临时脚本（例如针对某个季度数据的特殊构建脚本）；  
- 一次性的对比实验 / 调参脚本；  
- 不保证长期兼容，但希望“留痕迹、可回顾”的代码。  

约定：  

- **只要文件在 `experiments/` 内，就默认是“实验性质”**，不影响对上层 Agent 的稳定接口。  
- 如果某个实验脚本沉淀为稳定流程，建议迁移到 `scripts/` / `apps/` / `training/` 等正式目录。

---

## 10. 使用与维护建议

- **新增脚本时先想清楚：它属于哪一类？**
  - 在线推理逻辑 → `core/`
  - 命令行入口 → `apps/`
  - 数据生成 / 环境工具 → `scripts/`
  - 训练与离线评估 → `training/`
  - 一次性实验 → `experiments/`

- **新增数据文件时统一放入 `data/`，并在此 README 说明来源与用途。**

- **新增评估结果时统一放入 `results/` 或 `training/evaluation_results/`，按“行为评估 vs 模型评估”区分。**

后续如果我们调整结构或新增模块，建议同步更新本 `README.md`，保持“目录即文档”的可读性。 


