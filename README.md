# Impala Text-to-SQL Intelligent Assistant Pro (v3)

本项目是一个基于 Streamlit 的智能 Text-to-SQL 助手，支持通过自然语言查询 Impala 数据库。该版本（v3）在原有功能基础上进行了显著优化，包括模型缓存机制和自适应知识库文件处理。

## 主要更新 (v3)

1.  **模型缓存优化 (Model Caching)**
    *   **问题解决**: 解决了之前版本中每次对话都重新加载模型导致的卡顿问题。
    *   **机制**: 系统现在会缓存已初始化的模型链路。只要用户不主动在侧边栏修改模型配置（提供商、模型名称、API Key），系统将复用第一次加载的模型实例，大大提升了对话响应速度。

2.  **自适应知识库构建 (Adaptive Knowledge Base)**
    *   **问题解决**: 解除了对特定文件名的强制依赖。
    *   **机制**: 用户上传的 Excel 文件（Schema 或 问答样例）不再需要保持特定的文件名（如原先的 `table_info_test_v4.xlsx`）。系统会自动将其保存为标准内部文件名 (`knowledge_base_schema.xlsx` 和 `knowledge_base_qa.xlsx`)，只要文件内部结构（列名、格式）符合要求即可。

## 功能特性

*   **多模型支持**: 支持 Google Gemini, OpenAI, Qwen (通义千问), Moonshot (Kimi) 等多种大模型。
*   **本地 Embeddings**: 使用 HuggingFace 本地模型进行向量化，无需消耗 API Token，且支持离线索引构建。
*   **RAG (检索增强生成)**: 基于 FAISS 向量库，结合 Schema 信息和 Few-Shot 样例，提供更准确的 SQL 生成。
*   **SQL 自检**: 内置基本的 Impala SQL 语法检查规则，提示潜在错误。
*   **Google Sheets 集成**: 支持直接从 Google Sheets 同步知识库（需配置 Service Account）。

## 快速开始

### 1. 环境准备

确保已安装 Python 3.8+，并安装项目依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行应用

使用 Streamlit 启动应用：

```bash
streamlit run app_v3.py
```

### 3. 使用流程

1.  **配置模型**: 在左侧侧边栏输入 API Key，选择相应的模型供应商和模型名称。
2.  **构建知识库**:
    *   在侧边栏上传 Schema Excel 文件（包含表结构描述）。
    *   在侧边栏上传 问答样例 Excel 文件（包含历史问答对）。
    *   *注意：文件名不限，上传后系统会自动识别。*
3.  **开始对话**: 在主界面输入框中描述你的查询需求，助手将生成对应的 Impala SQL。

## 文件结构要求

虽然文件名不再受限，但上传的 Excel 文件内容需符合以下结构：

*   **Schema Excel**: 需包含表名、描述、字段列表（包含字段名、类型、中文名、枚举值、备注说明等列）。
*   **问答样例 Excel**: 需包含自然语言问题和对应的 SQL 语句（格式参考：`例1：...`）。

## 依赖库

主要依赖包括：
*   `streamlit`
*   `langchain` & `langchain-community`
*   `faiss-cpu`
*   `pandas`
*   `openpyxl`
