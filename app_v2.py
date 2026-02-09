import streamlit as st
import os
import re
import pandas as pd
import time
import json
import httpx

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

# 导入各平台 LLM 实现
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ========== 可选依赖（用于 Google Sheets 读取） ==========
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GS_AVAILABLE = True
except Exception:
    GS_AVAILABLE = False

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="Impala SQL 智能助手 Pro", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stMarkdown h3 {
        color: #4285F4;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Impala Text-to-SQL 智能助手 Pro")

# ========== 核心：自动识别并获取模型列表 ==========

def fetch_available_models(api_key):
    """根据 API Key 自动识别供应商并拉取可用模型"""
    if not api_key:
        return []

    # 1. 识别 Gemini (Google)
    if api_key.startswith("AIza"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append({"id": m.name.replace("models/", ""), "provider": "google"})
            return models
        except Exception:
            return []

    # 2. 识别 OpenAI 兼容格式 (Qwen, Kimi, OpenAI 等)
    # 通用的识别逻辑：尝试请求其模型列表接口
    providers = {
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "kimi": "https://api.moonshot.cn/v1",
        "openai": "https://api.openai.com/v1"
    }
    
    # 根据 Key 特征或尝试法
    target_url = providers["openai"] # 默认
    current_provider = "openai"
    
    # 简单特征判断
    if "sk-" in api_key:
        # 尝试通用的 OpenAI 兼容接口拉取
        # 这里为了演示，我们优先尝试用户最常用的 Qwen 和 Kimi
        for name, url in providers.items():
            try:
                response = httpx.get(f"{url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return [{"id": m["id"], "provider": name} for m in data.get("data", [])]
            except:
                continue
    return []



# ==========================================
# 1. 侧边栏：核心配置 (API Key & Model)
# ==========================================
with st.sidebar:
    st.header("🔑 模型配置")
    
    api_key = st.text_input("输入 API Key", type="password", help="支持 Gemini, Qwen, Kimi 等")
    
    # 动态获取并显示模型
    available_models = []
    if api_key:
        with st.spinner("正在识别 Key 并拉取模型..."):
            available_models = fetch_available_models(api_key)
    
    if available_models:
        model_display_names = [f"{m['provider']} | {m['id']}" for m in available_models]
        selected_display = st.selectbox("选择模型", model_display_names)
        
        # 提取选中的模型信息
        sel_idx = model_display_names.index(selected_display)
        target_model = available_models[sel_idx]["id"]
        target_provider = available_models[sel_idx]["provider"]
        st.success(f"已识别供应商: {target_provider.upper()}")
    else:
        if api_key:
            st.error("无法识别此 Key 或无法连接 API，请检查网络")
        st.stop()

    st.divider()



# ==========================================
# 3. 动态初始化 LLM 和 Embedding
# ==========================================

def get_llm(provider, model_name, api_key):
    if provider == "google":
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, streaming=True)
    else:
        # Qwen, Kimi 等均通过 ChatOpenAI 桥接
        base_urls = {
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "kimi": "https://api.moonshot.cn/v1",
            "openai": "https://api.openai.com/v1"
        }
        return ChatOpenAI(
            model=model_name, 
            openai_api_key=api_key, 
            base_url=base_urls.get(provider), 
            temperature=0, 
            streaming=True
        )

def get_embeddings(provider, api_key):
    if provider == "google":
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    else:
        # 注意：Kimi/Qwen 的 Embedding 模型名不同，这里简化处理
        # 实际生产中建议针对 provider 判断具体的 embedding model 名
        emb_model = "text-embedding-v2" if provider == "qwen" else "text-embedding-ada-002"
        base_urls = {
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "kimi": "https://api.moonshot.cn/v1",
            "openai": "https://api.openai.com/v1"
        }
        return OpenAIEmbeddings(model=emb_model, openai_api_key=api_key, base_url=base_urls.get(provider))




# ==========================================
# 2. 辅助工具函数
# ==========================================
def basic_impala_syntax_check(sql_text):
    errors = []
    warnings = []
    if '"' in sql_text and "'" in sql_text:
        warnings.append("⚠️ 检测到 SQL 中混用了单引号和双引号，Impala 中推荐统一使用单引号。")

    forbidden_funcs = ["getdate()", "to_char(", "sysdate"]
    for func in forbidden_funcs:
        if func.lower() in sql_text.lower():
            errors.append(f"❌ 检测到非 Impala 兼容函数: {func}")

    if "sum(" in sql_text.lower() or "count(" in sql_text.lower() or "avg(" in sql_text.lower():
        if "group by" not in sql_text.lower() and "over (" not in sql_text.lower():
            warnings.append("⚠️ 检测到聚合函数但未发现 GROUP BY 或窗口函数，请确认逻辑是否正确。")

    return errors, warnings

def save_uploaded_file(uploaded_file, dest_path):
    try:
        bytes_data = uploaded_file.read()
        with open(dest_path, "wb") as f:
            f.write(bytes_data)
        return True, None
    except Exception as e:
        return False, str(e)

def gsheet_to_excel_and_save(sheet_id_or_url: str, service_account_info: dict, dest_path: str):
    if not GS_AVAILABLE:
        raise RuntimeError("gspread 库未安装")

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly",
              "https://www.googleapis.com/auth/drive.readonly"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    gc = gspread.authorize(creds)

    try:
        if sheet_id_or_url.startswith("http"):
            sheet = gc.open_by_url(sheet_id_or_url)
        else:
            sheet = gc.open_by_key(sheet_id_or_url)
    except Exception as e:
        raise e

    writer = pd.ExcelWriter(dest_path, engine="openpyxl")
    for ws in sheet.worksheets():
        values = ws.get_all_values()
        if not values:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(values)
        df.to_excel(writer, sheet_name=ws.title, index=False, header=False)
    writer.close() # Pandas >= 1.5 使用 close() 自动保存

# ==========================================
# 3. 核心解析与索引构建
# ==========================================
# 缓存资源，当 API Key 变化时需重新加载 Embedding 模型
@st.cache_resource(show_spinner="正在构建知识库索引...")
def init_knowledge_base(api_key,provider):
    # --- 内部解析函数 ---
    def parse_schema_excel(file_path):
        all_tables = []
        try:
            xl = pd.ExcelFile(file_path)
            for sheet_name in xl.sheet_names:
                try:
                    df_raw = pd.read_excel(xl, sheet_name=sheet_name, header=None)
                    table_name = sheet_name
                    description = "无描述"

                    for i in range(min(5, len(df_raw))):
                        row_str = str(df_raw.iloc[i, 0])
                        if "table_name:" in row_str:
                            table_name = row_str.split('table_name:')[-1].strip()
                        elif "description:" in row_str:
                            description = row_str.split('description:')[-1].strip()

                    header_row_index = -1
                    for i, row in df_raw.iterrows():
                        row_values = [str(x) for x in row.values]
                        if "新字段" in row_values:
                            header_row_index = i
                            break

                    if header_row_index == -1: continue

                    df_columns = pd.read_excel(xl, sheet_name=sheet_name, skiprows=header_row_index)
                    df_columns.columns = [str(c).strip() for c in df_columns.columns]

                    columns_list = []
                    for _, row in df_columns.iterrows():
                        if pd.isna(row.get("新字段")): continue
                        field_name = str(row.get('新字段')).strip()
                        field_type = str(row.get('字段类型')).strip() if pd.notna(row.get('字段类型')) else ""
                        field_cn = str(row.get('中文名')).strip().replace('\n', ' ') if pd.notna(row.get('中文名')) else ""
                        col_info = f"- **{field_name}** ({field_type}): {field_cn}"
                        
                        enum_val = row.get('枚举值')
                        if pd.notna(enum_val) and str(enum_val).strip() != "":
                            col_info += f" | 枚举: {str(enum_val).strip().replace(chr(10), ' ')}"
                        
                        note_val = row.get('备注说明')
                        if pd.notna(note_val) and str(note_val).strip() != "":
                            col_info += f" | ⚠️备注: {str(note_val).strip().replace(chr(10), ' ')}"
                        
                        columns_list.append(col_info)

                    if columns_list:
                        all_tables.append({
                            "table_name": table_name,
                            "description": description,
                            "columns": columns_list
                        })
                except Exception:
                    continue
        except Exception:
            pass
        return all_tables

    def parse_qa_excel(file_path):
        examples = []
        try:
            df = pd.read_excel(file_path, header=None)
            raw_lines = [str(x).strip() for x in df.iloc[:, 0].tolist() if pd.notna(x) and str(x).strip() != '']
            buffer = []
            for line in raw_lines:
                if re.match(r'^例\s*\d+[：:].*', line):
                    if buffer:
                        full_text = "\n".join(buffer)
                        split_match = re.search(r'(.*?)(select\s.*)', full_text, re.IGNORECASE | re.DOTALL)
                        if split_match:
                            examples.append({"question": split_match.group(1).strip(), "sql": split_match.group(2).strip()})
                    buffer = [line]
                else:
                    buffer.append(line)
            if buffer:
                full_text = "\n".join(buffer)
                split_match = re.search(r'(.*?)(select\s.*)', full_text, re.IGNORECASE | re.DOTALL)
                if split_match:
                    examples.append({"question": split_match.group(1).strip(), "sql": split_match.group(2).strip()})
        except Exception:
            pass
        return examples

    if not os.path.exists("table_info_test_v4.xlsx") or not os.path.exists("简单需求样例.xlsx"):
        return None, None

    schema_data = parse_schema_excel("table_info_test_v4.xlsx")
    qa_data = parse_qa_excel("简单需求样例.xlsx")

    if not schema_data: return None, None

    schema_docs = [Document(page_content=f"表名: {t['table_name']}\n描述: {t['description']}\n字段列表:\n" + "\n".join(t['columns']), metadata={"type": "schema"}) for t in schema_data]
    qa_docs = [Document(page_content=q['question'], metadata={"sql": q['sql'], "type": "example"}) for q in qa_data]

    # 使用用户提供的 Key 初始化 Embeddings
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key_trigger)
    
    # 动态选择 Embedding
    embeddings = get_embeddings(provider, api_key)
    
    #retriever_s = FAISS.from_documents(schema_docs, embeddings).as_retriever(search_kwargs={"k": 10})
    #retriever_e = FAISS.from_documents(qa_docs, embeddings).as_retriever(search_kwargs={"k": 5})
    #return retriever_s, retriever_e
    
    retriever_s = FAISS.from_documents(schema_docs, embeddings).as_retriever(search_kwargs={"k": 10})
    retriever_e = None
    if qa_docs:
        retriever_e = FAISS.from_documents(qa_docs, embeddings).as_retriever(search_kwargs={"k": 5})

    return retriever_s, retriever_e

# ==========================================
# 4. 链条逻辑 (接受模型名称和 API Key)
# ==========================================
def get_sql_chain(retriever_schema, retriever_examples,provider, model_name, api_key):
    # 动态初始化 LLM
    #llm = ChatGoogleGenerativeAI(
    #    model=model_name, 
    #    temperature=0, 
    #    streaming=True,
    #    google_api_key=api_key
    #)
    
    llm = get_llm(provider, model_name, api_key)

    rephrase_template = """基于对话历史，将用户的最新问题改写为一个独立的、包含完整上下文的问题。
    对话历史:
    {chat_history}
    最新提问: {input}
    独立问题:"""
    rephrase_prompt = ChatPromptTemplate.from_template(rephrase_template)
    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    sql_template = """你是一个 Impala SQL 专家助手。请根据以下信息回答用户问题。

    【Schema 信息 (表结构)】
    {schema}

    【参考案例 (Few-Shot)】
    {examples}

    【用户当前问题】
    {question}

    请按照以下步骤思考并输出结果（请使用 Markdown 格式）：

    ### 步骤 1: 需求与字段分析
    * **分析**: 简述你对用户需求的理解（指标、维度、筛选）。
    * **表与字段选择**: 明确列出你决定使用的表名和关键字段，并解释原因。
        * 格式：`表名.字段名` (中文名) - [使用逻辑]
        * **注意**: 必须检查字段的"备注说明"和"枚举值"，确保逻辑符合业务定义。

    ### 步骤 2: SQL 编写与自检 (Chain of Thought)
    在编写 SQL 之前，请进行自我检查：
    * [Check] 是否使用了 Impala 兼容的语法（单引号字符串、Impala 日期函数）？
    * [Check] 是否存在多表连接？连接键是否在 Schema 中存在？
    * [Check] WHERE 条件中的时间范围是否符合用户描述？
    * [Check] 聚合计算（Sum/Count）是否正确配合了 Group By？

    ### 步骤 3: SQL 代码
    请生成最终的 SQL 代码。
    ```sql
    -- 在这里写 SQL
    ```
    """

    sql_prompt = ChatPromptTemplate.from_template(sql_template)

    def format_docs(docs): return "\n\n".join([d.page_content for d in docs])
    def format_qs(docs): return "\n\n".join([f"Q: {d.page_content}\nSQL: {d.metadata['sql']}" for d in docs])

    chain = (
        RunnablePassthrough.assign(
            standalone_question=lambda x: rephrase_chain.invoke(x) if x.get("chat_history") else x["input"]
        )
        | RunnablePassthrough.assign(
            schema=lambda x: format_docs(retriever_schema.invoke(x["standalone_question"])),
            examples=lambda x: format_qs(retriever_examples.invoke(x["standalone_question"])) if retriever_examples else "",
            question=lambda x: x["standalone_question"]
        )
        | sql_prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ==========================================
# 5. 侧边栏：知识库管理逻辑
# ==========================================
with st.sidebar:
    st.header("🗂️ 知识库管理")
    st.caption("首次使用请先构建知识库")

    # 显示文件状态
    has_files = os.path.exists("table_info_test_v4.xlsx") and os.path.exists("简单需求样例.xlsx")
    file_status = "✅ 已就绪" if has_files else "❌ 缺失文件"
    st.text(f"本地状态: {file_status}")

    with st.expander("🛠️ 上传或更新知识库"):
        st.subheader("1. 表结构 (Schema)")
        schema_source = st.radio("Schema 来源", ("上传 Excel", "Google Sheets"), key="s_src")
        
        if schema_source == "上传 Excel":
            up_schema = st.file_uploader("上传 table_info_test_v4.xlsx", type=["xlsx"])
            if st.button("保存 Schema Excel") and up_schema:
                save_uploaded_file(up_schema, "table_info_test_v4.xlsx")
                st.success("已保存!")
                st.rerun()
        else:
            s_id = st.text_input("Sheet ID/URL", key="s_id")
            s_sa = st.file_uploader("Service Account JSON", type=["json"], key="s_sa")
            if st.button("从 Sheets 同步 Schema") and s_id and s_sa:
                try:
                    sa_info = json.load(s_sa)
                    gsheet_to_excel_and_save(s_id, sa_info, "table_info_test_v4.xlsx")
                    st.success("同步成功!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.divider()
        st.subheader("2. 问答样例 (Few-Shot)")
        qa_source = st.radio("样例来源", ("上传 Excel", "Google Sheets"), key="q_src")
        
        if qa_source == "上传 Excel":
            up_qa = st.file_uploader("上传 简单需求样例.xlsx", type=["xlsx"])
            if st.button("保存样例 Excel") and up_qa:
                save_uploaded_file(up_qa, "简单需求样例.xlsx")
                st.success("已保存!")
                st.rerun()
        else:
            q_id = st.text_input("Sheet ID/URL", key="q_id")
            q_sa = st.file_uploader("Service Account JSON", type=["json"], key="q_sa_f")
            if st.button("从 Sheets 同步样例") and q_id and q_sa:
                try:
                    sa_info = json.load(q_sa)
                    gsheet_to_excel_and_save(q_id, sa_info, "简单需求样例.xlsx")
                    st.success("同步成功!")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if st.button("🔄 重置/清空对话"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 6. 主流程：初始化与对话
# ==========================================
if has_files:
    # 传入 user_api_key 作为 trigger，如果 key 变了，缓存失效重新加载
    rs, re_ = init_knowledge_base(api_key,target_provider)
    if rs:
        # 获取链对象，传入当前选择的模型和 API Key
        chain = get_sql_chain(rs, re_, selected_model, api_key)
    else:
        st.warning("知识库文件解析失败，请检查 Excel 格式。")
        st.stop()
else:
    st.info("👋 欢迎！请在左侧侧边栏上传 Excel 文件或连接 Google Sheets 以构建知识库。")
    st.stop()

# --- 对话界面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

if prompt := st.chat_input("请输入查询需求..."):
    st.chat_message("user", avatar="🧑‍💻").markdown(prompt)
    history = st.session_state.messages.copy()

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = chain.stream({
                "input": prompt,
                "chat_history": history
            })

            for chunk in stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            # SQL 自动检测
            sql_match = re.search(r"```sql\n(.*?)\n```", full_response, re.DOTALL)
            if sql_match:
                sql_code = sql_match.group(1)
                errors, warnings = basic_impala_syntax_check(sql_code)
                if errors or warnings:
                    report = "\n\n---\n**🔍 语法自检:**\n" + "\n".join([f"- {e}" for e in errors + warnings])
                    full_response += report
                    message_placeholder.markdown(full_response)

            st.session_state.messages.append(HumanMessage(content=prompt))
            st.session_state.messages.append(AIMessage(content=full_response))

        except Exception as e:
            st.error(f"⚠️ 发生错误: {str(e)}")
            st.caption("建议检查 API Key 额度或网络连接。")