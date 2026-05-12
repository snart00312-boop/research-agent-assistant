# Research Agent Assistant

面向研究生学习和科研场景的 RAG + Agent 应用。支持上传文档资料，通过检索增强生成（RAG）的方式基于资料内容进行问答，并返回引用来源。

## 技术栈

| 层 | 技术 |
|----|------|
| 大模型 API | 智谱 GLM-4.5-Air（OpenAI SDK 兼容） |
| Embedding | 智谱 embedding-3（1024 维） |
| LLM 框架 | LangChain |
| 文本切分 | RecursiveCharacterTextSplitter |
| 向量库 | FAISS |
| 运行环境 | Python 3.10+, venv |

## 项目结构

```
research-agent-assistant/
  app/
    main.py          # 命令行入口
    llm.py           # 大模型 API 调用
    loader.py        # 文档读取（txt）
    splitter.py      # 文本切分
    vectorstore.py   # 向量库（embedding + FAISS 存取 + 检索）
    rag.py           # RAG 问答（检索结果 → LLM 生成 + 引用）
  data/
    docs/
      test.txt       # 测试文档
  vectorstore/       # FAISS 索引持久化目录
  requirements.txt
  .env.example
  README.md
 大模型计划.md      # 详细开发计划（Day 1 ~ Day 15）
```

## 安装与运行

**1. 克隆项目**

```bash
git clone <repo-url>
cd research-agent-assistant
```

**2. 创建虚拟环境**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**3. 安装依赖**

```bash
pip install -r requirements.txt
```

**4. 配置 API Key**

```bash
cp .env.example .env
```

编辑 `.env`，填入你的智谱 API Key：

```env
ZHIPU_API_KEY=你的key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL=glm-4.5-air
```

**5. 运行**

```bash
python app/main.py
```

## 当前功能（截至 Day 8）

- 调用智谱 GLM-4.5-Air 大模型进行问答
- 读取 txt 文档，将文档内容作为上下文
- 使用 RecursiveCharacterTextSplitter 将长文档切分为 chunk
- 调用智谱 embedding-3 生成文本向量（1024 维）
- 使用 FAISS 存储和检索向量
- 根据用户问题检索 top-k 个相关 chunk，展示分数和来源
- 将检索到的 chunk 拼接进 prompt，生成基于文档的 RAG 回答
- 在回答中使用 `[1]`、`[2]` 标注引用，并列出来源文件和 chunk_id

## RAG 流程

```
文档 (txt)
  ↓ loader.load_txt()
原始文本
  ↓ splitter.split_text()
chunks
  ↓ vectorstore.save_chunks_to_faiss()
embedding → FAISS 索引
  ↓ vectorstore.retrieve_chunks()
检索结果（top-k chunks + 相似度分数）
  ↓ rag.build_rag_prompt()
带引用要求的 RAG prompt
  ↓ llm.call_llm()
答案 + 引用来源
```

## 各模块说明

### llm.py — 大模型调用

- `get_client()` — 创建 OpenAI 兼容客户端，连接智谱 API
- `call_llm(prompt)` — 传入 prompt，返回模型回答

### loader.py — 文档加载

- `load_txt(path)` — 读取 txt 文件，返回文本内容

### splitter.py — 文本切分

- `split_text(text, chunk_size, chunk_overlap)` — 将文本切分为 chunk
- `print_chunks(chunks)` — 打印每个 chunk 的长度和内容预览

### vectorstore.py — 向量库

- `ZhipuAIEmbeddings` — 自定义 embedding 类，封装智谱 embedding-3 API
- `save_chunks_to_faiss(chunks, source)` — 将 chunks 向量化并存入 FAISS
- `retrieve_chunks(question, top_k)` — 检索与问题相关的 top-k 个 chunk
- `print_retrieval_results(question, top_k)` — 格式化打印检索结果
- `load_faiss()` — 从本地加载 FAISS 索引

### rag.py — RAG 问答

- `build_rag_prompt(question, retrieved_chunks)` — 将问题和检索片段组织成 RAG prompt
- `generate_answer(question, retrieved_chunks)` — 调用大模型生成带引用的答案
- `ask_with_rag(question, top_k)` — 完成“检索 → 生成”的闭环
- `print_rag_answer(question, top_k)` — 在命令行打印答案、引用和检索片段

## 开发进度

| Day | 内容 | 状态 |
|-----|------|------|
| Day 1 | 项目初始化，创建目录结构 | ✅ |
| Day 2 | 调用大模型 API，命令行问答 | ✅ |
| Day 3 | 读取 txt 文档，基于文档问答 | ✅ |
| Day 4 | 整理代码结构，模块化分离 | ✅ |
| Day 5 | 文本切分 (RecursiveCharacterTextSplitter) | ✅ |
| Day 6 | 向量存储 (embedding + FAISS) | ✅ |
| Day 7 | 向量检索 (相似度搜索 + top-k) | ✅ |
| Day 8 | RAG 问答闭环 + 引用溯源 | ✅ |
| Day 9 | 多格式文档加载 (PDF/MD) | 🔲 |
| Day 10 | 查询改写 (Query Rewriting) | 🔲 |
| Day 11 | 混合检索 (BM25 + FAISS + RRF) | 🔲 |
| Day 12 | 重排序 (Cross-Encoder Reranking) | 🔲 |
| Day 13 | 评估体系（手写指标） | 🔲 |
| Day 14 | 索引优化 + 消融实验 | 🔲 |
| Day 15 | Streamlit 可视化 Demo | 🔲 |

> 详细计划（每 Day 的做什么/为什么/怎么做/验收标准）见 [大模型计划.md](./大模型计划.md)

## 技术取舍

- **FAISS vs Chroma**：Chroma 在 Windows 上存在 onnxruntime DLL 兼容性问题（0xC0000005 崩溃），切换到 FAISS 后解决了该问题。FAISS 由 Meta 维护，在 Windows 上更稳定，性能相当。
- **embedding-3 vs text-embedding-ada-002**：选用智谱 embedding-3，1024 维度在精度和存储成本间取得平衡，且与 LLM 使用同一 API 平台，配置更简单。

## 面试讲法

**这个项目是做什么的？**

面向研究生科研场景的 RAG + Agent 应用。用户可以上传论文、课程资料，系统对文档进行解析、切分、向量化存储，然后根据用户问题检索相关片段，调用大模型生成带引用的回答。

**为什么用 RAG？**

直接让大模型回答时，它不知道用户上传的私有文档内容，也容易产生幻觉。RAG 先从本地向量库中检索和问题相关的片段，再把这些片段作为上下文交给模型回答，让回答更贴近文档内容，同时可以追溯引用来源。

**为什么用 FAISS？**

最初使用 Chroma，但在 Windows 上遇到 onnxruntime 的 C++ DLL 初始化崩溃问题（0xC0000005）。切换到 FAISS 后稳定性大幅提升，且 FAISS 由 Meta 维护，在向量检索性能上有成熟验证。
