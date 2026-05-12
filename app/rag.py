from langchain_core.documents import Document

from llm import call_llm
from vectorstore import retrieve_chunks


RetrievedChunk = tuple[Document, float]


def build_rag_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    """
    Build the final prompt for RAG generation.

    The model only sees the retrieved chunks here, so the answer should stay
    grounded in the local document content instead of external knowledge.
    """
    context_parts = []

    for index, (doc, score) in enumerate(retrieved_chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        context_parts.append(
            f"""[{index}]
来源文件: {source}
chunk_id: {chunk_id}
距离分数: {score}
内容:
{doc.page_content}"""
        )

    context = "\n\n".join(context_parts)

    return f"""
你是一个研究生科研资料助手。
下面的参考文档片段是检索系统根据用户问题选出的候选内容。
请优先根据这些片段回答用户问题。
如果片段中出现了与问题相关的标题、段落、关键词或事实，必须基于这些信息总结回答。
只有在片段内容和问题完全无关时，才回答：文档中没有提到相关内容。

## 参考文档片段
{context}

## 用户问题
{question}

## 回答要求
1. 只基于参考文档片段回答，不要引入外部知识。
2. 在答案中的关键句后使用 [1]、[2] 这样的编号标注来源。
3. 如果多个片段都支持同一句话，可以写成 [1][2]。
4. 回答末尾必须列出“引用来源”，对应每个编号的来源文件和 chunk_id。
5. 不要轻易拒答。用户问题和文档原文表述不完全相同，也应该根据相关段落进行概括。
6. 如果文档片段中有“后续计划”“项目目标”“核心功能”等标题，用户询问对应内容时应直接总结这些段落。

## 回答
""".strip()


def format_citations(retrieved_chunks: list[RetrievedChunk]) -> str:
    lines = ["引用来源："]

    for index, (doc, _) in enumerate(retrieved_chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        lines.append(f"{index}. {source}，chunk_id: {chunk_id}")

    return "\n".join(lines)


def generate_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "没有检索到相关内容，无法基于文档回答。"

    prompt = build_rag_prompt(question, retrieved_chunks)
    answer = call_llm(prompt).strip()

    if "引用来源" not in answer:
        answer = f"{answer}\n\n{format_citations(retrieved_chunks)}"

    return answer


def ask_with_rag(question: str, top_k: int = 3) -> tuple[str, list[RetrievedChunk]]:
    retrieved_chunks = retrieve_chunks(question, top_k=top_k)
    answer = generate_answer(question, retrieved_chunks)

    return answer, retrieved_chunks


def print_rag_answer(question: str, top_k: int = 3) -> None:
    answer, retrieved_chunks = ask_with_rag(question, top_k=top_k)

    print("\nRAG 回答：")
    print("=" * 60)
    print(answer)

    print("\n检索到的片段：")
    print("=" * 60)

    if not retrieved_chunks:
        print("没有检索到相关内容")
        return

    for index, (doc, score) in enumerate(retrieved_chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        preview = doc.page_content[:300].replace("\n", " ")

        print(f"[{index}] 距离分数: {score}")
        print(f"来源文件: {source}")
        print(f"chunk_id: {chunk_id}")
        print(f"内容片段: {preview}")
        print("-" * 60)
