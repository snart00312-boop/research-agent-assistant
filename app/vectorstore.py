import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = str(PROJECT_ROOT / "vectorstore")
INDEX_NAME = "faiss_index"


class ZhipuAIEmbeddings(Embeddings):

    def __init__(self, model: str = "embedding-3", dimensions: int = 1024):
        load_dotenv(override=True)

        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("请先在 .env 中配置 ZHIPU_API_KEY")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/",
        )
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 64

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            response = self.client.embeddings.create(
                model=self.model,
                input=batch_texts,
                dimensions=self.dimensions,
            )

            data = sorted(response.data, key=lambda item: item.index)
            batch_embeddings = [item.embedding for item in data]

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_embedding_model():
    return ZhipuAIEmbeddings(
        model="embedding-3",
        dimensions=1024,
    )


def build_documents(chunks: list[str], source: str = "data/docs/test.txt"):
    documents = []

    for index, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source,
                "chunk_id": index,
            },
        )
        documents.append(doc)

    return documents


def save_chunks_to_faiss(chunks: list[str], source: str = "data/docs/test.txt"):
    if not chunks:
        print("没有 chunks 可以写入向量库")
        return None

    print("步骤 1：准备创建 embedding 模型", flush=True)
    embeddings = get_embedding_model()

    print("步骤 2：准备构建 documents", flush=True)
    documents = build_documents(chunks, source=source)

    print("步骤 3：测试 embedding 是否能正常调用", flush=True)
    test_vector = embeddings.embed_query("这是一个测试文本")
    print(f"embedding 调用成功，向量维度：{len(test_vector)}", flush=True)

    print("步骤 4：准备写入 FAISS", flush=True)

    vectorstore_path = Path(VECTORSTORE_DIR)
    vectorstore_path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    vectorstore.save_local(
        folder_path=VECTORSTORE_DIR,
        index_name=INDEX_NAME,
    )

    print(f"成功写入 {len(documents)} 个 chunks 到 FAISS", flush=True)
    print(f"向量库保存目录：{VECTORSTORE_DIR}", flush=True)

    return vectorstore


def load_faiss():
    embeddings = get_embedding_model()
    return FAISS.load_local(
        folder_path=VECTORSTORE_DIR,
        embeddings=embeddings,
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )


def retrieve_chunks(question: str, top_k: int = 3):
    """
    根据用户问题，从 FAISS 中检索 top_k 个相关 chunk。
    """
    vectorstore = load_faiss()

    results = vectorstore.similarity_search_with_score(
        query=question,
        k=top_k,
    )

    return results


def print_retrieval_results(question: str, top_k: int = 3):
    """
    打印检索结果，用于 Day 7 验收。
    """
    results = retrieve_chunks(question, top_k=top_k)

    if not results:
        print("没有检索到相关内容")
        return

    print("\n检索结果：")
    print("=" * 60)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"\n[{i}] 距离分数: {score}")
        print("-" * 60)

        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        print(f"来源文件: {source}")
        print(f"chunk_id: {chunk_id}")
        print("内容片段:")
        print(doc.page_content[:500])
        print("-" * 60)
