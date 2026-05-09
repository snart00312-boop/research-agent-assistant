from pathlib import Path

from llm import call_llm
from loader import load_txt
from splitter import split_text, print_chunks
from vectorstore import save_chunks_to_faiss



def build_prompt(document: str, question: str) -> str:
    prompt = f"""
你是一个研究生科研资料助手。
请严格根据下面提供的文档内容回答用户问题。
如果文档中没有相关信息，请回答：文档中没有提到相关内容。

文档内容：
{document}

用户问题：
{question}

请基于文档回答：
"""
    return prompt


# def main():
#     project_root = Path(__file__).resolve().parent.parent
#     doc_path = project_root / "data" / "docs" / "test.txt"
#
#     document = load_txt(str(doc_path))
#
#     print("研究生科研资料 Agent")
#     print("已加载文档：data/docs/test.txt")
#     print("输入 exit 退出程序")
#
#     while True:
#         question = input("\n请输入你的问题：").strip()
#
#         if not question:
#             print("问题不能为空，请重新输入。")
#             continue
#
#         if question.lower() in ["exit", "quit"]:
#             print("已退出")
#             break
#
#         prompt = build_prompt(document, question)
#
#         try:
#             answer = call_llm(prompt)
#         except Exception as e:
#             print(f"\n调用模型失败：{e}")
#             continue
#
#         print("\n模型回答：")
#         print(answer)



def main():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "docs" / "test.txt"

    text = load_txt(file_path)

    chunks = split_text(
        text,
        chunk_size=500,
        chunk_overlap=100,
    )

    print_chunks(chunks)

    save_chunks_to_faiss(
        chunks=chunks,
        source=str(file_path),
    )


if __name__ == "__main__":
    main()