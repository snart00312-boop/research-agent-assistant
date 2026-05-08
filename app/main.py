from pathlib import Path

from llm import call_llm
from loader import load_txt


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


def main():
    project_root = Path(__file__).resolve().parent.parent
    doc_path = project_root / "data" / "docs" / "test.txt"

    document = load_txt(str(doc_path))

    print("研究生科研资料 Agent")
    print("已加载文档：data/docs/test.txt")
    print("输入 exit 退出程序")

    while True:
        question = input("\n请输入你的问题：")

        if question.lower() in ["exit", "quit"]:
            print("已退出")
            break

        prompt = build_prompt(document, question)
        answer = call_llm(prompt)

        print("\n模型回答：")
        print(answer)


if __name__ == "__main__":
    main()