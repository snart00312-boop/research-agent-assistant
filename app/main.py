from rag import print_rag_answer

def main():
    print("Research Agent Assistant - Day 9 Multi-format RAG QA")
    print("输入问题后，系统会从 txt / md / pdf 入库内容中检索片段，并生成带引用的回答。")
    print("输入 q 退出。")

    while True:
        question = input("\n请输入你的问题：").strip()

        if question.lower() in ["q", "quit", "exit"]:
            print("已退出。")
            break

        if not question:
            print("问题不能为空。")
            continue

        try:
            print_rag_answer(question, top_k=3)
        except Exception as e:
            print(f"\nRAG 问答失败：{e}")


if __name__ == "__main__":
    main()
