from llm import call_llm


def main():
    print("Research Agent Assistant")
    print("输入 exit 退出程序")

    while True:
        question = input("\n请输入你的问题：")

        if question.lower() in ["exit", "quit", "q"]:
            print("已退出")
            break

        try:
            answer = call_llm(question)
            print("\n模型回答：")
            print(answer)
        except Exception as e:
            print("\n调用模型失败：")
            print(e)


if __name__ == "__main__":
    main()