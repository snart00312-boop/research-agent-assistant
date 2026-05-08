import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_client() -> OpenAI:
    api_key = os.getenv("ZHIPU_API_KEY")
    base_url = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

    if not api_key:
        raise ValueError("缺少 ZHIPU_API_KEY，请检查 .env 文件")

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def call_llm(prompt: str) -> str:
    client = get_client()
    model = os.getenv("ZHIPU_MODEL", "glm-4.5-air")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个帮助用户学习大模型应用开发的助手，回答要清晰、简洁、适合初学者。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.3,
        max_tokens=1024,
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    return response.choices[0].message.content