# Research Agent Assistant

一个面向研究生学习和科研场景的 RAG + Agent 应用。

## 当前进度

## Day 1：完成项目初始化
- 已创建基础目录结构
- 已创建命令行入口 app/main.py


## Day 2：调用大模型 API

已完成：

- 使用 `python-dotenv` 读取环境变量
- 使用 OpenAI SDK 调用智谱 GLM-4.5-Air
- 封装 `call_llm(prompt)` 函数
- 在命令行中输入问题并获得模型回答

当前模型配置：

```env
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_MODEL=glm-4.5-air
```

## 运行方式

```bash
python app/main.py
