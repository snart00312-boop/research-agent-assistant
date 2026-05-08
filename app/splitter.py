from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    把长文本切分成多个 chunk。

    参数：
    text: 原始长文本
    chunk_size: 每个 chunk 的最大长度
    chunk_overlap: 相邻 chunk 之间重叠的长度

    返回：
    chunks: list[str]
    """

    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_text(text)

    return chunks


def print_chunks(chunks: list[str], preview_len: int = 80):
    """
    打印每个 chunk 的编号、长度和内容片段。
    """

    print(f"共切分出 {len(chunks)} 个 chunk")
    print()

    for index, chunk in enumerate(chunks, start=1):
        preview = chunk[:preview_len].replace("\n", " ")

        print(f"Chunk {index}")
        print(f"长度：{len(chunk)}")
        print(f"内容片段：{preview}...")
        print("-" * 50)