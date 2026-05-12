from pathlib import Path
import fitz


def load_txt(path: str) -> str:
    """
    读取 txt 文件内容。

    Args:
        path: txt 文件路径

    Returns:
        文件中的文本内容
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if file_path.suffix.lower() != ".txt":
        raise ValueError("当前函数只支持读取 .txt 文件")

    return file_path.read_text(encoding="utf-8")


def load_markdown(path: str) -> str:
    """
    读取 Markdown 文件内容。

    Args:
        path: Markdown 文件路径，支持 .md / .markdown

    Returns:
        Markdown 文件中的文本内容
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if file_path.suffix.lower() not in [".md", ".markdown"]:
        raise ValueError("当前函数只支持读取 .md 或 .markdown 文件")

    return file_path.read_text(encoding="utf-8")


def load_pdf(path: str) -> str:
    """
    读取 PDF 文件内容。

    Args:
        path: PDF 文件路径

    Returns:
        PDF 中提取出来的文本内容
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("当前函数只支持读取 .pdf 文件")

    doc = fitz.open(path)

    texts = []

    for page_index, page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            texts.append(f"\n\n[Page {page_index + 1}]\n{text}")

    doc.close()

    return "\n".join(texts)


def load_document(path: str) -> str:
    """
    统一文档读取入口。

    根据文件后缀自动调用不同的读取函数：
    - .txt -> load_txt()
    - .md / .markdown -> load_markdown()
    - .pdf -> load_pdf()

    Args:
        path: 文档路径

    Returns:
        文档文本内容
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return load_txt(path)

    if suffix in [".md", ".markdown"]:
        return load_markdown(path)

    if suffix == ".pdf":
        return load_pdf(path)

    raise ValueError(f"不支持的文件格式: {suffix}")