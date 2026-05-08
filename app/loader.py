from pathlib import Path


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
        raise ValueError("当前只支持读取 .txt 文件")

    return file_path.read_text(encoding="utf-8")