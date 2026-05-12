import argparse
from pathlib import Path

from loader import load_document
from splitter import split_text
from vectorstore import build_documents, save_documents_to_faiss


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_document_path(path: str) -> Path:
    file_path = Path(path)

    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path

    return file_path


def ingest_document(path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
    ingest_documents(
        paths=[path],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def ingest_documents(paths: list[str], chunk_size: int = 500, chunk_overlap: int = 100) -> None:
    all_documents = []

    for path in paths:
        file_path = resolve_document_path(path)

        print("=" * 60)
        print(f"准备读取文档：{file_path}")

        text = load_document(str(file_path))
        print(f"文档读取完成，文本长度：{len(text)}")

        chunks = split_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        print(f"文本切分完成，chunk 数量：{len(chunks)}")

        documents = build_documents(
            chunks=chunks,
            source=str(file_path),
        )
        all_documents.extend(documents)

    print("=" * 60)
    print(f"准备写入向量库，总 document 数量：{len(all_documents)}")
    save_documents_to_faiss(all_documents)


def preview_document(path: str) -> None:
    file_path = resolve_document_path(path)
    print(f"准备读取文档：{file_path}")

    text = load_document(str(file_path))
    print(f"文档读取完成，文本长度：{len(text)}")
    print(text[:500])


def main() -> None:
    parser = argparse.ArgumentParser(description="Load documents and save them into FAISS.")
    parser.add_argument("paths", nargs="+", help="文档路径，支持 txt / md / pdf，可传入多个")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    ingest_documents(
        paths=args.paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
