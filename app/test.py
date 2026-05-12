import sys
from pathlib import Path

from loader import load_document


sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent

for filename in ["test.txt", "test.md", "test.pdf"]:
    file_path = BASE_DIR / "data" / "docs" / filename
    text = load_document(str(file_path))

    print("=" * 60)
    print(filename)
    print(f"文本长度：{len(text)}")
    print(text[:300])
