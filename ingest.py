import os
from typing import List, Dict
from preprocessor import preprocess
from config import DATA_PATH


def read_documents() -> List[Dict]:
    docs = []
    if not os.path.exists(DATA_PATH):
        return docs

    for filename in os.listdir(DATA_PATH):
        if not filename.endswith('.txt'):
            continue

        filepath = os.path.join(DATA_PATH, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        title = filename.replace('.txt', '').replace('_', ' ')
        docs.append({
            'title': title,
            'content': content,
            'source': filename
        })

    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                chunk = text[start:start + break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 50]


def process_documents() -> List[Dict]:
    docs = read_documents()
    processed = []

    for doc in docs:
        content = preprocess(doc['content'])
        chunks = chunk_text(content)

        for i, chunk in enumerate(chunks):
            processed.append({
                'id': f"{doc['source']}_{i}",
                'title': doc['title'],
                'content': chunk,
                'source': doc['source'],
                'chunk_index': i
            })

    return processed


if __name__ == "__main__":
    print("Processing documents...")
    chunks = process_documents()

    print(f"\nTotal documents: {len(os.listdir(DATA_PATH))}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print(f"\nSample chunk:")
        print(f"ID: {chunks[0]['id']}")
        print(f"Title: {chunks[0]['title']}")
        print(f"Content: {chunks[0]['content'][:200]}...")
