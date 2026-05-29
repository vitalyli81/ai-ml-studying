"""Minimal RAG chatbot — load docs, retrieve, answer with citations.

Deliberately naive. See README.md for the upgrades that turn this into a
portfolio piece. Concepts: ../../llms/rag.md
"""

import sys
from pathlib import Path

import chromadb
from anthropic import Anthropic

DOCS_DIR = Path(__file__).parent / "docs"
CHUNK_SIZE = 500  # characters; naive. Upgrade to token-based recursive splitting.
TOP_K = 3
MODEL = "claude-sonnet-4-6"

client = Anthropic()  # reads ANTHROPIC_API_KEY from env
chroma = chromadb.Client()  # in-memory; swap for PersistentClient to keep the index
collection = chroma.get_or_create_collection("docs")  # Chroma auto-embeds


def chunk(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Fixed-size character chunks. The simplest thing that works — and the
    first thing you'll replace once the eval harness shows it hurts recall."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def index_docs() -> int:
    """Load every .md/.txt in ./docs, chunk, embed, store. Run once per corpus."""
    ids, documents, metadatas = [], [], []
    for path in sorted(DOCS_DIR.glob("**/*")):
        if path.suffix not in {".md", ".txt"}:
            continue
        for j, piece in enumerate(chunk(path.read_text(encoding="utf-8"))):
            ids.append(f"{path.name}-{j}")
            documents.append(piece)
            metadatas.append({"source": path.name})  # for citations
    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)


def ask(question: str) -> dict:
    """Retrieve top-k chunks and generate a grounded, cited answer."""
    results = collection.query(query_texts=[question], n_results=TOP_K)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # Label each chunk with its source so the model can cite it.
    context = "\n\n".join(
        f"[Source: {src}]\n{text}" for src, text in zip(sources, chunks)
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(
            "Answer using ONLY the provided context. "
            'If the context does not contain the answer, say "I don\'t know." '
            "Always cite the source filename you used."
        ),
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
    )

    return {
        "answer": response.content[0].text,
        "sources": sorted(set(sources)),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python rag.py "your question"')
        sys.exit(1)

    n = index_docs()
    if n == 0:
        print(f"No documents found. Add .md/.txt files to {DOCS_DIR}/ first.")
        sys.exit(1)
    print(f"Indexed {n} chunks.\n")

    result = ask(" ".join(sys.argv[1:]))
    print(result["answer"])
    print(f"\nSources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
