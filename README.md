# باحث — Bahith

A semantic search engine built specifically for Arabic content. It combines Arabic-optimized embeddings, hybrid search (semantic + keyword), and LLM-powered answer generation with a clean RTL web interface.

Arabic text search is an underserved problem. Most search tools are built for English and don't handle Arabic's unique characteristics—diacritics, letter variants, right-to-left text. Bahith addresses this with proper Arabic preprocessing, multilingual embeddings, and a native Arabic interface.

## How It Works

```
User query (Arabic)
    ↓
Arabic preprocessing (normalize, remove diacritics)
    ↓
Parallel search:
    ├─ Semantic: embed query → ChromaDB similarity
    └─ Keyword: BM25 scoring
    ↓
Hybrid ranking (weighted fusion)
    ↓
Top-K results
    ↓
LLM generates answer using results as context
    ↓
Return: AI answer + ranked results
```

The frontend is a single-page Arabic-first interface. Full RTL support, search mode switching, color-coded relevance scores, and an AI-generated answer box with source attribution.

## Features

- **Three search modes**: Semantic (meaning-based), keyword (BM25), and hybrid (combines both)
- **Arabic text preprocessing**: Diacritics removal, letter normalization (alef variants, taa marbuta)
- **LLM answer generation**: RAG-style answers grounded in search results
- **Web crawler**: Crawl and index any Arabic website
- **Clean Arabic UI**: RTL layout, Arabic typography, responsive design

## Tech Stack

- **Embeddings**: sentence-transformers (Arabic-optimized model)
- **Vector store**: ChromaDB (local, persistent)
- **Keyword search**: rank-bm25
- **LLM**: OpenAI-compatible API
- **Backend**: FastAPI + uvicorn
- **Frontend**: Vanilla HTML/CSS/JS

## Setup

```bash
# Clone the repo
git clone https://github.com/SanaAraj/bahith-search.git
cd bahith-search

# Install dependencies
pip install -r requirements.txt

# Create data folder and seed with sample content
python seed_data.py

# Build the search indices
python embeddings.py

# Configure the LLM (create .env from template)
cp .env.example .env
# Edit .env with your API key

# Run the server
python main.py
```

Open http://localhost:8000 in your browser.

## Adding More Content

The system can index any Arabic text. Beyond the Wikipedia seed data, you can crawl websites:

```python
from crawler import crawl_url

# Crawl a single page
crawl_url('https://ar.wikipedia.org/wiki/موضوع', 'ويكيبيديا')

# Then re-index
python embeddings.py
```

Or add text files directly to the `data/` folder and re-run `python embeddings.py`.

## Usage

1. Enter an Arabic query in the search box
2. Select a search mode:
   - **هجين (Hybrid)**: Best of both—combines semantic understanding with exact matches
   - **بحث دلالي (Semantic)**: Finds conceptually similar content even with different wording
   - **بحث كلمات (Keyword)**: Traditional keyword matching, good for exact phrases
3. View the AI-generated answer and browse the ranked results

## Hybrid Search

The hybrid approach combines semantic and keyword scores:

```
final_score = α × semantic_score + (1 - α) × keyword_score
```

Default α is 0.7, favoring semantic understanding while still boosting exact keyword matches. Both scores are normalized to [0, 1] before combining.

## API

### POST /search

```json
{
  "query": "ما هي المملكة العربية السعودية",
  "mode": "hybrid",
  "top_k": 5
}
```

Response:

```json
{
  "query": "ما هي المملكة العربية السعودية",
  "answer": "المملكة العربية السعودية هي أكبر دولة في شبه الجزيرة العربية...",
  "results": [
    {
      "title": "المملكة العربية السعودية",
      "snippet": "...",
      "source": "المملكة_العربية_السعودية.txt",
      "score": 0.95
    }
  ],
  "total_results": 5,
  "search_time": 0.34
}
```

### GET /health

Returns `{"status": "ok"}` if the server is running.

## Configuration

Environment variables (`.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for LLM | required |
| `OPENAI_BASE_URL` | API endpoint | https://api.groq.com/openai/v1 |
| `MODEL_NAME` | LLM model ID | llama-3.1-8b-instant |
| `EMBEDDING_MODEL` | Embedding model | Omartificial-Intelligence-Space/Matryoshka-Arabic-STS-V1 |
| `SEARCH_ALPHA` | Hybrid weighting | 0.7 |
| `TOP_K` | Default results count | 5 |

Works with any OpenAI-compatible API provider (OpenAI, Together, Fireworks, etc).

## Project Structure

```
bahith-search/
├── main.py          # FastAPI app
├── search.py        # Hybrid search logic
├── embeddings.py    # Vector store operations
├── bm25.py          # BM25 keyword search
├── generate.py      # LLM answer generation
├── preprocessor.py  # Arabic text preprocessing
├── ingest.py        # Data ingestion pipeline
├── crawler.py       # Web crawler for Arabic sites
├── seed_data.py     # Sample data generator
├── config.py        # Configuration
├── static/
│   └── index.html   # Frontend
├── requirements.txt
└── .env.example
```
