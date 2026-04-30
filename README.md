## Tech Stack

| Layer | Tool |
|---|---|
| Document Loading | LangChain PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embedding | OpenAI text-embedding-3-small |
| Vector Store | pgvector (PostgreSQL) |
| LLM | GPT-4.1-mini via LangChain |

## Getting Started

**Prerequisites:** Python 3.10+, PostgreSQL with pgvector, OpenAI API key

```bash
pip install langchain langchain-community langchain-openai langchain-postgres openai psycopg
```

```sql
-- Enable pgvector in PostgreSQL
CREATE EXTENSION IF NOT EXISTS vector;
```

```bash
export OPENAI_API_KEY=your_key_here
```

Update `CONNECTION_STRING` in `main.py` to match your PostgreSQL setup.

## Usage

```python
# Ingest a single PDF
load_store_pdf_data("path/to/document.pdf")

# Ingest multiple PDFs (batch)
load_store_pdf_batch(["doc1.pdf", "doc2.pdf"])

# Query
results = query_data("What is the refund policy?")
build_prompt(query="What is the refund policy?", query_results=results)
```

## Key Design Decisions

**Why RecursiveCharacterTextSplitter?**  
Splits on natural boundaries (paragraphs → sentences → words) before falling 
back to character count — produces more semantically coherent chunks.

**Why pgvector over a dedicated vector DB?**  
For small-to-medium document sets, pgvector inside existing PostgreSQL needs 
no extra service. For millions of vectors, Pinecone/Weaviate would scale better.

**Chunk size 200, overlap 50 — why?**  
Small chunks keep each vector focused on one concept (better retrieval precision). 
Overlap prevents answers being cut off at chunk boundaries.

## Error Handling

- Missing or invalid `OPENAI_API_KEY`
- PDF file not found or corrupted  
- OpenAI API errors (auth, rate limit, timeout)
- pgvector connection failure
- Empty query input
- Batch ingestion: one failed file does not block others

## Roadmap

- [ ] Support `.txt` and `.docx` formats
- [ ] Add reranking (Cohere Rerank or cross-encoder)
- [ ] Expose as REST API (FastAPI)
- [ ] Pinecone integration for scale comparison
- [ ] Streaming LLM responses