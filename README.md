# LLM Project

## Semantic Search Demo
A semantic search demo using OpenAI Embeddings and pgvector.

## Features
- Batch insert documents with vector embeddings
- Cosine similarity-based semantic search
- Display similarity scores for search results

## Tech Stack
- Python, OpenAI API, PostgreSQL, pgvector, psycopg3

## Getting Started
1. Set environment variable `OPENAI_API_KEY`
2. Start PostgreSQL and enable pgvector extension
3. Insert documents: `python -m test_embedding.insert_documents`
4. Run search: `python -m test_embedding.semantic_search`

## Example
```python
semantic_search_by_query("how to speed up vector search")
# #1 similarity: 0.6375 | 向量数据库通过近似最近邻算法加速相似度搜索
# #2 similarity: 0.5663 | HNSW索引通过分层图结构实现高效的向量检索
```