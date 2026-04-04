import openai
import os
from db import get_conn
from pgvector.psycopg import register_vector
import psycopg
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def semantic_search_by_query(query, top_k=3):
    conn = get_conn()
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    
    embbeded_query = response.data[0].embedding

    sql = """SELECT description, 1 - (test_embedding.embedded_description <=> %s::vector) as similarity FROM test_embedding 
    ORDER BY test_embedding.embedded_description <=> %s::vector LIMIT %s"""

    with conn.cursor() as cur:
        cur.execute(sql, (embbeded_query, embbeded_query, top_k))
        results = cur.fetchall()
        for rank, (content, similarity) in enumerate(results, start=1):
            print(f"#{rank} 相似度 for {query}: {similarity:.4f} | {content}")


semantic_search_by_query("怎么加速向量搜索")
semantic_search_by_query("如何让模型回答更准确")
semantic_search_by_query("怎么构建AI应用")