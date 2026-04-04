import openai
import os
import psycopg
from pgvector.psycopg import register_vector

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

documents = [
    "Transformer架构通过自注意力机制处理序列数据",
    "RAG结合了检索系统和生成模型来提升回答质量",
    "向量数据库通过近似最近邻算法加速相似度搜索",
    "Fine-tuning在预训练模型基础上针对特定任务继续训练",
    "LangChain是一个用于构建LLM应用的开发框架",
    "HNSW索引通过分层图结构实现高效的向量检索",
    "Prompt engineering通过精心设计输入来优化模型输出",
    "Embedding将文本映射到高维向量空间保留语义信息",
]
def ingest_single_sentence():
    with psycopg.connect(
        dbname="postgres",
        user="hanson",
        password="",
        host="localhost",
        port=5432
    ) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for doc in documents:
                response = client.embeddings.create(
                    input = doc,
                    model = "text-embedding-3-small"
                )

                embedding = response.data[0].embedding

                cur.execute(
                    "INSERT INTO test_embedding(description, embedded_description) VALUES (%s, %s)",
                    (doc, embedding)
                )

                print(f"Inserted document: '{doc}' with embedding length {len(embedding)}")

            conn.commit()

    print("All documents have been inserted into the database with their embeddings.")

def ingest_documents():
     with psycopg.connect(
        dbname="postgres",
        user="hanson",
        password="",
        host="localhost",
        port=5432
    ) as conn:
         with conn.cursor() as cur:
            response = client.embeddings.create(
                input = documents,
                model="text-embedding-3-small"
            )
            
            data = [(doc, emb_obj.embedding) for doc, emb_obj in zip(documents, response.data)]
            cur.executemany("INSERT INTO test_embedding (description, embedded_description) values (%s, %s)", data)

            conn.commit()

ingest_documents()


