from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os


CONNECTION_STRING = "postgresql+psycopg://hanson@localhost:5432/postgres"
COLLECTION_NAME = "public"

def load_store_pdf_data():
    # 1. load pdf
    loader = PyPDFLoader("/Users/hanson/Project/AI_Project/llm_project/test_pdf.pdf")
    documents = loader.load()
    print(f"There are totally {len(documents)} pages")

    # 2. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )

    chunks = splitter.split_documents(documents)
    print(f"Chunking finished, there are {len(chunks)} of chunks")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ({len(chunk.page_content)} characters) ---")
        print(chunk.page_content)

    # 3. Store in pgvector
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING
    )

    print(f"Chunked documents have been saved in DB successfully! Totally {len(chunks)} of vectors")

def query_data(question):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING
    )

    results = vector_store.similarity_search(question, k = 3)

    print(f"\nQuering: {question}")
    print(f"Return most similar {len(results)} results")
    for i, doc in enumerate(results):
        print(f"\n--- {i + 1} place ---")
        print(doc.page_content)
        print(f"Source: {doc.metadata}")

    return results

def build_prompt(query, query_results):
    context = "\n\n".join([doc.page_content for doc in query_results])

    messages = [
        SystemMessage(content="""You are a helpful assistant. 
        Answer the question based only on the provided context.
        If the answer is not in the context, say "I don't know"."""),
        HumanMessage(content=f"""Context:
        {context}
        Question: {query}""")
    ]

    llm = ChatOpenAI(model = "gpt-4.1-mini", openai_api_key = os.environ.get("OPENAI_API_KEY"))
    response = llm.invoke(messages)

    print(f"Question is: {query}")
    print(f"\n Answer is: {response.content}")


question = "What is RAG?"
raw_query_results = query_data(question)
build_prompt(query=question, query_results=raw_query_results)