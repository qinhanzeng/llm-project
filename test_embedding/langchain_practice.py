from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, RateLimitError, APITimeoutError
import os


CONNECTION_STRING = "postgresql+psycopg://hanson@localhost:5432/postgres"
COLLECTION_NAME = "public"


def get_embeddings():
    """Create a unique embeddings to reuse it"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI API KEY is not set")
    
    return OpenAIEmbeddings(openai_api_key=api_key)

def load_store_pdf_data(file_path: str):
    """Error Field 1: file loading"""
    # 1. load pdf
    if not os.path.exists(file_path):
        print(f"[Error] No such a file: {file_path}")
        return
    
    try:
        loader = PyPDFLoader("/Users/hanson/Project/AI_Project/llm_project/test_pdf.pdf")
        documents = loader.load()
        print(f"There are totally {len(documents)} pages")

    except Exception as e:
        print(f"Error while loading PDF: {e}")
        return


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

    """Error Field 2 + 3: OpenAI embedding + pgvector storing data"""

    # 3. Store in pgvector
    try:
        embeddings = get_embeddings()
        vector_store = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING
        )

        print(f"Chunked documents have been saved in DB successfully! Totally {len(chunks)} of vectors")
    except AuthenticationError:
        print("[Error]: Invalid OpenAI API Key")
    except RateLimitError:
        print("[Error]: Rate limited, please try it later")
    except Exception as e:
        print(f"[Error]: Batch ingestion error: {e}")


def load_store_pef_batch(pdf_paths: list[str]):
    """Batch ingest mutiple pdf files"""
    all_chunks = []

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"No such file: {path}, skip...")
            continue
        try:
            loader = PyPDFLoader(path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 200,
                chunk_overlap = 50
            )

            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"[OK] {path} -> {len(chunks)} chunks")

        except Exception as e:
            print(f"Skip {path}, load failed: {e}")
            continue

    if not all_chunks:
        print("[Error] No document can be ingested")
        return
    
    try:
        embeddings = get_embeddings()
        PGVector.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING
        )
        print(f"Batch ingestion finished, totally {len(all_chunks)} chunks")
    except AuthenticationError:
        print("[Error] Invalid OpenAI API key")
    except RateLimitError:
        print("[Error] Rate limied, please try it later")
    except Exception as e:
        print("[Error] Batch ingestion failed: {e}")

def query_data(question):
    # Input check
    if not question or not question.strip():
        print("[Error] Input cannot be empty")
        return
    
    # Error Field 2 + 3: Embedding query + pgvector search
    try:
        embeddings = get_embeddings()
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING
        )
        results = vector_store.similarity_search_with_score(question, k = 3)
    except AuthenticationError:
        print("[Error] Invalid OpenAI API Key")
        return []
    except Exception as e:
        print(f"[Error] Query Failed: {e}")
        return []

    print(f"\nQuering: {question}")
    print(f"Return most similar {len(results)} results")
    for i, (doc, score) in enumerate(results):
        print(f"\n--- {i + 1} place (similarity score: {score:.4f})---")
        print(doc.page_content)
        print(f"Source: {doc.metadata}")

    return results

def build_prompt(query, query_results):
    # Input check
    if not query_results:
        print("No query results, cannot create answer")
        return 

    context = "\n\n".join([doc.page_content for doc, score in query_results])

    messages = [
        SystemMessage(content="""You are a helpful assistant. 
        Answer the question based only on the provided context.
        If the answer is not in the context, say "I don't know"."""),
        HumanMessage(content=f"""Context:
        {context}
        Question: {query}""")
    ]
    try:
        llm = ChatOpenAI(model = "gpt-4.1-mini", openai_api_key = os.environ.get("OPENAI_API_KEY"))
        response = llm.invoke(messages)
    except AuthenticationError:
        print("[Error] Invalid OpenAI API Key")
    except RateLimitError:
        print("[Error] Rate limited, please try it later")
    except APITimeoutError:
        print("[Error] Timout for API call")
    except Exception as e:
        print(f"[Error] LLM call failed: {e}")
    print(f"Question is: {query}")
    print(f"\n Answer is: {response.content}")


question = "What is the price of pgvector?"
raw_query_results = query_data(question)
build_prompt(query=question, query_results=raw_query_results)