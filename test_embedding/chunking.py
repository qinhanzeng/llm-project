from langchain_text_splitters import RecursiveCharacterTextSplitter

document = """Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by allowing them to access external knowledge bases during inference. Instead of relying solely on the information encoded in model parameters during training, RAG systems retrieve relevant documents or passages from a knowledge base and provide them as context to the language model. This approach has several advantages: it allows models to access up-to-date information without retraining, reduces hallucinations by grounding responses in retrieved facts, and enables the model to cite sources. The retrieval step typically uses dense vector search, where both the query and documents are encoded as embeddings in a high-dimensional vector space, and similarity is measured using cosine distance or dot product. Chunking strategy is critical in RAG pipelines because it determines the granularity of retrieved information."""

def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = chunk_size - overlap
    total_length = len(text)
    chunked_text = []
    for i in range(0, total_length, step):
        chunked_text.append(text[i: i+chunk_size])

    return chunked_text


def recursive_chunk(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)
    return chunks

res = recursive_chunk(document)
#res = fixed_size_chunk(document, 200, 50)
for i, chunk in enumerate(res):
    print(f"Chunk {i+1} | 长度: {len(chunk)}")
    print(f"内容: {chunk}")
    print("---")

