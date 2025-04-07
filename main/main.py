import psycopg2
from extract_store import generate_embeddings  # From File 1
from wolframalpha import ask_wolframalpha
from wikipedia import fetch_wikipedia_content, extract_wikipedia_search_term ,extract_german_search_term,fetch_wikipedia_full_content_german,fetch_and_clean_wikipedia_content # From File 3
import torch
import torch.nn.functional as F
import numpy as np
from openaiAPI import query_gpt  # From File 2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def write_to_file(file_name, content):
    """
    Writes content to a file.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)
def fetch_from_knowledge_base(query_embedding, db_config, top_k=8):
    """
    Fetches the top_k most relevant documents from the knowledge base.
    """
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    try:
        query = """
        
        SELECT 
            id, 
            content, 
            1 - (embedding <=> %s::vector) AS cosine_similarity 
        FROM documents 
        ORDER BY cosine_similarity DESC 
        LIMIT %s;
        """
        
        # Ensure query_embedding is a list
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        cursor.execute(query, (query_embedding, top_k))
        results = cursor.fetchall()

        documents = [{"id": row[0], "content": row[1], "similarity": row[2]} for row in results]
        return documents
    finally:
        cursor.close()
        conn.close()

def apply_rag_pipeline(question, db_config):
    """
    Executes the RAG pipeline to retrieve and rank documents from the knowledge base and Wikipedia.
    """
    query_embedding = generate_embeddings(question)

    # Step 1: Fetch from Knowledge Base
    kb_docs = fetch_from_knowledge_base(query_embedding, db_config, top_k=15)
    doc3 = {"answer": kb_docs[0]["content"], "extra": [doc["content"] for doc in kb_docs[1:]]}

    # Step 2: Fetch from Wikipedia
    # search_term = extract_german_search_term(question)
    search_term = "Hegel"
    print("search term")
    print(search_term)
    if search_term:
        # title, wiki_content = fetch_wikipedia_full_content_german(search_term)
        wiki_content = fetch_and_clean_wikipedia_content(search_term)
        if wiki_content:
            paragraphs = wiki_content.split("\n")
            paragraph_scores = []
            for paragraph in paragraphs:
                if paragraph.strip():  # Skip empty paragraphs
                    paragraph_embedding = generate_embeddings(paragraph)
                    score = F.cosine_similarity(
                        torch.tensor(query_embedding), torch.tensor(paragraph_embedding), dim=0
                    ).item()
                    paragraph_scores.append((paragraph, score))
            most_relevant_paragraph = max(paragraph_scores, key=lambda x: x[1])[0]
            extra_context = [p[0] for p in sorted(paragraph_scores, key=lambda x: -x[1])[1:20]]
            doc4 = {"answer": most_relevant_paragraph, "extra": extra_context}
        else:
            doc4 = {"answer": None, "extra": []}
    else:
        doc4 = {"answer": None, "extra": []}

    # Step 3: Generate $doc1 and $doc2
    doc1 = doc3["answer"]
    doc2 = {"kb": doc3["extra"], "wiki": doc4["extra"]}
    doc5 = query_gpt(
    question,
    context="You are a helpful assistant answering user questions with accurate, concise, and detailed information. always answer in english",
)
    doc6 = ask_wolframalpha(question)
    write_to_file(
    "doc-KB.txt",
    f"Answer: {doc3['answer']}\n\nExtra:\n" + "\n\n".join(doc3['extra']))
    write_to_file(
    "doc-wiki.txt",
    f"Answer: {doc4['answer']}\n\nExtra:\n" + "\n\n".join(doc4['extra']))
    write_to_file(
    "doc-openai.txt", doc5)
    write_to_file(
    "doc-wolfaramalpha.txt", doc6)
    return {"$doc1": doc1, "$doc2": doc2, "$doc3": doc3, "$doc4": doc4}

# Example usage
db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

question = "When was Hegel born?" 
results = apply_rag_pipeline(question, db_config)

# print("\nRAG Output:")
# for key, value in results.items():
#     print(f"{key}: {value}")
