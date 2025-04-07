import requests
import re
from extract_store import generate_embeddings  # From File 1
import torch
import torch.nn.functional as F
def search_wikipedia(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['query']['search']
    else:
        return []

def clean_html_tags(text):
    return re.sub(r'<.*?>', '', text)


def apply_rag_wiki_q(wiki_content):
            query_embedding = generate_embeddings(query)
            paragraphs = wiki_content.split("newResult")
            paragraph_scores = []
            for paragraph in paragraphs:
                if paragraph.strip():  # Skip empty paragraphs
                    paragraph_embedding = generate_embeddings(paragraph)
                    score = F.cosine_similarity(
                        torch.tensor(query_embedding), torch.tensor(paragraph_embedding), dim=0
                    ).item()
                    paragraph_scores.append((paragraph, score))
            most_relevant_paragraph = max(paragraph_scores, key=lambda x: x[1])[0]
            extra_context = [p[0] for p in sorted(paragraph_scores, key=lambda x: -x[1])[1:4]]
            doc4 = {"answer": most_relevant_paragraph, "extra": extra_context}
            #doc4 = {"answer": most_relevant_paragraph}

            print(doc4)


query = "When was Georg Wilhelm Friedrich Hegel born?"

results = search_wikipedia(query)

clean_content = ""
if results:
    for result in results:
        title = clean_html_tags(result['title'])
        snippet = clean_html_tags(result['snippet'])
        clean_content += (f"newResult {title} - {snippet} \n")
else:
    print("No results found or an error occurred.")

apply_rag_wiki_q(clean_content)