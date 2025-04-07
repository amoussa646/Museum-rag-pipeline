import requests
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
import json


# Initialize tokenizer and model for embedding generation
tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
model = AutoModel.from_pretrained('thenlper/gte-base')

# Utility: Normalize and pool embeddings
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_embeddings(text, metadata={}):
    """
    Generates embeddings for the given text using a pre-trained model.
    """
    combined_text = " ".join([text] + [v for k, v in metadata.items() if isinstance(v, str)])
    inputs = tokenizer(combined_text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    embeddings = average_pool(outputs.last_hidden_state, attention_mask)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()[0]  # Return as a numpy array


# Utility: Fetch Wikipedia content
def fetch_wikipedia_content(search_term):
    """
    Fetches the content of a Wikipedia page using the Wikipedia API.
    """
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MusefulBot/1.0 (https://example.com; contact: youremail@example.com)"}
    params = {
        "action": "query",
        "format": "json",
        "titles": search_term,
        "prop": "extracts",
        "redirects": True,
        "exintro": False,
        "explaintext": True,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id != "-1":  # Valid page
                title = page.get("title", "Unknown Title")
                extract = page.get("extract", "")
                if extract:
                    return title, extract
    return None, None  # If no valid page is found


# Utility: Extract Wikipedia search terms
def extract_wikipedia_search_term(question):
    """
    Extracts the most relevant search term from a user question.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(question)
    print(doc.ents)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "LOC"}:  # Adjust labels based on context
            return ent.text
    return None
def extract_german_search_term(question):
    """
    Extracts the most relevant search term from a user question in German.
    """
    nlp = spacy.load("de_core_news_sm")  # Load the German language model
    doc = nlp(question)
    
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "LOC", "MISC"}:  # Adjust labels for German
            return ent.text
    for token in doc:
        if token.pos_ == "PROPN":  # Proper noun
            return token.text
    return None
def fetch_wikipedia_full_content_german(search_term):
    """
    Fetches the full plain text content of a German Wikipedia page for a given search term.
    """
    url = "https://de.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MusefulBot/1.0 (https://example.com; contact: youremail@example.com)"}
    params = {
        "action": "query",
        "format": "json",
        "titles": search_term,  # Search term to query
        "prop": "extracts",  # Extract plain text content
        "redirects": True,  # Follow redirects if the page has one
        "exintro": False,  # Fetch full content (not just the introduction)
        "explaintext": True,  # Get plain text instead of HTML or Wikitext
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id != "-1":  # Check if it's a valid page
                return page.get("title", ""), page.get("extract", "")
    return None, None

# Example Usage
# search_term = "Georg Wilhelm Friedrich Hegel"
# title, content = fetch_wikipedia_full_content_german(search_term)
# if content:
#     print(f"Title: {title}\n")
#     print(f"Full Content:\n{content}")
# else:
#     print("Page not found.")
# Example Usage
# german_question = "Wann wurde Hegel geboren?"
# search_term = extract_german_search_term(german_question)
# print("Extracted Search Term:", search_term)

# Main Function: Apply RAG
def apply_rag(question, search_terms):
    """
    Fetches Wikipedia pages for the search terms, applies RAG to select the most relevant page,
    and then applies RAG again to select the most relevant paragraph.
    """
    query_embedding = generate_embeddings(question)
    pages = []

    # Step 1: Fetch the first 3 pages
    # for term in search_terms:
    print(search_terms[0])
    print(search_terms)
    title, content = fetch_wikipedia_content(search_terms)
    if content:
        pages.append({"title": title, "content": content})

    if not pages:
        print("No relevant pages found.")
        return None

    # Step 2: Decide the most relevant page using RAG
    page_scores = []
    for page in pages:
        page_embedding = generate_embeddings(page["content"][:512])  # Limit to first 512 characters
        score = F.cosine_similarity(torch.tensor(query_embedding), torch.tensor(page_embedding), dim=0).item()
        page_scores.append((page, score))
        print(page)

    most_relevant_page = max(page_scores, key=lambda x: x[1])[0]
    print(f"Most relevant page: {most_relevant_page['title']}")

    # Step 3: Apply RAG to paragraphs
    paragraphs = most_relevant_page["content"].split("\n")
    paragraph_scores = []
    for paragraph in paragraphs:
        if paragraph.strip():  # Skip empty paragraphs
            paragraph_embedding = generate_embeddings(paragraph)
            score = F.cosine_similarity(torch.tensor(query_embedding), torch.tensor(paragraph_embedding), dim=0).item()
            paragraph_scores.append((paragraph, score))

    most_relevant_paragraph = max(paragraph_scores, key=lambda x: x[1])[0]
    # print(f"Most relevant paragraph: {most_relevant_paragraph}")

    return most_relevant_paragraph


# # Example Usage
# question = "When was Hegel born?"
# search_terms = extract_wikipedia_search_term(question)
# print(search_terms)  # Top Wikipedia search terms
# result = apply_rag(question, search_terms)

# if result:
#     print("\nFinal Answer:")
#     print(result)
# import requests
# print(requests.get("https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&titles=pizza").text)
import requests
import re
def fetch_and_clean_wikipedia_content(title):
    """
    Fetches raw Wikitext content from a Wikipedia page and cleans it to produce plain text.
    
    Parameters:
        title (str): The title of the Wikipedia page to fetch.
    
    Returns:
        str: Cleaned plain text content of the Wikipedia page.
    """
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "MusefulBot/1.0 (https://example.com; contact: youremail@example.com)"}
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "redirects": True,
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id != "-1":  # Valid page
                raw_content = page.get("revisions", [{}])[0].get("*", "")
                return clean_wikitext(raw_content)
    return None

def clean_wikitext(raw_text):
    """
    Cleans Wikitext content to produce plain text.
    
    Parameters:
        raw_text (str): The raw Wikitext content.
    
    Returns:
        str: Cleaned plain text content.
    """
    # Remove templates (e.g., {{...}})
    cleaned_text = re.sub(r"\{\{.*?\}\}", "", raw_text, flags=re.DOTALL)
    # Remove file/image links (e.g., [[File:...]] or [[Image:...]])
    cleaned_text = re.sub(r"\[\[File:.*?\]\]", "", cleaned_text)
    cleaned_text = re.sub(r"\[\[Image:.*?\]\]", "", cleaned_text)
    # Remove category links (e.g., [[Category:...]])
    cleaned_text = re.sub(r"\[\[Category:.*?\]\]", "", cleaned_text)
    # Remove external links and anchors (e.g., [http://... label])
    cleaned_text = re.sub(r"\[http[^\]]*\]", "", cleaned_text)
    # Convert internal links to plain text (e.g., [[Article|label]] -> label)
    cleaned_text = re.sub(r"\[\[(?:[^\|\]]*\|)?([^\]]+)\]\]", r"\1", cleaned_text)
    # Remove HTML comments (e.g., <!-- Comment -->)
    cleaned_text = re.sub(r"<!--.*?-->", "", cleaned_text, flags=re.DOTALL)
    # Remove excessive newlines
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
    # print(cleaned_text.strip())
    return cleaned_text.strip()


# question = "why was hegel smart"
# query_embedding = generate_embeddings(question)
# # Example Usage
# title = extract_wikipedia_search_term(question)
# print(title)
# clean_text = fetch_and_clean_wikipedia_content(title)
# if clean_text:
#     print(clean_text)
# else:
#     print("Page not found or empty.")

# paragraphs = clean_text.split("\n")
# paragraph_scores = []
# for paragraph in paragraphs:
#         if paragraph.strip():  # Skip empty paragraphs
#             paragraph_embedding = generate_embeddings(paragraph)
#             score = F.cosine_similarity(torch.tensor(query_embedding), torch.tensor(paragraph_embedding), dim=0).item()
#             paragraph_scores.append((paragraph, score))

# most_relevant_paragraph = max(paragraph_scores, key=lambda x: x[1])[0]
# print(f"Most relevant paragraph: {most_relevant_paragraph}")