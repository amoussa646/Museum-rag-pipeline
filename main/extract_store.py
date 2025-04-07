import os
from docx import Document
import psycopg2
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize tokenizer and model for GTE-base
tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-base')
model = AutoModel.from_pretrained('thenlper/gte-base')


# Function to average-pool embeddings
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Function to generate embeddings
def generate_embeddings(text, metadata={}):
    combined_text = " ".join(
        [text] + [v for k, v in metadata.items() if isinstance(v, str)])

    inputs = tokenizer(combined_text, return_tensors='pt',
                       max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs['attention_mask']
    embeddings = average_pool(outputs.last_hidden_state, attention_mask)

    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.numpy().tolist()[0]


# Function to parse a .docx file
def parse_docx(file_path):
    doc = Document(file_path)
    parsed_data = []
    file_name = os.path.basename(file_path)

    current_section = {"section_number": None, "section_title": None, "content": ""}
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        
        if text.startswith("[Slot/Bild"):  # Detect section marker
            if current_section["content"]:  # Save the previous section
                parsed_data.append(current_section)
            current_section = {"section_number": text, "section_title": None, "content": ""}
        
        elif "Hegel:" in text:  # Detect section title
            current_section["section_title"] = text

        else:  # Append content to the current section
            current_section["content"] += f" {text}"
    
    # Append the last section
    if current_section["content"]:
        parsed_data.append(current_section)

    return file_name, parsed_data


# Function to store data into PostgreSQL
def store_in_postgres(data, db_config):
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            file_name TEXT NOT NULL,
            section_number TEXT,
            section_title TEXT,
            content TEXT NOT NULL,
            embedding vector(768)  -- Assuming the embedding size is 768
        );
    """)

    # Insert parsed data and embeddings
    for doc in data:
        file_name = doc['file_name']
        for section in doc['sections']:
            content = section["content"].strip()
            embedding = generate_embeddings(content)
            cursor.execute(
                """
                INSERT INTO documents (file_name, section_number, section_title, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (file_name, section["section_number"], section["section_title"], content, embedding)
            )
    
    conn.commit()
    cursor.close()
    conn.close()


# Function to process all .docx files in a folder
def parse_and_store(folder_path, db_config):
    all_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".docx"):
            file_path = os.path.join(folder_path, file_name)
            file_name, parsed_sections = parse_docx(file_path)
            all_data.append({"file_name": file_name, "sections": parsed_sections})
    
    store_in_postgres(all_data, db_config)
    print(f"Successfully processed and stored {len(all_data)} documents.")


# Example usage
folder_path = "texts"  # Folder containing .docx files
db_config = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "Invoker1",
    "host": "localhost",
    "port": "5433"
}

# Parse and store documents
#parse_and_store(folder_path, db_config)
