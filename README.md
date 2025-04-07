# Mus - Advanced Question Answering System

Mus is a sophisticated question-answering system that combines multiple data sources and AI models to provide comprehensive and accurate responses to user queries. The system integrates knowledge from Wikipedia, WolframAlpha, OpenAI's GPT models, and a custom knowledge base to deliver detailed answers.

## Features

- **Multi-source Information Retrieval**: Combines data from:

  - Wikipedia content
  - WolframAlpha computational knowledge engine
  - OpenAI's GPT models
  - Custom knowledge base with document embeddings

- **Advanced RAG (Retrieval-Augmented Generation) Pipeline**:

  - Semantic search across multiple data sources
  - Context-aware answer generation
  - Document embedding and similarity matching

- **Language Support**:

  - Primary support for English
  - German language capabilities for certain queries

- **Knowledge Base Integration**:
  - PostgreSQL database for storing and retrieving document embeddings
  - Support for processing and storing .docx files

## Prerequisites

- Python 3.x
- PostgreSQL database
- Required Python packages (see `requirements.txt`)
- API keys for:
  - OpenAI
  - WolframAlpha
  - News API (optional)

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd mus
```

2. Install required packages:

```bash
pip install -r requirements.txt
pip install python-dotenv
```

3. Set up environment variables:

```bash
# Copy the example .env file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

4. Configure the following environment variables in your `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
WOLFRAM_ALPHA_APP_ID=your_wolfram_alpha_app_id_here
NEWS_API_KEY=your_news_api_key_here
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_db_password_here
DB_HOST=localhost
DB_PORT=5433
```

5. Set up PostgreSQL database:

```bash
# Create database and configure connection
# The system will use the credentials from your .env file
```

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys and database credentials secure
- The `.env` file is included in `.gitignore` to prevent accidental commits
- Use different API keys for development and production environments

## Project Structure

```
mus/
├── main/
│   ├── main.py              # Main application logic
│   ├── wikipedia.py         # Wikipedia content retrieval
│   ├── extract_store.py     # Document processing and storage
│   ├── openaiAPI.py         # OpenAI integration
│   ├── wolframalpha.py      # WolframAlpha integration
│   └── test*.py             # Test and utility files
├── texts/                   # Directory for .docx files
├── .env                     # Environment variables (not in version control)
└── requirements.txt         # Python dependencies
```

## Usage

1. Start the application:

```bash
python main/main.py
```

2. Enter your question when prompted.

3. The system will:
   - Search across multiple data sources
   - Generate embeddings for semantic matching
   - Combine relevant information
   - Provide a comprehensive answer

## Configuration

The system can be configured through environment variables in the `.env` file:

- Database connection settings
- API keys for various services
- Embedding model parameters
- RAG pipeline settings

## Dependencies

- transformers
- torch
- psycopg2
- python-docx
- requests
- spacy
- openai
- numpy
- python-dotenv

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Acknowledgments

- OpenAI for GPT models
- WolframAlpha for computational knowledge
- Wikipedia for open knowledge
- The Hugging Face team for transformer models
