# Medical Chatbot

A retrieval-augmented generation (RAG) chatbot that provides medical information by querying a knowledge base of medical documents using advanced language models.

## Features

- **RAG-Powered Responses**: Combines document retrieval with OpenAI's GPT-4 for accurate, context-aware medical information
- **Persistent Chat History**: Maintains conversation context for more natural, coherent interactions
- **History-Aware Retrieval**: Reformulates follow-up questions based on chat history for better retrieval
- **Web Interface**: Clean, user-friendly chat interface built with Flask
- **Efficient Embeddings**: Uses HuggingFace's sentence transformers for fast document embeddings
- **Vector Storage**: Leverages Pinecone for scalable vector database management

## Tech Stack

- **Backend**: Flask, LangChain, OpenAI API
- **Embeddings**: HuggingFace Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: Pinecone
- **Frontend**: Bootstrap, jQuery
- **Document Processing**: PyPDF

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MedicalChatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

5. **Prepare your medical document**:
   Place your medical PDF in the `data/` directory and name it `medical-book.pdf` (or update the path in `app.py`)

6. **Initialize the vector index** (first run):
   ```bash
   python store_index.py
   ```
   This loads your PDF, chunks the documents, generates embeddings, and stores them in Pinecone.

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Access the chatbot**:
   Open your browser and navigate to `http://localhost:8080`

3. **Ask questions**:
   Type medical questions in the chat interface. The chatbot will:
   - Retrieve relevant document chunks from your medical knowledge base
   - Use conversation history to understand context
   - Generate a concise answer with proper citations

## Project Structure

```
MedicalChatbot/
├── app.py                 # Flask application entry point
├── store_index.py         # Vector index initialization script
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup configuration
├── data/
│   └── medical-book.pdf  # Medical knowledge base (PDF)
├── src/
│   ├── helper.py         # RAG chain initialization and utilities
│   ├── prompt.py         # System prompt templates
│   └── __init__.py
├── templates/
│   └── chat.html         # Web interface template
├── static/
│   └── style.css         # UI styling
└── research/
    └── trials.ipynb      # Development experiments and trials
```

## Key Components

### `app.py`
Flask web server with routes for the chat interface and message handling. Initializes the RAG chain on startup.

### `store_index.py`
Script to process your medical PDF:
- Loads and filters PDF pages
- Splits documents into 500-char chunks (100 char overlap)
- Generates embeddings using HuggingFace
- Stores vectors in Pinecone

### `src/helper.py`
Core RAG functionality:
- `init_rag()`: Initializes the retrieval chain with LLM and vector store
- Handles PDF loading, document chunking, and embedding generation
- Manages Pinecone index creation and vector upserts

### `src/prompt.py`
System prompt template that instructs the chatbot to answer concisely using retrieved context.

## Configuration

Key parameters in `src/helper.py`:

```python
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
DEFAULT_K = 3                                             # Number of documents to retrieve
chunk_size = 500                                          # Document chunk size in characters
chunk_overlap = 100                                       # Overlap between chunks
```

Modify these in `init_rag()` parameters for different configurations.

## Support & Documentation

- **Questions about LangChain**: See [LangChain Documentation](https://python.langchain.com)
- **Pinecone setup**: See [Pinecone Docs](https://docs.pinecone.io)
- **OpenAI API**: See [OpenAI Documentation](https://platform.openai.com/docs)

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## Maintainers

**Winata Tristan** - Initial development
- Email: winatatristan04@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This chatbot is designed to provide informational assistance only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for medical decisions.
