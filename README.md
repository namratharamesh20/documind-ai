DocuMind AI

DocuMind AI is a lightweight Streamlit application that turns any PDF into an interactive research assistant. Upload a document, ask questions, generate study guides, and create podcast-style scripts using Groq’s LLMs and local embeddings.

Features

Chat with your PDF using retrieval augmented generation
Free local embeddings using Sentence Transformers
Chroma vector search for accurate retrieval
Study guide generator with summaries, key points, definitions, and quiz questions
Podcast-style script generation between two AI hosts
Dark mode UI inspired by notebook and research tools
Automatic PDF text extraction and safe temporary file handling

Tech Stack

Streamlit
LangChain
Sentence Transformers
ChromaDB
Groq API
python-dotenv
PyPDF

Getting Started
Step 1: Clone the repository
git clone <repo-url>
cd <folder-name>

Step 2: Create and activate a virtual environment

Linux and macOS:

python -m venv .venv
source .venv/bin/activate


Windows:

python -m venv .venv
.venv\Scripts\activate

Step 3: Install dependencies
pip install -r requirements.txt

Step 4: Add your Groq API key

Create a file named .env in the project root:

GROQ_API_KEY=your_groq_api_key_here

Step 5: Run the application
streamlit run app.py

Folder Structure
app.py            # Main application logic
requirements.txt  # Dependencies
README.md         # Documentation
.env              # Environment variables (not included in repo)

How the App Works

You upload a PDF.
The text is read and split into chunks.
Each chunk is converted into embeddings using a local MiniLM model.
Chroma stores these embeddings and enables semantic search.
When you ask a question, the app retrieves the most relevant chunks.
These chunks are passed to Groq’s LLM, which generates an accurate answer.
The full text is also used for generating study guides and podcast scripts.

Required Environment Variable
GROQ_API_KEY=your_api_key_here

Deployment Notes

Works on Streamlit Cloud, Render, Railway, and similar services.
Make sure to configure GROQ_API_KEY as an environment variable on the hosting platform.
The embedding model downloads once during the first run.

License

MIT License.
