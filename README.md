# StudyMate-AI

StudyMate-AI is an AI-powered study assistant that helps students interact with their course materials.  
Users can upload documents such as PDF or TXT files and use AI to ask questions, generate summaries, create slide-style notes, listen to explanations, and export study materials as PDF.

The system combines document processing, semantic search, and large language models to provide intelligent, context-aware answers based on the uploaded material.

---

## Features

- Upload course materials in **PDF or TXT format**
- Ask **AI-powered questions grounded in the document**
- Generate **clear study summaries**
- Create **slide-style study notes**
- Convert explanations into **audio using Text-to-Speech**
- Export summaries as **PDF documents**

---

## Tech Stack

- Python
- Streamlit
- Azure OpenAI
- Sentence Transformers
- Semantic Search
- Text-to-Speech
- ReportLab

---

## Project Structure

```
StudyMate-AI
│
├── ui.py              # Streamlit user interface
├── agent.py           # AI agent logic (Q&A, summary, slides, TTS, PDF generation)
├── search.py          # Document chunking and semantic search
├── requirements.txt   # Project dependencies
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/study-mate-ai.git
cd study-mate-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment (Windows):

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root and add your Azure credentials:

```
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT=your_deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_SPEECH_KEY=your_speech_key
AZURE_SPEECH_REGION=your_region
```

⚠️ Never upload your `.env` file to GitHub.

---

## Running the Application

Start the Streamlit app:

```bash
streamlit run ui.py
```

The application will open in your browser.

---

## How It Works

1. User uploads a course document.
2. The document is split into smaller chunks.
3. Sentence embeddings are generated.
4. Semantic search retrieves the most relevant sections.
5. The AI model generates answers, summaries, or slide-style notes based on the retrieved content.

---

## Future Improvements

- Multi-document support
- Vector database integration
- Improved slide generation
- Voice-based question asking
- Better UI for studying