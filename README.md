# StudyMate-AI

StudyMate-AI is an AI-powered study assistant designed to help students interact with their course materials in a more efficient way.

Users can upload study documents such as **PDF or TXT files** and interact with them using AI. The system allows students to ask questions about the content, generate summaries, create slide-style study notes, listen to explanations through audio, and export generated study materials as PDF documents.

The platform integrates document processing, semantic search, and large language models to provide intelligent, context-aware responses based on the uploaded material.

---

## Features

StudyMate-AI offers several features to enhance the learning experience:

- Upload course materials in **PDF or TXT format**
- Ask **AI-powered questions based on the uploaded document**
- Generate **clear and structured study summaries**
- Create **slide-style study notes for quick review**
- Convert explanations into **audio using Text-to-Speech**
- Export summaries and notes as **PDF documents**

---

## Tech Stack

The project is built using the following technologies:

- **Python**
- **Streamlit**
- **Azure OpenAI**
- **Sentence Transformers**
- **Semantic Search**
- **Text-to-Speech**
- **ReportLab**
- **GitHub**
- **Visual Studio Code**
- **PyCharm**

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

The system processes documents and generates AI responses through the following steps:

**1.** The user uploads a course document.

**2.** The document is split into smaller text chunks.

**3.** Sentence embeddings are generated from these chunks.

**4.** Semantic search retrieves the most relevant sections related to the user's query.

**5**. The AI model generates answers, summaries, or slide-style notes based on the retrieved content.


---

## Future Improvements

Planned improvements for the project include:

- Support for multiple document uploads
- Integration with a vector database
- Improved automatic slide generation
- Voice-based question interaction
- Enhanced user interface for studying
