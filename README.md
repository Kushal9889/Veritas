# Veritas — Sprint 1

This repository contains a minimal RAG ingestion pipeline and scaffolding for early-stage experimentation.

Project layout

Veritas/
├── .env                  <-- YOUR SECRETS (Created from .env.example)
├── .gitignore            <-- What to keep out of version control
├── requirements.txt      <-- The Python dependencies
├── README.md
│
├── data/                 <-- The sensory input
│   └── raw/              <-- Place your PDF 10-Ks here
│
├── src/                  <-- The source code
│   ├── core/             <-- Shared logic
│   │   ├── __init__.py
│   │   └── config.py     <-- Loads .env variables safely
│   │   
│   ├── ingestion/        <-- WRITE PATH (Sprint 1 Focus)
│   │   ├── __init__.py
│   │   ├── loader.py     <-- PDF -> Text
│   │   └── vector_db.py  <-- Text -> Pinecone
│   │
│   └── retrieval/        <-- READ PATH (Sprint 2 Focus)
│       ├── __init__.py
│       └── graph.py
│
└── main.py               <-- ENTRY POINT (Temporary CLI for testing)


