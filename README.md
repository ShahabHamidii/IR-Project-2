# IR-Project-2
# Information Retrieval System (IR) - Python Implementation

## Overview
This project implements a simple **Information Retrieval (IR) system** using Python.  
The system demonstrates the fundamental components of IR, including:

- Document preprocessing
- Indexing (Term Frequency and Document Frequency)
- Query processing
- Document ranking

Although the system currently supports only **simple queries without Boolean operators**, it is designed in a **modular way** to allow future extensions such as Boolean queries, BM25 ranking, or integration with machine learning techniques.

---

## Features

1. **Tokenizer**: Converts documents and queries into individual terms (lowercased).
2. **Indexer**: Computes Term Frequency (TF) for each document and Document Frequency (DF) across the collection.
3. **TF-IDF Model**: Assigns weights to terms based on their importance, giving higher weight to informative terms.
4. **Query Processor**: Converts user queries into weighted vectors compatible with the TF-IDF model.
5. **Search Engine**: Scores and ranks documents based on their relevance to the query.

---

## How It Works

1. **Load Documents**: Read all documents from `documents.txt`.
2. **Build Index**: Compute TF and DF for each document.
3. **Process Query**: Convert user input into a query vector.
4. **Compute Scores**: Use TF-IDF to calculate relevance scores for each document.
5. **Rank and Display**: Sort documents by score and show the most relevant results at the top.

---

## Usage

1. Place all documents in `documents.txt`, one document per line.
2. Run the system:

```bash
python ir_system.py
Enter a query when prompted:


Enter your query: information retrieval
View ranked results:

Search Results:
Doc 2 | Score: 2.314
Doc 5 | Score: 1.872
Doc 1 | Score: 1.210
Project Structure

IR-Project/
│
├── ir_system.py       # Main Python implementation
├── documents.txt      # Text documents for searching
└── README.md          # Project documentation
Extensibility
The system is designed to be modular:

Scoring Models: TF-IDF can be replaced or extended with BM25, cosine similarity, or custom models.

Query Processing: Boolean operators or phrase queries can be added easily.

Dataset: Can handle larger datasets for practical IR applications.

Dependencies
Python 3.x

Standard libraries only (math, collections)

Author
Shahab Hamidi
