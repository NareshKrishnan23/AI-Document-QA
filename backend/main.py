import os
import faiss
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from dotenv import load_dotenv
from google import genai

import numpy as np
from google.genai import types


# ==========================================
# Create FastAPI application
# ==========================================

app = FastAPI()


# ==========================================
# Load environment variables
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==========================================
# Gemini embedding function
# ==========================================

def create_embeddings(texts, task_type):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )

    embeddings = [embedding.values for embedding in response.embeddings]

    return np.array(embeddings, dtype="float32")


# ==========================================
# FAISS configuration
# ==========================================

dimension = 768
index = faiss.IndexFlatL2(dimension)

document_chunks = []


# ==========================================
# CORS configuration
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Upload folder
# ==========================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ==========================================
# Extract text from PDF
# ==========================================

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ==========================================
# Split text into chunks
# ==========================================

def split_text_into_chunks(text, chunk_size=500):
    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

    return chunks


# ==========================================
# Home API
# ==========================================

@app.get("/")
def home():
    return {
        "message": "AI Document Q&A Backend is running!"
    }


# ==========================================
# Test API
# ==========================================

@app.get("/api/message")
def get_message():
    return {
        "message": "Hello from FastAPI Backend!"
    }


# ==========================================
# Gemini Test API
# ==========================================

@app.get("/api/gemini-test")
def gemini_test():
    for attempt in range(3):
        try:
            chat = client.chats.create(
                model="gemini-3.5-flash-lite"
            )

            response = chat.send_message(
                "Explain what a PDF is in one sentence."
            )

            return {
                "success": True,
                "response": response.text
            }

        except Exception as e:
            error = str(e)

            if "503" in error and attempt < 2:
                time.sleep(5)
                continue

            return {
                "success": False,
                "message": "Gemini service is temporarily unavailable.",
                "error": error
            }


# ==========================================
# PDF Upload API
# ==========================================

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):
    # Check file type
    if file.content_type != "application/pdf":
        return {
            "success": False,
            "message": "Only PDF files are allowed."
        }

    # Create file path
    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    # Save PDF
    with open(
        file_path,
        "wb"
    ) as buffer:
        buffer.write(
            await file.read()
        )

    return {
        "success": True,
        "message": "PDF uploaded successfully!",
        "filename": file.filename
    }


# ==========================================
# PDF Text Extraction API
# ==========================================

@app.get("/api/extract/{filename}")
def extract_pdf_text(filename: str):
    # Create file path
    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    # Check file
    if not os.path.exists(file_path):
        return {
            "success": False,
            "message": "File not found."
        }

    # Extract text
    text = extract_text_from_pdf(
        file_path
    )

    # Split text
    chunks = split_text_into_chunks(
        text
    )

    return {
        "success": True,
        "filename": filename,
        "total_chunks": len(chunks),
        "chunks": chunks
    }


# ==========================================
# Create FAISS Index
# ==========================================

@app.get("/api/create-index/{filename}")
def create_index(filename: str):
    global document_chunks

    file_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if not os.path.exists(file_path):
        return {
            "success": False,
            "message": "File not found."
        }

    # Extract text
    text = extract_text_from_pdf(file_path)

    if not text:
        return {
            "success": False,
            "message": "No text found in PDF."
        }

    # Split text
    chunks = split_text_into_chunks(text)

    # Make sure every chunk is a normal string
    clean_chunks = []

    for chunk in chunks:
        if chunk is not None:
            chunk = str(chunk).strip()

            if chunk:
                clean_chunks.append(chunk)

    if not clean_chunks:
        return {
            "success": False,
            "message": "No valid text chunks found."
        }

    print("Total chunks:", len(clean_chunks))
    print("First chunk type:", type(clean_chunks[0]))
    print("First chunk:", clean_chunks[0][:200])

    # Convert chunks into Gemini embeddings
    embeddings = create_embeddings(
        clean_chunks,
        "RETRIEVAL_DOCUMENT"
    )

    # Reset old index
    index.reset()

    # Add embeddings to FAISS
    index.add(embeddings)

    # Store chunks
    document_chunks = clean_chunks

    return {
        "success": True,
        "filename": filename,
        "total_chunks": len(clean_chunks),
        "message": "Chunks converted to embeddings and added to FAISS."
    }


# ==========================================
# Search API
# ==========================================

@app.get("/api/search")
def search_document(query: str):
    if index.ntotal == 0 or not document_chunks:
        return {
            "success": False,
            "message": "No document has been indexed. Please create the index first."
        }

    # Convert user's question into a Gemini query embedding
    query_embedding = create_embeddings(
        [query],
        "RETRIEVAL_QUERY"
    )

    # Search FAISS for the 3 most relevant chunks
    distances, indices = index.search(
        query_embedding,
        3
    )

    results = []

    for i in indices[0]:
        if i != -1 and i < len(document_chunks):
            results.append(document_chunks[i])

    if not results:
        return {
            "success": False,
            "message": "No relevant information found in the document."
        }

    return {
        "success": True,
        "query": query,
        "results": results
    }


# ==========================================
# Ask Question API
# ==========================================

@app.get("/api/ask")
def ask_question(query: str):
    # --------------------------------------
    # Check whether document is indexed
    # --------------------------------------

    if index.ntotal == 0 or not document_chunks:
        return {
            "success": False,
            "message": "No document has been indexed. Please create the index first."
        }

    # --------------------------------------
    # Step 1: Convert question into Gemini embedding
    # --------------------------------------

    try:
        query_embedding = create_embeddings(
            [str(query)],
            "RETRIEVAL_QUERY"
        )

    except Exception as e:
        return {
            "success": False,
            "message": "Failed to create question embedding.",
            "error": str(e)
        }

    # --------------------------------------
    # Step 2: Search relevant chunks using FAISS
    # --------------------------------------

    distances, indices = index.search(
        query_embedding,
        3
    )

    # --------------------------------------
    # Step 3: Get relevant chunks
    # --------------------------------------

    relevant_chunks = []

    for i in indices[0]:
        if (
            i != -1
            and i < len(document_chunks)
        ):
            relevant_chunks.append(
                document_chunks[i]
            )

    # --------------------------------------
    # Step 4: Check relevant chunks
    # --------------------------------------

    if not relevant_chunks:
        return {
            "success": False,
            "message": "No relevant information found in the document."
        }

    # --------------------------------------
    # Step 5: Combine relevant chunks
    # --------------------------------------

    context = "\n\n".join(
        relevant_chunks
    )

    # --------------------------------------
    # Step 6: Create prompt for Gemini
    # --------------------------------------

    prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{query}

If the answer cannot be found in the context, say:
"The answer is not available in the document."
"""

    # --------------------------------------
    # Step 7: Ask Gemini
    # --------------------------------------

    try:
        response = None

        for attempt in range(3):
            try:
                chat = client.chats.create(
                    model="gemini-3.5-flash-lite"
                )

                response = chat.send_message(
                    prompt
                )

                break

            except Exception as e:
                error = str(e)

                if "503" in error and attempt < 2:
                    time.sleep(5)
                    continue

                raise e

    except Exception as e:
        return {
            "success": False,
            "message": "Gemini service is temporarily unavailable. Please try again.",
            "error": str(e)
        }

    # --------------------------------------
    # Step 8: Return final result
    # --------------------------------------

    return {
        "success": True,
        "question": query,
        "context": context,
        "answer": response.text
    }
