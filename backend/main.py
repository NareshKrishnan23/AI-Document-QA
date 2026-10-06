import os
import time

import faiss
import numpy as np

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pypdf import PdfReader


# ==========================================
# Create FastAPI application
# ==========================================

app = FastAPI()

@app.post("/api/test")
async def test_post():
    print("🔥 TEST POST RECEIVED")
    return {
        "success": True,
        "message": "POST is working"
    }
    
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

    embeddings = [
        embedding.values
        for embedding in response.embeddings
    ]

    return np.array(
        embeddings,
        dtype="float32"
    )


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
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Upload folder
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

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

def split_text_into_chunks(
    text,
    chunk_size=500
):

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            i:i + chunk_size
        ]

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

            if (
                "503" in error
                and attempt < 2
            ):

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

    global document_chunks

    # --------------------------------------
    # Check file type
    # --------------------------------------

    if file.content_type != "application/pdf":

        return {
            "success": False,
            "message": "Only PDF files are allowed."
        }

    # --------------------------------------
    # Create file path
    # --------------------------------------

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    # --------------------------------------
    # Save PDF
    # --------------------------------------

    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            await file.read()
        )

    print("PDF uploaded:", file.filename)

    # --------------------------------------
    # Extract PDF text
    # --------------------------------------

    text = extract_text_from_pdf(
        file_path
    )

    if not text:

        return {
            "success": False,
            "message": "No text found in PDF."
        }

    # --------------------------------------
    # Split into chunks
    # --------------------------------------

    chunks = split_text_into_chunks(
        text
    )

    clean_chunks = []

    for chunk in chunks:

        if chunk is not None:

            chunk = str(chunk).strip()

            if chunk:

                clean_chunks.append(
                    chunk
                )

    if not clean_chunks:

        return {
            "success": False,
            "message": "No valid text chunks found."
        }

    print(
        "Total chunks:",
        len(clean_chunks)
    )

    # --------------------------------------
    # Reset FAISS
    # --------------------------------------

    index.reset()

    document_chunks = []

    # --------------------------------------
    # Create embeddings in small batches
    # --------------------------------------

    BATCH_SIZE = 2

    for start in range(
        0,
        len(clean_chunks),
        BATCH_SIZE
    ):

        batch = clean_chunks[
            start:start + BATCH_SIZE
        ]

        print(
            f"Embedding chunks "
            f"{start + 1} - "
            f"{start + len(batch)} "
            f"of {len(clean_chunks)}"
        )

        embeddings = create_embeddings(
            batch,
            "RETRIEVAL_DOCUMENT"
        )

        index.add(
            embeddings
        )

        document_chunks.extend(
            batch
        )

        del embeddings

    print(
        "Index created successfully."
    )

    # --------------------------------------
    # Return success
    # --------------------------------------

    return {
        "success": True,
        "filename": file.filename,
        "total_chunks": len(document_chunks),
        "message": "PDF uploaded and indexed successfully."
    }


# ==========================================
# Create FAISS Index
# ==========================================

@app.get("/api/create-index/{filename}")
def create_index(filename: str):

    global document_chunks

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

    # --------------------------------------
    # Extract text
    # --------------------------------------

    text = extract_text_from_pdf(
        file_path
    )

    if not text:

        return {
            "success": False,
            "message": "No text found in PDF."
        }

    # --------------------------------------
    # Split text
    # --------------------------------------

    chunks = split_text_into_chunks(
        text
    )

    # --------------------------------------
    # Clean chunks
    # --------------------------------------

    clean_chunks = []

    for chunk in chunks:

        if chunk is not None:

            chunk = str(chunk).strip()

            if chunk:

                clean_chunks.append(
                    chunk
                )

    if not clean_chunks:

        return {
            "success": False,
            "message": "No valid text chunks found."
        }

    # --------------------------------------
    # Debug information
    # --------------------------------------

    print(
        "Total chunks:",
        len(clean_chunks)
    )

    print(
        "First chunk type:",
        type(clean_chunks[0])
    )

    print(
        "First chunk:",
        clean_chunks[0][:200]
    )

    # --------------------------------------
    # Reset old FAISS index
    # --------------------------------------

    index.reset()

    document_chunks = []

    # --------------------------------------
    # Create embeddings in batches
    # --------------------------------------

    BATCH_SIZE = 2

    for start in range(
        0,
        len(clean_chunks),
        BATCH_SIZE
    ):

        batch = clean_chunks[
            start:start + BATCH_SIZE
        ]

        print(
            f"Creating embeddings for chunks "
            f"{start + 1} to "
            f"{start + len(batch)} "
            f"of {len(clean_chunks)}"
        )

        embeddings = create_embeddings(
            batch,
            "RETRIEVAL_DOCUMENT"
        )

        # Add embeddings to FAISS
        index.add(
            embeddings
        )

        # Store document chunks
        document_chunks.extend(
            batch
        )

        # Release embedding memory
        del embeddings

    # --------------------------------------
    # Return success
    # --------------------------------------

    return {
        "success": True,
        "filename": filename,
        "total_chunks": len(document_chunks),
        "message": "Chunks converted to embeddings and added to FAISS."
    }


# ==========================================
# Search API
# ==========================================

@app.get("/api/search")
def search_document(query: str):

    if (
        index.ntotal == 0
        or not document_chunks
    ):

        return {
            "success": False,
            "message": "No document has been indexed. Please create the index first."
        }

    # --------------------------------------
    # Create query embedding
    # --------------------------------------

    query_embedding = create_embeddings(
        [query],
        "RETRIEVAL_QUERY"
    )

    # --------------------------------------
    # Search FAISS
    # --------------------------------------

    distances, indices = index.search(
        query_embedding,
        3
    )

    results = []

    for i in indices[0]:

        if (
            i != -1
            and i < len(document_chunks)
        ):

            results.append(
                document_chunks[i]
            )

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

    if (
        index.ntotal == 0
        or not document_chunks
    ):

        return {
            "success": False,
            "message": "No document has been indexed. Please create the index first."
        }

    # --------------------------------------
    # Step 1: Create question embedding
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
    # Step 2: Search FAISS
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
    # Step 4: Check results
    # --------------------------------------

    if not relevant_chunks:

        return {
            "success": False,
            "message": "No relevant information found in the document."
        }

    # --------------------------------------
    # Step 5: Combine chunks
    # --------------------------------------

    context = "\n\n".join(
        relevant_chunks
    )

    # --------------------------------------
    # Step 6: Create Gemini prompt
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

                if (
                    "503" in error
                    and attempt < 2
                ):

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