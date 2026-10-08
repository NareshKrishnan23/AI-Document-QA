import os
import time

import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai
from google.genai import types

from pypdf import PdfReader


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Document Q&A Backend"
)


# =========================================================
# GLOBAL ERROR HANDLER
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):
    print("\n==========================================")
    print("UNHANDLED BACKEND ERROR")
    print("Method:", request.method)
    print("Path:", request.url.path)
    print("Error:", repr(exc))
    print("==========================================\n")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Backend error while processing the request.",
            "error": str(exc)
        }
    )


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    print(
        "WARNING: GEMINI_API_KEY is not set."
    )
else:
    print(
        "GEMINI_API_KEY loaded successfully."
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# DOCUMENT STORAGE
# =========================================================

# FAISS IS COMPLETELY REMOVED.

EMBEDDING_DIMENSION = 768

document_chunks = []

document_embeddings = None


# =========================================================
# CORS CONFIGURATION
# =========================================================

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# UPLOAD FOLDER
# =========================================================

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


# =========================================================
# CREATE GEMINI EMBEDDINGS
# =========================================================

def create_embeddings(
    texts,
    task_type
):
    if not texts:
        return np.empty(
            (
                0,
                EMBEDDING_DIMENSION
            ),
            dtype=np.float32
        )

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    embeddings = [
        embedding.values
        for embedding in response.embeddings
    ]

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# =========================================================
# NUMPY COSINE SIMILARITY SEARCH
# =========================================================

def search_embeddings(
    query_embedding,
    top_k=3
):
    global document_embeddings

    if document_embeddings is None:
        return []

    if len(document_embeddings) == 0:
        return []

    query_vector = np.asarray(
        query_embedding[0],
        dtype=np.float32
    )

    # -----------------------------------------
    # NORMALIZE QUERY
    # -----------------------------------------

    query_norm = np.linalg.norm(
        query_vector
    )

    if query_norm == 0:
        return []

    query_vector = (
        query_vector / query_norm
    )

    # -----------------------------------------
    # NORMALIZE DOCUMENT EMBEDDINGS
    # -----------------------------------------

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1,
        keepdims=True
    )

    document_norms = np.maximum(
        document_norms,
        1e-12
    )

    normalized_documents = (
        document_embeddings
        / document_norms
    )

    # -----------------------------------------
    # COSINE SIMILARITY
    # -----------------------------------------

    scores = (
        normalized_documents
        @ query_vector
    )

    # -----------------------------------------
    # GET TOP RESULTS
    # -----------------------------------------

    top_k = min(
        top_k,
        len(scores)
    )

    top_indices = np.argsort(
        scores
    )[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append(
            (
                int(index),
                float(scores[index])
            )
        )

    return results


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_text_from_pdf(
    file_path
):
    reader = PdfReader(
        file_path
    )

    text_parts = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        try:
            page_text = page.extract_text()

            if page_text:
                text_parts.append(
                    page_text
                )

        except Exception as e:
            print(
                f"Error extracting page {page_number}:",
                str(e)
            )

    return "\n".join(
        text_parts
    )


# =========================================================
# SPLIT TEXT INTO CHUNKS
# =========================================================

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


# =========================================================
# HOME API
# =========================================================

@app.api_route(
    "/",
    methods=["GET", "HEAD"]
)
def home():
    return {
        "success": True,
        "message": "AI Document Q&A Backend is running!"
    }


# =========================================================
# MESSAGE TEST API
# =========================================================

@app.get("/api/message")
def get_message():
    return {
        "success": True,
        "message": "Hello from FastAPI Backend!"
    }


# =========================================================
# GEMINI TEST API
# =========================================================

@app.get("/api/gemini-test")
def gemini_test():

    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents="Explain what a PDF is in one sentence."
            )

            return {
                "success": True,
                "response": response.text
            }

        except Exception as e:

            error = str(e)

            print(
                f"Gemini test attempt "
                f"{attempt + 1} failed:",
                error
            )

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


# =========================================================
# PDF UPLOAD API
# =========================================================

@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global document_chunks
    global document_embeddings

    print("\n==========================================")
    print("PDF UPLOAD STARTED")
    print("File:", file.filename)
    print("Content Type:", file.content_type)
    print("==========================================")

    # -----------------------------------------
    # CHECK FILE TYPE
    # -----------------------------------------

    if file.content_type != "application/pdf":
        return {
            "success": False,
            "message": "Only PDF files are allowed."
        }

    if not file.filename:
        return {
            "success": False,
            "message": "No filename received."
        }

    # -----------------------------------------
    # SAFE FILE NAME
    # -----------------------------------------

    safe_filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    # -----------------------------------------
    # SAVE PDF
    # -----------------------------------------

    try:

        file_data = await file.read()

        with open(
            file_path,
            "wb"
        ) as buffer:

            buffer.write(
                file_data
            )

        del file_data

        print(
            "PDF uploaded:",
            safe_filename
        )

    except Exception as e:

        print(
            "PDF save error:",
            str(e)
        )

        return {
            "success": False,
            "message": "Failed to save PDF.",
            "error": str(e)
        }

    # -----------------------------------------
    # EXTRACT PDF TEXT
    # -----------------------------------------

    try:

        text = extract_text_from_pdf(
            file_path
        )

        print(
            "Text extracted successfully."
        )

    except Exception as e:

        print(
            "PDF extraction error:",
            str(e)
        )

        return {
            "success": False,
            "message": "Failed to extract text from PDF.",
            "error": str(e)
        }

    if not text or not text.strip():
        return {
            "success": False,
            "message": "No text found in PDF."
        }

    # -----------------------------------------
    # SPLIT INTO CHUNKS
    # -----------------------------------------

    chunks = split_text_into_chunks(
        text
    )

    del text

    clean_chunks = []

    for chunk in chunks:

        if chunk is not None:

            chunk = str(
                chunk
            ).strip()

            if chunk:
                clean_chunks.append(
                    chunk
                )

    del chunks

    if not clean_chunks:
        return {
            "success": False,
            "message": "No valid text chunks found."
        }

    print(
        "Total chunks:",
        len(clean_chunks)
    )

    # -----------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------

    BATCH_SIZE = 2

    all_embedding_batches = []

    try:

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

            all_embedding_batches.append(
                embeddings
            )

            del embeddings

        # -------------------------------------
        # COMBINE EMBEDDINGS
        # -------------------------------------

        if all_embedding_batches:

            new_embeddings = np.vstack(
                all_embedding_batches
            ).astype(
                np.float32,
                copy=False
            )

        else:

            new_embeddings = np.empty(
                (
                    0,
                    EMBEDDING_DIMENSION
                ),
                dtype=np.float32
            )

        # -------------------------------------
        # SAVE DOCUMENT DATA
        # -------------------------------------

        document_chunks = clean_chunks

        document_embeddings = new_embeddings

        del all_embedding_batches

        print(
            "Document embeddings created successfully."
        )

        print(
            "Embedding shape:",
            document_embeddings.shape
        )

    except Exception as e:

        print(
            "Embedding error:",
            str(e)
        )

        return {
            "success": False,
            "message": "Failed to create document embeddings.",
            "error": str(e)
        }

    # -----------------------------------------
    # SUCCESS
    # -----------------------------------------

    return {
        "success": True,
        "filename": safe_filename,
        "total_chunks": len(document_chunks),
        "message": "PDF uploaded and indexed successfully."
    }


# =========================================================
# SEARCH API
# =========================================================

@app.get("/api/search")
def search_document(
    query: str
):

    global document_chunks
    global document_embeddings

    # -----------------------------------------
    # CHECK DOCUMENT
    # -----------------------------------------

    if (
        document_embeddings is None
        or len(document_embeddings) == 0
        or not document_chunks
    ):
        return {
            "success": False,
            "message": (
                "No document has been indexed. "
                "Please upload a PDF first."
            )
        }

    # -----------------------------------------
    # QUERY EMBEDDING
    # -----------------------------------------

    try:

        query_embedding = create_embeddings(
            [query],
            "RETRIEVAL_QUERY"
        )

    except Exception as e:

        print(
            "Query embedding error:",
            str(e)
        )

        return {
            "success": False,
            "message": "Failed to create query embedding.",
            "error": str(e)
        }

    # -----------------------------------------
    # SEARCH
    # -----------------------------------------

    matches = search_embeddings(
        query_embedding,
        top_k=3
    )

    del query_embedding

    results = []

    for index, score in matches:

        if (
            index >= 0
            and index < len(document_chunks)
        ):

            results.append({
                "text": document_chunks[index],
                "score": score
            })

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


# =========================================================
# ASK QUESTION API
# =========================================================

@app.get("/api/ask")
def ask_question(
    query: str
):

    global document_chunks
    global document_embeddings

    # -----------------------------------------
    # CHECK DOCUMENT
    # -----------------------------------------

    if (
        document_embeddings is None
        or len(document_embeddings) == 0
        or not document_chunks
    ):
        return {
            "success": False,
            "message": (
                "No document has been indexed. "
                "Please upload a PDF first."
            )
        }

    # -----------------------------------------
    # STEP 1
    # CREATE QUESTION EMBEDDING
    # -----------------------------------------

    try:

        query_embedding = create_embeddings(
            [str(query)],
            "RETRIEVAL_QUERY"
        )

    except Exception as e:

        print(
            "Question embedding error:",
            str(e)
        )

        return {
            "success": False,
            "message": "Failed to create question embedding.",
            "error": str(e)
        }

    # -----------------------------------------
    # STEP 2
    # SEARCH DOCUMENT
    # -----------------------------------------

    matches = search_embeddings(
        query_embedding,
        top_k=3
    )

    del query_embedding

    # -----------------------------------------
    # STEP 3
    # GET RELEVANT CHUNKS
    # -----------------------------------------

    relevant_chunks = []

    for index, score in matches:

        if (
            index >= 0
            and index < len(document_chunks)
        ):

            relevant_chunks.append(
                document_chunks[index]
            )

    # -----------------------------------------
    # STEP 4
    # CHECK RESULTS
    # -----------------------------------------

    if not relevant_chunks:
        return {
            "success": False,
            "message": "No relevant information found in the document."
        }

    # -----------------------------------------
    # STEP 5
    # COMBINE CONTEXT
    # -----------------------------------------

    context = "\n\n".join(
        relevant_chunks
    )

    # -----------------------------------------
    # STEP 6
    # GEMINI PROMPT
    # -----------------------------------------

    prompt = f"""
Answer the question using only the information
provided in the context.

Context:

{context}

Question:

{query}

Important instructions:

- Answer only from the document context.
- Do not invent information.
- Keep the answer clear and easy to understand.
- If the answer cannot be found in the context, say:

"The answer is not available in the document."
"""

    # -----------------------------------------
    # STEP 7
    # ASK GEMINI
    # -----------------------------------------

    try:

        response = None

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                break

            except Exception as e:

                error = str(e)

                print(
                    f"Gemini attempt "
                    f"{attempt + 1} failed:",
                    error
                )

                if (
                    "503" in error
                    and attempt < 2
                ):
                    time.sleep(5)
                    continue

                raise

        if response is None:
            return {
                "success": False,
                "message": "Gemini did not return a response."
            }

    except Exception as e:

        print(
            "Gemini error:",
            str(e)
        )

        return {
            "success": False,
            "message": (
                "Gemini service is temporarily unavailable. "
                "Please try again."
            ),
            "error": str(e)
        }

    # -----------------------------------------
    # STEP 8
    # FINAL RESPONSE
    # -----------------------------------------

    return {
        "success": True,
        "question": query,
        "context": context,
        "answer": response.text
    }


# =========================================================
# BACKEND START MESSAGE
# =========================================================

print("\n==========================================")
print("AI Document Q&A Backend Loaded")
print("FAISS: REMOVED")
print("Search: NumPy Cosine Similarity")
print(
    "Embedding Dimension:",
    EMBEDDING_DIMENSION
)
print("==========================================\n")

@app.post("/api/test-post")
async def test_post():
    print("==========================================")
    print("TEST POST RECEIVED")
    print("==========================================")

    return {
        "success": True,
        "message": "POST request reached FastAPI successfully!"
    }
    
    
@app.post("/api/upload-test")
async def upload_test(file: UploadFile = File(...)):
    print("==========================================")
    print("UPLOAD TEST RECEIVED")
    print("Filename:", file.filename)
    print("==========================================")

    content = await file.read()

    return {
        "success": True,
        "filename": file.filename,
        "size": len(content),
        "message": "Multipart upload reached FastAPI successfully!"
    }
    
@app.post("/api/upload-test-real")
async def upload_test_real(file: UploadFile = File(...)):
    print("==========================================")
    print("REAL PDF UPLOAD TEST STARTED")
    print("Filename:", file.filename)

    content = await file.read()

    print("PDF SIZE:", len(content))
    print("REAL PDF UPLOAD TEST FINISHED")
    print("==========================================")

    return {
        "success": True,
        "filename": file.filename,
        "size": len(content),
        "message": "Real PDF reached FastAPI successfully!"
    }    