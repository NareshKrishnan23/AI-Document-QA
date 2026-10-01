import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [question, setQuestion] = useState("");
 const [messages, setMessages] = useState([]);

  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  // Upload PDF and create FAISS index
  const processFile = async (file) => {
    if (!file) {
      return;
    }

    // Check PDF
    if (file.type !== "application/pdf") {
      setError("Please select a PDF file.");
      return;
    }

    setSelectedFile(file);
    setMessages([]);
    setError("");
    setUploading(true);

    try {
      // Step 1: Upload PDF
      const formData = new FormData();
      formData.append("file", file);

      const uploadResponse = await fetch(
        "http://127.0.0.1:8000/api/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      const uploadData = await uploadResponse.json();

      if (!uploadData.success) {
        setError(uploadData.message);
        setUploading(false);
        return;
      }

      // Step 2: Create FAISS index
      const indexResponse = await fetch(
        `http://127.0.0.1:8000/api/create-index/${encodeURIComponent(
          file.name
        )}`
      );

      const indexData = await indexResponse.json();

      if (!indexData.success) {
        setError(indexData.message);
        setUploading(false);
        return;
      }

      setUploading(false);

    } catch (error) {
      console.error("Upload error:", error);

      setError(
        "Something went wrong while processing the PDF."
      );

      setUploading(false);
    }
  };

  // Normal file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];

    processFile(file);
  };

  // Drag over
  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  // Drag leaves upload area
  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  // Drop file
  const handleDrop = (event) => {
    event.preventDefault();

    setIsDragging(false);

    const file = event.dataTransfer.files[0];

    processFile(file);
  };

  // Ask question
const handleAsk = async () => {
  if (!question.trim()) {
    setError("Please enter a question.");
    return;
  }

  if (!selectedFile) {
    setError("Please upload a PDF first.");
    return;
  }

  const currentQuestion = question.trim();

  setError("");
  setAsking(true);

  // Add user's question immediately
  setMessages((previousMessages) => [
    ...previousMessages,
    {
      type: "user",
      text: currentQuestion,
    },
  ]);

  setQuestion("");

  try {
    const response = await fetch(
      `http://127.0.0.1:8000/api/ask?query=${encodeURIComponent(
        currentQuestion
      )}`
    );

    const data = await response.json();

    if (data.success) {
      // Add AI answer
      setMessages((previousMessages) => [
        ...previousMessages,
        {
          type: "ai",
          text: data.answer,
        },
      ]);
    } else {
      setError(data.message);
    }

  } catch (error) {
    console.error("Ask error:", error);

    setError(
      "Failed to get an answer from the server."
    );
  }

  setAsking(false);
};

  return (
    <div className="app">

      {/* Background glow */}

      <div className="glow glow-one"></div>
      <div className="glow glow-two"></div>

      {/* Navbar */}

      <nav className="navbar">

        <div className="logo">
          <span className="logo-icon">✦</span>
          DocuMind
        </div>

        <div className="nav-badge">
          AI DOCUMENT Q&A
        </div>

      </nav>

      {/* Hero */}

      <section className="hero">

        <div className="hero-badge">
          ✨ Powered by AI + RAG
        </div>

        <h1>
          Chat with your
          <span> Documents.</span>
        </h1>

        <p>
          Upload your PDF and ask questions.
          <br />
          Get intelligent answers from your document.
        </p>

      </section>

      {/* Main */}

      <main className="main-container">

        {/* Upload Card */}

        <div className="upload-card">

          <label
            htmlFor="pdf-upload"
            className={`upload-area ${
              isDragging ? "drag-active" : ""
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >

            <div className="upload-icon">
              📄
            </div>

            <h2>
              {isDragging
                ? "Drop your PDF here"
                : "Upload your document"}
            </h2>

            <p>
              {isDragging
                ? "Release to upload your PDF"
                : "Drag & drop your PDF here"}
              <br />
              or click to browse
            </p>

            <span className="browse-button">
              Choose PDF
            </span>

            <small>
              PDF files only
            </small>

          </label>

          <input
            id="pdf-upload"
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            hidden
          />

          {/* File status */}

          {selectedFile && (
            <div className="file-status">

              <div className="file-left">

                <div className="pdf-icon">
                  PDF
                </div>

                <div>

                  <strong>
                    {selectedFile.name}
                  </strong>

                  <span>
                    {uploading
                      ? "Processing document..."
                      : "Document indexed successfully"}
                  </span>

                </div>

              </div>

              <div className="status-dot">
                {uploading ? "⏳" : "✓"}
              </div>

            </div>
          )}

        </div>

        {/* Question */}

        <div className="question-card">

          <div className="question-label">
            <span>✦</span>
            Ask your document
          </div>

          <div className="question-box">

            <input
              type="text"
              placeholder="What would you like to know?"
              value={question}
              onChange={(event) =>
                setQuestion(event.target.value)
              }
              disabled={uploading || asking}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  handleAsk();
                }
              }}
            />

            <button
              onClick={handleAsk}
              disabled={uploading || asking}
            >
              {asking ? "..." : "➤"}
            </button>

          </div>

          <div className="question-hint">
            Press Enter to ask
          </div>

        </div>

        {/* Error */}

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {/* Answer */}

    {messages.length > 0 && (
  <div className="chat-container">

    {messages.map((message, index) => (

      <div
        key={index}
        className={
          message.type === "user"
            ? "message user-message"
            : "message ai-message"
        }
      >

        {message.type === "ai" && (
          <div className="ai-avatar">
            ✦
          </div>
        )}

        <div className="message-content">

          <span className="message-label">
            {message.type === "user"
              ? "You"
              : "DocuMind AI"}
          </span>

          <p>
            {message.text}
          </p>

        </div>

      </div>

    ))}

  </div>
)}

        {/* Thinking */}

        {asking && (
          <div className="thinking-card">

            <div className="thinking-icon">
              ✦
            </div>

            <div>

              <strong>
                AI is thinking...
              </strong>

              <span>
                Searching your document for the best answer
              </span>

            </div>

          </div>
        )}

      </main>

      {/* Footer */}

      <footer>
        Built with React • FastAPI • FAISS • Gemini
      </footer>

    </div>
  );
}

export default App;