import { Router, Request, Response } from "express";
import fetch from "node-fetch";
import multer from "multer";
import fs from "fs";
import FormData from "form-data";

const router = Router();

// Configure multer for file uploads
const upload = multer({ 
  dest: "/tmp/uploads/",
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://127.0.0.1:8000";

// Streaming chat endpoint - proxies SSE from Python chatbot
router.post("/stream", async (req: Request, res: Response) => {
  const { message, session_id } = req.body;

  if (!message) {
    return res.status(400).json({ error: "Message is required" });
  }

  // Set SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  try {
    const response = await fetch(`${PYTHON_API_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id }),
    });

    if (!response.ok) {
      const error = await response.json();
      res.write(`data: ${JSON.stringify({ type: "error", content: error.detail || "Chat service error" })}\n\n`);
      res.end();
      return;
    }

    // Pipe the SSE stream from Python to the client
    const body = response.body as unknown as NodeJS.ReadableStream;
    if (!body) {
      res.write(`data: ${JSON.stringify({ type: "error", content: "No response body" })}\n\n`);
      res.end();
      return;
    }

    body.on("data", (chunk: Buffer) => {
      res.write(chunk);
    });

    body.on("end", () => {
      res.end();
    });

    body.on("error", (err: Error) => {
      console.error("Stream error:", err);
      res.write(`data: ${JSON.stringify({ type: "error", content: "Stream interrupted" })}\n\n`);
      res.end();
    });

    // Handle client disconnect
    req.on("close", () => {
      body.removeAllListeners();
    });

  } catch (err) {
    console.error("Chat stream error:", err);
    res.write(`data: ${JSON.stringify({ type: "error", content: "Chatbot service unreachable" })}\n\n`);
    res.end();
  }
});

// Main chat endpoint - proxies to Python chatbot
router.post("/", async (req: Request, res: Response) => {
  const { message, session_id } = req.body;

  if (!message) {
    return res.status(400).json({ error: "Message is required" });
  }

  try {
    const response = await fetch(`${PYTHON_API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id }),
    });

    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Chat API error:", err);
    return res.status(500).json({ error: "Chatbot service unreachable" });
  }
});

// Chat with document upload
router.post("/upload", upload.single("file"), async (req: Request, res: Response) => {
  if (!req.file) {
    return res.status(400).json({ error: "File is required" });
  }

  try {
    const formData = new FormData();
    formData.append("file", fs.createReadStream(req.file.path), {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });
    formData.append("message", req.body.message || "Please analyze this document");
    if (req.body.session_id) {
      formData.append("session_id", req.body.session_id);
    }

    const response = await fetch(`${PYTHON_API_URL}/chat/upload`, {
      method: "POST",
      body: formData,
      headers: formData.getHeaders(),
    });

    // Clean up temp file
    fs.unlinkSync(req.file.path);

    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Document upload error:", err);
    // Clean up temp file on error
    if (req.file?.path) {
      try { fs.unlinkSync(req.file.path); } catch {}
    }
    return res.status(500).json({ error: "Document processing service unreachable" });
  }
});

// Analyze document text directly
router.post("/analyze-document", async (req: Request, res: Response) => {
  const { document_text, session_id } = req.body;

  if (!document_text) {
    return res.status(400).json({ error: "Document text is required" });
  }

  try {
    const response = await fetch(`${PYTHON_API_URL}/analyze-document`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_text, session_id }),
    });

    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Document analysis error:", err);
    return res.status(500).json({ error: "Document analysis service unreachable" });
  }
});

// Crime reporting guidance
router.post("/crime-report", async (req: Request, res: Response) => {
  const { description, session_id } = req.body;

  if (!description) {
    return res.status(400).json({ error: "Description is required" });
  }

  try {
    const response = await fetch(`${PYTHON_API_URL}/crime-report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description, session_id }),
    });

    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Crime report error:", err);
    return res.status(500).json({ error: "Crime reporting service unreachable" });
  }
});

// Find lawyers via Python AI
router.post("/find-lawyer", async (req: Request, res: Response) => {
  const { query, location, specialization } = req.body;

  if (!query) {
    return res.status(400).json({ error: "Query is required" });
  }

  try {
    const response = await fetch(`${PYTHON_API_URL}/find-lawyer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, location, specialization }),
    });

    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Lawyer finder error:", err);
    return res.status(500).json({ error: "Lawyer search service unreachable" });
  }
});

// Get available specializations
router.get("/specializations", async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/specializations`);
    
    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Specializations error:", err);
    return res.status(500).json({ error: "Service unreachable" });
  }
});

// Get crime types
router.get("/crime-types", async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/crime-types`);
    
    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Crime types error:", err);
    return res.status(500).json({ error: "Service unreachable" });
  }
});

// Clear session
router.delete("/session/:sessionId", async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/session/${req.params.sessionId}`, {
      method: "DELETE",
    });
    
    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Clear session error:", err);
    return res.status(500).json({ error: "Service unreachable" });
  }
});

// Get session history
router.get("/session/:sessionId/history", async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/session/${req.params.sessionId}/history`);
    
    if (!response.ok) {
      const error = await response.json();
      return res.status(response.status).json(error);
    }

    const data = await response.json();
    return res.json(data);

  } catch (err) {
    console.error("Get history error:", err);
    return res.status(500).json({ error: "Service unreachable" });
  }
});

// Health check for Python service
router.get("/health", async (req: Request, res: Response) => {
  try {
    const response = await fetch(`${PYTHON_API_URL}/health`);
    
    if (!response.ok) {
      return res.status(503).json({ status: "unhealthy", python_service: "down" });
    }

    const data = await response.json();
    return res.json({ status: "healthy", python_service: data });

  } catch (err) {
    return res.status(503).json({ status: "unhealthy", python_service: "unreachable" });
  }
});

export default router;
