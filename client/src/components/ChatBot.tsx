import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Upload, X, FileText, AlertCircle, Loader2 } from "lucide-react";
import { 
  sendChatMessage, 
  sendChatMessageStream,
  uploadDocumentForAnalysis, 
  ChatResponse,
  StreamEvent,
  clearChatSession 
} from "../api";

interface Message {
  id: string;
  sender: "user" | "bot";
  content: string;
  intent?: string;
  isLoading?: boolean;
}

export function ChatBot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      sender: "bot",
      content: "Hello! I'm your AI legal assistant. I can help you with:\n\n• **Document Analysis** - Upload legal documents for analysis\n• **Crime Reporting** - Get guidance on reporting crimes\n• **Legal Questions** - Ask about Indian laws and statutes\n• **Find Lawyers** - Search for lawyers by specialization\n\nHow can I assist you today?",
    },
  ]);

  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const formatBotResponse = (data: ChatResponse): string => {
    let response = data.response || "";
    
    // Add lawyer info if available
    if (data.lawyers_found && data.lawyers_found.length > 0) {
      response += "\n\n**Found Lawyers:**\n";
      data.lawyers_found.forEach((lawyer, idx) => {
        response += `\n${idx + 1}. **${lawyer.name}** - ${lawyer.specialization}\n`;
        response += `   📍 ${lawyer.location} | ⭐ ${lawyer.rating} | 📞 ${lawyer.contact}\n`;
      });
    }
    
    return response;
  };

  const handleSend = async () => {
    if ((!input.trim() && !selectedFile) || isLoading) return;

    const userText = input;
    const uploadFile = selectedFile;
    setInput("");
    setSelectedFile(null);
    setError(null);

    // Add user message
    const userMessageId = Date.now().toString();
    setMessages(prev => [
      ...prev,
      { 
        id: userMessageId, 
        sender: "user", 
        content: uploadFile ? `📎 ${uploadFile.name}\n${userText || "Please analyze this document"}` : userText 
      },
    ]);

    // Add loading message
    const loadingId = (Date.now() + 1).toString();
    setMessages(prev => [
      ...prev,
      { id: loadingId, sender: "bot", content: "", isLoading: true },
    ]);

    setIsLoading(true);

    try {
      if (uploadFile) {
        // Upload document for analysis (non-streaming)
        const data = await uploadDocumentForAnalysis(
          uploadFile,
          userText || "Please analyze this document",
          sessionId || undefined
        );

        // Store session ID
        if (data.session_id) {
          setSessionId(data.session_id);
        }

        // Replace loading message with actual response
        setMessages(prev => 
          prev.map(msg => 
            msg.id === loadingId 
              ? { 
                  id: loadingId, 
                  sender: "bot", 
                  content: formatBotResponse(data),
                  intent: data.intent,
                  isLoading: false
                } 
              : msg
          )
        );
      } else {
        // Streaming chat message
        let streamedContent = "";

        await sendChatMessageStream(
          userText,
          sessionId || undefined,
          // onToken: update message content incrementally
          (token: string) => {
            streamedContent += token;
            const currentContent = streamedContent;
            setMessages(prev =>
              prev.map(msg =>
                msg.id === loadingId
                  ? { ...msg, content: currentContent, isLoading: false }
                  : msg
              )
            );
          },
          // onDone: finalize with metadata
          (metadata: StreamEvent) => {
            if (metadata.session_id) {
              setSessionId(metadata.session_id);
            }
            // Append lawyer info if available
            let finalContent = streamedContent;
            if (metadata.lawyers_found && metadata.lawyers_found.length > 0) {
              finalContent += "\n\n**Found Lawyers:**\n";
              metadata.lawyers_found.forEach((lawyer, idx) => {
                finalContent += `\n${idx + 1}. **${lawyer.name}** - ${lawyer.specialization}\n`;
                finalContent += `   📍 ${lawyer.location} | ⭐ ${lawyer.rating} | 📞 ${lawyer.contact}\n`;
              });
            }
            setMessages(prev =>
              prev.map(msg =>
                msg.id === loadingId
                  ? { ...msg, content: finalContent, intent: metadata.intent, isLoading: false }
                  : msg
              )
            );
          },
          // onError
          (errorMsg: string) => {
            setError(errorMsg);
            setMessages(prev =>
              prev.map(msg =>
                msg.id === loadingId
                  ? { ...msg, content: "❌ " + errorMsg, isLoading: false }
                  : msg
              )
            );
          }
        );
      }
    } catch (err: any) {
      console.error("Chat error:", err);
      setError(err.message || "Failed to get response");
      
      // Replace loading message with error
      setMessages(prev => 
        prev.map(msg => 
          msg.id === loadingId 
            ? { 
                id: loadingId, 
                sender: "bot", 
                content: "❌ " + (err.message || "Server unreachable. Please try again."),
                isLoading: false
              } 
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Check file size (10MB limit)
      if (file.size > 10 * 1024 * 1024) {
        setError("File too large. Maximum size is 10MB.");
        return;
      }
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleClearSession = async () => {
    if (sessionId) {
      try {
        await clearChatSession(sessionId);
      } catch (e) {
        // Ignore errors when clearing session
      }
    }
    setSessionId(null);
    setMessages([{
      id: "1",
      sender: "bot",
      content: "Session cleared! How can I help you today?",
    }]);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">

        {/* HEADER */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Bot className="w-8 h-8" />
              <div>
                <h2 className="text-lg font-semibold">Legal AI Assistant</h2>
                <p className="text-blue-100 text-sm">Powered by Indian Legal Knowledge</p>
              </div>
            </div>
            {sessionId && (
              <button
                onClick={handleClearSession}
                className="text-blue-100 hover:text-white text-sm px-3 py-1 rounded border border-blue-400 hover:border-white transition-colors"
              >
                New Chat
              </button>
            )}
          </div>
        </div>

        {/* ERROR BANNER */}
        {error && (
          <div className="bg-red-50 border-b border-red-200 px-4 py-2 flex items-center gap-2 text-red-700">
            <AlertCircle className="w-4 h-4" />
            <span className="text-sm">{error}</span>
            <button onClick={() => setError(null)} className="ml-auto">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* CHAT BODY */}
        <div className="h-[500px] overflow-y-auto p-6 space-y-4 bg-gray-50">
          {messages.map(msg => (
            <div
              key={msg.id}
              className={`flex gap-3 ${
                msg.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {msg.sender === "bot" && (
                <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}

              <div
                className={`max-w-[70%] p-4 rounded-lg ${
                  msg.sender === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-white border border-gray-200"
                }`}
              >
                {msg.isLoading ? (
                  <div className="flex items-center gap-2 text-gray-500">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Thinking...</span>
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap">
                    {msg.content.split('\n').map((line, i) => (
                      <p key={i} className={line.startsWith('**') ? 'font-semibold' : ''}>
                        {line.replace(/\*\*/g, '')}
                      </p>
                    ))}
                  </div>
                )}
                {msg.intent && (
                  <div className="mt-2 text-xs text-gray-400">
                    Intent: {msg.intent}
                  </div>
                )}
              </div>

              {msg.sender === "user" && (
                <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-gray-700" />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* FILE PREVIEW */}
        {selectedFile && (
          <div className="px-4 py-2 bg-blue-50 border-t border-blue-200 flex items-center gap-2">
            <FileText className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-700 flex-1 truncate">{selectedFile.name}</span>
            <button 
              onClick={() => setSelectedFile(null)}
              className="text-blue-600 hover:text-blue-800"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* INPUT */}
        <div className="p-4 bg-white border-t border-gray-200 flex gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-3 text-gray-500 hover:text-blue-600 hover:bg-gray-100 rounded-lg transition-colors"
            title="Upload document"
          >
            <Upload className="w-5 h-5" />
          </button>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder={selectedFile ? "Add a message about the document..." : "Describe your legal question..."}
            className="flex-1 px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-600 focus:outline-none"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || (!input.trim() && !selectedFile)}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
            Send
          </button>
        </div>

      </div>
    </div>
  );
}
