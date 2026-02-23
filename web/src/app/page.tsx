"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "I am Veritas. I have access to the company's financials (10-K), strategic risks (Graph), and vector memory. What do you need to know?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    let streamedContent = "";
    let sources: string[] = [];
    let messageIndex = -1;

    try {
      const baseUrl = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';
      const res = await fetch(`${baseUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg, user_id: "web-client" }),
      });

      if (!res.ok) throw new Error("Server error");

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No reader");

      const decoder = new TextDecoder();
      let buffer = "";

      // Add empty assistant message
      setMessages((prev) => {
        messageIndex = prev.length;
        return [...prev, { role: "assistant", content: "" }];
      });

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim();
            if (dataStr === '[DONE]') continue;

            try {
              const data = JSON.parse(dataStr);
              
              if (data.type === 'content') {
                streamedContent += data.text;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[messageIndex] = {
                    role: "assistant",
                    content: streamedContent,
                    sources: sources.length > 0 ? sources : undefined
                  };
                  return updated;
                });
              } else if (data.type === 'sources') {
                sources = data.sources || [];
              }
            } catch (e) {
              console.error('Parse error:', e, dataStr);
            }
          }
        }
      }

      // Final update with sources
      if (sources.length > 0) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[messageIndex] = {
            role: "assistant",
            content: streamedContent,
            sources: sources
          };
          return updated;
        });
      }

    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "⚠️ Error connecting to Veritas Brain. Is the backend running?" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-black text-gray-100 font-sans">
      {/* HEADER */}
      <header className="flex items-center px-6 py-4 border-b border-gray-800 bg-gray-950">
        <Sparkles className="w-6 h-6 text-emerald-500 mr-2" />
        <h1 className="text-xl font-semibold tracking-tight">Veritas <span className="text-gray-500">Financial Intelligence</span></h1>
      </header>

      {/* CHAT AREA */}
      <main className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-gray-800">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "assistant" && (
              <div className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center shrink-0">
                <Bot className="w-5 h-5 text-emerald-400" />
              </div>
            )}

            <div
              className={`max-w-2xl px-5 py-3 rounded-2xl leading-relaxed text-sm shadow-sm ${
                msg.role === "user"
                  ? "bg-emerald-600 text-white rounded-br-none"
                  : "bg-gray-900 border border-gray-800 text-gray-200 rounded-bl-none"
              }`}
            >
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown 
                  components={{
                    strong: ({node, ...props}) => <span className="font-bold text-emerald-300" {...props} />
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              </div>

              {/* SOURCES AREA */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-4 pt-3 border-t border-gray-700/50">
                  <p className="text-xs text-gray-500 font-medium mb-2">VERIFIED SOURCES:</p>
                  <div className="flex flex-wrap gap-2">
                    {msg.sources.map((src, i) => (
                      <div key={i} className="text-[10px] bg-emerald-950/30 px-2 py-1 rounded border border-emerald-900/50 text-emerald-400 font-mono">
                         {src}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {msg.role === "user" && (
              <div className="w-8 h-8 rounded-full bg-emerald-700 flex items-center justify-center shrink-0">
                <User className="w-5 h-5 text-white" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-4">
             <div className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center">
                <Bot className="w-5 h-5 text-emerald-400 animate-pulse" />
              </div>
              <div className="text-sm text-gray-500 flex items-center">Thinking...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* INPUT AREA */}
      <footer className="p-4 border-t border-gray-800 bg-gray-950">
        <div className="max-w-3xl mx-auto relative">
          <input
            type="text"
            className="w-full bg-gray-900 border border-gray-700 text-white rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 placeholder-gray-500"
            placeholder="Ask a strategic question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading}
            className="absolute right-2 top-2 p-1.5 bg-emerald-600 rounded-lg hover:bg-emerald-500 transition-colors disabled:opacity-50"
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
      </footer>
    </div>
  );
}