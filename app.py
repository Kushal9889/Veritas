"""
Veritas Flask API — Production-Optimized
G3: Returns metadata (category, sources, timing) alongside answer.
X2: Warms up reranker on startup.
Serves both /query (JSON) and /chat (SSE streaming) for Next.js frontend.
"""
import time
import json
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from src.orchestration.workflow import build_graph
from src.retrieval.search import warmup as warmup_reranker
from src.core.telemetry import get_telemetry

telemetry = get_telemetry('app')

app = Flask(__name__)

# CORS support for Next.js frontend on localhost:3000
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    return response

# Build graph once at startup
graph_app = build_graph()

# X2: Warm up reranker model on startup
try:
    print("🔥 Warming up reranker model...")
    warmup_reranker()
    print("✅ Reranker ready")
except Exception as e:
    print(f"⚠️ Reranker warmup failed: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    """JSON endpoint — returns answer + metadata."""
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    start_time = time.time()
    try:
        result = graph_app.invoke({'question': question})
        duration_ms = round((time.time() - start_time) * 1000, 2)
        answer = result.get('generation', 'No answer generated')

        response = {
            'answer': answer,
            'success': True,
            'metadata': {
                'category': result.get('query_category', 'unknown'),
                'duration_ms': duration_ms,
                'vector_results': len(result.get('documents', [])),
                'graph_results': len(result.get('graph_data', [])),
            }
        }
        return jsonify(response)

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        telemetry.log_error("Query failed", error=e, duration_ms=duration_ms)
        return jsonify({
            'error': str(e),
            'success': False,
            'metadata': {'duration_ms': duration_ms}
        }), 500


@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """SSE streaming endpoint for Next.js frontend."""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    def generate():
        start_time = time.time()
        try:
            # Collect streamed text from generate_node
            collected_text = []

            def stream_callback(text_chunk):
                collected_text.append(text_chunk)

            # Run the full graph — generate_node uses stream_callback internally
            # We need to invoke the graph and stream the result
            result = graph_app.invoke({'question': question})
            answer = result.get('generation', 'No answer generated')

            # Build source references from documents
            documents = result.get('documents', [])
            sources = []
            for doc in documents[:5]:
                # Extract source info from formatted doc string
                if doc.startswith('[') and ']' in doc:
                    source_ref = doc[1:doc.index(']')]
                    sources.append(source_ref)

            # Stream the answer in chunks for a natural feel
            chunk_size = 20  # characters per chunk
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'text': chunk})}\n\n"

            # Send sources
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Done signal
            duration_ms = round((time.time() - start_time) * 1000, 2)
            yield f"data: {json.dumps({'type': 'metadata', 'category': result.get('query_category', 'unknown'), 'duration_ms': duration_ms})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            telemetry.log_error("Chat failed", error=e)
            yield f"data: {json.dumps({'type': 'content', 'text': f'⚠️ Error: {str(e)}'})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'veritas'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
