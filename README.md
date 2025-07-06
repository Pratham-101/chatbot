## LLM Provider Configuration

This chatbot supports both Groq (cloud, free, fast) and Ollama (local, private) as LLM backends for all agentic and RAG workflows. No OpenAI key is required.

### To use Groq (default, recommended):

1. Get a free Groq API key from https://console.groq.com/keys
2. Set the following environment variables:
   ```bash
   export LLM_PROVIDER=groq
   export GROQ_API_KEY=your-groq-key
   export GROQ_MODEL=llama3-70b-8192  # or mixtral-8x7b-32768, gemma-7b-it, etc.
   ```
3. Start the backend as usual.

### To use Ollama (local):

1. Install Ollama from https://ollama.com/download and run a model, e.g.:
   ```bash
   ollama run llama3
   ```
2. Set the following environment variables:
   ```bash
   export LLM_PROVIDER=ollama
   export OLLAMA_MODEL=llama3  # or mistral, phi, etc.
   export OLLAMA_API_URL=http://127.0.0.1:11434/api/generate  # default
   ```
3. Start the backend as usual.

### Switching Providers
- Change the `LLM_PROVIDER` environment variable to `groq` or `ollama` as needed.
- No code changes required. 