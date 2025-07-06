# 🤖 Mutual Fund Chatbot

A production-ready mutual fund chatbot with multi-agent evaluation, real-time data integration, and ChatGPT-style responses.

## 🚀 Features

- Multi-Agent Architecture
- Real-time Data Integration
- Quality Evaluation
- Structured Responses
- Production Ready
- Modern UI

## 🛠️ Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd mutual-fund-chatbot

# Install dependencies
make install-dev

# Set environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# Run
make run-dev
```

## 📊 API Usage

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Ask question
curl -X POST http://localhost:8000/api/v1/query \n  -H "Content-Type: application/json" \n  -d '{"text": "Tell me about HDFC Defence Fund?"}'
```

## 🏗️ Architecture

- **API Layer**: FastAPI with structured logging
- **Services**: Multi-agent chatbot system
- **Data**: Vector store + real-time web search
- **LLM**: Groq integration with quality evaluation
- **UI**: Streamlit interface

## 📈 Production Features

- Docker containerization
- Health checks & monitoring
- Rate limiting & security
- Structured logging
- CI/CD ready

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit PR

## 📄 License

MIT License
