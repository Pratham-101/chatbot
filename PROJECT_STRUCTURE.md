# Clean Project Structure

```
mutual_fund_chatbot/
├── src/                    # Source code
│   ├── api/               # API layer
│   │   ├── routes/        # API routes (chat.py, health.py)
│   │   ├── middleware/    # Middleware
│   │   ├── models/        # Pydantic schemas
│   │   └── main.py        # FastAPI app
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration management
│   │   ├── logging.py     # Structured logging
│   │   └── exceptions.py  # Custom exceptions
│   ├── services/          # Business logic
│   │   ├── chatbot/       # Chatbot services
│   │   │   ├── agents/    # Multi-agent system
│   │   │   ├── enhanced_chatbot.py
│   │   │   └── knowledge_graph.py
│   │   ├── data/          # Data services
│   │   │   ├── real_time_data.py
│   │   │   ├── web_search.py
│   │   │   └── retrieval.py
│   │   └── llm/           # LLM services
│   │       ├── generation.py
│   │       └── response_quality.py
│   ├── utils/             # Utilities
│   │   └── helpers.py
│   └── ui/                # User interface
│       └── streamlit_app.py
├── tests/                 # Test suite
│   ├── unit/
│   └── integration/
├── docker/                # Docker configuration
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── requirements/          # Dependencies
├── main.py                # Entry point
├── pyproject.toml         # Project configuration
├── Makefile               # Build automation
└── README.md              # Project documentation
```
