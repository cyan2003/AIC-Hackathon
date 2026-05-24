# Intelligent Recruiter Agent - Setup Guide

## Overview
This guide helps new team members set up and run the Intelligent Recruiter Agent project.

## Prerequisites
- Python 3.11 or higher
- Git
- API keys for Chutes.ai and optionally OpenAI/Anthropic

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd intelligent-recruiter
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file and add your API keys
# Get your Chutes API key from: https://chutes.ai/app
# You can also add OpenAI/Anthropic keys if needed
```

### 5. Run the Application
```bash
streamlit run ui/app.py
```

The application will be available at http://localhost:8501

## Project Structure
```
intelligent-recruiter/
├── api/              # API endpoints (if applicable)
├── ingest/           # Document processing pipeline
├── retrieval/        # Search and ranking logic
├── agent/            # LangGraph agent implementation
├── ui/               # Streamlit interface components
├── cache/            # Temporary storage
├── observability/    # Logging and monitoring
├── evals/            # Testing and evaluation scripts
├── data/             # Persistent data storage
├── scripts/          # Utility scripts
├── tests/            # Unit and integration tests
├── requirements.txt  # Python dependencies
├── .env.example      # Environment template
└── README.md         # Project overview
```

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where possible
- Write descriptive function and variable names
- Add docstrings for public functions and classes

### Git Workflow
1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Commit with descriptive messages: `git commit -m "Add feature: description"`
4. Push to remote: `git push origin feature/your-feature-name`
5. Open a pull request for review

### Testing
- Write tests for new functionality
- Run tests with: `pytest` (when implemented)
- Aim for >80% test coverage

## Troubleshooting

### Common Issues
1. **Module not found errors**: Ensure you're in the virtual environment and dependencies are installed
2. **API key errors**: Verify your `.env` file contains valid API keys
3. **Port already in use**: Change the port in `.env` or stop the conflicting service
4. **Memory issues**: Consider reducing batch sizes in processing scripts

### Getting Help
- Check the logs in the `logs/` directory
- Review error messages in the Streamlit UI
- Consult the documentation in each module's directory
- Ask team members in the project communication channels

## Additional Resources
- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Chutes.ai Documentation](https://chutes.ai/docs)