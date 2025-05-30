# Hospital System Chatbot

A conversational AI system that helps users query hospital-related information using natural language. The system uses Neo4j for graph database storage and LangChain for RAG (Retrieval Augmented Generation) capabilities.

## Features

- Natural language querying of hospital data
- Graph-based data storage using Neo4j
- RAG implementation for enhanced response generation
- Interactive Streamlit frontend
- FastAPI backend

## Prerequisites

- Python 3.8+
- Neo4j Database
- Docker (optional)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Kjain02/hospital-system-chatbot.git
cd hospital-system-chatbot
```

2. Create and activate virtual environment:
```bash
python -m venv chatenv
# On Windows
.\chatenv\Scripts\activate
# On Unix/MacOS
source chatenv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:
```env
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
```

## Running the Application

### ETL Pipeline (Neo4j Data Loading)

1. Navigate to the ETL directory:
```bash
cd hospital_neo4j_etl
```

2. Run the ETL pipeline:
```bash
python src/hospital_csv_write.py
```

This will process and load the hospital data into your Neo4j database.

### Backend (FastAPI)

1. Navigate to the backend directory:
```bash
cd chatbot
```

2. Start the FastAPI server:
```bash
uvicorn src.main:app --reload
```

The backend will be available at `http://localhost:8000`

### Frontend (Streamlit)

1. Navigate to the frontend directory:
```bash
cd chatbot_frontend
```

2. Start the Streamlit application:
```bash
streamlit run src/main.py
```

The frontend will be available at `http://localhost:8501`

## Docker Deployment

To run the application using Docker:

1. Build the Docker images:
```bash
docker-compose build
```

2. Start the containers:
```bash
docker-compose up
```

## Project Structure

```
hospital-system-chatbot/
├── chatbot/                           # Backend FastAPI application
│   ├── src/
│   │   ├── agents/                    # LangChain agents
│   │   │   ├── __init__.py
│   │   │   └── hospital_rag_agent.py  # RAG agent implementation
│   │   ├── chains/                    # Custom chains
│   │   │   ├── __init__.py
│   │   │   ├── hospital_cypher_chain.py    # Neo4j query chain
│   │   │   └── hospital_review_chain.py    # Review analysis chain
│   │   ├── models/                    # Pydantic models
│   │   │   ├── __init__.py
│   │   │   └── hospital_rag_query.py  # Input/Output models
│   │   └── main.py                    # FastAPI application
│   ├── pyproject.toml                 # Backend dependencies
│   └── Dockerfile                     # Backend Docker configuration
├── chatbot_frontend/                  # Frontend Streamlit application
│   ├── src/
│   │   ├── __init__.py
│   │   └── main.py                    # Streamlit application
│   ├── pyproject.toml                 # Frontend dependencies
│   └── Dockerfile                     # Frontend Docker configuration
├── hospital_neo4j_etl/                # ETL pipeline for Neo4j
│   ├── src/
│   │   ├── entrypoint.sh
│   │   └── hospital_csv_write.py      # Data extracting and loading scripts
│   ├── pyproject.toml                 # ETL dependencies  
│   └── Dockerfile                     # ETL Docker configuration    
├── .env.example                       # Example environment variables
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Project dependencies
├── docker-compose.yml                 # Docker services configuration
└── README.md                          # Project documentation
```


## API Endpoints

- `POST /hospital-rag-agent`: Main endpoint for processing hospital-related queries
  - Input: Natural language query
  - Output: Structured response with intermediate steps

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.