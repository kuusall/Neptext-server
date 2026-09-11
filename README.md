# Neptext Server 🇳🇵

Neptext Server is a high-performance FastAPI-based REST API designed for advanced Nepali text processing. It provides a suite of Natural Language Processing (NLP) tools, including sentiment analysis, context-aware word prediction, and rule-based spell correction.

## Use of claude :

This is deployed in railway through docker image. So, claude contributed to just the API connection and Deployment part of it. FastAPI deployed in Railway for the 3 Models to run. FYI 

## 🚀 Features

- **Sentiment Analysis**: Classifies Nepali text into 5 distinct sentiment levels: `positive`, `semi_positive`, `neutral`, `semi_negative`, and `negative`.
- **Spell Correction**: Identifies misspelled Nepali words and provides intelligent suggestions based on a custom lexicon and n-gram models.
- **Word Prediction**: Context-aware next-word prediction to power autocomplete features, blending corpus frequency with recent token windows.
- **API Gateway**: Integrated logging and request tracking for all incoming API calls.

## 🛠️ Tech Stack

- **Language**: Python 3.10
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Server**: [Uvicorn](https://www.uvicorn.org/)
- **ML Libraries**: 
  - `PyTorch` & `HuggingFace Transformers` (for sentiment analysis)
  - `Scikit-Learn` & `Joblib` (for traditional ML models)
  - `NumPy` & `SciPy` (for matrix operations and feature extraction)
- **Containerization**: [Docker](https://www.docker.com/)

## 📦 Installation & Setup

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/neptext-server.git
   cd neptext-server
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   uvicorn app:app --reload
   ```
   The server will start at `https://neptext-server-production.up.railway.app`.

### Docker Deployment (Recommended)

Build and run the API using Docker for a consistent environment:

```bash
# Build the image
docker build -t neptext-server .

# Run the container
docker run -p 8000:8000 neptext-server
```

## 🔌 API Usage

Once the server is running, you can access the interactive API documentation at:
- **Swagger UI**: `https://neptext-server-production.up.railway.app/docs`
- **ReDoc**: `https://neptext-server-production.up.railway.app/redoc`

### Example Endpoints

#### 1. Sentiment Analysis
**Endpoint**: `POST /sentiment`
```bash
curl -X POST "https://neptext-server-production.up.railway.app/sentiment" \
  -H "Content-Type: application/json" \
  -d '{"text":"यो movie ramro cha 😊"}'
```

#### 2. Spell Correction
**Endpoint**: `POST /spell-correct`
```bash
curl -X POST "https://neptext-server-production.up.railway.app/spell-correct" \
  -H "Content-Type: application/json" \
  -d '{"text":"म नेपाल जाान्छु।", "suggest_only": true}'
```

#### 3. Word Prediction
**Endpoint**: `POST /word-predict`
```bash
curl -X POST "https://neptext-server-production.up.railway.app/word-predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"यो नीतिमा सुधार", "top_k": 5}'
```

## 📂 Project Structure

```text
.
├── app.py              # Main FastAPI entry point
├── config/             # Configuration and environment settings
├── model_dicts/        # Model weights, lexicons, and resource files
├── models/             # Model logic and preprocessing wrappers
├── schemas/            # Pydantic models for request/response validation
├── services/           # Business logic and service layer
├── requirements.txt    # Python dependencies
└── Dockerfile          # Docker configuration
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
