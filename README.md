# TravelBot

**Live Demo:** [https://travel.elevateaihcm.tech/](https://travel.elevateaihcm.tech/)

*If the demo link is unavailable, you can run the project locally by following the setup instructions below.*

*See also: [PROMPT_ENGINEERING.md](PROMPT_ENGINEERING.md), [USE_CASES.md](USE_CASES.md), and [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md) for more details.*

A sophisticated travel chatbot built with OpenAI SDK, featuring real-time chat, function calling, text-to-speech, and external API integrations for comprehensive travel planning assistance.

---

## 🌟 Features

### Core
- **🏨 Hotel Search:** Find and compare hotels with pricing, ratings, and amenities
- **✈️ Flight Search:** Search flights with multiple options for different dates
- **🌤️ Weather Info:** Get weather forecasts for travel planning
- **🎭 Local Activities:** Discover activities, attractions, and experiences
- **💬 Real-time Chat:** WebSocket-powered instant messaging
- **🔧 Function Calling:** OpenAI function calling for dynamic data retrieval
- **🗣️ Text-to-Speech:** Convert chatbot responses to speech for accessible, hands-free interaction

### Advanced
- **Multi-turn Conversations:** Context-aware conversation management
- **Conversation History:** Persistent chat history per session
- **Mock Data Generation:** Realistic travel data simulation
- **Responsive Web UI:** Modern, mobile-friendly interface
- **Connection Management:** Auto-reconnection and status indicators
- **Semantic Memory:** Uses OpenAI embeddings to vectorize conversation data for improved context retention and retrieval (via ChromaDB)

---

## 🚀 Technologies Used

- **Backend:** FastAPI + Python
- **AI:** OpenAI SDK (function calling & embeddings)
- **Frontend:** Vanilla JavaScript + WebSockets
- **Styling:** Custom CSS
- **Data:** Mock APIs for travel services
- **Vector DB:** ChromaDB for storing/retrieving vectorized conversation memory
- **Text-to-Speech:** Integrated TTS (browser-based or via API)

---

## 📋 Setup Instructions

### Prerequisites
- Python 3.8+
- OpenAI API key (and other API keys for full functionality)

### Installation

1. **Clone the project:**
   ```powershell
   git clone <your-repo-url>
   cd TravelAssistantChatbot_v2/TravelAssistantChatbot_v2
   ```
2. **Create virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   ```powershell
   copy .env.example .env
   # Edit .env with your API keys (see API_SETUP_GUIDE.md)
   ```
5. **Run the application:**
   - Using script: `./start.sh`
   - Or: `python main.py`
6. **Open in browser:** [http://localhost:8000](http://localhost:8000)

---

## 📁 Project Structure

```
TravelAssistantChatbot_v2/
├── .env               # Environment variables
├── .env.example       # Example env config
├── API_SETUP_GUIDE.md # API setup documentation
├── main.py            # Main FastAPI app
├── requirements.txt   # Python dependencies
├── start.sh           # Startup script
├── chroma_db/         # Vector DB for conversation memory
├── static/            # Static web assets (CSS, JS, images)
└── templates/         # Jinja2 HTML templates
    ├── index.html     # Main chat interface
    └── result.html    # Summary/result template
```

---

## 💡 Usage Examples

Try these prompts in the chat:
- "Plan a 3-day trip to Rome"
- "Find me a hotel in Paris for next weekend"
- "Search flights from New York to Tokyo"
- "What's the weather like in London?"
- "Suggest cultural activities in Barcelona"

---

## 🔧 API Endpoints

### WebSocket
- `ws://localhost:8000/ws/{conversation_id}` — Real-time chat

### REST API
- `POST /api/chat` — Send chat message
- `GET /api/conversation/{conversation_id}` — Get conversation history
- `DELETE /api/conversation/{conversation_id}` — Clear conversation
- `GET /health` — Health check

### Function Calling
The chatbot uses OpenAI's function calling feature with these functions:
- `search_hotels(destination, check_in, check_out, guests)`
- `search_flights(origin, destination, departure_date, return_date)`
- `get_weather(city, date)`
- `get_local_activities(destination, activity_type)`
- `exchange_currency(amount, from_currency, to_currency)`

---

## 🏗️ Architecture

### Backend
```
main.py                 # Main FastAPI application
├── Function Definitions # OpenAI function schemas
├── Mock APIs           # Simulated travel services
├── WebSocket Handler   # Real-time chat management
├── Conversation Manager # Chat history and context
├── Embedding & Vectorization # Uses OpenAI embeddings to vectorize messages
├── ChromaDB Integration # Stores and retrieves conversation vectors for semantic memory
├── Text-to-Speech API  # Converts text responses to audio (if enabled)
└── REST Endpoints      # API interfaces
```

### Frontend
```
templates/index.html    # Chat interface
├── TravelChatbot Class # Main chat logic
├── WebSocket Client    # Real-time communication
├── Message Formatting  # Markdown rendering
├── Text-to-Speech UI   # Play/pause audio for bot responses
└── UI Components       # Chat bubbles, indicators
```

---

## 🌐 Workshop Context

This project addresses the **Workshop 2** objective of building real-world chatbot systems using Azure OpenAI API. It demonstrates:
- Multi-turn Conversation Management
- Function Calling for External APIs
- Mock Data Generation
- Sophisticated Prompting Techniques
- Real-time Chat Interface
- Context-aware Responses

### Problem Solved
The chatbot addresses the real-life challenge of **travel planning complexity** by providing a single interface for:
- Hotel booking assistance
- Flight search and comparison
- Weather-based travel planning
- Local activity discovery
- Comprehensive itinerary planning

---

## 🚀 Getting Started

1. **Quick Start:**
   ```powershell
   python main.py
   ```
2. **Try these prompts:**
   - "Plan a 3-day trip to Rome"
   - "Find me a hotel in Paris for next weekend"
   - "Search flights from New York to Tokyo"
   - "What's the weather like in London?"
   - "Suggest cultural activities in Barcelona"

---

## 📝 Development Notes

- Mock APIs simulate realistic travel data with random pricing and availability
- Function calling enables dynamic data retrieval based on user queries
- WebSocket implementation provides real-time chat experience
- Conversation context is maintained across multiple exchanges
- **Embeddings & Vectorization:** User and bot messages are embedded using OpenAI models, vectorized, and stored in ChromaDB for semantic search and context retrieval
- **ChromaDB:** Efficient vector database for storing and retrieving conversation memory, enabling context-aware and relevant responses
- **Text-to-Speech:** Bot responses can be played as audio, improving accessibility and user experience
- Error handling and reconnection logic ensure robust user experience

### Important Notes
- The project includes a `.env.example` file — copy this to `.env` and add your API keys
- The `start.sh` script provides an easy way to launch the application
- The `Makefile` contains some legacy references (e.g., meeting transcript summarizer); you may update it for travel chatbot tasks as needed
- `.gitignore` is configured to protect secrets, data, and development artifacts

---

## 🛠️ Development Setup

For development, you may want to install additional tools:
```powershell
pip install black flake8 pytest

# Format code
black main.py

# Run linting
flake8 main.py
```

---

## 🔮 Future Enhancements

- Integration with real travel APIs (Booking.com, Expedia, etc.)
- User authentication and profile management
- Booking confirmation and itinerary saving
- Multi-language support
- Voice interface integration
- Trip budget tracking and recommendations

---
