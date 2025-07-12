# Travel Assistant Chatbot - LangChain Migration

## Overview
This travel chatbot has been completely rewritten using LangChain's agentic framework with the following major improvements:

## ✅ New Features Implemented

### 1. LangChain Agentic Framework
- **Tool Decorators**: All tools are now defined using LangChain's `@tool` decorator for better organization
- **Agent Executor**: Uses LangChain's `AgentExecutor` with OpenAI function calling
- **Proper Tool Management**: Tools are properly registered and managed through LangChain's system

### 2. Persistent Multi-User Chat History
- **ChromaDB Storage**: All conversations are now stored in ChromaDB with user context
- **Multi-User Support**: Each user has their own isolated conversation history
- **Persistent Storage**: Chat history survives application restarts
- **User Management**: Simple user ID system for switching between users

### 3. Enhanced Date Handling
- **Current Date Tool**: New `get_current_date_time()` tool provides real-time date information
- **Date Awareness**: LLM no longer relies on training cutoff for current date queries
- **Timezone Support**: Returns UTC time with proper timezone information
- **Context Integration**: Date information is automatically used when users ask about "today", "now", etc.

### 4. Improved Architecture
- **Better Error Handling**: More robust error management and fallbacks
- **Tool Result Format**: All tools return JSON strings for better parsing
- **Context Management**: Enhanced conversation context with relevant history retrieval
- **Memory Integration**: Smart retrieval of relevant past conversations and knowledge

## 🛠️ Technical Changes

### Tools with Decorators
```python
@tool
def get_current_date_time() -> str:
    """Get the current date and time. Use this tool when users ask about 'today', 'now', 'current time', or any date-related queries."""
    current_time = datetime.now(timezone.utc)
    return f"Current UTC date and time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}..."

@tool  
async def search_hotels(destination: str, check_in: str, check_out: str, guests: int = 2) -> str:
    """Search for hotels in a specific destination with check-in and check-out dates."""
    # Implementation...
```

### Multi-User Conversation Management
```python
class PersistentMultiUserConversationManager:
    def store_conversation(self, user_id: str, conversation_id: str, user_message: str, ai_response: str, tool_calls: Optional[List[Dict]] = None):
        # Stores conversation in ChromaDB with user context
        
    def get_conversation_history(self, user_id: str, conversation_id: str, limit: int = 10) -> List[Dict]:
        # Retrieves user-specific conversation history
```

### Enhanced WebSocket Support
- Updated WebSocket endpoints: `/ws/{user_id}/{conversation_id}`
- User session management in frontend
- Real-time user switching capability

## 🌐 API Changes

### New Endpoints
- `GET /api/users/{user_id}/conversations` - Get all conversations for a user
- `GET /api/conversation/{user_id}/{conversation_id}` - Get specific conversation
- `DELETE /api/conversation/{user_id}/{conversation_id}` - Clear specific conversation

### Updated Endpoints
- `POST /api/chat` - Now requires `user_id` field
- `WebSocket /ws/{user_id}/{conversation_id}` - User-specific WebSocket connections

## 🎯 Usage Examples

### Testing Date Awareness
Try asking:
- "What's the weather today in Ho Chi Minh City?"
- "What's the current date and time?"
- "Plan a trip for next week"

The bot will now use the `get_current_date_time` tool to get accurate current information.

### Testing Multi-User Support
1. Enter a User ID (e.g., "alice") and click "Set User ID"
2. Have a conversation
3. Switch to a different User ID (e.g., "bob") 
4. Notice that chat history is isolated per user
5. Switch back to "alice" to see persistent history

### Testing Persistent Storage
1. Have a conversation with the bot
2. Restart the application (`Ctrl+C` and `python main.py`)
3. Open the chat interface
4. Ask "What did we talk about before?" 
5. The bot should remember previous conversations from ChromaDB

## 🔧 Configuration

### Environment Variables
Make sure your `.env` file includes:
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_KEY_EMBEDDING=your_embedding_api_key  # Can be the same as above
OPENAI_BASE_URL=https://aiportalapi.stu-platform.live/jpe
OPENAI_MODEL_NAME=GPT-4o-mini

# Optional API keys for real data
RAPIDAPI_KEY=your_rapidapi_key
AMADEUS_API_KEY=your_amadeus_key
AMADEUS_API_SECRET=your_amadeus_secret
OPENWEATHER_API_KEY=your_openweather_key
EXCHANGERATE_API_KEY=your_exchangerate_key
FOURSQUARE_API_KEY=your_foursquare_key
```

## 🚀 Running the Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the application
python main.py
```

The application will be available at:
- **Chat Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔄 Migration Benefits

1. **Better Tool Management**: LangChain's decorator system is cleaner and more maintainable
2. **Persistent Memory**: Users no longer lose chat history on refresh/restart
3. **Multi-User Capability**: Multiple users can use the system simultaneously
4. **Accurate Date Handling**: No more outdated date assumptions from LLM training
5. **Improved Error Handling**: More robust error management and fallbacks
6. **Better Architecture**: Cleaner separation of concerns with LangChain agents

## 🧪 Testing Scenarios

1. **Date Query Test**: Ask "What's the weather today in Ho Chi Minh City?" and verify it uses current date
2. **Multi-User Test**: Switch between different user IDs and verify conversation isolation
3. **Persistence Test**: Restart the app and verify chat history is maintained
4. **Tool Functionality**: Test all travel tools (hotels, flights, weather, activities, currency)
5. **Memory Test**: Ask the bot to remember something, then ask about it later
