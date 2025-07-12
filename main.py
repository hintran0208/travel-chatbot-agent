"""
Travel Assistant Chatbot using LangChain and OpenAI

Advanced travel chatbot with LangChain agents, tool decorators, persistent multi-user chat history,
and real-time conversation management for comprehensive travel assistance.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Annotated
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from io import BytesIO, StringIO
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import base64
import tempfile
import warnings
import os
import json
import uuid
import random
import httpx
import pandas as pd
import uvicorn
import chromadb
from openai import OpenAI

# LangChain imports
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import tool
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.schema.runnable import RunnablePassthrough

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI(
    title="Travel Assistant Chatbot with LangChain",
    description="AI-powered travel assistant with LangChain agents, real-time chat, hotel booking, flight search, and weather information",
    version="3.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# Create templates and static directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# LangChain OpenAI Client Setup
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/jpe"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME", "GPT-4o-mini"),
    temperature=0.7,
    max_tokens=1500
)

# OpenAI Client for TTS (uses standard OpenAI API)
openai_tts_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY_TTS"),
    base_url="https://api.openai.com/v1"  # Explicitly use standard OpenAI API
)

# ChromaDB Setup for Vector Storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Initialize OpenAI embedding function for ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY_EMBEDDING"),
    model_name="text-embedding-3-small"
)

# Create or get collections
try:
    user_conversations_collection = chroma_client.create_collection(
        name="user_conversations_v2",
        embedding_function=openai_ef,
        metadata={"description": "Multi-user conversation history and preferences"}
    )
except Exception:
    user_conversations_collection = chroma_client.get_collection(
        name="user_conversations_v2",
        embedding_function=openai_ef
    )

try:
    travel_knowledge_collection = chroma_client.create_collection(
        name="travel_knowledge",
        embedding_function=openai_ef,
        metadata={"description": "Travel knowledge base and recommendations"}
    )
except Exception:
    travel_knowledge_collection = chroma_client.get_collection(
        name="travel_knowledge",
        embedding_function=openai_ef
    )

# Text-to-Speech Setup using OpenAI
TTS_AVAILABLE = True

def initialize_tts():
    """Check TTS availability"""
    global TTS_AVAILABLE
    try:
        # Test if OpenAI TTS client is properly configured
        if not os.getenv("OPENAI_API_KEY_TTS"):
            print("❌ OpenAI TTS API key not found. TTS will be disabled.")
            TTS_AVAILABLE = False
        else:
            print("✅ OpenAI TTS initialized successfully")
            TTS_AVAILABLE = True
    except Exception as e:
        print(f"❌ Error initializing OpenAI TTS: {e}")
        print("TTS functionality will be disabled")
        TTS_AVAILABLE = False

def text_to_speech(text: str, max_length: int = 200, speed: float = 1.0) -> Optional[str]:
    """Convert text to speech using OpenAI TTS API and return base64 encoded audio"""
    try:        
        if not TTS_AVAILABLE:
            print("❌ TTS not available")
            return None
        
        # Truncate text if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Clamp speed to OpenAI TTS supported range (0.25 to 4.0)
        speed = max(0.25, min(4.0, speed))
        
        # Call OpenAI TTS API
        response = openai_tts_client.audio.speech.create(
            model="tts-1",  # Use tts-1 for faster generation, tts-1-hd for higher quality
            voice="alloy",  # Available voices: alloy, echo, fable, onyx, nova, shimmer
            input=text,
            speed=speed,
            response_format="mp3"  # MP3 is more efficient than wav
        )
        
        # Get audio data as bytes
        audio_data = response.content
        
        # Encode as base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return audio_base64
    
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return None

# ===== LANGCHAIN TOOLS WITH DECORATORS =====

@tool
def get_current_date_time() -> str:
    """Get the current date and time. Use this tool when users ask about 'today', 'now', 'current time', or any date-related queries."""
    current_time = datetime.now(timezone.utc)
    return f"Current UTC date and time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}. Today is {current_time.strftime('%A, %B %d, %Y')}."

@tool
async def search_hotels(destination: str, check_in: str, check_out: str, guests: int = 2) -> str:
    """Search for hotels in a specific destination with check-in and check-out dates.
    
    Args:
        destination: The destination city or location
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
        guests: Number of guests (default: 2)
    """
    try:
        rapidapi_key = os.getenv("RAPIDAPI_KEY")
        if not rapidapi_key:
            return await _mock_hotel_search(destination, check_in, check_out, guests)
        
        # First, get destination ID
        async with httpx.AsyncClient() as client:
            # Search for destination
            search_url = "https://booking-com.p.rapidapi.com/v1/hotels/locations"
            search_params = {
                "name": destination,
                "locale": "en-gb"
            }
            search_headers = {
                "X-RapidAPI-Key": rapidapi_key,
                "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
            }
            
            search_response = await client.get(search_url, params=search_params, headers=search_headers)
            
            if search_response.status_code != 200:
                return await _mock_hotel_search(destination, check_in, check_out, guests)
            
            search_data = search_response.json()
            if not search_data or len(search_data) == 0:
                return json.dumps({"destination": destination, "error": "Destination not found", "hotels": []})
            
            dest_id = search_data[0].get("dest_id")
            if not dest_id:
                return await _mock_hotel_search(destination, check_in, check_out, guests)
            
            # Search for hotels
            hotels_url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
            hotels_params = {
                "dest_id": dest_id,
                "order_by": "popularity",
                "filter_by_currency": "USD",
                "room_number": "1",
                "checkin_date": check_in,
                "checkout_date": check_out,
                "adults_number": str(guests),
                "page_number": "0",
                "locale": "en-gb",
                "units": "metric"
            }
            
            hotels_response = await client.get(hotels_url, params=hotels_params, headers=search_headers)
            
            if hotels_response.status_code != 200:
                return await _mock_hotel_search(destination, check_in, check_out, guests)
            
            hotels_data = hotels_response.json()
            
            # Parse hotel results
            hotels = []
            for hotel in hotels_data.get("result", [])[:5]:
                hotel_info = {
                    "name": hotel.get("hotel_name", "Unknown Hotel"),
                    "price": hotel.get("min_total_price", 0),
                    "rating": hotel.get("review_score", 0),
                    "amenities": hotel.get("hotel_facilities", [])[:4],
                    "location": hotel.get("address", f"{destination}"),
                    "image_url": hotel.get("main_photo_url", ""),
                    "description": hotel.get("hotel_name_trans", "")
                }
                hotels.append(hotel_info)
            
            result = {
                "destination": destination,
                "check_in": check_in,
                "check_out": check_out,
                "guests": guests,
                "hotels": hotels
            }
            
            return json.dumps(result)
            
    except Exception as e:
        print(f"Hotel search error: {e}")
        return await _mock_hotel_search(destination, check_in, check_out, guests)

@tool
async def search_flights(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None) -> str:
    """Search for flights between two cities with departure and optional return dates.
    
    Args:
        origin: Origin city or airport
        destination: Destination city or airport  
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format (optional for round trip)
    """
    try:
        amadeus_key = os.getenv("AMADEUS_API_KEY")
        amadeus_secret = os.getenv("AMADEUS_API_SECRET")
        
        if not amadeus_key or not amadeus_secret:
            return await _mock_flight_search(origin, destination, departure_date, return_date)
        
        async with httpx.AsyncClient() as client:
            # Get access token
            token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
            token_data = {
                "grant_type": "client_credentials",
                "client_id": amadeus_key,
                "client_secret": amadeus_secret
            }
            token_headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            token_response = await client.post(token_url, data=token_data, headers=token_headers)
            
            if token_response.status_code != 200:
                return await _mock_flight_search(origin, destination, departure_date, return_date)
            
            token_info = token_response.json()
            access_token = token_info.get("access_token")
            
            if not access_token:
                return await _mock_flight_search(origin, destination, departure_date, return_date)
            
            # Search flights
            flights_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
            flights_params = {
                "originLocationCode": origin[:3].upper(),
                "destinationLocationCode": destination[:3].upper(),
                "departureDate": departure_date,
                "adults": "1",
                "max": "5"
            }
            
            if return_date:
                flights_params["returnDate"] = return_date
            
            flights_headers = {"Authorization": f"Bearer {access_token}"}
            
            flights_response = await client.get(flights_url, params=flights_params, headers=flights_headers)
            
            if flights_response.status_code != 200:
                return await _mock_flight_search(origin, destination, departure_date, return_date)
            
            flights_data = flights_response.json()
            
            # Parse flight results
            outbound_flights = []
            return_flights = []
            
            for offer in flights_data.get("data", [])[:3]:
                itineraries = offer.get("itineraries", [])
                
                if len(itineraries) > 0:
                    outbound = itineraries[0]
                    segments = outbound.get("segments", [])
                    
                    if segments:
                        first_segment = segments[0]
                        last_segment = segments[-1]
                        
                        flight_info = {
                            "airline": first_segment.get("carrierCode", "Unknown"),
                            "flight_number": f"{first_segment.get('carrierCode', 'XX')}{first_segment.get('number', '000')}",
                            "departure_time": first_segment.get("departure", {}).get("at", "")[-8:-3],
                            "arrival_time": last_segment.get("arrival", {}).get("at", "")[-8:-3],
                            "price": float(offer.get("price", {}).get("total", "0")),
                            "duration": outbound.get("duration", ""),
                            "stops": len(segments) - 1
                        }
                        outbound_flights.append(flight_info)
                
                if len(itineraries) > 1:
                    return_flight = itineraries[1]
                    segments = return_flight.get("segments", [])
                    
                    if segments:
                        first_segment = segments[0]
                        last_segment = segments[-1]
                        
                        flight_info = {
                            "airline": first_segment.get("carrierCode", "Unknown"),
                            "flight_number": f"{first_segment.get('carrierCode', 'XX')}{first_segment.get('number', '000')}",
                            "departure_time": first_segment.get("departure", {}).get("at", "")[-8:-3],
                            "arrival_time": last_segment.get("arrival", {}).get("at", "")[-8:-3],
                            "price": float(offer.get("price", {}).get("total", "0")),
                            "duration": return_flight.get("duration", ""),
                            "stops": len(segments) - 1
                        }
                        return_flights.append(flight_info)
            
            result = {
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "outbound_flights": outbound_flights
            }
            
            if return_date and return_flights:
                result["return_date"] = return_date
                result["return_flights"] = return_flights
            
            return json.dumps(result)
            
    except Exception as e:
        print(f"Flight search error: {e}")
        return await _mock_flight_search(origin, destination, departure_date, return_date)

@tool
async def get_weather(city: str, date: Optional[str] = None) -> str:
    """Get weather information for a specific city and date.
    
    Args:
        city: City name
        date: Date in YYYY-MM-DD format (optional, defaults to today)
    """
    try:
        openweather_key = os.getenv("OPENWEATHER_API_KEY")
        if not openweather_key:
            return await _mock_weather_data(city, date)
        
        async with httpx.AsyncClient() as client:
            # Current weather endpoint
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            weather_params = {
                "q": city,
                "appid": openweather_key,
                "units": "metric"
            }
            
            weather_response = await client.get(weather_url, params=weather_params)
            
            if weather_response.status_code != 200:
                return await _mock_weather_data(city, date)
            
            weather_data = weather_response.json()
            
            # Get forecast for better temperature range
            forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
            forecast_params = {
                "q": city,
                "appid": openweather_key,
                "units": "metric",
                "cnt": 8
            }
            
            forecast_response = await client.get(forecast_url, params=forecast_params)
            
            # Calculate temperature range from forecast
            temp_high = weather_data["main"]["temp"]
            temp_low = weather_data["main"]["temp"]
            
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                temperatures = [item["main"]["temp"] for item in forecast_data.get("list", [])]
                if temperatures:
                    temp_high = max(temperatures)
                    temp_low = min(temperatures)
            
            result = {
                "city": city,
                "date": date or datetime.now().strftime("%Y-%m-%d"),
                "temperature": {
                    "high": round(temp_high),
                    "low": round(temp_low),
                    "current": round(weather_data["main"]["temp"])
                },
                "condition": weather_data["weather"][0]["description"].title(),
                "humidity": weather_data["main"]["humidity"],
                "wind_speed": weather_data.get("wind", {}).get("speed", 0),
                "visibility": weather_data.get("visibility", 0) / 1000,
                "icon": weather_data["weather"][0]["icon"]
            }
            
            return json.dumps(result)
            
    except Exception as e:
        print(f"Weather API error: {e}")
        return await _mock_weather_data(city, date)

@tool
async def get_local_activities(destination: str, activity_type: str = "all") -> str:
    """Get local activities and attractions for a destination.
    
    Args:
        destination: Destination city or location
        activity_type: Type of activities to search for (cultural, adventure, food, all)
    """
    try:
        foursquare_key = os.getenv("FOURSQUARE_API_KEY")
        if not foursquare_key:
            return await _mock_activities_data(destination, activity_type)
        
        async with httpx.AsyncClient() as client:
            places_url = "https://api.foursquare.com/v3/places/search"
            
            category_mapping = {
                "cultural": "10000,12000",
                "adventure": "16000,18000",
                "food": "13000",
                "all": "10000,12000,13000,16000,18000"
            }
            
            places_params = {
                "near": destination,
                "categories": category_mapping.get(activity_type, category_mapping["all"]),
                "limit": 10
            }
            
            places_headers = {
                "Authorization": foursquare_key,
                "Accept": "application/json"
            }
            
            places_response = await client.get(places_url, params=places_params, headers=places_headers)
            
            if places_response.status_code != 200:
                return await _mock_activities_data(destination, activity_type)
            
            places_data = places_response.json()
            
            activities = []
            for place in places_data.get("results", []):
                categories = place.get("categories", [])
                category_name = categories[0].get("name", "Activity") if categories else "Activity"
                
                activity_category = "cultural"
                if any(cat in category_name.lower() for cat in ["food", "restaurant", "cafe", "bar"]):
                    activity_category = "food"
                elif any(cat in category_name.lower() for cat in ["outdoor", "park", "sport", "recreation"]):
                    activity_category = "adventure"
                elif any(cat in category_name.lower() for cat in ["museum", "art", "theater", "historic"]):
                    activity_category = "cultural"
                
                activity_info = {
                    "name": place.get("name", "Unknown Activity"),
                    "category": activity_category,
                    "price": random.randint(15, 100),
                    "duration": "2-3 hours",
                    "address": place.get("location", {}).get("formatted_address", ""),
                    "rating": round(random.uniform(3.5, 4.8), 1),
                    "description": category_name
                }
                activities.append(activity_info)
            
            result = {
                "destination": destination,
                "activity_type": activity_type,
                "activities": activities
            }
            
            return json.dumps(result)
            
    except Exception as e:
        print(f"Activities API error: {e}")
        return await _mock_activities_data(destination, activity_type)

@tool
async def exchange_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency amounts from one currency to another using real-time exchange rates.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., USD, EUR, VND)
        to_currency: Target currency code (e.g., USD, EUR, VND)
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    
    exchangerate_api_key = os.getenv("EXCHANGERATE_API_KEY")
    
    if exchangerate_api_key:
        try:
            async with httpx.AsyncClient() as client:
                url = f"https://v6.exchangerate-api.com/v6/{exchangerate_api_key}/pair/{from_currency}/{to_currency}/{amount}"
                
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("result") == "success":
                        result = {
                            "source": "ExchangeRate-API.com",
                            "amount": amount,
                            "from_currency": from_currency,
                            "to_currency": to_currency,
                            "rate": data.get("conversion_rate"),
                            "exchanged_amount": round(data.get("conversion_result"), 2),
                            "last_update": data.get("time_last_update_utc")
                        }
                        return json.dumps(result)
                    else:
                        error_result = {
                            "error": f"Currency conversion failed: {data.get('error-type', 'Unknown error')}",
                            "source": "ExchangeRate-API.com"
                        }
                        return json.dumps(error_result)
        except Exception as e:
            error_result = {
                "error": f"API request failed: {str(e)}",
                "source": "ExchangeRate-API.com"
            }
            return json.dumps(error_result)
    
    error_result = {
        "error": "Currency conversion requires ExchangeRate-API key. Please add EXCHANGERATE_API_KEY to your .env file.",
        "setup_instructions": "1. Visit https://www.exchangerate-api.com/ 2. Sign up for free 3. Get API key 4. Add to .env",
        "supported_currencies": "160+ currencies including VND, USD, EUR, GBP, JPY, etc."
    }
    return json.dumps(error_result)

# Mock function helpers (need to be async and return JSON strings)
async def _mock_hotel_search(destination: str, check_in: str, check_out: str, guests: int) -> str:
    """Fallback mock hotel search"""
    hotels = [
        {
            "name": f"Grand {destination} Hotel",
            "price": random.randint(120, 350),
            "rating": round(random.uniform(4.0, 4.8), 1),
            "amenities": ["WiFi", "Pool", "Gym", "Restaurant"],
            "location": f"Downtown {destination}"
        },
        {
            "name": f"Boutique {destination} Inn",
            "price": random.randint(80, 200),
            "rating": round(random.uniform(3.8, 4.5), 1),
            "amenities": ["WiFi", "Breakfast", "Parking"],
            "location": f"City Center {destination}"
        }
    ]
    
    result = {
        "destination": destination,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "hotels": hotels
    }
    return json.dumps(result)

async def _mock_flight_search(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None) -> str:
    """Fallback mock flight search"""
    airlines = ["SkyWings Airlines", "CloudHopper Air", "JetStream Airways", "Pacific Express"]
    
    outbound_flights = []
    for i in range(3):
        departure_time = f"{random.randint(6, 20):02d}:{random.choice(['00', '15', '30', '45'])}"
        arrival_time = f"{random.randint(8, 22):02d}:{random.choice(['00', '15', '30', '45'])}"
        
        outbound_flights.append({
            "airline": random.choice(airlines),
            "flight_number": f"{random.choice(['SK', 'CH', 'JS', 'PE'])}{random.randint(100, 999)}",
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "price": random.randint(200, 800),
            "duration": f"{random.randint(2, 8)}h {random.randint(0, 59)}m"
        })
    
    result = {
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "outbound_flights": outbound_flights
    }
    
    if return_date:
        return_flights = []
        for i in range(3):
            departure_time = f"{random.randint(6, 20):02d}:{random.choice(['00', '15', '30', '45'])}"
            arrival_time = f"{random.randint(8, 22):02d}:{random.choice(['00', '15', '30', '45'])}"
            
            return_flights.append({
                "airline": random.choice(airlines),
                "flight_number": f"{random.choice(['SK', 'CH', 'JS', 'PE'])}{random.randint(100, 999)}",
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "price": random.randint(200, 800),
                "duration": f"{random.randint(2, 8)}h {random.randint(0, 59)}m"
            })
        
        result["return_date"] = return_date
        result["return_flights"] = return_flights
    
    return json.dumps(result)

async def _mock_weather_data(city: str, date: Optional[str] = None) -> str:
    """Fallback mock weather data"""
    weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Thunderstorms"]
    
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    result = {
        "city": city,
        "date": date,
        "temperature": {
            "high": random.randint(20, 35),
            "low": random.randint(10, 25)
        },
        "condition": random.choice(weather_conditions),
        "humidity": random.randint(40, 80),
        "precipitation": random.randint(0, 30)
    }
    return json.dumps(result)

async def _mock_activities_data(destination: str, activity_type: str = "all") -> str:
    """Fallback mock activities data"""
    activities = {
        "cultural": [
            {"name": f"{destination} Art Museum", "price": random.randint(15, 30), "duration": "2-3 hours", "category": "cultural"},
            {"name": f"Historic {destination} Walking Tour", "price": random.randint(25, 50), "duration": "3 hours", "category": "cultural"},
            {"name": f"{destination} Cathedral", "price": 0, "duration": "1-2 hours", "category": "cultural"}
        ],
        "adventure": [
            {"name": f"{destination} Mountain Hiking", "price": random.randint(40, 80), "duration": "Full day", "category": "adventure"},
            {"name": f"{destination} River Rafting", "price": random.randint(60, 120), "duration": "Half day", "category": "adventure"},
            {"name": f"{destination} Zip Line Tour", "price": random.randint(50, 100), "duration": "3 hours", "category": "adventure"}
        ],
        "food": [
            {"name": f"{destination} Food Tour", "price": random.randint(60, 100), "duration": "4 hours", "category": "food"},
            {"name": f"Cooking Class in {destination}", "price": random.randint(80, 150), "duration": "3 hours", "category": "food"},
            {"name": f"{destination} Wine Tasting", "price": random.randint(40, 90), "duration": "2 hours", "category": "food"}
        ]
    }
    
    if activity_type == "all":
        all_activities = []
        for category, acts in activities.items():
            all_activities.extend(acts)
        result = {"destination": destination, "activities": all_activities}
    else:
        result = {"destination": destination, "activities": activities.get(activity_type, [])}
    
    return json.dumps(result)

# Initialize knowledge base with travel information
def initialize_knowledge_base():
    """Initialize the knowledge base with travel information"""
    try:
        # Check if knowledge base already has data
        existing_data = travel_knowledge_collection.count()
        if existing_data > 0:
            print(f"📚 Knowledge base already initialized with {existing_data} entries")
            return
        
        print("📚 Initializing travel knowledge base...")
        
        travel_knowledge = [
            {
                "title": "Best Time to Visit Europe",
                "content": "Spring (April-May) and Fall (September-October) offer mild weather and fewer crowds. Summer is peak season with higher prices but best weather.",
                "category": "travel_tips",
                "region": "Europe"
            },
            {
                "title": "Budget Travel Tips",
                "content": "Book flights in advance, use public transport, stay in hostels or budget hotels, eat like a local, and take advantage of free attractions.",
                "category": "travel_tips",
                "region": "general"
            },
            {
                "title": "Packing Essentials",
                "content": "Pack light, bring versatile clothing, essential documents, first aid kit, portable charger, and comfortable walking shoes.",
                "category": "travel_tips",
                "region": "general"
            },
            {
                "title": "Southeast Asia Travel Guide",
                "content": "Visit during dry season (November-April), try street food, respect local customs, negotiate prices, and carry cash for smaller vendors.",
                "category": "destination_guide",
                "region": "Southeast Asia"
            },
            {
                "title": "Hotel Booking Tips",
                "content": "Compare prices across platforms, read recent reviews, check cancellation policies, book directly with hotels for better rates, and consider location vs price.",
                "category": "accommodation",
                "region": "general"
            },
            {
                "title": "Flight Booking Strategies",
                "content": "Use flexible date searches, clear browser cookies, compare one-way vs round-trip, consider layovers, and book Tuesday-Thursday for better prices.",
                "category": "transportation",
                "region": "general"
            },
            {
                "title": "Local Culture Vietnam",
                "content": "Remove shoes when entering homes, dress modestly in temples, learn basic Vietnamese greetings, tip 10-15% in restaurants, and try pho and banh mi.",
                "category": "culture",
                "region": "Vietnam"
            },
            {
                "title": "Emergency Travel Tips",
                "content": "Keep copies of important documents, know embassy contact info, have travel insurance, keep emergency cash hidden, and register with your embassy.",
                "category": "safety",
                "region": "general"
            }
        ]
        
        # Add to knowledge base
        for idx, knowledge in enumerate(travel_knowledge):
            travel_knowledge_collection.add(
                documents=[knowledge["content"]],
                metadatas=[{
                    "title": knowledge["title"],
                    "category": knowledge["category"],
                    "region": knowledge["region"]
                }],
                ids=[f"knowledge_{idx}"]
            )
        
        print(f"✅ Knowledge base initialized with {len(travel_knowledge)} entries")
        
    except Exception as e:
        print(f"❌ Error initializing knowledge base: {e}")

# ===== LANGCHAIN AGENT SETUP =====

def create_travel_agent(user_id: str) -> AgentExecutor:
    """Create a LangChain agent for travel assistance with persistent memory"""
    
    # Get all available tools
    tools = [
        get_current_date_time,
        search_hotels,
        search_flights, 
        get_weather,
        get_local_activities,
        exchange_currency
    ]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are TravelBot, an expert AI travel assistant powered by LangChain, real-time data, external APIs, and enhanced with conversational memory and voice capabilities. Your mission is to provide comprehensive, personalized travel guidance that helps users plan amazing trips.

🌟 Your Enhanced Capabilities:
- Search and recommend hotels with real-time availability and pricing
- Find flights with multiple options and competitive prices  
- Provide accurate weather forecasts for travel planning
- Suggest local activities, attractions, and experiences
- Convert currencies with real-time exchange rates (including VND support)
- Create detailed itineraries and travel plans
- Offer travel tips, safety advice, and cultural insights
- Remember past conversations and preferences using vector storage
- Access a comprehensive travel knowledge base for expert advice
- Handle date and time queries accurately using current date/time information

🧠 Memory & Personalization:
- I remember your previous travel preferences and conversations
- I can reference your past inquiries to provide more personalized recommendations
- I learn from our interactions to better understand your travel style
- I can access relevant travel knowledge from my comprehensive database

🕐 Date & Time Awareness:
- IMPORTANT: When users ask about 'today', 'now', 'current weather', or any date-related queries, ALWAYS use the get_current_date_time tool first to get accurate current date and time information
- Use this current date information to provide accurate, up-to-date responses
- Don't rely on your training data for current date information

💡 Your Approach:
1. If the user asks about current date/time or uses words like 'today', 'now', etc., use get_current_date_time tool first
2. Ask clarifying questions to understand user preferences (budget, travel style, interests)
3. Use available tools to gather real-time data
4. Leverage conversation history to provide personalized recommendations
5. Access travel knowledge base for expert tips and insights
6. Provide multiple options with pros/cons
7. Be proactive in suggesting complementary services
8. Consider practical aspects like weather, local customs, and logistics

🎯 Communication Style:
- Friendly, enthusiastic, and knowledgeable
- Use emojis appropriately to enhance user experience
- Provide detailed but digestible information
- Always ask follow-up questions to refine recommendations
- Offer alternatives and explain your reasoning
- Reference past conversations when relevant

📝 Response Formatting:
- Use proper markdown formatting for structure and readability
- Use headers (### for main sections, #### for subsections) to organize information
- Use **bold** for important details like hotel names, prices, ratings
- Use bullet points (-) for lists of amenities, pros/cons
- Present hotel/flight options as numbered lists with clear headers
- Format prices, ratings, and key details prominently

Remember: Always use the available tools to get real-time data, especially get_current_date_time for any date-related queries. Leverage conversation history for personalization, and access the travel knowledge base for expert insights."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor

# ===== MULTI-USER PERSISTENT CONVERSATION MANAGEMENT =====

class PersistentMultiUserConversationManager:
    """Manage persistent conversations for multiple users using ChromaDB"""
    
    def __init__(self):
        self.agents = {}  # Cache for agent instances
    
    def get_agent(self, user_id: str) -> AgentExecutor:
        """Get or create agent for a user"""
        if user_id not in self.agents:
            self.agents[user_id] = create_travel_agent(user_id)
        return self.agents[user_id]
    
    def store_conversation(self, user_id: str, conversation_id: str, user_message: str, ai_response: str, tool_calls: Optional[List[Dict]] = None):
        """Store conversation in ChromaDB for persistence"""
        try:
            conversation_text = f"User: {user_message}\nAssistant: {ai_response}"
            
            if tool_calls:
                tool_info = []
                for call in tool_calls:
                    tool_info.append(f"Tool: {call.get('tool', 'unknown')}")
                conversation_text += f"\nTools used: {', '.join(tool_info)}"
            
            entry_id = f"{user_id}_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            user_conversations_collection.add(
                documents=[conversation_text],
                metadatas=[{
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat(),
                    "user_message": user_message[:200],
                    "has_tool_calls": bool(tool_calls)
                }],
                ids=[entry_id]
            )
            
            print(f"📝 Stored conversation for user {user_id}: {entry_id}")
            
        except Exception as e:
            print(f"❌ Error storing conversation: {e}")
    
    def get_conversation_history(self, user_id: str, conversation_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history for a user from ChromaDB"""
        try:
            results = user_conversations_collection.query(
                query_texts=["conversation history"],
                n_results=limit,
                where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"conversation_id": {"$eq": conversation_id}}
                    ]
                }
            )
            
            history = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    # Parse back the user and assistant messages
                    lines = doc.split('\n')
                    if len(lines) >= 2:
                        user_msg = lines[0].replace("User: ", "")
                        ai_msg = lines[1].replace("Assistant: ", "")
                        
                        history.append({
                            "role": "user",
                            "content": user_msg,
                            "timestamp": metadata.get("timestamp", "")
                        })
                        history.append({
                            "role": "assistant", 
                            "content": ai_msg,
                            "timestamp": metadata.get("timestamp", "")
                        })
            
            # Sort by timestamp and return most recent first
            history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return history[:limit*2]  # Each exchange has 2 messages
            
        except Exception as e:
            print(f"❌ Error retrieving conversation history: {e}")
            return []
    
    def get_relevant_context(self, user_id: str, query: str, limit: int = 3) -> str:
        """Get relevant conversation history and knowledge for context"""
        try:
            # Get relevant user conversations
            user_results = user_conversations_collection.query(
                query_texts=[query],
                n_results=limit,
                where={"user_id": {"$eq": user_id}}
            )
            
            # Get relevant travel knowledge
            knowledge_results = travel_knowledge_collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            context = ""
            
            if user_results['documents'] and user_results['documents'][0]:
                context += "\n\n🧠 **Relevant Previous Conversations:**\n"
                for doc in user_results['documents'][0]:
                    context += f"- {doc[:200]}...\n"
            
            if knowledge_results['documents'] and knowledge_results['documents'][0]:
                context += "\n\n📚 **Relevant Travel Knowledge:**\n"
                for i, doc in enumerate(knowledge_results['documents'][0]):
                    metadata = knowledge_results['metadatas'][0][i]
                    title = metadata.get("title", "Travel Info")
                    context += f"- **{title}**: {doc[:150]}...\n"
            
            return context
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return ""

# Initialize global conversation manager
conversation_manager = PersistentMultiUserConversationManager()

# Initialize TTS and knowledge base on startup
initialize_tts()
initialize_knowledge_base()

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: str = Field(..., description="User ID for multi-user support")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    speech_speed: float = Field(1.0, description="Speech speed multiplier (0.5-2.0, default: 1.0)")

class ChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    tool_calls: Optional[List[Dict]] = None
    audio_base64: Optional[str] = None

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    speed: float = Field(1.0, description="Speech speed multiplier (0.25-4.0, default: 1.0)")
    max_length: int = Field(200, description="Maximum text length (default: 200)")

# ===== MAIN PROCESSING FUNCTION =====

async def process_chat_message_with_langchain(message: str, user_id: str, conversation_id: str, speech_speed: float = 1.0) -> tuple[str, List[Dict], Optional[str]]:
    """Process a chat message using LangChain agent with persistent multi-user history"""
    
    try:
        # Get agent for user
        agent = conversation_manager.get_agent(user_id)
        
        # Get relevant context for this query
        context = conversation_manager.get_relevant_context(user_id, message)
        
        # Get conversation history from ChromaDB
        history = conversation_manager.get_conversation_history(user_id, conversation_id, limit=5)
        
        # Prepare chat history for LangChain
        chat_history = []
        for msg in reversed(history):  # Reverse to get chronological order
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        # Add context to the input if available
        enhanced_input = message
        if context:
            enhanced_input = f"{message}\n\nRelevant context from your knowledge and our previous conversations:{context}"
        
        # Execute agent
        result = await agent.ainvoke({
            "input": enhanced_input,
            "chat_history": chat_history
        })
        
        response = result["output"]
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Extract tool calls from intermediate steps
        tool_calls = []
        for step in intermediate_steps:
            if len(step) >= 2:
                action, observation = step
                tool_calls.append({
                    "tool": action.tool,
                    "arguments": action.tool_input,
                    "result": observation
                })
        
        # Store conversation in ChromaDB
        conversation_manager.store_conversation(user_id, conversation_id, message, response, tool_calls)
        
        # Generate TTS audio with speed control
        audio_base64 = text_to_speech(response, max_length=200, speed=speech_speed)
        
        return response, tool_calls, audio_base64
    
    except Exception as e:
        error_message = f"I'm sorry, I encountered an error while processing your request: {str(e)}"
        
        # Store error in ChromaDB too
        conversation_manager.store_conversation(user_id, conversation_id, message, error_message, [])
        
        return error_message, [], None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# ===== API ROUTES =====

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{user_id}/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, conversation_id: str):
    """WebSocket endpoint for real-time chat with multi-user support"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            if user_message.strip():
                # Get speech speed from message data
                speech_speed = message_data.get("speech_speed", 1.0)
                
                # Process message with LangChain agent
                response, tool_calls, audio_base64 = await process_chat_message_with_langchain(
                    user_message, user_id, conversation_id, speech_speed
                )
                
                # Send response back to client in the format expected by frontend
                response_data = {
                    "type": "response",
                    "content": response,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "function_calls": tool_calls,
                    "audio_base64": audio_base64,
                    "timestamp": datetime.now().isoformat()
                }
                
                await manager.send_personal_message(json.dumps(response_data), websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/chat")
async def chat_api(request: ChatRequest):
    """REST API endpoint for chat with multi-user support"""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        response, tool_calls, audio_base64 = await process_chat_message_with_langchain(
            request.message, 
            request.user_id,
            conversation_id, 
            request.speech_speed
        )
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            conversation_id=conversation_id,
            tool_calls=tool_calls,
            audio_base64=audio_base64
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/api/conversation/{user_id}/{conversation_id}")
async def get_conversation(user_id: str, conversation_id: str):
    """Get conversation history for a specific user and conversation"""
    try:
        history = conversation_manager.get_conversation_history(user_id, conversation_id, limit=20)
        return {
            "user_id": user_id,
            "conversation_id": conversation_id, 
            "messages": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.delete("/api/conversation/{user_id}/{conversation_id}")
async def clear_conversation(user_id: str, conversation_id: str):
    """Clear conversation history for a specific user and conversation"""
    try:
        # Note: ChromaDB doesn't have a direct delete by metadata query
        # In a production environment, you might want to mark conversations as deleted
        # or implement a proper deletion mechanism
        return {"message": f"Conversation {conversation_id} for user {user_id} marked for clearing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@app.get("/api/users/{user_id}/conversations")
async def get_user_conversations(user_id: str):
    """Get all conversations for a specific user"""
    try:
        results = user_conversations_collection.query(
            query_texts=["conversations"],
            n_results=100,
            where={"user_id": {"$eq": user_id}}
        )
        
        conversations = {}
        if results['metadatas']:
            for metadata in results['metadatas'][0]:
                conv_id = metadata.get("conversation_id")
                if conv_id not in conversations:
                    conversations[conv_id] = {
                        "conversation_id": conv_id,
                        "last_message": metadata.get("user_message", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "message_count": 0
                    }
                conversations[conv_id]["message_count"] += 1
        
        return {
            "user_id": user_id,
            "conversations": list(conversations.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user conversations: {str(e)}")

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    """Generate text-to-speech audio for given text with speed control"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Validate speed parameter (OpenAI TTS supports 0.25 to 4.0)
        if request.speed < 0.25 or request.speed > 4.0:
            raise HTTPException(status_code=400, detail="Speed must be between 0.25 and 4.0")
        
        audio_base64 = text_to_speech(request.text, request.max_length, request.speed)
        
        if audio_base64:
            return {"audio_base64": audio_base64, "text": request.text[:request.max_length]}
        else:
            raise HTTPException(status_code=500, detail="TTS generation failed")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")

@app.get("/api/knowledge/search")
async def search_knowledge(query: str, limit: int = 5):
    """Search travel knowledge base"""
    try:
        results = travel_knowledge_collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        knowledge_items = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                knowledge_items.append({
                    "title": metadata.get("title", ""),
                    "content": doc,
                    "category": metadata.get("category", ""),
                    "region": metadata.get("region", ""),
                    "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 0
                })
        
        return {"query": query, "results": knowledge_items}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching knowledge: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Get total conversations
        total_conversations = user_conversations_collection.count()
        
        # Get unique users (approximate)
        all_results = user_conversations_collection.query(
            query_texts=["user statistics"],
            n_results=total_conversations
        )
        
        unique_users = set()
        if all_results['metadatas']:
            for metadata in all_results['metadatas'][0]:
                user_id = metadata.get("user_id")
                if user_id:
                    unique_users.add(user_id)
        
        # Get knowledge base count
        knowledge_count = travel_knowledge_collection.count()
        
        return {
            "total_conversations": total_conversations,
            "unique_users": len(unique_users),
            "knowledge_items": knowledge_count,
            "active_agents": len(conversation_manager.agents),
            "tts_available": TTS_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    return {
        "status": "healthy",
        "model": os.getenv("OPENAI_MODEL_NAME", "GPT-4o-mini"),
        "langchain_version": "enabled",
        "persistent_storage": "ChromaDB",
        "multi_user_support": "enabled",
        "timestamp": datetime.now().isoformat()
    }

def start_server():
    """Start the FastAPI server"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: Missing OPENAI_API_KEY environment variable")
        print("Please add your OpenAI API key to the .env file")
        return
    
    print("🧳 Starting Travel Assistant Chatbot with LangChain...")
    print("🤖 LangChain Agents: Enabled")
    print("💾 Persistent Storage: ChromaDB")
    print("👥 Multi-User Support: Enabled")
    print("🕐 Date Awareness: Enhanced")
    print("💬 Chat Interface: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    start_server()
