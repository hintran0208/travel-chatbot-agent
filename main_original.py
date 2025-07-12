"""
Travel Assistant Chatbot using OpenAI SDK

Advanced travel chatbot with function calling, external API integrations,
and real-time conversation management for comprehensive travel assistance.
"""

# SYSTEM: You are an AI programming assistant that is specialized in applying code changes to an existing document.
# Follow Microsoft content policies.
# Avoid content that violates copyrights.
# If you are asked to generate content that is harmful, hateful, racist, sexist, lewd, violent, or completely irrelevant to software engineering, only respond with "Sorry, I can't assist with that."
# Keep your answers short and impersonal.
# The user has a code block that represents a suggestion for a code change and a python file opened in a code editor.
# Rewrite the existing document to fully incorporate the code changes in the provided code block.
# For the response, always follow these instructions:
# 1. Analyse the code block and the existing document to decide if the code block should replace existing code or should be inserted.
# 2. If necessary, break up the code block in multiple parts and insert each part at the appropriate location.
# 3. Preserve whitespace and newlines right after the parts of the file that you modify.
# 4. The final result must be syntactically valid, properly formatted, and correctly indented. It should not contain any ...existing code... comments.
# 5. Finally, provide the fully rewritten file. You must output the complete file.

import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
import uvicorn
import httpx
import random
import pandas as pd
import openpyxl
from io import BytesIO, StringIO
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import torch
from transformers import VitsModel, AutoTokenizer
import numpy as np
import base64
import soundfile as sf
import tempfile
import librosa
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# FastAPI app initialization
app = FastAPI(
    title="Travel Assistant Chatbot",
    description="AI-powered travel assistant with real-time chat, hotel booking, flight search, and weather information",
    version="2.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")

# Create templates and static directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# OpenAI Client Setup
client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/jpe"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

model_name = os.getenv("OPENAI_MODEL_NAME", "GPT-4o-mini")

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
        name="user_conversations",
        embedding_function=openai_ef,
        metadata={"description": "User conversation history and preferences"}
    )
except Exception:
    user_conversations_collection = chroma_client.get_collection(
        name="user_conversations",
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

# Text-to-Speech Setup
tts_model = None
tts_tokenizer = None

def initialize_tts():
    """Initialize TTS model (lazy loading)"""
    global tts_model, tts_tokenizer
    try:
        if tts_model is None:
            print("🔊 Initializing Text-to-Speech model...")
            tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
            tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            print("✅ TTS model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing TTS: {e}")
        print("TTS functionality will be disabled")

def text_to_speech(text: str, max_length: int = 200, speed: float = 1.0) -> Optional[str]:
    """Convert text to speech and return base64 encoded audio with speed control"""
    try:
        if tts_model is None or tts_tokenizer is None:
            print("TTS model not initialized")
            return None
        
        # Truncate text if too long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # Tokenize and generate speech
        inputs = tts_tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = tts_model(**inputs).waveform
        
        # Convert to numpy array and normalize
        waveform = output.squeeze().cpu().numpy()
        
        # Apply speed adjustment if needed
        if speed != 1.0:
            # Clamp speed to reasonable range
            speed = max(0.5, min(2.0, speed))
            
            # Use librosa for time stretching (speed adjustment)
            waveform = librosa.effects.time_stretch(waveform, rate=speed)
        
        # Create temporary file to save audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, waveform, 16000)  # 16kHz sample rate
            
            # Read the file and encode as base64
            with open(tmp_file.name, "rb") as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return audio_base64
    
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

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
        
        # Sample travel knowledge base
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

# Initialize TTS and knowledge base on startup
initialize_tts()
initialize_knowledge_base()

# Global conversation storage (in production, use Redis or database)
conversations: Dict[str, List[Dict]] = {}

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    speech_speed: float = Field(1.0, description="Speech speed multiplier (0.5-2.0, default: 1.0)")

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    function_calls: Optional[List[Dict]] = None
    audio_base64: Optional[str] = None

class ExportRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of messages to export")
    format: str = Field(..., description="Export format: 'excel' or 'txt'")
    filename: Optional[str] = Field(None, description="Custom filename")

class ExportMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    speed: float = Field(1.0, description="Speech speed multiplier (0.5-2.0, default: 1.0)")
    max_length: int = Field(200, description="Maximum text length (default: 200)")

"""
Travel Assistant Chatbot using OpenAI SDK

Advanced travel chatbot with function calling, external API integrations,
and real-time conversation management for comprehensive travel assistance.
"""

# Real API functions using external services
async def search_hotels(destination: str, check_in: str, check_out: str, guests: int = 2) -> Dict[str, Any]:
    """Search hotels using Booking.com RapidAPI"""
    try:
        rapidapi_key = os.getenv("RAPIDAPI_KEY")
        if not rapidapi_key:
            return {"error": "RapidAPI key not configured", "hotels": []}
        
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
                # Fallback to mock data if API fails
                return await _mock_hotel_search(destination, check_in, check_out, guests)
            
            search_data = search_response.json()
            if not search_data or len(search_data) == 0:
                return {"destination": destination, "error": "Destination not found", "hotels": []}
            
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
            for hotel in hotels_data.get("result", [])[:5]:  # Limit to 5 results
                hotel_info = {
                    "name": hotel.get("hotel_name", "Unknown Hotel"),
                    "price": hotel.get("min_total_price", 0),
                    "rating": hotel.get("review_score", 0),
                    "amenities": hotel.get("hotel_facilities", [])[:4],  # Limit amenities
                    "location": hotel.get("address", f"{destination}"),
                    "image_url": hotel.get("main_photo_url", ""),
                    "description": hotel.get("hotel_name_trans", "")
                }
                hotels.append(hotel_info)
            
            return {
                "destination": destination,
                "check_in": check_in,
                "check_out": check_out,
                "guests": guests,
                "hotels": hotels
            }
            
    except Exception as e:
        print(f"Hotel search error: {e}")
        return await _mock_hotel_search(destination, check_in, check_out, guests)

async def _mock_hotel_search(destination: str, check_in: str, check_out: str, guests: int) -> Dict[str, Any]:
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
    
    return {
        "destination": destination,
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "hotels": hotels
    }

async def search_flights(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None) -> Dict[str, Any]:
    """Search flights using Amadeus API"""
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
                "originLocationCode": origin[:3].upper(),  # Convert to IATA code
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
            
            for offer in flights_data.get("data", [])[:3]:  # Limit to 3 results
                itineraries = offer.get("itineraries", [])
                
                if len(itineraries) > 0:
                    # Outbound flight
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
                    # Return flight
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
            
            return result
            
    except Exception as e:
        print(f"Flight search error: {e}")
        return await _mock_flight_search(origin, destination, departure_date, return_date)

async def _mock_flight_search(origin: str, destination: str, departure_date: str, return_date: Optional[str] = None) -> Dict[str, Any]:
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
    
    return result

async def get_weather(city: str, date: Optional[str] = None) -> Dict[str, Any]:
    """Get weather using OpenWeatherMap API"""
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
                "cnt": 8  # Next 24 hours
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
                "visibility": weather_data.get("visibility", 0) / 1000,  # Convert to km
                "icon": weather_data["weather"][0]["icon"]
            }
            
            return result
            
    except Exception as e:
        print(f"Weather API error: {e}")
        return await _mock_weather_data(city, date)

async def _mock_weather_data(city: str, date: Optional[str] = None) -> Dict[str, Any]:
    """Fallback mock weather data"""
    weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Thunderstorms"]
    
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    return {
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

async def get_local_activities(destination: str, activity_type: str = "all") -> Dict[str, Any]:
    """Get local activities using Foursquare Places API"""
    try:
        foursquare_key = os.getenv("FOURSQUARE_API_KEY")
        if not foursquare_key:
            return await _mock_activities_data(destination, activity_type)
        
        async with httpx.AsyncClient() as client:
            # Search for places in the destination
            places_url = "https://api.foursquare.com/v3/places/search"
            
            # Map activity types to Foursquare categories
            category_mapping = {
                "cultural": "10000,12000",  # Arts & Entertainment, Museums
                "adventure": "16000,18000", # Outdoor & Recreation, Travel
                "food": "13000",            # Food & Beverage
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
            
            # Parse activities
            activities = []
            for place in places_data.get("results", []):
                categories = place.get("categories", [])
                category_name = categories[0].get("name", "Activity") if categories else "Activity"
                
                # Determine activity category
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
                    "price": random.randint(15, 100),  # Foursquare doesn't provide pricing
                    "duration": "2-3 hours",
                    "address": place.get("location", {}).get("formatted_address", ""),
                    "rating": round(random.uniform(3.5, 4.8), 1),
                    "description": category_name
                }
                activities.append(activity_info)
            
            return {
                "destination": destination,
                "activity_type": activity_type,
                "activities": activities
            }
            
    except Exception as e:
        print(f"Activities API error: {e}")
        return await _mock_activities_data(destination, activity_type)

async def _mock_activities_data(destination: str, activity_type: str = "all") -> Dict[str, Any]:
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
        return {"destination": destination, "activities": all_activities}
    else:
        return {"destination": destination, "activities": activities.get(activity_type, [])}

async def exchange_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Exchange currency using ExchangeRate-API (requires API key)"""
    
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    
    # Get ExchangeRate-API key (required)
    exchangerate_api_key = os.getenv("EXCHANGERATE_API_KEY")
    
    if exchangerate_api_key:
        try:
            async with httpx.AsyncClient() as client:
                # Use pair conversion endpoint for direct conversion
                url = f"https://v6.exchangerate-api.com/v6/{exchangerate_api_key}/pair/{from_currency}/{to_currency}/{amount}"
                
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("result") == "success":
                        return {
                            "source": "ExchangeRate-API.com",
                            "amount": amount,
                            "from_currency": from_currency,
                            "to_currency": to_currency,
                            "rate": data.get("conversion_rate"),
                            "exchanged_amount": round(data.get("conversion_result"), 2),
                            "last_update": data.get("time_last_update_utc")
                        }
                    else:
                        return {
                            "error": f"Currency conversion failed: {data.get('error-type', 'Unknown error')}",
                            "source": "ExchangeRate-API.com"
                        }
        except Exception as e:
            return {
                "error": f"API request failed: {str(e)}",
                "source": "ExchangeRate-API.com"
            }
    
    # No API key configured
    return {
        "error": "Currency conversion requires ExchangeRate-API key. Please add EXCHANGERATE_API_KEY to your .env file.",
        "setup_instructions": "1. Visit https://www.exchangerate-api.com/ 2. Sign up for free 3. Get API key 4. Add to .env",
        "supported_currencies": "160+ currencies including VND, USD, EUR, GBP, JPY, etc."
    }

# OpenAI function definitions for function calling
function_definitions = [
    {
        "name": "search_hotels",
        "description": "Search for hotels in a specific destination with check-in and check-out dates",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "The destination city or location"
                },
                "check_in": {
                    "type": "string", 
                    "description": "Check-in date in YYYY-MM-DD format"
                },
                "check_out": {
                    "type": "string",
                    "description": "Check-out date in YYYY-MM-DD format"
                },
                "guests": {
                    "type": "integer",
                    "description": "Number of guests (default: 2)"
                }
            },
            "required": ["destination", "check_in", "check_out"]
        }
    },
    {
        "name": "search_flights",
        "description": "Search for flights between two cities with departure and optional return dates",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Origin city or airport"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination city or airport"
                },
                "departure_date": {
                    "type": "string",
                    "description": "Departure date in YYYY-MM-DD format"
                },
                "return_date": {
                    "type": "string",
                    "description": "Return date in YYYY-MM-DD format (optional for round trip)"
                }
            },
            "required": ["origin", "destination", "departure_date"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get weather information for a specific city and date",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                },
                "date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format (optional, defaults to today)"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_local_activities",
        "description": "Get local activities and attractions for a destination",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Destination city or location"
                },
                "activity_type": {
                    "type": "string",
                    "enum": ["cultural", "adventure", "food", "all"],
                    "description": "Type of activities to search for"
                }
            },
            "required": ["destination"]
        }
    },
    {
        "name": "exchange_currency",
        "description": "Convert currency amounts from one currency to another using real-time exchange rates (supports VND and 160+ other currencies)",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Amount to convert"
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (e.g., USD, EUR, VND)"
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code (e.g., USD, EUR, VND)"
                }
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    }
]

# Function mapping
available_functions = {
    "search_hotels": search_hotels,
    "search_flights": search_flights,
    "get_weather": get_weather,
    "get_local_activities": get_local_activities,
    "exchange_currency": exchange_currency
}

def get_system_prompt() -> str:
    """Get the system prompt for the travel assistant"""
    return """You are TravelBot, an expert AI travel assistant powered by real-time data, external APIs, and enhanced with conversational memory and voice capabilities. Your mission is to provide comprehensive, personalized travel guidance that helps users plan amazing trips.

🌟 Your Enhanced Capabilities:
- Search and recommend hotels with real-time availability and pricing
- Find flights with multiple options and competitive prices
- Provide accurate weather forecasts for travel planning
- Suggest local activities, attractions, and experiences
- Convert currencies with real-time exchange rates (including VND support)
- Create detailed itineraries and travel plans
- Offer travel tips, safety advice, and cultural insights
- Remember past conversations and preferences using vector storage
- Provide audio responses through text-to-speech technology
- Access a comprehensive travel knowledge base for expert advice

🧠 Memory & Personalization:
- I remember your previous travel preferences and conversations
- I can reference your past inquiries to provide more personalized recommendations
- I learn from our interactions to better understand your travel style
- I can access relevant travel knowledge from my comprehensive database

🔊 Voice Capabilities:
- I can convert my responses to speech for hands-free interaction
- Audio responses are available for key recommendations and summaries
- Voice output helps with accessibility and multitasking while travel planning

💡 Your Approach:
1. Ask clarifying questions to understand user preferences (budget, travel style, interests)
2. Use available functions to gather real-time data
3. Leverage conversation history to provide personalized recommendations
4. Access travel knowledge base for expert tips and insights
5. Provide multiple options with pros/cons
6. Be proactive in suggesting complementary services
7. Consider practical aspects like weather, local customs, and logistics

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

Remember: Always use the available functions to get real-time data, leverage conversation history for personalization, and access the travel knowledge base for expert insights. When users ask about travel plans, proactively gather all relevant information they might need and offer audio summaries for key recommendations."""

async def process_chat_message(message: str, conversation_id: str, speech_speed: float = 1.0) -> tuple[str, List[Dict], Optional[str]]:
    """Process a chat message with function calling support, memory, and TTS"""
    
    # Get or create conversation history
    if conversation_id not in conversations:
        conversations[conversation_id] = [
            {"role": "system", "content": get_system_prompt()}
        ]
    
    # Get relevant conversation history from ChromaDB
    relevant_history = get_relevant_conversation_history(message, conversation_id, limit=2)
    
    # Get relevant travel knowledge from ChromaDB
    relevant_knowledge = get_relevant_travel_knowledge(message, limit=3)
    
    # Enhance system prompt with relevant context
    enhanced_context = ""
    
    if relevant_history:
        enhanced_context += "\n\n🧠 **Relevant Conversation History:**\n"
        for hist in relevant_history:
            enhanced_context += f"- {hist['content'][:200]}...\n"
    
    if relevant_knowledge:
        enhanced_context += "\n\n📚 **Relevant Travel Knowledge:**\n"
        for knowledge in relevant_knowledge:
            enhanced_context += f"- **{knowledge['title']}**: {knowledge['content'][:150]}...\n"
    
    # Add enhanced context to the conversation if available
    if enhanced_context:
        conversations[conversation_id].append({
            "role": "system",
            "content": f"Context from your knowledge base and previous conversations:{enhanced_context}"
        })
    
    # Add user message to conversation
    conversations[conversation_id].append({
        "role": "user", 
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    
    function_call_results = []
    
    try:
        # First API call with function calling enabled
        response = client.chat.completions.create(
            model=model_name,
            messages=conversations[conversation_id],
            functions=function_definitions,
            function_call="auto",
            temperature=0.7,
            max_tokens=1000
        )
        
        response_message = response.choices[0].message
        
        # Check if the model wants to call a function
        if response_message.function_call:
            # Extract function details
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)
            
            print(f"🔧 Calling function: {function_name} with args: {function_args}")
            
            # Call the function
            if function_name in available_functions:
                function_response = await available_functions[function_name](**function_args)
                function_call_results.append({
                    "function": function_name,
                    "arguments": function_args,
                    "result": function_response
                })
                
                # Add function call and response to conversation
                conversations[conversation_id].append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": function_name,
                        "arguments": json.dumps(function_args)
                    }
                })
                
                conversations[conversation_id].append({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
                
                # Get final response with function results
                final_response = client.chat.completions.create(
                    model=model_name,
                    messages=conversations[conversation_id],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                assistant_response = final_response.choices[0].message.content
            else:
                assistant_response = f"I tried to call function '{function_name}' but it's not available. Let me help you in another way."
        else:
            assistant_response = response_message.content
        
        # Add assistant response to conversation
        conversations[conversation_id].append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store conversation in ChromaDB for future reference
        store_user_conversation(conversation_id, message, assistant_response, function_call_results)
        
        # Generate TTS audio for the response with speed control
        audio_base64 = text_to_speech(assistant_response, speed=speech_speed)
        
        return assistant_response, function_call_results, audio_base64
    
    except Exception as e:
        error_message = f"I'm sorry, I encountered an error while processing your request: {str(e)}"
        conversations[conversation_id].append({
            "role": "assistant",
            "content": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store error in ChromaDB too
        store_user_conversation(conversation_id, message, error_message, [])
        
        return error_message, [], None

# Vector Database Functions
def store_user_conversation(conversation_id: str, user_message: str, assistant_response: str, function_calls: Optional[List[Dict]] = None):
    """Store user conversation in ChromaDB for future reference"""
    try:
        conversation_text = f"User: {user_message}\nAssistant: {assistant_response}"
        
        # Add function call context if available
        if function_calls:
            function_info = []
            for call in function_calls:
                function_info.append(f"Function: {call.get('function', 'unknown')}")
            conversation_text += f"\nFunctions used: {', '.join(function_info)}"
        
        # Create unique ID for this conversation entry
        entry_id = f"{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store in ChromaDB
        user_conversations_collection.add(
            documents=[conversation_text],
            metadatas=[{
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message[:200],  # Truncate for metadata
                "has_function_calls": bool(function_calls)
            }],
            ids=[entry_id]
        )
        
        print(f"📝 Stored conversation entry: {entry_id}")
        
    except Exception as e:
        print(f"❌ Error storing conversation: {e}")

def get_relevant_conversation_history(query: str, conversation_id: str, limit: int = 3) -> List[Dict]:
    """Retrieve relevant conversation history from ChromaDB"""
    try:
        # Search for similar conversations
        results = user_conversations_collection.query(
            query_texts=[query],
            n_results=limit,
            where={"conversation_id": {"$eq": conversation_id}}
        )
        
        relevant_history = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                relevant_history.append({
                    "content": doc,
                    "timestamp": metadata.get("timestamp", ""),
                    "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 0
                })
        
        return relevant_history
        
    except Exception as e:
        print(f"❌ Error retrieving conversation history: {e}")
        return []

def get_relevant_travel_knowledge(query: str, limit: int = 3) -> List[Dict]:
    """Retrieve relevant travel knowledge from ChromaDB"""
    try:
        results = travel_knowledge_collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        relevant_knowledge = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                relevant_knowledge.append({
                    "title": metadata.get("title", ""),
                    "content": doc,
                    "category": metadata.get("category", ""),
                    "region": metadata.get("region", ""),
                    "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 0
                })
        
        return relevant_knowledge
        
    except Exception as e:
        print(f"❌ Error retrieving travel knowledge: {e}")
        return []

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

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time chat"""
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
                
                # Process the message
                response, function_calls, audio_base64 = await process_chat_message(user_message, conversation_id, speech_speed)
                
                # Send response back to client
                await manager.send_personal_message(json.dumps({
                    "type": "response",
                    "content": response,
                    "function_calls": function_calls,
                    "audio_base64": audio_base64,
                    "timestamp": datetime.now().isoformat()
                }), websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/chat")
async def chat_api(request: ChatRequest):
    """REST API endpoint for chat"""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        response, function_calls, audio_base64 = await process_chat_message(
            request.message, 
            conversation_id, 
            request.speech_speed
        )
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            function_calls=function_calls,
            audio_base64=audio_base64
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/api/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id in conversations:
        # Filter out system messages for client
        user_messages = [
            msg for msg in conversations[conversation_id] 
            if msg.get("role") in ["user", "assistant"] and msg.get("content")
        ]
        return {"conversation_id": conversation_id, "messages": user_messages}
    
    return {"conversation_id": conversation_id, "messages": []}

@app.delete("/api/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"message": "Conversation cleared"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    return {
        "status": "healthy",
        "model": model_name,
        "active_conversations": len(conversations),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/conversation/{conversation_id}/knowledge")
async def get_conversation_knowledge(conversation_id: str, query: str = ""):
    """Get relevant knowledge and history for a conversation"""
    try:
        # Get conversation history
        history = []
        if conversation_id in conversations:
            user_messages = [
                msg for msg in conversations[conversation_id] 
                if msg.get("role") in ["user", "assistant"] and msg.get("content")
            ]
            history = user_messages
        
        # Get relevant knowledge if query provided
        knowledge = []
        if query:
            knowledge = get_relevant_travel_knowledge(query, limit=5)
        
        # Get conversation context from ChromaDB
        relevant_history = []
        if query:
            relevant_history = get_relevant_conversation_history(query, conversation_id, limit=3)
        
        return {
            "conversation_id": conversation_id,
            "current_history": history,
            "relevant_knowledge": knowledge,
            "relevant_history": relevant_history
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving knowledge: {str(e)}")

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    """Generate text-to-speech audio for given text with speed control"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Validate speed parameter
        if request.speed < 0.5 or request.speed > 2.0:
            raise HTTPException(
                status_code=400, 
                detail="Speech speed must be between 0.5 and 2.0"
            )
        
        audio_base64 = text_to_speech(
            request.text, 
            max_length=request.max_length, 
            speed=request.speed
        )
        
        if audio_base64:
            return {
                "audio_base64": audio_base64, 
                "text": request.text,
                "speed": request.speed,
                "max_length": request.max_length
            }
        else:
            raise HTTPException(status_code=500, detail="TTS generation failed")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.get("/api/knowledge/search")
async def search_knowledge(query: str, limit: int = 5):
    """Search travel knowledge base"""
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        knowledge = get_relevant_travel_knowledge(query, limit)
        
        return {
            "query": query,
            "results": knowledge,
            "total_results": len(knowledge)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge search error: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Get ChromaDB statistics
        conversation_count = user_conversations_collection.count()
        knowledge_count = travel_knowledge_collection.count()
        
        return {
            "active_conversations": len(conversations),
            "stored_conversations": conversation_count,
            "knowledge_base_entries": knowledge_count,
            "tts_available": tts_model is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

def start_server():
    """Start the FastAPI server"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: Missing OPENAI_API_KEY environment variable")
        print("Please set this variable in your .env file.")
        return
    
    print("🧳 Starting Travel Assistant Chatbot...")
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

def export_messages_to_excel(messages: List[Dict[str, Any]], filename: Optional[str] = None) -> BytesIO:
    """Export messages to Excel format"""
    if not filename:
        filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    # Prepare data for DataFrame - only content
    data = []
    for msg in messages:
        content = msg.get('content', '')
        if content.strip():  # Only add non-empty content
            data.append({'Message': content})
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Chat Messages', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Chat Messages']
        
        # Adjust column width for content
        worksheet.column_dimensions['A'].width = 100  # Message content
        
        # Style the header
        from openpyxl.styles import Font, PatternFill
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
    
    output.seek(0)
    return output

def export_messages_to_txt(messages: List[Dict[str, Any]], filename: Optional[str] = None) -> StringIO:
    """Export messages to TXT format"""
    if not filename:
        filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    output = StringIO()
    
    for i, msg in enumerate(messages):
        content = msg.get('content', '')
        if content.strip():  # Only add non-empty content
            output.write(f"{content}")
            # Add separator between messages (except for the last one)
            if i < len(messages) - 1:
                output.write("\n\n")
    
    output.seek(0)
    return output
    
    output.seek(0)
    return output

@app.post("/api/export")
async def export_messages(request: ExportRequest):
    """Export selected messages to Excel or TXT format"""
    try:
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided for export")
        
        if request.format.lower() not in ['excel', 'txt']:
            raise HTTPException(status_code=400, detail="Format must be 'excel' or 'txt'")
        
        # Generate filename if not provided
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not request.filename:
            base_filename = f"chat_export_{timestamp}"
        else:
            base_filename = request.filename.replace('.xlsx', '').replace('.txt', '')
        
        if request.format.lower() == 'excel':
            filename = f"{base_filename}.xlsx"
            file_content = export_messages_to_excel(request.messages, filename)
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            return StreamingResponse(
                BytesIO(file_content.getvalue()),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        else:  # txt format
            filename = f"{base_filename}.txt"
            file_content = export_messages_to_txt(request.messages, filename)
            media_type = "text/plain"
            
            return StreamingResponse(
                StringIO(file_content.getvalue()),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting messages: {str(e)}")