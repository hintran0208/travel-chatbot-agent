# API Setup Guide for TravelBot

*See also: [README.md](README.md), [PROMPT_ENGINEERING.md](PROMPT_ENGINEERING.md), and [USE_CASES.md](USE_CASES.md) for more details.*

# Travel APIs Setup Guide

This guide will help you register for all the free travel APIs used in TravelBot and obtain the necessary credentials.

## 🔑 API Services Registration

### 1. RapidAPI (for Hotel Search via Booking.com)
**Purpose**: Search for hotels with real pricing and availability
**Free Tier**: 1,000 requests/month

**Registration Steps**:
1. Visit: https://rapidapi.com/
2. Sign up for a free account
3. Navigate to: https://rapidapi.com/apidojo/api/booking/
4. Subscribe to the "Basic" plan (free)
5. Go to "Endpoints" → copy your API key from the header

**Required Credentials**:
- `RAPIDAPI_KEY`: Your RapidAPI key

---

### 2. Amadeus API (for Flight Search)
**Purpose**: Search for real flight options and pricing
**Free Tier**: 2,000 requests/month

**Registration Steps**:
1. Visit: https://developers.amadeus.com/
2. Click "Sign Up" and create a free account
3. Verify your email address
4. Go to "My Apps" → "Create New App"
5. Name your app (e.g., "TravelBot")
6. Select "Self-Service" API
7. Copy your API Key and API Secret

**Required Credentials**:
- `AMADEUS_API_KEY`: Your Amadeus API Key
- `AMADEUS_API_SECRET`: Your Amadeus API Secret

---

### 3. OpenWeatherMap API (for Weather Data)
**Purpose**: Get current weather and forecasts for destinations
**Free Tier**: 1,000 requests/day

**Registration Steps**:
1. Visit: https://openweathermap.org/api
2. Click "Sign Up" and create a free account
3. Verify your email address
4. Go to "API keys" in your account dashboard
5. Copy the default API key (or create a new one)

**Required Credentials**:
- `OPENWEATHER_API_KEY`: Your OpenWeatherMap API key

---

### 4. Foursquare Places API (for Local Activities)
**Purpose**: Discover local attractions, restaurants, and activities
**Free Tier**: 1,000 requests/day

**Registration Steps**:
1. Visit: https://developer.foursquare.com/
2. Click "Get Started" and sign up
3. Create a new project
4. Go to your project dashboard
5. Copy the API key from the "API Keys" section

**Required Credentials**:
- `FOURSQUARE_API_KEY`: Your Foursquare API key

---

### 5. ExchangeRate-API (for Currency Conversion)
**Purpose**: Convert currencies with real-time exchange rates (supports VND)
**Free Tier**: 2,000 requests/month

**Registration Steps**:
1. Visit: https://www.exchangerate-api.com/
2. Click "Get Free Key" 
3. Create account with email and password
4. Verify your email address
5. Copy your API key from the dashboard

**Required Credentials**:
- `EXCHANGERATE_API_KEY`: Your ExchangeRate-API key

**Supported Currencies**: 160+ including VND, USD, EUR, GBP, JPY, etc.

---

## 🛠️ Setup Instructions

### 1. Update your .env file
After registering for all services, update your `.env` file with the credentials:

```env
# OpenAI Configuration (already configured)
OPENAI_API_KEY=sk-IXzuMh_OeMdLACbNjuIj6g
OPENAI_BASE_URL=https://aiportalapi.stu-platform.live/jpe
OPENAI_MODEL_NAME=GPT-4o-mini

# RapidAPI Key (for hotel search via Booking.com API)
RAPIDAPI_KEY=your_rapidapi_key_here

# ExchangeRate-API (for currency conversion with VND support)
EXCHANGERATE_API_KEY=your_exchangerate_api_key_here

# Amadeus API (for flight search)
AMADEUS_API_KEY=your_amadeus_api_key_here
AMADEUS_API_SECRET=your_amadeus_api_secret_here

# OpenWeatherMap API (for weather data)
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Foursquare API (for local activities)
FOURSQUARE_API_KEY=your_foursquare_api_key_here
```

### 2. Test the APIs
Once all credentials are added, start the application:

```bash
python main.py
```

The system will automatically:
- Use real APIs when credentials are available
- Fall back to mock data if any API fails or credentials are missing
- Display helpful error messages in the console for debugging

---

## 📊 API Usage Limits Summary

| Service | Free Tier Limit | Purpose |
|---------|----------------|---------|
| RapidAPI/Booking.com | 1,000 requests/month | Hotel search |
| ExchangeRate-API | 2,000 requests/month | Currency conversion |
| Amadeus | 2,000 requests/month | Flight search |
| OpenWeatherMap | 1,000 requests/day | Weather data |
| Foursquare | 1,000 requests/day | Local activities |

---

## 🚨 Important Notes

1. **Rate Limits**: All APIs have rate limits. The application includes error handling to gracefully fall back to mock data if limits are exceeded.

2. **API Keys Security**: Never commit your actual API keys to version control. Keep them in your `.env` file only.

3. **Testing**: Start with one API at a time to ensure each integration works correctly.

4. **Monitoring**: Keep track of your API usage through each provider's dashboard to avoid exceeding free tier limits.

5. **Fallback System**: The application is designed to work even if some APIs are unavailable - it will use mock data as a fallback.

---

## 🔧 Troubleshooting

**If an API isn't working**:
1. Check that your API key is correctly set in the `.env` file
2. Verify your account is active and verified
3. Check the console output for specific error messages
4. Ensure you haven't exceeded your API limits
5. Try testing the API directly in the provider's documentation

**Common Issues**:
- **Amadeus**: Make sure to use the test environment URLs initially
- **RapidAPI**: Ensure you've subscribed to the Booking.com API specifically
- **OpenWeatherMap**: API keys may take up to 10 minutes to activate
- **Foursquare**: Make sure to use the v3 API endpoints
