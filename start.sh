#!/bin/bash

# TravelBot Startup Script

echo "🧳 Starting TravelBot - AI Travel Assistant..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "📝 Please edit .env file with your OpenAI API key"
        echo "   OPENAI_API_KEY=your_api_key_here"
    else
        echo "❌ .env.example file not found. Please create .env manually."
        exit 1
    fi
fi

# Activate virtual environment and start the server
echo "🚀 Starting server..."
.venv/bin/python main.py
