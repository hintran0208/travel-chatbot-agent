# Travel Assistant ChatBot - Deployment Guide

## 🚀 Deployment Options

This guide covers different deployment options for your Travel Assistant ChatBot with ChromaDB persistence.

### 📋 Prerequisites

1. **API Keys**: Ensure you have all required API keys in your `.env` file
2. **Docker**: Install Docker and Docker Compose for containerized deployment
3. **ChromaDB Data**: The application will automatically create and persist ChromaDB data

### 🐳 Docker Deployment (Recommended)

#### Quick Start

```bash
# 1. Copy environment variables
cp .env.example .env

# 2. Edit .env with your API keys
nano .env

# 3. Run deployment script
./deploy.sh
```

#### Manual Docker Commands

```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Check logs
docker-compose logs -f

# Stop the application
docker-compose down

# Stop and remove volumes (⚠️ This will delete ChromaDB data)
docker-compose down -v
```

### ☁️ Cloud Deployment Options

#### 1. Railway.app

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Railway will automatically use the `Procfile` for deployment
4. **Note**: Railway provides ephemeral storage, so ChromaDB data may not persist between deployments

#### 2. Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/[PROJECT-ID]/travel-chatbot

# Deploy to Cloud Run
gcloud run deploy travel-chatbot \
  --image gcr.io/[PROJECT-ID]/travel-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

#### 3. AWS ECS/Fargate

- Use the provided `Dockerfile`
- Create ECS task definition
- Configure persistent storage with EFS for ChromaDB data

#### 4. Heroku

```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-travel-chatbot

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key_here
# ... (set all other environment variables)

# Deploy
git push heroku main
```

### 💾 ChromaDB Data Persistence

#### Local Development

- Data stored in `./chroma_db/` directory
- Automatically created and managed

#### Docker Deployment

- Data persisted in Docker volume `chroma_data`
- Survives container restarts and updates
- To backup: `docker-compose exec travel-chatbot tar -czf /app/chroma_backup.tar.gz /app/chroma_data`

#### Cloud Deployment

- **Railway/Vercel**: Ephemeral storage (data lost on restart)
- **GCP/AWS/Azure**: Use persistent volumes or managed databases
- **Heroku**: Use Heroku Postgres with pgvector extension

### 🔧 Configuration

#### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_KEY_TTS=your_tts_api_key
OPENAI_API_KEY_EMBEDDING=your_embedding_api_key
OPENAI_BASE_URL=https://aiportalapi.stu-platform.live/jpe
OPENAI_MODEL_NAME=GPT-4o-mini

# ChromaDB
CHROMA_DB_PATH=./chroma_data

# Server
HOST=0.0.0.0
PORT=8000
```

#### Production Considerations

1. **Security**: Use secrets management for API keys
2. **Monitoring**: Set up health checks and logging
3. **Scaling**: Consider load balancing for high traffic
4. **Backup**: Regular ChromaDB data backups
5. **SSL/TLS**: Use HTTPS in production

### 🔍 Monitoring and Troubleshooting

#### Health Check

```bash
curl http://localhost:8000/health
```

#### View Logs

```bash
# Docker Compose
docker-compose logs -f

# Single container
docker logs travel-chatbot
```

#### Common Issues

1. **Port conflicts**: Change port in docker-compose.yml
2. **API key errors**: Verify all keys in .env file
3. **ChromaDB permissions**: Ensure write permissions to data directory
4. **Memory issues**: Increase container memory limits

### 📊 Performance Optimization

1. **ChromaDB**: Use SSD storage for better performance
2. **Memory**: Allocate sufficient RAM (minimum 2GB recommended)
3. **CPU**: Multi-core CPU improves concurrent request handling
4. **Caching**: Consider Redis for session management

### 🔄 Updates and Maintenance

```bash
# Pull latest changes
git pull origin main

# Rebuild and redeploy
docker-compose up --build -d

# View updated logs
docker-compose logs -f
```

For production environments, consider blue-green deployments to minimize downtime.
