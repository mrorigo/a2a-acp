# Deployment Guide

Deploy A2A-ACP to production with confidence using our comprehensive deployment strategies.

## Quick Deployment

### One-Command Deployment

```bash
# Deploy with automatic setup
./deploy.sh

# Application will be available at http://localhost:8000
# Health check: http://localhost:8000/health
```

### Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f a2a-acp

# Stop services
docker-compose down
```

## Production Deployment

### 1. Environment Configuration

Create production `.env` file:

```bash
# Core Settings
A2A_AGENT_COMMAND="/usr/local/bin/codex-acp"
A2A_AGENT_API_KEY="${OPENAI_API_KEY}"
A2A_AUTH_TOKEN="your-production-secret-token"
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=WARNING

# Security
PUSH_NOTIFICATIONS_ENABLED=true
PUSH_NOTIFICATION_HMAC_SECRET="your-hmac-secret"
RATE_LIMIT_PER_MINUTE=60

# Performance
MAX_CONCURRENT_TASKS=100
TASK_TIMEOUT=300
```

### 2. Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  a2a-acp:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - A2A_AGENT_COMMAND=/usr/local/bin/codex-acp
      - A2A_AGENT_API_KEY=${OPENAI_API_KEY}
      - A2A_AUTH_TOKEN=${A2A_AUTH_TOKEN}
      - LOG_LEVEL=WARNING
      - PUSH_NOTIFICATIONS_ENABLED=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
    networks:
      - a2a-acp-network

networks:
  a2a-acp-network:
    driver: bridge
```

### 3. Reverse Proxy Setup

#### Nginx Configuration

```nginx
# /etc/nginx/sites-available/a2a-acp
upstream a2a_acp_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Health check endpoint (no auth required)
    location /health {
        proxy_pass http://a2a_acp_backend;
        proxy_connect_timeout 10s;
        proxy_read_timeout 10s;
    }

    # API endpoints (require authentication)
    location / {
        proxy_pass http://a2a_acp_backend;

        # Authentication
        auth_request /auth;

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Authentication endpoint
    location /auth {
        internal;
        proxy_pass http://a2a_acp_backend/auth;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URI $request_uri;
    }
}
```

### 4. SSL/TLS Setup

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (add to crontab)
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring & Observability

### Health Check Endpoints

```bash
# Comprehensive health check (requires auth)
curl -X GET "http://localhost:8000/health" \
  -H "Authorization: Bearer your-token"

# Response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "healthy",
    "push_notifications": "healthy",
    "streaming": "healthy"
  },
  "version": "1.0.0"
}
```

### Metrics Endpoints

```bash
# Push notification delivery metrics
curl -X GET "http://localhost:8000/metrics/push-notifications" \
  -H "Authorization: Bearer your-token"

# System metrics
curl -X GET "http://localhost:8000/metrics/system" \
  -H "Authorization: Bearer your-token"
```

### Log Aggregation

```bash
# Application logs location
tail -f /app/logs/a2a_acp.log

# Setup log shipping (example with vector)
# /etc/vector/vector.toml
[sources.app_logs]
  type = "file"
  include = ["/app/logs/*.log"]

[sinks.elasticsearch]
  type = "elasticsearch"
  inputs = ["app_logs"]
  host = "http://elasticsearch:9200"
```

## Scaling

### Horizontal Scaling

```yaml
# Multi-instance docker-compose.scale.yml
version: '3.8'

services:
  a2a-acp:
    &a2a-acp
    build: .
    environment:
      - INSTANCE_ID=a2a-acp-${INSTANCE_NO}
    volumes:
      - shared_data:/app/data
    deploy:
      replicas: 3

  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - a2a-acp

volumes:
  shared_data:
    driver: nfs  # or cloud provider volume
```

### Database Scaling

For high-volume deployments:

```yaml
# External PostgreSQL for better performance
services:
  database:
    image: postgres:15
    environment:
      - POSTGRES_DB=a2a_acp
      - POSTGRES_USER=a2a_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  a2a-acp:
    depends_on:
      - database
    environment:
      - DATABASE_URL=postgresql://a2a_user:secure_password@database/a2a_acp
```

## Security Hardening

### Production Security Checklist

- [ ] **Strong Authentication**: Use cryptographically secure tokens
- [ ] **TLS Everywhere**: Enable HTTPS on all endpoints
- [ ] **Minimal Permissions**: Run container with non-root user
- [ ] **Network Security**: Use internal networks for inter-service communication
- [ ] **Regular Updates**: Keep base images and dependencies updated
- [ ] **Secret Management**: Use proper secret management (Vault, etc.)
- [ ] **Audit Logging**: Enable comprehensive audit logging
- [ ] **Rate Limiting**: Configure appropriate rate limits
- [ ] **Firewall Rules**: Restrict access to necessary ports only
- [ ] **Monitoring**: Set up security monitoring and alerting

### Container Security

```dockerfile
# Production Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r a2a && useradd -r -g a2a a2a

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

USER a2a
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

## Backup & Recovery

### Database Backup

```bash
# SQLite backup (if using SQLite)
cp /app/data/a2a_acp.db /backup/a2a_acp_$(date +%Y%m%d_%H%M%S).db

# PostgreSQL backup
pg_dump a2a_acp > /backup/a2a_acp_$(date +%Y%m%d_%H%M%S).sql
```

### Configuration Backup

```bash
# Backup configuration
tar -czf /backup/config_$(date +%Y%m%d_%H%M%S).tar.gz \
    .env \
    docker-compose*.yml \
    nginx.conf
```

### Recovery Procedure

```bash
# 1. Restore database
cp /backup/a2a_acp_20240115_100000.db /app/data/a2a_acp.db

# 2. Restore configuration
tar -xzf /backup/config_20240115_100000.tar.gz

# 3. Restart services
docker-compose down
docker-compose up -d
```

## Troubleshooting Deployment

### Common Deployment Issues

**Container won't start**
```bash
# Check detailed logs
docker-compose logs a2a-acp

# Verify environment variables
docker-compose exec a2a-acp env | grep A2A_
```

**Health check fails**
```bash
# Test health endpoint manually
curl -v http://localhost:8000/health

# Check if port is accessible
netstat -tlnp | grep 8000
```

**Agent connection issues**
```bash
# Verify agent binary exists
docker-compose exec a2a-acp which codex-acp

# Check agent logs
docker-compose exec a2a-acp ls -la /usr/local/bin/codex-acp
```

**Database connection problems**
```bash
# Check database file permissions
docker-compose exec a2a-acp ls -la data/

# Verify SQLite functionality
docker-compose exec a2a-acp sqlite3 data/a2a_acp.db "SELECT 1;"
```

## Performance Tuning

### Memory Optimization

```bash
# Set memory limits
export MAX_CONCURRENT_TASKS=50
export TASK_TIMEOUT=180

# Enable response caching
export TOOL_CACHE_ENABLED=true
export TOOL_CACHE_TTL=3600
```

### CPU Optimization

```bash
# Adjust async worker settings
export ASYNC_WORKERS=4

# Configure database connection pool
export DB_POOL_SIZE=10
export DB_MAX_OVERFLOW=20
```

## Cost Optimization

### Efficient Resource Usage

- **Right-size instances**: Start with small instances and scale based on load
- **Auto-scaling**: Use auto-scaling groups for variable loads
- **Resource limits**: Set appropriate CPU and memory limits per instance
- **Database choice**: Consider managed databases for easier scaling

---

**Deployment complete!** 🚀 For monitoring and maintenance, see [monitoring.md](monitoring.md).