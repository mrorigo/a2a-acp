FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install uv
RUN apt-get update && apt-get install -y make
RUN make install

# Install optional push notification dependencies
RUN uv pip install -e ".[notifications]"

EXPOSE 8000
ENV A2A_AUTH_TOKEN="your-secret-token"
ENV PUSH_NOTIFICATIONS_ENABLED=true

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.a2a_acp.main:create_app", "--host", "0.0.0.0", "--port", "8000"]