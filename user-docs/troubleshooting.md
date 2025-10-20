# Troubleshooting Guide

Solve common issues and get A2A-ACP running smoothly.

## Quick Diagnostics

### 1. Health Check

```bash
# Test basic connectivity
curl -X GET http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

**If health check fails:**
- Check if server is running: `ps aux | grep a2a-acp`
- Verify port 8000 is not blocked: `netstat -tlnp | grep 8000`
- Check logs: `tail -f logs/a2a_acp.log`

### 2. Agent Connectivity

```bash
# Test agent command directly
/usr/local/bin/codex-acp --help

# Check if agent binary exists and is executable
ls -la /usr/local/bin/codex-acp
```

## Common Issues & Solutions

### 🚫 "Agent not found" Error

**Symptoms**: Server starts but fails when processing requests.

**Causes & Solutions**:

1. **Incorrect agent path**
   ```bash
   # Find correct path
   which codex-acp
   # or
   find /usr -name "codex-acp" 2>/dev/null

   # Update A2A_AGENT_COMMAND with full path
   export A2A_AGENT_COMMAND="/full/path/to/codex-acp"
   ```

2. **Missing execute permissions**
   ```bash
   # Check and fix permissions
   ls -la /usr/local/bin/codex-acp
   chmod +x /usr/local/bin/codex-acp
   ```

3. **Agent not installed**
   ```bash
   # Install the agent (example for codex-acp)
   npm install -g @openai/codex-acp
   ```

### 🚫 Authentication Errors

**Symptoms**: "Unauthorized" or "Invalid token" errors.

**Causes & Solutions**:

1. **Missing auth token**
   ```bash
   # Set auth token
   export A2A_AUTH_TOKEN="your-secret-token"

   # Verify it's set
   echo $A2A_AUTH_TOKEN
   ```

2. **Token mismatch**
   ```bash
   # Ensure client sends correct Authorization header
   curl -H "Authorization: Bearer $A2A_AUTH_TOKEN" \
        http://localhost:8001/a2a/rpc
   ```

3. **API key issues**
   ```bash
   # For agent API keys
   export A2A_AGENT_API_KEY="${OPENAI_API_KEY}"
   echo $A2A_AGENT_API_KEY  # Should not be empty
   ```

### 🚫 Database Connection Issues

**Symptoms**: "Database connection failed" or SQLite errors.

**Causes & Solutions**:

1. **SQLite file permissions**
   ```bash
   # Check data directory
   ls -la data/
   chmod 755 data/
   chmod 644 data/a2a_acp.db
   ```

2. **WAL mode issues**
   ```bash
   # Check SQLite journal mode
   sqlite3 data/a2a_acp.db "PRAGMA journal_mode;"

   # Should return "wal" for proper concurrency
   ```

3. **Database corruption**
   ```bash
   # Check database integrity
   sqlite3 data/a2a_acp.db "PRAGMA integrity_check;"

   # If corrupted, restore from backup
   cp /backup/a2a_acp.db data/
   ```

### 🚫 Push Notification Failures

**Symptoms**: Webhooks not being delivered.

**Causes & Solutions**:

1. **Invalid webhook URL**
   ```bash
   # Test webhook endpoint manually
   curl -X POST https://your-app.com/webhooks/a2a \
        -H "Content-Type: application/json" \
        -d '{"test": "message"}'
   ```

2. **Network connectivity**
   ```bash
   # Check if webhook domain is reachable
   ping your-app.com
   curl -v https://your-app.com/webhooks/a2a
   ```

3. **Authentication issues**
   ```bash
   # Check webhook authentication
   curl -X POST https://your-app.com/webhooks/a2a \
        -H "Authorization: Bearer your-webhook-token" \
        -H "Content-Type: application/json" \
        -d '{"test": "authenticated"}'
   ```

### 🚫 Performance Issues

**Symptoms**: Slow responses or timeouts.

**Causes & Solutions**:

1. **High concurrent load**
   ```bash
   # Check concurrent tasks
   curl http://localhost:8000/metrics/system

   # Adjust limits if needed
   export MAX_CONCURRENT_TASKS=50
   ```

2. **Resource exhaustion**
   ```bash
   # Check system resources
   top -p $(pgrep -f a2a-acp)

   # Monitor memory usage
   free -h
   ```

3. **Database performance**
   ```bash
   # Check query performance
   sqlite3 data/a2a_acp.db "ANALYZE;"

   # Optimize if needed
   sqlite3 data/a2a_acp.db "VACUUM;"
   ```

## Debug Mode

### Enable Verbose Logging

```bash
# Set debug logging
export LOG_LEVEL=DEBUG

# Or run with debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Your debug code here
"
```

### Debug Agent Communication

```bash
# Run agent manually to test
A2A_AGENT_COMMAND="python tests/dummy_agent.py" \
python -m uvicorn src.a2a_acp.main:create_app --reload --log-level debug
```

### Protocol Debugging

```bash
# Enable protocol-level debugging
export A2A_DEBUG_PROTOCOL=true

# Check protocol translation logs
tail -f logs/a2a_acp.log | grep -E "(A2A|ZedACP|translation)"
```

## Log Analysis

### Application Logs

```bash
# View recent logs
tail -f logs/a2a_acp.log

# Search for errors
grep -i error logs/a2a_acp.log

# Filter by component
grep -E "(task_manager|zed_agent|push_notification)" logs/a2a_acp.log
```

### Database Logs

```bash
# Check for database errors
grep -i "database\|sqlite" logs/a2a_acp.log

# Verify database operations
sqlite3 data/a2a_acp.db "SELECT COUNT(*) FROM tasks;"
```

## Configuration Validation

### Environment Check

```bash
# Validate all required variables are set
python -c "
import os
required = ['A2A_AGENT_COMMAND', 'A2A_AUTH_TOKEN']
missing = [var for var in required if not os.getenv(var)]
if missing:
    print(f'❌ Missing: {missing}')
    exit(1)
else:
    print('✅ All required variables set')
"
```

### Agent Validation

```bash
# Test agent independently
eval $A2A_AGENT_COMMAND << 'EOF'
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}}
EOF
```

## Recovery Procedures

### Service Recovery

```bash
# If service is unresponsive
docker-compose restart a2a-acp

# Check restart policy
docker-compose ps

# View restart logs
docker-compose logs --tail 50 a2a-acp
```

### Database Recovery

```bash
# If database is corrupted
cp /backup/a2a_acp_$(date +%Y%m%d).db data/a2a_acp.db

# Verify integrity
sqlite3 data/a2a_acp.db "PRAGMA integrity_check;"
```

### Configuration Recovery

```bash
# If configuration is lost
cp /backup/.env.backup .env

# Validate configuration
source .env && echo "Config loaded successfully"
```

## Getting Help

### Community Support

1. **Check existing issues**: Search GitHub issues for similar problems
2. **Review documentation**: Ensure you're following current best practices
3. **Check logs**: Detailed error information is usually in application logs

### Reporting Issues

When reporting issues, include:

- **Version**: `a2a-acp --version`
- **Environment**: OS, Python version, agent version
- **Configuration**: Relevant environment variables (without secrets)
- **Logs**: Relevant log entries (sanitize sensitive data)
- **Steps to reproduce**: Exact steps that trigger the issue

### Professional Support

For production deployments needing guaranteed support:

- **Enterprise Support**: Contact for SLA-backed support
- **Consulting**: Architecture review and optimization services
- **Training**: Team training on A2A-ACP deployment and operation

## Advanced Debugging

### Database Inspection

```bash
# Check recent tasks
sqlite3 data/a2a_acp.db "SELECT id, status, created_at FROM tasks ORDER BY created_at DESC LIMIT 10;"

# Check message count
sqlite3 data/a2a_acp.db "SELECT COUNT(*) FROM messages;"

# Check contexts
sqlite3 data/a2a_acp.db "SELECT id, created_at, last_activity FROM contexts ORDER BY last_activity DESC;"
```

### Network Debugging

```bash
# Check open ports
netstat -tlnp | grep 8000

# Test port accessibility
telnet localhost 8000

# Check firewall rules
sudo iptables -L | grep 8000
```

### Memory Debugging

```bash
# Check for memory leaks
ps aux | grep a2a-acp

# Monitor memory usage over time
watch -n 5 'ps aux | grep a2a-acp'

# Check garbage collection (Python)
python -c "
import gc
print(f'Objects: {len(gc.get_objects())}')
gc.collect()
print(f'After GC: {len(gc.get_objects())}')
"
```

---

**Problem solved!** 🔧 If issues persist, check [deployment.md](deployment.md) or [configuration.md](configuration.md).