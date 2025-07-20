# 🔌 Streamlit Port Configuration

This guide explains how to run Streamlit apps on different ports and configure the server settings.

## 🚀 Quick Start

### **Method 1: Using the Custom Runner Scripts**

**Run Vertex Tuning App on Custom Port:**
```bash
# Run on port 8502
python run_vertex_tuning.py --port 8502

# Run on port 3000 with custom address
python run_vertex_tuning.py --port 3000 --address 0.0.0.0

# Run with custom config
python run_vertex_tuning.py --port 8502 --config .streamlit/custom_config.toml
```

**Run Any Streamlit App on Custom Port:**
```bash
# Run main app on port 8502
python run_streamlit.py app.py --port 8502

# Run AI app on port 8503
python run_streamlit.py app_ai.py --port 8503

# Run cloud app on port 8504
python run_streamlit.py app_cloud.py --port 8504

# Run local app on port 8505
python run_streamlit.py app_local.py --port 8505
```

### **Method 2: Using Environment Variables**

```bash
# Set environment variables
export STREAMLIT_PORT=8502
export STREAMLIT_ADDRESS=localhost

# Run with start_local.py
python start_local.py
```

### **Method 3: Direct Streamlit Command**

```bash
# Run directly with streamlit
streamlit run app_vertex_tuning.py --server.port 8502 --server.address localhost

# Run with custom config
streamlit run app_vertex_tuning.py --config .streamlit/config.toml --server.port 8502
```

## 📁 Configuration Files

### **Local Development Config** (`.streamlit/config.toml`)
```toml
[server]
port = 8501
address = "localhost"
headless = false
enableCORS = true
enableXsrfProtection = true

[browser]
serverAddress = "localhost"
serverPort = 8501
```

### **Cloud Deployment Config** (`streamlit_config.toml`)
```toml
[server]
port = 8080
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false
```

## 🔧 Port Configuration Options

### **1. Command Line Arguments**
```bash
# Basic port change
--port 8502

# Custom address and port
--server.port 8502 --server.address 0.0.0.0

# With config file
--config .streamlit/config.toml --server.port 8502
```

### **2. Environment Variables**
```bash
export STREAMLIT_PORT=8502
export STREAMLIT_ADDRESS=localhost
```

### **3. Configuration File**
```toml
[server]
port = 8502
address = "localhost"
```

### **4. Streamlit Config File**
Create `.streamlit/config.toml` in your project directory:
```toml
[server]
port = 8502
address = "localhost"
```

## 🌐 Common Port Usage

| Port | Common Use | Example |
|------|------------|---------|
| 8501 | Default Streamlit | `streamlit run app.py` |
| 8502 | Alternative Streamlit | `python run_streamlit.py app.py --port 8502` |
| 3000 | Development | `python run_vertex_tuning.py --port 3000` |
| 8080 | Cloud Deployment | `streamlit run app.py --server.port 8080` |
| 5000 | Flask Alternative | `python run_streamlit.py app.py --port 5000` |

## 🔄 Running Multiple Apps

You can run multiple Streamlit apps simultaneously on different ports:

```bash
# Terminal 1: Main app on 8501
python run_streamlit.py app.py --port 8501

# Terminal 2: Vertex tuning on 8502
python run_streamlit.py app_vertex_tuning.py --port 8502

# Terminal 3: AI app on 8503
python run_streamlit.py app_ai.py --port 8503

# Terminal 4: Cloud app on 8504
python run_streamlit.py app_cloud.py --port 8504
```

## 🛠️ Advanced Configuration

### **Custom Config File**
Create `.streamlit/custom_config.toml`:
```toml
[server]
port = 8502
address = "0.0.0.0"
headless = false
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"
serverPort = 8502

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### **Network Access**
To allow access from other devices on your network:
```bash
# Bind to all interfaces
python run_streamlit.py app.py --port 8502 --address 0.0.0.0

# Or use environment variable
export STREAMLIT_ADDRESS=0.0.0.0
python run_streamlit.py app.py --port 8502
```

### **HTTPS/SSL**
For production with SSL:
```bash
# Create SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Run with SSL
streamlit run app.py --server.port 8502 --server.address 0.0.0.0 --server.sslCertFile cert.pem --server.sslKeyFile key.pem
```

## 🔍 Troubleshooting

### **Port Already in Use**
```bash
# Check what's using the port
lsof -i :8501

# Kill the process
kill -9 <PID>

# Or use a different port
python run_streamlit.py app.py --port 8502
```

### **Permission Denied**
```bash
# On Linux/Mac, you might need sudo for ports < 1024
sudo python run_streamlit.py app.py --port 80

# Or use a higher port number
python run_streamlit.py app.py --port 8502
```

### **Firewall Issues**
```bash
# Allow port through firewall (Ubuntu/Debian)
sudo ufw allow 8502

# Allow port through firewall (macOS)
sudo pfctl -e
# Add rule to /etc/pf.conf
```

## 📋 Available Apps

List of available Streamlit apps in this project:

```bash
# List available apps
python run_streamlit.py --help

# Available apps:
# • app.py - Main image labeling app
# • app_ai.py - AI-enhanced image labeling
# • app_cloud.py - Cloud deployment version
# • app_local.py - Local development version
# • app_vertex_tuning.py - Vertex AI supervised tuning
```

## 🚀 Production Deployment

### **Docker with Custom Port**
```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8502
CMD ["streamlit", "run", "app.py", "--server.port", "8502", "--server.address", "0.0.0.0"]
```

### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'
services:
  streamlit:
    build: .
    ports:
      - "8502:8502"
    environment:
      - STREAMLIT_PORT=8502
      - STREAMLIT_ADDRESS=0.0.0.0
```

### **Kubernetes**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: streamlit-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: streamlit
  template:
    metadata:
      labels:
        app: streamlit
    spec:
      containers:
      - name: streamlit
        image: your-registry/streamlit-app:latest
        ports:
        - containerPort: 8502
        env:
        - name: STREAMLIT_PORT
          value: "8502"
        - name: STREAMLIT_ADDRESS
          value: "0.0.0.0"
---
apiVersion: v1
kind: Service
metadata:
  name: streamlit-service
spec:
  selector:
    app: streamlit
  ports:
  - port: 80
    targetPort: 8502
  type: LoadBalancer
```

## 📚 Additional Resources

- [Streamlit Configuration](https://docs.streamlit.io/library/advanced-features/configuration)
- [Streamlit Server Configuration](https://docs.streamlit.io/library/advanced-features/configuration#server-configuration)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

**Happy coding with custom ports! 🚀** 