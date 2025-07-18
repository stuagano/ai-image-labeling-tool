#!/usr/bin/env python3
"""
Service Management Script for Local Development
Manages FastAPI server and Streamlit app
"""

import subprocess
import time
import sys
import os
import signal
import requests
from threading import Thread

def check_service(url, name):
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return True, f"✅ {name} is running"
        else:
            return False, f"❌ {name} returned status {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"❌ {name} is not running: {e}"

def start_fastapi():
    """Start FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        subprocess.run([
            sys.executable, "api_server.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server stopped")
    except Exception as e:
        print(f"❌ FastAPI server error: {e}")

def start_streamlit():
    """Start Streamlit app"""
    print("🌐 Starting Streamlit app...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_local.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Streamlit app error: {e}")

def stop_services():
    """Stop all services"""
    print("🛑 Stopping services...")
    
    # Kill FastAPI server
    try:
        subprocess.run(["pkill", "-f", "api_server.py"], check=False)
        print("✅ FastAPI server stopped")
    except Exception as e:
        print(f"⚠️  Error stopping FastAPI: {e}")
    
    # Kill Streamlit app
    try:
        subprocess.run(["pkill", "-f", "streamlit run app_local.py"], check=False)
        print("✅ Streamlit app stopped")
    except Exception as e:
        print(f"⚠️  Error stopping Streamlit: {e}")

def status():
    """Check status of all services"""
    print("📊 Service Status")
    print("=" * 40)
    
    # Check FastAPI
    fastapi_ok, fastapi_msg = check_service("http://localhost:8000/health", "FastAPI Server")
    print(fastapi_msg)
    
    # Check Streamlit
    streamlit_ok, streamlit_msg = check_service("http://localhost:8501/_stcore/health", "Streamlit App")
    print(streamlit_msg)
    
    print("\n📋 URLs:")
    print("  • Streamlit App:  http://localhost:8501")
    print("  • FastAPI Server: http://localhost:8000")
    print("  • API Docs:       http://localhost:8000/docs")
    
    return fastapi_ok and streamlit_ok

def start_both():
    """Start both services"""
    print("🏠 Starting AI Image Labeling Tool Services")
    print("=" * 50)
    
    # Check if services are already running
    if status():
        print("\n⚠️  Services are already running!")
        return
    
    print("\n🔄 Starting services...")
    
    # Start FastAPI server in background
    api_thread = Thread(target=start_fastapi, daemon=True)
    api_thread.start()
    
    # Wait for API to start
    print("⏳ Waiting for FastAPI server to start...")
    for i in range(10):
        time.sleep(1)
        if check_service("http://localhost:8000/health", "FastAPI")[0]:
            print("✅ FastAPI server is ready!")
            break
    else:
        print("⚠️  FastAPI server may not be ready")
    
    # Start Streamlit app
    start_streamlit()

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python manage_services.py [start|stop|status|restart]")
        print("\nCommands:")
        print("  start   - Start both services")
        print("  stop    - Stop both services")
        print("  status  - Check service status")
        print("  restart - Restart both services")
        return
    
    command = sys.argv[1].lower()
    
    if command == "start":
        start_both()
    elif command == "stop":
        stop_services()
    elif command == "status":
        status()
    elif command == "restart":
        stop_services()
        time.sleep(2)
        start_both()
    else:
        print(f"Unknown command: {command}")
        print("Use: start, stop, status, or restart")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        stop_services()
        sys.exit(0) 