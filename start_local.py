#!/usr/bin/env python3
"""
Startup script for local development
Runs both FastAPI server and Streamlit app
"""

import subprocess
import time
import sys
import os
import signal
from threading import Thread

def run_fastapi_server():
    """Run the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        subprocess.run([
            sys.executable, "api_server.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 FastAPI server stopped")
    except Exception as e:
        print(f"❌ FastAPI server error: {e}")

def run_streamlit_app():
    """Run the Streamlit app"""
    print("🌐 Starting Streamlit app...")
    
    # Get port from environment or use default
    streamlit_port = os.getenv("STREAMLIT_PORT", "8501")
    streamlit_address = os.getenv("STREAMLIT_ADDRESS", "localhost")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_local.py",
            "--server.port", streamlit_port,
            "--server.address", streamlit_address
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Streamlit app error: {e}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ("streamlit", "streamlit"),
        ("fastapi", "fastapi"), 
        ("uvicorn", "uvicorn"),
        ("requests", "requests"),
        ("google-cloud-storage", "google.cloud.storage")
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check environment variables"""
    required_vars = ["GCS_BUCKET_NAME"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them before running the app")
        return False
    
    return True

def main():
    """Main function"""
    print("🏠 AI Image Labeling Tool - Local Development")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("Continuing anyway...")
    
    print("✅ All checks passed!")
    # Get port from environment or use default
    streamlit_port = os.getenv("STREAMLIT_PORT", "8501")
    streamlit_address = os.getenv("STREAMLIT_ADDRESS", "localhost")
    
    print("\n📋 Services:")
    print("  • FastAPI Server: http://localhost:8000")
    print(f"  • Streamlit App:  http://{streamlit_address}:{streamlit_port}")
    print("  • API Docs:       http://localhost:8000/docs")
    print("\n🔄 Starting services...")
    
    # Start FastAPI server in background
    api_thread = Thread(target=run_fastapi_server, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    time.sleep(3)
    
    # Start Streamlit app
    run_streamlit_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        sys.exit(0) 