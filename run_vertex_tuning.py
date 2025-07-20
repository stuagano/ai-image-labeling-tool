#!/usr/bin/env python3
"""
Run Vertex AI Supervised Tuning App
Configurable port and address for local development
"""

import os
import sys
import subprocess
import argparse

def main():
    """Main function to run the Vertex tuning app"""
    parser = argparse.ArgumentParser(description='Run Vertex AI Supervised Tuning App')
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Port to run Streamlit on (default: 8501)'
    )
    parser.add_argument(
        '--address', 
        type=str, 
        default='localhost',
        help='Address to bind to (default: localhost)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='.streamlit/config.toml',
        help='Path to Streamlit config file (default: .streamlit/config.toml)'
    )
    parser.add_argument(
        '--app', 
        type=str, 
        default='app_vertex_tuning.py',
        help='Streamlit app to run (default: app_vertex_tuning.py)'
    )
    
    args = parser.parse_args()
    
    print(f"🤖 Starting Vertex AI Supervised Tuning App")
    print(f"📍 Address: {args.address}")
    print(f"🔌 Port: {args.port}")
    print(f"📁 App: {args.app}")
    print(f"⚙️  Config: {args.config}")
    print("=" * 50)
    
    # Set environment variables
    os.environ['STREAMLIT_PORT'] = str(args.port)
    os.environ['STREAMLIT_ADDRESS'] = args.address
    
    # Build command
    cmd = [
        sys.executable, "-m", "streamlit", "run", args.app,
        "--server.port", str(args.port),
        "--server.address", args.address
    ]
    
    # Add config file if it exists
    if os.path.exists(args.config):
        cmd.extend(["--config", args.config])
    
    try:
        print(f"🚀 Starting app on http://{args.address}:{args.port}")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 