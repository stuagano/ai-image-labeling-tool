#!/usr/bin/env python3
"""
Generic Streamlit Runner
Run any Streamlit app on a custom port
"""

import os
import sys
import subprocess
import argparse

def main():
    """Main function to run Streamlit apps"""
    parser = argparse.ArgumentParser(description='Run Streamlit App on Custom Port')
    parser.add_argument(
        'app', 
        type=str,
        help='Streamlit app file to run (e.g., app_vertex_tuning.py)'
    )
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
    
    args = parser.parse_args()
    
    # Check if app file exists
    if not os.path.exists(args.app):
        print(f"❌ App file not found: {args.app}")
        print("Available apps:")
        for file in os.listdir('.'):
            if file.endswith('.py') and 'app' in file.lower():
                print(f"  • {file}")
        sys.exit(1)
    
    print(f"🚀 Starting Streamlit App")
    print(f"📁 App: {args.app}")
    print(f"📍 Address: {args.address}")
    print(f"🔌 Port: {args.port}")
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
        print(f"🌐 App will be available at: http://{args.address}:{args.port}")
        print("Press Ctrl+C to stop")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 