#!/usr/bin/env python3
"""
Run Integrated Image Labeling & Fine-Tuning App
Combines image labeling with Vertex AI supervised fine-tuning
"""

import os
import sys
import subprocess
import argparse

def main():
    """Main function to run the integrated app"""
    parser = argparse.ArgumentParser(description='Run Integrated Image Labeling & Fine-Tuning App')
    parser.add_argument(
        '--port', 
        type=int, 
        default=8502,
        help='Port to run Streamlit on (default: 8502)'
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
    
    print(f"🎯 Starting Integrated Image Labeling & Fine-Tuning App")
    print(f"📍 Address: {args.address}")
    print(f"🔌 Port: {args.port}")
    print(f"📁 App: app_integrated.py")
    print(f"⚙️  Config: {args.config}")
    print("=" * 60)
    
    # Set environment variables
    os.environ['STREAMLIT_PORT'] = str(args.port)
    os.environ['STREAMLIT_ADDRESS'] = args.address
    
    # Build command
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app_integrated.py",
        "--server.port", str(args.port),
        "--server.address", args.address
    ]
    
    # Note: Streamlit config is automatically loaded from .streamlit/config.toml
    # No need to pass --config flag as it's not supported in this version
    
    try:
        print(f"🚀 Starting integrated app on http://{args.address}:{args.port}")
        print("📋 Features available:")
        print("  • Image labeling with bounding boxes")
        print("  • AI-powered object detection (YOLO, Transformers, Gemini)")
        print("  • GCP authentication and Vertex AI integration")
        print("  • Gemini-powered prompt generation")
        print("  • Supervised fine-tuning job creation")
        print("  • Model deployment to Vertex AI endpoints")
        print("\nPress Ctrl+C to stop")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 