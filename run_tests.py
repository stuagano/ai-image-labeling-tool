#!/usr/bin/env python3
"""
Test Runner for Integrated Image Labeling App
Automatically runs functional tests with headless Chrome
"""

import os
import sys
import subprocess
import time
import argparse

def check_dependencies():
    """Check if testing dependencies are installed"""
    try:
        import selenium
        import requests
        print("✅ Testing dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing testing dependencies: {e}")
        print("💡 Install with: pip install -r requirements_testing.txt")
        return False

def check_chrome():
    """Check if Chrome is available"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.quit()
        print("✅ Chrome driver working")
        return True
    except Exception as e:
        print(f"❌ Chrome driver issue: {e}")
        print("💡 Install Chrome or chromedriver")
        return False

def start_app(port=8502):
    """Start the integrated app in background"""
    print(f"🚀 Starting integrated app on port {port}...")
    
    try:
        # Check if app is already running
        import requests
        response = requests.get(f"http://localhost:{port}", timeout=5)
        if response.status_code == 200:
            print(f"✅ App already running on port {port}")
            return True
    except:
        pass
    
    # Start the app
    try:
        cmd = [
            sys.executable, "run_integrated.py",
            "--port", str(port)
        ]
        
        # Start in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )
        
        # Wait for app to start
        print("⏳ Waiting for app to start...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            try:
                response = requests.get(f"http://localhost:{port}", timeout=2)
                if response.status_code == 200:
                    print(f"✅ App started successfully on port {port}")
                    return True
            except:
                pass
        
        print("❌ App failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start app: {e}")
        return False

def run_tests(app_url="http://localhost:8502", headless=True, verbose=False):
    """Run the functional tests"""
    print("🧪 Running functional tests...")
    
    try:
        from test_integrated_app import IntegratedAppTester
        
        tester = IntegratedAppTester(app_url, headless)
        results = tester.run_all_tests()
        
        return results
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return {"status": "FAIL", "message": str(e)}

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='Run functional tests for integrated app')
    parser.add_argument('--port', type=int, default=8502, help='App port (default: 8502)')
    parser.add_argument('--url', default=None, help='App URL (overrides port)')
    parser.add_argument('--no-headless', action='store_true', help='Run tests with visible browser')
    parser.add_argument('--no-start-app', action='store_true', help='Don\'t start app automatically')
    parser.add_argument('--check-only', action='store_true', help='Only check dependencies')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🧪 Integrated App Test Runner")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_only:
        print("✅ Dependency check completed")
        sys.exit(0)
    
    # Check Chrome
    if not check_chrome():
        print("⚠️  Chrome issues detected, but continuing...")
    
    # Determine app URL
    app_url = args.url or f"http://localhost:{args.port}"
    
    # Start app if needed
    if not args.no_start_app:
        if not start_app(args.port):
            print("❌ Failed to start app")
            sys.exit(1)
    
    # Run tests
    results = run_tests(app_url, not args.no_headless, args.verbose)
    
    # Report results
    if results["status"] == "PASS":
        print("\n🎉 All tests passed! The app is ready for use.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Tests completed with issues.")
        print(f"Passed: {results.get('passed', 0)}")
        print(f"Failed: {results.get('failed', 0)}")
        print(f"Warnings: {results.get('warned', 0)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 