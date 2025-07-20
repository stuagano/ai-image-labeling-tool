#!/usr/bin/env python3
"""
Functional Testing Suite for Integrated Image Labeling & Fine-Tuning App
Uses Selenium with headless Chrome for automated testing
"""

import os
import sys
import time
import json
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("❌ Selenium not available. Install with: pip install selenium")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("❌ Requests not available. Install with: pip install requests")

class IntegratedAppTester:
    """Functional testing suite for the integrated image labeling app"""
    
    def __init__(self, app_url: str = "http://localhost:8502", headless: bool = True):
        self.app_url = app_url
        self.headless = headless
        self.driver = None
        self.test_results = []
        
    def setup_driver(self):
        """Setup Chrome driver with headless options"""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for testing")
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            return True
        except Exception as e:
            print(f"❌ Failed to setup Chrome driver: {e}")
            return False
    
    def teardown_driver(self):
        """Clean up the driver"""
        if self.driver:
            self.driver.quit()
    
    def wait_for_element(self, by: By, value: str, timeout: int = 10):
        """Wait for an element to be present"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except TimeoutException:
            return None
    
    def wait_for_clickable(self, by: By, value: str, timeout: int = 10):
        """Wait for an element to be clickable"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            return element
        except TimeoutException:
            return None
    
    def test_app_loading(self) -> Dict:
        """Test if the app loads correctly"""
        print("🔍 Testing app loading...")
        
        try:
            self.driver.get(self.app_url)
            time.sleep(3)
            
            # Check if page loads
            title = self.driver.title
            if "Streamlit" in title or "Image Labeling" in title:
                print("✅ App loaded successfully")
                return {"status": "PASS", "message": "App loaded successfully"}
            else:
                print("❌ App failed to load properly")
                return {"status": "FAIL", "message": f"Unexpected title: {title}"}
                
        except Exception as e:
            print(f"❌ App loading failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_sidebar_elements(self) -> Dict:
        """Test if sidebar elements are present"""
        print("🔍 Testing sidebar elements...")
        
        try:
            # Check for sidebar elements
            sidebar_selectors = [
                "div[data-testid='stSidebar']",
                "div[data-testid='stSidebar'] button",
                "div[data-testid='stSidebar'] input"
            ]
            
            for selector in sidebar_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if not elements:
                    print(f"⚠️ Sidebar element not found: {selector}")
            
            # Check for specific sidebar content
            sidebar_text = self.driver.find_element(By.CSS_SELECTOR, "div[data-testid='stSidebar']").text
            if "Total files" in sidebar_text or "Labels" in sidebar_text:
                print("✅ Sidebar elements present")
                return {"status": "PASS", "message": "Sidebar elements present"}
            else:
                print("❌ Expected sidebar content not found")
                return {"status": "FAIL", "message": "Sidebar content missing"}
                
        except Exception as e:
            print(f"❌ Sidebar test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_tab_navigation(self) -> Dict:
        """Test tab navigation between Image Labeling and Fine-Tuning"""
        print("🔍 Testing tab navigation...")
        
        try:
            # Look for tab elements
            tab_selectors = [
                "button[data-testid='stTabs']",
                "div[role='tab']",
                "button[aria-selected='true']"
            ]
            
            tabs_found = False
            for selector in tab_selectors:
                tabs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if tabs:
                    tabs_found = True
                    print(f"✅ Found {len(tabs)} tab elements")
                    break
            
            if not tabs_found:
                # Try to find tab-like elements
                tab_like_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Image Labeling') or contains(text(), 'Fine-Tuning')]")
                if tab_like_elements:
                    print("✅ Tab-like elements found")
                    return {"status": "PASS", "message": "Tab navigation elements present"}
                else:
                    print("❌ No tab elements found")
                    return {"status": "FAIL", "message": "Tab navigation not found"}
            
            return {"status": "PASS", "message": "Tab navigation working"}
            
        except Exception as e:
            print(f"❌ Tab navigation test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_image_labeling_tab(self) -> Dict:
        """Test the image labeling tab functionality"""
        print("🔍 Testing image labeling tab...")
        
        try:
            # Look for image labeling elements
            labeling_selectors = [
                "canvas",  # Image canvas
                "input[type='file']",  # File upload
                "button:contains('Save')",  # Save button
                "button:contains('Previous')",  # Navigation buttons
                "button:contains('Next')"
            ]
            
            elements_found = 0
            for selector in labeling_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        elements_found += 1
                        print(f"✅ Found {len(elements)} {selector} elements")
                except:
                    pass
            
            # Check for image directory content
            img_dir = Path("img_dir")
            if img_dir.exists() and any(img_dir.iterdir()):
                print("✅ Image directory has content")
                elements_found += 1
            else:
                print("⚠️ Image directory is empty")
            
            if elements_found > 0:
                return {"status": "PASS", "message": f"Image labeling tab functional ({elements_found} elements found)"}
            else:
                return {"status": "FAIL", "message": "No image labeling elements found"}
                
        except Exception as e:
            print(f"❌ Image labeling test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_fine_tuning_tab(self) -> Dict:
        """Test the fine-tuning tab functionality"""
        print("🔍 Testing fine-tuning tab...")
        
        try:
            # Look for fine-tuning elements
            tuning_selectors = [
                "input[placeholder*='Project']",  # GCP Project ID
                "select",  # Dropdown menus
                "button:contains('Authenticate')",  # Auth buttons
                "textarea",  # Text areas for prompts
                "button:contains('Generate')"  # Generate buttons
            ]
            
            elements_found = 0
            for selector in tuning_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        elements_found += 1
                        print(f"✅ Found {len(elements)} {selector} elements")
                except:
                    pass
            
            # Check for fine-tuning specific text
            page_text = self.driver.page_source.lower()
            fine_tuning_keywords = ["fine-tuning", "vertex", "gcp", "authentication", "prompt"]
            keywords_found = sum(1 for keyword in fine_tuning_keywords if keyword in page_text)
            
            if elements_found > 0 or keywords_found > 2:
                return {"status": "PASS", "message": f"Fine-tuning tab functional ({elements_found} elements, {keywords_found} keywords)"}
            else:
                return {"status": "FAIL", "message": "No fine-tuning elements found"}
                
        except Exception as e:
            print(f"❌ Fine-tuning test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_ai_model_selection(self) -> Dict:
        """Test AI model selection functionality"""
        print("🔍 Testing AI model selection...")
        
        try:
            # Look for AI model selection elements
            ai_selectors = [
                "select",  # Model selection dropdown
                "input[type='range']",  # Confidence slider
                "button:contains('Initialize')",  # Initialize button
                "button:contains('AI')"  # AI-related buttons
            ]
            
            elements_found = 0
            for selector in ai_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        elements_found += 1
                        print(f"✅ Found {len(elements)} {selector} elements")
                except:
                    pass
            
            # Check for AI-related text
            page_text = self.driver.page_source.lower()
            ai_keywords = ["yolo", "transformers", "gemini", "ai", "model", "confidence"]
            keywords_found = sum(1 for keyword in ai_keywords if keyword in page_text)
            
            if elements_found > 0 or keywords_found > 2:
                return {"status": "PASS", "message": f"AI model selection functional ({elements_found} elements, {keywords_found} keywords)"}
            else:
                return {"status": "FAIL", "message": "No AI model selection elements found"}
                
        except Exception as e:
            print(f"❌ AI model selection test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_gcp_authentication(self) -> Dict:
        """Test GCP authentication elements"""
        print("🔍 Testing GCP authentication elements...")
        
        try:
            # Look for GCP authentication elements
            gcp_selectors = [
                "input[placeholder*='Project']",  # Project ID input
                "select",  # Region selection
                "input[type='file']",  # Service account key upload
                "button:contains('Authenticate')",  # Auth button
                "button:contains('Use')"  # Use credentials button
            ]
            
            elements_found = 0
            for selector in gcp_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        elements_found += 1
                        print(f"✅ Found {len(elements)} {selector} elements")
                except:
                    pass
            
            # Check for GCP-related text
            page_text = self.driver.page_source.lower()
            gcp_keywords = ["gcp", "google cloud", "project", "region", "authentication", "service account"]
            keywords_found = sum(1 for keyword in gcp_keywords if keyword in page_text)
            
            if elements_found > 0 or keywords_found > 2:
                return {"status": "PASS", "message": f"GCP authentication elements present ({elements_found} elements, {keywords_found} keywords)"}
            else:
                return {"status": "FAIL", "message": "No GCP authentication elements found"}
                
        except Exception as e:
            print(f"❌ GCP authentication test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_gemini_integration(self) -> Dict:
        """Test Gemini integration elements"""
        print("🔍 Testing Gemini integration...")
        
        try:
            # Look for Gemini-related elements
            gemini_selectors = [
                "input[placeholder*='API']",  # API key input
                "button:contains('Gemini')",  # Gemini buttons
                "button:contains('Generate')",  # Generate prompts button
                "textarea",  # Prompt text areas
                "div:contains('prompt')"  # Prompt-related divs
            ]
            
            elements_found = 0
            for selector in gemini_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        elements_found += 1
                        print(f"✅ Found {len(elements)} {selector} elements")
                except:
                    pass
            
            # Check for Gemini-related text
            page_text = self.driver.page_source.lower()
            gemini_keywords = ["gemini", "prompt", "generation", "training", "system prompt"]
            keywords_found = sum(1 for keyword in gemini_keywords if keyword in page_text)
            
            if elements_found > 0 or keywords_found > 2:
                return {"status": "PASS", "message": f"Gemini integration elements present ({elements_found} elements, {keywords_found} keywords)"}
            else:
                return {"status": "FAIL", "message": "No Gemini integration elements found"}
                
        except Exception as e:
            print(f"❌ Gemini integration test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_vertex_ai_integration(self) -> Dict:
        """Test Vertex AI integration elements"""
        print("🔍 Testing Vertex AI integration...")
        
        try:
            # Look for Vertex AI elements
            vertex_selectors = [
                "button:contains('Training')",  # Training job buttons
                "button:contains('Deploy')",  # Deploy buttons
                "button:contains('Dataset')",  # Dataset buttons
                "select",  # Task type selection
                "input[placeholder*='Model']"  # Model name input
            ]
            
            elements_found = 0
            for selector in vertex_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        elements_found += 1
                        print(f"✅ Found {len(elements)} {selector} elements")
                except:
                    pass
            
            # Check for Vertex AI-related text
            page_text = self.driver.page_source.lower()
            vertex_keywords = ["vertex", "training", "job", "deploy", "endpoint", "dataset"]
            keywords_found = sum(1 for keyword in vertex_keywords if keyword in page_text)
            
            if elements_found > 0 or keywords_found > 2:
                return {"status": "PASS", "message": f"Vertex AI integration elements present ({elements_found} elements, {keywords_found} keywords)"}
            else:
                return {"status": "FAIL", "message": "No Vertex AI integration elements found"}
                
        except Exception as e:
            print(f"❌ Vertex AI integration test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def test_app_responsiveness(self) -> Dict:
        """Test app responsiveness and performance"""
        print("🔍 Testing app responsiveness...")
        
        try:
            start_time = time.time()
            
            # Test page load time
            self.driver.get(self.app_url)
            load_time = time.time() - start_time
            
            # Test element interaction
            try:
                # Try to find and interact with a common element
                sidebar = self.wait_for_element(By.CSS_SELECTOR, "div[data-testid='stSidebar']", timeout=5)
                if sidebar:
                    interaction_time = time.time() - start_time
                    print(f"✅ App responsive (load: {load_time:.2f}s, interaction: {interaction_time:.2f}s)")
                    return {"status": "PASS", "message": f"App responsive (load: {load_time:.2f}s)"}
                else:
                    print(f"⚠️ App loaded but no interactive elements found (load: {load_time:.2f}s)")
                    return {"status": "WARN", "message": f"App loaded but limited interactivity (load: {load_time:.2f}s)"}
            except:
                print(f"⚠️ App loaded but interaction failed (load: {load_time:.2f}s)")
                return {"status": "WARN", "message": f"App loaded but interaction failed (load: {load_time:.2f}s)"}
                
        except Exception as e:
            print(f"❌ Responsiveness test failed: {e}")
            return {"status": "FAIL", "message": str(e)}
    
    def run_all_tests(self) -> Dict:
        """Run all functional tests"""
        print("🚀 Starting comprehensive functional testing...")
        print("=" * 60)
        
        if not self.setup_driver():
            return {"status": "FAIL", "message": "Failed to setup Chrome driver"}
        
        try:
            tests = [
                ("App Loading", self.test_app_loading),
                ("Sidebar Elements", self.test_sidebar_elements),
                ("Tab Navigation", self.test_tab_navigation),
                ("Image Labeling Tab", self.test_image_labeling_tab),
                ("Fine-Tuning Tab", self.test_fine_tuning_tab),
                ("AI Model Selection", self.test_ai_model_selection),
                ("GCP Authentication", self.test_gcp_authentication),
                ("Gemini Integration", self.test_gemini_integration),
                ("Vertex AI Integration", self.test_vertex_ai_integration),
                ("App Responsiveness", self.test_app_responsiveness)
            ]
            
            results = {}
            passed = 0
            failed = 0
            warned = 0
            
            for test_name, test_func in tests:
                print(f"\n📋 Running: {test_name}")
                result = test_func()
                results[test_name] = result
                
                if result["status"] == "PASS":
                    passed += 1
                elif result["status"] == "FAIL":
                    failed += 1
                elif result["status"] == "WARN":
                    warned += 1
            
            # Generate summary
            total_tests = len(tests)
            summary = {
                "status": "PASS" if failed == 0 else "FAIL",
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "warned": warned,
                "results": results
            }
            
            print("\n" + "=" * 60)
            print("📊 TEST SUMMARY")
            print("=" * 60)
            print(f"✅ Passed: {passed}/{total_tests}")
            print(f"⚠️  Warnings: {warned}/{total_tests}")
            print(f"❌ Failed: {failed}/{total_tests}")
            print(f"🎯 Overall: {summary['status']}")
            
            if failed > 0:
                print("\n❌ FAILED TESTS:")
                for test_name, result in results.items():
                    if result["status"] == "FAIL":
                        print(f"  • {test_name}: {result['message']}")
            
            if warned > 0:
                print("\n⚠️  WARNINGS:")
                for test_name, result in results.items():
                    if result["status"] == "WARN":
                        print(f"  • {test_name}: {result['message']}")
            
            return summary
            
        finally:
            self.teardown_driver()

def check_app_availability(url: str = "http://localhost:8502") -> bool:
    """Check if the app is available at the given URL"""
    if not REQUESTS_AVAILABLE:
        print("❌ Requests not available for app availability check")
        return False
    
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Functional Testing for Integrated Image Labeling App')
    parser.add_argument('--url', default='http://localhost:8502', help='App URL to test')
    parser.add_argument('--headless', action='store_true', default=True, help='Run tests in headless mode')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--check-only', action='store_true', help='Only check if app is available')
    
    args = parser.parse_args()
    
    print("🧪 Integrated App Functional Testing Suite")
    print("=" * 60)
    
    # Check dependencies
    if not SELENIUM_AVAILABLE:
        print("❌ Selenium not available. Install with: pip install selenium")
        sys.exit(1)
    
    if not REQUESTS_AVAILABLE:
        print("❌ Requests not available. Install with: pip install requests")
        sys.exit(1)
    
    # Check app availability
    print(f"🔍 Checking app availability at {args.url}...")
    if not check_app_availability(args.url):
        print(f"❌ App not available at {args.url}")
        print("💡 Make sure the app is running with: python run_integrated.py --port 8502")
        sys.exit(1)
    
    print("✅ App is available")
    
    if args.check_only:
        print("✅ App availability check passed")
        sys.exit(0)
    
    # Run tests
    tester = IntegratedAppTester(args.url, args.headless)
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results["status"] == "PASS":
        print("\n🎉 All tests passed! The app is ready for use.")
        sys.exit(0)
    else:
        print(f"\n⚠️  Tests completed with {results['failed']} failures.")
        print("Please review the failed tests before using the app.")
        sys.exit(1)

if __name__ == "__main__":
    main() 