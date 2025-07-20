# 🧪 Functional Testing Suite

Comprehensive automated testing for the Integrated Image Labeling & Fine-Tuning App using Selenium with headless Chrome.

## 🚀 Quick Start

### **Install Testing Dependencies**
```bash
# Install testing requirements
pip install -r requirements_testing.txt

# Or install manually
pip install selenium requests webdriver-manager
```

### **Run Tests**
```bash
# Quick test run (starts app automatically)
python run_tests.py

# Test with custom port
python run_tests.py --port 8503

# Test without starting app (if already running)
python run_tests.py --no-start-app

# Run with visible browser (for debugging)
python run_tests.py --no-headless

# Check dependencies only
python run_tests.py --check-only
```

### **Direct Test Execution**
```bash
# Run tests directly
python test_integrated_app.py

# Test specific URL
python test_integrated_app.py --url http://localhost:8502

# Check app availability only
python test_integrated_app.py --check-only
```

## 📋 Test Coverage

### **Core Functionality Tests**
- ✅ **App Loading**: Verifies the app starts and loads correctly
- ✅ **Sidebar Elements**: Checks for essential sidebar components
- ✅ **Tab Navigation**: Tests switching between Image Labeling and Fine-Tuning tabs
- ✅ **Image Labeling Tab**: Validates image annotation functionality
- ✅ **Fine-Tuning Tab**: Tests Vertex AI integration elements
- ✅ **AI Model Selection**: Verifies AI model configuration options
- ✅ **GCP Authentication**: Tests Google Cloud authentication elements
- ✅ **Gemini Integration**: Validates Gemini prompt generation features
- ✅ **Vertex AI Integration**: Tests training job and deployment elements
- ✅ **App Responsiveness**: Checks performance and interaction capabilities

### **Test Categories**

#### **UI/UX Tests**
- Page loading and responsiveness
- Element presence and accessibility
- Navigation and tab switching
- Form elements and inputs

#### **Feature Tests**
- Image labeling functionality
- AI model integration
- GCP authentication flow
- Fine-tuning workflow elements

#### **Integration Tests**
- Gemini API integration
- Vertex AI components
- Cloud storage elements
- Model deployment features

## 🛠️ Setup Requirements

### **System Requirements**
- Python 3.8+
- Google Chrome browser
- ChromeDriver (auto-installed by webdriver-manager)

### **Dependencies**
```bash
# Core testing dependencies
selenium>=4.0.0          # Web automation
requests>=2.25.0         # HTTP requests
webdriver-manager>=3.8.0 # Chrome driver management

# Optional enhancements
chromedriver-autoinstaller>=0.6.0  # Automatic ChromeDriver installation
```

### **Chrome Setup**
```bash
# Install Chrome (macOS)
brew install --cask google-chrome

# Install Chrome (Ubuntu)
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list'
sudo apt-get update
sudo apt-get install google-chrome-stable

# Install Chrome (Windows)
# Download from https://www.google.com/chrome/
```

## 🔧 Configuration

### **Test Configuration**
```python
# test_integrated_app.py
class IntegratedAppTester:
    def __init__(self, app_url: str = "http://localhost:8502", headless: bool = True):
        self.app_url = app_url
        self.headless = headless
```

### **Chrome Options**
```python
chrome_options = Options()
chrome_options.add_argument("--headless")           # Run in background
chrome_options.add_argument("--no-sandbox")         # Security bypass
chrome_options.add_argument("--disable-dev-shm-usage")  # Memory optimization
chrome_options.add_argument("--disable-gpu")        # GPU acceleration
chrome_options.add_argument("--window-size=1920,1080")  # Window size
```

### **Environment Variables**
```bash
# Set custom test configuration
export TEST_APP_URL="http://localhost:8502"
export TEST_HEADLESS="true"
export TEST_TIMEOUT="30"
```

## 📊 Test Results

### **Success Criteria**
- ✅ **PASS**: All tests pass, app is ready for use
- ⚠️ **WARN**: Some tests have warnings but app is functional
- ❌ **FAIL**: Critical tests failed, app needs attention

### **Sample Output**
```
🧪 Integrated App Test Runner
==================================================
✅ Testing dependencies available
✅ Chrome driver working
🚀 Starting integrated app on port 8502...
✅ App started successfully on port 8502

🔍 Testing app loading...
✅ App loaded successfully

🔍 Testing sidebar elements...
✅ Sidebar elements present

🔍 Testing tab navigation...
✅ Tab navigation working

📊 TEST SUMMARY
==================================================
✅ Passed: 10/10
⚠️  Warnings: 0/10
❌ Failed: 0/10
🎯 Overall: PASS

🎉 All tests passed! The app is ready for use.
```

### **Detailed Results**
```json
{
  "status": "PASS",
  "total_tests": 10,
  "passed": 10,
  "failed": 0,
  "warned": 0,
  "results": {
    "App Loading": {"status": "PASS", "message": "App loaded successfully"},
    "Sidebar Elements": {"status": "PASS", "message": "Sidebar elements present"},
    "Tab Navigation": {"status": "PASS", "message": "Tab navigation working"}
  }
}
```

## 🔍 Troubleshooting

### **Common Issues**

#### **Chrome Driver Issues**
```bash
# Error: ChromeDriver not found
pip install webdriver-manager
# Or manually install chromedriver
```

#### **Selenium Import Errors**
```bash
# Error: No module named 'selenium'
pip install selenium
```

#### **App Not Starting**
```bash
# Error: App not available
# Check if app is running manually
python run_integrated.py --port 8502

# Check port availability
lsof -i :8502
```

#### **Headless Mode Issues**
```bash
# Run with visible browser for debugging
python run_tests.py --no-headless
```

### **Debug Mode**
```bash
# Enable verbose output
python run_tests.py --verbose

# Run specific test
python test_integrated_app.py --url http://localhost:8502
```

### **Performance Issues**
```bash
# Increase timeout for slow systems
export TEST_TIMEOUT="60"
python run_tests.py

# Run with reduced window size
# Edit test_integrated_app.py chrome_options
```

## 🚀 CI/CD Integration

### **GitHub Actions**
```yaml
# .github/workflows/test.yml
name: Functional Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements_testing.txt
    - name: Install Chrome
      run: |
        sudo apt-get update
        sudo apt-get install -y google-chrome-stable
    - name: Run tests
      run: python run_tests.py
```

### **Local CI**
```bash
# Run tests before commit
#!/bin/bash
# pre-commit hook
python run_tests.py --no-start-app
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix issues before committing."
    exit 1
fi
echo "✅ Tests passed. Proceeding with commit."
```

## 📈 Test Development

### **Adding New Tests**
```python
def test_new_feature(self) -> Dict:
    """Test new feature functionality"""
    print("🔍 Testing new feature...")
    
    try:
        # Test logic here
        element = self.wait_for_element(By.CSS_SELECTOR, "selector")
        if element:
            return {"status": "PASS", "message": "New feature working"}
        else:
            return {"status": "FAIL", "message": "New feature not found"}
    except Exception as e:
        return {"status": "FAIL", "message": str(e)}
```

### **Custom Test Suites**
```python
# Create custom test runner
from test_integrated_app import IntegratedAppTester

class CustomTester(IntegratedAppTester):
    def test_custom_feature(self):
        # Custom test implementation
        pass

# Run custom tests
tester = CustomTester("http://localhost:8502")
results = tester.run_all_tests()
```

### **Test Data Management**
```python
# Create test data
def create_test_images():
    """Create sample images for testing"""
    import PIL.Image
    import numpy as np
    
    for i in range(5):
        img = PIL.Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(f"img_dir/test_image_{i}.jpg")
```

## 🔧 Advanced Configuration

### **Parallel Testing**
```python
# Run tests in parallel
import concurrent.futures

def run_parallel_tests():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_tests, f"http://localhost:{port}")
            for port in [8502, 8503, 8504]
        ]
        results = [future.result() for future in futures]
```

### **Custom Chrome Profiles**
```python
# Use custom Chrome profile
chrome_options.add_argument("--user-data-dir=/path/to/profile")
chrome_options.add_argument("--profile-directory=Default")
```

### **Screenshot Capture**
```python
# Capture screenshots on failure
def capture_screenshot(self, test_name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/{test_name}_{timestamp}.png"
    self.driver.save_screenshot(filename)
    print(f"📸 Screenshot saved: {filename}")
```

## 📚 Best Practices

### **Test Design**
1. **Isolation**: Each test should be independent
2. **Reliability**: Tests should be deterministic
3. **Speed**: Optimize for fast execution
4. **Coverage**: Test all critical paths
5. **Maintainability**: Keep tests simple and readable

### **Error Handling**
```python
# Robust error handling
try:
    element = self.wait_for_element(By.CSS_SELECTOR, "selector", timeout=10)
    if element:
        return {"status": "PASS", "message": "Element found"}
    else:
        return {"status": "FAIL", "message": "Element not found"}
except TimeoutException:
    return {"status": "FAIL", "message": "Timeout waiting for element"}
except Exception as e:
    return {"status": "FAIL", "message": f"Unexpected error: {e}"}
```

### **Performance Optimization**
```python
# Optimize test performance
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-plugins")
chrome_options.add_argument("--disable-images")  # For faster loading
```

## 📞 Support

### **Getting Help**
- Check the troubleshooting section
- Review test output for specific errors
- Run tests with `--verbose` for detailed output
- Use `--no-headless` to see browser interactions

### **Reporting Issues**
When reporting test failures, include:
- Test output with `--verbose` flag
- Chrome version and ChromeDriver version
- Operating system and Python version
- Steps to reproduce the issue

---

**Happy testing! 🧪** 