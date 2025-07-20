# 🔐 Google OAuth Authentication Setup

This guide will help you set up Google OAuth 2.0 authentication for the Integrated Image Labeling & Fine-Tuning App.

## 🚀 Quick Setup

### **1. Install OAuth Dependencies**
```bash
pip install google-auth-oauthlib google-api-python-client
```

### **2. Set Up Google Cloud OAuth Credentials**

#### **Step 1: Go to Google Cloud Console**
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project or create a new one
3. Enable the APIs you need:
   - Vertex AI API
   - Cloud Storage API
   - OAuth 2.0 API

#### **Step 2: Create OAuth 2.0 Credentials**
1. Go to **APIs & Services** > **Credentials**
2. Click **+ CREATE CREDENTIALS** > **OAuth 2.0 Client IDs**
3. Choose **Desktop application** as the application type
4. Give it a name (e.g., "Image Labeling App")
5. Click **Create**

#### **Step 3: Download Credentials**
1. After creating, click **Download JSON**
2. Rename the downloaded file to `client_secrets.json`
3. Place it in your project root directory (same level as `app_integrated.py`)

### **3. Configure OAuth Scopes**

The app uses these OAuth scopes:
- `https://www.googleapis.com/auth/cloud-platform` - Full access to Google Cloud
- `https://www.googleapis.com/auth/cloud-platform.projects` - Project access
- `https://www.googleapis.com/auth/cloud-platform.read-only` - Read-only access

## 🔧 Detailed Setup Instructions

### **Google Cloud Console Setup**

#### **1. Enable Required APIs**
```bash
# Enable these APIs in your Google Cloud project:
# - Vertex AI API
# - Cloud Storage API
# - OAuth 2.0 API
# - People API (for user info)
```

#### **2. Create OAuth Consent Screen**
1. Go to **APIs & Services** > **OAuth consent screen**
2. Choose **External** user type
3. Fill in required information:
   - App name: "Image Labeling & Fine-Tuning Tool"
   - User support email: Your email
   - Developer contact information: Your email
4. Add scopes:
   - `https://www.googleapis.com/auth/cloud-platform`
   - `https://www.googleapis.com/auth/userinfo.email`
   - `https://www.googleapis.com/auth/userinfo.profile`
5. Add test users (your email)
6. Save and continue

#### **3. Create OAuth 2.0 Client ID**
1. Go to **APIs & Services** > **Credentials**
2. Click **+ CREATE CREDENTIALS** > **OAuth 2.0 Client IDs**
3. Application type: **Desktop application**
4. Name: "Image Labeling App"
5. Click **Create**

#### **4. Download and Configure**
1. Download the JSON file
2. Rename to `client_secrets.json`
3. Place in project root
4. Update the file with your actual values:

```json
{
  "installed": {
    "client_id": "YOUR_ACTUAL_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-actual-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_ACTUAL_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
```

## 🎯 Using OAuth Authentication

### **1. Start the App**
```bash
python run_integrated.py --port 8502
```

### **2. Authenticate**
1. Open the app in your browser
2. In the left sidebar, select **"Google Account (OAuth 2.0)"**
3. Click **"🔐 Sign in with Google"**
4. A browser window will open for Google authentication
5. Sign in with your Google account
6. Grant the requested permissions
7. Return to the app - you should see "✅ Authenticated as: your-email@domain.com"

### **3. Use GCP Services**
Once authenticated, you can:
- Access Vertex AI services
- Create training jobs
- Deploy models
- Use Cloud Storage
- Access all GCP resources within your project

## 🔒 Security Best Practices

### **1. Credential Management**
- Never commit `client_secrets.json` to version control
- Add `client_secrets.json` to `.gitignore`
- Use environment variables for production

### **2. OAuth Consent Screen**
- Keep the consent screen in testing mode for development
- Add only necessary scopes
- Use test users for development

### **3. Production Deployment**
- Move to production OAuth consent screen
- Add proper privacy policy and terms of service
- Verify your app with Google

## 🐛 Troubleshooting

### **Common Issues**

#### **"client_secrets.json not found"**
- Ensure the file is in the project root directory
- Check the file name is exactly `client_secrets.json`
- Verify the JSON format is correct

#### **"OAuth not available"**
```bash
pip install google-auth-oauthlib google-api-python-client
```

#### **"Invalid client" error**
- Check that your OAuth client ID is correct
- Ensure the OAuth consent screen is configured
- Verify you're using the right project

#### **"Access denied" error**
- Add your email as a test user in OAuth consent screen
- Check that required APIs are enabled
- Verify project permissions

#### **Browser authentication issues**
- Ensure pop-ups are allowed
- Check that localhost is in redirect URIs
- Try using a different browser

### **Debug Mode**
```python
# Add to app_integrated.py for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📋 File Structure

```
streamlit-img-label/
├── app_integrated.py
├── client_secrets.json          # Your OAuth credentials
├── client_secrets.json.example  # Example file
├── requirements.txt
├── OAUTH_SETUP.md              # This guide
└── .gitignore                  # Should include client_secrets.json
```

## 🔄 OAuth Flow

1. **User clicks "Sign in with Google"**
2. **App opens browser for Google authentication**
3. **User signs in and grants permissions**
4. **Google redirects back to app with authorization code**
5. **App exchanges code for access token**
6. **App stores credentials in session state**
7. **User is authenticated for GCP services**

## 🎉 Success Indicators

When OAuth is working correctly, you should see:
- ✅ "Authenticated as: your-email@domain.com"
- ✅ "Vertex AI Manager initialized!"
- ✅ Access to all GCP features in the app
- ✅ No authentication errors in the console

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your Google Cloud Console setup
3. Ensure all dependencies are installed
4. Check the app logs for detailed error messages

---

**Happy authenticating! 🔐** 