"""
Test script to verify your Google Gemini API key works
Run this BEFORE using the main app to debug API issues
"""

def test_gemini_api():
    """Test Google Gemini API key"""
    
    print("=" * 60)
    print("🔍 GOOGLE GEMINI API KEY TEST")
    print("=" * 60)
    print()
    
    # Step 1: Check if library is installed
    print("Step 1: Checking if google-generativeai is installed...")
    try:
        import google.generativeai as genai
        print("✅ google-generativeai is installed")
    except ImportError:
        print("❌ google-generativeai NOT installed")
        print("   Run: pip install google-generativeai")
        return False
    
    print()
    
    # Step 2: Get API key
    print("Step 2: Getting API key...")
    api_key = input("Paste your Google Gemini API key here: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    if len(api_key) < 20:
        print("⚠️  Warning: API key seems too short")
        print(f"   Length: {len(api_key)} characters")
    
    print(f"✅ API key received ({len(api_key)} characters)")
    print(f"   Starts with: {api_key[:4]}...")
    print(f"   Ends with: ...{api_key[-4:]}")
    print()
    
    # Step 3: Configure API
    print("Step 3: Configuring API...")
    try:
        genai.configure(api_key=api_key)
        print("✅ API configured successfully")
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    print()
    
    # Step 4: Test models
    print("Step 4: Testing models...")
    
    models_to_test = [
        "gemini-pro",
        "gemini-1.5-flash", 
        "gemini-1.5-pro"
    ]
    
    working_model = None
    
    for model_name in models_to_test:
        print(f"\n   Testing {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Hello'")
            
            if response and response.text:
                print(f"   ✅ {model_name} WORKS!")
                print(f"      Response: {response.text[:50]}")
                working_model = model_name
                break
            else:
                print(f"   ⚠️  {model_name} - Empty response")
                
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ {model_name} failed")
            
            if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
                print(f"      Error: Invalid API key")
            elif "404" in error_msg:
                print(f"      Error: Model not found")
            elif "403" in error_msg or "permission" in error_msg.lower():
                print(f"      Error: Permission denied")
            elif "429" in error_msg or "quota" in error_msg.lower():
                print(f"      Error: Quota exceeded")
            else:
                print(f"      Error: {error_msg[:100]}")
    
    print()
    print("=" * 60)
    
    # Final result
    if working_model:
        print("✅ SUCCESS! Your API key works!")
        print(f"   Working model: {working_model}")
        print()
        print("You can now use this API key in ALOC Settings.")
        return True
    else:
        print("❌ FAILED! API key is not working properly")
        print()
        print("Common solutions:")
        print("1. Get a fresh API key from: https://aistudio.google.com/app/apikey")
        print("2. Enable 'Generative Language API' in Google Cloud Console")
        print("3. Wait 2-3 minutes after creating/enabling")
        print("4. Try a different Google account")
        return False


if __name__ == "__main__":
    test_gemini_api()
    
    print()
    input("Press Enter to exit...")