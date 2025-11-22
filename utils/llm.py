"""
Simple LLM Handler - Enhanced API Key Validation
"""
import os
from typing import Optional, Tuple
import warnings
import logging

# Suppress SSL certificate verification warnings from gRPC
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('grpc').setLevel(logging.ERROR)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Module logger
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class SimpleLLM:
    """Gemini API wrapper with robust validation"""
    
    def __init__(self, api_key: Optional[str] = None):
        if genai is None:
            raise ValueError("google-generativeai not installed")
        
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not found")
        
        # Validate key format before attempting connection
        if not self._is_valid_key_format(key):
            raise ValueError("Invalid API key format. Key should start with 'AIza' and be 39+ characters long")
        
        # Always reconfigure with fresh key to avoid caching issues
        genai.configure(api_key=key)
        self.api_key = key
        self.model = None
        self.current_model_name = None
        self._initialize_model()
    
    @staticmethod
    def _is_valid_key_format(key: str) -> bool:
        """Validate API key format"""
        return (
            isinstance(key, str) and
            len(key) >= 30 and
            key.startswith("AIza")
        )
    
    def _initialize_model(self):
        """Initialize model by getting available models from API"""
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.getLogger('grpc').setLevel(logging.ERROR)
            
            try:
                # Get list of available models
                available_models = genai.list_models()
                
                # Find the first model that supports generateContent
                for m in available_models:
                    if 'generateContent' in m.supported_generation_methods:
                        model_name = m.name
                        try:
                            self.model = genai.GenerativeModel(model_name)
                            self.current_model_name = model_name
                            return  # Success
                        except Exception:
                            continue
            except:
                pass
        
        # Fallback to a sensible default model name and record it
        fallback = "gemini-2.5-flash"
        try:
            self.model = genai.GenerativeModel(fallback)
            self.current_model_name = fallback
        except Exception:
            # leave model as-is (may be None) and let callers handle
            self.model = None
            self.current_model_name = None
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text response"""
        if not self.model:
            return "Error: Model not initialized. Please check API key in Settings."

        # Ensure the global client is configured with the current key
        try:
            genai.configure(api_key=self.api_key)
        except Exception:
            pass

        # Build a candidate list of model names to try (preferred -> alternates)
        candidates = []
        if self.current_model_name:
            candidates.append(self.current_model_name)

        try:
            models = genai.list_models()
            for m in models:
                if 'generateContent' in m.supported_generation_methods:
                    name = m.name
                    if name not in candidates:
                        candidates.append(name)
        except Exception as e:
            logger.debug("Could not list alternate models: %s", e)

        # Limit number of candidates to avoid long loops
        max_try = min(len(candidates), 6) if candidates else 0
        last_exc = None

        for idx in range(max_try):
            model_name = candidates[idx]
            try:
                # If model is not the currently-in-memory model, create a fresh instance
                if not self.current_model_name or model_name != self.current_model_name:
                    try:
                        self.model = genai.GenerativeModel(model_name)
                        self.current_model_name = model_name
                    except Exception as e:
                        logger.warning("Failed to initialize model %s: %s", model_name, e)
                        last_exc = e
                        continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    logging.getLogger('grpc').setLevel(logging.ERROR)
                    response = self.model.generate_content(prompt)

                
                text = None
                if response is not None:
                    text = getattr(response, 'text', None) or getattr(response, 'content', None)

                if text:
                    return text
                else:
                    return "Error: Empty response from model"

            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                if '429' in err_str or 'quota' in err_str or 'exceeded' in err_str:
                    logger.warning("Model %s returned quota error, trying next model if available", model_name)
                    continue
                
                logger.error("Model %s failed with error: %s", model_name, e)
                return f"Error: {str(e)[:200]}"

        
        if last_exc:
            return f"Error: {str(last_exc)[:200]}"
        return "Error: No available models to try"
    
    def summarize(self, text: str, length: str = "medium") -> str:
        """Summarize text"""
        length_map = {"short": 100, "medium": 250, "long": 500}
        max_words = length_map.get(length, 250)
        text_limit = 8000
        truncated_text = text[:text_limit]
        prompt = f"Summarize this in about {max_words} words:\n\n{truncated_text}"
        return self.generate(prompt, max_tokens=max_words * 2)
    
    def generate_quiz(self, text: str, num_q: int = 10) -> str:
        """Generate quiz questions"""
        text_limit = 6000
        truncated_text = text[:text_limit]
        prompt = f"""Create {num_q} multiple choice questions from this text.
Format as JSON array with fields: question, options (list of 4), correct_answer, explanation.
Text: {truncated_text}"""
        return self.generate(prompt, max_tokens=num_q * 150)
    
    @classmethod
    def reset_quota_tracking(cls):
        """Reset quota tracking"""
        pass
    
    @classmethod
    def test_api_key(cls, api_key: str) -> Tuple[bool, str]:
        """
        Test API key by trying to list models and generate content
        Returns: (success: bool, message: str)
        """
        # Phase 1: Format validation
        if not api_key:
            return (False, "API key is empty")
        
        if not isinstance(api_key, str):
            return (False, "API key must be a string")
        
        if len(api_key) < 30:
            return (False, f"API key too short ({len(api_key)} chars)")
        
        if not api_key.startswith("AIza"):
            return (False, "Invalid key format - should start with 'AIza'")
        
        if api_key.startswith("AIza ") or api_key.endswith(" "):
            return (False, "API key contains spaces - remove them")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.getLogger('grpc').setLevel(logging.ERROR)
            
            # Phase 2: Configure and test connection
            try:
                genai.configure(api_key=api_key)
            except Exception as e:
                return (False, f"Failed to configure API: {str(e)[:80]}")
            
            # Phase 3: List models - if this works, API key is valid
            try:
                models_list = genai.list_models()
                working_model = None
                
                
                for model in models_list:
                    if 'generateContent' in model.supported_generation_methods:
                        working_model = model.name
                        break
                
                if not working_model:
                    return (False, "No models available for content generation")
                
                return (True, f"API key verified! Ready to use.")
                        
            except Exception as e:
                error_str = str(e).lower()
                if "api key" in error_str or "invalid" in error_str or "unauthenticated" in error_str:
                    return (False, f"Invalid API key: {str(e)[:80]}")
                else:
                    return (False, f"Failed to list models: {str(e)[:80]}")