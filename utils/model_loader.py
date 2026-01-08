import os
import sys

# ADD PROJECT ROOT TO PATH FIRST
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException

# Create logger for this module
log = CustomLogger().get_logger("model_loader")


class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY"]

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Fallback to individual env vars
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise DocumentPortalException("Missing API keys", sys)

        log.info("API keys loaded successfully", num_keys=len(self.api_keys))


    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            log.error("API key not found", key=key)
            raise KeyError(f"API key for {key} is missing")
        return val


class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        log.info("Initializing ModelLoader")
        
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("Running in LOCAL mode: .env loaded")
        else:
            log.info("Running in PRODUCTION mode")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        """
        Load and return embedding model from HuggingFace.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            log.info("Embedding model loaded successfully", model=model_name)
            return embeddings
            
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "groq")
        
        log.info("Attempting to load LLM", provider=provider_key)

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key, available_providers=list(llm_block.keys()))
            raise ValueError(f"LLM provider '{provider_key}' not found in config")

        llm_config = llm_block[provider_key]
        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name, temperature=temperature)

        if provider == "groq":
            try:
                llm = ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                log.info("LLM loaded successfully", provider=provider, model=model_name)
                return llm
            except Exception as e:
                log.error("Failed to initialize Groq LLM", error=str(e))
                raise
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    try:
        log.info("=" * 50)
        log.info("Starting model loader tests")
        log.info("=" * 50)
        
        loader = ModelLoader()

        # Test Embedding
        print("\n=== Testing Embeddings ===")
        log.info("Starting embedding test")
        embeddings = loader.load_embeddings()
        print(f"✓ Embedding Model Loaded: {type(embeddings).__name__}")
        
        result = embeddings.embed_query("Hello, how are you?")
        print(f"✓ Embedding Result (first 10 values): {result[:10]}")
        print(f"✓ Embedding Dimension: {len(result)}")
        log.info("Embedding test completed successfully", dimension=len(result))

        # Test LLM
        print("\n=== Testing LLM ===")
        log.info("Starting LLM test")
        llm = loader.load_llm()
        print(f"✓ LLM Loaded: {type(llm).__name__}")
        
        response = llm.invoke("Hello, how are you?")
        print(f"✓ LLM Result: {response.content}")
        log.info("LLM test completed successfully", response_length=len(response.content))
        
        log.info("All tests completed successfully")
        
    except Exception as e:
        log.error("Error in model loader tests", error=str(e), error_type=type(e).__name__)
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()