from dotenv import load_dotenv
import os
import json, sys
from exception.custom_exception import ProjectCustomException
from logger import GLOBAL_LOGGER as logger

class APIKeyManager:
    
    def __init__(self, REQUIRED_KEYS):
        self.REQUIRED_KEYS = REQUIRED_KEYS
        self.api_keys = {}
        
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            logger.info("Running in LOCAL mode, loading environment variables from .env file")
        else:
            logger.info("Running in PRODUCTION mode")
        
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    logger.info(f"Loaded {key} from individual env var")

        # Final check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            logger.error("Missing required API keys", missing_keys=missing)
            raise ProjectCustomException("Missing API keys", sys)

        logger.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})
    
    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key for {key} is missing")
        return val
    
if __name__ == "__main__":
    # REQUIRED_KEYS = ['GOOGLE_API_KEY', 'OPENAI_API_KEY', 'GROQ_API_KEY', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION']
    REQUIRED_KEYS = ['AWS_ACCESS_KEY_ID']
    manager = APIKeyManager(REQUIRED_KEYS)
    