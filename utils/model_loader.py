import os
import sys
import json
from dotenv import load_dotenv
from exception.custom_exception import ProjectCustomException
from logger import GLOBAL_LOGGER as logger
from utils.config_loader import load_config

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq

class APIKeyManager:
    REQUIRED_KEYS = ['GOOGLE_API_KEY', 'OPENAI_API_KEY', 'GROQ_API_KEY']

    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("API_KEYS")

        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS is not a valid JSON object")
                self.api_keys = parsed
                logger.info("Loaded API_KEYS from ECS secret")
            except Exception as e:
                logger.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # Fallback to individual env vars
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

class ModelLoader:
    """
    Loads embedding models and LLMs based on config and environment.
    """

    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            logger.info("Running in LOCAL using .env file")
        else:
            logger.info("Running in PRODUCTION mode")

        self.api_key_mgr = APIKeyManager()
        self.config = load_config()
        logger.info("YAML config loaded", config_keys=list(self.config.keys()))
    
    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """

        provider = self.config.get("providers").get("embedding")
        embedding_block = self.config["embedding_model"]
       
        if provider not in embedding_block:
            logger.error("Embedding provider not found in config", provider=provider)
            raise ValueError(f"Embedding provider '{provider}' not found in config")
        
        embedding_config = embedding_block[provider]
        model_name = embedding_config.get("model_name")
        logger.info("Loading embedding model", provider=provider, model=model_name)

        if provider == "google":
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY")
            )
        elif provider == "openai":
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=self.api_key_mgr.get("OPENAI_API_KEY")
            )
        else:
            logger.error("Unsupported embedding provider", provider=provider)
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """
        provider = self.config.get("providers").get("llm")
        llm_block = self.config["llm"]

        if provider not in llm_block:
            logger.error("LLM provider not found in config", provider=provider)
            raise ValueError(f"LLM provider '{provider}' not found in config")

        llm_config = llm_block[provider]
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        logger.info("Loading LLM", provider=provider, model=model_name)
        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY"),
                temperature=temperature,
                max_output_tokens=max_tokens
            )

        elif provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                temperature=temperature,
            )

        elif provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=self.api_key_mgr.get("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=max_tokens
            )

        else:
            logger.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    # print(f"Embedding Model Loaded: {embeddings}")
    # result = embeddings.embed_query("Hello, how are you?")
    # print(f"Embedding Result: {result}")

    # Test LLM
    # llm = loader.load_llm()
    # print(f"LLM Loaded: {llm}")
    # result = llm.invoke("Hello, how are you?")
    # print(f"LLM Result: {result.content}")

    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    ### LOCAL QDRANT TESTING

    # client = QdrantClient(path="langchain_qdrant")
    # client.create_collection(
    #     collection_name="demo_collection",
    #     vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    # )
    # vector_store = QdrantVectorStore(
    #     client=client,
    #     collection_name="demo_collection",
    #     embedding=embeddings,
    # )
    # client.close()


    ### CLOUD QDRANT TESTING

    # url = os.getenv("QDRANT_URL")
    # api_key = os.getenv("QDRANT_API_KEY")
    # from langchain_core.documents import Document
    # document_1 = Document(
    #     page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    #     metadata={"source": "tweet"},
    # )
    # docs = [document_1] 
    # qdrant = QdrantVectorStore.from_documents(
    #     docs,
    #     embeddings,
    #     url=url,
    #     prefer_grpc=True,
    #     api_key=api_key,
    #     collection_name="my_documents",
    # )