import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from utils.APIKey_loader import APIKeyManager
from utils.model_loader import ModelLoader

class QdrantVDB:
    def __init__(self):
        api_key_mgr = APIKeyManager(['QDRANT_API_KEY', 'QDRANT_URL'])
        self.url = api_key_mgr.get("QDRANT_URL")
        self.api_key = api_key_mgr.get("QDRANT_API_KEY")
        if not self.api_key or not self.url:
            raise ValueError("Qdrant API key and URL must be provided in the env file.")

    def create_vector_store(self, embedding, collection_name, documents):
        vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embedding,
                url=self.url,
                prefer_grpc=True,
                api_key=self.api_key,
                collection_name=collection_name,
            )
        return vector_store
    
    def get_vector_store(self, embedding, collection_name):
        vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embedding,
                url=self.url,
                prefer_grpc=True,
                api_key=self.api_key,
                collection_name=collection_name,
            )
        return vector_store


if __name__ == "__main__":
    loader = ModelLoader()
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    from langchain_core.documents import Document
    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    )

    qdrant_vdb = QdrantVDB()
    qdrant_vdb.create_vector_store(embedding=embeddings,
        collection_name="test_collection",
        documents=[document_1])
    print("Vector store created successfully.")


    qdrant_vdb.get_vector_store(embedding=embeddings,
        collection_name="test_collection")
    print("Vector store retrieved successfully.")