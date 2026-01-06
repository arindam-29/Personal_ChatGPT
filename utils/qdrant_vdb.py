import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from utils.model_loader import ModelLoader

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

class Qdrantclient:
    def __init__(self):
        load_dotenv()
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        if not self.api_key or not self.url:
            raise ValueError("Qdrant API key and URL must be provided in the env file.")

    #     self.client = QdrantClient(
    #         url=self.url,
    #         api_key=self.api_key,
    #     )

    # def create_collection(self, collection_name, vector_size, distance_metric=Distance.COSINE):
    #     self.client.create_collection(
    #         collection_name=collection_name,
    #         vectors_config=VectorParams(size=vector_size, distance=distance_metric),
    #     )

    # def get_client(self):
    #     return self.client

class QdrantVDB:
    def __init__(self, embeddings, collection_name, documents):
        client = Qdrantclient()

        self.url = client.url
        self.api_key = client.api_key
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.documents = documents

    def create_vector_store(self):
        vector_store = QdrantVectorStore.from_documents(
                document=self.documents,
                embeddings=self.embeddings,
                url=self.url,
                prefer_grpc=True,
                api_key=self.api_key,
                collection_name=self.collection_name,
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

    qdrant_vdb = QdrantVDB(embeddings=embeddings,
        collection_name="test_collection",
        documents=[document_1])
    qdrant_vdb.create_vector_store()
    print("Vector store created successfully.")