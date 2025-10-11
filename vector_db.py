from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging

logger = logging.getLogger(__name__)

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection='docs', dim=3072):
        try:
            self.client = QdrantClient(url=url, timeout=30)
            self.collection = collection
            self.dim = dim
            
            # Test connection and create collection if needed
            if not self.client.collection_exists(self.collection):
                logger.info(f"Creating collection '{self.collection}' with dimension {dim}")
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
            else:
                logger.info(f"Using existing collection '{self.collection}'")
                
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {url}: {e}")
            raise RuntimeError(f"Qdrant connection failed. Make sure Qdrant server is running at {url}. Error: {e}")
    def upsert(self, ids, vectors, payloads):
        try:
            points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Successfully upserted {len(points)} points to collection '{self.collection}'")
        except Exception as e:
            logger.error(f"Failed to upsert to Qdrant: {e}")
            raise RuntimeError(f"Qdrant upsert failed: {e}")

    def search(self, query_vector, top_k: int = 5):
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                with_payload=True,
                limit=top_k,
            )
            contexts = []
            sources = set()

            for r in results:
                payload = getattr(r, "payload", None) or {}
                text = payload.get("text", "")
                source = payload.get("source", "")
                if text:
                    contexts.append(text)
                    sources.add(source)
            
            return {"contexts": contexts, "sources": list(sources)}
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            raise RuntimeError(f"Qdrant search failed: {e}")
    
    def delete_collection(self):
        """Delete the collection. Useful for dimension mismatches."""
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(self.collection)
                logger.info(f"Deleted collection '{self.collection}'")
            else:
                logger.info(f"Collection '{self.collection}' does not exist")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise RuntimeError(f"Collection deletion failed: {e}")