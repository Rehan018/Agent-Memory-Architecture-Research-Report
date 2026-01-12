import chromadb
import uuid
import time
from typing import List, Dict, Any
from datetime import datetime
from agent import MemoryInterface, Message
from llm import LLMClient
from memory_episodic import EpisodicMemory, Episode

class SemanticMemory(EpisodicMemory):
    """
    Architecture C: STM + Episodic + Semantic.
    Extends EpisodicMemory but adds a Vector DB layer for semantic retrieval.
    """
    def __init__(self, llm_client: LLMClient, stm_size: int = 5, file_path: str = "episodic_memory.json", db_path: str = "./chroma_db"):
        super().__init__(llm_client, stm_size, file_path)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="agent_memory")
        
    def add_message(self, message: Message):
        # 1. Standard STM + Episodic processing
        super().add_message(message)
        
        # 2. Semantic Processing
        # In a real system, we'd extract "facts" here. 
        # For this research prototype, we'll embed the raw message if it's from the user, 
        # acting as a "knowledge/preference" capture.
        if message.role == "user":
            self._store_semantic(message)

    def get_context(self, current_query: str = None) -> List[Message]:
        relevant_context = []
        
        # 1. Semantic Retrieval (Vector DB)
        if current_query:
            semantic_results = self._query_semantic(current_query)
            for res in semantic_results:
                relevant_context.append(Message(
                    role="system",
                    content=f"[Semantic Memory]: {res}",
                    metadata={"type": "semantic"}
                ))
                
        # 2. Episodic Retrieval (Scoring based) - from Parent
        episodic_context = super().get_context(current_query)
        
        # 3. Merge (Avoid duplicates is tricky, but for now simple append)
        # episodic_context includes STM at the end.
        
        # We put Semantic First, then Episodic, then STM.
        # But 'episodic_context' has (Episodes + STM).
        # So we prepend Semantic to that.
        return relevant_context + episodic_context

    def _store_semantic(self, message: Message):
        # Allow searching by content
        # ID must be unique
        msg_id = str(uuid.uuid4())
        self.collection.add(
            documents=[message.content],
            metadatas=[{"role": message.role, "timestamp": message.timestamp.isoformat()}],
            ids=[msg_id]
        )

    def _query_semantic(self, query: str, n_results: int = 2) -> List[str]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            # results['documents'] is a list of lists [[doc1, doc2]]
            if results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            print(f"Vector Scan Error: {e}")
            return []
    
    def clear(self):
        super().clear()
        try:
            self.chroma_client.delete_collection("agent_memory")
        except:
            pass
