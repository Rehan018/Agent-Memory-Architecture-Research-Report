import json
import time
import math
import os
import uuid
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
from agent import MemoryInterface, Message
from llm import LLMClient
@dataclass
class Episode:
    id: str
    content: str  # Summary of the interaction
    keywords: List[str]
    timestamp: float  # Unix timestamp
    metadata: Dict[str, Any]

class EpisodicMemory(MemoryInterface):
    """
    Architecture B: STM + Episodic Memory.
    - STM: Context Window (recent messages).
    - Episodic: Long-term summaries with scoring.
    """
    def __init__(self, llm_client: LLMClient, stm_size: int = 5, file_path: str = "episodic_memory.json"):
        self.llm_client = llm_client
        self.stm_window: List[Message] = []
        self.stm_limit = stm_size
        self.episodes: List[Episode] = []
        self.file_path = file_path
        self.load_memory()

    def add_message(self, message: Message):
        self.stm_window.append(message)
        
        # Check if we need to consolidate STM into Episodic
        # For simplicity, let's say after every N*2 turns, we summarize the oldest N messages
        if len(self.stm_window) > self.stm_limit * 2:
            self._consolidate_memory()
            
    def get_context(self, current_query: str = None) -> List[Message]:
        # 1. Get relevant episodes
        relevant_context = []
        if current_query:
            top_episodes = self._retrieve_episodes(current_query)
            for ep in top_episodes:
                # Format episode as a system message or special context message
                relevant_context.append(Message(
                    role="system", 
                    content=f"[Memory from {datetime.fromtimestamp(ep.timestamp)}]: {ep.content}",
                    metadata={"type": "episodic"}
                ))
        
        # 2. Append STM (Recent conversation)
        return relevant_context + self.stm_window[-self.stm_limit:]

    def _consolidate_memory(self):
        """Moves older messages from STM to a summarized Episode."""
        # Allow keeping last 'stm_limit' messages, summarize the rest
        messages_to_summarize = self.stm_window[:-self.stm_limit]
        self.stm_window = self.stm_window[-self.stm_limit:] # Keep recent
        
        if not messages_to_summarize:
            return

        text_block = "\n".join([f"{m.role}: {m.content}" for m in messages_to_summarize])
        
        # TODO: Use LLM to summarize. For now, we mock summary or use crude extraction.
        # summary = self.llm_client.generate_response(...)
        summary = f"Interaction loop where user said: {[m.content for m in messages_to_summarize if m.role == 'user']}"
        keywords = [w for w in text_block.split() if len(w) > 4] # Crude keyword extraction
        
        episode = Episode(
            id=str(uuid.uuid4()),
            content=summary,
            keywords=keywords,
            timestamp=time.time(),
            metadata={}
        )
        self.episodes.append(episode)
        self.save_memory()

    def _retrieve_episodes(self, query: str, top_k: int = 3) -> List[Episode]:
        """
        Scoring Logic:
        Score = (Keyword Match * 0.5) + (Recency Boost * 0.3) + (Decay Penalty)
        """
        query_words = set(query.lower().split())
        scored_episodes = []
        current_time = time.time()
        
        for ep in self.episodes:
            # 1. Keyword Score
            ep_keywords = set([k.lower() for k in ep.keywords])
            match_count = len(query_words.intersection(ep_keywords))
            keyword_score = match_count / (len(query_words) + 1) # Normalize roughly
            
            # 2. Recency / Decay
            # Simple exponential decay: e^(-delta_time / lambda)
            hours_passed = (current_time - ep.timestamp) / 3600
            decay_score = math.exp(-hours_passed / 24) # Decays over days
            
            # Total Score
            final_score = (keyword_score * 0.7) + (decay_score * 0.3)
            scored_episodes.append((final_score, ep))
            
        # Sort and return top K
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored_episodes[:top_k]]

    def save_memory(self):
        # Serialization logic
        data = [{
            "id": ep.id, 
            "content": ep.content, 
            "keywords": ep.keywords,
            "timestamp": ep.timestamp,
            "metadata": ep.metadata
        } for ep in self.episodes]
        with open(self.file_path, 'w') as f:
            json.dump(data, f)

    def load_memory(self):
        if not os.path.exists(self.file_path):
            return
        
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                self.episodes = [Episode(**item) for item in data]
        except Exception as e:
            print(f"Error loading memory: {e}")

    def clear(self):
        self.stm_window = []
        self.episodes = []
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
