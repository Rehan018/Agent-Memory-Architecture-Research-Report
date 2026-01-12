from typing import List, Dict, Any
import uuid
import time
from datetime import datetime
from agent import MemoryInterface, Message
from llm import LLMClient
from memory_semantic import SemanticMemory

class ReflectionMemory(SemanticMemory):
    """
    Architecture D: STM + Episodic + Semantic + Reflection.
    Adds a 'Reflector' that analyzes past interactions to create 'Lessons Learned'.
    These lessons are retrieved and injected as high-priority System Prompts.
    """
    def __init__(self, llm_client: LLMClient, stm_size: int = 5, file_path: str = "episodic_memory.json", db_path: str = "./chroma_db"):
        super().__init__(llm_client, stm_size, file_path, db_path)
        self.reflections: List[str] = [] 
        # In a real app, reflections should be stored solely in a dedicated VectorDB collection or file.
        # For simplicity, we'll store them in memory + append to a file.
        self.reflection_file = "reflections.txt"
        self._load_reflections()

    def add_message(self, message: Message):
        super().add_message(message)
        # Trigger reflection logic could go here (e.g. after every N turns or on 'error')
        # For this research, we will manually call reflect() from the agent/test loop.

    def get_context(self, current_query: str = None) -> List[Message]:
        # 1. Get Base Context (Semantic + Episodic + STM)
        base_context = super().get_context(current_query)
        
        # 2. Get Relevant Reflections
        # We want to check if any "lesson" applies to the current situation.
        # Ideally, we verify against the current query using Vector Search on the reflections.
        # For this prototype, we'll inject ALL recent reflections as high-level instructions
        # because the list is short. In production, we'd use semantic search for this too.
        
        reflection_msgs = []
        if self.reflections:
            content = "CRITICAL INSTRUCTIONS (Derived from past mistakes):\n" + "\n".join([f"- {r}" for r in self.reflections])
            reflection_msgs.append(Message(
                role="system",
                content=content,
                metadata={"type": "reflection", "priority": "high"}
            ))
            
        return reflection_msgs + base_context

    def reflect(self, recent_history: List[Message]):
        """
        Analyzes the recent history to find mistakes and generate a lesson.
        """
        history_text = "\n".join([f"{m.role}: {m.content}" for m in recent_history])
        
        prompt = [
            {"role": "system", "content": "You are a critical observer of an AI agent. Analyze the following interaction. If the agent made a mistake or the user was unhappy, formulate a 'Lesson Learned' or 'Rule' to prevent this in the future. If no mistake, return 'None'."},
            {"role": "user", "content": f"Interaction:\n{history_text}"}
        ]
        
        print("\n[Reflection Process Running...]")
        critique = self.llm_client.generate_response(prompt)
        
        if "None" not in critique and len(critique) > 5:
            print(f"[New Lesson Learned]: {critique}")
            self.reflections.append(critique)
            self._save_reflection(critique)
        else:
            print("[Reflection]: No new lessons.")

    def _save_reflection(self, text: str):
        with open(self.reflection_file, "a") as f:
            f.write(text + "\n")

    def _load_reflections(self):
        try:
            with open(self.reflection_file, "r") as f:
                self.reflections = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            pass
