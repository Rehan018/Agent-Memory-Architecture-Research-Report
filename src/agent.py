import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

class MemoryInterface:
    def add_message(self, message: Message):
        raise NotImplementedError

    def get_context(self, current_query: str = None) -> List[Message]:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

class BaseAgent:
    def __init__(self, name: str, memory_system: MemoryInterface, llm_client, tools: List[Any] = None):
        self.name = name
        self.memory = memory_system
        self.llm_client = llm_client
        self.tools = tools or []
        self.conversation_id = str(uuid.uuid4())

    def run(self, user_input: str) -> str:
        """
        Main execution loop:
        1.  Format input
        2.  Retrieve context from memory
        3.  Call LLM
        4.  Store interaction
        5.  Return response
        """
        # 1. Input message
        user_msg = Message(role="user", content=user_input)
        self.memory.add_message(user_msg)

        # 2. Retrieve Context
        context = self.memory.get_context(current_query=user_input)
        
        # 3. LLM Call
        # Convert Message objects to dicts for the LLM client
        messages = [{"role": m.role, "content": m.content} for m in context]
        
        # Insert System Prompt if needed (optional, logic can be added here)
        system_prompt = {"role": "system", "content": f"You are {self.name}, a helpful AI assistant."}
        messages.insert(0, system_prompt)

        print(f"[{self.name}] Thinking with {len(context)} messages context...")
        response_content = self.llm_client.generate_response(messages)
        
        # 4. Store Response
        agent_msg = Message(role="assistant", content=response_content)
        self.memory.add_message(agent_msg)

        return response_content
