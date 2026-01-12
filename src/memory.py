from typing import List
from agent import MemoryInterface, Message

class ContextWindowMemory(MemoryInterface):
    """
    Architecture A: Simple Context Window Memory.
    Keeps the last N messages.
    """
    def __init__(self, window_size: int = 10):
        self.messages: List[Message] = []
        self.window_size = window_size

    def add_message(self, message: Message):
        self.messages.append(message)
        # Simple FIFO truncation
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_context(self, current_query: str = None) -> List[Message]:
        return self.messages

    def clear(self):
        self.messages = []
