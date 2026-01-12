import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from agent import BaseAgent, Message
from memory_episodic import EpisodicMemory
from llm import LLMClient

def main():
    print("Testing Architecture B (Episodic Memory)...")
    
    # Mock LLM for summarization
    class MockLLM(LLMClient):
        def __init__(self):
            pass 
            
        def generate_response(self, messages, temperature=0.7):
            return "Mock Summary of interaction"

    # Setup with small STM to force consolidation
    # stm_size = 2. Consolidation triggers at > 4 msgs.
    memory = EpisodicMemory(llm_client=MockLLM(), stm_size=2) 
    
    # Don't mock agent behavior completely, strict flow:
    agent = BaseAgent(name="ArchB_Bot", memory_system=memory, llm_client=MockLLM())
    
    # 1. Fill Memory to trigger consolidation
    # We need 5 messages to trigger (> 2*2)
    inputs = [
        "I like apples", 
        "The sky is blue", 
        "I need a refund for my ticket", 
        "My ticket number is 12345",
        "Can you help me?"
    ]
    
    print("\n--- Interaction Phase ---")
    for inp in inputs:
        agent.run(inp)
        print(f"User input: {inp}")
        # STM check
        print(f"STM Size: {len(memory.stm_window)}")

    # Check if consolidation happened
    print("\n--- Consolidation Check ---")
    print(f"Episodes count: {len(memory.episodes)}")
    if memory.episodes:
        print(f"First Episode Content: {memory.episodes[0].content}")
        print(f"First Episode Keywords: {memory.episodes[0].keywords}")

    # 2. Test Retrieval
    print("\n--- Retrieval Phase ---")
    # New query unrelated to current STM but related to past episode
    query = "What was my ticket number?" 
    print(f"Query: {query}")
    
    context = memory.get_context(current_query=query)
    print(f"Retrieved Context Length: {len(context)}")
    for m in context:
        if m.role == "system":
             print(f"[RETRIEVED EPISODE]: {m.content}")
        else:
             print(f"[STM]: {m.content}")

    # Clean up
    memory.clear()

if __name__ == "__main__":
    main()
