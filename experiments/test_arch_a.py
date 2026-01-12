import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from agent import BaseAgent
from memory import ContextWindowMemory
from llm import LLMClient

def main():
    print("Testing Architecture A (Context Window)...")
    
    # Setup
    # Using 'mock' provider for testing without keys first, or 'ollama' if available. 
    # For now, let's assume valid API key is not present and mock behavior or use a specific test provider.
    # Actually, LLMClient defaults to OpenAI. Let's try to mock the client for this specific test 
    # if we don't want to make real calls, OR we can try real calls if key exists.
    
    # Let's mock the LLMClient for the unit test to ensure logic works independently of API.
    class MockLLM:
        def generate_response(self, messages):
            last = messages[-1]['content']
            return f"Echo: {last}"
            
    memory = ContextWindowMemory(window_size=3) # Small window to test forgetting
    agent = BaseAgent(name="ArchA_Bot", memory_system=memory, llm_client=MockLLM())

    # Interaction Loop
    inputs = ["Hello", "My name is Rehan", "What is 2+2?", "What is my name?"]
    
    for i, inp in enumerate(inputs):
        print(f"\nUser: {inp}")
        response = agent.run(inp)
        print(f"Agent: {response}")
        
        # Verify memory state
        print(f"Memory Size: {len(memory.messages)}")
        for m in memory.messages:
            print(f"- {m.role}: {m.content}")

    # The window size is 3. 
    # 1. User: Hello -> Mem: [User, Asst] (2)
    # 2. User: Name -> Mem: [User, Asst, User, Asst] -> Truncate to 3? 
    # Wait, ContextWindowMemory keeps LAST N messages. 
    # If window=3:
    # After step 1 (2 msgs): [U1, A1]
    # After step 2 (4 msgs): [A1, U2, A2] -> "My name is Rehan", "Echo..."
    # After step 3 (6 msgs): [A2, U3, A3] -> "What is 2+2", "Echo..."
    # After step 4 (8 msgs): [A3, U4, A4] -> "What is my name", "Echo..."
    # So "My name is Rehan" (U2) will be lost.
    
if __name__ == "__main__":
    main()
