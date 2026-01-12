import sys
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from agent import BaseAgent, Message
from memory_reflection import ReflectionMemory
from llm import LLMClient

def main():
    print("Testing Architecture D (Reflection Memory)...")
    
    # Cleanup
    if os.path.exists("./chroma_db_test_d"):
        shutil.rmtree("./chroma_db_test_d")
    if os.path.exists("reflections.txt"):
        os.remove("reflections.txt")

    # Mock LLM that simulates:
    # 1. Being bad initially (forgetting to ask for confirmation).
    # 2. Being a critic (generating the lesson).
    # 3. Being good (following the lesson).
    class MockLLM(LLMClient):
        def __init__(self):
            pass 
        def generate_response(self, messages, temperature=0.7):
            last_msg = messages[-1]['content']
            system_instruction = messages[0]['content'] if messages else ""
            
            # Critic Logic (Reflection)
            if "critical observer" in str(messages):
                return "ALWAYS ask for user confirmation before booking."
            
            # Agent Logic
            if "book a flight" in last_msg.lower():
                # Check if we have the lesson in system instructions
                if "ALWAYS ask for user confirmation" in str(messages):
                    return "I can help with that. Do you want me to proceed with booking?" # Good behavior
                else:
                    return "Booking confirmed!" # Bad behavior (assuming confirmation)
            
            return "I am helpful."

    memory = ReflectionMemory(
        llm_client=MockLLM(), 
        stm_size=2, 
        file_path="episodic_test_d.json",
        db_path="./chroma_db_test_d"
    )
    agent = BaseAgent(name="ArchD_Bot", memory_system=memory, llm_client=MockLLM())

    print("\n--- Phase 1: Naive Agent (Making a Mistake) ---")
    user_input = "Please book a flight to Paris."
    print(f"User: {user_input}")
    resp = agent.run(user_input)
    print(f"Agent: {resp}")
    
    # Simulate User Feedback
    feedback = "You didn't ask for confirmation! That's bad."
    print(f"User: {feedback}")
    agent.run(feedback)
    
    print("\n--- Phase 2: Reflection Triggered ---")
    # Manually trigger reflection on recent history
    # In real D-Arch, this happens periodically or on negative sentiment.
    history = memory.stm_window 
    memory.reflect(history)
    
    print("\n--- Phase 3: Improved Agent (After Reflection) ---")
    # Clear STM to simulate a new day/session, but Reflection persists
    memory.stm_window = [] 
    
    print(f"User: {user_input} (Again)")
    resp = agent.run(user_input)
    print(f"Agent: {resp}")
    
    if "Do you want me to proceed" in resp:
        print("SUCCESS: Agent applied the lesson learned!")
    else:
        print("FAILURE: Agent repeated the mistake.")

    # Cleanup
    memory.clear()
    if os.path.exists("./chroma_db_test_d"):
        shutil.rmtree("./chroma_db_test_d")
    if os.path.exists("reflections.txt"):
        os.remove("reflections.txt")
    if os.path.exists("episodic_test_d.json"):
        os.remove("episodic_test_d.json")

if __name__ == "__main__":
    main()
