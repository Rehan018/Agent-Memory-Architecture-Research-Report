import sys
import os
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from agent import BaseAgent
from memory_semantic import SemanticMemory
from llm import LLMClient

def main():
    print("Testing Architecture C (Semantic + Episodic)...")
    
    # Cleanup previous DB
    if os.path.exists("./chroma_db_test"):
        shutil.rmtree("./chroma_db_test")

    class MockLLM(LLMClient):
        def __init__(self):
            pass 
        def generate_response(self, messages, temperature=0.7):
            return "Mock Response"

    # Initialize Memory
    # Using a test path for Chroma
    memory = SemanticMemory(
        llm_client=MockLLM(), 
        stm_size=2, 
        file_path="episodic_test.json",
        db_path="./chroma_db_test"
    )
    
    agent = BaseAgent(name="ArchC_Bot", memory_system=memory, llm_client=MockLLM())

    print("\n--- Seeding Semantic Memory ---")
    # These should go into Vector DB
    facts = [
        "My favorite color is red.",
        "I live in Bangalore.",
        "I prefer vegetarian food."
    ]
    
    for f in facts:
        print(f"User: {f}")
        agent.run(f)
        
    print("\n--- Testing Retrieval ---")
    queries = [
        "What is my favorite color?", 
        "Where do I live?",
        "What food do I like?"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        # We manually inspect valid context retrieval to prove Vector DB works
        context = memory.get_context(current_query=q)
        found_semantic = False
        for m in context:
            if m.metadata.get("type") == "semantic":
                print(f"  [Found Semantic]: {m.content}")
                found_semantic = True
        
        if not found_semantic:
            print("  [!] Failed to retrieve semantic memory.")

    # Clean up
    memory.clear()
    if os.path.exists("./chroma_db_test"):
        shutil.rmtree("./chroma_db_test")
    if os.path.exists("episodic_test.json"):
        os.remove("episodic_test.json")

if __name__ == "__main__":
    main()
