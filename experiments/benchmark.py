import sys
import os
import shutil
import time
import json
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from agent import BaseAgent
from memory import ContextWindowMemory
from memory_episodic import EpisodicMemory
from memory_semantic import SemanticMemory
from memory_reflection import ReflectionMemory
from llm import LLMClient

# --- Setup Mock LLM ---
class BenchmarkLLM(LLMClient):
    def __init__(self):
        pass
    
    def generate_response(self, messages, temperature=0.7):
        last_msg = messages[-1]['content'].lower()
        
        # 1. Summarization check (for Episodic)
        # In real code, we'd detect if this is a summary request. 
        # Here we just assume specific call signatures or rely on the agent flow.
        # But our memory classes call generate_response directly for summaries.
        # We need a way to distinguish "Chat" from "Summary".
        # Our implementation of _consolidate_memory uses a mocked string, so it MIGHT call LLM.
        # Let's assume the memory classes use the mock provided.
        
        # 2. Retrieval/Q&A Logic
        # If the input asks a question, we check if the ANSWER is in the context.
        context_str = str(messages).lower()
        
        if "what is the secret code" in last_msg:
            if "blue_falcon_99" in context_str:
                return "The secret code is Blue_Falcon_99."
            else:
                return "I don't know the secret code."
                
        if "where are the keys" in last_msg:
            if "under the flower pot" in context_str:
                return "The keys are under the flower pot."
            else:
                return "I don't know where the keys are."
                
        return "I processed your input."

def run_benchmark():
    print("Starting Comparative Benchmark...")
    print("Scenario: 'Long-Running Conversation' (50 Turns)")
    
    # Cleanups
    if os.path.exists("./bench_db"): shutil.rmtree("./bench_db")
    if os.path.exists("./bench_episodic.json"): os.remove("./bench_episodic.json")
    if os.path.exists("reflections.txt"): os.remove("reflections.txt")

    # Define Architectures
    # We use small STM (window=2) to force heavy reliance on LTM strategies.
    mock_llm = BenchmarkLLM()
    
    agents = {
        "Arch_A": BaseAgent("A", ContextWindowMemory(window_size=2), mock_llm),
        "Arch_B": BaseAgent("B", EpisodicMemory(mock_llm, stm_size=2, file_path="bench_ep_b.json"), mock_llm),
        "Arch_C": BaseAgent("C", SemanticMemory(mock_llm, stm_size=2, file_path="bench_ep_c.json", db_path="./bench_db_c"), mock_llm),
        "Arch_D": BaseAgent("D", ReflectionMemory(mock_llm, stm_size=2, file_path="bench_ep_d.json", db_path="./bench_db_d"), mock_llm)
    }

    # Simulation Data
    # 1. Early Fact (will be pushed out of STM quickly)
    # 2. Distractor Steps (fill context)
    # 3. Recall Question
    
    turns = [
        ("User", "The secret code is Blue_Falcon_99."), # Fact 1
        ("User", "I am putting the keys under the flower pot."), # Fact 2
    ]
    
    # Add 20 turns of noise
    for i in range(20):
        turns.append(("User", f"Distractor query number {i} to fill context."))

    # Probing Questions
    probes = [
        ("What is the secret code?", "Blue_Falcon_99"),
        ("Where are the keys?", "flower pot")
    ]

    results = {name: {"recall": 0, "total": 0} for name in agents}

    print(f"\nRunning Simulation on {len(agents)} agents...")
    
    for name, agent in agents.items():
        print(f"\n--- Testing Agent: {name} ---")
        start_time = time.time()
        
        # 1. Run Conversation
        for role, txt in turns:
            agent.run(txt)
            
        # 2. Run Probes
        score = 0
        for question, answer_keyword in probes:
            resp = agent.run(question)
            if answer_keyword.lower() in resp.lower():
                score += 1
            else:
                print(f"  [Failure] Asked: '{question}', Got: '{resp}'")
        
        results[name]["recall"] = (score / len(probes)) * 100
        results[name]["time"] = round(time.time() - start_time, 2)
        
        # Cleanup specific agent artifacts
        if hasattr(agent.memory, "clear"):
            agent.memory.clear()

    # Report
    print("\n\n====== FINAL RESULTS ======")
    print(f"{'Architecture':<10} | {'Recall':<10} | {'Time (s)':<10}")
    print("-" * 35)
    for name, date in results.items():
        print(f"{name:<10} | {date['recall']}%       | {date['time']}")

    # Cleanup Global
    if os.path.exists("./bench_db_c"): shutil.rmtree("./bench_db_c")
    if os.path.exists("./bench_db_d"): shutil.rmtree("./bench_db_d")
    for f in ["bench_ep_b.json", "bench_ep_c.json", "bench_ep_d.json", "reflections.txt"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    run_benchmark()
