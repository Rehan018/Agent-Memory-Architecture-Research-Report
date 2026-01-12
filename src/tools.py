def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def search_knowledge_base(query: str) -> str:
    """Mock search tool."""
    return f"Results for '{query}': [Mock Data about policies/schedule]"

TOOLS = {
    "calculator": calculator,
    "search": search_knowledge_base
}
