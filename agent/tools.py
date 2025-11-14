import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

# Load .env file
load_dotenv(".env")

llm_for_tools = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
)

@tool
def explain_code(code: str) -> str:
    """Dynamically explain any code."""
    prompt = f"Explain the following code:\n\n{code}\n\nExplanation:"
    result = llm_for_tools.invoke(prompt)
    return result.content

@tool
def debug_code(code: str) -> str:
    """Debug the given code and fix errors."""
    prompt = f"Debug this code. Fix errors and explain what was wrong:\n\n{code}"
    result = llm_for_tools.invoke(prompt)
    return result.content

@tool
def generate_unity_script(prompt: str) -> str:
    """Generate Unity C# script dynamically."""
    query = f"Generate a Unity C# script for:\n\n{prompt}"
    result = llm_for_tools.invoke(query)
    return result.content
