# agent/agent.py
import json
import re
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda

# Import the tools (@tool-decorated)
from .tools import explain_code, debug_code, generate_unity_script

# Load environment variables
load_dotenv(".env")


# ======================================================
#  FIXED REGEX — supports </toolname> and </function>
# ======================================================
_TOOL_TAG_RE = re.compile(
    r"<(?P<name>[a-zA-Z0-9_]+)>\s*(\{.*?\})\s*</(?:\1|function)>",
    re.S
)


# ======================================================
#  FIND TOOL BY NAME
# ======================================================
def _find_tool_by_name(tools, name):
    for t in tools:
        t_name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if t_name == name:
            return t
    return None


# ======================================================
#  EXECUTE TOOL
# ======================================================
def _exec_tool(tool_obj, args):
    # StructuredTool invokes through "invoke"
    if hasattr(tool_obj, "invoke"):
        try:
            return tool_obj.invoke(args)
        except TypeError:
            if isinstance(args, dict) and len(args) == 1:
                return tool_obj.invoke(next(iter(args.values())))
            raise

    # Normal Python function
    if callable(tool_obj):
        if isinstance(args, dict):
            try:
                return tool_obj(**args)
            except TypeError:
                return tool_obj(next(iter(args.values())))
        return tool_obj(args)

    raise RuntimeError("Invalid tool object.")


# ======================================================
#  PROCESS RAW LLM TEXT AND EXECUTE TOOL CALL IF FOUND
# ======================================================
def _handle_llm_result_text(result_text: str, tools):
    match = _TOOL_TAG_RE.search(result_text)
    if not match:
        return result_text

    tool_name = match.group("name")
    json_text = match.group(2).strip()

    try:
        args = json.loads(json_text)
    except Exception:
        args = json_text.strip("{} ")

    tool_obj = _find_tool_by_name(tools, tool_name)
    if not tool_obj:
        return f"[Error] Tool not found: {tool_name}"

    try:
        return _exec_tool(tool_obj, args)
    except Exception as e:
        return f"[Tool execution error] {e}"


# ======================================================
#  AGENT CREATOR
# ======================================================
def create_agent():
    # Initialize LLM
    groq_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=groq_key,
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )

    # Tools list
    tools = [explain_code, debug_code, generate_unity_script]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful coding + Unity game development assistant."),
        ("user", "{input}")
    ])

    # Post-processing logic
    def post_process(result):
        try:
            # Structured tool call
            tool_calls = getattr(result, "tool_calls", None)
            if tool_calls:
                tc = tool_calls[0]
                name = tc.get("name")
                args = tc.get("args", {})

                tool_obj = _find_tool_by_name(tools, name)
                if not tool_obj:
                    return f"[Error] Tool not found: {name}"

                return _exec_tool(tool_obj, args)

            # Raw content text
            text = getattr(result, "content", None) or str(result)
            return _handle_llm_result_text(text, tools)

        except Exception as e:
            return f"[Post-process error] {e}"

    # Wrap post processor
    post_proc = RunnableLambda(lambda x: post_process(x))

    # Final agent pipeline: prompt → llm → tool_handler
    agent = RunnableSequence(prompt | llm_with_tools | post_proc)

    return agent
