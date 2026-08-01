import os
import json
from openai import OpenAI, APIError
from dotenv import load_dotenv

# Import Canvas and tool factory from updated tools.py
from tools_svg import Canvas, make_diagram_tools

load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("⚠️ WARNING: 'GROQ_API_KEY' is not set in environment or .env file!")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

SYSTEM_PROMPT = """You are an expert Diagram Agent. Your task is to help users draw and modify diagrams.
- Use `generate_diagram` when creating a new diagram from scratch.
- Use `modify_diagram` when modifying an existing element so that you do not wipe out other elements.
- Answer general conversation questions directly without calling tools.
- CRITICAL: When calling a tool, strictly follow its defined JSON schema and NEVER add extra arguments.
"""

MAX_STEPS = 10
USER_FRIENDLY_ERROR = "⚠️ The model is currently unable to answer. Please check your API key or network connection."


def compress_messages(messages: list, max_messages: int = 8) -> list:
    if len(messages) <= max_messages:
        return messages
    
    system_msg = messages[0]
    old_messages = messages[1:-4]
    recent_messages = messages[-4:]
    
    summary_prompt = [
        {"role": "system", "content": "Summarize the key information from this conversation concisely."},
        {"role": "user", "content": json.dumps([m if isinstance(m, dict) else m.model_dump() for m in old_messages])}
    ]
    
    try:
        summary_response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=summary_prompt
        )
        summary_text = summary_response.choices[0].message.content
        return [
            system_msg,
            {"role": "system", "content": f"Previous conversation summary: {summary_text}"}
        ] + recent_messages
    except Exception:
        return [system_msg] + recent_messages


def start_agent():
    canvas = Canvas()
    tool_map, tools_schema = make_diagram_tools(canvas)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=== AI Diagram Agent Started ===")
    print("Commands: 'exit' or 'quit' to stop | 'canvas' to view current elements\n")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "canvas":
            print(f"\n📊 Canvas Elements ({len(canvas.get_elements())}):")
            print(json.dumps(canvas.get_elements(), indent=2, ensure_ascii=False))
            continue
        if not user_input:
            continue
            
        messages.append({"role": "user", "content": user_input})
        messages = compress_messages(messages)
        
        step_count = 0
        
        while step_count < MAX_STEPS:
            step_count += 1
            
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto"
                )
            except APIError as e:
                print(f"\n[API Error Details]: {e.message}")
                print(f"Agent: {USER_FRIENDLY_ERROR}")
                messages.pop()
                break
            except Exception as e:
                print(f"\n[General Exception]: {e}")
                print(f"Agent: {USER_FRIENDLY_ERROR}")
                messages.pop()
                break
            
            response_message = response.choices[0].message
            messages.append(response_message)
            
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    print(f"\n⚙️  [TOOL EXECUTING]: Calling '{function_name}'...")
                    
                    if function_name in tool_map:
                        tool_function = tool_map[function_name]
                        result = tool_function(**arguments)
                    else:
                        result = f"Error: Tool '{function_name}' is not recognized."
                    
                    print(f"📋  [TOOL RESULT]: {result}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
                    
                messages = compress_messages(messages)
            else:
                print(f"\nAgent: {response_message.content}")
                break

        if step_count >= MAX_STEPS:
            print(f"\n⚠️ Reached step count limit ({MAX_STEPS}). Turning off turn execution.")

if __name__ == "__main__":
    start_agent()