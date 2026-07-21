import os
import json
from openai import OpenAI
from tools import TOOLS_SCHEMA, TOOL_MAP

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to local tools.
- Answer general knowledge questions directly using your own knowledge without calling tools.
- Only call tools when you explicitly need to read/write/edit files, run shell commands, or search the web.
"""

DANGEROUS_TOOLS = {"write_file", "edit_file", "run_command"}

def run_agent_turn(user_prompt: str):
    """Single turn execution for Evals or API calls"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )
        message = response.choices[0].message
        called_tool = None
        if message.tool_calls:
            called_tool = message.tool_calls[0].function.name
        return called_tool, message.content
    except Exception as e:
        return None, f"Error: {e}"

def start_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=== AI Agent Started with Human-in-the-Loop (type 'exit' to quit) ===")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input:
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
        except Exception as e:
            print(f"\n[API Error]: {e}")
            continue
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                print(f"\n[Agent Decision]: Wants to call '{function_name}' with args: {arguments}")
                
                if function_name in DANGEROUS_TOOLS:
                    approval = input(f"⚠️ [Security Gate] Allow execution of '{function_name}'? (y/n): ").strip().lower()
                    if approval != 'y':
                        result = "User denied this action."
                        print("[Gate Result]: Action DENIED by user.")
                    else:
                        tool_function = TOOL_MAP[function_name]
                        result = tool_function(**arguments)
                        print(f"[Tool Result]: {result}")
                else:
                    tool_function = TOOL_MAP[function_name]
                    result = tool_function(**arguments)
                    print(f"[Tool Result]: {result}")
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
                
                final_response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages
                )
                final_message = final_response.choices[0].message
                messages.append(final_message)
                print(f"\nAgent: {final_message.content}")
        else:
            print(f"\nAgent: {response_message.content}")

if __name__ == "__main__":
    start_agent()